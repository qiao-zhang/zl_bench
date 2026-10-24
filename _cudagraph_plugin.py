"""pytest plugin: swap `triton.testing.do_bench` for CUDA-graph timing.

Loaded via `-p _cudagraph_plugin` in run_pytest.sh's PLUGINS list (comment that
line out to time with plain do_bench). Replaces do_bench at session configure,
after Triton is loaded but before any benchmark calls it. Per measurement:

1. synchronize, so inputs still in flight on the default stream are complete;
2. warm fn up for the caller's warmup budget (_warmup_before_capture);
3. run fn once with host syncs turned into errors: a call that synchronizes
   with the host cannot be captured, so it goes straight to the fallback
   without an aborted capture (_find_host_sync);
4. time it with _graph_bench: triton's do_bench_cudagraph scheme (a graph of
   n back-to-back calls, replayed 10 times, median), with at most
   GRAPH_MAX_CALLS calls per graph; larger counts are reached by replaying
   the graph several times per sample, so a sample still spans the rep
   budget. An out-of-memory capture is retried with one call per graph.

A call that cannot be timed this way falls back to plain do_bench and prints
a BENCHMARK_DIRECT_NO_CUDAGRAPH marker into the log. Captures run under a
scratch CUDA RNG state (_capture_with_scratch_rng); an aborted capture is
released (_release_memory) and followed by one untimed call
(_absorb_aborted_capture) before the fallback.

Accelerators whose torch backend presents itself as cuda but implements only
part of the CUDA graph API (the T-Head PPU, and MetaX's MACA layer, which also
presents `torch.cuda`) still go through this path: every measurement tries the
graph first, and the optional helpers — sync-debug mode, graphsafe RNG state,
the RNG generator list — are skipped when missing instead of failing the
measurement. No backend needs a special case here; an op whose capture crashes
the process (a segfault cannot be caught) is opted out by name with
NO_CUDAGRAPH_OPS. When the graph mechanism itself is absent (no
torch.cuda.CUDAGraph, or capture reported unsupported) graph timing is turned
off for the rest of that pytest process, i.e. for this op: its remaining shapes
go straight to do_bench. The next op runs in a fresh process and tries again, so
ops that can be captured keep their graph timing.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import sys

import _timing_scope

_ORIGINAL_DO_BENCH = None  # triton.testing.do_bench, saved in pytest_configure
_MAX_CALLS_ENV = "GRAPH_MAX_CALLS"
DEFAULT_MAX_CALLS = 1000
MIN_SAFE_CALLS = 250  # smaller graphs read 1 us kernels too fast (graph_plan)
N_SAMPLES = 10  # graph replays per measurement, as in do_bench_cudagraph
# torch.cuda API the graph path needs; a backend that lacks one (the PPU and
# MACA compatibility layers implement most, but not necessarily all of it) never
# gets a capture attempt.
_GRAPH_APIS = ("CUDAGraph", "graph", "Stream", "Event", "synchronize", "current_stream", "stream",
               "current_device", "empty_cache")
_GRAPH_UNSUPPORTED = ""  # set once the mechanism itself is missing (graph_missing_api /
                         # _graph_mechanism_error); empty = try graphs
_unsupported_reported = False
# Operators whose capture crashes the process (a segfault cannot be caught and
# turned into the do_bench fallback) can be listed here, comma separated, to go
# straight to do_bench: NO_CUDAGRAPH_OPS=lstm,foo
_DISABLE_ENV = "NO_CUDAGRAPH_OPS"
_disabled_reported = False


def graph_disabled(op: str) -> bool:
    """True when op matches one of the names in NO_CUDAGRAPH_OPS (substring)."""
    names = [n.strip() for n in os.environ.get(_DISABLE_ENV, "").split(",") if n.strip()]
    return any(name in op for name in names)


def _missing_graph_api(torch_module) -> str:
    """First torch.cuda API graph timing needs that this backend lacks, else ""."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return "torch.cuda"
    return next((f"torch.cuda.{name}" for name in _GRAPH_APIS if not hasattr(cuda, name)), "")


def graph_missing_api() -> str:
    """_missing_graph_api for the torch that is installed; "" when a capture can be tried."""
    try:
        import torch
    except Exception as exc:
        return f"import torch: {type(exc).__name__}: {exc}"
    return _missing_graph_api(torch)


def _graph_mechanism_error(exc: Exception) -> str:
    """Reason when exc means graph timing cannot work on this platform at all, else "".

    A single op that cannot be captured (host sync, OOM, a kernel the graph
    rejects) must not disable graph timing for the ops after it, so only a
    missing Python API or an explicit "graphs/capture not supported" counts as
    the mechanism being absent."""
    if isinstance(exc, (AttributeError, NotImplementedError)):
        return f"{type(exc).__name__}: {exc}"
    msg = str(exc).lower()
    absent = ("not supported", "unsupported", "not implemented", "unavailable")
    if any(m in msg for m in absent) and any(c in msg for c in ("graph", "captur")):
        return " ".join(str(exc).split())[:160]
    return ""


def max_calls_per_graph() -> int:
    """Cap on the calls unrolled into one graph; 0 = no cap."""
    value = os.environ.get(_MAX_CALLS_ENV, "").strip()
    return int(value) if value else DEFAULT_MAX_CALLS


def graph_plan(estimate_ms: float, rep_ms: float, cap: int):
    """(calls per graph, replays per sample) for a call taking estimate_ms.

    The uncapped plan is do_bench_cudagraph's: rep_ms worth of calls in one
    graph, one replay per sample. Capture time grows linearly with the calls
    in the graph (about 12 ms per call for conv1d with padding), so with a cap
    the graph holds fewer calls and each sample replays it back to back until
    it covers the same number of calls. From 250 calls per graph up the
    replays add nothing measurable, even for 1 us kernels; smaller graphs read
    such kernels too fast. The plan is the same on both sides of an A/B."""
    n_total = 1000 if estimate_ms <= 0 else max(1, int(rep_ms / estimate_ms))
    n_calls = min(n_total, cap) if cap > 0 else n_total
    return n_calls, math.ceil(n_total / n_calls)


def summarize(times, quantiles, return_mode):
    """The same summaries as triton.testing.do_bench."""
    if quantiles is not None:
        ordered = sorted(times)
        out = []
        for q in quantiles:
            pos = q * (len(ordered) - 1)
            lo, hi = math.floor(pos), math.ceil(pos)
            out.append(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))
        return out[0] if len(out) == 1 else out
    if return_mode == "all":
        return list(times)
    return {"min": min, "max": max, "mean": statistics.fmean,
            "median": statistics.median}[return_mode](times)


def _run_once(fn, grad_to_none):
    """One call of fn, with the gradients the caller wants reset cleared first."""
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None
    fn()


def _is_fatal_cuda_error(exc: Exception) -> bool:
    """True for CUDA context errors (illegal address, OOM, device-side assert):
    the context is poisoned, so re-raise rather than fall back and record a bogus
    latency as success.

    Markers must match the error text itself, not torch's generic footer: torch
    appends "Compile with `TORCH_USE_CUDA_DSA` to enable device-side
    assertions." to EVERY AcceleratorError, so a loose "device-side assert"
    marker would make every capture failure fatal and disable the do_bench
    fallback."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "illegal memory access",
        "cudaerrorillegaladdress",
        "out of memory",
        "device-side assert triggered",
        "an illegal instruction",
        "misaligned address",
        "uncorrectable ecc",
    )
    return any(m in msg for m in markers)


def _warmup_before_capture(fn, warmup_ms, grad_to_none):
    """Warm fn up before cudagraph capture.

    do_bench_cudagraph only warms up 5 iterations internally — too few for the
    MoE kernels, where the first calls trigger Triton autotune config
    compilation, libtuner selection and lazy JIT/init. If that work leaks into
    the captured graph or the first timed iteration the latency is unstable
    run-to-run (most visibly on the small-M multi-kernel path). Absorb the
    one-off compile with a throwaway run, then loop for the caller's warmup time
    budget so the graph is captured at steady state.

    The loop is clamped to 5..200 iterations: the cap keeps sub-ms kernels from
    spinning through a large budget (FlagGems' Config.warm_up defaults to
    1000 ms) — 200 iterations settles them fine in practice — and the floor
    covers kernels whose single run exceeds the budget. If first-run jitter
    reappears on a new op, raising the cap is the first knob to try.
    """
    import time

    import torch

    # First call absorbs autotune/JIT compilation (can take seconds); discard it.
    _run_once(fn, grad_to_none)
    torch.cuda.synchronize()
    # Size the warmup loop from one timed steady-state run. dt_ms includes
    # launch overhead, so n errs low for tiny kernels; warmup needn't be exact.
    t0 = time.perf_counter()
    _run_once(fn, grad_to_none)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    budget = warmup_ms if (warmup_ms and warmup_ms > 0) else 25.0
    n = int(budget / dt_ms) if dt_ms > 0 else 25
    n = max(5, min(n, 200))
    for _ in range(n):
        _run_once(fn, grad_to_none)
    torch.cuda.synchronize()


def _capture_with_scratch_rng(call):
    """Run a graph capture under a throwaway CUDA RNG state.

    torch.cuda.graph's capture_begin unconditionally registers the default CUDA
    RNG generator and sets its capturing flag; when a capture aborts midway the
    epilogue may never run, the flag sticks, and every later CUDA RNG call in
    the process (the next shape's torch.randn included) raises "Offset
    increment outside graph capture". Repairing after the fact is unreliable —
    when the poisoning surfaces depends on which stage the capture died in —
    so prevent it instead: swap in a scratch state via graphsafe_set_state
    before capturing and restore the original state object afterwards, success
    or failure. The real generator never takes part in a capture, so poisoning
    can only land on the discarded scratch state, and _seed_plugin determinism
    is left untouched."""
    import torch
    gens = getattr(torch.cuda, "default_generators", None)
    if gens is None:
        return call()  # backend exposes no default generators: capture without a scratch state
    gen = gens[torch.cuda.current_device()]
    if not (hasattr(gen, "graphsafe_get_state") and hasattr(gen, "graphsafe_set_state")):
        return call()  # no graph-aware RNG state here: the real state takes part
    try:
        saved = gen.graphsafe_get_state()
        scratch = torch.Generator(device="cuda")
        scratch.manual_seed(gen.initial_seed())
        gen.graphsafe_set_state(scratch)
    except Exception:
        return call()  # scratch state unusable here: capture with the real state
    try:
        return call()
    finally:
        gen.graphsafe_set_state(saved)


def _release_memory():
    """Free what an aborted or failed capture left behind. The graph and its
    private memory pool stay alive in reference cycles until the collector
    runs; for inputs of tens of GiB a few of them leave too little memory for
    the next measurement. Call outside any except block: a live exception's
    traceback still references the graph."""
    import torch
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _find_host_sync(run):
    """Error text if one call of run synchronizes with the host, else None.

    A host sync (tensor.item(), nonzero, a device-to-host copy) is illegal
    during capture, so such a call can only be timed with plain do_bench.
    Detecting it up front with torch's sync debug mode avoids an aborted
    capture and the memory it would strand. Where the backend has no usable
    sync-debug mode (the PPU layer may not implement it), skip the probe and
    let the capture itself decide."""
    import torch
    get_mode = getattr(torch.cuda, "get_sync_debug_mode", None)
    set_mode = getattr(torch.cuda, "set_sync_debug_mode", None)
    if get_mode is None or set_mode is None:
        return None
    try:
        mode = get_mode()
        set_mode("error")
    except Exception:
        return None  # present but unusable here: not a reason to give up the graph
    try:
        run()
    except RuntimeError as exc:
        if "synchroniz" not in str(exc).lower():
            raise
        return " ".join(str(exc).split())
    finally:
        try:
            set_mode(mode)
        except Exception:
            pass
    torch.cuda.synchronize()
    return None


class _HostSync(RuntimeError):
    pass


def _graph_bench(fn, rep, grad_to_none, quantiles, return_mode):
    """CUDA-graph timing, following triton.testing.do_bench_cudagraph, with at
    most max_calls_per_graph() calls per graph (graph_plan)."""
    import torch

    def capture(n_calls):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_calls):
                _run_once(fn, grad_to_none)
        torch.cuda.synchronize()
        return g

    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.cuda.synchronize()
        n_calls, replays = graph_plan(start.elapsed_time(end) / 5, rep, max_calls_per_graph())
        graph = None
        oom = getattr(torch, "OutOfMemoryError", None)
        try:
            graph = capture(n_calls)
        except Exception as exc:
            if oom is None or not isinstance(exc, oom):
                raise
            # retried below, once the failed graph is no longer referenced
        if graph is None:
            _release_memory()
            op = os.environ.get("FLAGGEMS_PERF_CURRENT_OP", "<unknown>")
            print(f"CUDAGRAPH_OOM_RETRY op={op} calls_per_graph=1", flush=True)
            n_total = n_calls * replays
            n_calls, replays = 1, n_total
            graph = capture(1)  # a second OOM is real: the caller treats it as fatal
        times = []
        for _ in range(N_SAMPLES):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / (n_calls * replays))
    return summarize(times, quantiles, return_mode)


def _absorb_aborted_capture(fn):
    """Run fn once, discarding a non-fatal error, after a capture aborted.

    Some libraries keep per-handle state that the aborted capture leaves
    broken, and report it on their next call only: after a failed capture of
    linalg_matrix_rank(hermitian=True), the next cusolverDnXsyevd returns
    CUSOLVER_STATUS_EXECUTION_FAILED once and then works again, on any stream.
    Clearing the CUDA runtime's last error does not help, it is cusolver's own
    state. Without this throwaway call the plain do_bench fallback would fail
    the op although nothing is wrong with it.
    """
    import torch
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_fatal_cuda_error(exc):
            raise


def _report_direct(op: str, phase: str, failure) -> None:
    """One BENCHMARK_DIRECT_NO_CUDAGRAPH marker, the same shape as before PPU support."""
    from _term_style import YELLOW, paint
    print(
        paint(f"BENCHMARK_DIRECT_NO_CUDAGRAPH op={op} phase={phase} "
              f"reason={failure[0]} detail={failure[1]!r}", YELLOW),
        flush=True,
    )


def _patched_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                      quantiles=None, return_mode="mean"):
    """Drop-in replacement for triton.testing.do_bench using CUDA-graph timing."""
    global _GRAPH_UNSUPPORTED, _unsupported_reported, _disabled_reported
    op = os.environ.get("FLAGGEMS_PERF_CURRENT_OP", "<unknown>")
    fn = _timing_scope.timed(fn)  # its launches are the measurement's (_ir_dump_plugin)
    # A capture that segfaults cannot be caught; such ops are opted out by name
    # and timed with plain do_bench (the marker says phase=disabled).
    if graph_disabled(op):
        if not _disabled_reported:
            _disabled_reported = True
            _report_direct(op, "disabled", ("Disabled", f"{_DISABLE_ENV} matched {op}"))
        return _ORIGINAL_DO_BENCH(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none,
                                  quantiles=quantiles, return_mode=return_mode)
    # A backend without the graph API (or one that already rejected a capture in
    # this process) goes straight to do_bench: the capture could not work, and
    # every shape would pay for finding that out.
    missing = _GRAPH_UNSUPPORTED or graph_missing_api()
    if missing:
        if not _unsupported_reported:
            _unsupported_reported = True
            _report_direct(op, "capability", ("GraphUnsupported", missing))
        return _ORIGINAL_DO_BENCH(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none,
                                  quantiles=quantiles, return_mode=return_mode)
    phase = "sync"  # which step rejected the graph, for the fallback marker
    failure = None
    try:
        # The graph runs on a fresh side stream; input tensors produced just
        # before (quantize/topk in the input iter) may still be in flight, and
        # racing on them intermittently kills the context (illegal instruction
        # from a device trap on garbage indices). Sync before switching streams.
        import torch
        torch.cuda.synchronize()
        phase = "warmup"
        # Settle autotune/JIT before capture (rationale: _warmup_before_capture).
        _warmup_before_capture(fn, warmup, grad_to_none)
        phase = "host_sync"
        sync = _find_host_sync(lambda: _run_once(fn, grad_to_none))
        if sync:
            raise _HostSync(sync)
        phase = "capture"
        return _capture_with_scratch_rng(
            lambda: _graph_bench(fn, rep, grad_to_none, quantiles, return_mode))
    except Exception as exc:
        # Re-raise genuine device failures; anything else falls back to plain
        # do_bench, with the failing phase in the marker.
        if _is_fatal_cuda_error(exc):
            raise
        failure = (type(exc).__name__, " ".join(str(exc).split())[:160])
        mechanism = _graph_mechanism_error(exc)
        if mechanism:
            # Not this kernel's fault: the platform cannot do graphs. Skip the
            # remaining shapes of this op; the next op is a fresh process.
            _GRAPH_UNSUPPORTED = mechanism
    # Out of the except block, so the failed graph is unreferenced and can be freed.
    _report_direct(op, phase, failure)
    if phase == "capture":
        _release_memory()
        _absorb_aborted_capture(fn)
    return _ORIGINAL_DO_BENCH(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none,
                              quantiles=quantiles, return_mode=return_mode)


def pytest_configure(config):
    global _ORIGINAL_DO_BENCH
    import triton.testing

    from _term_style import tag
    _ORIGINAL_DO_BENCH = triton.testing.do_bench
    triton.testing.do_bench = _patched_do_bench
    _timing_scope.enabled = True
    cap = max_calls_per_graph()
    print(f"{tag('[cudagraph-plugin]')} triton.testing.do_bench → CUDA-graph timing "
          f"({f'at most {cap} calls per graph' if cap > 0 else 'no cap on calls per graph'}); "
          "a call that cannot be captured falls back to do_bench with a "
          "BENCHMARK_DIRECT_NO_CUDAGRAPH marker (skipped for the rest of the op when the "
          "platform has no graph support)", file=sys.stderr, flush=True)
    if 0 < cap < MIN_SAFE_CALLS:
        print(f"{tag('[cudagraph-plugin]')} warning: {_MAX_CALLS_ENV}={cap} is below "
              f"{MIN_SAFE_CALLS}; graphs that small read sub-microsecond kernels too fast",
              file=sys.stderr, flush=True)
