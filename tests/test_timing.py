from types import SimpleNamespace
import sys

import pytest

import _cudagraph_plugin as cg


def test_graph_plan_uncapped_is_do_bench_cudagraph():
    assert cg.graph_plan(0.01, 100, 0) == (10000, 1)
    assert cg.graph_plan(0.0, 100, 0) == (1000, 1)
    assert cg.graph_plan(500.0, 100, 0) == (1, 1)


def test_graph_plan_cap_keeps_the_total_call_count():
    calls, replays = cg.graph_plan(0.01, 100, 1000)
    assert calls == 1000 and replays == 10
    calls, replays = cg.graph_plan(0.03, 100, 1000)  # 3333 calls -> 1000 x 4
    assert calls == 1000 and replays == 4
    assert cg.graph_plan(1.0, 100, 1000) == (100, 1)


def test_max_calls_per_graph(monkeypatch):
    monkeypatch.delenv("GRAPH_MAX_CALLS", raising=False)
    assert cg.max_calls_per_graph() == cg.DEFAULT_MAX_CALLS
    monkeypatch.setenv("GRAPH_MAX_CALLS", "0")
    assert cg.max_calls_per_graph() == 0


def test_summarize_matches_do_bench_modes():
    times = [3.0, 1.0, 2.0, 4.0]
    assert cg.summarize(times, None, "median") == 2.5
    assert cg.summarize(times, None, "mean") == 2.5
    assert cg.summarize(times, None, "min") == 1.0
    assert cg.summarize(times, None, "all") == times
    assert cg.summarize(times, [0.5], "mean") == 2.5
    assert cg.summarize(times, [0.0, 1.0], "mean") == [1.0, 4.0]


@pytest.mark.parametrize("text, fatal", [
    ("CUDA error: an illegal memory access was encountered", True),
    ("operation not permitted when stream is capturing. Compile with `TORCH_USE_CUDA_DSA` to "
     "enable device-side assertions.", False),
])
def test_fatal_errors(text, fatal):
    assert cg._is_fatal_cuda_error(RuntimeError(text)) is fatal


def test_timing_scope():
    import _timing_scope as ts
    seen = []
    wrapped = ts.timed(lambda: seen.append(ts.in_measurement()))
    ts.enabled = True
    try:
        assert not ts.in_measurement()
        wrapped()
        assert seen == [True] and not ts.in_measurement()
    finally:
        ts.enabled = False
    assert ts.in_measurement()


_GRAPH_API_NAMES = ("CUDAGraph", "graph", "Stream", "Event", "synchronize",
                    "current_stream", "stream", "current_device", "empty_cache")


def test_missing_graph_api_spots_a_partial_cuda_backend():
    full = SimpleNamespace(cuda=SimpleNamespace(**{n: object() for n in _GRAPH_API_NAMES}))
    assert cg._missing_graph_api(full) == ""
    assert cg._missing_graph_api(SimpleNamespace(cuda=None)) == "torch.cuda"
    partial = SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=object(), graph=object()))
    assert cg._missing_graph_api(partial) == "torch.cuda.Stream"


def test_graph_mechanism_error_covers_the_platform_not_one_kernel(monkeypatch, tmp_path):
    """Only torch's own graph code reporting an absent mechanism switches graph
    timing off for the rest of the op. A kernel that cannot be captured — even
    with "not supported" in its message — keeps graph timing for the next shape,
    and so does an operator's own AttributeError or NotImplementedError."""
    torch_root = tmp_path / "torch"
    (torch_root / "cuda").mkdir(parents=True)
    (torch_root / "__init__.py").write_text("")
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(__file__=str(torch_root / "__init__.py")))

    def from_(filename, exc):
        try:
            exec(compile("raise exc", str(filename), "exec"), {"exc": exc})
        except Exception as raised:   # the traceback's last frame is the raise site
            return raised
        raise AssertionError("the statement did not raise")

    platform = [AttributeError("module 'torch.cuda' has no attribute 'CUDAGraph'"),
                NotImplementedError("CUDA graphs"),
                RuntimeError("CUDA graphs are not supported on this device"),
                RuntimeError("stream capture is not supported here")]
    for exc in platform:
        assert cg._graph_mechanism_error(
            from_(torch_root / "cuda" / "graphs.py", exc)), exc

    # An operation rejected inside a capture is that call's failure, not the
    # platform's — even when torch's own graph file raises it (a CUDA/C++ error
    # surfaces through capture_end()) and even with "not supported" in the message.
    states = [RuntimeError("operation not permitted when stream is capturing"),
              RuntimeError("capture is already active"),
              RuntimeError("operation not supported during graph capture"),
              RuntimeError("graph capture is not supported for this op")]
    for exc in states:
        assert cg._graph_mechanism_error(
            from_(torch_root / "cuda" / "graphs.py", exc)) == "", exc

    kernel = [RuntimeError("operation not supported during graph capture"),
              NotImplementedError("aten::gather not implemented for this backend"),
              AttributeError("'NoneType' object has no attribute 'dtype'"),
              AttributeError("'Widget' object has no attribute 'graph_config'"),
              RuntimeError("bf16 is not supported by this op")]
    for exc in kernel:
        assert cg._graph_mechanism_error(
            from_(tmp_path / "benchmark" / "test_gather.py", exc)) == "", exc


def test_a_missing_attribute_needs_both_the_owner_and_the_name(monkeypatch, tmp_path):
    """A missing attribute names a platform gap only when torch.cuda itself is the
    object and the attribute is one the graph path needs. `'Widget' object has no
    attribute 'graph'` and `torch.cuda.foo` are neither: reading the name alone
    switched graph timing off for the rest of the op."""
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(__file__=str(tmp_path / "torch" / "__init__.py")))
    assert cg._graph_mechanism_error(
        AttributeError("module 'torch.cuda' has no attribute 'CUDAGraph'"))
    assert cg._graph_mechanism_error(
        AttributeError("module 'torch.cuda' has no attribute 'graph'"))
    assert not cg._graph_mechanism_error(
        AttributeError("'Widget' object has no attribute 'graph'"))
    assert not cg._graph_mechanism_error(
        AttributeError("module 'torch.cuda' has no attribute 'foo'"))
    assert not cg._graph_mechanism_error(
        AttributeError("'NoneType' object has no attribute 'dtype'"))


@pytest.mark.skipif(sys.version_info < (3, 10),
                    reason="AttributeError.name/obj are filled in from 3.10")
def test_a_real_missing_attribute_uses_the_interpreters_owner():
    """The interpreter fills `name`/`obj` in, so the raise site decides: torch.cuda's
    stand-in missing CUDAGraph is the platform, any other object missing `graph`
    is not."""
    cuda = SimpleNamespace(__name__="torch.cuda", Stream=object())
    try:
        cuda.CUDAGraph
    except AttributeError as exc:
        assert cg._graph_mechanism_error(exc)
    else:
        raise AssertionError("the attribute lookup did not raise")

    widget = SimpleNamespace()
    try:
        widget.graph
    except AttributeError as exc:
        assert not cg._graph_mechanism_error(exc)
    else:
        raise AssertionError("the attribute lookup did not raise")


def test_platform_without_the_graph_api_calls_do_bench_directly(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cg, "graph_missing_api", lambda: "torch.cuda.CUDAGraph")
    monkeypatch.setattr(cg, "_GRAPH_UNSUPPORTED", "")
    monkeypatch.setattr(cg, "_unsupported_reported", False)
    monkeypatch.setattr(cg, "_ORIGINAL_DO_BENCH", lambda fn, **kw: calls.append(kw) or 1.0)
    assert cg._patched_do_bench(lambda: None, warmup=1, rep=2) == 1.0
    assert cg._patched_do_bench(lambda: None, warmup=1, rep=2) == 1.0
    assert len(calls) == 2  # the fallback still runs, once per shape
    out = capsys.readouterr().out
    assert out.count("BENCHMARK_DIRECT_NO_CUDAGRAPH") == 1  # marker printed once per op
    assert "phase=capability" in out


def test_a_kernel_that_cannot_be_captured_keeps_graphs_for_the_next_measurement(
        monkeypatch):
    """End to end: the first shape's capture fails with a kernel-level error, it
    falls back to do_bench, and the next measurement still captures — only that
    one shape is timed with plain do_bench."""
    fallbacks = []
    attempts = []
    monkeypatch.setattr(cg, "_GRAPH_UNSUPPORTED", "")
    monkeypatch.setattr(cg, "_unsupported_reported", False)
    monkeypatch.setattr(cg, "graph_missing_api", lambda: "")
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setattr(cg, "_ORIGINAL_DO_BENCH", lambda fn, **kw: fallbacks.append(kw) or 1.0)
    monkeypatch.setattr(cg, "_report_direct",
                        lambda op, phase, failure: fallbacks.append(phase))

    def warmup(fn, budget, grad_to_none):
        attempts.append("warmup")
        if len(attempts) == 1:
            raise RuntimeError("operation not supported during graph capture")

    monkeypatch.setattr(cg, "_warmup_before_capture", warmup)
    monkeypatch.setattr(cg, "_find_host_sync", lambda run: "")
    monkeypatch.setattr(cg, "_capture_with_scratch_rng", lambda run: run())
    monkeypatch.setattr(cg, "_graph_bench", lambda fn, rep, grad_to_none, q, m: 2.0)

    assert cg._patched_do_bench(lambda: None, warmup=1, rep=2) == 1.0   # that shape fell back
    assert cg._patched_do_bench(lambda: None, warmup=1, rep=2) == 2.0   # the next one graphed
    assert cg._GRAPH_UNSUPPORTED == ""        # the failure was the kernel's, not the platform's
    assert attempts == ["warmup", "warmup"]   # the second measurement tried a capture
    # The failing phase is reported once, then the fallback runs once; the second
    # measurement does neither.
    assert fallbacks == ["warmup", {"warmup": 1, "rep": 2, "grad_to_none": None,
                                    "quantiles": None, "return_mode": "mean"}]


def test_graph_disabled_matches_names(monkeypatch):
    monkeypatch.setenv("NO_CUDAGRAPH_OPS", "lstm, foo")
    assert cg.graph_disabled("lstm")
    assert cg.graph_disabled("main:lstm")
    assert cg.graph_disabled("foo")
    assert not cg.graph_disabled("mm")


def test_disabled_op_goes_straight_to_do_bench(monkeypatch, capsys):
    calls = []
    monkeypatch.setenv("NO_CUDAGRAPH_OPS", "lstm")
    monkeypatch.setenv("FLAGGEMS_PERF_CURRENT_OP", "lstm")
    monkeypatch.setattr(cg, "_ORIGINAL_DO_BENCH", lambda fn, **kw: calls.append(kw) or 1.0)
    monkeypatch.setattr(cg, "_disabled_reported", False)
    monkeypatch.setattr(cg, "_GRAPH_UNSUPPORTED", "")
    assert cg._patched_do_bench(lambda: None) == 1.0
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "phase=disabled" in out
