from types import SimpleNamespace

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


def test_graph_mechanism_error_covers_the_platform_not_one_kernel():
    assert cg._graph_mechanism_error(AttributeError("no CUDAGraph"))
    assert cg._graph_mechanism_error(NotImplementedError("CUDA graphs"))
    assert cg._graph_mechanism_error(RuntimeError("stream capture is not supported here"))
    # A kernel that cannot be captured (or an unsupported dtype) must not disable
    # graph timing for the ops after it.
    assert not cg._graph_mechanism_error(
        RuntimeError("operation not permitted when stream is capturing"))
    assert not cg._graph_mechanism_error(RuntimeError("bf16 is not supported by this op"))


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
