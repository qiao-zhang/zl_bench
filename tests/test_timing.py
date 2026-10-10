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
