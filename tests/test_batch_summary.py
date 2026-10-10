import batch_summary

PRETTY_LOG = """\
Operator: softmax  (dtype=torch.bfloat16, mode=kernel, level=core)
Status     Torch Latency (ms)   Gems Latency (ms)   Gems Speedup  Size Detail
SUCCESS              0.010000            0.008000          1.250  [torch.Size([1024, 1024])]
FAILED                    N/A                 N/A            N/A  [torch.Size([64, 512, 512])]
"""

UPSTREAM_LOG = """\
Operator: mm  Performance Test (dtype=torch.float16, mode=kernel,level=core)
SUCCESS   0.003175   0.003074   1.033   36.843   [torch.Size([384, 384])]
"""


def test_parse_pretty_and_upstream_tables():
    rows = batch_summary.parse_run_log(PRETTY_LOG + UPSTREAM_LOG)
    assert [r["fg_op_name"] for r in rows] == ["softmax", "softmax", "mm"]
    assert rows[0]["dtype"] == "torch.bfloat16"
    assert (rows[0]["torch_lat"], rows[0]["gems_lat"], rows[0]["speedup"]) == (
        "0.010000", "0.008000", "1.250")
    assert rows[0]["size_detail"] == "[torch.Size([1024, 1024])]"
    assert rows[1]["row_status"] == "FAILED"
    assert rows[2]["extra_metrics"] == "36.843"


def test_classify():
    rows = batch_summary.parse_run_log(PRETTY_LOG)
    assert batch_summary.classify(0, rows[:1], "") == "PASS"
    assert batch_summary.classify(0, rows, "") == "FAIL"
    assert batch_summary.classify(1, rows[:1], "") == "FAIL"
    assert batch_summary.classify(0, [], "== 1 skipped in 0.2s ==") == "SKIP"
    assert batch_summary.classify(0, [], "== 1 passed in 0.2s ==") == "FAIL"


def test_marker_notes():
    text = ("BENCHMARK_DIRECT_NO_CUDAGRAPH op=x phase=capture\n" * 2
            + "AUTOTUNE_REPLAY_FALLBACK reason=key_missing\n")
    assert batch_summary.marker_notes(text) == "no_cudagraph_rows=2;replay_fallback=1"
    assert batch_summary.marker_notes("") == ""
