import pytest

import rlc_ops
import rlc_report


TABLE = """\
RLC Enhance 开启前后算子性能对比,,,,,,
环境,,,,,,
,,,,,,
#,算子,bf16,,,,,
,,收益,,,退化,,
,,最弱,最强,shape,最轻,最重,shape,
1,kron,,,,,,
2,"outer
vllm:outer",,,,,,
3,"uniform_
main:uniform_inplace",,,,,,
"""


def _table(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(TABLE, encoding="utf-8")
    return path


def test_parse_cell_reads_the_display_name_and_inventory_op():
    assert rlc_ops.parse_cell("kron") == ("kron", "", "kron", None)
    assert rlc_ops.parse_cell("outer\nvllm:outer") == ("outer", "vllm", "outer", None)
    # the table's label and the inventory op differ for two tests that share a label
    assert rlc_ops.parse_cell("uniform_\nmain:uniform_inplace") == \
        ("uniform_", "main", "uniform_inplace", None)
    assert rlc_ops.parse_cell("x\nmain:y@z") == ("x", "main", "y", "z")
    assert rlc_ops.parse_cell("vllm:foo") == ("vllm:foo", "vllm", "foo", None)
    # a single line carries the same `[repo:]op[@op_file]` spec as the second line
    assert rlc_ops.parse_cell("main:kron@kron") == ("main:kron@kron", "main", "kron", "kron")
    assert rlc_ops.parse_cell("kron@kron") == ("kron@kron", "", "kron", "kron")
    assert rlc_ops.parse_cell("") is None


def test_read_table_skips_the_title_and_header_rows(tmp_path):
    entries = rlc_ops.read_table(_table(tmp_path))
    assert [(e.display, e.repo, e.op) for e in entries] == [
        ("kron", "", "kron"),
        ("outer", "vllm", "outer"),
        ("uniform_", "main", "uniform_inplace"),
    ]


def test_resolve_picks_the_only_op_file(monkeypatch, tmp_path):
    inv = {
        "main": [{"op": "kron", "op_file": "kron", "collected": "yes"},
                 {"op": "uniform_inplace", "op_file": "uniform", "collected": "yes"}],
        "vllm": [{"op": "outer", "op_file": "outer", "collected": "yes"}],
    }
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(_table(tmp_path)))
    assert specs == ["main:kron@kron", "vllm:outer@outer", "main:uniform_inplace@uniform"]
    assert not errors


def test_resolve_expands_every_test_for_one_name(monkeypatch, tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("#,算子\n1,kron\n", encoding="utf-8")
    inv = {"main": [{"op": "kron", "op_file": "kron", "collected": "yes"},
                    {"op": "kron", "op_file": "kron_other", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(path))
    assert specs == ["main:kron@kron", "main:kron@kron_other"]
    assert [r["status"] for r in report] == ["multi", "multi"]
    assert not errors


def _row(label="main:kron@kron", dtype="torch.bfloat16", gap=0.5, a=0.05, b=0.02,
         gate="changed", flags="", verdict="faster", row=0):
    return rlc_report.Row(label, "main/kron", "kron", dtype, row, gap, a, b,
                          gate, flags, verdict)


def test_summarizing_splits_gain_and_loss_rows():
    rows = [_row(gap=1.5, row=0), _row(gap=0.5, row=1),
            _row(gap=-0.2, row=2, a=0.02, b=0.025)]
    stats = rlc_report.summarizing(rows, {}, None, {}, "main:kron@kron", "bf16")
    assert stats["rows"] == 3
    assert stats["gain"] == (0.5, 1.5, 2)
    assert stats["loss"] == (-0.2, -0.2, 1)
    assert stats["cvt"] == (None, None, False)


def test_summarizing_reads_the_gate_convert_layout_count(tmp_path):
    (tmp_path / "base" / "H1").mkdir(parents=True)
    (tmp_path / "new" / "H1").mkdir(parents=True)
    (tmp_path / "base" / "H1" / "k.digest.json").write_text('{"ops": {"ttg.convert_layout": 2}}')
    (tmp_path / "new" / "H1" / "k.digest.json").write_text('{"ops": {"ttg.convert_layout": 0}}')
    gate = {("main/kron", "kron", "torch.bfloat16", 0): ["H1"]}
    stats = rlc_report.summarizing([_row()], gate, tmp_path, {}, "main:kron@kron", "bf16")
    assert stats["cvt"] == (2, 0, False)
    assert rlc_report.fmt_cvt(stats["cvt"]) == "2→0"


def test_summarizing_marks_an_unchanged_kernel_with_its_convert_count(tmp_path):
    (tmp_path / "base" / "H1").mkdir(parents=True)
    (tmp_path / "new" / "H1").mkdir(parents=True)
    (tmp_path / "base" / "H1" / "k.digest.json").write_text('{"ops": {"ttg.convert_layout": 2}}')
    (tmp_path / "new" / "H1" / "k.digest.json").write_text('{"ops": {"ttg.convert_layout": 2}}')
    kernels = {("main/kron", "kron", "torch.bfloat16", 0): ["H1"]}
    stats = rlc_report.summarizing([_row()], {}, tmp_path, {}, "main:kron@kron", "bf16",
                                   None, kernels)
    assert stats["cvt"] == (2, 2, True)
    assert rlc_report.fmt_cvt(stats["cvt"]) == "2 (kernel unchanged)"


def test_load_run_kernels_reads_the_a_side_rows_tsv(tmp_path):
    path = tmp_path / "r1_a" / "main" / "kron" / "rows.tsv"
    path.parent.mkdir(parents=True)
    path.write_text("op_name\tdtype\trow\tshape\tkernels\n"
                    "kron\ttorch.bfloat16\t0\t4x4\tH1\n"
                    "kron\ttorch.bfloat16\t1\t8x8\tH1,H2\n", encoding="utf-8")
    assert rlc_report.load_run_kernels(tmp_path) == {
        ("main/kron", "kron", "torch.bfloat16", 0): ["H1"],
        ("main/kron", "kron", "torch.bfloat16", 1): ["H1", "H2"],
    }
    assert rlc_report.load_run_kernels(None) == {}


def test_dtype_groups_and_formatters():
    assert rlc_report.dtype_group("torch.bfloat16") == "bf16"
    assert rlc_report.dtype_group("torch.float16") == "fp16"
    assert rlc_report.dtype_group("torch.complex64") == "fp8 / complex64"
    assert rlc_report.dtype_group("torch.int32") is None
    assert rlc_report.fmt_time(0.0) == "<0.1 µs"
    assert rlc_report.fmt_time(0.0005) == "0.5 µs"
    assert rlc_report.fmt_time(0.05) == "50 µs"
    assert rlc_report.fmt_time(12.09) == "12.09 ms"


def test_notes_are_automatic_and_manual(monkeypatch, tmp_path):
    rows = [_row(gap=-0.2, flags="do_bench", verdict="slower", gate="unchanged")]
    notes = {"main:kron@kron": "选到不同 autotune"}
    stats = rlc_report.summarizing(rows, {}, None, notes, "main:kron@kron", "bf16")
    assert "do_bench" in stats["note"]
    assert "选到不同 autotune" in stats["note"]


def test_xlsx_has_the_expected_groups(tmp_path):
    pytest.importorskip("openpyxl")
    from openpyxl import load_workbook

    path = tmp_path / "out.xlsx"
    rows = [{"name": "kron", "label": "main:kron@kron", "rows": 2,
             "groups": {"bf16": rlc_report.summarizing(
                 [_row(gap=1.5, row=0), _row(gap=-0.1, row=1)], {}, None, {},
                 "main:kron@kron", "bf16")},
             "saving": 0.03, "cost": 0.001, "note": ""}]
    rlc_report.write_xlsx(path, rows, "title", "env")
    ws = load_workbook(path).active
    assert ws.max_column == 27
    assert ws.cell(4, 3).value == "bf16"
    assert ws.cell(4, 10).value == "fp16"
    assert ws.cell(4, 17).value == "fp8 / complex64"
    assert ws.cell(6, 3).value == "最弱"
    assert ws.cell(7, 2).value == "kron"
    assert ws.cell(7, 3).value == "+150.0%"
    assert ws.cell(7, 6).value == "-10.0%"


def _one_op_table(tmp_path, cell):
    path = tmp_path / "one.csv"
    path.write_text(f"#,算子\n1,\"{cell}\"\n", encoding="utf-8")
    return path


def test_resolve_runs_both_repos_when_both_have_the_op(monkeypatch, tmp_path):
    inv = {"main": [{"op": "outer", "op_file": "outer", "collected": "yes"}],
           "vllm": [{"op": "outer", "op_file": "outer", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "outer")))
    assert specs == ["main:outer@outer", "vllm:outer@outer"]
    assert [r["status"] for r in report] == ["both", "both"]
    assert not errors


def test_resolve_uses_vllm_for_an_op_that_only_exists_there(monkeypatch, tmp_path):
    inv = {"main": [{"op": "kron", "op_file": "kron", "collected": "yes"}],
           "vllm": [{"op": "fused_marlin_moe_w8a16_fp8",
                     "op_file": "fused_marlin_moe_w8a16_fp8", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "fused_marlin_moe_w8a16_fp8")))
    assert specs == ["vllm:fused_marlin_moe_w8a16_fp8@fused_marlin_moe_w8a16_fp8"]
    assert report[0]["status"] == "vllm"
    assert not errors


def test_resolve_records_an_op_missing_from_both_repos(monkeypatch, tmp_path):
    inv = {"main": [{"op": "kron", "op_file": "kron", "collected": "yes"}],
           "vllm": [{"op": "outer", "op_file": "outer", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "no_such_op")))
    assert specs == []
    assert report == [{"line": 2, "display": "no_such_op", "repo": "",
                       "op": "no_such_op", "op_file": "", "status": "missing"}]
    assert not errors  # recorded, not fatal


def test_resolve_pins_one_op_file_from_the_first_line(monkeypatch, tmp_path):
    """`main:kron@kron` on a single line is the same spec as the two-line form:
    it must pin that benchmark file instead of being read as an op named
    `kron@kron`, and `kron@kron` must pin it in both repos."""
    inv = {"main": [{"op": "kron", "op_file": "kron", "collected": "yes"},
                    {"op": "kron", "op_file": "kron_other", "collected": "yes"}],
           "vllm": [{"op": "kron", "op_file": "kron", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)

    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "main:kron@kron")))
    assert specs == ["main:kron@kron"]
    assert [r["status"] for r in report] == ["main"]
    assert not errors

    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "kron\nmain:kron@kron")))
    assert specs == ["main:kron@kron"]
    assert not errors

    # no repo prefix: the pinned file is looked up in both repos
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "kron@kron")))
    assert specs == ["main:kron@kron", "vllm:kron@kron"]
    assert [r["status"] for r in report] == ["both", "both"]
    assert not errors


def test_resolve_reports_a_pinned_op_file_that_is_missing(monkeypatch, tmp_path):
    inv = {"main": [{"op": "kron", "op_file": "kron", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "main:kron@nope")))
    assert specs == []
    assert errors == ["line 2: no main:kron@nope in the inventory"]
    assert report == [{"line": 2, "display": "main:kron@nope", "repo": "main",
                       "op": "kron", "op_file": "nope", "status": "missing"}]


def test_resolve_matches_the_benchmark_op_name(monkeypatch, tmp_path):
    inv = {"main": [{"op": "perf_fft", "op_file": "fft", "op_names": "fft",
                     "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "fft")))
    assert specs == ["main:perf_fft@fft"]
    assert report[0]["status"] == "main"
    assert not errors


def test_resolve_matches_the_op_file_when_op_names_is_empty(monkeypatch, tmp_path):
    inv = {"main": [{"op": "scaled_grouped_mm_benchmark", "op_file": "scaled_grouped_mm",
                     "op_names": "", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "scaled_grouped_mm")))
    assert specs == ["main:scaled_grouped_mm_benchmark@scaled_grouped_mm"]
    assert not errors


def test_resolve_expands_several_tests_with_one_op_name(monkeypatch, tmp_path):
    inv = {"main": [{"op": "fused_moe_fp8", "op_file": "fused_moe_fp8",
                     "op_names": "fused_experts_impl", "collected": "yes"},
                    {"op": "fused_moe_int8", "op_file": "fused_moe_int8",
                     "op_names": "fused_experts_impl", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "fused_experts_impl")))
    assert specs == ["main:fused_moe_fp8@fused_moe_fp8",
                     "main:fused_moe_int8@fused_moe_int8"]
    assert [r["status"] for r in report] == ["multi", "multi"]
    assert not errors


def test_resolve_expands_all_tests_in_both_repos(monkeypatch, tmp_path):
    inv = {"main": [{"op": "perf_router_gemm", "op_file": "blas_perf_parallel",
                     "op_names": "router_gemm", "collected": "yes"},
                    {"op": "perf_router_gemm", "op_file": "router_gemm",
                     "op_names": "router_gemm", "collected": "yes"}],
           "vllm": [{"op": "perf_router_gemm", "op_file": "router_gemm",
                     "op_names": "router_gemm", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(
        _one_op_table(tmp_path, "router_gemm")))
    assert specs == ["main:perf_router_gemm@blas_perf_parallel",
                     "main:perf_router_gemm@router_gemm",
                     "vllm:perf_router_gemm@router_gemm"]
    assert [r["status"] for r in report] == ["multi", "multi", "multi"]
    assert not errors


def test_display_name_marks_multi_and_both(tmp_path):
    sources = {
        "main:fused_moe_fp8@fused_moe_fp8": {"display": "fused_experts_impl",
                                             "status": "multi"},
        "vllm:outer@outer": {"display": "outer", "status": "both"},
    }
    assert rlc_report.display_name("main:fused_moe_fp8@fused_moe_fp8", {}, sources) == \
        "fused_moe_fp8\nfused_experts_impl · main · multi"
    assert rlc_report.display_name("vllm:outer@outer", {}, sources) == \
        "outer\nouter · vllm · both"
    # without a source row the old same-name rule applies
    assert rlc_report.display_name("main:outer@outer", {"outer": 2}, {}) == \
        "outer\nmain:outer@outer"
    assert rlc_report.display_name("main:kron@kron", {"kron": 1}, {}) == "kron"


def test_display_name_keeps_the_op_file_for_one_name_in_several_files():
    """Two benchmark files with one op name (the `multi` case) must stay apart:
    the second line names the table row both were expanded from, so without the
    op_file on the first line the two rows read exactly the same."""
    sources = {
        "main:perf_router_gemm@blas_perf_parallel": {"display": "router_gemm",
                                                     "status": "multi"},
        "main:perf_router_gemm@router_gemm": {"display": "router_gemm", "status": "multi"},
    }
    op_files = {("main", "perf_router_gemm"): {"blas_perf_parallel", "router_gemm"}}
    a = rlc_report.display_name("main:perf_router_gemm@blas_perf_parallel", {}, sources, op_files)
    b = rlc_report.display_name("main:perf_router_gemm@router_gemm", {}, sources, op_files)
    assert a == "perf_router_gemm@blas_perf_parallel\nrouter_gemm · main · multi"
    assert b == "perf_router_gemm@router_gemm\nrouter_gemm · main · multi"
    assert a != b
    # one file per op name keeps the plain first line
    assert rlc_report.display_name("main:kron@kron", {}, {}, {("main", "kron"): {"kron"}}) == "kron"
    assert rlc_report.display_name("main:kron@kron", {"kron": 2}, {},
                                   {("main", "kron"): {"kron"}}) == "kron\nmain:kron@kron"


def test_load_sources_reads_the_provenance_csv(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text("line,display,repo,op,op_file,status\n"
                    "7,router_gemm,main,perf_router_gemm,router_gemm,multi\n",
                    encoding="utf-8")
    sources = rlc_report.load_sources(path)
    assert sources["main:perf_router_gemm@router_gemm"]["display"] == "router_gemm"
    assert sources["main:perf_router_gemm@router_gemm"]["status"] == "multi"
    assert rlc_report.load_sources(None) == {}
    assert rlc_report.load_sources(tmp_path / "missing.csv") == {}


def test_read_table_order_unpacks_the_resolve_result(monkeypatch, tmp_path):
    import rlc_ops
    monkeypatch.setattr(rlc_ops, "resolve",
                        lambda entries: (["main:kron@kron"], [], []))
    table = tmp_path / "t.csv"
    table.write_text("#,算子\n1,kron\n", encoding="utf-8")
    assert rlc_report.read_table_order(table, ["main:kron@kron"]) == \
        {"main:kron@kron": 0}


def test_load_notes_ignores_a_missing_file(tmp_path):
    assert rlc_report.load_notes(None) == {}
    assert rlc_report.load_notes(tmp_path / "missing.csv") == {}


def test_load_do_bench_reasons_reads_the_marker(tmp_path):
    log = tmp_path / "r1_a" / "main" / "kron" / "run.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "BENCHMARK_DIRECT_NO_CUDAGRAPH op=kron phase=host_sync reason=_HostSync "
        "detail='tensor.item()' \n"
        "BENCHMARK_DIRECT_NO_CUDAGRAPH op=kron phase=warmup reason=OutOfResources "
        "detail='out of memory'\n", encoding="utf-8")
    assert rlc_report.load_do_bench_reasons(tmp_path) == {
        "main/kron": "host 同步；预热显存不足"}
    assert rlc_report.load_do_bench_reasons(None) == {}
    assert rlc_report.load_do_bench_reasons(tmp_path / "missing") == {}


def test_describe_do_bench_covers_the_phases():
    assert rlc_report.describe_do_bench("host_sync", "_HostSync") == "host 同步"
    assert rlc_report.describe_do_bench("capability", "GraphUnsupported") == "平台无 graph API"
    assert rlc_report.describe_do_bench("warmup", "OutOfResources") == "预热显存不足"
    assert rlc_report.describe_do_bench("capture", "AcceleratorError") == \
        "capture 失败: AcceleratorError"


def test_summarizing_appends_the_do_bench_reason():
    rows = [_row(gap=-0.2, flags="do_bench", verdict="slower")]
    stats = rlc_report.summarizing(rows, {}, None, {}, "main:kron@kron", "bf16",
                                   {"main/kron": "host 同步"})
    assert "do_bench 计时（host 同步）" in stats["note"]
