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
    assert rlc_ops.parse_cell("kron") == ("kron", "main", "kron", None, False)
    assert rlc_ops.parse_cell("outer\nvllm:outer") == ("outer", "vllm", "outer", None, True)
    # the table's label and the inventory op differ for two tests that share a label
    assert rlc_ops.parse_cell("uniform_\nmain:uniform_inplace") == \
        ("uniform_", "main", "uniform_inplace", None, True)
    assert rlc_ops.parse_cell("x\nmain:y@z") == ("x", "main", "y", "z", True)
    assert rlc_ops.parse_cell("vllm:foo") == ("vllm:foo", "vllm", "foo", None, True)
    assert rlc_ops.parse_cell("") is None


def test_read_table_skips_the_title_and_header_rows(tmp_path):
    entries = rlc_ops.read_table(_table(tmp_path))
    assert [(e.display, e.repo, e.op) for e in entries] == [
        ("kron", "main", "kron"),
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


def test_resolve_reports_an_ambiguous_op(monkeypatch, tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("#,算子\n1,kron\n", encoding="utf-8")
    inv = {"main": [{"op": "kron", "op_file": "kron", "collected": "yes"},
                    {"op": "kron", "op_file": "kron_other", "collected": "yes"}]}
    monkeypatch.setattr(rlc_ops.run_batch, "load_inventory", lambda: inv)
    specs, report, errors = rlc_ops.resolve(rlc_ops.read_table(path))
    assert specs == []
    assert report[0]["status"] == "ambiguous"
    assert any("ambiguous" in e for e in errors)


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
    assert stats["cvt"] == (None, None)


def test_summarizing_reads_the_gate_convert_layout_count(tmp_path):
    (tmp_path / "base" / "H1").mkdir(parents=True)
    (tmp_path / "new" / "H1").mkdir(parents=True)
    (tmp_path / "base" / "H1" / "k.digest.json").write_text('{"ops": {"ttg.convert_layout": 2}}')
    (tmp_path / "new" / "H1" / "k.digest.json").write_text('{"ops": {"ttg.convert_layout": 0}}')
    gate = {("main/kron", "kron", "torch.bfloat16", 0): ["H1"]}
    stats = rlc_report.summarizing([_row()], gate, tmp_path, {}, "main:kron@kron", "bf16")
    assert stats["cvt"] == (2, 0)
    assert rlc_report.fmt_cvt(stats["cvt"]) == "2→0"


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

