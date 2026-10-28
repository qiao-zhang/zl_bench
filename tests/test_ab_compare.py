import csv
import json

import ab_compare

HEADER = "Operator: softmax  (dtype=torch.bfloat16, mode=kernel, level=core)\n"


def _log(lats, failed=()):
    lines = [HEADER]
    for i, lat in enumerate(lats):
        if i in failed:
            lines.append("FAILED   N/A   N/A   N/A  [torch.Size([8])]\n")
        else:
            lines.append(f"SUCCESS  0.010000  {lat:.6f}  1.000  [torch.Size([{i + 1}])]\n")
    return "".join(lines)


def test_load_rows_keeps_positions_across_failed_rows(tmp_path):
    (tmp_path / "run.log").write_text(_log([0.1, 0.2, 0.3], failed={1}))
    rows = ab_compare.load_rows(tmp_path)
    assert sorted(rows) == [("softmax", "torch.bfloat16", 0), ("softmax", "torch.bfloat16", 2)]
    assert rows[("softmax", "torch.bfloat16", 2)][0] == 0.3


def test_nocg_tables():
    text = ("BENCHMARK_DIRECT_NO_CUDAGRAPH op=softmax phase=host_sync\n" + _log([0.1])
            + HEADER.replace("bfloat16", "float16"))
    assert ab_compare.nocg_tables(text) == {("softmax", "torch.bfloat16")}


def test_nocg_tables_marks_every_table_of_a_process_wide_fallback():
    """`phase=disabled` / `phase=capability` are reported once per process (one
    process, one op), so every table of that run.log used plain do_bench, not
    just the first one."""
    for phase in ("disabled", "capability"):
        text = (f"BENCHMARK_DIRECT_NO_CUDAGRAPH op=softmax phase={phase} reason=X\n"
                + _log([0.1]) + HEADER.replace("bfloat16", "float16"))
        assert ab_compare.nocg_tables(text) == {("softmax", "torch.bfloat16"),
                                                ("softmax", "torch.float16")}


def test_parse_range():
    assert ab_compare.parse_range(None) is None
    assert ab_compare.parse_range("7-12") == (7, 12)
    assert ab_compare.parse_range("3") == (3, 3)
    assert ab_compare.parse_range("-6") == (1, 6)
    assert ab_compare.parse_range("7-")[0] == 7


def _ab_dir(tmp_path, rounds, a_lat, b_lat, skip_b=()):
    """An ab_run.py output dir with one op of two rows; row 1 is 10% faster on B."""
    out = tmp_path / "ab"
    for k in range(1, rounds + 1):
        for side, lat in (("a", a_lat), ("b", b_lat)):
            if side == "b" and k in skip_b:
                continue
            run = out / f"r{k}_{side}" / "main" / "softmax"
            run.mkdir(parents=True)
            jitter = 1 + 0.001 * k
            (run / "run.log").write_text(_log([0.1 * jitter, lat * jitter]))
            (out / f"r{k}_{side}" / "ops_status.csv").write_text(
                "repo,op,op_file,run_name,status\nmain,softmax,softmax,softmax,PASS\n")
    (out / "ab.json").write_text(json.dumps({"record": "rec", "sides": {}}))
    return out


def test_analyze_pairs_rounds_and_applies_the_gate(tmp_path):
    out = _ab_dir(tmp_path, 6, 0.2, 0.2 / 1.1, skip_b={3})
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "rows.tsv").write_text(
        "run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n"
        "main/softmax\tmain:softmax@softmax\tsoftmax\ttorch.bfloat16\t0\t1\t0\t\n"
        "main/softmax\tmain:softmax@softmax\tsoftmax\ttorch.bfloat16\t1\t2\t1\tH1\n")
    res = {r.row.row: r for r in ab_compare.analyze(out, gate)}
    assert res[0].row.rounds == [1, 2, 4, 5, 6]  # round 3 has no B side
    assert res[0].gate == "unchanged" and res[1].gate == "changed"
    assert res[1].stats.verdict == "undecided"  # 5 rounds < MIN_ROUNDS
    res = {r.row.row: r for r in ab_compare.analyze(out, gate, min_rounds=5)}
    assert res[0].stats.verdict == "same"
    assert res[1].stats.verdict == "faster" and abs(res[1].stats.gap - 0.1) < 1e-4
    only_late = ab_compare.analyze(out, gate, (4, 6))
    assert only_late[0].row.rounds == [4, 5, 6]
    assert ab_compare.op_labels(out) == {"main/softmax": "main:softmax@softmax"}


# ---------------- accuracy gate ----------------

def test_accuracy_of_reads_each_side():
    manifest = {"accuracy": {"sides": {"a": {"main/softmax": {"status": "pass"}},
                                       "b": {"main/softmax": {"status": "fail"}}}}}
    assert ab_compare.accuracy_of(manifest, "main/softmax") == "a:pass,b:fail"
    assert ab_compare.accuracy_of({}, "main/softmax") == ""


def test_report_accuracy_shows_counts_and_skipped_ops(capsys):
    """The gate decides whether an op has an A/B at all, so the report names the ops it
    dropped and how many are unverified (no_unit test upstream)."""
    manifest = {"accuracy": {
        "tests_dir": "tests",
        "sides": {"a": {"main/abs": {"status": "pass"}, "main/add": {"status": "pass"}},
                  "b": {"main/abs": {"status": "no_test"}, "main/add": {"status": "fail"}}},
        "skipped": {"main/add": "b:fail (FAILED tests/test_add.py::test_add - mismatch)"}}}
    ab_compare.report_accuracy(manifest, {"main/add": "main:add@add"})
    out = capsys.readouterr().out
    assert "accuracy gate (tests/ unit test per side): 2 ops checked" in out
    assert "A: pass 2" in out and "B: fail 1, no_test 1" in out
    assert "no_test: 1" in out
    assert "skipped A/B (a side did not pass): 1" in out
    assert "main:add@add" in out and "b:fail" in out


def test_report_accuracy_is_silent_without_a_gate(capsys):
    ab_compare.report_accuracy({"record": "rec"}, {})
    assert capsys.readouterr().out == ""


def test_write_csv_carries_the_accuracy_column(tmp_path):
    out = _ab_dir(tmp_path, 6, 0.2, 0.2 / 1.1)
    manifest = {"accuracy": {"sides": {"a": {"main/softmax": {"status": "pass"}},
                                       "b": {"main/softmax": {"status": "no_test"}}}}}
    path = tmp_path / "rows.csv"
    ab_compare.write_csv(path, ab_compare.analyze(out, None, min_rounds=5),
                         ab_compare.op_labels(out), manifest)
    rows = list(csv.DictReader(path.open(newline="")))
    assert rows and {r["accuracy"] for r in rows} == {"a:pass,b:no_test"}
