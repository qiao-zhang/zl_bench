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


def test_nocg_tables_ignores_the_plugin_banner():
    """The cudagraph plugin's startup banner names the marker without reporting a
    fallback ("... falls back to do_bench with a BENCHMARK_DIRECT_NO_CUDAGRAPH
    marker ..."). Reading the name alone marked the first table of every run.log as
    plain do_bench, which turned every op's note into a do_bench warning."""
    banner = ("[cudagraph-plugin] triton.testing.do_bench → CUDA-graph timing (at most 1000 "
              "calls per graph); a call that cannot be captured falls back to do_bench with a "
              "BENCHMARK_DIRECT_NO_CUDAGRAPH marker (skipped for the rest of the op when the "
              "platform has no graph support)\n")
    text = (banner + _log([0.1])
            + banner + HEADER.replace("bfloat16", "float16") + _log([0.2]))
    assert ab_compare.nocg_tables(text) == set()


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


def _mark_direct(run_dir, phase="host_sync"):
    """Prepend the fallback marker, so nocg_tables() marks that run's table."""
    log = run_dir / "run.log"
    log.write_text(f"BENCHMARK_DIRECT_NO_CUDAGRAPH op=softmax phase={phase} reason=X\n"
                   + log.read_text())


def test_timer_mismatch_only_when_the_two_sides_differ(tmp_path):
    """One side captured a graph, the other timed with plain do_bench: the gap mixes two
    clocks, so the row is flagged."""
    out = _ab_dir(tmp_path, 6, 0.2, 0.2 / 1.1)

    def row1():
        return [r for r in ab_compare.analyze(out, min_rounds=5) if r.row.row == 1][0].row

    assert not ab_compare.timer_mismatch(row1())  # both sides on graphs
    for k in range(1, 7):
        _mark_direct(out / f"r{k}_a" / "main" / "softmax")
    direct = row1()
    assert ab_compare.timer_mismatch(direct)
    assert direct.nocg_a and not direct.nocg_b and direct.nocg
    for k in range(1, 7):
        _mark_direct(out / f"r{k}_b" / "main" / "softmax")
    both = row1()
    assert both.nocg and not ab_compare.timer_mismatch(both)


def test_drop_timer_mismatch_removes_the_rows(tmp_path):
    out = _ab_dir(tmp_path, 6, 0.2, 0.2 / 1.1)
    for k in range(1, 7):
        _mark_direct(out / f"r{k}_a" / "main" / "softmax")
    assert ab_compare.analyze(out, min_rounds=5)
    assert ab_compare.analyze(out, min_rounds=5, drop_timer_mismatch=True) == []


def test_report_and_csv_flag_timer_mismatch(tmp_path, capsys):
    out = _ab_dir(tmp_path, 6, 0.2, 0.2 / 1.1)
    for k in range(1, 7):
        _mark_direct(out / f"r{k}_a" / "main" / "softmax")
    results = ab_compare.analyze(out, min_rounds=5)
    manifest = json.loads((out / "ab.json").read_text())
    ab_compare.report(results, ab_compare.op_labels(out), manifest, False, False)
    printed = capsys.readouterr().out
    assert "timer mismatch (one side graph, the other plain do_bench)" in printed
    assert "[do_bench, timer_mismatch]" in printed

    path = tmp_path / "rows.csv"
    ab_compare.write_csv(path, results, ab_compare.op_labels(out), manifest)
    row = next(csv.DictReader(path.open(newline="")))
    assert "timer_mismatch" in row["flags"].split(";")
    assert row["timer"] == "mixed"
    assert row["timer_rounds"] == "1:a,2:a,3:a,4:a,5:a,6:a"


def test_compare_pair_counts_the_mismatched_rows(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "run.log").write_text(
        "BENCHMARK_DIRECT_NO_CUDAGRAPH op=softmax phase=host_sync reason=X\n" + _log([0.1, 0.2]))
    (b / "run.log").write_text(_log([0.1, 0.2]))
    _, rows, mismatched = ab_compare.compare_pair(a, b, "softmax", 0.03, False)
    assert (rows, mismatched) == (2, 2)
    assert "timer_mismatch 2 rows" in capsys.readouterr().out


def _ab_dir_marks(tmp_path, rounds, op="softmax"):
    """An ab_run.py-shaped dir from {round: {"a": direct|None, "b": direct|None}}.

    direct=True writes the fallback marker for that side, None means that side has no
    sample in that round (the round then does not count for the row). B is 10% faster
    so a decided verdict exists."""
    out = tmp_path / "ab"
    for k, sides in rounds.items():
        for side, direct in sides.items():
            if direct is None:
                continue
            run = out / f"r{k}_{side}" / "main" / op
            run.mkdir(parents=True)
            pre = (f"BENCHMARK_DIRECT_NO_CUDAGRAPH op={op} phase=host_sync reason=X\n"
                   if direct else "")
            lat = 0.1 if side == "a" else 0.09
            (run / "run.log").write_text(pre + _log([lat]))
            (out / f"r{k}_{side}" / "ops_status.csv").write_text(
                f"repo,op,op_file,run_name,status\nmain,{op},{op},{op},PASS\n")
    (out / "ab.json").write_text(json.dumps({"record": "rec", "sides": {}}))
    return out


def test_timer_mismatch_is_judged_per_round_not_on_the_union(tmp_path):
    """Review case 1a: round 1 falls back on A, round 2 on B. Every counted round mixes
    two clocks, so the row is mismatched even though both sides fell back somewhere."""
    out = _ab_dir_marks(tmp_path, {1: {"a": True, "b": False}, 2: {"a": False, "b": True}})
    rows = ab_compare.analyze(out, min_rounds=2)
    assert len(rows) == 1
    row = rows[0].row
    assert ab_compare.timer_mismatch(row)
    assert ab_compare.timer_mode(row) == "mixed"
    assert ab_compare.timer_detail(row) == "1:a,2:b"
    assert ab_compare.analyze(out, min_rounds=2, drop_timer_mismatch=True) == []


def test_a_round_only_one_side_finished_does_not_mark_the_row(tmp_path):
    """Review case 1b: rounds 1-6 captured graphs on both sides; round 7 only A ran and
    fell back. Round 7 does not count, so it must not turn the six good rounds into a
    mismatch (which would silently drop them)."""
    rounds = {k: {"a": False, "b": False} for k in range(1, 7)}
    rounds[7] = {"a": True, "b": None}
    out = _ab_dir_marks(tmp_path, rounds)
    rows = ab_compare.analyze(out, min_rounds=5)
    assert len(rows) == 1
    row = rows[0].row
    assert row.rounds == [1, 2, 3, 4, 5, 6]
    assert not ab_compare.timer_mismatch(row)
    assert ab_compare.timer_mode(row) == "graph"
    assert len(ab_compare.analyze(out, min_rounds=5, drop_timer_mismatch=True)) == 1


def test_confirm_refuses_a_recheck_timed_differently(tmp_path):
    """Review case 2: the main rounds are clean, the recheck rounds fall back on A only.
    Such a recheck cannot confirm the verdict, and its samples are not merged."""
    rounds = {k: {"a": False, "b": False} for k in range(1, 7)}
    for k in range(7, 10):
        rounds[k] = {"a": True, "b": False}
    out = _ab_dir_marks(tmp_path, rounds)
    res = ab_compare.confirm(out, None, (7, 9), 0.05, drop_timer_mismatch=True,
                             min_rounds=5, threshold=0.05, alpha=0.05, clear=0.05)
    assert res[0].stats.verdict == "faster"
    assert res[0].confirm == "timer_mismatch"
    assert res[0].new is not None                  # 复测统计仍给出，供参考
    assert len(res[0].row.rounds) == 6             # 样本没有合并
    assert res[0].row.direct_rounds == tuple((k, False, False) for k in range(1, 7))


def test_confirm_merges_the_marks_of_the_recheck_rounds(tmp_path):
    """The positive half of case 2: a clean recheck confirms and its marks are merged, so
    the report/CSV describe all nine rounds."""
    rounds = {k: {"a": False, "b": False} for k in range(1, 10)}
    out = _ab_dir_marks(tmp_path, rounds)
    res = ab_compare.confirm(out, None, (7, 9), 0.05, min_rounds=5, threshold=0.05,
                             alpha=0.05, clear=0.05)
    assert res[0].confirm == "confirmed"
    assert len(res[0].row.rounds) == 9
    assert res[0].row.direct_rounds == tuple((k, False, False) for k in range(1, 10))
    assert ab_compare.timer_mode(res[0].row) == "graph"


def _mark_fallback(run_dir):
    """Prepend the replay-fallback marker, so load_samples() flags that run."""
    log = run_dir / "run.log"
    log.write_text("AUTOTUNE_REPLAY_FALLBACK op=softmax config=0 sha=deadbeef\n" + log.read_text())


def test_confirm_keeps_a_recheck_only_replay_fallback(tmp_path):
    """AUTOTUNE_REPLAY_FALLBACK names no round, so a recheck that tripped it must keep the
    warning once its samples are merged: otherwise the merged row reads as clean although
    it now contains live-tuned points."""
    rounds = {k: {"a": False, "b": False} for k in range(1, 10)}
    out = _ab_dir_marks(tmp_path, rounds)
    for k in range(7, 10):
        _mark_fallback(out / f"r{k}_a" / "main" / "softmax")
    main = ab_compare.analyze(out, None, exclude=(7, 9), min_rounds=5)
    assert not main[0].row.fallback            # 主轮次本身干净
    res = ab_compare.confirm(out, None, (7, 9), 0.05, min_rounds=5, threshold=0.05,
                             alpha=0.05, clear=0.05)
    assert res[0].confirm == "confirmed"
    assert res[0].row.fallback                 # 复测的回退没有丢


def _result(confirm="", *, gap=0.01, gate="changed", nocg=False, run="main/softmax",
            verdict="faster"):
    """A Result for the pure selection helpers (no run dir needed)."""
    row = ab_compare.Row(run, "softmax", "torch.bfloat16", 0, "", [0.1], [0.1], [1],
                         nocg, False, nocg, nocg, ((1, nocg, nocg),))
    stats = ab_compare.ab_stats.RowStats(1, 1, 0.1, 0.1 / (1 + gap), gap, 0.01, verdict)
    return ab_compare.Result(row, stats, gate, confirm)


def test_recheck_ops_keeps_a_short_recheck_but_not_an_unfixable_one():
    """--recheck-list feeds ab_run.py --add-rounds, so a recheck that was merely too short
    belongs in it; a timer mismatch or a failed reproduction does not improve with more
    rounds and stays out (as do controls)."""
    labels = {"main/softmax": "main:softmax@softmax"}
    results = [_result(""), _result("insufficient_rounds"), _result("timer_mismatch"),
               _result("not_reproduced"), _result("", gate="unchanged"),
               _result("insufficient_rounds", gap=0.5)]   # 0.5 的 gap 不算 weak
    ops, rows = ab_compare.recheck_ops(results, labels, 0.02)
    assert (ops, rows) == (["main:softmax@softmax"], 2)


def test_confirm_marks_a_short_recheck_as_insufficient_rounds(tmp_path):
    """Two fresh rounds that agree on their timing are still too few to confirm: the row
    is marked and keeps its recheck statistics, and its samples are not merged. (The old
    early return left confirm empty, indistinguishable from "no recheck arranged".)"""
    rounds = {k: {"a": False, "b": False} for k in range(1, 7)}
    rounds[7] = {"a": False, "b": False}
    rounds[8] = {"a": False, "b": False}
    out = _ab_dir_marks(tmp_path, rounds)
    res = ab_compare.confirm(out, None, (7, 8), 0.05, min_rounds=5, threshold=0.05,
                             alpha=0.05, clear=0.05)[0]
    assert res.stats.verdict == "faster"
    assert res.confirm == "insufficient_rounds"
    assert res.new is not None and res.new.n_a == 2
    assert len(res.row.rounds) == 6                 # 没有合并


def test_confirm_prefers_timer_mismatch_over_a_short_recheck(tmp_path):
    """Timing is judged before the round count: two mismatched rounds already say the
    recheck cannot confirm anything, even though it is also too short."""
    rounds = {k: {"a": False, "b": False} for k in range(1, 7)}
    rounds[7] = {"a": True, "b": False}
    rounds[8] = {"a": True, "b": False}
    out = _ab_dir_marks(tmp_path, rounds)
    res = ab_compare.confirm(out, None, (7, 8), 0.05, min_rounds=5, threshold=0.05,
                             alpha=0.05, clear=0.05)[0]
    assert res.confirm == "timer_mismatch"
    assert res.new is not None


def _marked_result(confirm="", *, a_direct=False, b_direct=False, gap=0.1):
    """A Result with explicit per-round timing marks, for the CSV writer."""
    row = ab_compare.Row("main/softmax", "softmax", "torch.bfloat16", 0, "",
                         [0.1], [0.1], [1], a_direct or b_direct, False,
                         a_direct, b_direct, ((1, a_direct, b_direct),))
    stats = ab_compare.ab_stats.RowStats(1, 1, 0.1, 0.1, gap, 0.01, "faster")
    return ab_compare.Result(row, stats, "changed", confirm)


def test_write_csv_marks_unconfirmed_rows_explicitly(tmp_path):
    """A reader that never looks at `confirm` still has to see that a gap is not
    confirmed: the row carries `unconfirmed` in flags and `gap_confirmed=no`."""
    results = [_marked_result(),
               _marked_result(confirm="not_reproduced"),
               _marked_result(confirm="insufficient_rounds"),
               _marked_result(confirm="timer_mismatch"),
               _marked_result(a_direct=True)]      # 主轮次口径不一致
    path = tmp_path / "rows.csv"
    ab_compare.write_csv(path, results, {"main/softmax": "main:softmax@softmax"}, {})
    rows = list(csv.DictReader(path.open(newline="")))
    assert [r["gap_confirmed"] for r in rows] == ["yes", "no", "no", "no", "no"]
    assert rows[0]["flags"] == ""
    assert rows[1]["flags"] == "unconfirmed"
    assert rows[4]["flags"] == "do_bench;timer_mismatch;unconfirmed"


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
