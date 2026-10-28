import argparse
import json

import pytest

import ab_compare
import ab_run
import run_batch


def _args(gpus=None, a_env=None, b_env=None):
    return argparse.Namespace(gpus=gpus, sides={
        "a": {"name": "A", "pythonpath": "", "env": a_env or {}},
        "b": {"name": "B", "pythonpath": "", "env": b_env or {}},
    })


def test_auto_gpus_reads_the_sides_environment(monkeypatch):
    """The cards come from the sides the ops run under, not from this shell: a
    parent CUDA_VISIBLE_DEVICES must not win over the sides' MetaX list."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    args = _args(a_env={"MACA_VISIBLE_DEVICES": "2,3"},
                 b_env={"MACA_VISIBLE_DEVICES": "2,3"})
    assert ab_run.auto_gpus(args) == ["2", "3"]


def test_auto_gpus_takes_an_explicit_list(monkeypatch):
    monkeypatch.setenv("MACA_VISIBLE_DEVICES", "2,3")
    assert ab_run.auto_gpus(_args(gpus="4,5")) == ["4", "5"]


def test_auto_gpus_accepts_the_same_cards_in_another_order(monkeypatch):
    """The mask order is irrelevant: every op is pinned to a single card, so
    A=2,3 and B=3,2 select the same two cards. The A side's order is kept."""
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = _args(a_env={"MACA_VISIBLE_DEVICES": "2,3"},
                 b_env={"MACA_VISIBLE_DEVICES": "3,2"})
    assert ab_run.auto_gpus(args) == ["2", "3"]


def test_auto_gpus_rejects_a_repeated_card(monkeypatch):
    """Same set, different size: a repeated index is not the same card list."""
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = _args(a_env={"MACA_VISIBLE_DEVICES": "2,2"},
                 b_env={"MACA_VISIBLE_DEVICES": "2"})
    with pytest.raises(SystemExit):
        ab_run.auto_gpus(args)


def test_auto_gpus_rejects_sides_that_see_different_cards(monkeypatch):
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = _args(a_env={"MACA_VISIBLE_DEVICES": "2,3"}, b_env={"MACA_VISIBLE_DEVICES": "4"})
    with pytest.raises(SystemExit) as exc:
        ab_run.auto_gpus(args)
    assert "different GPUs" in str(exc.value)


def test_main_dispatches_the_sides_gpu_list(monkeypatch, tmp_path):
    args = argparse.Namespace(
        gpus=None, out=tmp_path, record=tmp_path, gate=None, add_rounds=0,
        until_decided=False, max_rounds=14, step=4,
        sides={"a": {"name": "A", "pythonpath": "", "env": {"MACA_VISIBLE_DEVICES": "2,3"}},
               "b": {"name": "B", "pythonpath": "", "env": {"MACA_VISIBLE_DEVICES": "2,3"}}})
    monkeypatch.setattr(ab_run, "parse_args", lambda: args)
    monkeypatch.setattr(run_batch, "load_inventory", lambda: {})
    monkeypatch.setattr(run_batch, "resolve_selection",
                        lambda a, inv: [run_batch.Case("main", "abs", "abs", "abs")])
    monkeypatch.setattr(ab_run, "check_manifest", lambda a: None)
    monkeypatch.setattr(ab_run, "done_rounds", lambda out: {})
    monkeypatch.setattr(ab_run, "record_cost", lambda rec: {})
    monkeypatch.setattr(ab_run, "plan", lambda cases, a, done: [])
    seen = {}

    class FakeRunner:
        failed = []

        def __init__(self, a):
            pass

        def run(self, items, gpus, cost):
            seen["gpus"] = gpus

        def summarize(self):
            pass

    monkeypatch.setattr(ab_run, "Runner", FakeRunner)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    ab_run.main()
    assert seen["gpus"] == ["2", "3"]


# ---------------- accuracy gate ----------------

def _checkout(tmp_path, label="main"):
    """A fake upstream checkout: only the tests/ dir matters to the gate."""
    root = tmp_path / label
    (root / "tests").mkdir(parents=True)
    return root


def _case(repo="main", op="softmax", op_file=None):
    return run_batch.Case(repo, op, op_file or op, op)


def _gate_args(tmp_path):
    return argparse.Namespace(
        out=tmp_path / "ab", accuracy_timeout=60, use_flagtune=0,
        sides={"a": {"name": "A", "pythonpath": "", "env": {}},
               "b": {"name": "B", "pythonpath": "", "env": {}}})


def _manifest_args(tmp_path, **kw):
    base = dict(out=tmp_path / "ab", record=tmp_path / "rec", dtypes="", metrics="",
                shape_file=None, use_flagtune=0, accuracy_gate=False,
                sides={"a": {"name": "A", "pythonpath": "", "env": {}},
                       "b": {"name": "B", "pythonpath": "", "env": {}}})
    base.update(kw)
    return argparse.Namespace(**base)


def test_accuracy_env_uses_the_measured_tuning_space(monkeypatch, tmp_path):
    """--use-flagtune reaches the unit test as USE_FLAGTUNE, exactly as run_batch._spawn
    passes it to the measured runs: the upstream benchmark picks its tuning space from
    that variable, so a gate run must not verify a different configuration than the
    A/B it stands for — neither a shell value nor a side's own --a-env may win."""
    monkeypatch.setenv("USE_FLAGTUNE", "0")  # the shell value the CLI overrides
    args = _gate_args(tmp_path)
    args.use_flagtune = 1
    assert ab_run.accuracy_env(args, "a", "0")["USE_FLAGTUNE"] == "1"

    args = _gate_args(tmp_path)
    args.sides["a"]["env"] = {"USE_FLAGTUNE": "1"}  # a side that asks for the other space
    args.use_flagtune = 0
    assert ab_run.accuracy_env(args, "a", "0")["USE_FLAGTUNE"] == "0"


def test_check_manifest_resumes_a_dir_recorded_before_metrics(tmp_path):
    """ab.json written before --metrics existed has no such key; resuming that dir
    with the default metric list must not look like a different setup."""
    args = _manifest_args(tmp_path)
    args.out.mkdir()
    (args.out / ab_compare.MANIFEST).write_text(json.dumps({
        "record": str(args.record), "sides": args.sides, "dtypes": "",
        "shape_file": "", "use_flagtune": 0}))
    ab_run.check_manifest(args)  # no SystemExit
    # An explicitly different metric list still refuses to continue that dir.
    with pytest.raises(SystemExit, match="different metrics"):
        ab_run.check_manifest(_manifest_args(tmp_path, metrics="latency"))


def test_check_manifest_refuses_another_setup(tmp_path):
    args = _manifest_args(tmp_path)
    ab_run.check_manifest(args)  # writes ab.json
    assert json.loads((args.out / ab_compare.MANIFEST).read_text())["metrics"] == ""
    with pytest.raises(SystemExit, match="different dtypes"):
        ab_run.check_manifest(_manifest_args(tmp_path, dtypes="bfloat16"))
    with pytest.raises(SystemExit, match="different use_flagtune"):
        ab_run.check_manifest(_manifest_args(tmp_path, use_flagtune=1))


def _row(status, detail=""):
    return {"status": status, "target": "t", "return_code": "0", "elapsed_ms": "1",
            "log": "", "detail": detail}


def test_test_target_finds_the_upstream_unit_test(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    (root / "tests" / "test_softmax.py").write_text("def test_softmax():\n    pass\n")
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    assert ab_run.test_target(_case()) == "tests/test_softmax.py::test_softmax"


def test_test_target_is_none_when_the_function_is_missing(monkeypatch, tmp_path):
    """The benchmark inventory is not 1:1 with the tests (perf_ and inplace variants
    share one test), so a file without that function means no_test, not a failure."""
    root = _checkout(tmp_path)
    (root / "tests" / "test_abs.py").write_text("def test_abs():\n    pass\n")
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    assert ab_run.test_target(_case(op="abs_inplace", op_file="abs")) is None


def test_test_target_is_none_without_a_test_file(monkeypatch, tmp_path):
    monkeypatch.setitem(ab_run.REPOS, "main", _checkout(tmp_path))
    assert ab_run.test_target(_case()) is None


def test_test_target_keeps_an_op_file_subdir(monkeypatch, tmp_path):
    """op_file keeps a benchmark subdir (benchmark/test_FLA/test_x.py), so the test sits
    under tests/test_FLA/."""
    root = _checkout(tmp_path)
    (root / "tests" / "test_FLA").mkdir()
    (root / "tests" / "test_FLA" / "test_x.py").write_text("def test_x():\n    pass\n")
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    assert ab_run.test_target(_case(op="x", op_file="FLA/test_x")) \
        == "tests/test_FLA/test_x.py::test_x"


def test_test_target_refuses_a_path_outside_tests(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    (root / "test_escape.py").write_text("def test_escape():\n    pass\n")
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    assert ab_run.test_target(_case(op="escape", op_file="../test_escape")) is None


@pytest.mark.parametrize("rc,timed_out,text,status", [
    (0, False, "1 passed in 0.01s", ab_run.GATE_PASS),
    (0, False, "2 passed, 1 skipped in 0.01s", ab_run.GATE_PASS),
    (0, False, "1 skipped in 0.01s", ab_run.GATE_NO_TEST),  # nothing was verified
    (0, False, "", ab_run.GATE_NO_TEST),                    # no summary at all
    (1, False, "1 failed in 0.1s", ab_run.GATE_FAIL),
    (5, False, "no tests ran in 0.01s", ab_run.GATE_NO_TEST),
    (2, False, "", ab_run.GATE_ERROR),     # interrupted
    (4, False, "", ab_run.GATE_ERROR),     # usage error, e.g. a conftest that wants a flag
    (0, True, "1 passed in 0.01s", ab_run.GATE_TIMEOUT),
])
def test_gate_status(rc, timed_out, text, status):
    assert ab_run.gate_status(rc, timed_out, text) == status


def test_pytest_counts_reads_the_last_summary_line():
    assert ab_run.pytest_counts("....\n3 passed, 2 skipped, 1 error in 1.2s\n") == {
        "passed": 3, "skipped": 2, "error": 1}
    assert ab_run.pytest_counts("no summary here") == {}


def test_gate_detail_keeps_pytests_own_summary_lines():
    failed = "..\nFAILED tests/test_add.py::test_add - AssertionError: tensors differ\n"
    assert ab_run.gate_detail(failed, ab_run.GATE_FAIL).startswith("FAILED tests/test_add.py")
    skipped = "s\nSKIPPED [2] tests/test_abs.py:12: needs fp8\n1 skipped in 0.01s\n"
    assert ab_run.gate_detail(skipped, ab_run.GATE_NO_TEST).startswith("SKIPPED [2]")
    assert ab_run.gate_detail(failed, ab_run.GATE_PASS) == ""


def test_run_accuracy_runs_no_subprocess_without_a_test(monkeypatch, tmp_path):
    monkeypatch.setitem(ab_run.REPOS, "main", _checkout(tmp_path))
    monkeypatch.setattr(ab_run.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("no subprocess without a unit test"))
    row = ab_run.run_accuracy(_case(), "a", _gate_args(tmp_path), "0")
    assert row["status"] == ab_run.GATE_NO_TEST and row["target"] == ""


def test_run_accuracy_records_the_side_and_the_log(monkeypatch, tmp_path):
    """The unit test runs under the side's own environment, in its own repo checkout,
    and its log is kept under accuracy/<side>/."""
    root = _checkout(tmp_path)
    (root / "tests" / "test_softmax.py").write_text("def test_softmax():\n    pass\n")
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    monkeypatch.setitem(ab_run.REPOS, "main", root)
    seen = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
        seen.update(cmd=cmd, cwd=cwd, env=env)
        stdout.write("1 failed\nFAILED tests/test_softmax.py::test_softmax - mismatch\n")
        return FakeProc()

    args = _gate_args(tmp_path)
    args.sides["b"]["pythonpath"] = "/path/dev/python"
    monkeypatch.setattr(ab_run.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ab_run.run_batch, "_wait", lambda proc, timeout: (1, False))
    row = ab_run.run_accuracy(_case(), "b", args, "3")
    assert seen["cmd"][-1] == "tests/test_softmax.py::test_softmax"
    assert seen["cwd"] == str(root)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert seen["env"]["PYTHONPATH"].startswith("/path/dev/python")
    assert row["status"] == ab_run.GATE_FAIL and "mismatch" in row["detail"]
    assert row["log"] == "accuracy/b/main/softmax/gate.log"
    assert (args.out / row["log"]).is_file()


def test_accuracy_gate_drops_an_op_a_side_fails(monkeypatch, tmp_path):
    args = _gate_args(tmp_path)
    args.out.mkdir()
    keep, drop = _case(op="abs"), _case(op="add")

    def fake(case, side, a, gpu):
        if case.op == "add" and side == "b":
            return _row(ab_run.GATE_FAIL, "FAILED tests/test_add.py::test_add - mismatch")
        return _row(ab_run.GATE_PASS)

    monkeypatch.setattr(ab_run, "run_accuracy", fake)
    kept, skipped = ab_run.accuracy_gate([keep, drop], args, ["0"])
    assert [c.op for c in kept] == ["abs"]
    assert list(skipped) == ["main/add"] and "b:fail" in skipped["main/add"]
    data = json.loads((args.out / "ab.json").read_text())["accuracy"]
    assert data["sides"]["a"]["main/add"]["status"] == ab_run.GATE_PASS
    assert data["sides"]["b"]["main/add"]["status"] == ab_run.GATE_FAIL
    assert list(data["skipped"]) == ["main/add"]


def test_accuracy_gate_reuses_recorded_results(monkeypatch, tmp_path):
    """Resuming and --add-rounds must not rerun a gate that already passed."""
    args = _gate_args(tmp_path)
    args.out.mkdir()
    (args.out / "ab.json").write_text(json.dumps({
        "record": "rec",
        "accuracy": {"sides": {"a": {"main/softmax": {"status": ab_run.GATE_PASS}},
                               "b": {"main/softmax": {"status": ab_run.GATE_NO_TEST}}}}}))
    monkeypatch.setattr(ab_run, "run_accuracy",
                        lambda *a: pytest.fail("a recorded gate ran again"))
    kept, skipped = ab_run.accuracy_gate([_case()], args, ["0"])
    assert [c.op for c in kept] == ["softmax"] and skipped == {}
    data = json.loads((args.out / "ab.json").read_text())
    assert data["record"] == "rec"  # the manifest fields survive the rewrite
    assert data["accuracy"]["sides"]["b"]["main/softmax"]["status"] == ab_run.GATE_NO_TEST


def test_main_measures_only_the_ops_that_pass_the_gate(monkeypatch, tmp_path):
    good, bad = _case(op="abs"), _case(op="add")
    args = argparse.Namespace(
        gpus="0", out=tmp_path, record=tmp_path, gate=None, add_rounds=0, until_decided=False,
        max_rounds=14, step=2, accuracy_gate=True, accuracy_timeout=60, dtypes="", metrics="",
        op_timeout=60, shape_file=None, use_flagtune=0, busy_gpu="ignore",
        sides={"a": {"name": "A", "pythonpath": "", "env": {}},
               "b": {"name": "B", "pythonpath": "", "env": {}}})
    monkeypatch.setattr(ab_run, "parse_args", lambda: args)
    monkeypatch.setattr(run_batch, "load_inventory", lambda: {})
    monkeypatch.setattr(run_batch, "resolve_selection", lambda a, inv: [good, bad])
    monkeypatch.setattr(ab_run, "check_manifest", lambda a: None)
    monkeypatch.setattr(ab_run, "done_rounds", lambda out: {})
    monkeypatch.setattr(ab_run, "record_cost", lambda rec: {})
    monkeypatch.setattr(ab_run, "accuracy_gate",
                        lambda cases, a, gpus: ([good], {"main/add": "b:fail"}))
    seen = {}

    def fake_plan(cases, a, done):
        seen["cases"] = cases
        return []

    monkeypatch.setattr(ab_run, "plan", fake_plan)

    class FakeRunner:
        failed = []

        def __init__(self, a):
            pass

        def run(self, items, gpus, cost):
            pass

        def summarize(self):
            pass

    monkeypatch.setattr(ab_run, "Runner", FakeRunner)
    ab_run.main()
    assert [c.op for c in seen["cases"]] == ["abs"]
