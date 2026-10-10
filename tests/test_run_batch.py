import argparse

import pytest

import run_batch


def _row(repo, op, op_file, collected="yes", marked_skip=""):
    return {"repo": repo, "op": op, "op_file": op_file,
            "collected": collected, "marked_skip": marked_skip}


INVENTORY = {
    "main": [
        _row("main", "abs", "abs"),
        _row("main", "perf_x", "a"),
        _row("main", "perf_x", "FLA/b"),
        _row("main", "skipped", "skipped", marked_skip="yes"),
        _row("main", "npu_only", "npu_only", collected="no"),
        _row("main", "dismissed", "dismissed"),
    ],
    "vllm": [_row("vllm", "abs", "abs")],
}


def _args(tmp_path, **kw):
    dismiss = tmp_path / "dismiss.txt"
    dismiss.write_text("# comment\ndismissed\nvllm:nothing\n")
    base = dict(all=False, ops=None, ops_file=None, repos=None, dismiss=dismiss,
                include_marked_skip=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_load_dismiss(tmp_path):
    assert run_batch.load_dismiss(_args(tmp_path).dismiss) == {
        ("main", "dismissed"), ("vllm", "nothing")}


def test_select_all_filters_and_names_duplicates(tmp_path):
    cases = run_batch.resolve_selection(_args(tmp_path, all=True), INVENTORY)
    assert [c.key for c in cases] == [
        "main:abs@abs", "main:perf_x@a", "main:perf_x@FLA/b", "vllm:abs@abs"]
    assert [c.run_name for c in cases] == ["abs", "perf_x@a", "perf_x@FLA_b", "abs"]


def test_select_specs(tmp_path):
    cases = run_batch.resolve_selection(
        _args(tmp_path, ops=["abs", "vllm:abs", "perf_x@FLA/b"]), INVENTORY)
    assert [c.key for c in cases] == ["main:abs@abs", "vllm:abs@abs", "main:perf_x@FLA/b"]


@pytest.mark.parametrize("spec", ["perf_x", "nope", "other:abs", "abs@nope"])
def test_select_rejects_bad_specs(tmp_path, spec):
    with pytest.raises(SystemExit):
        run_batch.resolve_selection(_args(tmp_path, ops=[spec]), INVENTORY)


def test_partition_is_stable_and_complete(tmp_path):
    cases = run_batch.resolve_selection(_args(tmp_path, all=True), INVENTORY)
    first = run_batch.partition(cases, ["0", "1", "2"])
    again = run_batch.partition(list(reversed(cases)), ["0", "1", "2"])
    assert {g: sorted(c.key for c in cs) for g, cs in first.items()} == \
        {g: sorted(c.key for c in cs) for g, cs in again.items()}
    assert sorted(c.key for cs in first.values() for c in cs) == sorted(c.key for c in cases)


SMI_GPUS = "0, GPU-aaa, 3\n1, GPU-bbb, 1034\n2, GPU-ccc, 20480\n"
SMI_APPS = "GPU-bbb, 4242, python, 1024\nGPU-bbb, 4243, /usr/bin/foo, 10\n"


def test_gpu_parsing():
    gpus = run_batch.parse_gpus(SMI_GPUS)
    assert gpus["0"] == gpus["GPU-aaa"] == ("GPU-aaa", 3)
    assert run_batch.parse_compute_apps(SMI_APPS, "GPU-aaa") == []
    assert run_batch.parse_compute_apps(SMI_APPS, "GPU-bbb") == [
        "pid 4242 python (1024 MiB)", "pid 4243 /usr/bin/foo (10 MiB)"]


def test_gpu_users(monkeypatch):
    out = {"--query-gpu=index,uuid,memory.used": SMI_GPUS}
    monkeypatch.setattr(run_batch, "_smi", lambda args: out.get(args[0], SMI_APPS))
    assert run_batch.gpu_users("0") == []
    assert len(run_batch.gpu_users("1")) == 2
    assert run_batch.gpu_users("2") == ["20480 MiB in use by processes not visible here"]
    assert run_batch.gpu_users("9") is None


def test_wait_for_idle_gpu_policies(monkeypatch):
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu: ["pid 1 x (1 MiB)"])
    assert run_batch.wait_for_idle_gpu("0", "ignore") == ""
    assert run_batch.wait_for_idle_gpu("0", "warn") == "gpu_busy"
    calls = iter([["pid 1 x (1 MiB)"], []])
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu: next(calls))
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: None)
    assert run_batch.wait_for_idle_gpu("0", "wait").startswith("gpu_wait=")


def test_parse_env():
    assert run_batch.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(SystemExit):
        run_batch.parse_env(["novalue"])


def test_replay_dir(tmp_path):
    case = run_batch.Case("vllm", "abs", "abs", "abs")
    args = argparse.Namespace(replay_root=tmp_path)
    assert run_batch.replay_dir(case, args) == tmp_path / "vllm" / "abs"
    (tmp_path / "run.log").write_text("")
    assert run_batch.replay_dir(case, args) == tmp_path  # a single-op record
    assert run_batch.replay_dir(case, argparse.Namespace(replay_root=None)) is None
