import argparse

import pytest

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
