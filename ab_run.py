#!/usr/bin/env python3
"""Repeated-measures A/B: replay one record on two compiler setups, interleaved on one
GPU per op, over several rounds.

A side is a compiler setup: a PYTHONPATH (a FlagTree python/ dir) and/or environment
variables (compile switches). Both sides replay the same record, so they run the same
autotune configs on the same shapes. Each (op, round) is one work item: the op's A and
B runs back to back on one GPU, A first in odd rounds and B first in even rounds, so
per-GPU bias and slow drift cancel within the item and order effects cancel across
rounds. Items share one queue (longest first) over all GPUs; each run is a fresh
process, so rounds are independent samples. ab_compare.py turns the rounds into
per-row verdicts.

  # 6 rounds of the ops the gate selected, one build with a compile switch
  python ab_run.py --record runs/base_multi --ops-file runs/gate_x/changed_ops.txt \\
      --a-env FLAGTREE_XXX=0 --b-env FLAGTREE_XXX=1 --out runs/ab_x --rounds 6
  # two builds
  python ab_run.py --record runs/base_multi --ops softmax \\
      --a-pythonpath /path/base/python --b-pythonpath /path/dev/python --out runs/ab_y
  # then: add rounds to ops with undecided rows, up to 14
  python ab_run.py ... --out runs/ab_x --until-decided --max-rounds 14
  # fresh rounds for an independent check (judge them with ab_compare.py --rounds)
  python ab_run.py ... --out runs/ab_x --ops-file recheck.txt --add-rounds 6

  # same, but run each side's unit test first and skip the ops it fails
  python ab_run.py ... --accuracy-gate --out runs/ab_x
  # ... and point the gate at tests whose function name differs from the op
  python ab_run.py ... --accuracy-gate --accuracy-map ops/accuracy_map.txt --out runs/ab_x

Output: <out>/r<k>_a and <out>/r<k>_b, each laid out like a run_batch.py batch dir
(ops_status.csv, summary.csv, <repo>/<run_name>/), and <out>/ab.json with the sides.
Rerunning with the same --out resumes: finished runs are kept, failed ones rerun.

With --accuracy-gate each side's unit test for the op (tests/test_<op_file>.py::test_<op>
of that repo) runs before the first round, and an op a side does not pass gets no A/B
at all: a compiler that changes the numbers is not a performance result. An op whose
repo has no matching test is not dropped -- it is recorded as no_test, since the
benchmark inventory and the tests are not 1:1. --accuracy-map names the test of the
ops the automatic lookup cannot find (its test function may carry any name), one
`[repo:]op[@op_file] tests/...py[::test_func]` per line. The outcome, including why
an op was skipped, goes to ab.json and into ab_compare.py's report; the no_test ops
are searched for in both checkouts, and the candidates written to
<out>/accuracy_no_test.md and <out>/accuracy_map.suggested.txt (a commented
--accuracy-map draft) -- see accuracy_hints.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ab_compare
import accuracy_hints
import batch_summary
import run_batch
from gen_inventory import REPOS, ZL_BENCH

LOG = "[ab]"
SIDES = ("a", "b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    run_batch.add_selection_args(p)
    s = p.add_argument_group("sides")
    s.add_argument("--record", type=Path, required=True,
                   help="record to replay: a run_batch.py batch dir, or a single-op run dir")
    for side in SIDES:
        S = side.upper()
        s.add_argument(f"--{side}-pythonpath", default="",
                       help=f"PYTHONPATH prepended for side {S} (a FlagTree python/ dir)")
        s.add_argument(f"--{side}-env", action="append", default=[], metavar="K=V",
                       help=f"environment variable for side {S} (repeatable)")
        s.add_argument(f"--{side}-name", default=S, help=f"display name of side {S}")

    r = p.add_argument_group("rounds")
    r.add_argument("--out", type=Path, default=None,
                   help="output dir (default runs/ab_<timestamp>); an existing one is resumed")
    r.add_argument("--rounds", type=int, default=6,
                   help="make sure every selected op has rounds 1..N (default %(default)s)")
    r.add_argument("--add-rounds", type=int, default=0, metavar="N",
                   help="instead run N new rounds per selected op, numbered after the highest "
                        "round in --out, for a check on fresh data")
    r.add_argument("--until-decided", action="store_true",
                   help="afterwards add rounds to ops with undecided rows (ab_compare.py's "
                        "rule) until every row is decided or --max-rounds is reached")
    r.add_argument("--max-rounds", type=int, default=14,
                   help="round cap for --until-decided (default %(default)s)")
    r.add_argument("--step", type=int, default=2,
                   help="rounds added per --until-decided pass (default %(default)s)")
    r.add_argument("--gate", type=Path, default=None,
                   help="ir_gate.py output dir: --until-decided extends only ops with an "
                        "undecided changed row (controls ride along on those ops)")
    ab_compare.rule_args(p)

    e = p.add_argument_group("execution")
    e.add_argument("--gpus", default=None,
                   help="comma-separated GPU list (default: the sides' own environment, "
                        "MACA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES, else every device their "
                        "SMI tool lists; both sides must see the same cards)")
    e.add_argument("--dtypes", default="", help='dtype whitelist, e.g. "bfloat16 float16"')
    e.add_argument("--metrics", default="",
                   help='metric whitelist, e.g. "latency"; empty = upstream default. '
                        '"latency" skips the baseline op, which an A/B of two compiler builds '
                        'does not need (both sides are compared on the gems latency alone)')
    e.add_argument("--shape-file", type=Path, default=None,
                   help="shape set (default: the record's archived shapes.yaml per op)")
    e.add_argument("--op-timeout", type=int, default=3600,
                   help="per-run timeout in seconds, 0 = none (default %(default)s)")
    e.add_argument("--use-flagtune", type=int, default=0, choices=(0, 1))
    e.add_argument("--busy-gpu", choices=("wait", "warn", "ignore"), default="wait",
                   help="when another process uses a GPU: wait (default), warn, or ignore")

    g = p.add_argument_group("accuracy gate")
    g.add_argument("--accuracy-gate", action="store_true",
                   help="before the A/B, run each side's unit test for every op and skip "
                        "the ops a side does not pass; the test is the exact "
                        "tests/test_<op_file>.py::test_<op>, else every test function whose "
                        "name contains it, else the one --accuracy-map names. The outcome "
                        "goes to ab.json and into ab_compare.py's report")
    g.add_argument("--accuracy-map", type=Path, default=None, metavar="FILE",
                   help="the gate's test per op, for the ops the automatic lookup cannot "
                        "locate: one '<[repo:]op[@op_file]> <target> [target...]' per line, "
                        "target a path under that repo's tests/ with an optional ::test_func "
                        "(relative to the repo root), '#' comments")
    g.add_argument("--accuracy-timeout", type=int, default=1800,
                   help="timeout of one unit-test run in seconds, 0 = none (default %(default)s)")
    args = p.parse_args()

    if args.add_rounds and args.until_decided:
        raise SystemExit(f"{LOG} --add-rounds and --until-decided do not mix: run them one after the other")
    args.record = args.record.resolve()
    if not args.record.is_dir():
        raise SystemExit(f"{LOG} no record dir {args.record}")
    if args.shape_file:
        args.shape_file = args.shape_file.resolve()
    args.accuracy_map_file = args.accuracy_map.resolve() if args.accuracy_map else None
    args.accuracy_map = load_accuracy_map(args.accuracy_map_file) if args.accuracy_map_file else {}
    if args.accuracy_map and not args.accuracy_gate:
        print(f"{LOG} note: --accuracy-map only applies together with --accuracy-gate",
              flush=True)
    args.out = (args.out or ZL_BENCH / "runs" / f"ab_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    # absolute paths, so that a resumed run with a relative path matches ab.json
    args.sides = {side: {"name": getattr(args, f"{side}_name"),
                         "pythonpath": os.path.abspath(getattr(args, f"{side}_pythonpath"))
                         if getattr(args, f"{side}_pythonpath") else "",
                         "env": run_batch.parse_env(getattr(args, f"{side}_env"))}
                  for side in SIDES}
    if args.sides["a"]["pythonpath"] == args.sides["b"]["pythonpath"] and \
            args.sides["a"]["env"] == args.sides["b"]["env"]:
        print(f"{LOG} note: both sides are identical, this measures the noise floor", flush=True)
    stacks = side_stacks(args.sides)
    if len(set(stacks.values())) > 1:
        shown = ", ".join(f"{side.upper()}={stacks[side] or 'plain cuda'}" for side in SIDES)
        print(f"{LOG} warning: the sides run on different device stacks ({shown}); "
              "the gap would mix the stacks, not just the compiler switch", flush=True)
    return args


def side_stacks(sides: Dict[str, dict]) -> Dict[str, str]:
    """Each side's device stack (run_batch.accelerator), read from that side's env.

    The idle check and the subprocess use the same per-side environment, so a side
    that sets MACA_PATH / PPU_SDK is checked with its own vendor's tool."""
    return {side: run_batch.accelerator(run_batch.op_env(cfg["env"])) or ""
            for side, cfg in sides.items()}


# ---------------- manifest ----------------

def manifest(args: argparse.Namespace) -> dict:
    out = {"record": str(args.record), "sides": args.sides, "dtypes": args.dtypes,
           "metrics": args.metrics,
           "shape_file": str(args.shape_file or ""), "use_flagtune": args.use_flagtune}
    # Only when on: an existing dir recorded without it stays resumable.
    if getattr(args, "accuracy_gate", False):
        out["accuracy_gate"] = True
    return out


# Fields added after output dirs were already being written: a dir recorded
# without one resumes on that key's default. Only defaults belong here, so a dir
# recorded with a non-default value still refuses a run that asks for another.
LEGACY_DEFAULTS = {"metrics": ""}


def check_manifest(args: argparse.Namespace) -> None:
    """Write ab.json, or make sure a resumed dir was run with the same setup."""
    path = args.out / ab_compare.MANIFEST
    want = manifest(args)
    if path.is_file():
        have = json.loads(path.read_text())
        for key, default in LEGACY_DEFAULTS.items():
            have.setdefault(key, default)
        diff = [k for k in want if have.get(k) != want[k]]
        if diff:
            raise SystemExit(f"{LOG} {args.out} was run with a different {', '.join(diff)}; "
                             "use another --out")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(want, indent=1) + "\n")


# ---------------- accuracy gate ----------------

GATE_DIR = "accuracy"      # <out>/accuracy/<side>/<repo>/<run_name>/gate.log
GATE_TESTS = "tests"       # upstream unit tests, next to benchmark/
GATE_PASS = "pass"
GATE_FAIL = "fail"
GATE_TIMEOUT = "timeout"
GATE_NO_TEST = "no_test"   # the repo has no unit test for this op: unverified, not a failure
GATE_ERROR = "error"       # pytest could not run at all (usage/internal error)
GATE_SKIP = (GATE_FAIL, GATE_TIMEOUT, GATE_ERROR)
PYTEST_NO_TESTS_RC = 5
# pytest's summary counts: "5 passed, 2 skipped in 3.2s"
_COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)")


def _test_functions(path: Path) -> List[str]:
    """The pytest nodes the module certainly defines, in file order.

    `accuracy_hints.test_nodes` keeps the class path (`TestA::test_x`), skips functions
    nested in a function, and leaves out the nodes pytest may not have (a `def` under an
    `if`/`try`, a method of a class whose base cannot be resolved): a node that is not
    there makes pytest exit 4, which the gate reads as an error, not as no_test."""
    return accuracy_hints.test_nodes(path)


def op_spec(spec: str) -> Tuple[str, str, str]:
    """`[repo:]op[@op_file]` -> (repo, op, op_file), read as --ops/--ops-file read it.
    op_file is empty when the spec leaves it out, which is how an --accuracy-map
    entry covers an op whatever benchmark file it sits in."""
    repo, _, rest = spec.rpartition(":")
    repo = repo or "main"
    op, _, op_file = rest.partition("@")
    if repo not in REPOS:
        raise ValueError(f"unknown repo {repo!r} (known: {', '.join(REPOS)})")
    if not op:
        raise ValueError("no op name")
    return repo, op, op_file


def map_target(repo: str, token: str) -> str:
    """One target of an --accuracy-map line: a file under that repo's tests/, with an
    optional ::test_func, relative to the repo root. Canonicalised to `tests/<rel>`
    so the recorded target reads the same for every spelling of the path."""
    rel, _, node = token.partition("::")
    root = REPOS[repo].resolve()
    tests_dir = (root / GATE_TESTS).resolve()
    path = (root / rel).resolve()
    if tests_dir not in path.parents:
        raise ValueError(f"{token!r}: target must be a file under {GATE_TESTS}/ of the "
                         f"{repo} repo")
    if not path.is_file():
        raise ValueError(f"{token!r}: no such file")
    shown = f"{GATE_TESTS}/{path.relative_to(tests_dir).as_posix()}"
    return f"{shown}::{node}" if node else shown


def load_accuracy_map(path: Path) -> Dict[Tuple[str, str, str], str]:
    """`--accuracy-map`: {(repo, op, op_file): pytest target} for the ops whose unit
    test the automatic lookup cannot name.

    One line per op, `[repo:]op[@op_file] target [target...]`, `#` comments. A line
    may name several targets (several test functions of one file) and may leave
    op_file out to cover the op in any benchmark file. Every target is checked here,
    at startup: a mistyped path should stop the run, not resurface as no_test after
    hours of benchmarking."""
    if not path.is_file():
        raise SystemExit(f"{LOG} no accuracy map {path}")
    entries: Dict[Tuple[str, str, str], str] = {}
    errors: List[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        parts = text.split()
        where = f"{path}:{lineno}"
        if len(parts) < 2:
            errors.append(f"{where}: expected '<[repo:]op[@op_file]> <tests/target.py"
                          "[::test_func]>'")
            continue
        try:
            key = op_spec(parts[0])
            targets = [map_target(key[0], token) for token in parts[1:]]
        except ValueError as exc:
            errors.append(f"{where}: {exc}")
            continue
        if key in entries:
            errors.append(f"{where}: {parts[0]} is mapped twice")
            continue
        entries[key] = " ".join(targets)
    if errors:
        for e in errors:
            print(f"{LOG} error: {e}", file=sys.stderr)
        raise SystemExit(f"{LOG} the accuracy map {path} has errors")
    if not entries:
        raise SystemExit(f"{LOG} the accuracy map {path} names no op")
    return entries


def mapped_test_target(case: run_batch.Case,
                       mapping: Optional[Dict[Tuple[str, str, str], str]]) -> Optional[str]:
    """The map's target for the case: the entry with its op_file first, then the one
    without (which covers the op in any benchmark file)."""
    if not mapping:
        return None
    for key in ((case.repo, case.op, case.op_file), (case.repo, case.op, "")):
        if mapping.get(key):
            return mapping[key]
    return None


def auto_test_target(case: run_batch.Case) -> Optional[str]:
    """`tests/test_<op_file>.py::test_<op>` of the case's repo, or None when that repo
    has no unit test for the op.

    The exact name wins; when the file has no such function, every test whose function
    name contains `test_<op>` is taken instead (`test_chunk_kda_matches_torch_reference`
    for `chunk_kda`, `test_abs_` for `abs`). A test defined in a class keeps its class
    path (`TestGridSampleEdgeCases::test_edge`), because that is the node pytest runs;
    the name lookup itself is on the function part. The inventory of benchmarks and the
    tests are not 1:1 (`perf_`-prefixed functions, inplace and out variants share one
    test, some benchmarks have no test at all), so a missing match is a normal outcome
    (no_test, the op still gets an A/B) and not an error."""
    tests_dir = (REPOS[case.repo] / GATE_TESTS).resolve()
    path = (tests_dir / f"test_{case.op_file}.py").resolve()
    if tests_dir not in path.parents or not path.is_file():
        return None
    name = f"test_{case.op}"
    nodes = _test_functions(path)
    leaves = [(node, accuracy_hints.node_leaf(node)) for node in nodes]
    picked = [node for node, leaf in leaves if leaf == name] or \
             [node for node, leaf in leaves if name in leaf]
    if not picked:
        return None
    shown = f"{GATE_TESTS}/test_{case.op_file}.py"
    return " ".join(f"{shown}::{node}" for node in picked)


def test_target(case: run_batch.Case,
                mapping: Optional[Dict[Tuple[str, str, str], str]] = None) -> Optional[str]:
    """The pytest target(s) of the case's unit test, or None when there is none.

    The map comes first -- it names the test of an op whose function carries another
    name -- then the automatic lookup. Several targets are separated by spaces."""
    return mapped_test_target(case, mapping) or auto_test_target(case)


def accuracy_env(args: argparse.Namespace, side: str, gpu: str) -> Dict[str, str]:
    """The side's environment for its unit test: its PYTHONPATH and variables, but
    none of the benchmark harness's (no replay, no shape set, no dtype filter).

    `USE_FLAGTUNE` is not a harness variable: the upstream benchmark reads it to
    pick its tuning space, so the test must see the value the measured runs get
    (run_batch._spawn sets it the same way) — otherwise a gate run and the A/B it
    stands for would test different configurations."""
    env = run_batch.op_env(args.sides[side]["env"])
    for key in ("OP", "OP_FILE", "OUT_DIR", "REPLAY_FROM", "SHAPE_FILE", "DTYPES", "METRICS",
                "PARALLEL_WARMUP_GPUS", "FLAGGEMS_PERF_IR_DUMP_DIR"):
        env.pop(key, None)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    pythonpath = args.sides[side]["pythonpath"]
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [pythonpath, env.get("PYTHONPATH")]))
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "NO_COLOR": "1", "FLAGGEMS_PERF_COLOR": "never",
                "USE_FLAGTUNE": str(getattr(args, "use_flagtune", 0))})
    if run_batch.accelerator(env) == "metax":
        env["MACA_VISIBLE_DEVICES"] = gpu
    return env


def gate_status(rc: int, timed_out: bool, text: str = "") -> str:
    """pytest's exit code -> gate status.

    5 means nothing was collected, and rc 0 with every test skipped means the tests
    need something this box lacks: both verify nothing, so they are no_test rather
    than a pass."""
    if timed_out:
        return GATE_TIMEOUT
    if rc == 0:
        return GATE_PASS if pytest_counts(text).get("passed") else GATE_NO_TEST
    if rc == PYTEST_NO_TESTS_RC:
        return GATE_NO_TEST
    if rc in (2, 3, 4):  # interrupted / internal error / usage error
        return GATE_ERROR
    return GATE_FAIL


def pytest_counts(text: str) -> Dict[str, int]:
    """pytest's summary counts ("5 passed, 2 skipped in 3.2s"), from the last line
    that carries them."""
    for line in reversed(text.splitlines()):
        found = _COUNT_RE.findall(line)
        if found:
            return {word.rstrip("s"): int(n) for n, word in found}
    return {}


def gate_detail(text: str, status: str) -> str:
    """One line saying why a gate run did not pass: pytest's own summary lines."""
    if status == GATE_PASS:
        return ""
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip().startswith(("FAILED ", "ERROR ", "SKIPPED "))]
    if not lines:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:]
    return "; ".join(lines[:3])[:300]


def run_accuracy(case: run_batch.Case, side: str, args: argparse.Namespace,
                 gpu: str) -> Dict[str, str]:
    """One (op, side): its unit test in a fresh process, like one A/B run."""
    mapping = getattr(args, "accuracy_map", None) or {}
    target = test_target(case, mapping)
    if target is None:
        return {"status": GATE_NO_TEST, "target": "", "return_code": "", "elapsed_ms": "0",
                "log": "", "detail": f"no {GATE_TESTS}/test_{case.op_file}.py::test_{case.op}"
                                     ", and no --accuracy-map entry"}
    out_dir = args.out / GATE_DIR / side / case.repo / case.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "gate.log"
    t0 = time.time()
    # -ra: skip and failure summary lines, because a test that skipped half its dtypes
    # (or all of them) still exits 0.
    cmd = ["python", "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", *target.split()]
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, cwd=str(REPOS[case.repo]), env=accuracy_env(args, side, gpu),
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        rc, timed_out = run_batch._wait(proc, args.accuracy_timeout)
    text = log.read_text(errors="replace")
    status = gate_status(rc, timed_out, text)
    return {"status": status, "target": target, "return_code": str(rc),
            "elapsed_ms": str(int((time.time() - t0) * 1000)),
            "log": log.relative_to(args.out).as_posix(), "detail": gate_detail(text, status)}


def load_gate(out: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """{side: {<repo>/<run_name>: row}} recorded in an earlier invocation."""
    path = out / ab_compare.MANIFEST
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
    except ValueError:
        return {}
    sides = (data.get("accuracy") or {}).get("sides") or {}
    return {side: dict(rows) for side, rows in sides.items() if isinstance(rows, dict)}


def save_gate(out: Path, gate: Dict[str, Dict[str, Dict[str, str]]],
              skipped: Dict[str, str], map_file: str = "", map_entries: int = 0) -> None:
    """Merge the gate results into ab.json, next to the manifest fields."""
    path = out / ab_compare.MANIFEST
    data = json.loads(path.read_text()) if path.is_file() else {}
    accuracy = {"tests_dir": GATE_TESTS,
                "skipped": {run: why for run, why in sorted(skipped.items())},
                "sides": {side: gate[side] for side in SIDES if gate.get(side)}}
    if map_file:
        # Where the targets of the ops the automatic lookup cannot name came from.
        accuracy["map"] = {"file": map_file, "entries": map_entries}
    data["accuracy"] = accuracy
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def gate_reusable(case: run_batch.Case, row: Optional[Dict[str, str]],
                  mapping: Optional[Dict[Tuple[str, str, str], str]]) -> bool:
    """Whether a recorded row spares the gate another run: a pass of the same target,
    or a no_test with no target still (a --accuracy-map entry or a widened match that
    now locates a test reruns it, so a better map takes effect on the same --out).

    The target is part of the key, not just the outcome: a pass recorded for another
    test (an --accuracy-map entry that now names a stricter one) says nothing about
    the test this run would execute, so it reruns. A recorded pass that carries no
    target at all (an ab.json from before targets were recorded) reruns once and then
    carries one, so this costs at most one extra gate per op."""
    status = (row or {}).get("status")
    current = test_target(case, mapping)
    if status == GATE_PASS:
        recorded = (row or {}).get("target") or ""
        return bool(recorded) and recorded == current
    return status == GATE_NO_TEST and current is None


def accuracy_gate(cases: List[run_batch.Case], args: argparse.Namespace,
                  gpus: List[str]) -> Tuple[List[run_batch.Case], Dict[str, str]]:
    """Run each side's unit test for every selected op (--accuracy-gate) and drop the
    ops a side fails. Returns the cases to measure and {run: why it was dropped}.

    Recorded results are reused when they still describe the test this run would run
    (a pass of the same target, a no_test with no target), so resuming and
    --add-rounds do not rerun the gate. Ops without a matching test keep their A/B:
    the gate cannot verify them, which ab.json records as no_test rather than as a
    pass."""
    mapping = getattr(args, "accuracy_map", None) or {}
    gate: Dict[str, Dict[str, Dict[str, str]]] = {
        side: dict(load_gate(args.out).get(side) or {}) for side in SIDES}
    work: List[Tuple[run_batch.Case, str]] = []
    reused = 0
    for case in cases:
        run = f"{case.repo}/{case.run_name}"
        for side in SIDES:
            if gate_reusable(case, gate[side].get(run), mapping):
                reused += 1
            else:
                work.append((case, side))
    print(f"{LOG} accuracy gate: {len(cases)} ops x {len(SIDES)} sides, {len(work)} to run"
          + (f", {reused} already recorded" if reused else ""), flush=True)

    if work:
        todo: "queue.Queue" = queue.Queue()
        for it in work:
            todo.put(it)
        done = [0]
        lock = threading.Lock()

        def worker(gpu: str) -> None:
            while True:
                try:
                    case, side = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    row = run_accuracy(case, side, args, gpu)
                except Exception as exc:  # a harness error must not take the worker down
                    row = {"status": GATE_ERROR, "target": "", "return_code": "EXC",
                           "elapsed_ms": "0", "log": "", "detail": f"harness_exc={exc}"}
                with lock:
                    gate[side][f"{case.repo}/{case.run_name}"] = row
                    done[0] += 1
                    print(f"  ({done[0]:>4}/{len(work)}) {side} {case.repo}:{case.run_name}  "
                          f"{row['status']}"
                          + (f"  ({row['detail'][:100]})" if row["detail"] else ""), flush=True)

        threads = [threading.Thread(target=worker, args=(g,)) for g in gpus[:len(work)]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    skipped: Dict[str, str] = {}
    for case in cases:
        run = f"{case.repo}/{case.run_name}"
        bad = [(side, gate[side].get(run) or {}) for side in SIDES]
        bad = [(side, row) for side, row in bad if row.get("status") in GATE_SKIP]
        if bad:
            skipped[run] = "; ".join(f"{side}:{row['status']}"
                                     + (f" ({row['detail']})" if row.get("detail") else "")
                                     for side, row in bad)
    for side in SIDES:
        counts = Counter((gate[side].get(f"{c.repo}/{c.run_name}") or {}).get("status", "?")
                         for c in cases)
        print(f"{LOG} accuracy gate {side.upper()}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    save_gate(args.out, gate, skipped,
              map_file=str(getattr(args, "accuracy_map_file", "") or ""),
              map_entries=len(mapping))
    if skipped:
        print(f"{LOG} {len(skipped)} ops skipped by the accuracy gate (recorded in ab.json)",
              flush=True)
    write_no_test_hints(args.out, cases, gate)
    return [c for c in cases if f"{c.repo}/{c.run_name}" not in skipped], skipped


def write_no_test_hints(out: Path, cases: List[run_batch.Case],
                        gate: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    """Write what the gate could not test, plus where in tests/ it might be.

    A no_test op has no accuracy evidence, and the missing ones are exactly what
    --accuracy-map is for: accuracy_hints.py searches both checkouts and leaves
    <out>/accuracy_no_test.md and <out>/accuracy_map.suggested.txt. The whole gate
    record is used, not just this invocation's cases, so a resume that re-ran one op
    does not drop the others; the files are rewritten even when nothing is left. A
    report that fails must not take the A/B down with it, so this only prints."""
    try:
        md, draft, count = accuracy_hints.write_from_gate(out, cases, gate)
    except Exception as exc:  # a hint file is not worth failing the run over
        print(f"{LOG} accuracy hints failed: {exc}", flush=True)
        return
    if count:
        print(f"{LOG} {count} ops without a test: {md.name} and {draft.name}", flush=True)
    else:
        print(f"{LOG} every op has a test: {md.name} rewritten empty", flush=True)


# ---------------- rounds and cost ----------------

def done_rounds(out: Path) -> Dict[str, Dict[int, float]]:
    """{<repo>/<run_name>: {round: seconds of both sides}} of the finished rounds."""
    per: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for k, side, d in ab_compare.round_dirs(out):
        status = d / "ops_status.csv"
        if not status.is_file():
            continue
        for r in csv.DictReader(status.open(newline="")):
            if r["status"] in ("PASS", "SKIP"):
                per[f"{r['repo']}/{r['run_name']}"][k][side] = int(r["elapsed_ms"] or 0) / 1000
    return {run: {k: s["a"] + s["b"] for k, s in rs.items() if len(s) == 2}
            for run, rs in per.items()}


def record_cost(record: Path) -> Dict[str, float]:
    """Seconds per op of the record run (a proxy until the first round is done)."""
    status = record / "ops_status.csv"
    out = {}
    if status.is_file():
        for r in csv.DictReader(status.open(newline="")):
            out[f"{r['repo']}/{r['run_name']}"] = 2 * int(r["elapsed_ms"] or 0) / 1000
    return out


def plan(cases, args, done) -> List[Tuple[run_batch.Case, int]]:
    """(case, round) items of this invocation, before any --until-decided pass."""
    items = []
    if args.add_rounds:
        start = max((k for k, _, _ in ab_compare.round_dirs(args.out)), default=0) + 1
        for c in cases:
            items += [(c, k) for k in range(start, start + args.add_rounds)]
    else:
        for c in cases:
            have = done.get(f"{c.repo}/{c.run_name}", {})
            items += [(c, k) for k in range(1, args.rounds + 1) if k not in have]
    return items


# ---------------- execution ----------------

class Runner:
    """Runs (case, round) items over the GPUs; keeps the per-round ops_status.csv."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.status: Dict[str, Dict[str, Dict[str, str]]] = {}  # round dir -> {case key: row}
        self.touched = set()
        self.failed = []

    def _status(self, batch: str) -> Dict[str, Dict[str, str]]:
        if batch not in self.status:
            path = self.args.out / batch / "ops_status.csv"
            rows = list(csv.DictReader(path.open(newline=""))) if path.is_file() else []
            self.status[batch] = {f"{r['repo']}:{r['op']}@{r['op_file']}": r for r in rows}
        return self.status[batch]

    def _flush(self, batch: str) -> None:
        path = self.args.out / batch / "ops_status.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=run_batch.STATUS_FIELDS)
            w.writeheader()
            w.writerows(self.status[batch].values())
        os.replace(tmp, path)

    def side_args(self, side: str) -> argparse.Namespace:
        a, cfg = self.args, self.args.sides[side]
        return argparse.Namespace(
            resume=True, dtypes=a.dtypes, metrics=a.metrics, op_timeout=a.op_timeout,
            shape_file=a.shape_file,
            replay_root=a.record, use_flagtune=a.use_flagtune, busy_gpu=a.busy_gpu,
            env=cfg["env"], pythonpath=cfg["pythonpath"])

    def run(self, items: List[Tuple[run_batch.Case, int]], gpus: List[str],
            cost: Dict[str, float]) -> None:
        if not items:
            return
        items = sorted(items, key=lambda it: (-cost.get(f"{it[0].repo}/{it[0].run_name}", 60.0), it[1]))
        work: "queue.Queue" = queue.Queue()
        for it in items:
            work.put(it)
        done = [0]
        t0 = time.time()

        def worker(gpu: str) -> None:
            while True:
                try:
                    case, k = work.get_nowait()
                except queue.Empty:
                    return
                line = []
                for side in (SIDES if k % 2 else SIDES[::-1]):
                    batch = f"r{k}_{side}"
                    try:
                        row = run_batch.run_case(case, self.args.out / batch, gpu,
                                                 self.side_args(side))
                    except Exception as exc:  # a harness error must not take the worker down
                        row = {"repo": case.repo, "op": case.op, "op_file": case.op_file,
                               "run_name": case.run_name, "gpu": gpu, "status": "FAIL",
                               "return_code": "EXC", "elapsed_ms": "0", "rows_total": "0",
                               "rows_success": "0", "notes": f"harness_exc={exc}",
                               "out_dir": str(self.args.out / batch / case.repo / case.run_name)}
                    with self.lock:
                        status = self._status(batch)
                        # A side finished in an earlier invocation keeps its own row (and time).
                        if "resumed" not in row["notes"].split(";") or case.key not in status:
                            status[case.key] = row
                        self._flush(batch)
                        self.touched.add(batch)
                        if row["status"] in ("FAIL", "TIMEOUT"):
                            self.failed.append(f"{case.repo}:{case.run_name} r{k}{side}")
                    line.append(f"{side}:{row['status']}/{int(row['elapsed_ms']) // 1000}s"
                                + (f"({row['notes']})" if row["notes"] else ""))
                with self.lock:
                    done[0] += 1
                    print(f"  ({done[0]:>4}/{len(items)}) r{k:<3} [gpu{gpu}] "
                          f"{case.repo}:{case.run_name}  " + "  ".join(line), flush=True)

        threads = [threading.Thread(target=worker, args=(g,)) for g in gpus[:len(items)]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"{LOG} {len(items)} items in {time.time() - t0:.0f} s", flush=True)

    def summarize(self) -> None:
        for batch in sorted(self.touched):
            batch_summary.summarize(self.args.out / batch)


def auto_gpus(args: argparse.Namespace) -> List[str]:
    """The GPU list both sides run on.

    `--gpus` wins. Otherwise the sides' own environments decide (their
    MACA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES, or the device count their SMI
    tool reports), because the sides are what the ops run under: the parent
    shell's visibility variable must not silently win. Two sides that see
    different sets of cards are an error -- A and B have to measure on the same
    ones; a different order is not, since every op is pinned to a single card,
    so the A side's order is kept."""
    if args.gpus:
        return run_batch.visible_gpus(args.gpus)
    per_side = {side: run_batch.visible_gpus(None, run_batch.op_env(cfg["env"]))
                for side, cfg in args.sides.items()}
    # Compare the sets (and the counts, so a repeated index still counts):
    # `MACA_VISIBLE_DEVICES=2,3` and `3,2` are the same two cards.
    seen = {(frozenset(gpus), len(gpus)) for gpus in per_side.values()}
    if len(seen) > 1:
        shown = ", ".join(f"{side.upper()}={per_side[side]}" for side in SIDES)
        raise SystemExit(f"{LOG} the sides see different GPUs ({shown}); "
                         "pass --gpus to pin the list")
    return list(per_side[SIDES[0]])


def undecided_runs(args: argparse.Namespace) -> set:
    """<repo>/<run_name> of the ops with an undecided row (a changed row when --gate is
    given)."""
    return {r.row.run for r in ab_compare.analyze(args.out, args.gate, **ab_compare.rule_of(args))
            if r.stats.verdict == "undecided" and r.gate != "unchanged"}


def main() -> None:
    args = parse_args()
    cases = run_batch.resolve_selection(args, run_batch.load_inventory())
    if not cases:
        raise SystemExit(f"{LOG} the selection is empty")
    check_manifest(args)
    gpus = auto_gpus(args)
    print(f"{LOG} {len(cases)} ops, {len(gpus)} GPUs, out={args.out}", flush=True)
    for side in SIDES:
        cfg = args.sides[side]
        print(f"{LOG} {side.upper()} = {cfg['name']}: PYTHONPATH={cfg['pythonpath'] or '(inherited)'} "
              + " ".join(f"{k}={v}" for k, v in cfg["env"].items()), flush=True)

    if getattr(args, "accuracy_gate", False):
        cases, skipped = accuracy_gate(cases, args, gpus)
        for run, why in sorted(skipped.items()):
            print(f"{LOG} accuracy gate: no A/B for {run} -- {why}", flush=True)
        if not cases:
            raise SystemExit(f"{LOG} every selected op failed the accuracy gate")
        print(f"{LOG} {len(cases)} ops left to measure", flush=True)

    runner = Runner(args)
    done = done_rounds(args.out)
    cost = record_cost(args.record)
    cost.update({run: statistics.median(rs.values()) for run, rs in done.items() if rs})
    items = plan(cases, args, done)
    runner.run(items, gpus, cost)

    by_run = {f"{c.repo}/{c.run_name}": c for c in cases}
    finished = sum(map(len, done.values()))
    while args.until_decided:
        done = done_rounds(args.out)
        if runner.failed and sum(map(len, done.values())) <= finished:
            print(f"{LOG} the last pass completed no round; stopping", flush=True)
            break
        finished = sum(map(len, done.values()))
        cost.update({run: statistics.median(rs.values()) for run, rs in done.items() if rs})
        need = {run for run in undecided_runs(args)
                if run in by_run and max(done.get(run, {0: 0})) < args.max_rounds}
        if not need:
            break
        extra = []
        for run in sorted(need):
            last = max(done.get(run, {0: 0}))
            extra += [(by_run[run], k) for k in range(last + 1, min(last + args.step, args.max_rounds) + 1)]
        print(f"{LOG} {len(need)} ops with undecided rows: {len(extra)} more items", flush=True)
        runner.run(extra, gpus, cost)

    runner.summarize()
    confirm = ""
    if args.add_rounds:
        first = min(k for _, k in items)
        confirm = f" --confirm {first}-{first + args.add_rounds - 1}"
    print(f"{LOG} done: python ab_compare.py {args.out}"
          + (f" --gate {args.gate}" if args.gate else "") + confirm, flush=True)
    if runner.failed:
        print(f"{LOG} failed runs: {', '.join(runner.failed)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
