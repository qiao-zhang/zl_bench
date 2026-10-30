#!/usr/bin/env python3
"""A/B comparison of Gems latency, per result row.

Repeated-measures A/B (the output dir of ab_run.py): every row has one sample per
round and side, and gets a verdict from both samples (ab_stats.py):

  python ab_compare.py runs/ab_x [--gate runs/gate_x] [--rounds 7-12] [--csv rows.csv] [-v]

  verdict  faster / slower: |median gap| > --threshold and Mann-Whitney p < --alpha
           same: within the threshold, or no detectable shift (p >= --clear)
           undecided: fewer than --min-rounds samples, or in between (add rounds)
  gap      median(A) / median(B) - 1, > 0 means B is faster

With --gate (an ir_gate.py output dir), rows whose kernels all compiled to the same
binary on both sides are controls: they cannot differ, so any verdict other than
same/undecided among them is a false positive, and their rate is the measured
false-positive rate of the rule. Only rows with a changed (or unknown) kernel count
as results. --rounds restricts the analysis to a range of rounds.

With ab_run.py --accuracy-gate, ab.json also records each side's unit-test outcome per
op (tests/test_<op_file>.py::test_<op>, or no_test when the repo has no such test). The
report ends with that section and --csv carries an `accuracy` column; ops a side did
not pass have no A/B rounds, so they appear only there.

Independent recheck of weak results, on fresh rounds:

  python ab_compare.py runs/ab_x --gate G --recheck-list recheck.txt   # ops with weak rows
  python ab_run.py ... --out runs/ab_x --ops-file recheck.txt --add-rounds 6   # say rounds 15-20
  python ab_compare.py runs/ab_x --gate G --confirm 15-20

A faster/slower row (not a control) is weak if |gap| < --weak-gap (2%), |gap| < 2x
the larger coefficient of variation of its two samples, or its table was timed by
plain do_bench. Timing is read per round: a row whose two sides timed the same round
differently (one captured a CUDA graph, the other fell back to plain do_bench) has no
comparable gap at all -- the report counts those rows, --csv carries `timer`
(graph / do_bench / mixed) plus `timer_rounds` naming the rounds behind a mixed one, and
--drop-timer-mismatch removes them from the analysis. --csv also writes `gap_confirmed`
(no/yes) and puts `unconfirmed` into `flags` for every row whose gap is no evidence of a
kernel difference: its timing mixed two clocks, or the recheck did not settle the verdict
(see UNCONFIRMED_CONFIRMS). A round only one side finished
never decides anything: it does not count for the row and it does not mark it.
With
--confirm, every row is judged on the other rounds first; each faster/slower row with
samples in the confirm rounds is then confirmed only if those rounds alone show a gap
beyond the threshold in the same direction with a one-sided Mann-Whitney
p < --confirm-alpha (0.05). A recheck is judged in this order: first whether its own
rounds were timed consistently (`timer_mismatch`), then whether it has at least three
rounds (`insufficient_rounds`), and only then the verdict — neither of the first two
confirms or merges anything, and --recheck-list keeps `insufficient_rounds` rows so
`ab_run.py --add-rounds` can extend them.
Confirmed rows report the gap over all rounds; rows that are
not reproduced become undecided.

Single comparison of two run dirs or two run_batch.py batch dirs (one sample per
row, no verdict; B usually replayed A's record):

  python ab_compare.py runs/softmax_A runs/softmax_B [--min 0.03] [--quiet]
  python ab_compare.py runs/batch_A runs/batch_B

Ratio = A/B Gems latency (>1 means B is faster). Rows are paired by (table op_name,
dtype, row index) and count only if SUCCESS on both sides. A single comparison is a
screen: run-to-run noise and per-GPU bias are a few percent; use ab_run.py for
conclusions. A row whose table one side timed with a CUDA graph and the other with
plain do_bench is counted on that op's line as `timer_mismatch N rows`, and the
summary repeats the total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import ab_stats
from batch_summary import OPERATOR_HEADER_RE, parse_run_log

NOCG = "BENCHMARK_DIRECT_NO_CUDAGRAPH"
FALLBACK = "AUTOTUNE_REPLAY_FALLBACK"
MANIFEST = "ab.json"
# Fallback phases the plugin reports once per process (one process, one op), so
# they hold for every table of that op: NO_CUDAGRAPH_OPS turns graph timing off
# by name, and a backend without the graph API never attempts a capture. The
# per-measurement phases (host_sync, warmup, capture, sync) mark one table.
_NOCG_PHASE_RE = re.compile(r"\bphase=(\w+)")
_NOCG_PROCESS_PHASES = frozenset({"capability", "disabled"})

Key = Tuple[str, str, int]  # (table op_name, dtype, row index)


# ---------------- run.log ----------------

def load_rows(run_dir: Path) -> Dict[Key, Tuple[float, str]]:
    """{(op_name, dtype, row): (Gems latency, size detail)} of the SUCCESS rows. The
    row index counts every row of the table, failed ones included, so that rows pair
    by position even when one side fails a row."""
    log = run_dir / "run.log"
    if not log.is_file():
        return {}
    out: Dict[Key, Tuple[float, str]] = {}
    seq: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in parse_run_log(log.read_text(errors="replace")):
        key = (r["fg_op_name"], r["dtype"])
        i = seq[key]
        seq[key] += 1
        if r["row_status"] != "SUCCESS":
            continue
        try:
            lat = float(r["gems_lat"])
        except ValueError:
            continue
        if lat > 0:
            out[key + (i,)] = (lat, r["size_detail"])
    return out


def nocg_tables(text: str) -> set:
    """(op_name, dtype) of the tables with a row timed by plain do_bench.

    A per-measurement fallback marker is printed while its table's rows are
    measured, before that table's header, so it marks the header that follows.
    A process-wide one (`phase=capability`, `phase=disabled`) is printed once for
    the whole op -- the plugin reports it on the first measurement and stays
    quiet -- so it marks every table of that run.log, not just the first.

    Only a marker carrying `phase=` counts. The cudagraph plugin's startup banner
    names the marker ("... falls back to do_bench with a BENCHMARK_DIRECT_NO_CUDAGRAPH
    marker ...") without reporting a fallback; reading the name alone marked the
    first table of every run.log as plain do_bench."""
    out, pending, process_wide = set(), 0, False
    for line in text.splitlines():
        s = line.strip()
        if NOCG in s:
            phase = _NOCG_PHASE_RE.search(s)
            if phase is None:
                continue  # the banner, not a fallback report
            if phase.group(1) in _NOCG_PROCESS_PHASES:
                process_wide = True
            else:
                pending += 1
            continue
        h = OPERATOR_HEADER_RE.match(s)
        if h:
            if pending or process_wide:
                out.add((h.group("name"), h.group("dtype").strip()))
            pending = 0
    return out


def _fallback(run_dir: Path) -> bool:
    log = run_dir / "run.log"
    return log.is_file() and FALLBACK in log.read_text(errors="replace")


# ---------------- single comparison ----------------

def _direct_tables(run_dir: Path) -> set:
    """(op_name, dtype) tables this run timed by plain do_bench instead of a CUDA graph."""
    log = run_dir / "run.log"
    if not log.is_file():
        return set()
    return nocg_tables(log.read_text(errors="replace"))


def compare_pair(a_dir: Path, b_dir: Path, label: str, thr: float,
                 verbose: bool) -> Optional[Tuple[float, int, int]]:
    """Screen one op pair. Returns (geomean, rows, timer-mismatch rows).

    A row whose table one side captured in a CUDA graph and the other timed by plain
    do_bench carries the difference between two clocks, not between two kernels, so it
    is counted separately instead of silently entering the geomean."""
    a, b = load_rows(a_dir), load_rows(b_dir)
    keys = [k for k in a if k in b]
    if not keys:
        return None
    direct_a, direct_b = _direct_tables(a_dir), _direct_tables(b_dir)
    mismatched = {k for k in keys if (k[:2] in direct_a) != (k[:2] in direct_b)}
    ratios = [a[k][0] / b[k][0] for k in keys]
    gm = ab_stats.geomean(ratios)
    flags = []
    if mismatched:
        flags.append(f"timer_mismatch {len(mismatched)} rows")
    if _fallback(b_dir):
        flags.append("replay_fallback")
    flag = "  [" + ", ".join(flags) + "]" if flags else ""
    print(f"{label:60s} rows={len(keys):4d} geomean={gm:.3f} "
          f"min={min(ratios):.3f} max={max(ratios):.3f}{flag}")
    if verbose:
        for k, r in zip(keys, ratios, strict=True):
            if abs(r - 1) >= thr:
                mark = "  [timer_mismatch]" if k in mismatched else ""
                print(f"    {r:6.3f}  {a[k][0]:.6f} -> {b[k][0]:.6f}  {k[0]} {k[1]} "
                      f"#{k[2]} {a[k][1][:100]}{mark}")
    return gm, len(keys), len(mismatched)


def single(args: argparse.Namespace) -> None:
    if (args.b / "run.log").is_file():
        if compare_pair(args.a, args.b, args.b.name, args.min, not args.quiet) is None:
            sys.exit("no SUCCESS rows to pair")
        return
    results = []
    for b_dir in sorted(d for d in args.b.glob("*/*") if (d / "run.log").is_file()):
        a_dir = args.a / b_dir.parent.name / b_dir.name
        r = compare_pair(a_dir, b_dir, f"{b_dir.parent.name}:{b_dir.name}", args.min,
                         not args.quiet)
        if r:
            results.append(r)
    if not results:
        sys.exit("no ops to pair")
    gm = ab_stats.geomean([r[0] for r in results])
    print(f"== {len(results)} ops, geomean of per-op geomeans {gm:.4f}; "
          f"<0.97: {sum(r[0] < 0.97 for r in results)}, "
          f">1.03: {sum(r[0] > 1.03 for r in results)}")
    mism = sum(r[2] for r in results)
    if mism:
        ops_m = sum(1 for r in results if r[2])
        print(f"== timer mismatch (one side graph, the other plain do_bench): {mism} rows in "
              f"{ops_m} ops -> those ratios compare two clocks, not two kernels; re-record "
              "both sides the same way or use ab_run.py")


# ---------------- repeated-measures A/B ----------------

def parse_range(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """"7-12" / "7-" / "-6" / "3" -> inclusive (lo, hi); None = every round."""
    if not text:
        return None
    lo, sep, hi = text.partition("-")
    if not sep:
        return int(lo), int(lo)
    return int(lo) if lo else 1, int(hi) if hi else 10 ** 9


def round_dirs(ab_dir: Path, rng: Optional[Tuple[int, int]] = None,
               exclude: Optional[Tuple[int, int]] = None) -> List[Tuple[int, str, Path]]:
    """[(round, side, dir)] of r<k>_a / r<k>_b, ordered by round; rng keeps, exclude drops
    an inclusive range of rounds."""
    out = []
    for d in ab_dir.glob("r*_[ab]"):
        head, _, side = d.name.partition("_")
        if d.is_dir() and head[1:].isdigit():
            k = int(head[1:])
            if rng is not None and not rng[0] <= k <= rng[1]:
                continue
            if exclude is not None and exclude[0] <= k <= exclude[1]:
                continue
            out.append((k, side, d))
    return sorted(out)


def op_labels(ab_dir: Path) -> Dict[str, str]:
    """<repo>/<run_name> -> repo:op@op_file, from the rounds' ops_status.csv."""
    out = {}
    for _, _, d in round_dirs(ab_dir):
        status = d / "ops_status.csv"
        if status.is_file():
            for r in csv.DictReader(status.open(newline="")):
                out[f"{r['repo']}/{r['run_name']}"] = f"{r['repo']}:{r['op']}@{r['op_file']}"
    return out


class Row(NamedTuple):
    run: str          # <repo>/<run_name>
    op_name: str
    dtype: str
    row: int
    size: str
    a: List[float]
    b: List[float]
    rounds: List[int]  # rounds with a sample on both sides
    nocg: bool         # a counted round timed the table by plain do_bench (either side)
    fallback: bool     # a side did not replay every recorded config
    nocg_a: bool = False   # A timed it by plain do_bench in at least one counted round
    nocg_b: bool = False   # B timed it by plain do_bench in at least one counted round
    # (round, A plain do_bench, B plain do_bench) for the rounds that count, in order.
    # Kept per round because the fallback belongs to a measurement, not to the op: a
    # round only one side finished must not decide anything (see timer_mismatch).
    direct_rounds: Tuple[Tuple[int, bool, bool], ...] = ()


def _marks(direct_rounds: Tuple[Tuple[int, bool, bool], ...]) -> Tuple[bool, bool, bool]:
    """(nocg, nocg_a, nocg_b) of a row, from the marks of its counted rounds."""
    a = any(da for _, da, _ in direct_rounds)
    b = any(db for _, _, db in direct_rounds)
    return a or b, a, b


def timer_mode(row: Row) -> str:
    """How this row was timed: "graph", "do_bench" or "mixed".

    graph     every counted round captured a CUDA graph on both sides
    do_bench  every counted round timed with plain do_bench on both sides
    mixed     anything else: a round where the sides differ (that gap mixes two clocks),
              or counted rounds that disagree with one another"""
    pairs = {(da, db) for _, da, db in row.direct_rounds}
    if pairs == {(True, True)}:
        return "do_bench"
    if pairs <= {(False, False)}:
        return "graph"
    return "mixed"


def timer_mismatch(row: Row) -> bool:
    """True when one side timed a counted round with a CUDA graph and the other with plain
    do_bench. Judged per round, not on the whole run: a table that fell back on A in one
    round and on B in another is mismatched in both, and a round only one side finished
    says nothing about the rounds that count.

    `report()` counts these rows, `write_csv()` flags them, and
    `analyze(drop_timer_mismatch=True)` removes them."""
    return any(da != db for _, da, db in row.direct_rounds)


# confirm= states in which the row's numbers must not be read as a measured gain/loss.
# Kept in sync with rlc_report.CONFIRM_REASONS, which adds the Chinese note titles.
UNCONFIRMED_CONFIRMS = ("timer_mismatch", "insufficient_rounds", "not_reproduced")


def row_unconfirmed(row: Row, confirm: str = "") -> bool:
    """True when the row's numbers are no evidence of a kernel difference.

    Either its own timing mixed two clocks (`timer_mismatch`), or the recheck settled
    nothing / refuted the verdict (`confirm` in UNCONFIRMED_CONFIRMS, passed from the
    Result since confirm belongs to the comparison, not to the samples). write_csv()
    puts `unconfirmed` into `flags` and `no` into `gap_confirmed` for these rows, so a
    reader that never looks at `confirm` still sees that the gap is not confirmed."""
    return timer_mismatch(row) or confirm in UNCONFIRMED_CONFIRMS


def timer_detail(row: Row) -> str:
    """The counted rounds behind a `mixed` timer: "k:a+b" both sides plain do_bench,
    "k:a" / "k:b" one side only, in round order. Empty unless timer_mode() is mixed;
    counted rounds it does not name captured a graph on both sides."""
    if timer_mode(row) != "mixed":
        return ""
    parts = []
    for k, da, db in row.direct_rounds:
        if da and db:
            parts.append(f"{k}:a+b")
        elif da:
            parts.append(f"{k}:a")
        elif db:
            parts.append(f"{k}:b")
    return ",".join(parts)


def load_samples(ab_dir: Path, rng: Optional[Tuple[int, int]] = None,
                 exclude: Optional[Tuple[int, int]] = None) -> List[Row]:
    """Every row with its per-round samples. A round counts for a row only if both of
    its sides measured the row."""
    per: Dict[Tuple[str, str, str, int], Dict[int, Dict[str, float]]] = defaultdict(dict)
    size: Dict[Tuple[str, str, str, int], str] = {}
    # Per round and side: the (run, op_name, dtype) tables that fell back to plain
    # do_bench. Recorded per round because it belongs to a measurement, not to the op:
    # a round only one side finished is recorded but never consulted.
    nocg: Dict[str, Dict[int, set]] = {"a": {}, "b": {}}
    fallback = set()
    for k, side, d in round_dirs(ab_dir, rng, exclude):
        for run_dir in sorted(p for p in d.glob("*/*") if (p / "run.log").is_file()):
            run = f"{run_dir.parent.name}/{run_dir.name}"
            text = (run_dir / "run.log").read_text(errors="replace")
            nocg[side].setdefault(k, set()).update({(run,) + t for t in nocg_tables(text)})
            if FALLBACK in text:
                fallback.add(run)
            for (op_name, dtype, i), (lat, sz) in load_rows(run_dir).items():
                key = (run, op_name, dtype, i)
                per[key].setdefault(k, {})[side] = lat
                size.setdefault(key, sz)
    rows = []
    for key, by_round in sorted(per.items()):
        rounds = sorted(k for k, s in by_round.items() if len(s) == 2)
        if not rounds:
            continue
        direct = tuple(
            (k, key[:3] in nocg["a"].get(k, ()), key[:3] in nocg["b"].get(k, ()))
            for k in rounds)
        nocg_all, direct_a, direct_b = _marks(direct)
        rows.append(Row(*key, size[key], [by_round[k]["a"] for k in rounds],
                        [by_round[k]["b"] for k in rounds], rounds,
                        nocg_all, key[0] in fallback, direct_a, direct_b, direct))
    return rows


def load_gate(gate_dir: Path) -> Dict[Tuple[str, str, str, int], str]:
    """{(run, op_name, dtype, row): changed 1/0/?} from ir_gate.py's rows.tsv."""
    path = gate_dir / "rows.tsv"
    if not path.is_file():
        raise SystemExit(f"no {path}: run ir_gate.py on a record made with row tracking")
    out = {}
    for r in csv.DictReader(path.open(newline=""), delimiter="\t"):
        out[(r["run"], r["op_name"], r["dtype"], int(r["row"]))] = r["changed"]
    return out


def gate_flag(gate: Dict[Tuple[str, str, str, int], str], row: Row) -> str:
    """changed / unchanged / unknown for a row; a single-op record has run "" in the gate."""
    flag = gate.get((row.run, row.op_name, row.dtype, row.row))
    if flag is None:
        flag = gate.get(("", row.op_name, row.dtype, row.row))
    return {"1": "changed", "0": "unchanged"}.get(flag or "?", "unknown")


class Result(NamedTuple):
    row: Row
    stats: ab_stats.RowStats
    gate: str          # changed / unchanged / unknown, or "" without --gate
    confirm: str = ""  # --confirm: confirmed / not_reproduced / "" (not rechecked)
    new: Optional[ab_stats.RowStats] = None  # statistics of the confirm rounds alone


def analyze(ab_dir: Path, gate_dir: Optional[Path] = None, rng: Optional[Tuple[int, int]] = None,
            exclude: Optional[Tuple[int, int]] = None,
            drop_timer_mismatch: bool = False, **rule) -> List[Result]:
    gate = load_gate(gate_dir) if gate_dir else None
    rows = load_samples(ab_dir, rng, exclude)
    if drop_timer_mismatch:
        rows = [r for r in rows if not timer_mismatch(r)]
    out = []
    for row in rows:
        out.append(Result(row, ab_stats.compare(row.a, row.b, **rule),
                          gate_flag(gate, row) if gate is not None else ""))
    return out


def is_weak(r: Result, weak_gap: float) -> bool:
    """A faster/slower row whose evidence is thin: a small gap, a gap within twice the
    samples' coefficient of variation, timing by plain do_bench, or two sides timed
    by different clocks."""
    if r.stats.verdict not in ("faster", "slower"):
        return False
    cv = max(statistics.pstdev(x) / statistics.fmean(x) for x in (r.row.a, r.row.b))
    return (abs(r.stats.gap) < weak_gap or abs(r.stats.gap) < 2 * cv
            or r.row.nocg or timer_mismatch(r.row))


def recheck_ops(results: List[Result], labels: Dict[str, str],
                weak_gap: float) -> Tuple[List[str], int]:
    """(ops, weak rows) a fresh-round recheck should target.

    Every weak row of a changed (or unknown) kernel, except the ones more rounds cannot
    fix: insufficient_rounds belongs here — more rounds is exactly what fixes it — while
    timer_mismatch and not_reproduced do not improve with more rounds, so both stay out."""
    weak = [r for r in results
            if r.gate != "unchanged" and r.confirm in ("", "insufficient_rounds")
            and is_weak(r, weak_gap)]
    return sorted({labels.get(r.row.run, r.row.run) for r in weak}), len(weak)


def confirm(ab_dir: Path, gate_dir: Optional[Path], rng: Tuple[int, int], confirm_alpha: float,
            drop_timer_mismatch: bool = False, **rule) -> List[Result]:
    """Judge every row on the rounds outside rng, then recheck its faster/slower verdict on
    the rounds inside rng alone (one-sided, in the original direction).

    A recheck whose own rounds were timed inconsistently (one side graph, the other plain
    do_bench) cannot confirm anything: it is reported as `timer_mismatch` and its samples
    are not merged, whatever --drop-timer-mismatch says about the main rounds."""
    main = analyze(ab_dir, gate_dir, exclude=rng,
                   drop_timer_mismatch=drop_timer_mismatch, **rule)
    new = {(r.run, r.op_name, r.dtype, r.row): r for r in load_samples(ab_dir, rng)}
    out = []
    for r in main:
        n = new.get((r.row.run, r.row.op_name, r.row.dtype, r.row.row))
        if r.stats.verdict not in ("faster", "slower") or n is None:
            out.append(r)
            continue
        alternative = "greater" if r.stats.verdict == "faster" else "less"
        p = ab_stats.mann_whitney_u(n.a, n.b, alternative)[1]
        gap = statistics.median(n.a) / statistics.median(n.b) - 1
        stats_new = ab_stats.RowStats(len(n.a), len(n.b), statistics.median(n.a),
                                      statistics.median(n.b), gap, p, "")
        # Timing first: a recheck that mixed two clocks says nothing whatever its round
        # count, and it is the reason no confirmation is possible.
        if timer_mismatch(n):
            out.append(Result(r.row, r.stats, r.gate, "timer_mismatch", stats_new))
            continue
        # Then the round count. "no recheck was arranged" (confirm == "") and "arranged,
        # but too short" have to stay distinguishable: the latter is what --add-rounds
        # fixes, so --recheck-list keeps it.
        if len(n.rounds) < 3:
            out.append(Result(r.row, r.stats, r.gate, "insufficient_rounds", stats_new))
            continue
        ok = p < confirm_alpha and (gap > rule["threshold"] if r.stats.verdict == "faster"
                            else gap < -rule["threshold"])
        if ok:
            direct = r.row.direct_rounds + n.direct_rounds
            nocg_all, direct_a, direct_b = _marks(direct)
            # fallback stays a per-run warning (AUTOTUNE_REPLAY_FALLBACK names no round),
            # so merging keeps it when either set of rounds tripped it: the merged samples
            # contain live-tuned points either way.
            both = r.row._replace(a=r.row.a + n.a, b=r.row.b + n.b,
                                  rounds=r.row.rounds + n.rounds, nocg=nocg_all,
                                  nocg_a=direct_a, nocg_b=direct_b, direct_rounds=direct,
                                  fallback=r.row.fallback or n.fallback)
            all_stats = ab_stats.compare(both.a, both.b, **rule)._replace(verdict=r.stats.verdict)
            out.append(Result(both, all_stats, r.gate, "confirmed", stats_new))
        else:
            out.append(Result(r.row, r.stats._replace(verdict="undecided"), r.gate,
                              "not_reproduced", stats_new))
    return out


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


# ---------------- accuracy gate ----------------

def accuracy_sides(manifest: dict) -> Dict[str, Dict[str, Dict[str, str]]]:
    """{side: {<repo>/<run_name>: row}} that ab_run.py --accuracy-gate recorded."""
    sides = (manifest.get("accuracy") or {}).get("sides") or {}
    return {side: dict(rows) for side, rows in sides.items() if isinstance(rows, dict)}


def accuracy_of(manifest: dict, run: str) -> str:
    """`a:pass,b:pass` for one op, empty when the A/B ran without a gate."""
    sides = accuracy_sides(manifest)
    if not sides:
        return ""
    out = []
    for side in ("a", "b"):
        row = (sides.get(side) or {}).get(run)
        if row is not None:
            out.append(f"{side}:{row.get('status', '?')}")
    return ",".join(out)


def report_accuracy(manifest: dict, labels: Dict[str, str]) -> None:
    """The accuracy gate's outcome: it is what decides whether an op has an A/B at
    all, so a report without it would hide why ops are missing."""
    acc = manifest.get("accuracy") or {}
    sides = accuracy_sides(manifest)
    skipped = acc.get("skipped") or {}
    if not sides and not skipped:
        return
    runs = sorted({run for rows in sides.values() for run in rows})
    tests_dir = acc.get("tests_dir", "tests")
    print(f"\n== accuracy gate ({tests_dir}/ unit test per side): {len(runs)} ops checked")
    for side in ("a", "b"):
        rows = sides.get(side) or {}
        if not rows:
            continue
        counts = Counter((row or {}).get("status", "?") for row in rows.values())
        print(f"   {side.upper()}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    unverified = sum(1 for rows in sides.values() for row in rows.values()
                     if (row or {}).get("status") == "no_test")
    if unverified:
        print(f"   no_test: {unverified} op-sides have no unit test upstream, "
              "their A/B result is not backed by an accuracy check")
    if skipped:
        print(f"   skipped A/B (a side did not pass): {len(skipped)}")
        for run, why in sorted(skipped.items())[:20]:
            print(f"     {labels.get(run, run):56s} {str(why)[:150]}")
        if len(skipped) > 20:
            print(f"     ... {len(skipped) - 20} more (ab.json: accuracy.skipped)")


def report(results: List[Result], labels: Dict[str, str], manifest: dict,
           verbose: bool, quiet: bool) -> None:
    sides = manifest.get("sides", {})
    for s in ("a", "b"):
        cfg = sides.get(s, {})
        what = " ".join(([f"PYTHONPATH={cfg['pythonpath']}"] if cfg.get("pythonpath") else [])
                        + [f"{k}={v}" for k, v in (cfg.get("env") or {}).items()])
        print(f"{s.upper()} = {cfg.get('name', s.upper())}: {what or '(caller environment)'}")
    gated = any(r.gate for r in results)
    tested = [r for r in results if not gated or r.gate != "unchanged"]
    controls = [r for r in results if gated and r.gate == "unchanged"]
    if results:
        n = [len(r.row.rounds) for r in results]
        print(f"record {manifest.get('record', '?')}; rounds per row {min(n)}-{max(n)}; "
              "gap = median(A)/median(B) - 1, > 0: B faster")

    by_op: Dict[str, List[Result]] = defaultdict(list)
    for r in tested:
        by_op[r.row.run].append(r)
    if not quiet and by_op:
        print(f"\n{'op':56s} {'rows':>5} {'faster':>7} {'slower':>7} {'same':>6} {'undec':>6}"
              f" {'fastest':>8} {'slowest':>8}")
        for run, rs in sorted(by_op.items()):
            v = defaultdict(int)
            for r in rs:
                v[r.stats.verdict] += 1
            fast = max((r.stats.gap for r in rs if r.stats.verdict == "faster"), default=None)
            slow = min((r.stats.gap for r in rs if r.stats.verdict == "slower"), default=None)
            flags = []
            if any(r.row.nocg for r in rs):
                flags.append("do_bench")
            if any(timer_mismatch(r.row) for r in rs):
                flags.append("timer_mismatch")
            if any(r.row.fallback for r in rs):
                flags.append("replay_fallback")
            fast_s = _pct(fast) if fast is not None else "-"
            slow_s = _pct(slow) if slow is not None else "-"
            print(f"{labels.get(run, run):56s} {len(rs):5d} {v['faster']:7d} {v['slower']:7d}"
                  f" {v['same']:6d} {v['undecided']:6d} {fast_s:>8} {slow_s:>8}"
                  + (f"  [{', '.join(flags)}]" if flags else ""))
            if verbose:
                for r in rs:
                    if r.stats.verdict in ("faster", "slower", "undecided"):
                        print(f"    {r.stats.verdict:9s} {_pct(r.stats.gap):>8} p={r.stats.p:.2g} "
                              f"n={r.stats.n_a} {r.row.op_name} {r.row.dtype} #{r.row.row} "
                              f"{r.row.size[:90]}")
    v = defaultdict(int)
    for r in tested:
        v[r.stats.verdict] += 1
    what = "rows with a changed kernel" if gated else "rows"
    print(f"\n== {what}: {len(tested)}; faster {v['faster']}, slower {v['slower']}, "
          f"same {v['same']}, undecided {v['undecided']}")
    modes = Counter(timer_mode(r.row) for r in tested)
    if modes.get("mixed") or modes.get("do_bench"):
        print(f"   timing: graph {modes.get('graph', 0)}, do_bench {modes.get('do_bench', 0)}, "
              f"mixed {modes.get('mixed', 0)}")
    mismatch = [r for r in tested if timer_mismatch(r.row)]
    if mismatch:
        a_direct = sum(1 for r in mismatch if r.row.nocg_a)
        b_direct = sum(1 for r in mismatch if r.row.nocg_b)
        ops_m = len({r.row.run for r in mismatch})
        print(f"   timer mismatch (one side graph, the other plain do_bench): "
              f"{len(mismatch)} rows in {ops_m} ops "
              f"(A do_bench {a_direct}, B do_bench {b_direct}) -> their gap is not comparable")
        for label, op_name, dtype, detail in sorted({(labels.get(r.row.run, r.row.run),
                                                      r.row.op_name, r.row.dtype,
                                                      timer_detail(r.row))
                                                     for r in mismatch})[:10]:
            print(f"    {label} {op_name} {dtype}  rounds {detail}")
    rechecked = [r for r in tested if r.confirm]
    if rechecked:
        counts = Counter(r.confirm for r in rechecked)
        parts = [f"confirmed {counts.get('confirmed', 0)}"]
        if counts.get("not_reproduced"):
            parts.append(f"not reproduced {counts['not_reproduced']} (now undecided)")
        if counts.get("timer_mismatch"):
            parts.append(f"refused, timer mismatch {counts['timer_mismatch']}")
        if counts.get("insufficient_rounds"):
            parts.append(f"too few rounds {counts['insufficient_rounds']}")
        print(f"   rechecked on fresh rounds: {len(rechecked)}; " + ", ".join(parts))
    if gated:
        unknown = sum(1 for r in tested if r.gate == "unknown")
        if unknown:
            print(f"   ({unknown} of them unattributed: no rows.tsv or a kernel failed to recompile)")
        fp = [r for r in controls if r.stats.verdict in ("faster", "slower")]
        decided = [r for r in controls if r.stats.verdict != "undecided"]
        rate = f" ({len(fp) / len(decided):.2%} of the {len(decided)} decided)" if decided else ""
        print(f"== controls (identical binaries): {len(controls)}; called faster/slower: "
              f"{len(fp)}{rate}, undecided {len(controls) - len(decided)}")
        for r in fp[:10]:
            print(f"    false positive: {labels.get(r.row.run, r.row.run)} {r.row.op_name} "
                  f"{r.row.dtype} #{r.row.row} {_pct(r.stats.gap)} p={r.stats.p:.2g}")
    report_accuracy(manifest, labels)


def write_csv(path: Path, results: List[Result], labels: Dict[str, str],
              manifest: dict) -> None:
    fields = ["op", "run", "op_name", "dtype", "row", "size_detail", "gate", "accuracy",
              "rounds", "median_a_ms", "median_b_ms", "gap", "p", "verdict", "flags",
              "gap_confirmed", "timer", "timer_rounds", "confirm", "confirm_rounds",
              "confirm_gap", "confirm_p"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            flags = ";".join(x for x, on in (("do_bench", r.row.nocg),
                                             ("timer_mismatch", timer_mismatch(r.row)),
                                             ("unconfirmed",
                                              row_unconfirmed(r.row, r.confirm)),
                                             ("replay_fallback", r.row.fallback)) if on)
            w.writerow([labels.get(r.row.run, r.row.run), r.row.run, r.row.op_name, r.row.dtype,
                        r.row.row, r.row.size, r.gate, accuracy_of(manifest, r.row.run),
                        r.stats.n_a, f"{r.stats.median_a:.6g}",
                        f"{r.stats.median_b:.6g}", f"{r.stats.gap:.6f}",
                        "" if math.isnan(r.stats.p) else f"{r.stats.p:.4g}", r.stats.verdict, flags,
                        "no" if row_unconfirmed(r.row, r.confirm) else "yes",
                        timer_mode(r.row), timer_detail(r.row),
                        r.confirm, r.new.n_a if r.new else "",
                        f"{r.new.gap:.6f}" if r.new else "", f"{r.new.p:.4g}" if r.new else ""])


def rule_args(p: argparse.ArgumentParser) -> None:
    """Verdict rule flags, shared with ab_run.py."""
    g = p.add_argument_group("verdict rule (repeated-measures A/B)")
    g.add_argument("--min-rounds", type=int, default=ab_stats.MIN_ROUNDS,
                   help="samples per side before a row gets a verdict (default %(default)s)")
    g.add_argument("--threshold", type=float, default=ab_stats.THRESHOLD,
                   help="smallest |gap| that counts as a difference (default %(default)s)")
    g.add_argument("--alpha", type=float, default=ab_stats.ALPHA,
                   help="Mann-Whitney p below which a gap is real (default %(default)s)")
    g.add_argument("--clear", type=float, default=ab_stats.CLEAR,
                   help="p at or above which a row is called same (default %(default)s)")


def rule_of(args: argparse.Namespace) -> dict:
    return {"min_rounds": args.min_rounds, "threshold": args.threshold, "alpha": args.alpha,
            "clear": args.clear}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", type=Path, help="ab_run.py output dir, or the A side: run dir or batch dir")
    p.add_argument("b", type=Path, nargs="?", help="B side of a single comparison")
    p.add_argument("--gate", type=Path, default=None,
                   help="ir_gate.py output dir: unchanged rows become controls")
    p.add_argument("--rounds", default=None, metavar="LO-HI",
                   help="only these rounds, e.g. 7-12 (default: all)")
    p.add_argument("--csv", type=Path, default=None, help="write every row's statistics here")
    p.add_argument("--recheck-list", type=Path, default=None, metavar="FILE",
                   help="write the ops with a weak faster/slower row (an --ops-file for "
                        "ab_run.py --add-rounds)")
    p.add_argument("--weak-gap", type=float, default=0.02,
                   help="|gap| below which a decided row is weak (default %(default)s)")
    p.add_argument("--confirm", default=None, metavar="LO-HI",
                   help="recheck rounds: judge on the other rounds, confirm on these alone")
    p.add_argument("--confirm-alpha", type=float, default=0.05,
                   help="one-sided p a recheck must reach (default %(default)s)")
    p.add_argument("--drop-timer-mismatch", action="store_true",
                   help="drop rows whose two sides timed the table with different clocks "
                        "(one graph, one plain do_bench): their gap is not comparable")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list each faster / slower / undecided row")
    p.add_argument("--min", type=float, default=0.03,
                   help="single comparison: per-row detail threshold (default %(default)s)")
    p.add_argument("--quiet", action="store_true", help="summary lines only")
    rule_args(p)
    args = p.parse_args()

    if args.b is not None:
        if (args.gate or args.rounds or args.confirm or args.recheck_list or args.csv
                or args.drop_timer_mismatch):
            sys.exit("--gate, --rounds, --confirm, --recheck-list, --csv and "
                     "--drop-timer-mismatch need an ab_run.py dir (rounds on both sides)")
        single(args)
        return
    manifest_path = args.a / MANIFEST
    if not manifest_path.is_file():
        sys.exit(f"{args.a} has no {MANIFEST}: give an ab_run.py output dir, or two dirs to compare")
    if args.confirm:
        if args.rounds:
            sys.exit("--confirm and --rounds do not mix")
        results = confirm(args.a, args.gate, parse_range(args.confirm), args.confirm_alpha,
                          drop_timer_mismatch=args.drop_timer_mismatch, **rule_of(args))
    else:
        results = analyze(args.a, args.gate, parse_range(args.rounds),
                          drop_timer_mismatch=args.drop_timer_mismatch, **rule_of(args))
    if not results:
        sys.exit("no row has a sample on both sides")
    labels = op_labels(args.a)
    manifest = json.loads(manifest_path.read_text())
    report(results, labels, manifest, args.verbose, args.quiet)
    if args.csv:
        write_csv(args.csv, results, labels, manifest)
        print(f"== per-row statistics: {args.csv}")
    if args.recheck_list:
        ops, rows = recheck_ops(results, labels, args.weak_gap)
        args.recheck_list.write_text("".join(f"{op}\n" for op in ops))
        print(f"== {rows} weak rows in {len(ops)} ops: {args.recheck_list}")


if __name__ == "__main__":
    main()
