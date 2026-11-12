#!/usr/bin/env python3
"""Build the RLC comparison table (xlsx) from an ab_compare.py CSV and a gate dir.

Input is the per-row statistics that ab_compare.py writes with --csv, plus the
ir_gate.py output for the layout-conversion counts:

  python ab_compare.py runs/ab_x --gate runs/gate_x --csv runs/ab_x/rows.csv
  python rlc_report.py --csv runs/ab_x/rows.csv --gate runs/gate_x \\
      --table ops/rlc_table.csv --notes ops/rlc_notes.csv --out rlc_report.xlsx

One row per op; columns grouped by dtype (bf16 / fp16 / float32 / fp8-complex64), each
with the weakest and strongest gain, the lightest and heaviest regression and
how many shapes they cover, plus the ttg.convert_layout count before -> after
from the gate and the verdict spread. The last columns are the largest absolute
saving/cost, a note, and a `*` column counting how often `slower` appears in that
row's other cells -- filter the sheet on "not 0" to keep only the ops some dtype
called slower. A `关键` column right of the op name marks the ops listed in --key-ops
as 1 and every other op as 0.

Definitions (matching the hand-made table):
  gain    a result row whose gap > 0 (gap = median(A)/median(B) - 1, B faster)
  loss    a result row whose gap < 0
  weakest/strongest   min/max gain among gain rows (strongest = max)
  lightest/heaviest   max/min loss among loss rows (heaviest = most negative)
  shape   rows showing a gain (or a loss) / every row of that op and dtype
  saving/cost         max(median(A) - median(B)) / max(median(B) - median(A))
  original            median(A) of the row the cost came from -- what 多出 was measured
                      against; empty when the op has no regression
  cvt     sum of ttg.convert_layout over the op's changed kernels, base -> new
  verdict how many rows ab_compare.py called same/faster/slower/undecided, out of every
          row of that op and dtype (N/M). A row whose gap only carries the sign of its
          noise -- `same` because p >= --clear -- still has a gain/loss figure above,
          so this column is what tells a real shift from a noisy sign: `same 4/5` with
          `退化 最重 -11.8%` is one small-shape row's jitter, not a regression.
  key     1 when the op appears in --key-ops (the ops/key_ops_*.txt lists), else 0
  slower  how many times `slower` appears in the row's other cells. The verdict cells
          are the ones that carry it, so 0 means no dtype of the op was called slower.

Rows kept out of the gain/loss and saving/cost figures — each with its own note giving
the row count, and --include-unconfirmed counting them again:

  timer_mismatch                 a counted round mixed a CUDA graph with plain do_bench
  confirm=timer_mismatch         the recheck's own rounds mixed two clocks
  confirm=insufficient_rounds    the recheck had fewer than three rounds
  confirm=not_reproduced         the recheck did not reproduce the verdict

The rows of an op a side's unit test rejects (the CSV's `accuracy` cell carries a
fail/timeout/error, or `flags` carries `accuracy_failed`) are kept out too, and
--include-accuracy-failed counts them again: ab_run.py --accuracy-gate leaves the
rounds of an earlier invocation on disk, so the row is still shown, but a saving
measured from a build the test rejected is not a saving of the current compiler.
`no_test` and an empty cell are not failures — the gate never ran, or had nothing to
run. Every note's "counted / left out" tail is computed from the same `excluded()` the
summaries use, over that note's own rows: a row carrying both marks (a rejected op and
an unconfirmed recheck) is never reported as counted while the other reason still keeps
it out, and a note whose rows only partly count says "其中 M 行未计入汇总".

A plain undecided row (no confirm) stays in the summaries: nothing was measured
against it.

Rows are ordered key ops first (those listed in --key-ops), then every other op. Inside
each group the op whose 多出 was measured against the largest original latency comes
first; the --table order only breaks ties, so a run without --key-ops still reads in
table order.

The table only orders and names the ops when --table is given; every op found in
the CSV is still reported. Rows whose dtype has no column here (int8, int32,
...) still count towards the op's saving/cost, and the note names them, so a
filtered view is not mistaken for a missing result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

LOG = "[rlc-report]"
DTYPE_GROUPS = (
    ("bf16", ("torch.bfloat16",)),
    ("fp16", ("torch.float16", "torch.half")),
    ("float32", ("torch.float32",)),
    ("fp8 / complex64", ("torch.float8_e4m3fn", "torch.float8_e5m2", "torch.float8_e4m3fnuz",
                         "torch.float8_e5m2fnuz", "torch.complex64", "torch.complex128")),
)
# Groups whose column block starts with a `类型` cell naming the exact dtype (fp8 and
# complex64 share one block, so the cell says which of them the numbers belong to).
TYPED_GROUPS = ("fp8 / complex64",)
COLUMNS = ("#", "算子", "bf16", "fp16", "float32", "fp8 / complex64",
           "最大绝对变化", "备注")
# The `判定` cell lists the verdicts in this order (ab_compare.py's wording), one
# `name N/M` line per verdict that occurs.
VERDICT_COLUMNS = ("same", "faster", "slower", "undecided")
# Cells of a key-op list that are headers, not operator names (a CSV table can be
# passed instead of the plain `ops/key_ops_*.txt` list).
KEY_OPS_IGNORED = {"#", "算子", "op", "name", "序号"}
# The verdict word the `*` column counts: ab_compare.py calls a row `slower` when the
# gap survived both the threshold and the Mann-Whitney test, so 0 means every row of
# the op was either same/undecided or only sign-noise.
SLOWER = "slower"


class Row(NamedTuple):
    label: str
    run: str
    op_name: str
    dtype: str
    row: int
    gap: float
    a_ms: float
    b_ms: float
    gate: str
    flags: str
    verdict: str
    # ab_compare.py --csv: "", confirmed, not_reproduced, or timer_mismatch (the recheck
    # could not confirm because its own rounds mixed two clocks).
    confirm: str = ""
    # ab_compare.py --csv: `a:pass,b:pass`, `a:pass,b:fail`, ... or "" without a gate.
    accuracy: str = ""


class Group(NamedTuple):
    label: str
    dtype: str
    rows: List[Row]


def lift_csv_field_limit() -> None:
    """Parse fields larger than the csv module's 128 KiB default.

    `size_detail` names every input of a row, so an op with many/large tensors writes
    a field far past that default and the parser stops with `field larger than field
    limit`, not with a bad row. The cap is a parser default, not a limit the input
    respects, so raise it to what this platform's C long holds (sys.maxsize overflows
    on 32-bit builds)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


lift_csv_field_limit()


def read_ab_csv(path: Path) -> List[Row]:
    out = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out.append(Row(r["op"], r["run"], r["op_name"], r["dtype"], int(r["row"]),
                               float(r["gap"]), float(r["median_a_ms"]), float(r["median_b_ms"]),
                               r.get("gate", ""), r.get("flags", ""), r.get("verdict", ""),
                               r.get("confirm") or "", r.get("accuracy") or ""))
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"{LOG} bad row in {path}: {r} ({exc})") from exc
    return out


def dtype_group(dtype: str) -> Optional[str]:
    for name, dtypes in DTYPE_GROUPS:
        if dtype in dtypes:
            return name
    return None


def read_table_order(path: Optional[Path], labels: List[str]) -> Dict[str, int]:
    """label -> position, from the comparison table (its 算子 column).

    The table names an op without its op_file; rlc_ops.resolve maps it to the
    same `repo:op@op_file` label ab_compare.py writes. If the table cannot be
    resolved (an op missing from the inventory), fall back to op-name order."""
    if not path:
        return {}
    try:
        from rlc_ops import read_table, resolve
    except ImportError:  # pragma: no cover - rlc_ops.py sits next to this script
        return {}
    entries = read_table(path)
    try:
        specs, _, _ = resolve(entries)
    except SystemExit as exc:
        print(f"{LOG} ordering by op name only: {exc}", file=sys.stderr)
        specs = []
    order = {spec: i for i, spec in enumerate(specs)}
    by_op: Dict[str, int] = {}
    for i, spec in enumerate(specs):
        by_op.setdefault(spec.partition(":")[2].partition("@")[0], i)
    for i, entry in enumerate(entries):
        by_op.setdefault(entry.op, i)
    for label in labels:
        if label not in order:
            op = label.partition(":")[2].partition("@")[0]
            if op in by_op:
                order[label] = by_op[op]
    return order


def load_notes(path: Optional[Path]) -> Dict[str, str]:
    """Optional `op,note` CSV; a missing file means no manual notes."""
    if not path or not path.is_file():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r.get("op") or r.get("label") or "").strip()
            note = (r.get("note") or "").strip()
            if key and note:
                out[key] = note
    return out


def load_key_ops(path: Optional[Path]) -> Set[str]:
    """Operator names listed in `path`, the way ops/key_ops_ppu.txt writes them.

    One name per line (`main:mm@mm`), `#` comments and blank lines skipped; a line
    with commas is split into cells, so a CSV table with a `算子` column works too.
    A missing file (or no --key-ops) means no op is a key op."""
    if not path or not path.is_file():
        return set()
    names = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        for line in f:
            text = "" if line.lstrip().startswith("#") else line.split("#", 1)[0]
            for cell in text.split(","):
                name = cell.strip()
                if name and not name.isdigit() and name not in KEY_OPS_IGNORED:
                    names.add(name)
    return names


def key_op(label: str, key_ops: Set[str]) -> bool:
    """Whether `label` (`repo:op@op_file`) names an op in the key-op list.

    The lists spell the label the way the reports do (`main:mm@mm`, `vllm:foo`), but a
    bare op name is accepted too -- hand-edited lists usually drop the repo."""
    if not key_ops:
        return False
    repo, _, rest = label.partition(":")
    op, _, _ = rest.partition("@")
    return bool({label, op, rest, f"{repo}:{op}"} & key_ops)


_DO_BENCH_RE = re.compile(
    r"BENCHMARK_DIRECT_NO_CUDAGRAPH\b.*?\bphase=(\w+)\s+reason=(\w+)")


def describe_do_bench(phase: str, reason: str) -> str:
    """Reason to append to the do_bench note."""
    if phase == "host_sync":
        return "host 同步"
    if phase == "capability":
        return "平台无 graph API"
    if phase == "sync":
        return "测量前同步失败"
    if phase == "disabled":
        return "capture 已禁用"
    if phase == "warmup" and reason == "OutOfResources":
        return "预热显存不足"
    return f"{phase} 失败: {reason}"


def load_do_bench_reasons(ab_dir: Optional[Path]) -> Dict[str, str]:
    """`<repo>/<run_name>` -> why its tables fell back to do_bench.

    Reads the BENCHMARK_DIRECT_NO_CUDAGRAPH markers in r<k>_<side>/<repo>/<op>/
    run.log of an ab_run.py output. Rows without a marker get no entry."""
    if not ab_dir or not ab_dir.is_dir():
        return {}
    found: Dict[str, set] = defaultdict(set)
    for log in ab_dir.glob("r*_*/*/*/run.log"):
        run = f"{log.parent.parent.name}/{log.parent.name}"
        text = log.read_text(errors="replace")
        for phase, reason in _DO_BENCH_RE.findall(text):
            found[run].add(describe_do_bench(phase, reason))
    return {run: "；".join(sorted(why)) for run, why in found.items()}


def load_gate(gate_dir: Optional[Path]) -> Dict[Tuple[str, str, str, int], List[str]]:
    """(run, op_name, dtype, row) -> cache hashes of its changed kernels."""
    if not gate_dir:
        return {}
    path = gate_dir / "rows.tsv"
    if not path.is_file():
        raise SystemExit(f"{LOG} no {path}; run ir_gate.py with row tracking")
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["run"], r["op_name"], r["dtype"], int(r["row"]))] = \
                [h for h in r.get("changed_kernels", "").split(",") if h]
    return out


def cvt_counts(gate_dir: Optional[Path], hashes: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """ttg.convert_layout before -> after, summed over the row's changed kernels."""
    if not gate_dir or not hashes:
        return None, None
    before = after = 0
    seen_before = seen_after = False
    for h in hashes:
        for side, attr in (("base", "before"), ("new", "after")):
            for digest in (gate_dir / side / h).glob("*.digest.json"):
                ops = json.loads(digest.read_text()).get("ops", {})
                n = int(ops.get("ttg.convert_layout", 0))
                if attr == "before":
                    before += n
                    seen_before = True
                else:
                    after += n
                    seen_after = True
    return (before if seen_before else None), (after if seen_after else None)


def load_run_kernels(ab_dir: Optional[Path]) -> Dict[Tuple[str, str, str, int], List[str]]:
    """(run, op_name, dtype, row) -> cache hashes of the kernels the row ran.

    Read from the A-side rounds' rows.tsv (written at run time, so it covers
    unchanged rows too; the gate's rows.tsv only lists changed kernels)."""
    if not ab_dir or not ab_dir.is_dir():
        return {}
    out = {}
    for path in ab_dir.glob("r*_a/*/*/rows.tsv"):
        run = f"{path.parent.parent.name}/{path.parent.name}"
        with path.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                ks = [k for k in (r.get("kernels") or "").split(",") if k and k != "?"]
                if ks:
                    out.setdefault((run, r["op_name"], r["dtype"], int(r["row"])), ks)
    return out


def fmt_pct(x: float) -> str:
    return f"{x:+.1%}"


def fmt_time(ms: float) -> str:
    if ms >= 1.0:
        return f"{ms:.2f} ms"
    if ms >= 0.01:
        return f"{ms * 1000:.0f} µs"
    if ms >= 0.0001:
        return f"{ms * 1000:.1f} µs"
    return "<0.1 µs"


def fmt_cvt(pair) -> str:
    """`before→after`, or `x (kernel unchanged)` when the gate saw no change."""
    if len(pair) == 2:
        before, after = pair
        unchanged = False
    else:
        before, after, unchanged = pair
    if unchanged:
        x = before if before is not None else after
        return "" if x is None else f"{x} (kernel unchanged)"
    if after is None:
        return ""
    return f"{'?' if before is None else before}→{after}"


def fmt_verdict(counts: Dict[str, int], total: int) -> str:
    """`same 5/5`, or one `name N/M` line per verdict that occurs.

    M is every row of the op and dtype, so the cell says which share of the rows each
    verdict covers. A CSV written before ab_compare.py had a `verdict` column carries
    none, and those rows are reported as 未标注 rather than folded into `same`."""
    if not total:
        return ""
    parts = [f"{name} {counts[name]}/{total}" for name in VERDICT_COLUMNS if counts.get(name)]
    unmarked = total - sum(counts.get(name, 0) for name in VERDICT_COLUMNS)
    if unmarked or not parts:
        parts.append(f"未标注 {unmarked}/{total}")
    return "\n".join(parts)


def fp8_type(dtypes: List[str]) -> str:
    return "complex64" if any("complex" in d for d in dtypes) else "fp8"


def row_entry(table: Optional[Dict[Tuple[str, str, str, int], List[str]]],
              run: str, op_name: str, dtype: str, row: int) -> List[str]:
    """The gate/kernels entry for one result row, with the same "" fallback
    ab_compare.gate_flag() applies: a single-op gate records its run as "" (see
    ir_gate.py's RunDir.run), so a report built against such a gate must fall back
    too — otherwise every row reads as "kernel unchanged"."""
    if not table:
        return []
    return table.get((run, op_name, dtype, row)) or table.get(("", op_name, dtype, row)) or []


# confirm= states that mean the main-round verdict did not come out of the recheck, or
# (not_reproduced) was rechecked and failed to reproduce. Each gets its own note and is
# kept out of the gain/loss and saving/cost summaries unless --include-unconfirmed.
CONFIRM_REASONS = {
    "timer_mismatch": "复测计时口径不一致，主轮次结论未确认",
    "insufficient_rounds": "复测样本不足，主轮次结论未确认",
    "not_reproduced": "复测未重现",
}


def unconfirmed(row: Row) -> bool:
    """A row whose numbers must not count as a measured gain or loss.

    Either the row's own timing is not comparable (a counted round mixed a CUDA graph
    with plain do_bench: the `timer_mismatch` flag), or the recheck settled nothing —
    it mixed two clocks, it had too few rounds, or the verdict did not reproduce
    (`confirm` in CONFIRM_REASONS). A plain undecided row (no `confirm`, e.g. too few
    main rounds to decide anything) is *not* unconfirmed: nothing was measured against
    it. The summaries skip unconfirmed rows by default; --include-unconfirmed puts them
    back.

    ab_compare.py --csv also writes `unconfirmed` into `flags` (and `no` into
    `gap_confirmed`), so a CSV whose `confirm` column was dropped still marks them."""
    flags = row.flags.split(";")
    return (row.confirm in CONFIRM_REASONS or "unconfirmed" in flags
            or "timer_mismatch" in flags)


# The statuses ab_compare.ACCURACY_FAIL rejects, read from the `accuracy` cell of the
# CSV. Kept here (instead of importing ab_compare) so this script still works on a CSV
# alone; test_rlc_report.py checks the two stay in step.
ACCURACY_FAIL = ("fail", "timeout", "error")


def accuracy_failed(row: Row) -> bool:
    """A row whose op a side's unit test now rejects.

    Such an op keeps the rounds an earlier invocation recorded (ab_compare.py shows
    them, and this report keeps its rows too), but its numbers describe the build the
    test rejected: a saving computed from them is not a saving of the current compiler.
    Empty (no --accuracy-gate) and `no_test` are *not* failures — the gate never ran, or
    had nothing to run, and the A/B stands as measured.

    ab_compare.py --csv also writes `accuracy_failed` into `flags`, so a CSV whose
    `accuracy` column was dropped still marks them (as with `unconfirmed`)."""
    if "accuracy_failed" in row.flags.split(";"):
        return True
    return any(part.split(":", 1)[1] in ACCURACY_FAIL
               for part in row.accuracy.split(",") if ":" in part)


def excluded(row: Row, include_unconfirmed: bool = False,
             include_accuracy_failed: bool = False) -> bool:
    """Whether the gain/loss and saving/cost summaries leave this row out."""
    return ((not include_unconfirmed and unconfirmed(row))
            or (not include_accuracy_failed and accuracy_failed(row)))


def counted_tail(rows: List[Row], include_unconfirmed: bool,
                 include_accuracy_failed: bool) -> str:
    """How the summaries treated this kind of row: all of them, none, or — when a row
    carries both marks and only one switch is on — some.

    Read off `excluded()`, the single place that decides it. Asking the two switches
    separately is what let a note say "仍计入汇总" for a row the other reason still
    kept out: with `accuracy=a:pass,b:fail` and `confirm=insufficient_rounds` on the
    same row, `--include-accuracy-failed` alone counts nothing."""
    dropped = sum(1 for r in rows
                  if excluded(r, include_unconfirmed, include_accuracy_failed))
    total = len(rows)
    if not dropped:
        return f"{total} 行，仍计入汇总"
    if dropped == total:
        return f"{total} 行，未计入汇总"
    return f"{total} 行，其中 {dropped} 行未计入汇总"


def row_notes(rows: List[Row], reasons: Optional[Dict[str, str]],
              notes: Dict[str, str], label: str, include_unconfirmed: bool = False,
              include_accuracy_failed: bool = False, tell_accuracy: bool = True) -> List[str]:
    """The warnings a set of result rows carries: do_bench timing with its reason and how
    many of the rows it covers, a timer that differs between the two sides of a round,
    whatever the recheck settled (mixed clocks, too few rounds, or not reproduced — one
    note each), a generic note for an unconfirmed row whose reason this CSV does not
    carry (a dropped `confirm` column), a replay/config mismatch, unchanged kernels
    despite a difference, the manual note.

    They stay apart on purpose: a mismatched main round makes that row's gap
    uncomparable, a refused recheck only means the main-round verdict was never
    confirmed, and not_reproduced is the opposite of both — the verdict was tested on
    fresh rounds and did not hold. `include_unconfirmed` / `include_accuracy_failed` are
    the two switches of the summaries, and every note derives its "counted / left out"
    tail from `excluded()` over its own rows (`counted_tail`): a row can carry both marks,
    so what one switch alone would do is not what happened to that row.
    `tell_accuracy` turns the accuracy note off where it would only repeat what the op's
    note already said (the per-group calls).

    The do_bench note is a table-level warning, not a per-row fallback count.
    ab_compare.nocg_tables() decides per (op_name, dtype) table: one fallback marker
    marks the whole table and every row of it then carries `flags=do_bench`. So
    "do_bench 计时（表级标记覆盖 3/8 行；host 同步）" reads "the tables behind 3 of the
    8 rows here carry a fallback marker" — a single shape falling back already yields
    the table's every row, and a fraction below 1 only appears when the note spans
    several tables (the op's note merges dtypes). It is not a fallback ratio, and the
    markers carry no per-shape identity to compute one from.

    Both the dtype group and the op call this with the rows they own — the op with
    all of them — so a warning that only a row of a column-less dtype (int8, ...)
    carries still reaches the op's note."""
    flags = {f for r in rows for f in r.flags.split(";") if f}
    out = []
    # First: a row whose op a side's unit test rejects has numbers that describe the
    # build the test rejected, so its "gain" may be a numerical regression. Say that
    # before the timing caveats, and say whether it was counted.
    failed = [r for r in rows if accuracy_failed(r)]
    if failed and tell_accuracy:
        cells = "；".join(sorted({r.accuracy for r in failed if r.accuracy}))
        # A CSV trimmed down to `flags` still marks the row (accuracy_failed) but has no
        # cell to quote: say the same thing without the "a:pass,b:fail" prefix.
        out.append(f"精度失败（{cells + '；' if cells else ''}历史性能仅供参考，"
                   f"{counted_tail(failed, include_unconfirmed, include_accuracy_failed)}）")
    if "do_bench" in flags:
        bench = [r for r in rows if "do_bench" in r.flags.split(";")]
        why = sorted({(reasons or {}).get(r.run, "") for r in bench} - {""})
        out.append(f"do_bench 计时（表级标记覆盖 {len(bench)}/{len(rows)} 行"
                   + (f"；{'；'.join(why)}" if why else "") + "）")
    mismatched = [r for r in rows if "timer_mismatch" in r.flags.split(";")]
    if mismatched:
        out.append("A/B 计时口径不一致，性能差异不可直接比较（"
                   f"{counted_tail(mismatched, include_unconfirmed, include_accuracy_failed)}）")
    for reason, title in CONFIRM_REASONS.items():
        hits = [r for r in rows if r.confirm == reason]
        if not hits:
            continue
        out.append(f"{title}（"
                   f"{counted_tail(hits, include_unconfirmed, include_accuracy_failed)}）")
    # Rows marked unconfirmed but carrying no reason this report can name: a --csv whose
    # `confirm` column was dropped (only `flags=unconfirmed` survives), or a state a newer
    # ab_compare.py writes. They are excluded all the same, so say so instead of leaving
    # the note silent about why their numbers are missing.
    generic = [r for r in rows
               if unconfirmed(r) and r.confirm not in CONFIRM_REASONS
               and "timer_mismatch" not in r.flags.split(";")]
    if generic:
        out.append(f"结果未确认（"
                   f"{counted_tail(generic, include_unconfirmed, include_accuracy_failed)}）")
    if "replay_fallback" in flags:
        out.append("replay 回退，config 不一致")
    if rows and all(r.gate == "unchanged" for r in rows) and \
            any(r.verdict in ("faster", "slower") for r in rows):
        out.append("指令未增删")
    manual = notes.get(label) or (notes.get(rows[0].op_name) if rows else "")
    if manual:
        out.append(manual)
    return out


def summarizing(rows: List[Row], gate: Dict[Tuple[str, str, str, int], List[str]],
                gate_dir: Optional[Path], notes: Dict[str, str], label: str,
                group: str, reasons: Optional[Dict[str, str]] = None,
                kernels: Optional[Dict[Tuple[str, str, str, int], List[str]]] = None,
                include_unconfirmed: bool = False,
                include_accuracy_failed: bool = False) -> dict:
    # Rows with a timer mismatch stay in `rows` (the table still counts them and warns
    # about them) but are left out of the gain/loss ranges: their gap compares two clocks,
    # or its verdict was never confirmed by the recheck. Rows whose op a side's unit test
    # rejects are left out for a stronger reason: their numbers belong to the build the
    # test rejected, so a "gain" there is not a gain of the current compiler.
    counted = [r for r in rows
               if not excluded(r, include_unconfirmed, include_accuracy_failed)]
    gains = [r for r in counted if r.gap > 0]
    losses = [r for r in counted if r.gap < 0]
    total = len(rows)
    changed_hashes = sorted({h for r in rows
                             for h in row_entry(gate, r.run, r.op_name, r.dtype, r.row)})
    all_hashes = sorted({h for r in rows for h in (
        row_entry(gate, r.run, r.op_name, r.dtype, r.row) or
        row_entry(kernels, r.run, r.op_name, r.dtype, r.row))})
    hashes = changed_hashes or all_hashes
    unchanged = not changed_hashes and bool(all_hashes)
    cvt_before = cvt_after = None
    if hashes:
        cvt_before, cvt_after = cvt_counts(gate_dir, hashes)
    note = row_notes(rows, reasons, notes, label,
                     include_unconfirmed=include_unconfirmed,
                     include_accuracy_failed=include_accuracy_failed)
    return {
        "rows": total,
        "dtypes": sorted({r.dtype for r in rows}),
        "gain": (min((r.gap for r in gains), default=None),
                 max((r.gap for r in gains), default=None), len(gains)),
        "loss": (max((r.gap for r in losses), default=None),
                 min((r.gap for r in losses), default=None), len(losses)),
        # verdict -> rows, over every row of the group (not just `counted`): the table
        # shows the spread next to gain/loss, so M has to be the same denominator as
        # the shape counts. An empty verdict (a CSV from before the column existed) is
        # left out here and reported as 未标注 by fmt_verdict.
        "verdicts": dict(Counter(r.verdict for r in rows if r.verdict)),
        "cvt": (cvt_before, cvt_after, unchanged),
        "note": "；".join(note),
    }


def build_groups(rows: List[Row]) -> List[Group]:
    by: Dict[str, Dict[str, List[Row]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        group = dtype_group(r.dtype)
        if group:
            by[r.label][group].append(r)
    return [Group(label, group, rs) for label, groups in by.items()
            for group, rs in groups.items()]


def ungrouped_note(rows: List[Row]) -> str:
    """Note for the rows whose dtype the table has no column for (int8, ...).

    They are not shown per dtype, but they are neither dropped nor hidden: the op
    list and its saving/cost include them, and this note says which dtypes carry
    them, so a table without an int8 column does not read as "int8 is fine"."""
    counts = Counter(r.dtype for r in rows if dtype_group(r.dtype) is None)
    if not counts:
        return ""
    shown = ", ".join(f"{dtype.replace('torch.', '')} {n} 行"
                      for dtype, n in sorted(counts.items()))
    return f"{shown} 未按 dtype 分列"


def op_entry(label: str, rows: List[Row], groups: Dict[str, List[Row]],
             gate: Dict[Tuple[str, str, str, int], List[str]],
             gate_dir: Optional[Path], notes: Dict[str, str], reasons: Dict[str, str],
             kernels: Dict[Tuple[str, str, str, int], List[str]],
             include_unconfirmed: bool = False,
             include_accuracy_failed: bool = False) -> dict:
    """One table row: the per-dtype statistics of the op, plus its saving/cost and
    its warnings over every row it has — rows of a dtype without a column included.

    So an op that only ran int8 still appears, its change is not lost, and a
    warning only that dtype carries (do_bench timing, a replay/config mismatch, a
    manual note) still reaches the table: those rows are in no group's note. A
    warning only *one dtype's* rows produce is kept too, prefixed with that dtype
    ("bf16 指令未增删"): the table has one note per op, and "these rows kept their
    kernels" stops being true for the op as soon as another dtype changed.

    The gain/loss and saving/cost figures skip rows with a timer mismatch unless
    include_unconfirmed is set, and rows whose op a side's unit test rejects unless
    include_accuracy_failed is set; the note says how many rows were left out."""
    stats = {group: summarizing(rs, gate, gate_dir, notes, label, group, reasons, kernels,
                                include_unconfirmed=include_unconfirmed,
                                include_accuracy_failed=include_accuracy_failed)
             for group, rs in groups.items()}
    counted = [r for r in rows
               if not excluded(r, include_unconfirmed, include_accuracy_failed)]
    # The largest change each way, plus the original latency behind the cost: "多出
    # 0.02 ms" only reads as a regression next to what it was measured against.
    saving_row = max(counted, key=lambda r: r.a_ms - r.b_ms, default=None)
    cost_row = max(counted, key=lambda r: r.b_ms - r.a_ms, default=None)
    saving = saving_row.a_ms - saving_row.b_ms if saving_row is not None else 0.0
    cost = cost_row.b_ms - cost_row.a_ms if cost_row is not None else 0.0
    merged = row_notes(rows, reasons, notes, label,
                       include_unconfirmed=include_unconfirmed,
                       include_accuracy_failed=include_accuracy_failed)
    for group, group_rows in groups.items():
        # The op's note already says a rejected op's rows are out of the summaries, so
        # the per-group copy would repeat it once per dtype with nothing new to say.
        for text in row_notes(group_rows, reasons, notes, label,
                              include_unconfirmed=include_unconfirmed,
                              include_accuracy_failed=include_accuracy_failed,
                              tell_accuracy=False):
            labelled = f"{group} {text}"
            if text not in merged and labelled not in merged:
                merged.append(labelled)
    note = "；".join(text for text in dict.fromkeys(merged + [ungrouped_note(rows)]) if text)
    return {"label": label, "rows": len(rows), "groups": stats,
            "saving": max(saving, 0.0), "cost": max(cost, 0.0),
            # 原始: the A-side latency of the row the cost came from. An op that only
            # got faster has no cost row, so the cell stays empty -- 0.0 would be a
            # value, and it would sort as the smallest one.
            "cost_base": cost_row.a_ms if cost_row is not None and cost > 0 else None,
            "note": note}


def load_sources(path: Optional[Path]) -> Dict[str, dict]:
    """`repo:op@op_file` -> the rlc_ops.py provenance row (display/repo/status)."""
    if not path or not path.is_file():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            repo, op = (r.get("repo") or "").strip(), (r.get("op") or "").strip()
            if not repo or not op:
                continue
            op_file = (r.get("op_file") or "").strip()
            label = f"{repo}:{op}@{op_file}" if op_file else f"{repo}:{op}"
            out[label] = r
    return out


def display_name(label: str, sources: Optional[Dict[str, dict]] = None) -> str:
    """`repo:op@op_file` -> the full label, spelled the way the CSV and the gate dirs
    spell it.

    The column used to carry the bare op name (`coalesce`) and only spelled the file
    out when two rows would have read the same, which left a reader reconstructing the
    benchmark file from a short name. The label itself is the name now; a multi/both
    row keeps a second line naming the table row it was expanded from, as provenance
    rather than as part of the name."""
    src = (sources or {}).get(label)
    if src and src.get("status") in ("multi", "both"):
        repo, _, rest = label.partition(":")
        shown = (src.get("display") or rest.partition("@")[0]).strip()
        return f"{label}\n{shown} · {repo} · {src['status']}"
    return label


def write_xlsx(path: Path, rows: List[dict], title: str, env: str) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise SystemExit(
            f"{LOG} openpyxl is needed for xlsx output: pip install openpyxl") from None

    wb = Workbook()
    ws = wb.active
    ws.title = "RLC"
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    head_fill = PatternFill("solid", fgColor="D9E2F3")
    group_fill = PatternFill("solid", fgColor="EDEDED")

    # The layout follows DTYPE_GROUPS: adding a dtype group widens the sheet without
    # touching the column positions below.
    def typed(group: str) -> bool:
        return group in TYPED_GROUPS

    def width(group: str) -> int:
        return 9 if typed(group) else 8

    heads = [(1, 1, "#"), (2, 2, "算子"), (3, 3, "关键")]
    subs: List[Tuple[int, int, str]] = []
    leaves = ["#", "算子", "关键"]
    widths = [5, 38, 6]
    col = 4
    for group, _ in DTYPE_GROUPS:
        heads.append((col, col + width(group) - 1, group))
        if typed(group):
            subs.append((col, col, "类型"))
            leaves.append("类型")
            widths.append(16)
            col += 1
        subs += [(col, col + 2, "收益"), (col + 3, col + 5, "退化"),
                 (col + 6, col + 6, "cvt"), (col + 7, col + 7, "判定")]
        leaves += ["最弱", "最强", "shape", "最轻", "最重", "shape", "cvt", "判定"]
        widths += [9, 9, 7, 9, 9, 7, 7, 13]
        col += 8
    heads += [(col, col + 2, "最大绝对变化"), (col + 3, col + 3, "备注"),
              (col + 4, col + 4, "*")]
    subs += [(col, col, "节省"), (col + 1, col + 1, "原始"), (col + 2, col + 2, "多出"),
             (col + 4, col + 4, "*")]
    leaves += ["节省", "原始", "多出", "备注", "*"]
    widths += [10, 10, 10, 34, 8]
    ncols = len(widths)
    # 备注 is no longer the last column: the `*` slower count sits right of it.
    note_col = ncols - 1

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.cell(1, 1, title).font = Font(bold=True, size=14)
    ws.cell(1, 1).alignment = center
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    ws.cell(2, 1, env).alignment = center

    # header: row 4 groups, row 5 sub-groups, row 6 leaf columns
    for c1, c2, text in heads:
        ws.merge_cells(start_row=4, start_column=c1, end_row=4, end_column=c2)
        cell = ws.cell(4, c1, text)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.fill = group_fill
    for c1, c2, text in subs:
        ws.merge_cells(start_row=5, start_column=c1, end_row=5, end_column=c2)
        cell = ws.cell(5, c1, text)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.fill = head_fill
    for i, text in enumerate(leaves, 1):
        cell = ws.cell(6, i, text)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.fill = head_fill
    for row in range(4, 7):
        for col in range(1, ncols + 1):
            ws.cell(row, col).border = border

    for i, r in enumerate(rows, 1):
        out = [i, r["name"], 1 if r["key"] else 0]
        for group, _ in DTYPE_GROUPS:
            g = r["groups"].get(group)
            if g is None:
                out += [""] * width(group)
                continue
            if typed(group):
                out.append(fp8_type(g["dtypes"]))
            out += [fmt_pct(g["gain"][0]) if g["gain"][0] is not None else "",
                    fmt_pct(g["gain"][1]) if g["gain"][1] is not None else "",
                    f"{g['gain'][2]}/{g['rows']}",
                    fmt_pct(g["loss"][0]) if g["loss"][0] is not None else "",
                    fmt_pct(g["loss"][1]) if g["loss"][1] is not None else "",
                    f"{g['loss'][2]}/{g['rows']}",
                    fmt_cvt(g["cvt"]),
                    fmt_verdict(g["verdicts"], g["rows"])]
        out += [fmt_time(r["saving"]) if r["saving"] else "—",
                fmt_time(r["cost_base"]) if r.get("cost_base") else "—",
                fmt_time(r["cost"]) if r["cost"] else "—", r["note"]]
        # The `*` column: how often the cells to its left call a row slower. The verdict
        # cells are the ones that carry the word; a manual note may mention it too, and
        # counting those keeps the filter honest instead of silently dropping the cell.
        out.append(sum(str(value).count(SLOWER) for value in out))
        for col, value in enumerate(out, 1):
            cell = ws.cell(6 + i, col, value)
            cell.border = border
            cell.alignment = left if col in (2, note_col) else center

    for i, width_ in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width_
    ws.freeze_panes = "A7"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True,
                   help="per-row statistics from ab_compare.py --csv")
    p.add_argument("--gate", type=Path, default=None, help="ir_gate.py output dir (cvt counts)")
    p.add_argument("--table", type=Path, default=None,
                   help="comparison table CSV, for the op order and display names")
    p.add_argument("--notes", type=Path, default=None,
                   help="CSV with columns op,note (manual notes appended to the auto ones)")
    p.add_argument("--ab-dir", type=Path, default=None,
                   help="ab_run.py output dir, read for the do_bench fallback reasons "
                        "(default: the directory of --csv)")
    p.add_argument("--sources", type=Path, default=Path("ops/rlc_ops_sources.csv"),
                   help="rlc_ops.py provenance CSV, used to mark multi/both ops "
                        "(default %(default)s)")
    p.add_argument("--key-ops", type=Path, default=None,
                   help="operator list whose entries are key ops (ops/key_ops_*.txt: one "
                        "name per line, # comments allowed); the table writes 1 in the "
                        "关键 column for them and 0 for every other op")
    p.add_argument("--out", type=Path, default=Path("rlc_report.xlsx"))
    p.add_argument("--title", default="RLC Enhance 开启前后算子性能对比")
    p.add_argument("--env", default="", help="environment line under the title")
    p.add_argument("--include-unconfirmed", action="store_true",
                   help="count rows with a timer mismatch (a round that mixed a CUDA graph "
                        "with plain do_bench, or a recheck that could not confirm) in the "
                        "gain/loss and saving/cost summaries; by default they are excluded "
                        "and the note says how many")
    p.add_argument("--include-accuracy-failed", action="store_true",
                   help="count the rows of an op a side's unit test rejects (--accuracy-gate "
                        "recorded a fail/timeout/error) in the gain/loss and saving/cost "
                        "summaries; by default they are excluded -- those rounds describe the "
                        "build the test rejected -- and the note says so")
    args = p.parse_args()

    rows = read_ab_csv(args.csv)
    if not rows:
        raise SystemExit(f"{LOG} no rows in {args.csv}")
    gate = load_gate(args.gate)
    notes = load_notes(args.notes)
    sources = load_sources(args.sources)
    key_ops = load_key_ops(args.key_ops)
    reasons = load_do_bench_reasons(args.ab_dir or args.csv.parent)
    kernels = load_run_kernels(args.ab_dir or args.csv.parent)
    groups = build_groups(rows)
    labels = sorted({r.label for r in rows})  # every op in the CSV, grouped or not
    order = read_table_order(args.table, labels)
    labels.sort(key=lambda label: (order.get(label, len(order)), label))
    by_label: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_label[r.label].append(r)
    grouped: Dict[str, Dict[str, List[Row]]] = defaultdict(dict)
    for g in groups:
        grouped[g.label][g.dtype] = g.rows

    out = []
    for label in labels:
        out.append({"name": display_name(label, sources),
                    "label": label,
                    "key": key_op(label, key_ops),
                    **op_entry(label, by_label[label], grouped[label], gate, args.gate,
                               notes, reasons, kernels,
                               include_unconfirmed=args.include_unconfirmed,
                               include_accuracy_failed=args.include_accuracy_failed)})
    # Key ops first, each group by the original latency behind its 多出 (largest first);
    # the table order breaks ties, so a run without --key-ops still reads in table order.
    out.sort(key=lambda r: (0 if r["key"] else 1, -(r["cost_base"] or 0.0),
                            order.get(r["label"], len(order)), r["label"]))

    write_xlsx(args.out, out, args.title, args.env)
    print(f"{LOG} {len(out)} ops, {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
