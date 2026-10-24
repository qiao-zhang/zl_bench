#!/usr/bin/env python3
"""Build the RLC comparison table (xlsx) from an ab_compare.py CSV and a gate dir.

Input is the per-row statistics that ab_compare.py writes with --csv, plus the
ir_gate.py output for the layout-conversion counts:

  python ab_compare.py runs/ab_x --gate runs/gate_x --csv runs/ab_x/rows.csv
  python rlc_report.py --csv runs/ab_x/rows.csv --gate runs/gate_x \\
      --table ops/rlc_table.csv --notes ops/rlc_notes.csv --out rlc_report.xlsx

One row per op; columns grouped by dtype (bf16 / fp16 / fp8-complex64), each
with the weakest and strongest gain, the lightest and heaviest regression and
how many shapes they cover, plus the ttg.convert_layout count before -> after
from the gate. The last columns are the largest absolute saving/cost and a note.

Definitions (matching the hand-made table):
  gain    a result row whose gap > 0 (gap = median(A)/median(B) - 1, B faster)
  loss    a result row whose gap < 0
  weakest/strongest   min/max gain among gain rows (strongest = max)
  lightest/heaviest   max/min loss among loss rows (heaviest = most negative)
  shape   rows showing a gain (or a loss) / every row of that op and dtype
  saving/cost         max(median(A) - median(B)) / max(median(B) - median(A))
  cvt     sum of ttg.convert_layout over the op's changed kernels, base -> new

The table only orders and names the ops when --table is given; every op found in
the CSV is still reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

LOG = "[rlc-report]"
DTYPE_GROUPS = (
    ("bf16", ("torch.bfloat16",)),
    ("fp16", ("torch.float16", "torch.half")),
    ("fp8 / complex64", ("torch.float8_e4m3fn", "torch.float8_e5m2", "torch.float8_e4m3fnuz",
                         "torch.float8_e5m2fnuz", "torch.complex64", "torch.complex128")),
)
COLUMNS = ("#", "算子", "bf16", "fp16", "fp8 / complex64", "最大绝对变化", "备注")


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


class Group(NamedTuple):
    label: str
    dtype: str
    rows: List[Row]


def read_ab_csv(path: Path) -> List[Row]:
    out = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out.append(Row(r["op"], r["run"], r["op_name"], r["dtype"], int(r["row"]),
                               float(r["gap"]), float(r["median_a_ms"]), float(r["median_b_ms"]),
                               r.get("gate", ""), r.get("flags", ""), r.get("verdict", "")))
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"{LOG} bad row in {path}: {r} ({exc})")
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


def fp8_type(dtypes: List[str]) -> str:
    return "complex64" if any("complex" in d for d in dtypes) else "fp8"


def summarizing(rows: List[Row], gate: Dict[Tuple[str, str, str, int], List[str]],
                gate_dir: Optional[Path], notes: Dict[str, str], label: str,
                group: str, reasons: Optional[Dict[str, str]] = None,
                kernels: Optional[Dict[Tuple[str, str, str, int], List[str]]] = None) -> dict:
    gains = [r for r in rows if r.gap > 0]
    losses = [r for r in rows if r.gap < 0]
    total = len(rows)
    changed_hashes = sorted({h for r in rows for h in gate.get(
        (r.run, r.op_name, r.dtype, r.row), [])})
    all_hashes = sorted({h for r in rows for h in (
        gate.get((r.run, r.op_name, r.dtype, r.row), []) or
        (kernels or {}).get((r.run, r.op_name, r.dtype, r.row), []))})
    hashes = changed_hashes or all_hashes
    unchanged = not changed_hashes and bool(all_hashes)
    cvt_before = cvt_after = None
    if hashes:
        cvt_before, cvt_after = cvt_counts(gate_dir, hashes)
    flags = {f for r in rows for f in r.flags.split(";") if f}
    note = []
    if "do_bench" in flags:
        why = sorted({(reasons or {}).get(r.run, "") for r in rows
                      if "do_bench" in r.flags.split(";")} - {""})
        note.append("do_bench 计时" + (f"（{'；'.join(why)}）" if why else ""))
    if "replay_fallback" in flags:
        note.append("replay 回退，config 不一致")
    if rows and all(r.gate == "unchanged" for r in rows) and \
            any(r.verdict in ("faster", "slower") for r in rows):
        note.append("指令未增删")
    manual = notes.get(label) or notes.get(rows[0].op_name)
    if manual:
        note.append(manual)
    return {
        "rows": total,
        "dtypes": sorted({r.dtype for r in rows}),
        "gain": (min((r.gap for r in gains), default=None),
                 max((r.gap for r in gains), default=None), len(gains)),
        "loss": (max((r.gap for r in losses), default=None),
                 min((r.gap for r in losses), default=None), len(losses)),
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


def display_name(label: str, op_names: Dict[str, int],
                 sources: Optional[Dict[str, dict]] = None) -> str:
    """`repo:op@op_file` -> the table's row label, plus a second line when the
    op needs distinguishing: `display · repo · multi/both` for a name that
    expanded to several tests, or `repo:op@op_file` for same-named tests."""
    repo, _, rest = label.partition(":")
    op, _, op_file = rest.partition("@")
    src = (sources or {}).get(label)
    if src and src.get("status") in ("multi", "both"):
        shown = (src.get("display") or op).strip()
        return f"{op}\n{shown} · {repo} · {src['status']}"
    return f"{op}\n{label}" if op_names.get(op, 0) > 1 else op


def write_xlsx(path: Path, rows: List[dict], title: str, env: str) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise SystemExit(f"{LOG} openpyxl is needed for xlsx output: pip install openpyxl")

    wb = Workbook()
    ws = wb.active
    ws.title = "RLC"
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    head_fill = PatternFill("solid", fgColor="D9E2F3")
    group_fill = PatternFill("solid", fgColor="EDEDED")

    ncols = 27
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.cell(1, 1, title).font = Font(bold=True, size=14)
    ws.cell(1, 1).alignment = center
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    ws.cell(2, 1, env).alignment = center

    # header: row 4 groups, row 5 sub-groups, row 6 leaf columns
    heads = [
        (1, 1, "#"), (2, 2, "算子"),
        (3, 9, "bf16"), (10, 16, "fp16"), (17, 24, "fp8 / complex64"),
        (25, 26, "最大绝对变化"), (27, 27, "备注"),
    ]
    for c1, c2, text in heads:
        ws.merge_cells(start_row=4, start_column=c1, end_row=4, end_column=c2)
        cell = ws.cell(4, c1, text)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.fill = group_fill
    subs = [(3, 5, "收益"), (6, 8, "退化"), (9, 9, "cvt"),
            (10, 12, "收益"), (13, 15, "退化"), (16, 16, "cvt"),
            (17, 17, "类型"), (18, 20, "收益"), (21, 23, "退化"), (24, 24, "cvt"),
            (25, 25, "节省"), (26, 26, "多出")]
    for c1, c2, text in subs:
        ws.merge_cells(start_row=5, start_column=c1, end_row=5, end_column=c2)
        cell = ws.cell(5, c1, text)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.fill = head_fill
    leaves = ["#", "算子",
              "最弱", "最强", "shape", "最轻", "最重", "shape", "cvt",
              "最弱", "最强", "shape", "最轻", "最重", "shape", "cvt",
              "类型", "最弱", "最强", "shape", "最轻", "最重", "shape", "cvt",
              "节省", "多出", "备注"]
    for i, text in enumerate(leaves, 1):
        cell = ws.cell(6, i, text)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.fill = head_fill
    for row in range(4, 7):
        for col in range(1, ncols + 1):
            ws.cell(row, col).border = border

    for i, r in enumerate(rows, 1):
        out = [i, r["name"]]
        for group in ("bf16", "fp16", "fp8 / complex64"):
            g = r["groups"].get(group)
            if g is None:
                out += [""] * (8 if group == "fp8 / complex64" else 7)
                continue
            if group == "fp8 / complex64":
                out.append(fp8_type(g["dtypes"]))
            out += [fmt_pct(g["gain"][0]) if g["gain"][0] is not None else "",
                    fmt_pct(g["gain"][1]) if g["gain"][1] is not None else "",
                    f"{g['gain'][2]}/{g['rows']}",
                    fmt_pct(g["loss"][0]) if g["loss"][0] is not None else "",
                    fmt_pct(g["loss"][1]) if g["loss"][1] is not None else "",
                    f"{g['loss'][2]}/{g['rows']}",
                    fmt_cvt(g["cvt"])]
        out += [fmt_time(r["saving"]) if r["saving"] else "—",
                fmt_time(r["cost"]) if r["cost"] else "—", r["note"]]
        for col, value in enumerate(out, 1):
            cell = ws.cell(6 + i, col, value)
            cell.border = border
            cell.alignment = left if col in (2, 27) else center

    widths = [5, 26, 9, 9, 7, 9, 9, 7, 7, 9, 9, 7, 9, 9, 7, 7,
              16, 9, 9, 7, 9, 9, 7, 7, 10, 10, 34]
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
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
    p.add_argument("--out", type=Path, default=Path("rlc_report.xlsx"))
    p.add_argument("--title", default="RLC Enhance 开启前后算子性能对比")
    p.add_argument("--env", default="", help="environment line under the title")
    args = p.parse_args()

    rows = read_ab_csv(args.csv)
    if not rows:
        raise SystemExit(f"{LOG} no rows in {args.csv}")
    gate = load_gate(args.gate)
    notes = load_notes(args.notes)
    sources = load_sources(args.sources)
    reasons = load_do_bench_reasons(args.ab_dir or args.csv.parent)
    kernels = load_run_kernels(args.ab_dir or args.csv.parent)
    groups = build_groups(rows)
    labels = sorted({g.label for g in groups})
    order = read_table_order(args.table, labels)
    labels.sort(key=lambda label: (order.get(label, len(order)), label))
    op_names = defaultdict(int)
    for label in labels:
        op_names[label.partition(":")[2].partition("@")[0]] += 1

    by_label: Dict[str, Dict[str, List[Row]]] = defaultdict(dict)
    for g in groups:
        by_label[g.label][g.dtype] = g.rows

    out = []
    for label in labels:
        groups_of = by_label[label]
        all_rows = [r for rs in groups_of.values() for r in rs]
        stats = {group: summarizing(rs, gate, args.gate, notes, label, group,
                                    reasons, kernels)
                 for group, rs in groups_of.items()}
        saving = max((r.a_ms - r.b_ms for r in all_rows), default=0.0)
        cost = max((r.b_ms - r.a_ms for r in all_rows), default=0.0)
        note = "；".join(dict.fromkeys(s["note"] for s in stats.values() if s["note"]))
        out.append({"name": display_name(label, op_names, sources), "label": label,
                    "rows": len(all_rows), "groups": stats,
                    "saving": max(saving, 0.0), "cost": max(cost, 0.0), "note": note})

    write_xlsx(args.out, out, args.title, args.env)
    print(f"{LOG} {len(out)} ops, {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
