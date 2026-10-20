#!/usr/bin/env python3
"""Turn the hand-maintained RLC comparison table (CSV) into an ops file.

run_batch.py and ab_run.py take an ops file with one `[repo:]op[@op_file]` per
line. The comparison table's "算子" column holds a display label; when two tests
share that label, a second line carries the inventory's `repo:op`:

    outer
    vllm:outer                 -> vllm:outer@outer
    uniform_
    main:uniform_inplace       -> main:uniform_inplace@uniform

A name without a prefix is looked up in both repos: when only one has it, that
one runs; when both have it, both run (main first) and the sidecar report marks
the pair `both`; when neither has it, the line is written as a `# UNRESOLVED`
comment in the ops file and recorded in the report instead of failing the run.

Usage:
  python rlc_ops.py --table ops/rlc_table.csv --out ops/rlc_ops.txt
  python run_batch.py --ops-file ops/rlc_ops.txt ...
  python ab_run.py --ops-file ops/rlc_ops.txt ...

`--report` (default ops/rlc_ops_sources.csv) records every line as
`main` / `vllm` / `both` / `missing` / `ambiguous`, so the provenance of each op
is auditable. Ambiguous names (several op_files in one repo) remain an error:
they cannot be resolved automatically.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import run_batch

LOG = "[rlc-ops]"
OP_COLUMN = 1  # the table's second column is "算子"
REPORT_FIELDS = ["line", "display", "repo", "op", "op_file", "status"]


class Entry(NamedTuple):
    line: int
    display: str
    repo: str
    op: str
    op_file: Optional[str]
    explicit: bool  # the cell named a repo (or an op_file), so do not search both


def parse_cell(cell: str) -> Optional[Tuple[str, str, str, Optional[str], bool]]:
    """(display, repo, op, op_file, explicit) of one 算子 cell.

    The first line is the table's display name; the optional second line is the
    inventory's `repo:op`. A trailing `@op_file` is accepted for an op that
    appears in several benchmark files. Without a second line (and without a
    `repo:` in the first), the entry is left unqualified and resolve() searches
    both repos."""
    lines = [ln.strip() for ln in cell.splitlines() if ln.strip()]
    if not lines:
        return None
    display = lines[0]
    repo, op, op_file, explicit = "main", display, None, False
    if len(lines) > 1:
        spec = lines[1]
        repo, sep, rest = spec.partition(":")
        if not sep or not repo or not rest:
            raise ValueError(f"expected `repo:op` on the second line, got {spec!r}")
        op, _, op_file = rest.partition("@")
        op_file = op_file or None
        explicit = True
    elif ":" in display:
        repo, _, op = display.partition(":")
        explicit = True
    return display, repo, op, op_file, explicit


def read_table(path: Path) -> List[Entry]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8-sig")))
    # The table starts with a title and an environment line, then a two-row
    # header whose second column is 算子. Data starts after the last header row.
    start = 0
    for i, row in enumerate(rows):
        if len(row) > OP_COLUMN and row[OP_COLUMN].strip() == "算子":
            start = i + 1
            break
    else:
        raise SystemExit(f"{LOG} no header row with 算子 in column {OP_COLUMN + 1}: {path}")
    # Skip the remaining header rows (收益/退化/最弱/...), which have an empty
    # op column, and keep every data row that has one.
    out = []
    for i, row in enumerate(rows[start:], start=start + 1):
        cell = row[OP_COLUMN] if len(row) > OP_COLUMN else ""
        parsed = parse_cell(cell)
        if parsed is None:
            continue
        display, repo, op, op_file, explicit = parsed
        out.append(Entry(i, display, repo, op, op_file, explicit))
    return out


def _inventory() -> Tuple[Dict[Tuple[str, str, str], str], Dict[Tuple[str, str], List[str]]]:
    inv = run_batch.load_inventory()
    index: Dict[Tuple[str, str, str], str] = {}
    by_pair: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for label, rows in inv.items():
        for r in rows:
            if r.get("collected") == "no":
                continue
            index[(label, r["op"], r["op_file"])] = r["op_file"]
            by_pair[(label, r["op"])].append(r["op_file"])
    return index, by_pair


def resolve(entries: List[Entry]) -> Tuple[List[str], List[dict], List[str]]:
    """(specs, report rows, errors). A name missing from both repos is reported,
    not an error; an ambiguous name (several op_files in one repo) still is."""
    index, by_pair = _inventory()
    specs: List[str] = []
    report: List[dict] = []
    errors: List[str] = []

    def add(entry: Entry, repo: str, op_file: str, status: str) -> None:
        specs.append(f"{repo}:{entry.op}@{op_file}")
        report.append({"line": entry.line, "display": entry.display, "repo": repo,
                       "op": entry.op, "op_file": op_file, "status": status})

    for e in entries:
        if e.op_file:
            if (e.repo, e.op, e.op_file) not in index:
                errors.append(f"line {e.line}: no {e.repo}:{e.op}@{e.op_file} in the inventory")
                report.append({"line": e.line, "display": e.display, "repo": e.repo,
                               "op": e.op, "op_file": e.op_file, "status": "ambiguous"})
                continue
            add(e, e.repo, e.op_file, e.repo)
            continue

        if e.explicit:
            candidates = sorted(set(by_pair.get((e.repo, e.op), [])))
            if not candidates:
                errors.append(f"line {e.line}: no {e.repo}:{e.op} in the inventory")
                report.append({"line": e.line, "display": e.display, "repo": e.repo,
                               "op": e.op, "op_file": "", "status": "missing"})
            elif len(candidates) > 1:
                errors.append(f"line {e.line}: {e.repo}:{e.op} is ambiguous, add "
                              f"`{e.repo}:<op_file>` on a second line; candidates: "
                              + ", ".join(candidates))
                report.append({"line": e.line, "display": e.display, "repo": e.repo,
                               "op": e.op, "op_file": "", "status": "ambiguous"})
            else:
                add(e, e.repo, candidates[0], e.repo)
            continue

        # No repo in the table: search both.
        found = {repo: sorted(set(by_pair.get((repo, e.op), [])))
                 for repo in ("main", "vllm")}
        present = {repo: cands for repo, cands in found.items() if cands}
        if not present:
            print(f"{LOG} line {e.line}: {e.display} is in neither main nor vllm; "
                  "recorded as unresolved", file=sys.stderr)
            report.append({"line": e.line, "display": e.display, "repo": "",
                           "op": e.op, "op_file": "", "status": "missing"})
            continue
        ambiguous = {repo: cands for repo, cands in present.items() if len(cands) > 1}
        if ambiguous:
            for repo, cands in ambiguous.items():
                errors.append(f"line {e.line}: {repo}:{e.op} is ambiguous, add "
                              f"`{repo}:<op_file>` on a second line; candidates: "
                              + ", ".join(cands))
            report.append({"line": e.line, "display": e.display, "repo": "",
                           "op": e.op, "op_file": "", "status": "ambiguous"})
            continue
        status = "both" if len(present) == 2 else next(iter(present))
        if status == "both":
            print(f"{LOG} line {e.line}: {e.display} is in both repos; running main and vllm",
                  file=sys.stderr)
        for repo, cands in present.items():
            add(e, repo, cands[0], status)
    return specs, report, errors


def write_report(path: Path, report: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(report)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", type=Path, required=True, help="the comparison table saved as CSV")
    p.add_argument("--out", type=Path, default=Path("ops/rlc_ops.txt"),
                   help="ops file for run_batch.py / ab_run.py (default %(default)s)")
    p.add_argument("--report", type=Path, default=Path("ops/rlc_ops_sources.csv"),
                   help="provenance report (default %(default)s)")
    args = p.parse_args()

    entries = read_table(args.table)
    specs, report, errors = resolve(entries)
    lines = list(specs)
    for r in report:
        if r["status"] == "missing":
            lines.append(f"# UNRESOLVED line {r['line']}: {r['display']} "
                         "(not in main or vllm)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(s + "\n" for s in lines), encoding="utf-8")
    write_report(args.report, report)
    counts = defaultdict(int)
    for r in report:
        counts[r["status"]] += 1
    print(f"{LOG} {len(specs)} ops -> {args.out}; sources -> {args.report} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})", flush=True)
    if errors:
        raise SystemExit(f"{LOG} {len(errors)} unresolved op(s):\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
