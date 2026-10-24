#!/usr/bin/env python3
"""Turn the hand-maintained RLC comparison table (CSV) into an ops file.

run_batch.py and ab_run.py take an ops file with one `[repo:]op[@op_file]` per
line. The comparison table's "算子" column holds a display label; when two tests
share that label, a second line carries the inventory's `repo:op`:

    outer
    vllm:outer                 -> vllm:outer@outer
    uniform_
    main:uniform_inplace       -> main:uniform_inplace@uniform

A name without a prefix is looked up in both repos (test function, then the
benchmark's `op_name`, then the op_file): when only one repo has it, that one
runs; when both have it, both run (main first) and the sidecar report marks the
pair `both`; when neither has it, the line is written as a `# UNRESOLVED`
comment in the ops file and recorded in the report instead of failing the run.
A trailing `@op_file` pins one benchmark file and is honoured on the first line
as well (`main:kron@kron`, `kron@kron`), not only on the second.

Usage:
  python rlc_ops.py --table ops/rlc_table.csv --out ops/rlc_ops.txt
  python run_batch.py --ops-file ops/rlc_ops.txt ...
  python ab_run.py --ops-file ops/rlc_ops.txt ...

`--report` (default ops/rlc_ops_sources.csv) records every line as
`main` / `vllm` / `both` / `multi` / `missing`, so the provenance of each op is
auditable. A name that matches several tests is expanded to every match, marked
`multi`, so the batch runs all of them; add `@op_file` (on either line, with or
without a `repo:` prefix) to run just one.
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
Match = Tuple[str, str]  # (inventory op, op_file)


class Entry(NamedTuple):
    line: int
    display: str
    repo: str  # "" when the cell named none: resolve() searches both repos
    op: str
    op_file: Optional[str]


def parse_cell(cell: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """(display, repo, op, op_file) of one 算子 cell.

    The first line is the table's display name; the optional second line is the
    inventory's `repo:op`. A trailing `@op_file` is accepted on either line, and
    on the first line alone, for an op that appears in several benchmark files:
    it pins that file instead of running every match. Without a repo the entry
    is left unqualified and resolve() searches both repos."""
    lines = [ln.strip() for ln in cell.splitlines() if ln.strip()]
    if not lines:
        return None
    display = lines[0]
    repo, spec = "", lines[1] if len(lines) > 1 else display
    if len(lines) > 1:
        repo, sep, spec = spec.partition(":")
        if not sep or not repo or not spec:
            raise ValueError(f"expected `repo:op` on the second line, got {lines[1]!r}")
    elif ":" in spec:
        repo, _, spec = spec.partition(":")
        if not repo or not spec:
            raise ValueError(f"expected `repo:op`, got {display!r}")
    op, _, op_file = spec.partition("@")
    return display, repo, op, op_file or None


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
        display, repo, op, op_file = parsed
        out.append(Entry(i, display, repo, op, op_file))
    return out


def _inventory() -> Tuple[Dict[Tuple[str, str], List[Match]],
                          Dict[Tuple[str, str], List[Match]],
                          Dict[Tuple[str, str], List[Match]]]:
    """(by_op, by_op_name, by_op_file) -> [(inventory op, op_file), ...]."""
    inv = run_batch.load_inventory()
    by_op: Dict[Tuple[str, str], List[Match]] = defaultdict(list)
    by_name: Dict[Tuple[str, str], List[Match]] = defaultdict(list)
    by_file: Dict[Tuple[str, str], List[Match]] = defaultdict(list)
    for label, rows in inv.items():
        for r in rows:
            if r.get("collected") == "no":
                continue
            match = (r["op"], r["op_file"])
            by_op[(label, r["op"])].append(match)
            for name in (r.get("op_names") or "").split(";"):
                name = name.strip()
                if name:
                    by_name[(label, name)].append(match)
            by_file[(label, r["op_file"])].append(match)
            base = r["op_file"].rsplit("/", 1)[-1]
            if base != r["op_file"]:
                by_file[(label, base)].append(match)
    return by_op, by_name, by_file


def _candidates(repo: str, name: str,
                by_op: Dict[Tuple[str, str], List[Match]],
                by_name: Dict[Tuple[str, str], List[Match]],
                by_file: Dict[Tuple[str, str], List[Match]]) -> List[Match]:
    """The table writes either the test function, the benchmark's op_name or the
    op_file; try them in that order and stop at the first kind that matches."""
    for table in (by_op, by_name, by_file):
        found = sorted(set(table.get((repo, name), [])))
        if found:
            return found
    return []


def resolve(entries: List[Entry]) -> Tuple[List[str], List[dict], List[str]]:
    """(specs, report rows, errors). A name missing from both repos is reported,
    not an error; a name that matches several tests in one repo (or in both) is
    expanded to every match, marked `multi`, so the batch runs all of them."""
    by_op, by_name, by_file = _inventory()
    specs: List[str] = []
    report: List[dict] = []
    errors: List[str] = []

    def add(entry: Entry, repo: str, match: Match, status: str) -> None:
        op, op_file = match
        specs.append(f"{repo}:{op}@{op_file}")
        report.append({"line": entry.line, "display": entry.display, "repo": repo,
                       "op": op, "op_file": op_file, "status": status})

    def expand(entry: Entry, repo: str, matches: List[Match], status: str) -> None:
        for match in matches:
            add(entry, repo, match, status)

    def candidates(entry: Entry, repo: str) -> List[Match]:
        matches = _candidates(repo, entry.op, by_op, by_name, by_file)
        if entry.op_file:  # the cell pins one benchmark file
            matches = [m for m in matches if m[1] == entry.op_file]
        return matches

    for e in entries:
        # "" (no repo in the cell) searches both; an op_file pins one benchmark
        # file in whichever repos have it.
        repos = [e.repo] if e.repo else ["main", "vllm"]
        present: Dict[str, List[Match]] = {}
        for repo in repos:
            matches = candidates(e, repo)
            if matches:
                present[repo] = matches
        if not present:
            spec = f"{e.repo}:{e.op}" if e.repo else e.op
            if e.op_file:
                spec += f"@{e.op_file}"
            if e.repo:
                errors.append(f"line {e.line}: no {spec} in the inventory")
            else:
                print(f"{LOG} line {e.line}: {e.display} is in neither main nor vllm; "
                      "recorded as unresolved", file=sys.stderr)
            report.append({"line": e.line, "display": e.display, "repo": e.repo,
                           "op": e.op, "op_file": e.op_file or "", "status": "missing"})
            continue
        multi = any(len(matches) > 1 for matches in present.values())
        status = "multi" if multi else ("both" if len(present) == 2 else next(iter(present)))
        if multi:
            total = sum(len(matches) for matches in present.values())
            print(f"{LOG} line {e.line}: {e.display} matches {total} tests; running all",
                  file=sys.stderr)
        elif status == "both":
            print(f"{LOG} line {e.line}: {e.display} is in both repos; running main and vllm",
                  file=sys.stderr)
        for repo, matches in present.items():
            expand(e, repo, matches, status)
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
