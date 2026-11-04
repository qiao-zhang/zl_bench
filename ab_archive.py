#!/usr/bin/env python3
"""Archive one AB experiment's evidence for a selected set of ops.

Inputs are the directories run_batch.py / ab_run.py / ir_gate.py leave behind:
a record batch (or a single-op run), the AB rounds, the optional compile gate
and the overall CSV/XLSX reports.  For the selected ops every existing
evidence root is copied below --out under the same path it has below --root,
the reports go to reports/, and archive_manifest.json plus ARCHIVE_README.txt
describe what was copied and what was missing.

The rules this implements are in docs/ab_archive.zh-CN.md; the acceptance
criteria there (A01-A42) are what tests/test_ab_archive.py checks.

  python ab_archive.py --root runs/experiment --ab-dir runs/experiment/ab \\
      --record runs/experiment/base --gate runs/experiment/gate \\
      --csv runs/experiment/ab/rows.csv --xlsx rlc_report.xlsx \\
      --ops main:chunk_kda --out runs/analysis_selected --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

import ab_compare

LOG = "[archive]"
REPOS = ("main", "vllm")
MANIFEST = "archive_manifest.json"
README = "ARCHIVE_README.txt"
REPORT_DIR = "reports"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
TMP_SUFFIX = ".tmp"
LOCK_PREFIX = "~$"
BATCH_FILES = ("ops_status.csv", "summary.csv")
AB_ROOT_FILES = ("ab.json", "accuracy_no_test.md", "accuracy_map.suggested.txt")
GATE_ROOT_FILES = ("variants.tsv", "report.tsv", "rows.tsv", "changed_ops.txt", "errors.log")
GATE_SIDES = ("base", "new")
SIDE_MARK = "######## "
ERROR_MARK = "== "


class ArchiveError(Exception):
    """A fatal problem: nothing is published, the staging dir says why."""


# ---------------- small helpers ----------------

class OpKey(NamedTuple):
    repo: str
    op: str
    op_file: str

    @property
    def label(self) -> str:
        name = f"{self.op}@{self.op_file}" if self.op_file else self.op
        return f"{self.repo}:{name}"


class Report(NamedTuple):
    path: Path
    size: int
    sha256: str


class Entry(NamedTuple):
    role: str
    scope: str
    archive: str          # posix path below --out, "." for the --out root
    base: Path            # files are made relative to this to get their target
    files: Tuple[Path, ...]
    names: Tuple[str, ...] = ()


def _is_reparse(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and attrs & flag)


def excluded(path: Path) -> bool:
    """Pipeline scratch, caches and lock files; .complete stays."""
    name = path.name
    if name in SKIP_DIRS:
        return True
    if name.endswith(TMP_SUFFIX):
        return True
    return name.startswith(LOCK_PREFIX) and name.endswith(".xlsx")


def iter_files(root: Path) -> Iterator[Path]:
    """Every file under root, dot files included; links are refused."""
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for item in entries:
                path = Path(item.path)
                st = item.stat(follow_symlinks=False)
                if _is_reparse(st):
                    raise ArchiveError(f"symbolic link / reparse point in source tree: {path}")
                if item.is_dir(follow_symlinks=False):
                    if not excluded(path):
                        stack.append(path)
                elif not excluded(path):
                    yield path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def rel_posix(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        raise ArchiveError(f"{path} is outside --root {root}") from None
    text = PurePosixPath(*rel.parts).as_posix()
    return text or "."


def source_rel(root: Path, path: Path) -> str:
    """Source path for the manifest: relative when under --root, absolute otherwise."""
    try:
        return PurePosixPath(*path.relative_to(root).parts).as_posix()
    except ValueError:
        return str(path)


def read_status(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def key_of(row: Dict[str, str]) -> Optional[OpKey]:
    repo = (row.get("repo") or "").strip()
    op = (row.get("op") or "").strip()
    if not repo or not op:
        return None
    return OpKey(repo, op, (row.get("op_file") or "").strip())


def run_name_of(row: Dict[str, str]) -> str:
    return (row.get("run_name") or "").strip()


def parse_spec(spec: str) -> OpKey:
    repo, _, rest = spec.rpartition(":")
    repo = repo or "main"
    if repo not in REPOS:
        raise ArchiveError(f"{spec!r}: unknown repo {repo!r} (known: {', '.join(REPOS)})")
    op, _, op_file = rest.partition("@")
    if not op:
        raise ArchiveError(f"{spec!r}: no op name")
    return OpKey(repo, op, op_file)


def load_specs(args: argparse.Namespace) -> List[OpKey]:
    specs: List[str] = list(args.ops or [])
    if args.ops_file is not None:
        try:
            text = args.ops_file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise ArchiveError(f"cannot read --ops-file {args.ops_file}: {exc}") from exc
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                specs.append(line)
    if not specs:
        raise ArchiveError("no ops selected: use --ops or --ops-file")
    out: List[OpKey] = []
    seen: Set[OpKey] = set()
    for spec in specs:
        key = parse_spec(spec)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---------------- experiment index ----------------

class Experiment:
    """Historical identity index built from the status CSVs and ab.json."""

    def __init__(self, root: Path, ab_dir: Path, record: Path, gate: Optional[Path]):
        self.root = root
        self.ab_dir = ab_dir
        self.record = record
        self.gate = gate
        self.manifest: dict = {}
        self.record_layout = "batch"
        self.record_keys: Dict[OpKey, str] = {}
        self.rounds: List[Tuple[int, str, Path]] = []
        self.round_keys: Dict[Tuple[int, str], Dict[OpKey, str]] = {}
        self.run_index: Dict[Tuple[str, str], Set[OpKey]] = {}
        self.accuracy: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.accuracy_keys: Dict[Tuple[str, str], OpKey] = {}
        self.notes: List[str] = []
        self._load()

    # -- loading

    def _load(self) -> None:
        self._load_record()
        self._load_rounds()
        self._load_accuracy()

    def _load_record(self) -> None:
        if self._looks_batch(self.record):
            self.record_layout = "batch"
            for row in read_status(self.record / "ops_status.csv"):
                key, run = key_of(row), run_name_of(row)
                if key and run:
                    self.record_keys.setdefault(key, run)
                    self.run_index.setdefault((key.repo, run), set()).add(key)
            return
        self.record_layout = "single-run"
        self._load_single_record()

    @staticmethod
    def _looks_batch(record: Path) -> bool:
        if (record / "ops_status.csv").is_file():
            return True
        return any((p / "run.log").is_file() or (p / "ttir").is_dir()
                   for p in record.glob("*/*") if p.is_dir())

    def _load_single_record(self) -> None:
        run_name = self.record.name
        found: Set[OpKey] = set()
        for parent in (self.record.parent, self.record.parent.parent):
            status = parent / "ops_status.csv"
            if not status.is_file():
                continue
            for row in read_status(status):
                key = key_of(row)
                if key and run_name_of(row) == run_name:
                    found.add(key)
            if found:
                break
        if len(found) != 1:
            hint = ", ".join(sorted(k.label for k in found)) if found else "no matching run_name"
            raise ArchiveError(
                f"{self.record}: cannot identify the op of a single-op record from a parent "
                f"ops_status.csv ({hint}); run.log/rows.tsv names are only a cross-check")
        key = found.pop()
        self.record_keys[key] = run_name
        self.run_index.setdefault((key.repo, run_name), set()).add(key)

    def _load_rounds(self) -> None:
        for index, side, path in ab_compare.round_dirs(self.ab_dir):
            self.rounds.append((index, side, path))
            rows = read_status(path / "ops_status.csv")
            if not rows:
                self.notes.append(f"round {index}{side}: no ops_status.csv, participation check degraded")
            keys: Dict[OpKey, str] = {}
            for row in rows:
                key, run = key_of(row), run_name_of(row)
                if key and run:
                    keys.setdefault(key, run)
                    self.run_index.setdefault((key.repo, run), set()).add(key)
            self.round_keys[(index, side)] = keys

    def _load_accuracy(self) -> None:
        path = self.ab_dir / ab_compare.MANIFEST
        if not path.is_file():
            raise ArchiveError(f"{self.ab_dir} has no {ab_compare.MANIFEST}")
        try:
            data = json.loads(path.read_text())
        except ValueError as exc:
            raise ArchiveError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or "record" not in data or "sides" not in data:
            raise ArchiveError(f"{path} is not an ab_run.py manifest: it needs 'record' and 'sides'")
        self.manifest = data
        sides = (data.get("accuracy") or {}).get("sides") or {}
        unresolved: List[str] = []
        for side in ("a", "b"):
            rows = sides.get(side) or {}
            if not isinstance(rows, dict):
                continue
            self.accuracy[side] = {str(run): row for run, row in rows.items() if isinstance(row, dict)}
            for run in self.accuracy[side]:
                try:
                    key = self._map_run(run)
                except ArchiveError as exc:
                    unresolved.append(str(exc))
                    continue
                if key is None:
                    unresolved.append(f"accuracy run {run} has no status CSV entry")
                else:
                    self.accuracy_keys[(side, run)] = key
        if unresolved:
            raise ArchiveError("cannot map accuracy records to ops: " + "; ".join(sorted(unresolved)))

    def _map_run(self, run: str) -> Optional[OpKey]:
        repo, _, run_name = run.partition("/")
        candidates = set(self.run_index.get((repo, run_name), set()))
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            participating = self._round_keys() | set(self.accuracy_keys.values())
            narrowed = candidates & participating
            if len(narrowed) == 1:
                return next(iter(narrowed))
            raise ArchiveError(
                f"accuracy run {run} maps to several ops: "
                + ", ".join(sorted(k.label for k in candidates)))
        return None

    def _round_keys(self) -> Set[OpKey]:
        out: Set[OpKey] = set()
        for keys in self.round_keys.values():
            out |= set(keys)
        return out

    # -- queries

    def participating(self) -> Set[OpKey]:
        return self._round_keys() | set(self.accuracy_keys.values())

    def known(self) -> Set[OpKey]:
        return set(self.record_keys) | self._round_keys() | set(self.accuracy_keys.values())

    def run_for(self, key: OpKey) -> Optional[str]:
        if key in self.record_keys:
            return self.record_keys[key]
        for index, side in self.round_keys:
            run = self.round_keys[(index, side)].get(key)
            if run:
                return run
        return None


# ---------------- selection ----------------

def select(exp: Experiment, specs: Sequence[OpKey]) -> List[OpKey]:
    known = exp.known()
    participating = exp.participating()
    chosen: List[OpKey] = []

    def accept(key: OpKey) -> None:
        if key not in participating:
            raise ArchiveError(f"{key.label}: not part of this AB (no round or accuracy record in --ab-dir)")
        if key not in chosen:
            chosen.append(key)

    for spec in specs:
        if spec.op_file:
            if spec not in known:
                raise ArchiveError(f"{spec.label}: no such op in this experiment")
            accept(spec)
            continue
        candidates = sorted(k for k in known if k.repo == spec.repo and k.op == spec.op)
        if not candidates:
            raise ArchiveError(f"{spec.label}: no such op in this experiment")
        if len(candidates) > 1:
            files = ", ".join(k.op_file or "<unknown>" for k in candidates)
            raise ArchiveError(f"{spec.label}: ambiguous, add @op_file (candidates: {files})")
        accept(candidates[0])
    return chosen


# ---------------- gate helpers ----------------

def read_variants(gate: Path) -> List[Dict[str, str]]:
    path = gate / "variants.tsv"
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def parse_gate_errors(gate: Path) -> Dict[Tuple[str, str], str]:
    """{(side, cache_hash): text} from errors.log's '######## side' + '== <ttir>' blocks."""
    path = gate / "errors.log"
    if not path.is_file():
        return {}
    out: Dict[Tuple[str, str], str] = {}
    side, current = "", None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(SIDE_MARK):
            side = line[len(SIDE_MARK):].strip()
            current = None
        elif line.startswith(ERROR_MARK):
            ttir = Path(line[len(ERROR_MARK):].strip())
            current = (side, ttir.parent.name)
            out[current] = line
        elif current is not None:
            out[current] += "\n" + line
    return out


def gate_compiled_base(gate: Path) -> bool:
    """Whether this gate ever compiled the base side.

    Either the error log has a base section (a failure) or base/ holds a real
    artifact; a leftover empty directory is not evidence (ir_gate.py only
    removes errors.log, never base/).
    """
    errors = gate / "errors.log"
    if errors.is_file():
        for line in errors.read_text(errors="replace").splitlines():
            if line.strip() == SIDE_MARK + "base":
                return True
    base = gate / "base"
    return base.is_dir() and any(base.rglob("*.ttgir"))


def expected_hashes(run_dirs: Sequence[Path]) -> Set[str]:
    """Hashes a gate can recompile: a .ttir with its compile-options .json."""
    out: Set[str] = set()
    for run_dir in run_dirs:
        ttir = run_dir / "ttir"
        if not ttir.is_dir():
            continue
        for path in ttir.glob("*/*.ttir"):
            if path.with_suffix(".json").is_file():
                out.add(path.parent.name)
    return out


# ---------------- plan ----------------

class Plan:
    def __init__(self) -> None:
        self.entries: List[Entry] = []
        self.reports: List[Report] = []
        self.files: List[Tuple[Path, str]] = []
        self.missing: List[dict] = []
        self.not_applicable: List[dict] = []
        self.notes: List[str] = []
        self.hashes: List[dict] = []
        self.selected: List[dict] = []
        self.record_layout = "batch"

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p, _ in self.files)

    def sources_json(self, root: Path) -> List[dict]:
        out = []
        for entry in self.entries:
            item = {
                "role": entry.role,
                "scope": entry.scope,
                "source": source_rel(root, entry.base),
                "archive": entry.archive,
                "files": len(entry.files),
                "bytes": sum(p.stat().st_size for p in entry.files),
            }
            if entry.names:
                item["names"] = list(entry.names)
            out.append(item)
        return out

    def reports_json(self, status: str) -> List[dict]:
        return [{
            "source": str(report.path),
            "archive": f"{REPORT_DIR}/{report.path.name}",
            "size": report.size,
            "sha256": report.sha256,
            "verified": status == "completed",
        } for report in self.reports]

    def manifest(self, args: argparse.Namespace, status: str,
                 errors: Sequence[dict] = ()) -> dict:
        data = {
            "format": 1,
            "created": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "evidence_complete": not self.missing,
            "errors": list(errors),
            "inputs": {
                "root": str(args.root),
                "ab_dir": str(args.ab_dir),
                "record": str(args.record),
                "gate": str(args.gate) if args.gate else "",
                "csv": str(args.csv),
                "xlsx": str(args.xlsx),
                "out": str(args.out),
            },
            "record_layout": self.record_layout,
            "selected": self.selected,
            "sources": self.sources_json(args.root),
            "reports": self.reports_json(status),
            "hashes": self.hashes,
            "missing": self.missing,
            "not_applicable": self.not_applicable,
            "notes": self.notes,
        }
        if args.manifest_detail == "files":
            data["files_detail"] = [
                {"source": source_rel(args.root, src), "archive": dst, "bytes": src.stat().st_size}
                for src, dst in self.files
            ]
        return data


def add_dir(plan: Plan, role: str, scope: str, path: Path, root: Path) -> None:
    plan.entries.append(Entry(role, scope, rel_posix(root, path), path, tuple(iter_files(path))))


def add_files(plan: Plan, role: str, scope: str, base: Path, files: Sequence[Path],
              root: Path, names: Sequence[str] = ()) -> None:
    real = tuple(p for p in files if p.is_file() and not excluded(p))
    if not real and not names:
        return
    plan.entries.append(Entry(role, scope, rel_posix(root, base), base, real, tuple(names)))


def add_group(plan: Plan, role: str, scope: str, base: Path, candidates: Sequence[Path],
              root: Path) -> None:
    present = tuple(p for p in candidates if p.is_file() and not excluded(p))
    if not present:
        return
    plan.entries.append(Entry(role, scope, rel_posix(root, base), base, present,
                              tuple(p.name for p in present)))


def build_reports(args: argparse.Namespace) -> List[Report]:
    out: List[Report] = []
    for path in (args.csv, args.xlsx):
        if not path.is_file():
            raise ArchiveError(f"report not found: {path}")
        out.append(Report(path, path.stat().st_size, sha256_of(path)))
    if args.csv.name == args.xlsx.name:
        raise ArchiveError(f"--csv and --xlsx have the same file name {args.csv.name!r}")
    return out


def _missing(item: str, path: Path, reason: str, root: Path, **extra: str) -> dict:
    data = {"item": item, "path": source_rel(root, path), "reason": reason}
    data.update(extra)
    return data


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def recorded_baseline(run_dirs: Sequence[Path], cache_hash: str) -> str:
    """'ok' / 'missing' / 'no-record' for the recorded ttgir a gate falls back to."""
    for run_dir in run_dirs:
        index = run_dir / "ttgir" / "index.tsv"
        if not index.is_file():
            continue
        for row in read_tsv(index):
            if row.get("cache_hash") != cache_hash or int(row.get("launches") or 0) <= 0:
                continue
            path = (run_dir / "ttgir" / (row.get("shape") or "") / (row.get("kernel") or "")
                    / (row.get("file") or ""))
            return "ok" if path.is_file() else "missing"
    return "no-record"


def build_gate(plan: Plan, exp: Experiment, keys: Sequence[OpKey],
               run_dirs: Sequence[Path]) -> None:
    gate = exp.gate
    assert gate is not None
    variants = read_variants(gate)
    errors = parse_gate_errors(gate)
    compiled_base = gate_compiled_base(gate)
    base_failed = any(side == "base" for side, _ in errors)
    by_hash = {row.get("cache_hash", ""): row for row in variants}
    expected = expected_hashes(run_dirs)

    for cache_hash in sorted(expected):
        row = by_hash.get(cache_hash, {})
        plan.hashes.append({
            "hash": cache_hash,
            "selected": True,
            "changed": row.get("changed", ""),
            "first_diff": row.get("first_diff", ""),
            "ttgir": row.get("ttgir", ""),
        })

    for side in GATE_SIDES:
        files: List[Path] = []
        for cache_hash in sorted(expected):
            path = gate / side / cache_hash
            row = by_hash.get(cache_hash, {})
            if path.is_dir():
                files.extend(iter_files(path))
                continue
            if side == "base" and not compiled_base:
                if row.get("changed") != "?":
                    continue  # the baseline came from the recorded ttgir, gate has no base result
                if ("base", cache_hash) in errors:
                    plan.missing.append(_missing(f"gate base {cache_hash}", path,
                                                 "compilation failed (see errors.log)",
                                                 exp.root, hash=cache_hash, side=side))
                elif recorded_baseline(run_dirs, cache_hash) == "missing":
                    plan.missing.append(_missing(f"gate base {cache_hash}", path,
                                                 "recorded baseline ttgir is gone", exp.root,
                                                 hash=cache_hash, side=side))
                else:
                    plan.not_applicable.append({
                        "item": f"gate base {cache_hash}",
                        "reason": "no --base and no recorded baseline for this variant",
                    })
                continue
            if row.get("changed") in ("0", "1"):
                reason = "a compiled variant has no result on this side"
            elif (side, cache_hash) in errors:
                reason = "compilation failed (see errors.log)"
            else:
                reason = "no result"
            plan.missing.append(_missing(f"gate {side} {cache_hash}", path, reason, exp.root,
                                         hash=cache_hash, side=side))
        if files:
            add_files(plan, "gate", "op", gate, files, exp.root)

    if not compiled_base and not base_failed:
        plan.not_applicable.append({
            "item": "gate base side",
            "reason": "ir_gate.py ran without --base, so gate/base never exists",
        })
    add_group(plan, "gate", "global", gate, [gate / name for name in GATE_ROOT_FILES], exp.root)


def check_run_dir(plan: Plan, run_dir: Path, root: Path) -> None:
    """A run marked complete by .complete must still have its run.log."""
    if (run_dir / ".complete").is_file() and not (run_dir / "run.log").is_file():
        plan.missing.append(_missing(
            source_rel(root, run_dir), run_dir / "run.log",
            ".complete marks a PASS/SKIP run but run.log is gone", root))


def check_ttir(plan: Plan, run_dir: Path, root: Path) -> None:
    """index.tsv lists the variants that should have ttir; classify what is gone by level."""
    index = run_dir / "ttgir" / "index.tsv"
    if not index.is_file():
        return
    wanted: List[Tuple[str, str]] = []
    for row in read_tsv(index):
        if int(row.get("launches") or 0) <= 0:
            continue
        cache_hash = (row.get("cache_hash") or "").strip()
        kernel = (row.get("kernel") or "").strip()
        if cache_hash and kernel:
            wanted.append((cache_hash, kernel))
    if not wanted:
        return
    label = source_rel(root, run_dir)
    ttir = run_dir / "ttir"
    if not ttir.is_dir():
        plan.not_applicable.append({
            "item": f"{label}/ttir",
            "reason": "no ttir at all (cache evicted or IR dump disabled)",
        })
        return
    absent = sorted({cache_hash for cache_hash, _ in wanted if not (ttir / cache_hash).is_dir()})
    if absent:
        plan.not_applicable.append({
            "item": f"{label}/ttir/<hash>",
            "reason": "variant ttir not in the cache",
            "hashes": absent,
        })
    for cache_hash, kernel in wanted:
        directory = ttir / cache_hash
        if not directory.is_dir():
            continue
        for suffix in (".ttir", ".json"):
            path = directory / f"{kernel}{suffix}"
            if not path.is_file():
                plan.missing.append(_missing(
                    f"{label} ttir {cache_hash}", path,
                    "recorded variant has no ttir file", root,
                    hash=cache_hash, kernel=kernel))


def build_plan(exp: Experiment, keys: Sequence[OpKey], args: argparse.Namespace) -> Plan:
    plan = Plan()
    plan.record_layout = exp.record_layout
    plan.notes.extend(exp.notes)
    plan.reports = build_reports(args)
    record_dirs: List[Path] = []
    not_in_rounds: Dict[str, List[str]] = {}

    for key in keys:
        if exp.record_layout == "single-run":
            source, run = exp.record, exp.record.name
        else:
            run = exp.record_keys.get(key)
            source = exp.record / key.repo / run if run else None
        if source is None or not source.is_dir():
            plan.missing.append(_missing(key.label, exp.record / key.repo / (run or key.op),
                                         "record run directory is missing", exp.root,
                                         op=key.label))
        else:
            add_dir(plan, "record", "op", source, exp.root)
            check_run_dir(plan, source, exp.root)
            check_ttir(plan, source, exp.root)
            record_dirs.append(source)
        plan.selected.append({
            "spec": key.label, "repo": key.repo, "op": key.op, "op_file": key.op_file,
            "run_name": run or "", "rounds": [], "sides": [],
            "accuracy": {},
        })
    if exp.record_layout == "batch":
        add_group(plan, "record", "global", exp.record,
                  [exp.record / name for name in BATCH_FILES], exp.root)

    by_spec = {item["spec"]: item for item in plan.selected}
    for index, side, path in exp.rounds:
        for key in keys:
            run = exp.round_keys[(index, side)].get(key)
            if run is None:
                not_in_rounds.setdefault(key.label, []).append(f"{index}{side}")
                continue
            run_dir = path / key.repo / run
            item = by_spec[key.label]
            if index not in item["rounds"]:
                item["rounds"].append(index)
            if side not in item["sides"]:
                item["sides"].append(side)
            if run_dir.is_dir():
                add_dir(plan, "ab", "op", run_dir, exp.root)
                check_run_dir(plan, run_dir, exp.root)
                check_ttir(plan, run_dir, exp.root)
            else:
                plan.missing.append(_missing(
                    key.label, run_dir, "AB round run directory is missing", exp.root,
                    op=key.label, round=index, side=side))
        add_group(plan, "ab", "global", path, [path / name for name in BATCH_FILES], exp.root)
    add_group(plan, "ab", "global", exp.ab_dir,
              [exp.ab_dir / name for name in AB_ROOT_FILES], exp.root)

    for (side, run), key in sorted(exp.accuracy_keys.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if key not in keys:
            continue
        run_name = run.partition("/")[2]
        accuracy_dir = exp.ab_dir / "accuracy" / side / key.repo / run_name
        row = exp.accuracy.get(side, {}).get(run) or {}
        item = by_spec[key.label]["accuracy"].setdefault(side, {})
        item["status"] = row.get("status", "")
        item["target"] = row.get("target", "")
        if accuracy_dir.is_dir():
            add_dir(plan, "accuracy", "op", accuracy_dir, exp.root)
        elif row.get("log"):
            plan.missing.append(_missing(
                f"{key.label} accuracy {side}", accuracy_dir,
                f"accuracy log declared but missing ({row.get('status', '')})", exp.root,
                op=key.label, side=side))
        else:
            plan.notes.append(f"{key.label}: no accuracy log on side {side} ({row.get('status', '')})")

    if exp.gate is not None:
        build_gate(plan, exp, keys, record_dirs)

    for label, rounds in not_in_rounds.items():
        plan.notes.append(f"{label}: not part of rounds {', '.join(rounds)}")

    plan.files = expand(plan)
    return plan


def expand(plan: Plan) -> List[Tuple[Path, str]]:
    """(source file, target path below --out) for every entry and report."""
    owners: Dict[Path, str] = {}
    files: List[Tuple[Path, str]] = []
    targets: Dict[str, Path] = {}
    for entry in plan.entries:
        for path in entry.files:
            owner = f"{entry.role}/{entry.scope}"
            if path in owners:
                raise ArchiveError(
                    f"evidence roots overlap: {path} is claimed by {owners[path]} and {owner}")
            owners[path] = owner
            try:
                rel = path.relative_to(entry.base)
            except ValueError:
                raise ArchiveError(f"{path} is not below {entry.base}") from None
            inner = PurePosixPath(*rel.parts).as_posix()
            target = inner if entry.archive == "." else f"{entry.archive}/{inner}"
            if target in targets:
                raise ArchiveError(f"two sources map to {target}")
            targets[target] = path
            files.append((path, target))
    for report in plan.reports:
        target = f"{REPORT_DIR}/{report.path.name}"
        if target in targets:
            raise ArchiveError(f"two sources map to {target}")
        targets[target] = report.path
        files.append((report.path, target))
    reserved = {MANIFEST, README}
    for target in targets:
        if target in reserved:
            raise ArchiveError(f"{target} would overwrite an archive artifact")
    return files


# ---------------- validation ----------------

def check_manifest_file(ab_dir: Path) -> dict:
    path = ab_dir / ab_compare.MANIFEST
    if not path.is_file():
        raise ArchiveError(f"{ab_dir} has no {ab_compare.MANIFEST} (not an ab_run.py output dir?)")
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise ArchiveError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "record" not in data or "sides" not in data:
        raise ArchiveError(f"{path} is not an ab_run.py manifest: it needs 'record' and 'sides'")
    return data


def check_gate(gate: Path) -> None:
    if not (gate / "variants.tsv").is_file():
        raise ArchiveError(
            f"{gate} does not look like an ir_gate.py output: no variants.tsv "
            "(the default output is runs/ir_gate_<timestamp>/, --gate should point there)")


def validate_target(out: Path, sources: Sequence[Path], reports: Sequence[Path]) -> None:
    if out.exists():
        if not out.is_dir():
            raise ArchiveError(f"--out exists and is not a directory: {out}")
        if any(out.iterdir()):
            raise ArchiveError(f"--out is not empty: {out}")
    for source in sources:
        if out == source or is_within(out, source) or is_within(source, out):
            raise ArchiveError(f"--out {out} overlaps source {source}")
    for report in reports:
        if out == report or is_within(report, out):
            raise ArchiveError(f"--out {out} would contain or overwrite {report}")


def available_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


# ---------------- output ----------------

def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _publish(staging: Path, out: Path) -> None:
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise ArchiveError(f"--out became non-empty before publishing: {out}")
        out.rmdir()
    os.rename(staging, out)


def _selected_line(item: dict) -> str:
    rounds = ",".join(map(str, item["rounds"])) if item["rounds"] else "-"
    return f"{item['spec']}  (run_name {item['run_name'] or '?'}, rounds {rounds})"


def _readme(plan: Plan, args: argparse.Namespace) -> str:
    lines = [
        "AB evidence archive",
        "",
        f"created      : {datetime.now().isoformat(timespec='seconds')}",
        f"ab dir       : {args.ab_dir}",
        f"record       : {args.record} ({plan.record_layout})",
        f"gate         : {args.gate if args.gate else '(not provided)'}",
        f"reports      : {', '.join(r.path.name for r in plan.reports)}",
        "",
        "Layout",
        "  base/, ab/, gate/   evidence roots, relative to --root (unchanged layout)",
        f"  {REPORT_DIR}/            the overall CSV/XLSX; they are NOT kept at their original path",
        f"  {MANIFEST}  machine-readable manifest (sources, reports, missing, notes)",
        "",
        "Selected ops",
    ]
    lines += ["  - " + _selected_line(item) for item in plan.selected]
    if plan.missing:
        lines += ["", f"Missing evidence ({len(plan.missing)}), see {MANIFEST} for all:"]
        lines += [f"  - {item['item']}: {item['reason']}" for item in plan.missing[:20]]
    if plan.not_applicable:
        lines += ["", "Not applicable"]
        lines += [f"  - {item['item']}: {item['reason']}" for item in plan.not_applicable]
    lines += [
        "",
        "Notes",
        "  - The overall CSV/XLSX are copied byte for byte; the XLSX was produced by rlc_report.py whose",
        "    --table/--notes inputs are manual and not archived, so it cannot be recomputed from this pack.",
        "  - The CSV columns depend on the ab_compare.py command that produced them; not recorded here.",
        "  - Absolute paths inside ab.json/run.log are untouched; nothing here is guaranteed to resume.",
        "  - When gate rows.tsv is absent, changes cannot be located per shape.",
    ]
    return "\n".join(lines) + "\n"


def print_plan(plan: Plan, args: argparse.Namespace) -> None:
    print(f"{LOG} record layout: {plan.record_layout}", flush=True)
    for item in plan.selected:
        print(f"{LOG} selected {item['spec']} (run_name {item['run_name'] or '?'})", flush=True)
    by_role: Dict[str, int] = {}
    for entry in plan.entries:
        by_role[entry.role] = by_role.get(entry.role, 0) + 1
    print(f"{LOG} {len(plan.entries)} evidence roots, {len(plan.files)} files, "
          f"{plan.total_bytes} bytes: " + ", ".join(f"{k} {v}" for k, v in sorted(by_role.items())),
          flush=True)
    for item in plan.not_applicable:
        print(f"{LOG} not applicable: {item['item']} ({item['reason']})", flush=True)
    for note in plan.notes:
        print(f"{LOG} note: {note}", flush=True)
    for item in plan.missing[:20]:
        print(f"{LOG} missing: {item['item']} ({item['reason']})", flush=True)
    if len(plan.missing) > 20:
        print(f"{LOG} missing: ... and {len(plan.missing) - 20} more in {MANIFEST}", flush=True)


def copy_out(plan: Plan, args: argparse.Namespace) -> int:
    out = args.out
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{out.name}.staging-"))
    _write_json(staging / MANIFEST, plan.manifest(args, "running"))
    total = len(plan.files)
    total_bytes = plan.total_bytes
    copied = 0
    last = time.monotonic()
    try:
        for index, (src, target) in enumerate(plan.files, 1):
            dst = staging / target
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += src.stat().st_size
            now = time.monotonic()
            if (index % args.progress_files == 0 or now - last >= args.progress_seconds
                    or index == total):
                print(f"{LOG} copied {index}/{total} files ({copied}/{total_bytes} bytes)",
                      flush=True)
                last = now
        for report in plan.reports:
            got = sha256_of(staging / REPORT_DIR / report.path.name)
            if got != report.sha256:
                raise ArchiveError(f"copied report differs from source: {report.path}")
        (staging / README).write_text(_readme(plan, args), encoding="utf-8")
        _write_json(staging / MANIFEST, plan.manifest(args, "completed"))
        _publish(staging, out)
    except Exception as exc:  # noqa: BLE001 - every failure keeps the staging dir for diagnosis
        data = plan.manifest(args, "failed", errors=[{
            "stage": "copy",
            "error": f"{type(exc).__name__}: {exc}",
        }])
        data["staging"] = str(staging)
        try:
            _write_json(staging / MANIFEST, data)
        except OSError:
            pass
        print(f"{LOG} FAILED: {exc}", file=sys.stderr, flush=True)
        print(f"{LOG} staging kept at {staging}", file=sys.stderr, flush=True)
        return 1
    print(f"{LOG} published {out} ({total} files, {plan.total_bytes} bytes)", flush=True)
    return 3 if (args.fail_on_missing and plan.missing) else 0


# ---------------- entry point ----------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ab_archive.py",
        description="Copy the evidence of selected ops from one AB experiment into a folder.")
    parser.add_argument("--root", type=Path, required=True,
                        help="common root of the directory sources (relative layout is kept)")
    parser.add_argument("--ab-dir", type=Path, required=True, help="ab_run.py output dir")
    parser.add_argument("--record", type=Path, required=True,
                        help="record batch dir or a single-op run dir used by this AB")
    parser.add_argument("--gate", type=Path, default=None, help="ir_gate.py output dir (optional)")
    parser.add_argument("--csv", type=Path, required=True, help="overall CSV report")
    parser.add_argument("--xlsx", type=Path, required=True, help="overall XLSX report")
    parser.add_argument("--ops", nargs="*", default=None, metavar="[repo:]op[@op_file]",
                        help="ops to archive (repo defaults to main)")
    parser.add_argument("--ops-file", type=Path, default=None,
                        help="file with one [repo:]op[@op_file] per line, # comments")
    parser.add_argument("--out", type=Path, required=True, help="target folder (must be empty)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    parser.add_argument("--fail-on-missing", action="store_true",
                        help="exit 3 after publishing when evidence is missing")
    parser.add_argument("--manifest-detail", choices=("summary", "files"), default="summary",
                        help="summary aggregates per evidence root; files adds every file")
    parser.add_argument("--progress-files", type=int, default=1000, metavar="N",
                        help="print progress every N files (default %(default)s)")
    parser.add_argument("--progress-seconds", type=float, default=10.0, metavar="M",
                        help="print progress at least every M seconds (default %(default)s)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root.resolve()
    args.ab_dir = args.ab_dir.resolve()
    args.record = args.record.resolve()
    args.gate = args.gate.resolve() if args.gate else None
    args.csv = args.csv.resolve()
    args.xlsx = args.xlsx.resolve()
    args.out = args.out.resolve()
    if args.ops_file is not None:
        args.ops_file = args.ops_file.resolve()
    try:
        for path, name in ((args.root, "--root"), (args.ab_dir, "--ab-dir"),
                           (args.record, "--record")):
            if not path.is_dir():
                raise ArchiveError(f"{name} is not a directory: {path}")
        if args.gate is not None:
            if not args.gate.is_dir():
                raise ArchiveError(f"--gate is not a directory: {args.gate}")
            check_gate(args.gate)
        check_manifest_file(args.ab_dir)
        specs = load_specs(args)
        exp = Experiment(args.root, args.ab_dir, args.record, args.gate)
        keys = select(exp, specs)
        plan = build_plan(exp, keys, args)
        sources = [args.root, args.ab_dir, args.record]
        if args.gate is not None:
            sources.append(args.gate)
        validate_target(args.out, sources, [r.path for r in plan.reports])
        free = available_bytes(args.out.parent)
        if free < plan.total_bytes:
            raise ArchiveError(f"not enough space in {args.out.parent}: "
                               f"need {plan.total_bytes} bytes, {free} free")
        print_plan(plan, args)
        if args.dry_run:
            print(f"{LOG} dry-run: nothing written", flush=True)
            return 3 if (args.fail_on_missing and plan.missing) else 0
        return copy_out(plan, args)
    except ArchiveError as exc:
        print(f"{LOG} error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
