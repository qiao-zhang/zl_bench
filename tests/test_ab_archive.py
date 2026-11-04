"""Tests for ab_archive.py: every case builds its own tiny AB experiment on disk."""

import csv
import json
import shutil
from pathlib import Path
from typing import NamedTuple, Optional

import ab_archive
import ab_compare
import run_batch

STATUS = run_batch.STATUS_FIELDS


def write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def status_row(op: str = "abs", op_file: str = "abs", run_name: str = "abs",
               repo: str = "main") -> dict:
    row = {field: "" for field in STATUS}
    row.update({"repo": repo, "op": op, "op_file": op_file, "run_name": run_name,
                "status": "PASS", "return_code": "0", "elapsed_ms": "10",
                "rows_total": "1", "rows_success": "1"})
    return row


def write_status(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS)
        writer.writeheader()
        writer.writerows(rows)


def write_run(run: Path, kernel: str = "abs_kernel", cache_hash: str = "H_abs") -> None:
    write(run / "run.log", "Operator: abs  (dtype=torch.float16, mode=kernel, level=core)\n")
    write(run / "rows.tsv",
          "op_name\tdtype\trow\tshape\tkernels\tscope\n"
          f"abs\ttorch.float16\t0\t4x4\t{cache_hash}\ttimed\n")
    write(run / "ttgir" / "index.tsv",
          "shape\tkernel\tfile\tlaunches\tsweep_launches\tnum_warps\tnum_stages\t"
          "constexprs\tcache_hash\n"
          f"4x4\t{kernel}\tw4s3.ttgir\t1\t0\t4\t3\t{{}}\t{cache_hash}\n")
    write(run / "ttir" / cache_hash / f"{kernel}.ttir", "module {}\n")
    write(run / "ttir" / cache_hash / f"{kernel}.json", "{}\n")
    write(run / ".complete", "PASS\n")


class Exp(NamedTuple):
    root: Path
    record: Path
    ab: Path
    gate: Optional[Path]
    csv: Path
    xlsx: Path


def build(base: Path, ops=("abs",), rounds: int = 2, gate: bool = False,
          accuracy: bool = True, skipped: Optional[dict] = None) -> Exp:
    root = base / "exp"
    record = root / "base"
    rows = [status_row(op=name, op_file=name, run_name=name) for name in ops]
    write_status(record / "ops_status.csv", rows)
    write(record / "summary.csv", "op\n" + "\n".join(ops) + "\n")
    for name in ops:
        write_run(record / "main" / name, kernel=f"{name}_kernel", cache_hash=f"H_{name}")
    ab = root / "ab"
    manifest = {
        "record": str(record),
        "sides": {"a": {"name": "A", "pythonpath": "", "env": {}},
                  "b": {"name": "B", "pythonpath": "", "env": {}}},
    }
    if accuracy:
        manifest["accuracy"] = {
            "skipped": dict(skipped or {}),
            "sides": {side: {f"main/{name}": {
                "status": "pass", "target": f"tests/test_{name}.py",
                "log": f"accuracy/{side}/main/{name}/gate.log"} for name in ops}
                for side in ("a", "b")},
        }
    write(ab / ab_compare.MANIFEST, json.dumps(manifest))
    for index in range(1, rounds + 1):
        for side in ("a", "b"):
            round_dir = ab / f"r{index}_{side}"
            write_status(round_dir / "ops_status.csv", rows)
            write(round_dir / "summary.csv", "op\n" + "\n".join(ops) + "\n")
            for name in ops:
                write_run(round_dir / "main" / name, kernel=f"{name}_kernel",
                          cache_hash=f"H_{name}")
    for side in ("a", "b"):
        for name in ops:
            write(ab / "accuracy" / side / "main" / name / "gate.log", "ok\n")
    csv_path = write(ab / "rows.csv", "run,op,row\n")
    xlsx = write(base / "rlc_report.xlsx", "xlsx-bytes")
    gate_dir = None
    if gate:
        gate_dir = root / "gate"
        write(gate_dir / "variants.tsv",
              "op\tcache_hash\tkernel\tchanged\tfirst_diff\tttgir\top_count_diff\n"
              + "".join(f"main:{name}@{name}\tH_{name}\t{name}_kernel\t1\tttgir\tover\t\n"
                        for name in ops))
        write(gate_dir / "report.tsv", "op\n")
        write(gate_dir / "rows.tsv", "run\n")
        write(gate_dir / "changed_ops.txt", "main:abs@abs\n")
        for name in ops:
            for side in ("base", "new"):
                target = gate_dir / side / f"H_{name}"
                write(target / f"{name}_kernel.ttgir", "module {}\n")
                write(target / f"{name}_kernel.digest.json", "{}\n")
    return Exp(root, record, ab, gate_dir, csv_path, xlsx)


def run(exp: Exp, out: Path, extra=(), record: Optional[Path] = None,
        ab_dir: Optional[Path] = None, csv_path: Optional[Path] = None,
        xlsx: Optional[Path] = None, gate: Optional[Path] = None) -> int:
    argv = [
        "--root", str(exp.root),
        "--ab-dir", str(ab_dir or exp.ab),
        "--record", str(record or exp.record),
        "--csv", str(csv_path or exp.csv),
        "--xlsx", str(xlsx or exp.xlsx),
        "--out", str(out),
        *extra,
    ]
    if gate is not None:
        argv += ["--gate", str(gate)]
    return ab_archive.main(argv)


def manifest_of(out: Path) -> dict:
    return json.loads((out / ab_archive.MANIFEST).read_text(encoding="utf-8"))


def test_archives_selected_ops_without_gate(tmp_path):
    exp = build(tmp_path)
    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs"]) == 0

    assert (out / "base" / "main" / "abs" / "run.log").is_file()
    assert (out / "ab" / "r1_a" / "main" / "abs" / "run.log").is_file()
    assert (out / "ab" / "r2_b" / "main" / "abs" / "run.log").is_file()
    assert (out / "ab" / "accuracy" / "a" / "main" / "abs" / "gate.log").is_file()
    assert not (out / "ab" / "rows.csv").exists()
    assert (out / "reports" / "rows.csv").read_bytes() == exp.csv.read_bytes()
    assert (out / "reports" / "rlc_report.xlsx").read_bytes() == exp.xlsx.read_bytes()

    data = manifest_of(out)
    assert data["status"] == "completed"
    assert data["evidence_complete"] is True
    roles = {(item["role"], item["scope"]) for item in data["sources"]}
    assert ("record", "op") in roles
    assert ("record", "global") in roles
    assert ("ab", "op") in roles
    assert ("accuracy", "op") in roles
    assert ("gate", "op") not in roles
    assert all("names" in item for item in data["sources"] if item["scope"] == "global")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()
                and p.name not in {ab_archive.MANIFEST, ab_archive.README})
    counted = sum(item["bytes"] for item in data["sources"]) + \
        sum(item["size"] for item in data["reports"])
    assert counted == total


def test_gate_copies_expected_hashes_and_root_files(tmp_path):
    exp = build(tmp_path, gate=True)
    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "main:abs@abs"], gate=exp.gate) == 0

    assert (out / "gate" / "base" / "H_abs" / "abs_kernel.ttgir").is_file()
    assert (out / "gate" / "new" / "H_abs" / "abs_kernel.ttgir").is_file()
    assert (out / "gate" / "variants.tsv").is_file()
    data = manifest_of(out)
    assert any(item["selected"] for item in data["hashes"])
    assert ("gate", "global") in {(item["role"], item["scope"]) for item in data["sources"]}


def test_ambiguous_op_needs_op_file(tmp_path):
    exp = build(tmp_path)
    extra = status_row(op="abs", op_file="abs_other", run_name="abs@abs_other")
    rows = list(csv.DictReader((exp.record / "ops_status.csv").open(newline="")))
    write_status(exp.record / "ops_status.csv", rows + [extra])
    write_run(exp.record / "main" / "abs@abs_other", kernel="abs_other_kernel",
              cache_hash="H2")
    for index in (1, 2):
        for side in ("a", "b"):
            round_dir = exp.ab / f"r{index}_{side}"
            rows = list(csv.DictReader((round_dir / "ops_status.csv").open(newline="")))
            write_status(round_dir / "ops_status.csv", rows + [extra])
            write_run(round_dir / "main" / "abs@abs_other", kernel="abs_other_kernel",
                      cache_hash="H2")

    first = tmp_path / "first"
    assert run(exp, first, ["--ops", "abs"]) == 1
    assert not first.exists()
    assert run(exp, tmp_path / "second", ["--ops", "main:abs@abs"]) == 0


def test_op_only_in_record_is_not_part_of_this_ab(tmp_path):
    exp = build(tmp_path)
    ghost = status_row(op="ghost", op_file="ghost", run_name="ghost")
    rows = list(csv.DictReader((exp.record / "ops_status.csv").open(newline="")))
    write_status(exp.record / "ops_status.csv", rows + [ghost])
    write_run(exp.record / "main" / "ghost", kernel="ghost_kernel", cache_hash="HG")

    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "ghost"]) == 1
    assert not out.exists()


def test_single_run_record_uses_parent_status(tmp_path):
    exp = build(tmp_path, rounds=1)
    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs"], record=exp.record / "main" / "abs") == 0
    assert (out / "base" / "main" / "abs" / "run.log").is_file()
    assert manifest_of(out)["record_layout"] == "single-run"


def test_single_run_record_without_parent_status_is_rejected(tmp_path):
    exp = build(tmp_path, rounds=1)
    record = exp.record / "main" / "abs"
    (exp.record / "ops_status.csv").unlink()
    rows = list(csv.DictReader((exp.ab / "r1_a" / "ops_status.csv").open(newline="")))
    assert rows
    assert run(exp, tmp_path / "archive", ["--ops", "abs"], record=record) == 1


def test_missing_run_log_counts_as_missing_and_strict_exit(tmp_path):
    exp = build(tmp_path)
    (exp.ab / "r1_a" / "main" / "abs" / "run.log").unlink()

    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs"]) == 0
    data = manifest_of(out)
    assert any("run.log" in item["reason"] for item in data["missing"])

    strict = tmp_path / "strict"
    assert run(exp, strict, ["--ops", "abs", "--fail-on-missing"]) == 3


def test_dry_run_writes_nothing_and_nonempty_target_is_rejected(tmp_path):
    exp = build(tmp_path)
    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs", "--dry-run"]) == 0
    assert not out.exists()

    write(out / "keep.txt", "x")
    assert run(exp, out, ["--ops", "abs"]) == 1


def test_manifest_detail_files(tmp_path):
    exp = build(tmp_path)
    summary = tmp_path / "summary"
    assert run(exp, summary, ["--ops", "abs"]) == 0
    assert "files_detail" not in manifest_of(summary)

    full = tmp_path / "full"
    assert run(exp, full, ["--ops", "abs", "--manifest-detail", "files"]) == 0
    detail = manifest_of(full)["files_detail"]
    assert detail and all("archive" in item and "bytes" in item for item in detail)


def test_ttir_missing_levels(tmp_path):
    whole = build(tmp_path / "whole")
    shutil.rmtree(whole.record / "main" / "abs" / "ttir")
    out = tmp_path / "a"
    assert run(whole, out, ["--ops", "abs"]) == 0
    assert any(item["item"].endswith("/ttir")
               for item in manifest_of(out)["not_applicable"])

    one_hash = build(tmp_path / "one")
    shutil.rmtree(one_hash.record / "main" / "abs" / "ttir" / "H_abs")
    out = tmp_path / "b"
    assert run(one_hash, out, ["--ops", "abs"]) == 0
    assert any(item["item"].endswith("/ttir/<hash>")
               for item in manifest_of(out)["not_applicable"])

    file_gone = build(tmp_path / "file")
    (file_gone.record / "main" / "abs" / "ttir" / "H_abs" / "abs_kernel.ttir").unlink()
    out = tmp_path / "c"
    assert run(file_gone, out, ["--ops", "abs", "--fail-on-missing"]) == 3


def test_accuracy_only_participation_ignores_skipped(tmp_path):
    exp = build(tmp_path, rounds=0, skipped={"main/other": "a:fail"})
    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs"]) == 0
    data = manifest_of(out)
    assert any(item["spec"] == "main:abs@abs" for item in data["selected"])


def test_overlapping_evidence_roots_are_rejected(tmp_path):
    exp = build(tmp_path, rounds=1)
    record = exp.record / "main" / "abs"
    nested_ab = record / "ab"
    shutil.copytree(exp.ab, nested_ab)
    csv_path = write(tmp_path / "report.csv", "x")
    xlsx = write(tmp_path / "report.xlsx", "x")

    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs"], record=record, ab_dir=nested_ab,
               csv_path=csv_path, xlsx=xlsx) == 1
    assert not out.exists()


def test_gate_without_variants_is_rejected(tmp_path):
    exp = build(tmp_path, gate=True)
    (exp.gate / "variants.tsv").unlink()
    assert run(exp, tmp_path / "archive", ["--ops", "abs"], gate=exp.gate) == 1


def test_gate_base_all_failed_is_missing(tmp_path):
    exp = build(tmp_path, gate=True)
    shutil.rmtree(exp.gate / "base")
    write(exp.gate / "errors.log",
          "######## base\n== /tmp/H_abs/abs_kernel.ttir\ntraceback\n")
    variants = (exp.gate / "variants.tsv").read_text().replace("\t1\t", "\t?\t")
    (exp.gate / "variants.tsv").write_text(variants)

    out = tmp_path / "archive"
    assert run(exp, out, ["--ops", "abs"], gate=exp.gate) == 0
    data = manifest_of(out)
    assert any("base" in item["item"] and "compilation failed" in item["reason"]
               for item in data["missing"])


def test_invalid_ab_json_is_rejected(tmp_path):
    exp = build(tmp_path)
    (exp.ab / ab_compare.MANIFEST).write_text('{"accuracy": {}}')
    assert run(exp, tmp_path / "archive", ["--ops", "abs"]) == 1


def test_empty_out_dir_is_accepted(tmp_path):
    exp = build(tmp_path)
    out = tmp_path / "archive"
    out.mkdir()
    assert run(exp, out, ["--ops", "abs"]) == 0
