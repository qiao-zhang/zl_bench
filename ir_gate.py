#!/usr/bin/env python3
"""Compile-diff gate: recompile the recorded TTIR with two compiler builds offline and
report which kernels, result rows and ops the change can affect.

Input is the ttir/<cache hash>/{<kernel>.ttir, <kernel>.json} that _ir_dump_plugin
writes into each run dir (only the variants the measurement actually used) and the
run's rows.tsv (which kernels each result row ran). No GPU is needed. A compiler
change usually touches few kernels, so pass this gate first and A/B only what it
selects, one to two orders of magnitude faster than a blind full run.

  # two builds (each value is a FlagTree python/ dir, used as PYTHONPATH)
  python ir_gate.py runs/batch_x --base /path/old/python --new /path/new/python
  # one build, two compile switches: --base-env / --new-env add env variables
  python ir_gate.py runs/batch_x --base $PWD/python --new $PWD/python \\
      --base-env FLAGTREE_XXX=0 --new-env FLAGTREE_XXX=1

Each variant is compiled down to the backend's binary stage on both sides (--stage
binary, the default: cubin on NVIDIA, hgbin on the PPU backend). A variant is
unchanged only if that binary and the launch metadata Triton derives (shared
memory, warps, ...) are identical: then the measured kernel is the same binary.
TTGIR alone is not enough: a reordering, or a change in a later stage, leaves the
TTGIR ops intact but changes the code the backend's assembler emits. The stage
chain is read from the backend itself (backend.add_stages), never hardcoded, so a
run recorded on PPU compares hgbin and a run recorded on NVIDIA compares cubin.
Line info is turned off on both sides (TRITON_DISABLE_LINE_INFO=1, unless an
--*-env sets it), so a moved source location alone does not count as a change; it
does not change the code.

--stage ttgir stops after make_ttgir (faster, for a first look); without --base the
TTGIR recorded at run time is the baseline, so only --stage ttgir is possible and
the batch must have been recorded with the baseline build.

Outputs in <out>/:
  variants.tsv     every variant: changed (1 / 0 / ? = a side failed to compile),
                   first stage that differs, TTGIR verdict (same / order / changed),
                   and the TTGIR ops whose count changed (e.g. "ttg.local_load:+2")
  report.tsv       the changed and unknown variants only
  rows.tsv         every result row of runs that have rows.tsv: changed if any kernel
                   it ran changed (input for ab_compare.py --gate)
  changed_ops.txt  ops with a changed or unknown row (or variant, for runs without
                   rows.tsv), for run_batch.py / ab_run.py --ops-file
  base/ new/       the recompiled ttgir per <cache hash>/<kernel>.ttgir, for diffing
  errors.log       compile failures
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from run_batch import parse_env

LOG = "[ir-gate]"
# NVIDIA's stage chain, used only for digests written before a run recorded its
# backend's own chain (and for --stage ttgir, where the recorded ttgir is the
# baseline). New digests carry "stages" and "binary"; see _compile_one.
STAGES = ("ttgir", "llir", "ptx", "cubin")
# Launch parameters Triton derives during compilation; a difference changes the launch
# even when the code is identical.
LAUNCH_META = ("shared", "num_warps", "num_ctas", "cluster_dims", "tmem_size", "global_scratch_size",
               "profile_scratch_size")


# ---------------- collection ----------------

class Variant(NamedTuple):
    ttir: Path                # ttir/<cache hash>/<kernel>.ttir, next to <kernel>.json
    label: str                # first op that used it (variants are shared): repo:op@op_file
    recorded: Optional[Path]  # ttgir compiled at record time (ttgir/<shape>/<kernel>/<file>)

    @property
    def key(self) -> Path:
        return Path(self.ttir.parent.name) / self.ttir.with_suffix(".ttgir").name


class RunDir(NamedTuple):
    path: Path
    label: str  # repo:op@op_file, the run_batch.py op spec (dir name for a single-op run)
    run: str    # <repo>/<run_name> inside a batch, "" for a single-op run dir


def run_dirs(roots: List[Path]) -> List[RunDir]:
    """Batch dirs expand to <repo>/<run_name>/; a single-op run dir is taken as is."""
    out = []
    for root in roots:
        if (root / "ttir").is_dir():
            out.append(RunDir(root, root.name, ""))
            continue
        spec = {}
        status = root / "ops_status.csv"
        if status.is_file():
            # changed_ops.txt must feed back into run_batch.py: label as repo:op@op_file
            for r in csv.DictReader(status.open(newline="")):
                spec[(r["repo"], r["run_name"])] = f"{r['repo']}:{r['op']}@{r['op_file']}"
        for d in sorted(p for p in root.glob("*/*") if (p / "ttir").is_dir()):
            key = (d.parent.name, d.name)
            out.append(RunDir(d, spec.get(key, f"{key[0]}:{key[1]}"), f"{key[0]}/{key[1]}"))
    return out


def run_hashes(run_dir: Path) -> Set[str]:
    """Cache hashes of the variants a run used (the ones it dumped)."""
    return {p.parent.name for p in (run_dir / "ttir").glob("*/*.ttir")}


def collect(roots: List[Path]) -> List[Variant]:
    """Every recompilable variant, deduplicated by cache hash (ops often share kernels)."""
    seen: Dict[str, Variant] = {}
    for rd in run_dirs(roots):
        recorded: Dict[str, Path] = {}
        index = rd.path / "ttgir" / "index.tsv"
        if index.is_file():
            for row in csv.DictReader(index.open(), delimiter="\t"):
                if int(row["launches"] or 0) > 0:
                    recorded.setdefault(row["cache_hash"], rd.path / "ttgir" / row["shape"]
                                        / row["kernel"] / row["file"])
        for ttir in sorted((rd.path / "ttir").glob("*/*.ttir")):
            h = ttir.parent.name
            if h not in seen and ttir.with_suffix(".json").is_file():
                seen[h] = Variant(ttir, rd.label, recorded.get(h))
    return list(seen.values())


def read_rows(run_dir: Path) -> Optional[List[Dict[str, str]]]:
    """The run's rows.tsv (written by _ir_dump_plugin), None if it has none."""
    path = run_dir / "rows.tsv"
    if not path.is_file():
        return None
    return list(csv.DictReader(path.open(newline=""), delimiter="\t"))


# ---------------- comparison helpers ----------------

_LOC_RE = re.compile(r" loc\((?:[^()]|\([^()]*\))*\)")
_SSA_RE = re.compile(r"%[\w#.:-]+")
# The op of an MLIR line: optional results ("%a, %b:2 = "), then the dialect.op name.
_OP_RE = re.compile(r"^\s*(?:%[\w#.:-]+(?:\s*,\s*%[\w#.:-]+)*\s*=\s*)?\"?([a-z_][\w]*\.[\w.]+)")


def _normalize(text: str, exact: bool) -> str:
    """TTGIR without loc info; with exact=False also without SSA names and line order,
    so only the multiset of ops/types/layouts is left (the "order" verdict)."""
    lines = [_LOC_RE.sub("", ln) for ln in text.splitlines() if not ln.startswith("#loc")]
    if exact:
        return "\n".join(lines)
    return "\n".join(sorted(_SSA_RE.sub("%", ln).strip() for ln in lines))


def _digest(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()[:16]


def op_counts(text: str) -> Dict[str, int]:
    """Number of each TTGIR op (dialect.op) in a module."""
    return dict(Counter(m.group(1) for m in map(_OP_RE.match, text.splitlines()) if m))


def op_diff(base: Optional[dict], new: Optional[dict]) -> str:
    """Every op whose count changed, largest change first: "ttg.local_load:+2 ..."."""
    if not base or not new:
        return ""
    a, b = base.get("ops", {}), new.get("ops", {})
    d = {op: b.get(op, 0) - a.get(op, 0) for op in set(a) | set(b)}
    d = sorted(((op, n) for op, n in d.items() if n), key=lambda x: (-abs(x[1]), x[0]))
    return " ".join(f"{op}:{n:+d}" for op, n in d)


def ttgir_digests(text: str) -> dict:
    return {"ttgir": _digest(_normalize(text, True)), "ttgir_norm": _digest(_normalize(text, False)),
            "ops": op_counts(text)}


def stage_chain(digest: Optional[dict]) -> List[str]:
    """Stages a digest recorded, in compile order; NVIDIA's chain for old digests."""
    return list((digest or {}).get("stages") or STAGES)


def binary_stage(digest: Optional[dict]) -> str:
    """The backend's final (binary) stage: cubin on NVIDIA, hgbin on PPU."""
    return (digest or {}).get("binary") or STAGES[-1]


def compare(base: Optional[dict], new: Optional[dict], stage: str) -> Tuple[str, str, str]:
    """(changed 1/0/?, first differing stage, TTGIR verdict) of one variant.

    stage is "binary" (the backend's final stage) or "ttgir". Digests written
    before the stage chain was recorded fall back to STAGES, so NVIDIA runs
    recorded by an older zl_bench compare exactly as they did."""
    if not base or not new:
        return "?", "error", "?"
    if base["ttgir"] == new["ttgir"]:
        ttgir = "same"
    elif base["ttgir_norm"] == new["ttgir_norm"]:
        ttgir = "order"
    else:
        ttgir = "changed"
    binary = binary_stage(new)
    target = binary if stage == "binary" else stage
    stages = stage_chain(new)
    if target in stages:
        stages = stages[:stages.index(target) + 1]
    first = next((s for s in stages if base.get(s) != new.get(s)), "")
    if not first and target == binary and base.get("meta") != new.get("meta"):
        first = "meta"
    return ("1" if first else "0"), first or "-", ttgir


# ---------------- compilation (in a subprocess; PYTHONPATH picks the compiler) ----------------

def _init_worker():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)  # MLIR diagnostics flood stderr; failures reach errors.log as tracebacks


def _compile_one(ttir_path: str, dest: str, stage: str) -> Optional[str]:
    """Compile one variant up to `stage`; write <dest>.ttgir and <dest>.digest.json."""
    try:
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import IRSource, make_backend

        ttir = Path(ttir_path)
        meta = json.loads(ttir.with_suffix(".json").read_text())
        t = meta.get("target") or {}
        target = GPUTarget(t.get("backend", "cuda"), t.get("arch", 90), t.get("warp_size", 32))
        backend = make_backend(target)
        fields = backend.parse_options({}).__dict__.keys()
        opts = {k: tuple(v) if isinstance(v, list) else v
                for k, v in meta.items() if k in fields}
        options = backend.parse_options(opts)
        context = ir.context()
        src = IRSource(str(ttir), context, backend)
        stages: dict = {}
        backend.add_stages(stages, options, src.language)
        # The backend decides its own chain: NVIDIA has ttir/ttgir/llir/ptx/cubin,
        # the PPU backend ttir/ttgir/llir/hgbin. Compile and compare up to the
        # backend's final stage, so a PPU run gates on hgbin instead of failing on
        # a missing "ptx" stage.
        names = list(stages)
        binary = names[-1] if names else ""
        stop = binary if stage == "binary" else stage
        if "ttgir" not in names or stop not in names:
            return (f"--stage {stop!r} is not in this backend's chain {names} "
                    f"(its binary stage is {binary!r})")
        if names.index(stop) < names.index("ttgir"):
            return f"--stage {stop!r} stops before ttgir; nothing to compare"
        chain = names[names.index("ttgir"):names.index(stop) + 1]
        module = backend.make_ir(src, options, backend.get_codegen_implementation(options),
                                 backend.get_module_map(), context)
        md = {"hash": meta.get("hash", "x"), "target": target, **options.__dict__}
        out: dict = {"stages": chain, "binary": binary}
        for name in chain:
            module = stages[name](module, md)
            if name == "ttgir":
                text = str(module)
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest + ".ttgir").write_text(text)
                out.update(ttgir_digests(text))
            else:
                out[name] = _digest(module)
        out["meta"] = {k: md[k] for k in LAUNCH_META if k in md}
        Path(dest + ".digest.json").write_text(json.dumps(out, default=str))
        return None
    except Exception:
        return traceback.format_exc(limit=4)


def _worker_main(list_file: str, jobs: int, stage: str) -> None:
    """Subprocess entry: each list_file line is "<ttir>\\t<dest prefix>"."""
    pairs = [ln.split("\t") for ln in Path(list_file).read_text().splitlines() if ln]
    errors = []
    with cf.ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        futs = {ex.submit(_compile_one, s, d, stage): s for s, d in pairs}
        for f in cf.as_completed(futs):
            err = f.result()
            if err:
                errors.append(f"== {futs[f]}\n{err}")
    print(json.dumps({"compiled": len(pairs) - len(errors), "errors": errors}))


def _dest(out: Path, side: str, v: Variant) -> Path:
    return out / side / v.ttir.parent.name / v.ttir.stem


def compile_side(name: str, items: List[Variant], out: Path, pythonpath: str,
                 extra_env: Dict[str, str], jobs: int, stage: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [pythonpath, env.get("PYTHONPATH")]))
    env.setdefault("TRITON_DISABLE_LINE_INFO", "1")
    env.update(extra_env)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for v in items:
            f.write(f"{v.ttir}\t{_dest(out, name, v)}\n")
        list_file = f.name
    proc = subprocess.run([sys.executable, __file__, "_worker", list_file, str(jobs), stage],
                          env=env, text=True, capture_output=True)
    os.unlink(list_file)
    if proc.returncode != 0:
        raise SystemExit(f"{LOG} {name} compile process failed:\n{proc.stderr[-2000:]}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    if res["errors"]:
        with (out / "errors.log").open("a") as fh:
            fh.write(f"######## {name}\n" + "\n".join(res["errors"]) + "\n")
    print(f"{LOG} {name}: compiled {res['compiled']}/{len(items)} "
          f"({len(res['errors'])} failed)", flush=True)
    return len(res["errors"])


def _load_digest(path: Path) -> Optional[dict]:
    p = path.with_name(path.name + ".digest.json")
    return json.loads(p.read_text()) if p.is_file() else None


# ---------------- main ----------------

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        return
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="batch dirs or single-op run dirs")
    p.add_argument("--new", required=True,
                   help="PYTHONPATH of the new compiler (a FlagTree python/ dir)")
    p.add_argument("--base", default=None,
                   help="PYTHONPATH of the baseline compiler; without it the recorded ttgir is "
                        "the baseline (--stage ttgir only)")
    p.add_argument("--new-env", action="append", default=[], metavar="K=V")
    p.add_argument("--base-env", action="append", default=[], metavar="K=V")
    p.add_argument("--stage", default="binary",
                   help="last stage to compile and compare (default %(default)s): 'binary' is "
                        "the backend's final stage (cubin on NVIDIA, hgbin on the PPU backend), "
                        "'ttgir' stops after make_ttgir, or name a stage of the backend")
    p.add_argument("--out", type=Path, default=None, help="output dir (default runs/ir_gate_<timestamp>)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 16)
    args = p.parse_args()

    stage = args.stage
    if not args.base and stage != "ttgir":
        print(f"{LOG} no --base: comparing against the recorded ttgir (--stage ttgir); "
              "changes below TTGIR are not visible", flush=True)
        stage = "ttgir"
    runs = run_dirs(args.runs)
    items = collect(args.runs)
    if not items:
        raise SystemExit(f"{LOG} nothing to recompile: expected ttir/<cache hash>/ in the run dirs")
    out = (args.out or Path(__file__).resolve().parent / "runs" /
           f"ir_gate_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "errors.log").unlink(missing_ok=True)
    print(f"{LOG} {len(items)} variants of {len(runs)} runs -> {out} (stage {stage})", flush=True)

    compile_side("new", items, out, args.new, parse_env(args.new_env), args.jobs, stage)
    if args.base:
        compile_side("base", items, out, args.base, parse_env(args.base_env), args.jobs, stage)

    # ---- per variant
    status: Dict[str, Tuple[str, str, str]] = {}
    lines = []
    binary_names: Set[str] = set()
    for v in items:
        new = _load_digest(_dest(out, "new", v))
        if args.base:
            base = _load_digest(_dest(out, "base", v))
        else:
            base = ttgir_digests(v.recorded.read_text()) if v.recorded and v.recorded.is_file() else None
        if new:
            binary_names.add(binary_stage(new))
        changed, first, ttgir = compare(base, new, stage)
        h = v.ttir.parent.name
        status[h] = (changed, first, ttgir)
        lines.append([v.label, h, v.ttir.stem, changed, first, ttgir, op_diff(base, new)])
    head = "op\tcache_hash\tkernel\tchanged\tfirst_diff\tttgir\top_count_diff\n"
    lines.sort()
    (out / "variants.tsv").write_text(head + "".join("\t".join(r) + "\n" for r in lines))
    (out / "report.tsv").write_text(head + "".join("\t".join(r) + "\n" for r in lines if r[3] != "0"))

    # ---- per row and per op
    changed_ops, row_lines = [], []
    row_counts: Counter = Counter()
    no_rows = 0
    for rd in runs:
        rows = read_rows(rd.path)
        if rows is None:
            no_rows += 1
            flags = {status.get(h, ("?",))[0] for h in run_hashes(rd.path)}
            if flags & {"1", "?"}:
                changed_ops.append(rd.label)
            continue
        op_changed = False
        for r in rows:
            ks = [k for k in r["kernels"].split(",") if k] if r["kernels"] != "?" else None
            if ks is None:
                flag, which = "?", ""
            else:
                flags = {k: status.get(k, ("?",))[0] for k in ks}
                which = ",".join(k for k, f in flags.items() if f != "0")
                flag = "1" if "1" in flags.values() else "?" if "?" in flags.values() else "0"
            row_counts[flag] += 1
            op_changed |= flag != "0"
            row_lines.append([rd.run, rd.label, r["op_name"], r["dtype"], r["row"], r["shape"],
                              flag, which])
        if op_changed:
            changed_ops.append(rd.label)
    (out / "rows.tsv").write_text(
        "run\top\top_name\tdtype\trow\tshape\tchanged\tchanged_kernels\n"
        + "".join("\t".join(r) + "\n" for r in row_lines))
    (out / "changed_ops.txt").write_text("".join(f"{label}\n" for label in sorted(set(changed_ops))))

    # ---- summary
    flags = Counter(s[0] for s in status.values())
    firsts = Counter(s[1] for s in status.values() if s[0] == "1")
    hidden = sum(1 for s in status.values() if s[0] == "1" and s[2] == "same")
    order = sum(1 for s in status.values() if s[0] == "1" and s[2] == "order")
    print(f"{LOG} variants: unchanged {flags['0']}, changed {flags['1']}, unknown {flags['?']} "
          f"(compile failure, see errors.log)")
    if flags["1"]:
        print(f"{LOG}   first difference at: "
              + ", ".join(f"{k} {n}" for k, n in sorted(firsts.items(), key=lambda kv: -kv[1])))
        if stage == "binary" or stage in binary_names:
            print(f"{LOG}   changed with identical TTGIR {hidden}, with only reordered TTGIR {order}")
    if row_lines:
        print(f"{LOG} rows: changed {row_counts['1']}, unchanged {row_counts['0']}, "
              f"unknown {row_counts['?']} (of {len(row_lines)})")
    if no_rows:
        print(f"{LOG} {no_rows} runs have no rows.tsv (recorded before row tracking): judged per op")
    print(f"{LOG} {len(set(changed_ops))} ops to A/B: {out / 'changed_ops.txt'}")


if __name__ == "__main__":
    main()
