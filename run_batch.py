#!/usr/bin/env python3
"""Batch driver: run run_pytest.sh once per op from the inventory, sharded across GPUs.

Each op runs in its own subprocess (a CUDA crash only loses that op) with its
own output dir <batch>/<repo>/<op>/, laid out exactly like a single-op
run_pytest.sh run (run.log / shapes.yaml / autotune_records / ttgir / ttir).
This layer only orchestrates:

- sharding: ops are hashed to GPUs by name; each GPU runs its ops serially,
  timed exactly like a single-op run. Parallel GPUs couple through power and
  thermals, so treat batch numbers as a screen and confirm suspects with
  single-op runs;
- timeout: --op-timeout kills a hung op (TIMEOUT, no retry, batch goes on);
- retries: pytest rc=1 (failure/flaky) and signal kills (rc>=128) are retried
  up to twice, stopping early when the same rc repeats. Under --dtypes there
  are two fallbacks: upstream's "can't be supported by this op" reruns without
  the filter (dtype_fallback), and any other failure with no SUCCESS row gets
  one last run without it (dtype_rescue);
- resume: --resume reuses ops with a .complete marker (written for PASS/SKIP
  only, so FAIL/TIMEOUT rerun);
- replay: --replay-root points at an earlier batch dir and passes
  REPLAY_FROM=<root>/<repo>/<op> per op, same semantics as single-op replay;
- idle GPUs: before each op starts, the accelerator management tool
  (nvidia-smi on NVIDIA, ppu-smi on the T-Head PPU, mx-smi on MetaX) must show
  no other process on its GPU (--busy-gpu wait, the default; warn or ignore to
  relax it), since a neighbour on the same GPU skews both the timing and the
  autotune choices. GPU numbers follow that tool (CUDA_DEVICE_ORDER=PCI_BUS_ID).
  Without any of them the batch assumes one device, so list the cards explicitly
  with --gpus.

Outputs: <batch>/ops_status.csv (one row per op) and summary.csv (one row per
result-table row, written by batch_summary.py).

Examples:
  # Full screen (every collectable op of both repos, minus marked_skip and dismiss)
  python run_batch.py --all --gpus 0,1,2,3,4,5,6,7 --dtypes bfloat16

  # A subset (repo prefix defaults to main; @op_file disambiguates duplicates)
  python run_batch.py --ops softmax vllm:fused_marlin_moe_w4a16_mxfp4 nextafter_@nextafter_

  # Re-measure with pinned configs (same semantics as single-op REPLAY_FROM)
  python run_batch.py --ops-file my_ops.txt --replay-root runs/batch_xxx
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import batch_summary
from gen_inventory import OPS_DIR, REPOS, ZL_BENCH

DEFAULT_SHAPE_FILE = ZL_BENCH / "shape" / "all_ops_single.yaml"
LOG = "[batch]"
MAX_RETRIES = 2
BUSY_POLL_S = 30
BUSY_REPORT_S = 300
# Accelerator management tool, the source of GPU numbers and the idle check:
# nvidia-smi on NVIDIA, ppu-smi on the T-Head PPU, mx-smi on MetaX (each is the
# name the matching FlagGems vendor reports as device_query_cmd). ZL_BENCH_SMI
# overrides the choice, for a box where several are installed or the tool has
# another name.
SMI_ENV = "ZL_BENCH_SMI"
SMI_TOOLS = ("nvidia-smi", "ppu-smi", "mx-smi")
# The tool that ships with each vendor stack (see accelerator()). A box often
# carries an nvidia-smi that cannot enumerate the devices of another vendor, so
# the stack is the stronger signal than the tool order.
VENDOR_SMI = {"ppu": "ppu-smi", "metax": "mx-smi"}
# The tools do not share their query spelling: nvidia-smi takes
# --list-gpus/--query-gpu, ppu-smi -L/--query-ppu, mx-smi follows the MACA
# release. Try the narrower spellings first, then the NVIDIA one; a tool
# accepts one and rejects the others, so the first non-empty answer wins.
SMI_LIST_ARGS = (["-L"], ["--list-gpus"], ["--list-ppus"])
SMI_GPU_ARGS = (
    ["--query-ppu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
    ["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
)
SMI_APPS_ARGS = (
    ["--query-compute-apps=ppu_uuid,pid,process_name,used_memory",
     "--format=csv,noheader,nounits"],
    ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
     "--format=csv,noheader,nounits"],
)
_LIST_GPU_RE = re.compile(r"^\s*(?:GPU|PPU)\s*#?\s*\d+\s*:")
# mx-smi's summary block states the device count in its `Attached GPUs: N` line
# (FlagGems' gpu_check_metax.sh reads the same field); used when no -L spelling
# works.
_ATTACHED_RE = re.compile(r"^\s*Attached\b[^:\n]*:\s*(\d+)\s*$", re.MULTILINE)
# Device-visibility variables, the stack-specific one first: the MACA runtime
# (MetaX) reads MACA_VISIBLE_DEVICES, NVIDIA and the PPU compatibility layer
# read CUDA_VISIBLE_DEVICES.
VISIBLE_ENVS = ("MACA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
# Memory in use that marks a GPU busy even when the management tool lists no
# process on it (processes of other containers or PID namespaces are not
# listed). An idle GPU shows a few MiB.
BUSY_MEMORY_MIB = 1024

STATUS_FIELDS = [
    "repo", "op", "op_file", "run_name", "gpu", "status", "return_code",
    "elapsed_ms", "rows_total", "rows_success", "notes", "out_dir",
]

DTYPE_UNSUPPORTED_MARK = "can't be supported by this op"


# ---------------- inventory and selection ----------------

class Case(NamedTuple):
    repo: str
    op: str
    op_file: str
    run_name: str  # output dir name under <batch>/<repo>/

    @property
    def key(self) -> str:
        return f"{self.repo}:{self.op}@{self.op_file}"


def load_inventory() -> Dict[str, List[Dict[str, str]]]:
    inv: Dict[str, List[Dict[str, str]]] = {}
    for label in REPOS:
        path = OPS_DIR / f"inventory_{label}.csv"
        if path.is_file():
            with path.open(newline="") as f:
                inv[label] = list(csv.DictReader(f))
    if not inv:
        raise SystemExit(f"{LOG} no ops/inventory_*.csv; run gen_inventory.py first")
    return inv


def _run_name(row: Dict[str, str], dup_ops: Set[Tuple[str, str]]) -> str:
    # The dir name is the op, plus @op_file when the op name repeats within the
    # repo. Duplicates are judged on the whole inventory, not the selection, so
    # resume/replay paths stay stable. op_file may contain a subdir (test_FLA/),
    # so "/" becomes "_".
    op = row["op"]
    if (row["repo"], op) not in dup_ops:
        return op
    return f"{op}@{row['op_file'].replace('/', '_')}"


def _dup_ops(inv: Dict[str, List[Dict[str, str]]]) -> Set[Tuple[str, str]]:
    dup = set()
    for label, rows in inv.items():
        counts = Counter(r["op"] for r in rows)
        dup |= {(label, op) for op, n in counts.items() if n > 1}
    return dup


def load_dismiss(path: Path) -> Set[Tuple[str, str]]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo, _, op = line.rpartition(":")
        out.add((repo or "main", op))
    return out


def resolve_selection(args: argparse.Namespace,
                      inv: Dict[str, List[Dict[str, str]]]) -> List[Case]:
    dup = _dup_ops(inv)
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    by_op: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for label, rows in inv.items():
        for r in rows:
            index[(label, r["op"], r["op_file"])] = r
            by_op.setdefault((label, r["op"]), []).append(r)

    dismissed = load_dismiss(args.dismiss)

    def make_case(row: Dict[str, str]) -> Case:
        return Case(row["repo"], row["op"], row["op_file"], _run_name(row, dup))

    if args.all:
        cases = []
        for label, rows in inv.items():
            if args.repos and label not in args.repos:
                continue
            for r in rows:
                if r.get("collected") == "no":
                    continue
                if r.get("marked_skip") == "yes" and not args.include_marked_skip:
                    continue
                if (label, r["op"]) in dismissed:
                    continue
                cases.append(make_case(r))
        return cases

    specs: List[str] = list(args.ops or [])
    if args.ops_file:
        for line in args.ops_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                specs.append(line)
    if not specs:
        raise SystemExit(f"{LOG} no ops selected: use --all, --ops or --ops-file")

    cases, errors = [], []
    for spec in specs:
        repo, _, rest = spec.rpartition(":")
        repo = repo or "main"
        op, _, op_file = rest.partition("@")
        if repo not in inv:
            errors.append(f"{spec}: unknown repo {repo!r}")
            continue
        if op_file:
            row = index.get((repo, op, op_file))
            if row is None:
                errors.append(f"{spec}: no such inventory entry")
                continue
            cases.append(make_case(row))
            continue
        rows = by_op.get((repo, op)) or []
        if not rows:
            errors.append(f"{spec}: no such op in the inventory")
        elif len(rows) > 1:
            files = ", ".join(r["op_file"] for r in rows)
            errors.append(f"{spec}: ambiguous, add @op_file (candidates: {files})")
        else:
            cases.append(make_case(rows[0]))
    if errors:
        for e in errors:
            print(f"{LOG} error: {e}", file=sys.stderr)
        raise SystemExit(1)
    return cases


def add_selection_args(p: argparse.ArgumentParser) -> None:
    """Op selection flags, shared with ab_run.py."""
    sel = p.add_argument_group("op selection")
    sel.add_argument("--all", action="store_true",
                     help="the whole inventory (minus collected=no / marked_skip / dismiss)")
    sel.add_argument("--ops", nargs="*", default=None, metavar="[repo:]op[@op_file]",
                     help="ops to run (repo defaults to main; @op_file disambiguates duplicates)")
    sel.add_argument("--ops-file", type=Path, default=None,
                     help="file with one [repo:]op[@op_file] per line, # comments")
    sel.add_argument("--repos", nargs="*", default=None, choices=list(REPOS),
                     help="restrict --all to these repos (default: all)")
    sel.add_argument("--dismiss", type=Path, default=OPS_DIR / "dismiss.txt",
                     help="exclusion list (default ops/dismiss.txt; --all only)")
    sel.add_argument("--include-marked-skip", action="store_true",
                     help="keep ops with an unconditional skip mark under --all")


def parse_env(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    """K=V strings -> dict."""
    env = {}
    for kv in pairs or ():
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise SystemExit(f"expected K=V, got {kv!r}")
        env[k] = v
    return env


# ---------------- GPUs ----------------

def op_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment an op runs in: this process's, plus the batch's --env pairs.

    Device enumeration, the idle check and the subprocess all read this one view,
    so a vendor stack passed with `--env MACA_PATH=...` selects the same tool the
    op itself runs under. ab_run.py passes each side's env, so A and B are checked
    with their own stack."""
    env = dict(os.environ)
    env.update(extra or {})
    return env


def accelerator(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The vendor stack in use: metax / ppu, or None when no stack marker is set
    (a plain NVIDIA box, or a caller that only sets ZL_BENCH_SMI). `env` defaults
    to this process's environment."""
    env = os.environ if env is None else env
    if env.get("MACA_PATH") or env.get("MACA_VISIBLE_DEVICES"):
        return "metax"
    if env.get("PPU_SDK"):
        return "ppu"
    return None


def smi_tool(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The accelerator management tool in use, or None when none is usable.

    nvidia-smi on NVIDIA, ppu-smi on the T-Head PPU, mx-smi on MetaX;
    ZL_BENCH_SMI overrides. Once the vendor stack is known, only its own tool is
    acceptable: another vendor's tool would inspect the wrong devices, so a
    missing tool means None (with a warning from the caller) instead of a
    fallback. A box without a stack marker keeps the plain tool order."""
    env = os.environ if env is None else env
    explicit = env.get(SMI_ENV, "").strip()
    if explicit:
        return explicit
    vendor = accelerator(env)
    expected = VENDOR_SMI.get(vendor or "")
    if expected:
        return expected if shutil.which(expected, path=env.get("PATH")) else None
    return next((tool for tool in SMI_TOOLS if shutil.which(tool, path=env.get("PATH"))), None)


def visible_gpus(arg: Optional[str],
                 env: Optional[Dict[str, str]] = None) -> List[str]:
    if arg:
        return [g.strip() for g in arg.split(",") if g.strip()]
    env = os.environ if env is None else env
    for key in VISIBLE_ENVS:
        value = env.get(key, "").strip()
        if value:
            return [g.strip() for g in value.split(",") if g.strip()]
    # -L lists devices on nvidia-smi/ppu-smi; mx-smi answers in its own spelling
    # per MACA release, so fall back to its summary block.
    n = _device_count(_smi_first(SMI_LIST_ARGS, env=env) or _smi([], env=env) or "")
    return [str(i) for i in range(n)] if n else ["0"]


def _device_count(text: str) -> int:
    """Devices in a management tool's listing (`GPU 0: ...` on nvidia-smi,
    `PPU 0: ...` on ppu-smi, `GPU#0: ...` on some MACA releases), else in
    mx-smi's `Attached GPUs: N` summary line."""
    lines = [line for line in text.splitlines() if line.strip()]
    n = sum(1 for line in lines if _LIST_GPU_RE.match(line))
    if n:
        return n
    attached = _ATTACHED_RE.search(text)
    return int(attached.group(1)) if attached else 0


def _smi(args: List[str], env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """stdout of `<tool> --args`, or None when there is no tool or it rejects the
    spelling (non-zero exit). "" means the query succeeded with no rows: an idle
    accelerator lists no compute process, which is not "the tool cannot tell".

    The tool runs under the same environment that selected it: a tool reached
    through `--env PATH=...` must also find its libraries (`LD_LIBRARY_PATH`)
    there, and a bare name must resolve against that PATH, not this process's."""
    env = op_env(env)
    tool = smi_tool(env)
    if tool is None:
        return None
    try:
        return subprocess.run([tool, *args], text=True, capture_output=True,
                              timeout=30, check=True, env=env).stdout
    except Exception:
        return None


def _smi_first(arg_sets: Sequence[List[str]], csv_query: bool = False,
               env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """First spelling the tool accepts, an empty answer included.

    `--query-compute-apps` prints nothing when no process uses the GPU, and
    treating that as a rejected spelling would discard the memory-based busy
    check. `csv_query` still rejects a usage block that the tool printed with
    exit 0: it has no CSV fields, so it is not an answer to a `--query-*`."""
    for args in arg_sets:
        out = _smi(args, env)
        if out is None:
            continue
        if csv_query and not _looks_like_csv(out):
            continue
        return out
    return None


def _looks_like_csv(text: str) -> bool:
    """CSV rows, or nothing at all for an empty result set."""
    rows = [line for line in text.splitlines() if line.strip()]
    return not rows or all("," in line for line in rows)


def parse_gpus(text: str) -> Dict[str, Tuple[str, int]]:
    """`<tool> --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits`
    (nvidia-smi, or ppu-smi/mx-smi which mirror its options) -> {index: (uuid,
    MiB in use)}, also keyed by uuid."""
    out = {}
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[1]:
            used = int(parts[2]) if parts[2].isdigit() else 0
            out[parts[0]] = out[parts[1]] = (parts[1], used)
    return out


def parse_compute_apps(text: str, uuid: str) -> List[str]:
    """Processes of `<tool> --query-compute-apps=gpu_uuid,pid,process_name,used_memory
    --format=csv,noheader,nounits` that run on the GPU with this uuid."""
    out = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            out.append(f"pid {parts[1]} {parts[2]} ({parts[3]} MiB)")
    return out


def gpu_users(gpu: str, env: Optional[Dict[str, str]] = None) -> Optional[List[str]]:
    """What is using gpu (a management-tool index or a GPU uuid): its processes, or
    the memory in use when no process is listed; empty when idle, None if the tool
    cannot tell."""
    gpus = _smi_first(SMI_GPU_ARGS, csv_query=True, env=env)
    apps = _smi_first(SMI_APPS_ARGS, csv_query=True, env=env)
    info = parse_gpus(gpus).get(gpu) if gpus else None
    if info is None or apps is None:
        return None
    uuid, used = info
    procs = parse_compute_apps(apps, uuid)
    if not procs and used >= BUSY_MEMORY_MIB:
        procs = [f"{used} MiB in use by processes not visible here"]
    return procs


_smi_missing_reported = threading.Event()


def _no_smi_reason(env: Optional[Dict[str, str]] = None) -> str:
    """Why the device query is impossible, for the one warning the batch prints."""
    tool = smi_tool(env)
    if tool is not None:
        return (f"{tool} cannot list GPU processes "
                "(tried -L/--query-ppu/--query-gpu and their compute-apps forms)")
    vendor = accelerator(env)
    expected = VENDOR_SMI.get(vendor or "")
    if expected:
        # The stack decided the vendor, so its own tool or none: another vendor's
        # tool would inspect the wrong devices.
        return (f"the {vendor} stack is in use but {expected} is not installed "
                f"(install it or set {SMI_ENV})")
    return (f"no accelerator management tool (install nvidia-smi, ppu-smi or mx-smi, "
            f"or set {SMI_ENV})")


def wait_for_idle_gpu(gpu: str, policy: str,
                      env: Optional[Dict[str, str]] = None) -> str:
    """Hold until no other process uses gpu (policy "wait"), or only report it ("warn").
    Returns a note for ops_status.csv: "", gpu_busy or gpu_wait=<seconds>s.

    `env` is the op's environment (see op_env), so the tool asked here is the one
    the op itself will run under."""
    if policy == "ignore":
        return ""
    procs = gpu_users(gpu, env)
    if procs is None:
        if not _smi_missing_reported.is_set():
            _smi_missing_reported.set()
            reason = _no_smi_reason(env)
            print(f"{LOG} warning: {reason}; not checking that GPUs are idle", flush=True)
        return ""
    if not procs:
        return ""
    if policy == "warn":
        print(f"{LOG} warning: gpu{gpu} is in use by {'; '.join(procs)}; measuring anyway",
              flush=True)
        return "gpu_busy"
    t0 = last = time.time()
    print(f"{LOG} gpu{gpu} is in use by {'; '.join(procs)}; waiting for it to be idle", flush=True)
    while procs:
        time.sleep(BUSY_POLL_S)
        procs = gpu_users(gpu, env) or []
        if procs and time.time() - last >= BUSY_REPORT_S:
            last = time.time()
            print(f"{LOG} gpu{gpu} still in use ({int(last - t0)} s): {'; '.join(procs)}", flush=True)
    return f"gpu_wait={int(time.time() - t0)}s"


def partition(cases: List[Case], gpus: List[str]) -> Dict[str, List[Case]]:
    """Stable hash of (repo, run_name): independent of selection order, so a
    resumed op lands on the same GPU."""
    parts: Dict[str, List[Case]] = {g: [] for g in gpus}
    for c in cases:
        digest = hashlib.sha256(f"{c.repo}/{c.run_name}".encode()).digest()
        parts[gpus[int.from_bytes(digest[:8], "big") % len(gpus)]].append(c)
    return parts


# ---------------- per-op execution ----------------

def _spawn(case: Case, out_dir: Path, gpu: str, args: argparse.Namespace,
           dtypes: str) -> subprocess.Popen:
    # One view of the environment for the vendor decision here and the idle check
    # in run_case: the caller's shell plus the batch's --env pairs.
    env = op_env(getattr(args, "env", None))
    # The batch arguments win: same-named variables from the caller's shell must
    # not leak into run_pytest.sh. PARALLEL_WARMUP_GPUS: the batch already
    # shards across GPUs, so no per-op parallelism.
    for key in ("PARALLEL_WARMUP_GPUS", "REPLAY_FROM", "SHAPE_FILE", "DTYPES"):
        env.pop(key, None)
    # GPU numbers as the management tool shows them, which the idle check relies on.
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    pythonpath = getattr(args, "pythonpath", None)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [pythonpath, env.get("PYTHONPATH")]))
    env.update({
        "OP": case.op,
        "OP_FILE": case.op_file,
        "FLAGGEMS_DIR": str(REPOS[case.repo]),
        "OUT_DIR": str(out_dir),
        "CUDA_VISIBLE_DEVICES": gpu,
        "FLAGGEMS_PERF_COLOR": "never",
        "NO_COLOR": "1",
        "USE_FLAGTUNE": str(args.use_flagtune),
    })
    # The MACA runtime (MetaX) reads its own visibility variable, and one
    # inherited from the caller's shell or passed by --env would override this
    # op's sharding.
    if accelerator(env) == "metax":
        env["MACA_VISIBLE_DEVICES"] = gpu
    if args.shape_file:  # unset under replay: run_pytest.sh reuses the A side's archive
        env["SHAPE_FILE"] = str(args.shape_file)
    if dtypes:
        env["DTYPES"] = dtypes
    replay = replay_dir(case, args)
    if replay:
        env["REPLAY_FROM"] = str(replay)
    return subprocess.Popen(
        ["bash", str(ZL_BENCH / "run_pytest.sh")],
        cwd=str(ZL_BENCH), env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def replay_dir(case: Case, args: argparse.Namespace) -> Optional[Path]:
    """The recorded run an op replays: <replay_root>/<repo>/<run_name>, or a
    single-op record given as replay_root itself (it has a run.log)."""
    root = getattr(args, "replay_root", None)
    if not root:
        return None
    return root if (root / "run.log").is_file() else root / case.repo / case.run_name


def _wait(proc: subprocess.Popen, timeout: int) -> Tuple[int, bool]:
    """(rc, timed_out); on timeout the whole process group gets SIGTERM, then SIGKILL."""
    try:
        return proc.wait(timeout=timeout or None), False
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # the group exited right at the deadline
        proc.wait()
        return 124, True


def run_case(case: Case, batch_dir: Path, gpu: str,
             args: argparse.Namespace) -> Dict[str, str]:
    out_dir = batch_dir / case.repo / case.run_name
    marker = out_dir / ".complete"
    # The same environment _spawn hands the op, so the idle check asks the tool
    # the op will actually run under (ab_run.py passes each side's env).
    env = op_env(getattr(args, "env", None))

    if args.resume and marker.is_file() and (out_dir / "run.log").is_file():
        row = _classify(case, out_dir, gpu, 0, 0, "", timed_out=False)
        row["notes"] = _join("resumed", row["notes"])
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)

    dtypes = args.dtypes or ""
    t0 = time.time()
    rcs: List[int] = []
    notes = ""
    timed_out = False
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            # keep the previous attempt's log
            log = out_dir / "run.log"
            if log.is_file():
                log.rename(out_dir / f"run.attempt{attempt - 1}.log")
        notes = _join(notes, wait_for_idle_gpu(gpu, getattr(args, "busy_gpu", "ignore"), env))
        rc, timed_out = _wait(_spawn(case, out_dir, gpu, args, dtypes),
                              args.op_timeout)
        rcs.append(rc)
        if timed_out:
            break
        log_text = _read_log(out_dir)
        # Unsupported dtype: rerun without --dtypes (once, outside the retry budget).
        if rc != 0 and dtypes and DTYPE_UNSUPPORTED_MARK in log_text:
            dtypes = ""
            notes = _join(notes, "dtype_fallback")
            continue
        if rc == 0 or not (rc == 1 or rc >= 128):
            break
        if len(rcs) >= 2 and rcs[-1] == rcs[-2]:
            break  # deterministic: same rc twice in a row

    # dtype rescue: the filtered run failed without a single SUCCESS row, so try
    # once more without the filter. Covers baselines that fail to compile for
    # the filtered dtype (torch jiterator special functions etc.) with an error
    # other than upstream's "can't be supported by this op".
    if dtypes and not timed_out:
        probe = _classify(case, out_dir, gpu, rcs[-1], 0, "", False)
        if probe["status"] == "FAIL" and probe["rows_success"] == "0":
            log = out_dir / "run.log"
            if log.is_file():
                log.rename(out_dir / f"run.attempt{len(rcs) - 1}.log")
            notes = _join(notes, "dtype_rescue",
                          wait_for_idle_gpu(gpu, getattr(args, "busy_gpu", "ignore"), env))
            rc, timed_out = _wait(_spawn(case, out_dir, gpu, args, ""),
                                  args.op_timeout)
            rcs.append(rc)
    elapsed_ms = int((time.time() - t0) * 1000)

    if len(rcs) > 1:
        notes = _join(notes, "retry_rc=" + "->".join(map(str, rcs)))
    row = _classify(case, out_dir, gpu, rcs[-1], elapsed_ms, notes, timed_out)
    if row["status"] in ("PASS", "SKIP"):
        tmp = marker.with_name(".complete.tmp")
        tmp.write_text(row["status"] + "\n")
        os.replace(tmp, marker)
    return row


def _read_log(out_dir: Path) -> str:
    log = out_dir / "run.log"
    return log.read_text(errors="replace") if log.is_file() else ""


def _join(*notes: str) -> str:
    return ";".join(n for n in notes if n)


def _classify(case: Case, out_dir: Path, gpu: str, rc: int, elapsed_ms: int,
              notes: str, timed_out: bool) -> Dict[str, str]:
    text = _read_log(out_dir)
    rows = batch_summary.parse_run_log(text)
    if timed_out:
        status = "TIMEOUT"
    else:
        status = batch_summary.classify(rc, rows, text)
    n_skipped = batch_summary.pytest_skipped_count(text)
    if status == "PASS" and n_skipped:
        notes = _join(notes, f"partial_skip={n_skipped}")
    return {
        "repo": case.repo,
        "op": case.op,
        "op_file": case.op_file,
        "run_name": case.run_name,
        "gpu": gpu,
        "status": status,
        "return_code": str(rc),
        "elapsed_ms": str(elapsed_ms),
        "rows_total": str(len(rows)),
        "rows_success": str(sum(1 for r in rows if r["row_status"] == "SUCCESS")),
        "notes": _join(notes, batch_summary.marker_notes(text)),
        "out_dir": str(out_dir),
    }


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(p)

    run = p.add_argument_group("execution")
    run.add_argument("--gpus", default=None,
                     help="comma-separated GPU list (default: MACA_VISIBLE_DEVICES / "
                          "CUDA_VISIBLE_DEVICES, else every device the SMI tool lists)")
    run.add_argument("--batch-dir", type=Path, default=None,
                     help="output dir (default runs/batch_<timestamp>; required with --resume)")
    run.add_argument("--op-timeout", type=int, default=1800,
                     help="per-op timeout in seconds, 0 = none (default %(default)s)")
    run.add_argument("--resume", action="store_true",
                     help="reuse ops in --batch-dir that carry a .complete marker")
    run.add_argument("--replay-root", type=Path, default=None,
                     help="earlier batch dir; each op replays <root>/<repo>/<op>")
    run.add_argument("--shape-file", type=Path, default=None,
                     help=f"shape set (default {DEFAULT_SHAPE_FILE.relative_to(ZL_BENCH)}; "
                          "shape/all_ops_multi.yaml for a deep run). With --replay-root "
                          "the default is each op's archived shapes.yaml from the A side")
    run.add_argument("--dtypes", default="",
                     help='dtype whitelist, space separated (e.g. "bfloat16"); empty = upstream sweep')
    run.add_argument("--use-flagtune", type=int, default=0, choices=(0, 1),
                     help="passed through as USE_FLAGTUNE (default 0; must match on both A/B sides)")
    run.add_argument("--env", action="append", default=[], metavar="K=V",
                     help="extra environment variable for every op, e.g. a compiler switch "
                          "(repeatable)")
    run.add_argument("--busy-gpu", choices=("wait", "warn", "ignore"), default="wait",
                     help="when another process uses an op's GPU: wait until it is idle "
                          "(default), warn and run, or ignore")
    args = p.parse_args()
    args.env = parse_env(args.env)

    if args.resume and not args.batch_dir:
        raise SystemExit(f"{LOG} --resume needs an explicit --batch-dir")
    if args.shape_file is None and not args.replay_root:
        args.shape_file = DEFAULT_SHAPE_FILE
    if args.shape_file:
        args.shape_file = args.shape_file.resolve()
        if not args.shape_file.is_file():
            raise SystemExit(f"{LOG} no shape file {args.shape_file}")
    if args.replay_root:
        args.replay_root = args.replay_root.resolve()
        if not args.replay_root.is_dir():
            raise SystemExit(f"{LOG} no replay dir {args.replay_root}")
    return args


def main() -> None:
    args = parse_args()
    inv = load_inventory()
    cases = resolve_selection(args, inv)
    if not cases:
        raise SystemExit(f"{LOG} the selection is empty")

    gpus = visible_gpus(args.gpus, op_env(args.env))
    batch_dir = (args.batch_dir or
                 ZL_BENCH / "runs" / f"batch_{datetime.now():%Y%m%d-%H%M%S}")
    batch_dir = batch_dir.resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)

    parts = {g: cs for g, cs in partition(cases, gpus).items() if cs}
    print(f"{LOG} {len(cases)} ops -> {len(parts)} GPUs "
          f"(shards: {[len(cs) for cs in parts.values()]})  out={batch_dir}")
    if args.replay_root:
        print(f"{LOG} replay mode: REPLAY_FROM={args.replay_root}/<repo>/<op>")

    lock = threading.Lock()
    results: Dict[str, Dict[str, str]] = {}
    status_csv = batch_dir / "ops_status.csv"
    wname = max((len(c.run_name) for c in cases), default=8)

    def flush_status() -> None:
        rows = [results[c.key] for c in cases if c.key in results]
        tmp = status_csv.with_name(status_csv.name + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, status_csv)

    def run_partition(gpu: str, part: List[Case]) -> None:
        for case in part:
            try:
                row = run_case(case, batch_dir, gpu, args)
            except Exception as exc:  # a harness error must not take the shard down
                row = {
                    "repo": case.repo, "op": case.op, "op_file": case.op_file,
                    "run_name": case.run_name, "gpu": gpu, "status": "FAIL",
                    "return_code": "EXC", "elapsed_ms": "0", "rows_total": "0",
                    "rows_success": "0", "notes": f"harness_exc={exc}",
                    "out_dir": str(batch_dir / case.repo / case.run_name),
                }
            with lock:
                results[case.key] = row
                flush_status()
                print(f"  ({len(results):>4}/{len(cases)})  {row['status']:<7} "
                      f"{row['elapsed_ms']:>8}ms  rows={row['rows_success']}/"
                      f"{row['rows_total']:<4} {case.repo}:{case.run_name:<{wname}} "
                      f"[gpu{gpu}]"
                      + (f"  {row['notes']}" if row["notes"] else ""),
                      flush=True)

    if len(parts) == 1:
        gpu, part = next(iter(parts.items()))
        run_partition(gpu, part)
    else:
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            futures = [pool.submit(run_partition, g, p) for g, p in parts.items()]
            for f in futures:
                f.result()

    summary = batch_summary.summarize(batch_dir)
    rows = [results[c.key] for c in cases if c.key in results]
    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ("PASS", "SKIP", "FAIL", "TIMEOUT")}
    print(f"{LOG} done: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"{LOG} status:  {status_csv}")
    print(f"{LOG} summary: {summary}")
    bad = [r for r in rows if r["status"] in ("FAIL", "TIMEOUT")]
    if bad:
        print(f"{LOG} failed: " +
              ", ".join(f"{r['repo']}:{r['run_name']}" for r in bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
