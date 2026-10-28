import argparse

import pytest

import run_batch


def _row(repo, op, op_file, collected="yes", marked_skip=""):
    return {"repo": repo, "op": op, "op_file": op_file,
            "collected": collected, "marked_skip": marked_skip}


INVENTORY = {
    "main": [
        _row("main", "abs", "abs"),
        _row("main", "perf_x", "a"),
        _row("main", "perf_x", "FLA/b"),
        _row("main", "skipped", "skipped", marked_skip="yes"),
        _row("main", "npu_only", "npu_only", collected="no"),
        _row("main", "dismissed", "dismissed"),
    ],
    "vllm": [_row("vllm", "abs", "abs")],
}


def _args(tmp_path, **kw):
    dismiss = tmp_path / "dismiss.txt"
    dismiss.write_text("# comment\ndismissed\nvllm:nothing\n")
    base = dict(all=False, ops=None, ops_file=None, repos=None, dismiss=dismiss,
                include_marked_skip=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_load_dismiss(tmp_path):
    assert run_batch.load_dismiss(_args(tmp_path).dismiss) == {
        ("main", "dismissed"), ("vllm", "nothing")}


def test_select_all_filters_and_names_duplicates(tmp_path):
    cases = run_batch.resolve_selection(_args(tmp_path, all=True), INVENTORY)
    assert [c.key for c in cases] == [
        "main:abs@abs", "main:perf_x@a", "main:perf_x@FLA/b", "vllm:abs@abs"]
    assert [c.run_name for c in cases] == ["abs", "perf_x@a", "perf_x@FLA_b", "abs"]


def test_select_specs(tmp_path):
    cases = run_batch.resolve_selection(
        _args(tmp_path, ops=["abs", "vllm:abs", "perf_x@FLA/b"]), INVENTORY)
    assert [c.key for c in cases] == ["main:abs@abs", "vllm:abs@abs", "main:perf_x@FLA/b"]


@pytest.mark.parametrize("spec", ["perf_x", "nope", "other:abs", "abs@nope"])
def test_select_rejects_bad_specs(tmp_path, spec):
    with pytest.raises(SystemExit):
        run_batch.resolve_selection(_args(tmp_path, ops=[spec]), INVENTORY)


def test_partition_is_stable_and_complete(tmp_path):
    cases = run_batch.resolve_selection(_args(tmp_path, all=True), INVENTORY)
    first = run_batch.partition(cases, ["0", "1", "2"])
    again = run_batch.partition(list(reversed(cases)), ["0", "1", "2"])
    assert {g: sorted(c.key for c in cs) for g, cs in first.items()} == \
        {g: sorted(c.key for c in cs) for g, cs in again.items()}
    assert sorted(c.key for cs in first.values() for c in cs) == sorted(c.key for c in cases)


SMI_GPUS = "0, GPU-aaa, 3\n1, GPU-bbb, 1034\n2, GPU-ccc, 20480\n"
SMI_APPS = "GPU-bbb, 4242, python, 1024\nGPU-bbb, 4243, /usr/bin/foo, 10\n"


def test_gpu_parsing():
    gpus = run_batch.parse_gpus(SMI_GPUS)
    assert gpus["0"] == gpus["GPU-aaa"] == ("GPU-aaa", 3)
    assert run_batch.parse_compute_apps(SMI_APPS, "GPU-aaa") == []
    assert run_batch.parse_compute_apps(SMI_APPS, "GPU-bbb") == [
        "pid 4242 python (1024 MiB)", "pid 4243 /usr/bin/foo (10 MiB)"]


def test_gpu_users(monkeypatch):
    out = {"--query-ppu=index,uuid,memory.used": SMI_GPUS,
           "--query-compute-apps=gpu_uuid,pid,process_name,used_memory": SMI_APPS}
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: out.get(args[0]))
    assert run_batch.gpu_users("0") == []
    assert len(run_batch.gpu_users("1")) == 2
    assert run_batch.gpu_users("2") == ["20480 MiB in use by processes not visible here"]
    assert run_batch.gpu_users("9") is None


def test_gpu_users_falls_back_to_the_nvidia_spelling(monkeypatch):
    out = {"--query-gpu=index,uuid,memory.used": SMI_GPUS,
           "--query-compute-apps=gpu_uuid,pid,process_name,used_memory": SMI_APPS}
    seen = []

    def fake(args, env=None):
        seen.append(args[0])
        return out.get(args[0])

    monkeypatch.setattr(run_batch, "_smi", fake)
    assert run_batch.gpu_users("0") == []
    assert seen[0] == "--query-ppu=index,uuid,memory.used"  # PPU spelling tried first
    assert "--query-gpu=index,uuid,memory.used" in seen


# Real `mx-smi` (2.3.4 / MACA 3.8.1.3) output: device lines carry no colon before
# the product name, memory is reported in KB, processes in a box that says
# `no process found` on an idle card.
MX_LISTING = (
    "mx-smi  version: 2.3.4\n"
    "GPU#0    MXC550      0000:2b:00.0   Available (UUID: GPU-0bde36ca-2b0e-3108-0512-6a8e65e21b58)\n"
    "GPU#1    MXC550      0000:3a:00.0   Available (UUID: GPU-bc1ae580-0fd1-8ee6-247e-c780293b629a)\n"
)
MX_MEMORY = (
    "Attached GPUs                                     : 2\n"
    "GPU#0  MXC550  0000:2b:00.0\n"
    "    Memory\n"
    "        vis_vram total                            : 67108864 KB\n"
    "        vis_vram used                             : 879036 KB\n"
    "        vis_vram usage                            : 1.3 %\n"
    "        vram total                                : 67108864 KB\n"
    "        vram used                                 : 879036 KB\n"
    "GPU#1  MXC550  0000:3a:00.0\n"
    "    Memory\n"
    "        vis_vram used                             : 3145728 KB\n"
    "        vram used                                 : 3145728 KB\n"
)
MX_PROCS_IDLE = (
    "+---------------------------------------------------------------------------------+\n"
    "| Process:                                                                        |\n"
    "|  GPU                    PID         Process Name                 GPU Memory     |\n"
    "|                                                                  Usage(MiB)     |\n"
    "|=================================================================================|\n"
    "|  no process found                                                               |\n"
    "+---------------------------------------------------------------------------------+\n"
)
MX_PROCS_BUSY = MX_PROCS_IDLE.replace(
    "|  no process found                                                               |",
    "|  0      1787940     python -m pytest           1024                             |")
# A neighbour's torchrun (pt_elastic) holds a device file on every card with 0 MiB:
# this is what --show-all-process prints on a box that also runs someone else's job.
MX_PROCS_ALL = (
    "|  0                  2921416         python                       1258           |\n"
    "|  0                  2921417         python                       0              |\n"
    "|  0                  2923230         python                       0              |\n"
    "|  0                  2923640         pt_elastic                   0              |\n"
    "|  5                  2921416         python                       0              |\n"
    "|  5                  2923640         pt_elastic                   0              |\n"
)
MX_MEMORY_IDLE = MX_MEMORY.replace("3145728", "879036")  # both cards at the driver baseline


def test_mx_memory_and_processes_parsing():
    assert run_batch.parse_mx_memory(MX_MEMORY) == {"0": 879036, "1": 3145728}
    assert run_batch.parse_mx_processes(MX_PROCS_IDLE) == {}
    assert run_batch.parse_mx_processes(MX_PROCS_BUSY) == {
        "0": ["pid 1787940 python -m pytest (1024 MiB)"]}
    assert run_batch.parse_mx_processes(MX_PROCS_ALL)["0"] == [
        "pid 2921416 python (1258 MiB)", "pid 2921417 python (0 MiB)",
        "pid 2923230 python (0 MiB)", "pid 2923640 pt_elastic (0 MiB)"]


def test_gpu_users_ignores_idle_mx_smi_handles(monkeypatch):
    """0 MiB rows are idle device-file handles, not users: counting them would make
    --busy-gpu wait hang forever on a box where a neighbour's torchrun opened every
    card."""
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "mx-smi")

    def fake(args, env=None):
        if args[0] in ("--show-process", "--show-all-process"):
            return MX_PROCS_ALL
        return MX_MEMORY_IDLE

    monkeypatch.setattr(run_batch, "_smi", fake)
    assert run_batch.gpu_users("0") == ["pid 2921416 python (1258 MiB)"]
    assert run_batch.gpu_users("1") == []  # only 0 MiB handles (none at all here)


# `mx-smi --show-process` (the preferred spelling) on the same box: same box drawing,
# but only the process that is actually running.
MX_PROCS_RUNNING = (
    "+---------------------------------------------------------------------------------+\n"
    "| Process:                                                                        |\n"
    "|  GPU                    PID         Process Name                 GPU Memory     |\n"
    "|                                                                  Usage(MiB)     |\n"
    "|=================================================================================|\n"
    "|  1                  2929140         python                       5054           |\n"
    "+---------------------------------------------------------------------------------+\n"
)


def test_gpu_users_uses_show_process_before_the_wider_spelling(monkeypatch):
    """--show-process reports only the running process, so it is asked first; the
    listing of idle device-file handles is the fallback only."""
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "mx-smi")
    seen = []

    def fake(args, env=None):
        seen.append(args[0])
        return MX_PROCS_RUNNING if args[0] == "--show-process" else MX_MEMORY

    monkeypatch.setattr(run_batch, "_smi", fake)
    assert run_batch.gpu_users("1") == ["pid 2929140 python (5054 MiB)"]
    assert seen[0] == "--show-memory"
    assert "--show-process" in seen
    assert "--show-all-process" not in seen  # never reached once the precise one answers


def test_visible_gpus_accepts_a_colonless_mx_smi_listing(monkeypatch):
    """mx-smi 2.3.x writes `GPU#0    MXC550   0000:2b:00.0`, with no colon for the
    old regex to key on."""
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: MX_LISTING if args == ["-L"] else None)
    assert run_batch.visible_gpus(None) == ["0", "1"]


def test_gpu_users_uses_the_mx_smi_blocks(monkeypatch):
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "mx-smi")
    out = {"--show-memory": MX_MEMORY, "--show-all-process": MX_PROCS_IDLE}
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: out.get(args[0]))
    # 879036 KB (858 MiB) is the driver baseline of an idle card, below BUSY_MEMORY_MIB.
    assert run_batch.gpu_users("0") == []
    # 3145728 KB with no process listed: another container's, or not visible here.
    assert run_batch.gpu_users("1") == ["3072 MiB in use by processes not visible here"]
    assert run_batch.gpu_users("9") is None


def test_gpu_users_reports_the_mx_smi_process_rows(monkeypatch):
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "mx-smi")
    out = {"--show-memory": MX_MEMORY, "--show-all-process": MX_PROCS_BUSY}
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: out.get(args[0]))
    assert run_batch.gpu_users("0") == ["pid 1787940 python -m pytest (1024 MiB)"]


def test_smi_name_strips_the_directory():
    """ZL_BENCH_SMI may name a path; the dialect follows the tool's name."""
    assert run_batch.smi_name("mx-smi") == "mx-smi"
    assert run_batch.smi_name("/opt/maca/bin/mx-smi") == "mx-smi"
    assert run_batch.smi_name(None) == ""


def test_gpu_users_reads_a_path_valued_smi_tool(monkeypatch):
    """`ZL_BENCH_SMI=/opt/maca/bin/mx-smi` must still take the mx-smi query path:
    the --query-* spellings mx-smi rejects would make gpu_users return None and
    skip the idle check, so a busy card would be measured anyway."""
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "/opt/maca/bin/mx-smi")
    out = {"--show-memory": MX_MEMORY, "--show-all-process": MX_PROCS_BUSY}
    seen = []

    def fake(args, env=None):
        seen.append(args[0])
        return out.get(args[0])

    monkeypatch.setattr(run_batch, "_smi", fake)
    assert run_batch.gpu_users("0") == ["pid 1787940 python -m pytest (1024 MiB)"]
    assert not any(arg.startswith("--query") for arg in seen)


def test_no_smi_reason_names_the_mx_smi_spellings_for_a_path(monkeypatch):
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "/opt/maca/bin/mx-smi")
    reason = run_batch._no_smi_reason({})
    assert "--show-memory" in reason and "--show-all-process" in reason


def test_mx_gpu_index_maps_a_uuid(monkeypatch):
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: MX_LISTING if args == ["-L"] else None)
    assert run_batch._mx_gpu_index("1") == "1"
    assert run_batch._mx_gpu_index("GPU-bc1ae580-0fd1-8ee6-247e-c780293b629a") == "1"
    assert run_batch._mx_gpu_index("GPU-unknown") is None


def test_wait_for_idle_gpu_policies(monkeypatch):
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu, env=None: ["pid 1 x (1 MiB)"])
    assert run_batch.wait_for_idle_gpu("0", "ignore") == ""
    assert run_batch.wait_for_idle_gpu("0", "warn") == "gpu_busy"
    calls = iter([["pid 1 x (1 MiB)"], []])
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu, env=None: next(calls))
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: None)
    assert run_batch.wait_for_idle_gpu("0", "wait").startswith("gpu_wait=")


def test_parse_env():
    assert run_batch.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(SystemExit):
        run_batch.parse_env(["novalue"])


def test_replay_dir(tmp_path):
    case = run_batch.Case("vllm", "abs", "abs", "abs")
    args = argparse.Namespace(replay_root=tmp_path)
    assert run_batch.replay_dir(case, args) == tmp_path / "vllm" / "abs"
    (tmp_path / "run.log").write_text("")
    assert run_batch.replay_dir(case, args) == tmp_path  # a single-op record
    assert run_batch.replay_dir(case, argparse.Namespace(replay_root=None)) is None


def test_smi_tool_prefers_the_override_then_what_is_installed(monkeypatch):
    monkeypatch.delenv("ZL_BENCH_SMI", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("MACA_PATH", raising=False)
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_batch.shutil, "which",
                        lambda tool, path=None: f"/usr/bin/{tool}" if tool == "ppu-smi" else None)
    assert run_batch.smi_tool() == "ppu-smi"
    monkeypatch.setenv("ZL_BENCH_SMI", "/opt/bin/my-smi")
    assert run_batch.smi_tool() == "/opt/bin/my-smi"


def test_smi_tool_prefers_ppu_smi_under_the_ppu_sdk(monkeypatch):
    monkeypatch.delenv("ZL_BENCH_SMI", raising=False)
    monkeypatch.delenv("MACA_PATH", raising=False)
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PPU_SDK", "/usr/local/PPU_SDK")
    monkeypatch.setattr(run_batch.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    assert run_batch.smi_tool() == "ppu-smi"


def test_visible_gpus_reads_ppu_smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "ppu-smi")
    monkeypatch.setattr(run_batch.subprocess, "run",
                        lambda cmd, **kw: argparse.Namespace(
                            stdout="PPU 0: T-Head Zhenwu 810E\nPPU 1: T-Head Zhenwu 810E\n"))
    assert run_batch.visible_gpus(None) == ["0", "1"]


def test_visible_gpus_without_a_tool_assumes_one_card(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: None)
    assert run_batch.visible_gpus(None) == ["0"]


def test_gpu_users_falls_back_to_memory_when_no_process_is_listed(monkeypatch):
    """A process query that succeeds with no rows is an idle answer, not a
    failure: the 20480 MiB card must still report its invisible processes."""
    out = {"--query-ppu=index,uuid,memory.used": SMI_GPUS,
           "--query-compute-apps=gpu_uuid,pid,process_name,used_memory": ""}
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: out.get(args[0]))
    assert run_batch.gpu_users("0") == []  # 3 MiB and no process: idle
    assert run_batch.gpu_users("2") == ["20480 MiB in use by processes not visible here"]


def test_gpu_users_ignores_a_usage_block_printed_by_a_csv_query(monkeypatch):
    """A tool that answers an unknown option with a usage block and exit 0 is not
    an answer: that spelling is skipped and the next one is tried."""
    out = {"--query-ppu=index,uuid,memory.used": "Usage: ppu-smi [options]\n",
           "--query-gpu=index,uuid,memory.used": SMI_GPUS,
           "--query-compute-apps=gpu_uuid,pid,process_name,used_memory": SMI_APPS}
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: out.get(args[0]))
    assert len(run_batch.gpu_users("1")) == 2


def test_wait_for_idle_gpu_sees_a_gpu_whose_processes_are_not_listed(monkeypatch):
    monkeypatch.setattr(run_batch, "gpu_users", lambda gpu, env=None: ["20480 MiB in use"])
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: None)
    assert run_batch.wait_for_idle_gpu("0", "warn") == "gpu_busy"


# ---------------- MetaX (MACA) ----------------

MACA_SUMMARY = ("======================= MetaX System Management Interface "
                "=======================\n"
                "Attached GPUs                            : 8\n")


def test_accelerator_reads_the_vendor_stack(monkeypatch):
    for key in ("MACA_PATH", "MACA_VISIBLE_DEVICES", "PPU_SDK"):
        monkeypatch.delenv(key, raising=False)
    assert run_batch.accelerator() is None
    monkeypatch.setenv("PPU_SDK", "/usr/local/PPU_SDK")
    assert run_batch.accelerator() == "ppu"
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    assert run_batch.accelerator() == "metax"
    assert run_batch.accelerator({"MACA_VISIBLE_DEVICES": "0"}) == "metax"


def test_smi_tool_prefers_mx_smi_under_the_maca_stack(monkeypatch):
    monkeypatch.delenv("ZL_BENCH_SMI", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    monkeypatch.setattr(run_batch.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    assert run_batch.smi_tool() == "mx-smi"


def test_smi_tool_picks_mx_smi_from_the_visibility_variable(monkeypatch):
    """accelerator() is the single vendor source: MACA_VISIBLE_DEVICES alone must
    still choose mx-smi, never the nvidia-smi that happens to be installed."""
    monkeypatch.delenv("ZL_BENCH_SMI", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("MACA_PATH", raising=False)
    monkeypatch.setenv("MACA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(run_batch.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    assert run_batch.accelerator() == "metax"
    assert run_batch.smi_tool() == "mx-smi"


def test_visible_gpus_reads_maca_visible_devices_first(monkeypatch):
    monkeypatch.setenv("MACA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert run_batch.visible_gpus(None) == ["2", "3"]


def test_visible_gpus_reads_the_mx_smi_summary(monkeypatch):
    """mx-smi answers in its own query spelling, so the count also comes from
    its `Attached GPUs: N` line (the field FlagGems' gpu_check_metax.sh reads)."""
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_batch, "smi_tool", lambda: "mx-smi")
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: MACA_SUMMARY if not args else None)
    assert run_batch.visible_gpus(None) == [str(i) for i in range(8)]


def test_visible_gpus_accepts_a_gpu_hash_listing(monkeypatch):
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(run_batch, "smi_tool", lambda env=None: "mx-smi")
    monkeypatch.setattr(run_batch, "_smi", lambda args, env=None: (
        "GPU#0: MetaX C500\nGPU#1: MetaX C500\n" if args == ["-L"] else None))
    assert run_batch.visible_gpus(None) == ["0", "1"]


def _spawn_args(**kw):
    base = dict(env={}, pythonpath=None, shape_file=None, replay_root=None, use_flagtune=0)
    base.update(kw)
    return argparse.Namespace(**base)


def _capture_spawn(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_batch.subprocess, "Popen", lambda cmd, **kw: seen.update(kw))
    return seen


def test_spawn_sets_the_maca_visibility_variable_on_metax(monkeypatch, tmp_path):
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    monkeypatch.setenv("MACA_VISIBLE_DEVICES", "0,1")  # the caller's mask must not win
    seen = _capture_spawn(monkeypatch)
    case = run_batch.Case("main", "abs", "abs", "abs")
    run_batch._spawn(case, tmp_path / "out", "3", _spawn_args(), "")
    assert seen["env"]["MACA_VISIBLE_DEVICES"] == "3"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_spawn_passes_the_metric_whitelist(monkeypatch, tmp_path):
    """--metrics latency skips the baseline op; it travels as METRICS, and an unset
    value must not leak the caller's shell variable into the op."""
    monkeypatch.setenv("METRICS", "latency_base")  # the caller's value must not win
    seen = _capture_spawn(monkeypatch)
    case = run_batch.Case("main", "abs", "abs", "abs")
    run_batch._spawn(case, tmp_path / "a", "0", _spawn_args(metrics="latency"), "")
    assert seen["env"]["METRICS"] == "latency"

    run_batch._spawn(case, tmp_path / "b", "0", _spawn_args(), "")
    assert "METRICS" not in seen["env"]


def test_spawn_takes_the_maca_stack_from_a_batch_env(monkeypatch, tmp_path):
    for key in ("MACA_PATH", "MACA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    seen = _capture_spawn(monkeypatch)
    case = run_batch.Case("main", "abs", "abs", "abs")
    run_batch._spawn(case, tmp_path / "out", "2", _spawn_args(env={"MACA_PATH": "/opt/maca"}), "")
    assert seen["env"]["MACA_VISIBLE_DEVICES"] == "2"


def test_spawn_leaves_the_maca_variable_alone_off_metax(monkeypatch, tmp_path):
    for key in ("MACA_PATH", "MACA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    seen = _capture_spawn(monkeypatch)
    case = run_batch.Case("main", "abs", "abs", "abs")
    run_batch._spawn(case, tmp_path / "out", "0", _spawn_args(), "")
    assert "MACA_VISIBLE_DEVICES" not in seen["env"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_smi_tool_refuses_another_vendors_tool(monkeypatch):
    """Once the stack says MetaX, only mx-smi is acceptable: an installed
    nvidia-smi cannot enumerate MetaX devices, so None (plus a warning) beats
    inspecting the wrong card."""
    monkeypatch.delenv("ZL_BENCH_SMI", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("MACA_PATH", raising=False)
    monkeypatch.setenv("MACA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(run_batch.shutil, "which",
                        lambda tool, path=None: "/usr/bin/nvidia-smi"
                        if tool == "nvidia-smi" else None)
    assert run_batch.accelerator() == "metax"
    assert run_batch.smi_tool() is None
    assert "mx-smi is not installed" in run_batch._no_smi_reason()


def test_smi_tool_reads_the_stack_from_the_given_env(monkeypatch):
    """`--env MACA_PATH=...` reaches the idle check: the tool comes from the op's
    environment, not only from this process's."""
    for key in ("ZL_BENCH_SMI", "PPU_SDK", "MACA_PATH", "MACA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(run_batch.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    assert run_batch.smi_tool() == "nvidia-smi"  # no stack marker: the plain order
    assert run_batch.smi_tool(run_batch.op_env({"MACA_PATH": "/opt/maca"})) == "mx-smi"


def test_smi_runs_the_tool_with_the_given_env(monkeypatch):
    """The tool runs under the environment that selected it: a `--env PATH` /
    `LD_LIBRARY_PATH` must reach the subprocess, otherwise a tool found there is
    not loadable and the query fails silently."""
    monkeypatch.delenv("ZL_BENCH_SMI", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, kw=kw)
        return argparse.Namespace(stdout="")

    monkeypatch.setattr(run_batch.subprocess, "run", fake_run)
    monkeypatch.setattr(run_batch.shutil, "which", lambda tool, path=None: f"/opt/maca/bin/{tool}")
    env = run_batch.op_env({"PATH": "/opt/maca/bin:/usr/bin",
                            "LD_LIBRARY_PATH": "/opt/maca/lib"})
    assert run_batch._smi(["-L"], env) == ""
    assert seen["cmd"][0] == "mx-smi"  # the name is resolved against the given PATH
    assert seen["kw"]["env"]["PATH"] == "/opt/maca/bin:/usr/bin"
    assert seen["kw"]["env"]["LD_LIBRARY_PATH"] == "/opt/maca/lib"


def test_visible_gpus_reads_the_given_env(monkeypatch):
    monkeypatch.delenv("MACA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert run_batch.visible_gpus(None, {"MACA_VISIBLE_DEVICES": "2,3"}) == ["2", "3"]


def test_run_case_checks_idle_with_the_merged_env(monkeypatch, tmp_path):
    """The idle check and the subprocess must see the same environment: a
    `--env MACA_PATH` that makes the op run on MetaX must also make the check
    ask mx-smi."""
    seen = {}
    args = argparse.Namespace(resume=False, dtypes="", op_timeout=0, busy_gpu="warn",
                              env={"MACA_PATH": "/opt/maca"}, pythonpath=None,
                              shape_file=None, replay_root=None, use_flagtune=0)
    monkeypatch.setattr(run_batch, "_spawn", lambda *a, **kw: None)
    monkeypatch.setattr(run_batch, "_wait", lambda proc, timeout: (0, False))
    monkeypatch.setattr(run_batch, "_classify",
                        lambda case, out_dir, gpu, rc, ms, notes, timed_out: {
                            "repo": case.repo, "op": case.op, "status": "PASS",
                            "return_code": "0", "notes": notes})
    monkeypatch.setattr(run_batch, "wait_for_idle_gpu",
                        lambda gpu, policy, env=None: seen.update(env=env) or "")
    case = run_batch.Case("main", "abs", "abs", "abs")
    run_batch.run_case(case, tmp_path, "0", args)
    assert seen["env"]["MACA_PATH"] == "/opt/maca"
    assert run_batch.accelerator(seen["env"]) == "metax"
