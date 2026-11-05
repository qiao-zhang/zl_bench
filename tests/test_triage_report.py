import csv

import triage_report

FAIL_LOG = """\
>>> op=eq_  mode=record  USE_FLAGTUNE=0
>>> triton=/opt/triton (git abc1234)
>>> env: FLAGTREE_BACKEND=metax
===================== test session starts ======================
platform linux -- Python 3.12.3, pytest-9.0.3
______________________ test_eq_ ______________________

    def test_eq_():
>       assert fn(x)
E       RuntimeError: PassManager::run failed
E       loc("eq_kernel":12:5): error: unsupported conversion
E       note: see compiler log

benchmark/test_eq_.py:42: RuntimeError
=========================== short test summary info ============================
FAILED benchmark/test_eq_.py::test_eq_ - RuntimeError: PassManager::run failed
"""

MLIR_BEFORE_LOG = """\
______________________ test_put ______________________

error: 'tt.store' op operand type mismatch
LLVM ERROR: cannot select: intrinsic %llvm.maca.barrier
E   RuntimeError: Internal Triton llir codegen error:
E   see stderr above
"""

SKIP_LOG = """\
______________________ test_mm ______________________
=========================== short test summary info ============================
SKIPPED [1] benchmark/test_mm.py:30: needs TransformerEngine >= 1.0
SKIPPED [2] benchmark/test_mm.py:44: needs TransformerEngine >= 1.0
XFAIL benchmark/test_mm.py::test_mm
"""

TABLE_LOG = """\
Operator: mkldnn_x  (dtype=torch.bfloat16, mode=kernel, level=core)
Status     Torch Latency (ms)   Gems Latency (ms)   Gems Speedup  Size Detail
SUCCESS                    N/A                 N/A            N/A  [torch.Size([4, 4])]
FAILED                     N/A                 N/A            N/A  [torch.Size([8, 8])]
"""

EMPTY_LOG = """\
>>> op=harness  mode=record  USE_FLAGTUNE=0
collected 1 item
.
"""


def test_failure_block_keeps_the_multiline_message():
    block = triage_report.failure_block(FAIL_LOG)
    assert block[0] == "RuntimeError: PassManager::run failed"
    assert any("unsupported conversion" in line for line in block)
    assert any(line.startswith("note:") for line in block)
    # The short summary repeats the first line; the block must not run into it.
    assert not any(line.startswith("FAILED benchmark/") for line in block)


def test_failure_block_picks_up_diagnostics_printed_before_the_exception():
    """Triton writes MLIR/LLVM diagnostics to stderr *before* the RuntimeError line."""
    block = triage_report.failure_block(MLIR_BEFORE_LOG)
    assert block[0].startswith("error: 'tt.store'")
    assert any("LLVM ERROR" in line for line in block)


def test_failure_block_without_e_lines_is_empty():
    assert triage_report.failure_block(TABLE_LOG) == []


def test_classify_maps_shell_messages_to_root_causes():
    cases = {
        "E   RuntimeError: PassManager::run failed": ("compiler_pass", "issue"),
        "E   RuntimeError: Internal Triton mlir opt error:": ("compiler_mlir", "issue"),
        "E   RuntimeError: Internal Triton llir codegen error:": ("compiler_llir", "issue"),
        "E   triton.compiler.errors.CompilationError: at 56:14:": ("compiler_kernel", "issue"),
        "E   RuntimeError: Triton Error [MACA]: memory size or pointer value too large "
        "to fit in 32 bit": ("runtime_address", "issue"),
        "E   triton.runtime.errors.OutOfResources: out of resource: shared memory, "
        "Required: 92160, Hardware limit: 65536.": ("smem_limit", "config"),
        "E   torch.AcceleratorError: CUDA error: an illegal memory access was encountered":
            ("crash", "investigate"),
        "E   AssertionError: Tensor-likes are not equal!": ("accuracy", "investigate"),
        "E   AssertionError: Input shapes should have M >= 1, N >= 1 and K >= 16":
            ("invalid_config", "config"),
        "E   RuntimeError: USE_FLASH_ATTENTION was not enabled for build.": ("env_missing", "env"),
        "E   ModuleNotFoundError: No module named 'vllm_metax'": ("missing_module", "env"),
        "E   AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute "
        "'make_nv_mma_shared_encoding_attr'": ("binding_mismatch", "env"),
        "E   TypeError: 'function' object is not subscriptable": ("upstream_code", "upstream"),
        "E   AttributeError: module 'flag_gems' has no attribute 'swapdims_'":
            ("upstream_code", "upstream"),
        "E   ValueError: Expected a cuda device, but got: cpu": ("cpu_only", "dismiss"),
        "E   RecursionError: maximum recursion depth exceeded": ("recursion", "investigate"),
    }
    for line, expected in cases.items():
        block = triage_report.failure_block(line + "\n")
        assert triage_report.classify(block)[:2] == expected, line


def test_classify_falls_back_to_unclassified():
    block = triage_report.failure_block("E   SomethingBrandNew: kaboom\n")
    assert triage_report.classify(block)[:2] == ("unclassified", "unknown")


def test_signature_folds_per_shape_values_but_keeps_small_numbers():
    text = ("E   OutOfResources: out of resource: shared memory, Required: 196608, "
            "Hardware limit: 65536. Reducing block sizes\n")
    _, _, signature = triage_report.classify(triage_report.failure_block(text))
    assert "Required: N" in signature
    assert "196608" not in signature


def test_skip_reasons_dedups_and_drops_the_location():
    reasons = triage_report.skip_reasons(SKIP_LOG)
    assert reasons == ["needs TransformerEngine >= 1.0", "xfail"]


def test_normalize_strips_addresses_and_shapes():
    out = triage_report.normalize("fault at 0x7f3a2b in torch.Size([1024, 2048]) size 4096")
    assert out == "fault at 0xADDR in SHAPE size N"


def _write_batch(root, rows, logs):
    (root / "ops_status.csv").write_text(
        "repo,op,op_file,run_name,gpu,status,return_code,elapsed_ms,rows_total,"
        "rows_success,notes,out_dir\n" + rows, encoding="utf-8")
    for rel, text in logs.items():
        log = root / rel
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")


BATCH_ROWS = (
    "main,good,good,good,0,PASS,0,10,2,2,,/x\n"
    "main,eq_,eq_,eq_,0,FAIL,1,10,0,0,retry_rc=1->1,/x\n"
    "main,mkldnn_x,mkldnn_x,mkldnn_x,0,FAIL,0,10,2,1,,/x\n"
    "main,harness,harness,harness,0,FAIL,0,10,0,0,,/x\n"
    "vllm,mm,mm,mm,0,SKIP,0,10,0,0,resumed,/x\n"
)


def test_scan_and_report_end_to_end(tmp_path):
    _write_batch(tmp_path, BATCH_ROWS, {
        "main/good/run.log": "Operator: good  (dtype=torch.bfloat16, mode=kernel, level=core)\n"
                             "SUCCESS 1 1 1  [torch.Size([4])]\n",
        "main/eq_/run.log": FAIL_LOG,
        "main/mkldnn_x/run.log": TABLE_LOG,
        "main/harness/run.log": EMPTY_LOG,
        "vllm/mm/run.log": SKIP_LOG,
    })
    verdicts = triage_report.scan_batch(tmp_path)
    by_op = {v.op: v for v in verdicts}

    assert by_op["eq_"].category == "compiler_pass"
    assert by_op["eq_"].action == "issue"
    assert by_op["mkldnn_x"].category == "row_failed"      # rc=0 but a FAILED table row
    assert by_op["harness"].category == "no_rows"          # rc=0, nothing at all
    assert by_op["mm"].category == "skip"
    assert "TransformerEngine" in by_op["mm"].skip_reason
    assert by_op["good"].category == ""

    info = triage_report.platform_info([FAIL_LOG])
    assert info["backend"] == "metax"
    assert info["triton"] == "/opt/triton (git abc1234)"

    md = triage_report.render_markdown(verdicts, tmp_path, info)
    assert "1. 概览" in md
    assert "`compiler_pass`" in md
    assert "FLAGGEMS_DIR=${FLAGGEMS_MAIN_DIR:-../FlagGems} OP=eq_ OP_FILE=eq_" in md
    assert "TransformerEngine" in md
    assert "FLAGTREE_BACKEND=metax" in md

    csv_path = triage_report.write_csv(verdicts, tmp_path / "triage.csv")
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["op"] for r in rows] == ["good", "eq_", "mkldnn_x", "harness", "mm"]
    assert rows[1]["category"] == "compiler_pass"


def test_signal_return_code_is_recognised():
    assert triage_report._signal_return_code("134")
    assert triage_report._signal_return_code("139")
    assert triage_report._signal_return_code("-11")
    assert not triage_report._signal_return_code("0")
    assert not triage_report._signal_return_code("1")
    assert not triage_report._signal_return_code("")


def test_signal_killed_op_becomes_a_crash_without_a_traceback(tmp_path):
    """rc=134 has no `E` line at all; reporting it as `no_rows` would bury it."""
    _write_batch(tmp_path, "vllm,chunk_gdn2,FLA/test_chunk_gdn2,chunk_gdn2,0,FAIL,134,10,0,0,,/x\n", {
        "vllm/chunk_gdn2/run.log": EMPTY_LOG,
    })
    verdict = triage_report.scan_batch(tmp_path)[0]
    assert verdict.category == "crash"
    assert verdict.action == "investigate"


def test_attempt_log_is_used_when_the_surviving_log_has_no_error(tmp_path):
    """A dtype_rescue run renames the failing log to run.attempt0.log."""
    _write_batch(tmp_path, "main,mmw,mmw,mmw,0,FAIL,1,10,0,0,dtype_rescue,/x\n", {
        "main/mmw/run.log": EMPTY_LOG,
        "main/mmw/run.attempt0.log": FAIL_LOG,
    })
    verdict = triage_report.scan_batch(tmp_path)[0]
    assert verdict.category == "compiler_pass"


def test_missing_run_log_does_not_crash(tmp_path):
    _write_batch(tmp_path, "main,gone,gone,gone,0,FAIL,1,10,0,0,,/x\n", {})
    verdict = triage_report.scan_batch(tmp_path)[0]
    assert verdict.category == "no_rows"
    assert verdict.log.endswith("gone")


def test_dismiss_text_groups_the_not_our_defect_actions():
    verdicts = [
        triage_report.OpVerdict("main", "flash", "flash", "flash", "FAIL", "1", "",
                                category="env_missing", action="env", why="缺实现"),
        triage_report.OpVerdict("vllm", "swap", "swap", "swap", "FAIL", "1", "",
                                category="upstream_code", action="upstream", why="上游"),
        triage_report.OpVerdict("main", "eq_", "eq_", "eq_", "FAIL", "1", "",
                                category="compiler_pass", action="issue", why="后端"),
    ]
    text = triage_report.dismiss_text(verdicts, today="2026-11-04")
    assert "# ==== env_missing [缺实现] [2026-11-04]" in text
    assert "\nflash\n" in text
    assert "\nvllm:swap\n" in text
    assert "eq_" not in text            # an issue stays out of dismiss


def test_dismiss_text_is_empty_without_candidates():
    verdicts = [triage_report.OpVerdict("main", "a", "a", "a", "FAIL", "1", "",
                                        category="compiler_pass", action="issue")]
    assert triage_report.dismiss_text(verdicts) == ""
