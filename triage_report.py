#!/usr/bin/env python3
"""Attribute a batch's SKIP/FAIL ops to root causes and write a triage report.

Everything it needs is already on disk in the batch dir, so the same batch gives the
same report on any platform (NVIDIA / PPU / MetaX):
- the per-op verdict comes from ops_status.csv (run_batch.py);
- the failure evidence comes from each op's run.log -- the whole multi-line pytest
  exception block, which keeps the MLIR/LLVM diagnostics and source excerpt that the
  first `E` line alone hides;
- the skip reason comes from the `-rsx` short summary at the end of the same log;
- per-row failures come from batch_summary's result-table parser.

Classification matches on message keywords, never on the vendor name, so the rules
work unchanged across backends. Add a rule when a new shell message shows up.

Standalone use:
  python triage_report.py runs/batch_xxx                    # triage.md + triage.csv
  python triage_report.py runs/batch_xxx --top 3            # ops listed per group
  python triage_report.py runs/batch_xxx --dismiss-out /tmp/dismiss_candidate.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import batch_summary

# ---------------------------------------------------------------- log patterns

# pytest prefixes every line of the exception message with `E   `; the diagnostics
# that Triton prints to stderr (MLIR/LLVM) keep their own prefixes and often land
# *before* the `E` line, in the order the compiler wrote them.
E_LINE_RE = re.compile(r"^E {2,}(?P<msg>.*)$")
BARE_DIAG_RE = re.compile(r"^(?:error|warning|note|remark):\s|^LLVM ERROR|^loc\(")
# `_____ test_name _____` separators end an exception block.
BLOCK_BOUNDARY_RE = re.compile(r"^[_=]{5,}")
# `-rsx` prints `SKIPPED [1] benchmark/test_x.py:30: reason` at the end of the log.
SKIP_RE = re.compile(r"^SKIPPED\s+\[\d+\]\s+.+?:\d+:\s*(?P<reason>.*)$")
XFAIL_RE = re.compile(r"^XFAIL\s+\S+")

ENV_LINE_RE = re.compile(r"^>>> env:\s*(?P<env>.*)$")
TRITON_LINE_RE = re.compile(
    r"^>>> triton=(?P<path>\S*)(?:\s+\(git\s+(?P<rev>[0-9a-f]+)\))?")
BACKEND_RE = re.compile(r"FLAGTREE_BACKEND=(\S+)")
PLATFORM_RE = re.compile(r"^platform\s+(?P<os>\S+)\s+--\s+Python\s+(?P<py>\S+?)[,\s]")

# ------------------------------------------------------------- normalisation

ADDR_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b")
SHAPE_RE = re.compile(r"torch\.Size\(\[[^\]]*\]\)")
# Keep 1-2 digit numbers (`32 bit`, `M >= 1` stay readable), fold the rest so that
# per-shape values do not split one root cause into many groups.
BIG_NUM_RE = re.compile(r"\b\d{3,}\b")
WS_RE = re.compile(r"\s+")


def normalize(text: str, limit: int = 200) -> str:
    """One greppable signature line: no addresses, shapes or per-shape numbers."""
    out = SHAPE_RE.sub("SHAPE", text.strip())
    out = ADDR_RE.sub("0xADDR", out)
    out = BIG_NUM_RE.sub("N", out)
    out = WS_RE.sub(" ", out)
    return out[:limit]


# -------------------------------------------------------------------- rules
#
# Ordered: the first match wins, so a specific pattern must come before the generic
# one it would otherwise be swallowed by (`libtriton` before `has no attribute`).
# `action` drives how the report groups the op:
#   investigate - likely a real defect, worth a minimal repro
#   issue       - same root cause across many ops, file one backend/driver issue
#   config      - the measured config or shape set does not fit the hardware
#   env         - the environment lacks a dependency or a build option
#   upstream    - upstream test/benchmark code or an unimplemented op
#   dismiss     - not applicable on this kind of hardware
#   unknown     - no evidence in the log, look by hand


@dataclass(frozen=True)
class Rule:
    category: str
    action: str
    pattern: re.Pattern
    why: str


def _rule(category: str, action: str, pattern: str, why: str) -> Rule:
    return Rule(category, action, re.compile(pattern), why)


RULES: Sequence[Rule] = (
    _rule("crash", "investigate",
          r"illegal memory access|segmentation fault|sigsegv|sigabrt|core dumped",
          "进程崩溃/非法访存：最小化到单 shape 复跑，抓 core 或缩小 shape 定位越界读写"),
    _rule("accuracy", "investigate",
          r"tensor-likes are not (close|equal)|mismatched elements|"
          r"greatest (absolute|relative) difference|not close to",
          "结果与参考实现不符：这是真问题，优先于性能排查"),
    _rule("smem_limit", "config",
          r"out of resource: shared memory",
          "单 kernel 申请的 shared memory 超过硬件上限：换更小的 block/更少的 stage，或从清单里去掉"),
    _rule("oom", "config",
          r"out of memory|outofmemoryerror",
          "显存不足：缩小 shape 集，或确认该卡没有别的进程"),
    _rule("invalid_config", "config",
          r"invalid configuration argument|invalid launch|input shapes should have",
          "launch 参数或 shape 不满足约束：检查 shape 集与 autotune 空间"),
    _rule("runtime_address", "issue",
          r"too large to fit in 32 bit",
          "运行时把大小/地址按 32 位传递：同族算子合并成一个后端 issue"),
    _rule("compiler_pass", "issue",
          r"passmanager::run failed",
          "编译器 pass 内抛异常：先看异常前的 MLIR 诊断，再拿 ttir 离线复现"),
    _rule("compiler_mlir", "issue",
          r"internal triton mlir opt error",
          "MLIR 层优化崩溃：诊断在冒号后面，同族算子合并成一个 issue"),
    _rule("compiler_llir", "issue",
          r"internal triton llir codegen error",
          "LLVM 代码生成崩溃：多为后端不支持的 op/类型组合"),
    _rule("compiler_kernel", "issue",
          r"triton\.compiler\.errors\.compilationerror|compilationerror:",
          "kernel 本身编译失败：看源码片段与 `^` 标记指向的那一行"),
    _rule("binding_mismatch", "env",
          r"libtriton|no attribute 'make_",
          "Triton Python binding 与后端不一致：对齐 triton 与后端版本"),
    _rule("env_missing", "env",
          r"not enabled for build|not compiled with|requires compiling pytorch with|"
          r"cutlass not supported|not built with",
          "当前构建未包含该实现：装依赖/换构建，或 dismiss"),
    _rule("missing_module", "env",
          r"modulenotfounderror|importerror:",
          "缺 Python 包：装上或 dismiss"),
    _rule("cpu_only", "dismiss",
          r"expected a cuda device|expected out tensor to have device cpu",
          "该算子是 CPU/mkldnn 路径：在加速卡上不适用"),
    _rule("upstream_code", "upstream",
          r"'function' object is not subscriptable|has no attribute|nameerror|"
          r"json\.decoder\.jsondecodeerror|missing \d+ required positional argument|"
          r"unexpected keyword argument|attributeerror",
          "上游 benchmark/op 代码问题：与上游对齐，或确认是版本错配"),
    _rule("recursion", "investigate",
          r"recursionerror",
          "递归超限：多为算子实现在该后端下的自引用，单独复现"),
)

ACTION_ORDER = ("investigate", "issue", "config", "env", "upstream", "dismiss", "unknown")

ACTION_TITLE = {
    "investigate": "优先深挖（可能是真缺陷）",
    "issue": "合并成一个后端 issue（同根因）",
    "config": "调 config / shape 集",
    "env": "补环境或换构建",
    "upstream": "上游代码问题",
    "dismiss": "平台不适用，建议 dismiss",
    "unknown": "需人工确认（无异常块）",
}


# --------------------------------------------------------------- extraction


def failure_block(text: str, max_lines: int = 40, lead_lines: int = 20) -> List[str]:
    """The first exception's full message, plus the diagnostics printed before it.

    Returns [] when the log has no `E` line at all (signal crash, no result rows,
    or a harness that failed without a traceback).
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if E_LINE_RE.match(line)), None)
    if start is None:
        return []

    lead: List[str] = []
    j = start - 1
    while j >= 0 and len(lead) < lead_lines:
        stripped = lines[j].strip()
        if not stripped or E_LINE_RE.match(lines[j]) or BLOCK_BOUNDARY_RE.match(stripped):
            break
        if not BARE_DIAG_RE.match(stripped):
            break
        lead.append(stripped)
        j -= 1

    block = list(reversed(lead))
    for line in lines[start:start + max_lines]:
        match = E_LINE_RE.match(line)
        if match:
            block.append(match.group("msg").strip())
            continue
        stripped = line.strip()
        if BLOCK_BOUNDARY_RE.match(stripped):
            break
        if not stripped:
            continue
        if BARE_DIAG_RE.match(stripped):
            block.append(stripped)
            continue
        block.append(stripped)

    out: List[str] = []
    for line in block:
        if line and line not in out:
            out.append(line)
    return out


def classify(block: List[str]) -> Tuple[str, str, str]:
    """(category, action, signature) for one exception block."""
    if not block:
        return "unknown", "unknown", ""
    joined = "\n".join(block).lower()
    for rule in RULES:
        if not rule.pattern.search(joined):
            continue
        hit = next((line for line in block if rule.pattern.search(line.lower())), block[0])
        return rule.category, rule.action, normalize(hit)
    return "unclassified", "unknown", normalize(block[0])


def skip_reasons(text: str) -> List[str]:
    """Skip/xfail reasons from the `-rsx` short summary, normalised and deduplicated."""
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = SKIP_RE.match(stripped)
        reason = normalize(match.group("reason")) if match else ""
        if not match and XFAIL_RE.match(stripped):
            reason = "xfail"
        if reason and reason not in out:
            out.append(reason)
    return out


def platform_info(texts: List[str]) -> Dict[str, str]:
    """Backend / compiler / OS / Python, taken from any run.log header."""
    info: Dict[str, str] = {}
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if "backend" not in info:
                match = ENV_LINE_RE.match(stripped)
                if match and (hit := BACKEND_RE.search(match.group("env"))):
                    info["backend"] = hit.group(1)
                    info["compile_env"] = normalize(match.group("env"), 400)
            if "triton" not in info:
                match = TRITON_LINE_RE.match(stripped)
                if match and match.group("path"):
                    rev = match.group("rev")
                    info["triton"] = f"{match.group('path')} (git {rev})" if rev else match.group("path")
            if "os" not in info:
                match = PLATFORM_RE.match(stripped)
                if match:
                    info["os"] = match.group("os")
                    info["python"] = match.group("py")
        if {"backend", "triton", "os"} <= info.keys():
            break
    return info


# ------------------------------------------------------------------- report


@dataclass
class OpVerdict:
    repo: str
    op: str
    op_file: str
    run_name: str
    status: str
    return_code: str
    notes: str
    category: str = ""
    action: str = ""
    signature: str = ""
    why: str = ""
    skip_reason: str = ""
    rows_total: int = 0
    rows_failed: int = 0
    evidence: List[str] = field(default_factory=list)
    log: str = ""

    @property
    def key(self) -> str:
        return f"{self.repo}:{self.op}"

    @property
    def dir_key(self) -> str:
        return f"{self.repo}/{self.run_name}"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_ops_status(batch_dir: Path) -> List[Dict[str, str]]:
    path = batch_dir / "ops_status.csv"
    if not path.is_file():
        raise SystemExit(f"no ops_status.csv in {batch_dir}; point it at a run_batch.py "
                         f"batch dir or run batch_summary.py first")
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _op_dir(batch_dir: Path, row: Dict[str, str]) -> Optional[Path]:
    """The op's output dir; falls back to the op name for batches predating run_name.
    A missing dir still yields the expected path, so the report can name it."""
    for name in (row.get("run_name"), row.get("op")):
        if name and (batch_dir / row["repo"] / name).is_dir():
            return batch_dir / row["repo"] / name
    name = row.get("run_name") or row.get("op")
    return batch_dir / row["repo"] / name if name else None


def _signal_return_code(value: str) -> bool:
    """True for the 128+N or negative return code of a signal-killed process."""
    try:
        rc = int(value)
    except (TypeError, ValueError):
        return False
    return rc < 0 or rc >= 128


def _attempt_log(texts: List[Tuple[Path, str]]) -> Optional[List[str]]:
    """A rescue attempt renames the earlier log to run.attempt<N>.log -- use it when
    the surviving run.log carries no exception block."""
    for _, text in texts:
        block = failure_block(text)
        if block:
            return block
    return None


def scan_batch(batch_dir: Path) -> List[OpVerdict]:
    batch_dir = batch_dir.resolve()
    verdicts: List[OpVerdict] = []
    for row in _load_ops_status(batch_dir):
        verdict = OpVerdict(
            repo=row.get("repo", ""),
            op=row.get("op", ""),
            op_file=row.get("op_file", ""),
            run_name=row.get("run_name") or row.get("op", ""),
            status=row.get("status", ""),
            return_code=row.get("return_code", ""),
            notes=row.get("notes", ""),
        )
        out_dir = _op_dir(batch_dir, row)
        text = ""
        if out_dir is not None:
            verdict.log = str(out_dir)
            log = out_dir / "run.log"
            if log.is_file():
                text = _read(log)

        if verdict.status == "FAIL":
            block = failure_block(text)
            if not block and out_dir is not None:
                attempts = sorted(out_dir.glob("run.attempt*.log"))
                block = _attempt_log([(p, _read(p)) for p in attempts]) or []
            if block:
                verdict.category, verdict.action, verdict.signature = classify(block)
                verdict.why = next((r.why for r in RULES if r.category == verdict.category), "")
                verdict.evidence = block
            else:
                rows = batch_summary.parse_run_log(text)
                verdict.rows_total = len(rows)
                verdict.rows_failed = sum(1 for r in rows if r["row_status"] != "SUCCESS")
                if _signal_return_code(verdict.return_code):
                    # 128+signal (or a negative rc): the process died, so there is no
                    # traceback to attribute -- the log only has what it printed before.
                    verdict.category, verdict.action = "crash", "investigate"
                    verdict.why = ("进程被信号终止（rc≥128 或负值），日志里没有 traceback："
                                   "单独复跑该算子，必要时抓 core")
                    verdict.signature = f"rc={verdict.return_code} signal-terminated, no traceback"
                elif verdict.rows_failed:
                    verdict.category, verdict.action = "row_failed", "unknown"
                    verdict.why = "结果行有 FAILED：该 shape 的测量没跑通，看同目录 run.log 对应行"
                    verdict.signature = f"{verdict.rows_failed}/{verdict.rows_total} result rows FAILED"
                else:
                    verdict.category, verdict.action = "no_rows", "unknown"
                    verdict.why = ("既没有异常块也没有结果行：多为收集中断（rc=2）、"
                                   "harness 自身用例，或测试没有走到 benchmark")
                    verdict.signature = f"rc={verdict.return_code} no exception block, no result rows"
        elif verdict.status == "SKIP":
            reasons = skip_reasons(text)
            verdict.skip_reason = " | ".join(reasons) if reasons else "no SKIPPED line in run.log"
            verdict.category = "skip"
            verdict.signature = normalize(verdict.skip_reason)
        verdicts.append(verdict)
    return verdicts


# ---------------------------------------------------------------- rendering


def _group(verdicts: List[OpVerdict], keyfn) -> List[Tuple[str, List[OpVerdict]]]:
    buckets: Dict[str, List[OpVerdict]] = {}
    for verdict in verdicts:
        buckets.setdefault(keyfn(verdict), []).append(verdict)
    return sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _counts(verdicts: List[OpVerdict]) -> Counter:
    return Counter(v.status for v in verdicts)


def _op_list(verdicts: List[OpVerdict], top: int) -> str:
    keys = [v.key for v in verdicts]
    if top > 0 and len(keys) > top:
        return ", ".join(f"`{k}`" for k in keys[:top]) + f" … (+{len(keys) - top})"
    return ", ".join(f"`{k}`" for k in keys)


def _evidence_md(verdict: OpVerdict, limit: int) -> List[str]:
    lines = []
    for row in verdict.evidence[:limit]:
        lines.append(f"      {row}")
    if len(verdict.evidence) > limit:
        lines.append(f"      … (+{len(verdict.evidence) - limit} 行，见 {verdict.dir_key}/run.log)")
    return lines


def dismiss_text(verdicts: List[OpVerdict], actions: Sequence[str] = ("env", "upstream", "dismiss"),
                 today: Optional[str] = None) -> str:
    """A paste-ready ops/dismiss.txt block for the groups that are not our defect."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    out: List[str] = []
    for action in actions:
        for category, group in _group([v for v in verdicts if v.action == action], lambda v: v.category):
            if not group:
                continue
            why = next((v.why for v in group if v.why), category)
            out.append("")
            out.append(f"# ==== {category} [{why}] [{today}]")
            for verdict in sorted(group, key=lambda v: (v.repo, v.op)):
                out.append(verdict.key if verdict.repo != "main" else verdict.op)
    return "\n".join(out).strip() + "\n" if out else ""


def render_markdown(verdicts: List[OpVerdict], batch_dir: Path, info: Dict[str, str],
                    top: int = 0, evidence_lines: int = 12) -> str:
    counts = _counts(verdicts)
    total = len(verdicts)
    fails = [v for v in verdicts if v.status == "FAIL"]
    skips = [v for v in verdicts if v.status == "SKIP"]

    lines: List[str] = []
    lines.append(f"# {batch_dir.name} — SKIP/FAIL 归因报告")
    lines.append("")
    lines.append(f"- 批次目录：`{batch_dir}`")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if info.get("backend"):
        lines.append(f"- 后端：`FLAGTREE_BACKEND={info['backend']}`")
    if info.get("triton"):
        lines.append(f"- 编译器：`{info['triton']}`")
    if info.get("os"):
        lines.append(f"- 平台：{info['os']} / Python {info.get('python', '?')}")
    if info.get("compile_env") and info["compile_env"] != f"FLAGTREE_BACKEND={info.get('backend', '')}":
        lines.append(f"- 编译环境：`{info['compile_env']}`")
    lines.append("- 归因只读批次目录里的产物，与平台无关；分类按错误消息关键词匹配，不认厂商名。")
    lines.append("")

    lines.append("## 1. 概览")
    lines.append("")
    lines.append("| 状态 | 算子数 | 占比 |")
    lines.append("| --- | --- | --- |")
    for status in ("PASS", "SKIP", "FAIL", "TIMEOUT"):
        n = counts.get(status, 0)
        if n or status in ("FAIL", "SKIP"):
            share = f"{n * 100 / total:.1f}%" if total else "-"
            lines.append(f"| {status} | {n} | {share} |")
    lines.append(f"| 合计 | {total} | {'100%' if total else '-'} |")
    lines.append("")
    if fails:
        rc = Counter(v.return_code for v in fails)
        lines.append("FAIL 的返回码分布：" + "，".join(
            f"`rc={k}` × {v}" for k, v in sorted(rc.items(), key=lambda kv: -kv[1])))
        lines.append("")
        lines.append("`rc=0` 表示 pytest 正常退出但结果不合预期（结果行 FAILED 或没有结果行）；"
                     "`rc=1` 是普通失败；`rc=2` 多为收集中断；负数或 ≥128 是信号终止，"
                     "日志里不会有 Python traceback。")
        lines.append("")

    lines.append("## 2. FAIL 根因分组")
    lines.append("")
    if not fails:
        lines.append("本批次没有 FAIL。")
        lines.append("")
    else:
        lines.append("| # | 类别 | 建议动作 | 算子数 | 归一化错误签名 | 算子 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for i, (category, group) in enumerate(_group(fails, lambda v: v.category), 1):
            action = group[0].action
            sig = group[0].signature.replace("|", "\\|")[:90]
            lines.append(f"| {i} | `{category}` | {ACTION_TITLE.get(action, action)} | {len(group)} "
                         f"| `{sig}` | {_op_list(group, top)} |")
        lines.append("")

        lines.append("### 2.1 各组证据")
        lines.append("")
        for category, group in _group(fails, lambda v: v.category):
            lines.append(f"#### `{category}` — {len(group)} 个算子")
            lines.append("")
            if group[0].why:
                lines.append(f"- 处置：{group[0].why}")
            lines.append(f"- 算子：{_op_list(group, 0)}")
            variants = Counter(v.signature for v in group if v.signature)
            if len(variants) > 1:
                lines.append(f"- 组内有 {len(variants)} 种错误签名（按算子数）：")
                for sig, n in variants.most_common(3):
                    lines.append(f"  - `{sig[:110]}` × {n}")
                if len(variants) > 3:
                    lines.append(f"  - … 其余 {len(variants) - 3} 种见 triage.csv 的 signature 列")
            sample = group[0]
            lines.append(f"- 样例（`{sample.dir_key}/run.log`）：")
            lines.append("")
            lines.append("  ```")
            lines.extend(_evidence_md(sample, evidence_lines) if sample.evidence
                         else [f"      {sample.signature}"])
            lines.append("  ```")
            lines.append("")
            lines.append("  复现：")
            lines.append("")
            lines.append("  ```bash")
            repo_dir = "${FLAGGEMS_MAIN_DIR:-../FlagGems}" if sample.repo == "main" \
                else "${FLAGGEMS_VLLM_DIR:-../FlagGems-vllm}"
            lines.append(f"  FLAGGEMS_DIR={repo_dir} OP={sample.op} "
                         f"OP_FILE={sample.op_file} bash run_pytest.sh")
            lines.append("  ```")
            lines.append("")

    lines.append("## 3. SKIP 原因分组")
    lines.append("")
    if not skips:
        lines.append("本批次没有 SKIP。")
        lines.append("")
    else:
        lines.append("| # | 原因 | 算子数 | 算子 |")
        lines.append("| --- | --- | --- | --- |")
        for i, (reason, group) in enumerate(_group(skips, lambda v: v.signature), 1):
            lines.append(f"| {i} | {reason[:110]} | {len(group)} | {_op_list(group, top)} |")
        lines.append("")
        lines.append("SKIP 基本都是上游 `skipif`（缺参考实现、缺依赖、架构/厂商限制），"
                     "不需要处理，但要与 FAIL 分开统计。")
        lines.append("")

    investigate = [v for v in fails if v.action == "investigate"]
    lines.append("## 4. 需要人工处理")
    lines.append("")
    if investigate:
        lines.append(f"### 4.1 优先深挖 — {len(investigate)} 个算子")
        lines.append("")
        for category, group in _group(investigate, lambda v: v.category):
            lines.append(f"- `{category}`（{len(group)}）：{_op_list(group, 0)}")
            if group[0].why:
                lines.append(f"  - {group[0].why}")
        lines.append("")
    else:
        lines.append("### 4.1 优先深挖")
        lines.append("")
        lines.append("无：没有精度不符或崩溃类失败。")
        lines.append("")

    by_action = [(a, [v for v in fails if v.action == a]) for a in ACTION_ORDER]
    by_action = [(a, g) for a, g in by_action if g]
    if by_action:
        lines.append("### 4.2 按建议动作汇总")
        lines.append("")
        lines.append("| 建议动作 | 算子数 | 类别 |")
        lines.append("| --- | --- | --- |")
        for action, group in by_action:
            cats = ", ".join(f"`{c}`×{len(g)}"
                             for c, g in _group(group, lambda v: v.category))
            lines.append(f"| {ACTION_TITLE.get(action, action)} | {len(group)} | {cats} |")
        lines.append("")

    unclassified = [v for v in verdicts if v.action == "unknown"]
    if unclassified:
        lines.append("### 4.3 需人工确认（日志里没有异常块）")
        lines.append("")
        lines.append(f"{len(unclassified)} 个算子无法从日志归因：结果行有 FAILED（`row_failed`，"
                     "看该 shape 的测量日志），或压根没有结果行（`no_rows`：收集中断、harness 自身用例，"
                     "或测试没走到 benchmark）。")
        lines.append("")
        for verdict in unclassified[:20]:
            lines.append(f"- `{verdict.dir_key}`：{verdict.category}，rc={verdict.return_code}，"
                         f"看 `{verdict.dir_key}/run.log` 尾部")
        if len(unclassified) > 20:
            lines.append(f"- … 其余 {len(unclassified) - 20} 个见 triage.csv")
        lines.append("")

    text = dismiss_text(fails)
    lines.append("## 5. dismiss 候选（可直接追加到 `ops/dismiss.txt`）")
    lines.append("")
    if text:
        lines.append("仅包含「补环境 / 上游问题 / 平台不适用」三类；确认后粘贴，并按实际原因改注释。")
        lines.append("")
        lines.append("```")
        lines.extend(text.rstrip("\n").splitlines())
        lines.append("```")
    else:
        lines.append("无。")
    lines.append("")
    lines.append("注意 `ops/dismiss.txt` 只对 `--all` 生效，`--ops-file` 不会过滤它；"
                 "同名算子会连同它的其他 `op_file` 一起被排除。")
    lines.append("")
    return "\n".join(lines)


def write_csv(verdicts: List[OpVerdict], path: Path) -> Path:
    fields = ["repo", "op", "op_file", "status", "return_code", "category", "action",
              "signature", "why", "skip_reason", "rows_total", "rows_failed", "evidence", "notes"]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for verdict in verdicts:
            writer.writerow({
                "repo": verdict.repo, "op": verdict.op, "op_file": verdict.op_file,
                "status": verdict.status, "return_code": verdict.return_code,
                "category": verdict.category, "action": verdict.action,
                "signature": verdict.signature, "why": verdict.why,
                "skip_reason": verdict.skip_reason,
                "rows_total": verdict.rows_total, "rows_failed": verdict.rows_failed,
                "evidence": " ⏎ ".join(verdict.evidence[:6]),
                "notes": verdict.notes,
            })
    tmp.replace(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch_dir", type=Path, help="run_batch.py batch dir (or a single-op run dir)")
    p.add_argument("--md", type=Path, default=None,
                   help="Markdown report (default <batch>/triage.md)")
    p.add_argument("--csv", type=Path, default=None,
                   help="per-op CSV (default <batch>/triage.csv)")
    p.add_argument("--top", type=int, default=0,
                   help="max ops listed per group in the tables (0 = all)")
    p.add_argument("--evidence-lines", type=int, default=12,
                   help="exception lines shown per group (default 12)")
    p.add_argument("--dismiss-out", type=Path, default=None,
                   help="also write the ops/dismiss.txt candidate block here")
    args = p.parse_args()

    if not args.batch_dir.is_dir():
        raise SystemExit(f"no such dir: {args.batch_dir}")
    batch_dir = args.batch_dir.resolve()
    verdicts = scan_batch(batch_dir)

    texts: List[str] = []
    for verdict in verdicts[:40]:
        if verdict.log:
            log = Path(verdict.log) / "run.log"
            if log.is_file():
                texts.append(_read(log))
    info = platform_info(texts)

    md_path = args.md or batch_dir / "triage.md"
    csv_path = args.csv or batch_dir / "triage.csv"
    md_path.write_text(render_markdown(verdicts, batch_dir, info, args.top, args.evidence_lines),
                       encoding="utf-8")
    write_csv(verdicts, csv_path)

    counts = _counts(verdicts)
    print(f"[triage] {counts.get('PASS', 0)} PASS / {counts.get('SKIP', 0)} SKIP / "
          f"{counts.get('FAIL', 0)} FAIL / {counts.get('TIMEOUT', 0)} TIMEOUT", file=sys.stderr)
    for category, group in _group([v for v in verdicts if v.status == "FAIL"], lambda v: v.category):
        print(f"[triage]   {category:<16} {len(group):>4}  {group[0].action}", file=sys.stderr)
    print(f"[triage] report: {md_path}", file=sys.stderr)
    print(f"[triage] csv:    {csv_path}", file=sys.stderr)
    if args.dismiss_out:
        text = dismiss_text([v for v in verdicts if v.status == "FAIL"])
        args.dismiss_out.write_text(text, encoding="utf-8")
        print(f"[triage] dismiss candidate: {args.dismiss_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
