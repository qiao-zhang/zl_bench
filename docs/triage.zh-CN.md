# A 侧 record 之后的 SKIP/FAIL 归因与报告

本文说明如何在 A 侧 record（`run_batch.py` 跑完一批算子）之后，把 SKIP 和 FAIL 快速归类到根因、
生成一份可归档的报告，并把「不是我们的问题」的部分收敛成 `ops/dismiss.txt` 条目。

归因**只读批次目录里已有的产物**，不依赖任何厂商工具（`nvidia-smi` / `ppu-smi` / `mx-smi` 都不调用），
所以同一份批次目录在 NVIDIA、PPU、MetaX 上得到同一份报告；换成别的加速卡也不用改脚本。
所有命令都在 `zl_bench/` 目录下执行。

---

## 1. 什么时候用

| 时机 | 做什么 |
| --- | --- |
| A 侧 record 刚跑完 | 先看 `FAIL`/`SKIP` 的构成，再决定这批数据能不能进 A/B |
| 换了上游 commit 或平台 | 重建清单前，先弄清上一轮失败里哪些是清单问题、哪些是平台问题 |
| 准备写 `ops/dismiss.txt` | 直接用报告第 5 节的候选块 |
| 修完环境后 | `--resume` 只重跑失败集，再跑一次归因确认收敛 |

它**不做**的事：不做 A/B 判定（那是 `ab_run.py` / `ab_compare.py`），不修改批次目录里的产物，
不重跑任何算子。

---

## 2. 输入：报告读哪些产物

| 产物 | 位置 | 用途 |
| --- | --- | --- |
| `ops_status.csv` | 批次根目录 | 每个算子的 `status` / `return_code` / `notes`，是报告的主表 |
| `run.log` | `<batch>/<repo>/<run_name>/` | **主要证据**：pytest 的完整异常块、`-rsx` 的 skip 原因、编译器诊断 |
| `run.attempt<N>.log` | 同上 | `dtype_rescue` 前一次尝试的日志；当前 `run.log` 没有异常块时用它 |
| `summary.csv` | 批次根目录 | 报告不读它：结果行的 `row_status` 由 `batch_summary.parse_run_log` 从 `run.log` 现算，所以缺了也不影响 |

**为什么以 `run.log` 为主证据。** pytest 会把异常消息的**每一行**都加 `E   ` 前缀，所以
`RuntimeError: PassManager::run failed` 只是壳，MLIR pass 的 `error:` / `loc("...")`、`CompilationError`
的源码片段和 `^` 标记都在紧随其后的行里；Triton 打到 stderr 的诊断（`error:`、`LLVM ERROR:`）甚至
**打印在异常行之前**且不带 `E` 前缀。只抓第一条 `E` 行（`grep -m1`）正好把这些全丢掉——
这正是「失败原因看不出来」的常见原因。

---

## 3. 快速开始

```bash
cd zl_bench
python triage_report.py runs/metax/all_1104/base
```

默认在批次目录里写两个文件：

```
runs/metax/all_1104/base/
├── triage.md     # Markdown 报告，可直接归档/贴进 issue
└── triage.csv    # 每个算子一行，便于二次统计
```

终端同时打印每个根因类别的算子数，用来扫一眼构成：

```
[triage] 1543 PASS / 115 SKIP / 138 FAIL / 0 TIMEOUT
[triage]   compiler_pass       30  issue
[triage]   runtime_address     26  issue
[triage]   smem_limit          10  config
...
```

### 3.1 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `batch_dir` | 必填 | `run_batch.py` 的批次目录，也可以是单个算子的 run 目录 |
| `--md` | `<batch>/triage.md` | Markdown 报告路径 |
| `--csv` | `<batch>/triage.csv` | 逐算子 CSV 路径 |
| `--top` | `0`（全部列出） | 表格里每个类别最多列几个算子；算子多的批次用 `--top 5` 让报告更好读 |
| `--evidence-lines` | `12` | 每组展示的异常行数 |
| `--dismiss-out` | 不写 | 额外把 dismiss 候选块写到一个文件 |

```bash
# 算子多、只想看构成
python triage_report.py runs/big_batch --top 5

# 顺便产出可粘贴的 dismiss 候选
python triage_report.py runs/big_batch --dismiss-out /tmp/dismiss_candidate.txt
```

---

## 4. 报告怎么读

报告固定五节，编号不随内容变化（没有 FAIL 时第 2 节写「本批次没有 FAIL」），便于跨批次对比。

### 第 1 节：概览

状态计数 + FAIL 的 `return_code` 分布。返回码本身就是第一层归因：

| `return_code` | 含义 | 报告里的去向 |
| --- | --- | --- |
| `0` 但 `status=FAIL` | pytest 正常退出，但结果行有 `FAILED`，或压根没有结果行 | `row_failed` / `no_rows` |
| `1` | 普通测试失败，日志里有 traceback | 按异常内容归类 |
| `2` | 收集中断 / pytest 内部错误（多为 import 期就炸） | 多为 `no_rows`，看日志前 30 行 |
| `≥128` 或负数 | 进程被信号终止（`134`=SIGABRT、`139`=SIGSEGV） | `crash`，日志里不会有 traceback |

`notes` 列的取值同样值得先扫一遍：`retry_rc=1->1` 说明重试过且两次返回码一致（确定复现，不用再猜偶发）；
`dtype_rescue` 说明最后一次是不带 dtype 限制的重跑；`partial_skip=N` 表示部分结果行是 SKIP。

### 第 2 节：FAIL 根因分组

一张表加每组证据。分组按**根因类别**，不是按算子名，所以 30 个同族算子通常塌成 1 行。
每组给出：建议动作、算子清单、样例的完整异常块、以及一条可直接复制的复现命令。
组内若有多条不同的归一化签名，会列出前 3 条及计数——同类别不等于同根因，签名不同就值得分别抽样。

### 第 3 节：SKIP 原因分组

按 `-rsx` 打出的 `SKIPPED [n] path:line: reason` 聚合。SKIP 基本都是上游 `skipif`（缺参考实现、
缺依赖、架构/厂商限制），不需要处理，但要与 FAIL 分开统计，否则失败率会被高估。

### 第 4 节：需要人工处理

- **4.1 优先深挖**：`crash`（非法访存、信号终止）和 `accuracy`（结果不符）——这两类最可能是真缺陷，
  优先级高于性能排查。
- **4.2 按建议动作汇总**：一张表看清「要提 issue 的有几类、要装依赖的有几个、要调 config 的有几个」。
- **4.3 需人工确认**：`row_failed`（有 `FAILED` 结果行）和 `no_rows`（没有任何结果行）。

### 第 5 节：dismiss 候选

把 `env` / `upstream` / `dismiss` 三类动作的算子按类别分组，生成可直接追加到 `ops/dismiss.txt` 的文本块，
带类别、原因和日期注释。

---

## 5. 归因规则

分类只匹配**错误消息关键词**，不认厂商名，所以规则跨平台通用。顺序敏感：先匹配的规则胜出，
具体模式必须排在泛化模式之前（例如 `libtriton` 排在 `has no attribute` 之前）。

| 类别 | 建议动作 | 匹配关键词（节选） | 处置 |
| --- | --- | --- | --- |
| `crash` | investigate | `illegal memory access`、`segmentation fault`、`sigsegv`、`sigabrt`、`core dumped` | 最小化到单 shape 复跑，抓 core |
| `accuracy` | investigate | `tensor-likes are not close/equal`、`mismatched elements`、`greatest absolute/relative difference` | 真问题，优先排查 |
| `smem_limit` | config | `out of resource: shared memory` | 换更小的 block / 更少的 stage，或从清单去掉 |
| `oom` | config | `out of memory`、`outofmemoryerror` | 缩小 shape 集；确认卡没被别的进程占用 |
| `invalid_config` | config | `invalid configuration argument`、`invalid launch`、`input shapes should have` | 检查 shape 集与 autotune 空间 |
| `runtime_address` | issue | `too large to fit in 32 bit` | 大小/地址按 32 位传递，同族合并成一个后端 issue |
| `compiler_pass` | issue | `passmanager::run failed` | 看异常前的 MLIR 诊断，再拿 ttir 离线复现 |
| `compiler_mlir` | issue | `internal triton mlir opt error` | 诊断在冒号后面 |
| `compiler_llir` | issue | `internal triton llir codegen error` | 多为后端不支持的 op/类型组合 |
| `compiler_kernel` | issue | `triton.compiler.errors.compilationerror`、`compilationerror:` | 看源码片段与 `^` 标记那一行 |
| `binding_mismatch` | env | `libtriton`、`no attribute 'make_` | Triton 与后端版本不一致 |
| `env_missing` | env | `not enabled for build`、`not compiled with`、`requires compiling pytorch with`、`cutlass not supported` | 装依赖或换构建 |
| `missing_module` | env | `modulenotfounderror`、`importerror:` | 装包 |
| `cpu_only` | dismiss | `expected a cuda device`、`expected out tensor to have device cpu` | 该算子是 CPU/mkldnn 路径 |
| `upstream_code` | upstream | `'function' object is not subscriptable`、`has no attribute`、`nameerror`、`jsondecodeerror`、`missing N required positional argument` | 与上游对齐，或确认版本错配 |
| `recursion` | investigate | `recursionerror` | 单独复现递归路径 |
| `row_failed` | unknown | —（`rc=0` 且有 `FAILED` 结果行） | 看该 shape 的测量日志 |
| `no_rows` | unknown | —（无异常块、无结果行） | 收集中断、harness 自身用例，或测试没走到 benchmark |
| `unclassified` | unknown | —（有异常块但没命中任何规则） | 需要新增一条规则 |

**新增规则**：在 `triage_report.py` 的 `RULES` 里加一条 `_rule(category, action, 正则, 处置说明)`，
并把类别补进 `ACTION_TITLE`；再在 `tests/test_triage_report.py::test_classify_maps_shell_messages_to_root_causes`
里加一行用例。规则只依赖报错文本，不需要平台分支。

签名会做归一化（去掉 `0x…` 地址、`torch.Size([…])`、三位以上的数字），这样不同 shape 的同一个问题
落到同一个签名里；`32 bit`、`M >= 1` 这类小数保留，不影响可读性。

---

## 6. 跨平台

脚本本身不含任何平台分支，因为归因对象是**日志文本**，不是硬件。平台只影响日志里出现哪些消息：

- 报告头部的后端、编译器、OS、Python 取自 `run.log` 开头的
  `>>> env: FLAGTREE_BACKEND=…`、`>>> triton=… (git …)`、`platform … -- Python …`，缺哪行就不显示哪项。
- 同一个类别在不同平台上可能数量差别很大，属于正常现象，例如：
  - `smem_limit` 在 SM 较小的加速卡上更常见（`Hardware limit: 65536` 这类上限写在消息里）；
  - `binding_mismatch` / `env_missing` 取决于该平台的构建选项和依赖包；
  - `cpu_only`（mkldnn / 量化算子）在任何加速卡上都一样不适用。
- 算子在 PPU / MetaX 上能不能被 pytest 收集，与清单有关：`ops/inventory_*.csv` 是在 NVIDIA 上生成的，
  换平台后建议用 `gen_inventory.py --verify-collect` 重建（见 README「跟进上游更新」），
  这样不可收集的算子会落到 `collected=no`，`--all` 不再选它们。

---

## 7. 四类「壳错误」怎么深挖到真原因

这几类错误的第一行只是外壳，真正的诊断在其后（或之前）。下面给的是不依赖脚本的手工方法，
用来在报告指出的样例算子上进一步定位。

### 7.1 手工抓完整异常块

```bash
k=runs/metax/all_1104/base/main/sgn

# 首个失败块 + 后续 25 行（含不带 E 前缀的诊断行）
awk '/^E +/{f=1} f{print; if(++n>=25) exit}' "$k/run.log"

# 往前看 60 行：MLIR/LLVM 诊断通常打印在 RuntimeError 之前
n=$(grep -n 'PassManager::run failed' "$k/run.log" | head -1 | cut -d: -f1)
sed -n "$(( n>60 ? n-60 : 1 )),$((n+10))p" "$k/run.log"
```

### 7.2 各类的诊断在哪

| 外层消息 | 真诊断位置 |
| --- | --- |
| `RuntimeError: PassManager::run failed` | 异常**之前**的 `error:` / `loc("…")` / `LLVM ERROR:` 行；都没有则说明是 C++ 侧 assert，需要离线复现 |
| `RuntimeError: Internal Triton mlir opt error:` | 冒号后面（常换行）的 MLIR 诊断 |
| `RuntimeError: Internal Triton llir codegen error:` | 后面跟的 LLVM 报错（`Cannot select`、不支持的 intrinsic 等） |
| `triton.compiler.errors.CompilationError: at N:M:` | kernel 源码片段 + `^` 指向出错行 + 真正的 Python 异常 |

同族算子报同一条外壳消息时，**只深挖一个**就够了：先挑最简单的一元算子复现，定性后再回到报告里
按类别批量处置。

### 7.3 单算子复现（保留完整日志与 IR）

```bash
FLAGGEMS_DIR=../FlagGems OP=sgn OP_FILE=sgn OUT_DIR=/tmp/dbg_sgn bash run_pytest.sh
tail -120 /tmp/dbg_sgn/run.log
```

产物里 `<OUT_DIR>/ttir/<CACHE_HASH>/<kernel>.ttir` 与同名 `.json`（编译参数）会留下来，可用于离线重编译：

```bash
python ir_gate.py /tmp/dbg_sgn --base "$TRITON_PYTHON" --new "$TRITON_PYTHON" \
  --stage ttgir --out /tmp/dbg_gate
```

其中 `$TRITON_PYTHON` 是该构建的 `python/` 目录（同一构建自比即可，目的是强制重编译并暴露诊断）。

---

## 8. 从归因到收敛

### 8.1 写 dismiss

报告第 5 节给出的块已经按「类别 + 原因 + 日期」分好组，确认后追加到 `ops/dismiss.txt`。
格式规则见该文件开头：一行一个 `op`（main 仓）或 `repo:op`（vllm 仓），`#` 开头是注释。

两条容易踩的坑：

- `--dismiss` **只对 `--all` 生效**，用 `--ops-file` 重跑时不会过滤它；
- 排除粒度是 `(repo, op)`，同一个算子若有多个 `op_file`，会被一起排除。

### 8.2 只重跑失败集

```bash
# 失败清单，带 @op_file，避免同名算子歧义
awk -F, 'NR>1 && $6=="FAIL"{print $1":"$2"@"$3}' runs/metax/all_1104/base/ops_status.csv \
  > /tmp/fail_ops.txt

python run_batch.py --ops-file /tmp/fail_ops.txt --resume \
  --batch-dir runs/metax/all_1104/base --busy-gpu ignore
```

`--resume` 复用 PASS/SKIP，只重跑 FAIL/TIMEOUT，所以「修一批 → 归因一次」可以反复迭代。
每轮结束再跑一次 `triage_report.py`，看类别数量是否收敛。

### 8.3 收敛的判据

| 情况 | 处置 |
| --- | --- |
| 报告里只剩 `env` / `upstream` / `dismiss` / `cpu_only` | 写进 `ops/dismiss.txt`，可以进 A/B |
| 还有 `issue` 类（编译器/运行时同根因） | 先提 issue，A/B 时按类别排除，并在报告里注明 |
| 还有 `investigate`（crash / accuracy） | **不要进 A/B**：A/B 只比延迟，算错的算子照样出计时行 |
| 还有 `unknown` | 逐个看 `run.log` 尾部，必要时补归因规则 |

---

## 9. 与 A/B 的衔接

A 侧的 `record` 批次同时是后面 A/B 的记录源，所以归因要在**进 A/B 之前**做完：

```bash
# 1. record（A 侧）
python run_batch.py --all --gpus 0,1,2,3,4,5,6,7 --dtypes bfloat16 \
  --batch-dir runs/<platform>/<tag>/base

# 2. 归因，把失败构成弄清楚
python triage_report.py runs/<platform>/<tag>/base

# 3. 过编译门控（只测 kernel 真变了的算子）
python ir_gate.py runs/<platform>/<tag>/base --base "$BASE_PY" --new "$NEW_PY" --out runs/<tag>/gate

# 4. 重复测量 A/B
python ab_run.py --record runs/<platform>/<tag>/base --ops-file runs/<tag>/gate/changed_ops.txt \
  --a-pythonpath "$BASE_PY" --b-pythonpath "$NEW_PY" --gate runs/<tag>/gate --out runs/<tag>/ab
python ab_compare.py runs/<tag>/ab --gate runs/<tag>/gate
```

报告里的分组结果还有一个用途：`ab_compare.py` 的判定只对**成功产出计时行**的算子有效，
把已知失败集固定下来（dismiss 或按类别排除），才能保证两轮 A/B 的算子集合一致、可比。

---

## 10. 局限

- **信号终止（`rc≥128`）没有 traceback**：只能归到 `crash`，具体越界点要单独复现。
- **`rc=0` 且没有异常块**：报告只能区分「有 `FAILED` 结果行」和「完全没有结果行」，具体原因要看该 shape 的测量日志。
- **同一类别不等于同一根因**：分组是基于归一化签名和关键词的启发式判断。30 个算子落在 `compiler_pass`
  通常确实是一个 pass 的问题，但仍建议抽 1–2 个按第 7 节确认，再合并成一个 issue。
- **日志被截断时**（例如 `--op-timeout` 杀掉进程），异常块可能不完整，报告会显示能拿到的部分。
- **报告不判断性能**：`PASS` 只表示跑通并产出了计时行，不代表数字可信（多卡并发、CUDA graph 回退都会影响）。
