# RLC Enhance A/B 测试指南（PPU）

本文说明如何用 `ops/key_ops.txt` 里的算子，在同一份 FlagTree 构建上比较
`FLAGTREE_RLC_ENHANCE=0`（A 侧，基线）与 `FLAGTREE_RLC_ENHANCE=1`（B 侧，开启 RLC）
的 kernel 延迟差异。所有命令都在 `zl_bench/` 目录下执行。

---

## 1. 测试口径

- **A/B 是什么**：用 A 侧环境录制一次基线（record），之后两侧 replay 同一份 record，也就是跑同一批 shape、同一组 autotune config；每个（算子, 轮次）在同一张卡上先后跑 A 和 B，奇数轮先 A、偶数轮先 B，以抵消卡间偏差和顺序效应。
- **结论怎么读**：每个结果行给出`gap = median(A) / median(B) - 1`，`gap > 0` 表示 **B（RLC 开）更快**。
- **测的不是什么**：不是两边各自最优 config 的对比，而是“同一 config 下，只改`FLAGTREE_RLC_ENHANCE` 这一个编译开关”的效果。
- **统计判定**：默认至少 6 轮样本，双重条件（差距阈值 + Mann-Whitney 检验），见第 4 节；不足则判 `undecided` 并可用 `--until-decided` 自动补轮。

### 1.1 `FLAGTREE_RLC_ENHANCE` 的作用方式

| 事实                                                                     | 位置                                                                                                                                                           |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PPU 后端默认 **False**，NVIDIA 后端默认 True                       | `FlagTree/python/triton/knobs.py`（`ppu_knobs.rlc_enhance` / `nvidia_knobs.rlc_enhance`）                                                                |
| 编译期 knob：`env_bool` 在 import 时求值，进程内改 `os.environ` 无效 | 同上                                                                                                                                                           |
| 作用于 `make_ttgir` 的三次 layout-conversion pass                      | `FlagTree/third_party/ppu/backend/compiler.py`：`add_remove_layout_conversions(pm, knobs.ppu.rlc_enhance, enable_rlc_enhance_store_layout_remat=True)` ×3 |

因此：

- 必须在**启动子进程之前**用环境变量注入；本项目通过 `run_batch.py --env` /
  `ab_run.py --a-env/--b-env` 完成，不要试图在测试内部改。
- 本 A/B 在 PPU 上跑，A 侧必须显式写 `=0`（默认值就是关，但显式写可防止环境里残留变量），
  B 侧写 `=1`。
- 子开关：RLC 的四个子特性（`enable-rlc-enhance-cost-based`、
  `enable-rlc-enhance-backward-propagation`、`enable-rlc-enhance-small-component-solving`、
  `enable-rlc-enhance-store-layout-rematerialization`）在 `python/src/passes.cc` 里默认都是 `True`；
  自 FlagTree commit `17288c4b4`（`enable rlc enhance for ppu`，2026-10-26）起，PPU 后端
  `third_party/ppu/backend/compiler.py` 三处调用都是
  `add_remove_layout_conversions(pm, knobs.ppu.rlc_enhance, enable_rlc_enhance_store_layout_remat=True)`，
  即 B 侧是 **5/5 全开**（更早的构建若仍硬编码 `False`，B 侧只生效 4/5，报告里需注明这一口径）。
- 生效前提：这四个子开关只有在构建时定义了 `__FLAGTREE_RLC_ENHANCE__` 才会被
  `passes.cc` 的 `#ifdef` 分支透传（`FLAGTREE_BACKEND=ppu` 时由 `cmake/FlagTreeOptions.cmake`
  自动加这个宏）。缺宏时参数被整体丢弃、RLC 完全不生效，A/B 两侧会编出同一份二进制——门控里
  该算子全是 `unchanged`，正好可以用来自检这份构建是否带 RLC。

---

## 2. 前置条件

| 项       | 要求                                                                                               |
| -------- | -------------------------------------------------------------------------------------------------- |
| 工作目录 | `zl_bench` 目录                                                                     |
| FlagTree | 已构建、可 `import triton`；A/B 两侧是**同一个构建**，只切环境变量                         |
| 算子仓库 | `../FlagGems` 与 `../FlagGems-vllm`（可用 `FLAGGEMS_MAIN_DIR` / `FLAGGEMS_VLLM_DIR` 覆盖） |
| 加速卡   | PPU；`ppu-smi` 可用（或设 `ZL_BENCH_SMI`），且每张卡在算子启动前必须空闲                       |
| Python   | 依赖见 `requirements.txt`；仅生成 xlsx 报告时需要 `openpyxl`                                   |
| 算子清单 | `ops/key_ops.txt`（56 个 `# key` 分组，122 条 `repo:op@op_file`）                        |

`ops/key_ops.txt` 的约定：

- `#` 开头的行（分组标题与备注）会被加载器跳过；
- 每个非注释行必须是 `[repo:]op[@op_file]`，**行尾不要加注释**，否则 `op_file` 会带上注释文本；
- 其中 `vllm:sparse_attn_triton`（inventory 标 `marked_skip=yes`）和两条
  `vllm:fp8_fp4_*`（`collected=no`）因为是显式列出，仍会被执行；解读结果时留意这一点。

### 环境自检

```bash
TRITON_PYTHON=$(python -c 'import triton, os; print(os.path.dirname(os.path.dirname(triton.__file__)))')
echo "$TRITON_PYTHON"
```

结果可能为：

```shell
/usr/local/lib/python3.12/site-packages
```

因为待测试FlagTree已经安装了。

确保主开关能够被读到：

```shell
FLAGTREE_RLC_ENHANCE=0 python -c "from triton import knobs; print('A side rlc_enhance =', knobs.ppu.rlc_enhance)"
FLAGTREE_RLC_ENHANCE=1 python -c "from triton import knobs; print('B side rlc_enhance =', knobs.ppu.rlc_enhance)"

ppu-smi -L                 # 确认卡号（不同版本也支持 --list-gpus / --list-ppus）
wc -l ops/key_ops.txt  # 259 行
```

上面的两条 `python -c` 应分别打印 `False` 与 `True`。若第二个仍是 `False`，说明`PYTHONPATH` 指向了另一套 triton，或环境里已有同名变量覆盖。

另外，运行：

```bash
python - <<'PY'
import pathlib, triton
root = pathlib.Path(triton.__file__).parent / "backends"
print("triton:", triton.__version__, triton.__file__)
for p in sorted(root.glob("*/compiler.py")):
    text = p.read_text(errors="replace")
    print(f"{p.parent.name:8} remat=True:{'enable_rlc_enhance_store_layout_remat=True' in text}")
PY
```

应该能看到：

```Shell
ppu      remat=True:True
```

---

## 3. 完整流程

### 设置环境

```bash
cd /root/qiao/zl_bench
GPUS=0,1,2,3,4,5,6,7 # 需要用ppu-smi来确定
source /usr/local/PPU_SDK/envsetup.sh
export ZL_BENCH_SMI=ppu-smi
```

### A 侧录制基线（=0）

```shell
python run_batch.py \
  --ops-file ops/key_ops.txt \
  --shape-file shape/all_ops_multi.yaml \
  --env FLAGTREE_RLC_ENHANCE=0 \
  --gpus "$GPUS" \
  --dtypes "bfloat16 float16 float32 complex64" \
  --op-timeout 7200 \
  --batch-dir runs/rlc_ppu_key_base
```

关键参数：

| 参数                                              | 说明                                                                      |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| `--ops-file ops/key_ops.txt`                | 选中的 122 条 op；默认 repo 前缀已在文件里写明                            |
| `--shape-file shape/all_ops_multi.yaml`         | kernel 变体取决于 shape，不传时默认 `shape/all_ops_single.yaml`，单 shape 记录看不到只用其他 shape 的变体；但多 shape 会显著拉长 record 与 A/B 时间                           |
| `--env FLAGTREE_RLC_ENHANCE=0`                  | 传给每个算子子进程（`run_pytest.sh`）                                   |
| `--gpus "$GPUS"`                                        | 卡号，按 `ppu-smi` 的输出；不传则取 `CUDA_VISIBLE_DEVICES` 或自动枚举 |
| `--dtypes "bfloat16 float16 float32 complex64"` | dtype 白名单；可与 A/B 阶段的 `--dtypes` 对应，两侧必须一致             |
| `--op-timeout 7200`                             | 多 shape 集建议调大（默认 1800）                                          |
| `--batch-dir runs/rlc_ppu_key_base`                 | record 目录，后续 `--record` 指向它                                     |

- record 必须用 **A 侧** 的环境生成
- 每个算子一个独立进程：CUDA crash 只丢该算子；FAIL/TIMEOUT 可 `--resume` 重跑。

这一步跑完后，先检查 `runs/rlc_ppu_key_base/ops_status.csv`：大量 FAIL/TIMEOUT 时不要进入 A/B，先解决环境问题（缺依赖、显存、`dismiss.txt` 未覆盖的上游 skip 等）。

### 可选：IR 门控（同一构建，两个开关值；不需要 GPU）

```shell
mkdir -p /tmp/zl_empty_pythonpath

python ir_gate.py runs/rlc_ppu_key_base \
  --base /tmp/zl_empty_pythonpath \
  --new /tmp/zl_empty_pythonpath \
  --base-env FLAGTREE_RLC_ENHANCE=0 \
  --new-env FLAGTREE_RLC_ENHANCE=1 \
  --out runs/rlc_ppu_key_gate
```

- 两侧是**同一构建**，区别只在 `--base-env` / `--new-env`；`--base` / `--new` 给的是空目录，是为了回落到已安装的 Triton，而不是 FlagTree 源码目录（作为 `PYTHONPATH`）。
- 不需要 GPU：把记录下来的 TTIR 离线编译到后端二进制（PPU 上是 hgbin），比较二进制和 launch 元数据（shared memory、warp 数等）。
- 产物：`variants.tsv`（逐变体）、`report.tsv`（仅变化/未知）、`rows.tsv`（逐结果行）、`changed_ops.txt`（有变化行的算子）、`base/` 与 `new/`（重编译的 TTGIR，可 diff）。
- 快速初看可用 `--stage ttgir`，但 TTGIR 相同不代表二进制相同，正式判定用默认 `binary`。
- 门控的价值：大多数 kernel 在开关前后编译成同一个二进制，这些行不可能有性能差异，在 `ab_compare.py --gate` 里作为**对照组**，用来实测判定规则的误报率。如果跳过当前步时，第 3、4 步都去掉 `--gate`；代价是没有“kernels 相同”的对照组，判定更保守也更容易混入噪声。

### 多轮 A/B

```shell
python ab_run.py --record runs/rlc_ppu_key_base \
  --ops-file ops/key_ops.txt \
  --a-env FLAGTREE_RLC_ENHANCE=0 \
  --b-env FLAGTREE_RLC_ENHANCE=1 \
  --a-name RLC-enhance-off \
  --b-name RLC-enhance-on \
  --gpus "$GPUS" \
  --dtypes "bfloat16 float16 float32 complex64" \
  --rounds 6 --until-decided --max-rounds 16 --step 2 \
  --gate runs/rlc_ppu_key_gate \
  --op-timeout 7200 \
  --out runs/rlc_ppu_key_ab
```

| 参数                        | 说明                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `--record`                | 必填，第 1 步的批次目录（或单个算子 run 目录）                                          |
| `--ops-file`              | 与 `--ops` 二选一；这里直接复用同一份 key 文件                                          |
| `--a-env` / `--b-env`   | 两侧各自的编译开关，可重复给多个 `K=V`                                                  |
| `--a-name` / `--b-name` | 报告里的显示名                                                                            |
| `--rounds 6`              | 保证每个算子有第 1..6 轮                                                                  |
| `--until-decided`         | 第 4.4 步的分析规则下仍有 `undecided` 行时，每轮补 `--step` 轮，直到 `--max-rounds` |
| `--gate`                  | 只影响 `--until-decided` 的补轮对象：只补“有 undecided 变化行”的算子                  |
| `--op-timeout`            | 单次运行超时，默认 3600 秒                                                                |
| `--out`                   | 输出目录；**用同一个 `--out` 重跑即断点续跑**                                     |

注意：

- 不要传 `--shape-file`：不传时两侧沿用 record 存档的 `shapes.yaml`。
- `--dtypes` 由这一个参数同时作用于两侧，不存在“两侧不一致”的问题；但建议与 record 阶段保持一致，否则 record 的 config 覆盖范围与本次结果行集合可能对不上（配对键是`(op_name, dtype, row)`，dtype 不同就是另一批行）。
- `--add-rounds N` 与 `--until-decided` 不能同时用；前者用于第 4.5 步的复测。
- 每次运行都是新进程，`run_pytest.sh` 为每次运行准备独立的 `TRITON_CACHE_DIR`（`<run>/.triton_cache`，结束后删除），因此 A/B 不会互相命中编译缓存。
- 如果两侧 `--a-env` / `--b-env` 完全相同，`ab_run.py` 会提示“both sides are identical, this measures the noise floor”，此时结果就是噪声底。

`ab.json` 会记录 record、两侧 env/PYTHONPATH、`--dtypes`、`--shape-file`、`--use-flagtune`；
用同一 `--out` 续跑时这些值必须一致，否则直接拒绝。


### 判定与报告

```shell
python ab_compare.py runs/rlc_ppu_key_ab --gate runs/rlc_ppu_key_gate -v \
  --csv runs/rlc_ppu_key_ab/rows.csv
```


| 参数                    | 说明                                                        |
| ----------------------- | ----------------------------------------------------------- |
| `--gate`              | 把两侧二进制相同的行标为对照组（`unchanged`），不计入结果 |
| `-v`                  | 列出每个 `faster` / `slower` / `undecided` 行         |
| `--csv`               | 写出逐行统计，供后续筛选或 `rlc_report.py` 使用           |
| `--rounds LO-HI`      | 只用指定轮次（例如 `--rounds 1-6`）                       |
| `--recheck-list FILE` | 写出含“弱结果”的算子清单，供复测                          |
| `--confirm LO-HI`     | 复测轮次；先用其余轮次判定，再用复测轮次单独确认            |

输出中的 `gap`、`p`、`verdict`、`flags` 含义见后。没有 `--gate` 时，所有行都当作结果，`unchanged` 对照组与误报率一栏为空。


### 弱结果复测（可选，但建议）

差距小、噪声大或被 `do_bench` 计时的行，是用同一批样本“既挑出来又判定”的，会被高估。
按三步处理：

```bash
# 1) 找出弱结果算子
python ab_compare.py runs/rlc_ppu_key_ab --gate runs/rlc_ppu_key_gate \
  --recheck-list runs/rlc_ppu_key_ab/recheck.txt

# 2) 给它们补全新轮次（其余 side 参数必须与首次完全一致）
python ab_run.py --record runs/rlc_ppu_key_base \
  --ops-file runs/rlc_ppu_key_ab/recheck.txt \
  --a-env FLAGTREE_RLC_ENHANCE=0 \
  --b-env FLAGTREE_RLC_ENHANCE=1 \
  --a-name RLC-enhance-off --b-name RLC-enhance-on \
  --gpus "$GPUS" \
  --dtypes "bfloat16 float16 float32 complex64" \
  --gate runs/rlc_ppu_key_gate \
  --op-timeout 7200 \
  --out runs/rlc_ppu_key_ab --add-rounds 6

# 3) 用 ab_run 打印的范围做 confirm（示例为 15-20，以实际输出为准）
python ab_compare.py runs/rlc_ppu_key_ab --gate runs/rlc_ppu_key_gate \
  --confirm 15-20 --csv runs/rlc_ppu_key_ab/rows.csv
```

弱结果的定义：`|gap| < --weak-gap`（默认 2%）、或 `|gap|` 小于两侧样本变异系数的 2 倍、或该表由普通 `do_bench` 计时。被 confirm 的 `faster`/`slower` 行用全部轮次报告 gap；未复现的行退回 `undecided`。

### 可选：xlsx 报告

`rlc_report.py` 可以把 `rows.csv` 和门控结果整理成表格：

```bash
python rlc_report.py \
  --csv runs/rlc_ppu_key_ab/rows.csv \
  --gate runs/rlc_ppu_key_gate \
  --ab-dir runs/rlc_ppu_key_ab \
  --out rlc_ppu_key_report.xlsx
```

注意 `--sources` 默认指向 `ops/rlc_ops_sources.csv`（由 `rlc_ops.py` 从对比表生成），本次清单是手工维护的 `ops/key_ops.txt`，两者不配套；若不需要表行来源信息，保持默认即可，若混入了旧表的行，用 `--sources` 指向本次的来源文件或先补一份。

---

## 4. 判定规则

每个（结果行, 轮次, 侧）一个 latency 样本。对一行：

```
gap = median(A) / median(B) - 1        # > 0: B 更快
p   = 双侧 Mann-Whitney U 检验
```

| 条件                                          | verdict                                      |
| --------------------------------------------- | -------------------------------------------- |
| 任一侧样本数 < `--min-rounds`（6）          | `undecided`（继续补轮）                    |
| `abs(gap) <= --threshold`（0.005，即 0.5%） | `same`                                     |
| `p < --alpha`（0.01）                       | `faster`（gap > 0）/ `slower`（gap < 0） |
| `p >= --clear`（0.10）                      | `same`                                     |
| 其他                                          | `undecided`                                |

- Mann-Whitney 不假设延迟分布；样本 ≤ 8 且无并列值时用精确分布，否则用带并列校正的
  正态近似，与 `scipy.stats.mannwhitneyu` 默认行为一致，无需 scipy。
- `--gate` 给出的对照组（两侧二进制相同）如果出现 `faster`/`slower`，就是误报；
  它们占对照组的比例即该规则在本批数据上的实测误报率。
- `--until-decided` 会反复补轮，直到所有行判定或达到 `--max-rounds`（默认为14）。
- 一轮 A/B 的结果是“在该 record 的 config 下”的差异；要比较两侧各自最优 config，
  需另外处理 libtuner 缓存（`~/.flaggems/config_cache/TunedConfig_*.db`），
  两种口径不要混用。

---

## 5. 产物

```
runs/rlc_ppu_key_base/                  # record（A 侧）
  ops_status.csv                        # 每算子一行：状态/耗时/行数/备注
  summary.csv                           # 每个结果行一行
  main/<op>/  vllm/<op>/                # run.log、shapes.yaml、rows.tsv、
                                        # autotune_records/、ttir/、ttgir/
runs/rlc_ppu_key_gate/                  # 门控
  variants.tsv  report.tsv  rows.tsv  changed_ops.txt  errors.log
  base/  new/                           # 两侧重编译的 TTGIR，可 diff
runs/rlc_ppu_key_ab/                    # A/B
  ab.json                               # record 与两侧配置清单
  r1_a/ r1_b/ ... rN_a/ rN_b/           # 每轮每侧一个批次目录（结构同上）
  rows.csv                              # 逐行统计
  recheck.txt                           # 弱结果算子清单（如生成）
```

`run.log` 里两类标记会影响解读：

- `AUTOTUNE_REPLAY_FALLBACK`：record 里缺该 config，或它在新编译器下编译失败，
  该测量点改为现场调优（不再是同一 config）；`_no_evict` 后缀表示重试仍读到坏 config。
- `BENCHMARK_DIRECT_NO_CUDAGRAPH`：该调用无法 graph capture，改用普通 `do_bench` 计时，
  噪声更大，亚毫秒 kernel 尤其明显。

`ab_compare.py` 会在 `flags` 列标出这些情况；只有带 `op=... phase=...` 的动态标记才算一次回退，
插件启动横幅里提到的标记名不算。

**注意标记的粒度是「表」，不是「行」**：`ab_compare.nocg_tables()` 以 `(op_name, dtype)` 为单位判定，
表内只要有一次回退标记，整张表都会被标记，随后该表**所有结果行**都会带上 `do_bench`（`rows.csv` 的
`flags` / `timer` 列同理）。所以 `rlc_report.py` 备注写的是保守口径：

```text
do_bench 计时（表级标记覆盖 3/8 行；host 同步）
```

含义是「这条备注覆盖的 8 行里，有 3 行所在的表带回了回退标记」，**不是**「3 行真的回退、其余 5 行
仍是 CUDA graph 计时」：同表内一次回退就会让整表所有行都被计入（8 个 shape 只有 1 个回退时，组级备注
会显示 8/8）；只有在备注跨多张表汇总时（算子级备注会合并多个 dtype）才会出现小于 1 的分数。
要拿到真实回退行数，必须先让标记带上 shape/测量的身份（当前 marker 只有 `op=`/`phase=`/`reason=`），
仅调整备注计数并不能还原。

---

## 6. 注意事项

1. **两侧必须同一构建**：本 A/B 只切 `FLAGTREE_RLC_ENHANCE`。若要比较两个构建，
   改用 `--a-pythonpath` / `--b-pythonpath` 指向各自的 FlagTree `python/` 目录。
2. **record 必须用 A 侧环境**：B 侧 replay 它记录的 config；record 用错侧等于换了基线。
3. **`--dtypes` 两侧一致**，`--shape-file` 在 A/B 阶段不要传（沿用 record 的存档）。
4. **`USE_FLAGTUNE` 两侧必须相同**；`--use-flagtune` 默认 0。
5. **卡必须空闲**：默认 `--busy-gpu wait`，同卡上的其他进程会同时干扰计时与 autotune；
   PPU 上用 `ppu-smi` 判断，必要时 `ZL_BENCH_SMI=...`。
6. **编译缓存已隔离**：每次 run 使用独立的 `TRITON_CACHE_DIR` 并在结束后删除，
   不存在 A 侧产物被 B 侧复用的风险；不要手动把两侧指向同一个缓存目录。
7. **显式选中的 skip 项也会跑**：`vllm:sparse_attn_triton`（`marked_skip=yes`）与
   `vllm:fp8_fp4_*`（`collected=no`）在 `--ops-file` 里是显式条目，加载器不会过滤。
8. **注释位置**：`ops/key_ops.txt` 的备注是独立 `#` 行；若在 op 行尾加注释，
   `op_file` 会带上注释文本而解析失败。
9. **断点续跑**：用同一 `--out` 重跑 `ab_run.py`，已完成轮次保留，失败的重跑；
   `ab.json` 不一致会拒绝续跑。
10. **不要混用 `--add-rounds` 与 `--until-decided`**：先跑完 until-decided，再做复测。
11. **多 shape 会显著变慢**：record 与每轮 A/B 都要按 shape 数放大，记得同时调大
    `--op-timeout`（record 默认 1800，A/B 默认 3600）。

---

## 7. 规模与时间估算

`ops/key_ops.txt` 展开为 122 条 op（main 66、vllm 56）。

- 6 轮 A/B = 122 × 6 × 2 ≈ **1464 次单算子运行**。
- 单 shape、每算子 10–30 秒（主要是进程启动与 import）时：单卡约 4–12 小时；
  8 卡并行约 **0.5–1.5 小时**，`--until-decided` 可能再补 2–8 轮。
- 多 shape 集（如 `shape/all_ops_multi.yaml`）按 shape 数近似成倍增加；
  `chunk_gdn2` 这类慢算子可能单次就要数十分钟，`--op-timeout` 要给足。
