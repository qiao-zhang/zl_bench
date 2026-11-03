# Design notes

**English** | [简体中文](design.zh-CN.md)

How zl_bench gets comparable numbers out of the FlagGems benchmarks, and what to keep in mind when reading them. For usage, see the [README](../README.md).

- [Plugins](#plugins)
- [Why replay](#why-replay)
- [Timing](#timing)
- [IR dump](#ir-dump)
- [Statistics](#statistics)
- [Accuracy gate](#accuracy-gate)
- [Shape override](#shape-override)
- [Parallel sweep](#parallel-sweep)
- [Upstream assumptions](#upstream-assumptions)

## Plugins

`run_pytest.sh` loads these with `-p` from its `PLUGINS` array:

| Plugin | Role | Disable when |
|---|---|---|
| `_parallel_warmup_plugin` | with `PARALLEL_WARMUP_GPUS=N`, runs the autotune sweep on N GPUs ([Parallel sweep](#parallel-sweep)) | never needed: without the variable it does nothing |
| `_seed_plugin` | seeds random, numpy and torch so data-dependent ops (sort, topk, ...) get byte-identical inputs | never |
| `_shape_override_plugin` | makes the shape yaml win over upstream code that bypasses it ([Shape override](#shape-override)) | never |
| `_autotune_record_plugin` | records or replays the choices of Triton autotune and libtuner ([Why replay](#why-replay)) | never |
| `_cudagraph_plugin` | times with warmed-up CUDA graphs, falling back to `do_bench` with a marker ([Timing](#timing)) | to match someone else's plain `do_bench` numbers |
| `_pretty_report_plugin` | folds inputs that are identical across all rows of the result table and adds color; column titles and the `SUCCESS`/`FAILED` words stay as upstream prints them, so log parsing is unaffected | to match upstream's raw table verbatim |
| `_ir_dump_plugin` | records each compiled variant, its real use per shape and the kernels each result row ran, and writes `ttgir/`, `ttir/` and `rows.tsv` at exit ([IR dump](#ir-dump)) | never |

## Why replay

### Two tuning layers

| Mechanism | Stored in | After a compiler change | A/B risk |
|---|---|---|---|
| Triton `@triton.autotune` | process memory | every process sweeps again | timing noise can make A and B pick **different configs** |
| FlagGems `@libtuner` (incl. the FlagTune expanded space) | `~/.flaggems/config_cache/*.db` (sqlite, persists across processes) | **still valid**: table names hash only the kernel source and the config space | B reuses A's winner and measures "the compiler change under the old config" rather than "each side at its best" |

libtuner also stores a BenchmarkCache in the same db: the measured latency of every config of a sweep. With a warm db even a record run does not re-measure and picks the winner from stored latencies; "record sweeps live" only holds for a cold db.

### Magnitude

Config differences can outweigh the change under test. Two record runs of the same shapes and settings, a dozen minutes apart, can differ by around 10% in mean speedup, shifted in the same direction across shapes rather than randomly. Diffing their `autotune_records/*.json` usually shows whole steps, not tweaks: `num_stages`, `BLOCK_SIZE_*` and `num_warps` all jump. Record-mode numbers therefore only provide the configs for replay; a conclusion backed by record data alone is unverified.

### Each side at its best

The libtuner cache is invalidated by changes to the kernel source, `tune_configs.yaml`, the expand yaml or `USE_FLAGTUNE`, and by a new Triton major version or GPU model; zl_bench only clears its own `TRITON_CACHE_DIR`, never that db. To compare each side at its own best config, delete `~/.flaggems/config_cache/TunedConfig_*.db` or point `FLAGGEMS_DB_URL` at a throwaway file before each side. Leaving out `REPLAY_FROM` is not enough: with a warm cache both sides hit the same winner anyway. Both views are valid; state which one a conclusion uses.

### Record and replay details

- **Record** reads back the config of each call from `cache[key]` after `run()` instead of looking for new cache keys, since a warm libtuner cache never gets new keys. Single-config kernels, which skip the cache, record their only config.
- **Replay** writes the recorded config into the cache before every call, unconditionally: skipping keys that are already cached would leave a warm libtuner cache tuning on its own.
- Configs with a `pre_hook` (e.g. the TMA configs of Hopper mm): libtuner reattaches the hook to a replayed config by matching its kwargs, so these record and replay normally. Plain `@triton.autotune` has no such mechanism; those configs are not recorded and replay as `key_missing` fallbacks.
- **Fallback**: a key missing from the record, or a recorded config that fails to compile with the new compiler, falls back to live tuning and logs `AUTOTUNE_REPLAY_FALLBACK reason=...`. A failed config can be evicted from Triton's in-memory cache but not from libtuner's sqlite cache; the marker then ends in `_no_evict`, because the retry reads the same config again.
- To check that a record covers a kernel, look for its entry in `autotune_records/<op>.json`.

## Timing

### Warmup before capture

Upstream `--mode kernel` times with `triton.testing.do_bench`; `_cudagraph_plugin` replaces it with CUDA-graph timing that follows `triton.testing.do_bench_cudagraph`: a graph of back-to-back calls covering the 100 ms repetition budget, replayed 10 times, median per call. It measures kernel time without launch overhead. (Upstream's own `--mode cudagraph` neither warms up nor falls back, so it is not used.)

`do_bench_cudagraph` warms up for only 5 iterations, too few for ops whose first calls compile autotune configs, run libtuner selection or JIT-initialize. If that one-off work leaks into the captured graph or the first timed iteration, latency jitters between runs; on the M=1 multi-kernel path the first round can read about a third of steady state. The plugin therefore:

1. synchronizes, so inputs still being produced on the default stream are complete before the side-stream capture;
2. runs the op once and discards the result, absorbing compilation;
3. runs it for the caller's warmup budget (`Config.warm_up`), with the count derived from one steady-state run and clamped to 5–200;
4. runs it once more with torch's sync debug mode set to error, to find host synchronization ([Fallback](#fallback-to-do_bench));
5. captures and times.

If a new op still shows first-round jitter, raise the cap in `_warmup_before_capture` first.

### Graph size

Capture time grows linearly with the number of calls in the graph, and for some ops each call is expensive to capture: about 12 ms for `conv1d_padding`, so filling 100 ms with thousands of calls costs half a minute or more per measurement; recording its 13 fp16 shapes took 67 minutes uncapped and 30 minutes capped. The plugin therefore puts at most `GRAPH_MAX_CALLS` (default 1000) calls into one graph and reaches the budget by replaying the graph back to back within each of the 10 timed samples, so each sample still spans the same number of calls.

Back-to-back replays add nothing measurable from 250 calls per graph up: `abs` on 64 elements reads 1.03 µs per call at 250, 500 and 1000 calls per graph and uncapped, and 23.0 µs on 4096×4096 at every size. Below that, small graphs read tiny kernels too fast (0.86 µs at 100 calls), so keep the cap at 250 or more. The graph size itself shifts some ops slightly: `conv1d_padding` reads 45.9 µs per call at 500 calls per graph and 46.8 µs at 2000, as each call's outputs grow the graph's private memory pool. Both sides of an A/B use the same plan, so comparisons are unaffected; absolute numbers of such ops can differ from an uncapped `do_bench_cudagraph` by that much. `GRAPH_MAX_CALLS=0` restores the uncapped graph.

### Fallback to do_bench

Some ops cannot be captured: host syncs, dynamic allocations, autograd backward, illegal stream operations. They are timed with plain `do_bench` instead, and `run.log` gets `BENCHMARK_DIRECT_NO_CUDAGRAPH op=... phase=... reason=...`, where `phase` is the step that failed (`sync`, `warmup`, `host_sync` or `capture`). These safeguards keep the fallback working:

- **Host syncs are found before capture.** A call that synchronizes with the host (`tensor.item()`, `nonzero`, a device-to-host copy) can never be captured. Running it once under `torch.cuda.set_sync_debug_mode("error")` detects that without starting a capture, and the measurement goes straight to `do_bench` (`phase=host_sync`).
- **Memory of a failed capture is released.** An aborted capture keeps its graph and private memory pool alive in reference cycles until the garbage collector runs; with inputs of tens of GiB a few of them leave too little memory for the next measurement. The plugin collects and empties the cache before the fallback. A capture that runs out of memory is retried once with one call per graph (`CUDAGRAPH_OOM_RETRY`), still timed by graph replay.
- **Fatal errors are matched on the error text itself.** Illegal addresses, OOM, device-side asserts and the like are re-raised rather than timed. torch appends an "enable device-side assertions" hint to every CUDA error, so matching loose substrings would make every capture failure fatal.
- **Capture runs under a throwaway RNG state.** An aborted capture can leave the default CUDA generator in capturing mode, after which every `torch.randn` in the process raises "Offset increment outside graph capture". The generator's state is swapped for a scratch one during capture and restored afterwards, so the real generator never takes part in a capture.
- **An aborted capture is followed by one untimed call.** cusolver keeps the aborted capture in its handle state and reports `CUSOLVER_STATUS_EXECUTION_FAILED` once on the next call before recovering (e.g. `linalg_eigvals`, `linalg_matrix_rank`).
- **A platform without the graph API degrades once, not per shape.** Sync-debug mode, the graph-safe RNG state and the generator list are optional: a backend that lacks one captures anyway (the PPU and MACA compatibility layers implement only part of the API). A backend without `torch.cuda.CUDAGraph`, `Stream` or `Event` goes straight to `do_bench` (`phase=capability`), and a capture the platform rejects as unsupported turns graphs off for the remaining shapes of that op. The next op runs in a fresh process and tries again, so ops that can be captured keep their graph timing.

### Accuracy of fallback rows

Fallback does not bias an A/B, since both sides fall back the same way for the same op, but `do_bench` includes the Python launch path in every iteration. On sub-millisecond kernels the two methods can differ by 2× or more and rows fluctuate more, so do not over-read absolute values or differences under about 10% on small shapes. Fallback happens per measurement, and one table can mix both methods; `grep BENCHMARK_DIRECT_NO_CUDAGRAPH run.log` shows which rows are affected.

## IR dump

`_ir_dump_plugin` works on the run's private Triton cache, keyed by the cache dir of each compiled kernel:

- **Compile listener**: writes a `__constexprs.json` sidecar with the constexpr values, signature dtypes and per-argument specializations (e.g. `tt.divisibility`) next to each kernel, since the ttgir alone does not say which autotune variant it is.
- **Shape tracking**: publishes the shape being fed. Upstream has two input paths, the legacy `get_input_iter` generator and the Case API (`get_case_iter` + `build_inputs` when `supports_cases()` is true, e.g. the `UnaryPointwiseBenchmark` and `BlasBenchmark` families); both are wrapped, in place so upstream's `supports_cases()` identity checks keep working, and yield the same shape labels.
- **Launch tracking**: counts launches per (kernel, shape), separately inside and outside `Autotuner._bench`. A variant is "used" if it ran at least once outside the sweep.
- **Row tracking**: upstream creates one `BenchmarkMetrics` per result row right before measuring it. From then until the next input is built, every launch outside the sweep and inside the timed function belongs to that row; `_timing_scope` marks the timed function, so launches the timing harness itself makes (e.g. fills torch issues when a capture begins) stay out. When a result table is printed, its rows are noted in print order. `rows.tsv` lists, per (op_name, dtype) table and row index as in `run.log`, the shape label and the cache hashes of the kernels the row ran. A benchmark that times without `triton.testing.do_bench` (a few call `do_bench_cudagraph` themselves), or a run without `_cudagraph_plugin`, has no timed function; its rows get every launch made while they were current, marked `scope=row`. That set can include kernels that only build inputs, which errs on the side of calling a row changed.
- **Dump at exit**: for each used variant, the ttgir goes to `ttgir/<shape>/<kernel>/<name>.ttgir` and the ttir plus Triton's metadata json to `ttir/<cache hash>/`. Sweep losers are listed in `ttgir/index.tsv` only. Dumping at process exit rather than session end keeps the IR of a run that crashed.

File names keep only what tells the variants of a (shape, kernel) group apart: constexprs whose value differs within the group, abbreviated per word (`BLOCK_SIZE_M` → `BSM`; legend in `naming.md`), then `w{warps}s{stages}`. Remaining collisions are broken by signature dtype (`__fp16`), then argument specialization (`__EMdiv16`), then a hash prefix.

The json in `ttir/` is the compile options for recompilation: Triton's metadata with `num_warps` restored from the ttgir module, because warp specialization rewrites the metadata value to the sum over partitions (e.g. 12), which the compile option rejects.

## Statistics

### One sample per round

A round runs each side once per op, in a fresh process with its own Triton cache, and contributes one latency per result row and side. Within a measurement the plugin already takes the median of 10 graph replays; what remains between rounds is run-to-run noise: GPU clocks and thermals, memory placement, the GPU a round lands on. That noise is what a verdict has to beat, so the unit of replication is the round, not the replay.

Pairing A and B on one GPU back to back cancels the per-GPU bias (a few percent between the GPUs of one box) and slow drift; alternating which side goes first cancels order effects such as a warm L2 or clock ramp.

### Verdict

`ab_stats.compare` judges a row from its A and B samples:

| Verdict | Condition |
|---|---|
| `undecided` | fewer than `--min-rounds` (6) samples on either side |
| `same` | \|gap\| ≤ `--threshold` (0.5%), or Mann-Whitney p ≥ `--clear` (0.1) |
| `faster` / `slower` | \|gap\| > threshold and p < `--alpha` (0.01) |
| `undecided` | otherwise: a gap, but not yet significant |

`gap = median(A) / median(B) − 1`. The Mann-Whitney U test compares ranks, so it needs no assumption about the latency distribution, which is skewed and sometimes bimodal (some kernels alternate between two speeds from run to run). It uses the exact null distribution for small samples without ties and the normal approximation otherwise, as `scipy.stats.mannwhitneyu` does. With 6 rounds per side the smallest attainable two-sided p is 0.0022, so `--alpha 0.01` is reachable only when the two samples barely overlap. The threshold keeps a statistically clear but immaterial shift (a fraction of a percent) out of the results.

### Controls and false positives

With `--gate`, rows whose kernels compiled to identical binaries on both sides cannot differ; they go through the same rule as the changed rows, and every `faster` or `slower` among them is a false positive. Their rate is the measured false-positive rate of the rule on this machine and this record, which a fixed α alone does not give: run-to-run effects that are not random noise (a bimodal kernel, a slow GPU on some rounds) show up here. Controls ride along on the ops that are measured anyway, so they cost nothing extra.

### More rounds, and fresh rounds

`--until-decided` adds rounds to ops with undecided rows. Testing again after each addition inflates the false-positive rate a little (sequential testing); the controls, which get the same extra rounds, show by how much.

A row near the threshold, or decided on few rounds, has been both selected and judged on the same samples, which overstates small effects (the winner's curse). The fix is an independent check: `ab_run.py --add-rounds N` runs N new rounds, and `ab_compare.py --confirm` judges each faster/slower row on those alone, with a one-sided test in the original direction at α = 0.05. One-sided is right because the direction was fixed before the new data existed; a row that passes both tests is a false positive only if both are, which the independence of the new rounds makes rare. A confirmed row reports the gap over all rounds; a row that is not reproduced becomes undecided rather than same, since a small real effect can also miss on a few rounds.

## Accuracy gate

`ab_run.py --accuracy-gate` runs each side's unit test for an op before the first round and drops the op from the A/B when a side does not pass it. An A/B compares latency and nothing else: upstream's `benchmark/base.py` asserts nothing about the numbers (`torch_op` is only the baseline it times), so a compiler that changes an op's arithmetic still yields timing rows on both sides, and the report would call the difference a speedup. The gate is what stands between a numerical regression and a performance verdict.

- **What runs.** The cases in that repo's `tests/test_<op_file>.py` (`op_file` keeps a benchmark subdir, so `benchmark/test_FLA/test_x.py` maps to `tests/test_FLA/test_x.py`), located in three steps: the exact `test_<op>`; when the file has no such function, every test function whose name contains `test_<op>` (`chunk_kda`'s test is `test_chunk_kda_matches_torch_reference`, and a file holding `test_abs` and `test_abs_` gives the exact one for `abs`); else `--accuracy-map`. A case defined in a class keeps its class path (`tests/test_grid_sample.py::TestGridSampleEdgeCases::test_edge`): a bare `::test_edge` is a node pytest cannot collect, so the op would read as `no_test`. A class holds tests only under pytest's own rules: the `Test*` prefix (where a custom constructor — an `__init__` or `__new__`, its own or inherited — makes pytest skip the class), or a `unittest.TestCase` subclass resolved through this module's imports and aliases (its own `__init__` is fine; a base whose name merely ends in `TestCase` proves nothing), and `__test__` takes part in collection the way pytest reads it: `= False` opts a class or module out (the flag is inherited like any other attribute, so a subclass is skipped too unless it sets `__test__ = True`), `= True` also makes a plain class without the `Test*` prefix a test container (the constructor rule still applies), the **last** assignment in a scope wins, and an assignment under `if`/`try` or a non-literal value leaves the class uncertain — for a module, every node in it. An inherited value is looked up along the **C3 MRO** for the first class that defines `__test__` (in a diamond, a base's effective value is not what `getattr` returns); an inconsistent hierarchy, an unresolvable base, or a base that is not a name at all (`make_base()`, `Generic[T]`) leaves the class uncertain — an unreadable expression must not read as "no bases", which is what collects a class pytest skips. Bases are captured as the classes bound **at the definition**, not as names, so redefining an ancestor later does not rewrite what an existing class inherits from (that is what `__bases__` does). Only the *identity* inherits: a `unittest.TestCase` subclass is collected whatever its name, while being collectable does not pass to a subclass — `class Helper(TestBase)` is collected only if it is itself a `Test*` class or a TestCase (a node pytest does not have is rc 4). Bases are resolved in module scope, in definition order: an import inside a function binds nothing at module level, a later `class Base` / `Base = ...` or **another import** of the same name takes an earlier `TestCase` alias back (what another module exports is unknowable, so the name becomes unknown instead of keeping the old identity), a conditional import leaves the name unknown; and a `from x import *` whose exports cannot be listed drops every existing binding to unknown (it may re-export a `TestCase` alias as something else), until a later explicit import or class definition re-establishes one; a base that cannot be resolved (a TestCase subclass imported from elsewhere) leaves the class uncertain; a `def` under an `if`/`try` is not picked automatically either — when the branch did not run, pytest exits 4 with "not found" and the op would be dropped as an error rather than `no_test`, so those wait for an `--accuracy-map` entry. It runs in a fresh process with `cwd` at the repo root, that side's `PYTHONPATH` and `--a-env` / `--b-env`, and one GPU. Same test a developer runs by hand, so it carries the repo's own tolerance: FlagGems' `gems_assert_close` compares against the torch reference on CPU with a per-dtype `rtol` (1.3e-6 fp32, 1e-3 fp16, 0.016 bf16, 0 for exact dtypes) plus `atol` 1e-4.
- **The map (`--accuracy-map`).** For the ops whose names and tests disagree — benchmark `act_quant_perf`, test `test_act_quant_triton` — a file names the test: one `[repo:]op[@op_file] target [target...]` per line, `target` a path under that repo's `tests/` (relative to the repo root) with an optional `::test_func`, or the whole file; an entry without `@op_file` covers the op whatever benchmark file it sits in, and several targets run in one pytest call. Naming the whole file means every case in it gates the op, including the ones it does not cover. The map is checked at startup (repo name, target under `tests/`, file exists, no op mapped twice): a typo stops the run instead of surfacing as a pile of `no_test` hours later. It beats the automatic lookup, and `ab.json`'s `accuracy.map` records which file and how many entries were used; `ab_compare.py` prints that line.
- **What counts as a pass.** rc 0 *and* at least one passed test. A module whose tests all skipped (backend or dtype unavailable on this box) verified nothing and is `no_test`; rc 5 (nothing collected) is `no_test` too. rc 2/3/4 are `error` (interrupted, internal, usage — e.g. a conftest that wants a flag), anything else `fail`, a timeout `timeout`. `fail`, `timeout` and `error` drop the op; `no_test` does not.
- **The benchmark inventory is not 1:1 with the tests.** Even after the wider match, 471 of the 1684 inventoried ops locate no test: `perf_`-prefixed functions and inplace/out variants often share one test, some benchmarks have none, and some names have nothing to do with their test function (`act_quant_perf` → `test_act_quant_triton`). Dropping those would remove nearly a third of the ops for a reason that says nothing about the compiler, so they run with `no_test` recorded, and the report says how many results are not backed by an accuracy check. Those 471 are what `--accuracy-map` is for: every entry turns one `no_test` into a real check.
- **Turning `no_test` into candidates (`accuracy_hints.py`).** At the end of the gate, those ops are searched for and two files land next to the A/B output: `accuracy_no_test.md` lists each op with its benchmark file and candidate tests, and `accuracy_map.suggested.txt` is the same candidates as an `--accuracy-map` draft, **every line commented** — uncomment the one that is right (one line per op; mapping an op twice is rejected at startup). Candidates are ranked by how they were matched, each saying why: `[同名文件]` (`tests/test_<op_file>.py` exists but has no `test_<op>`, so its own cases are listed), `[文件名相近]`/`[文件名包含]`, `[函数名命中]` (the file name says nothing but a function carries the op), `[benchmark 名命中]` (only the benchmark file name hits a function name — the weakest), and a whole-file target when a name-similar file has no related case (it gates the op with every case in it). Matching is done on `_` boundaries, otherwise `mul` matches `cumulative` and `mhc` matches `mhc_bwd`. A whole-file target is only a fallback: it sorts after every named case and does not count as "a candidate was found", so an empty same-name file (or a name-similar file with no related case) never hides a usable node in another file; a file that parses to no cases only means "nothing was parsed" (import failures and dynamic definitions look the same), not that pytest collects nothing. The all-files function sweep only runs for the ops whose file names matched nothing. Every candidate is a node pytest may collect: a case defined in a class keeps its class path (`TestA::test_x`); a `test_*` method of a plain class (no `Test*` prefix, no `TestCase` base imported here, or a class with `__init__` / `__test__ = False`) is not a case, and neither is a `def test_*` inside a function. A case under an `if`/`try` is listed with `·条件定义`: it exists only when its branch ran, and `ab_run.py`'s automatic lookup leaves it alone. The report is rebuilt from the **whole** gate record in `ab.json`, so a resume that only re-ran some ops does not drop the others; a run that leaves no `no_test` behind rewrites it as an empty list instead of leaving the old one standing. An existing `--out` can be re-searched any time: `python accuracy_hints.py --ab-dir runs/ab_x`.
- **Recorded per (op, side), not just printed.** `ab.json` gets an `accuracy` object next to the manifest fields: `sides.<a|b>.<repo>/<op>` with status, target, return code, duration and the path of its log under `<out>/accuracy/<side>/<repo>/<op>/gate.log`, plus `skipped: {op: why}` with pytest's first summary line, and `map: {file, entries}` when a map was used. A recorded `pass` is reused when the same `--out` is resumed and under `--add-rounds`, so the gate is paid once; a recorded `no_test` is reused too, unless a test can now be located for it (a new `--accuracy-map` entry, or a match the wider lookup finds) — that row is then stale and runs again. `--accuracy-timeout` (default 1800 s) caps a single run, killed by process group like a timed-out benchmark.
- **In the report.** `ab_compare.py` prints the section after the verdicts — per side counts, the number of `no_test` op-sides, and every skipped op with the reason — and `--csv` carries an `accuracy` column (`a:pass,b:fail`) per row, plus `accuracy_failed` in `flags` so a consumer that keeps only the markers still sees it. An op skipped by the gate has no rounds, so it appears only in that section; one measured in an earlier invocation keeps its old rows, and those rows are shown but not counted: the report lists them under `accuracy failed (a side the gate rejected)` with `historical performance only`, because their medians describe the build the test rejected. `--recheck-list` leaves them out too — more rounds cannot fix a rejected side, and `ab_run.py --add-rounds` would gate the op out again and stop. `no_test` and an empty cell are not failures — the gate never ran, or had nothing to run, and their A/B stands. `rlc_report.py` reads the same cell out of `--csv` (or the flag, when the column was dropped): such an op still gets its table row and a note ("精度失败（a:pass,b:fail；历史性能仅供参考，N 行，未计入汇总）"), but its share of the gain/loss and saving/cost figures is zero unless `--include-accuracy-failed` is given. That note's counted/left-out tail follows what the summaries actually did, not which switch was passed: a row carrying both marks is never called counted while the other reason still keeps it out, and a note whose rows only partly count reads `其中 M 行未计入汇总`.
- **What it does not do.** It does not make the verdict more reliable: the gate answers "are the numbers still right on both sides", the statistics answer "is the difference real". It also cannot catch a difference both sides share (a wrong tolerance in the test, an op whose reference is itself the gems implementation).

## Shape override

Upstream reads shapes in `Benchmark.set_shapes()`, falling back from the op_name key to an MRO class-name key to the base-class `DEFAULT_SHAPES`. A good share of benchmarks bypass that. `_shape_override_plugin` closes three gaps, each by reusing the class's own input construction and only swapping the source of shapes, so the measured kernel stays the same; ops without a yaml entry keep their upstream shapes.

1. **`set_shapes()` overridden with a hardcoded list** (e.g. `fused_marlin_moe_*`): `Benchmark.init_user_config`, which runs right after `set_shapes()`, is wrapped to put the yaml shapes back. Case-API benchmarks enumerate `self.shapes`, so this covers them too. It steps aside when the subclass normalizes the arity after reading the yaml (BLAS (B, M, N, K) ↔ (M, N, K)), detected as both sides having a single but different arity.
2. **`get_input_iter` with a hardcoded list that never reads `self.shapes`** (conv padding, pool, nll, scaled_softmax families): yields one input per yaml shape through the class's standard `input_fn(shape, dtype, device)`. Classes without such an `input_fn` are left alone.
3. **Bespoke input construction**, registered by class name: `UpsampleBicubic2dAaBackwardBenchmark`, `FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark` and `FlashmlaSparseBenchmark` build their inputs from config tables or `TestParam` objects; `CutlassScaledMMPerfKit` samples cases from an (M, N, K) list, which is replaced while the quantization-mode product stays as upstream defines it. These couple to upstream internals, and a mismatch shows up as a benchmark failure, not as silently different inputs.

Benchmarks that never read shapes at all, such as `scaled_mm_benchmark` (parameterized by cases), always run their built-in shapes.

## Parallel sweep

With many shapes the sweep dominates: for `fused_marlin_moe_w4a16_mxfp4` with 53 shapes, a record run took 3755 s while the measured kernels summed to 49 ms, and 2645 of the 2858 compiled variants were sweep losers; replaying the same configs took 196 s.

`_parallel_warmup_plugin` runs before the record/replay plugin configures. It splits the shapes of `OP` round-robin (shape lists are usually sorted, and contiguous blocks would give one shard all the large shapes), runs one pytest subprocess per shard pinned to its own GPU, merges the shards' records into the run's `autotune_records/<op>.json`, and switches the run to replay those configs. The measurement that follows is the normal single-GPU path. Subprocesses are used instead of pytest-xdist because importing FlagGems creates a CUDA context, which does not survive fork-based workers.

The measurement itself is never parallel: processes saturating several GPUs of one box couple through power limits and thermals, and per-GPU latency then depends on the neighbours. Within the sweep the same coupling can change which config wins. A key with different winners in different shards counts as a conflict, the first value is kept, and the log reports the count. It is the trust level of the speed-up: zero is fine, many means rerunning without the variable before drawing conclusions. Shard pinning respects an inherited `CUDA_VISIBLE_DEVICES`; if no shard returns configs, the run falls back to serial tuning and keeps the shard logs.

## Upstream assumptions

- The upstream benchmark framework: the `Benchmark` class in `benchmark/base.py` (`set_shapes`, `init_user_config`, `get_input_iter`, and the Case API in the main repo), the conftest options `--shape_file`, `--level`, `--mode` and `--dtypes`, `consts.BenchmarkResult` (a table is logged by printing it) and `consts.BenchmarkMetrics` (one per result row, created right before the row is measured; row tracking relies on that). Each repo ships its own copy; the plugins use whichever `benchmark` package is running.
- LibTuner lives in `<package>.utils.libentry`; `_autotune_record_plugin` patches whichever package (`flag_gems` or `flaggems_vllm`) was imported.
- Triton provides `knobs.compilation.listener`, `kernel_load_end_hook` and `launch_enter_hook` (FlagTree's Triton 3.6 does).
- The plugins monkeypatch upstream internals. If behavior changes after a major upstream update, first check that each plugin still prints its registration line at startup.
