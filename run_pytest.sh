#!/bin/bash
# Single-op benchmark entry. Edit the config section or override any setting
# with the environment variable of the same name:
#   OP=softmax OP_FILE=softmax SHAPE_FILE=my_shapes.yaml bash run_pytest.sh
# Outputs land in runs/<op>_<timestamp>/: run.log, shapes.yaml,
# autotune_records/, ttgir/, ttir/, rows.tsv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================== config ==============================
# Every setting can be overridden by its environment variable; see README.

# Op under test: $FLAGGEMS_DIR/benchmark/test_<OP_FILE>.py::test_<OP>
OP="${OP:-fused_marlin_moe_w4a16_mxfp4}"
OP_FILE="${OP_FILE:-fused_marlin_moe_w4a16_mxfp4}"

# Upstream repo under test: FlagGems (package flag_gems, the default) or
# FlagGems-vllm (package flaggems_vllm). The default is the FlagGems checkout
# next to this repo, or FLAGGEMS_MAIN_DIR if set (the same one run_batch.py uses).
FLAGGEMS_DIR="${FLAGGEMS_DIR:-${FLAGGEMS_MAIN_DIR:-$(dirname "$SCRIPT_DIR")/FlagGems}}"

# Shape set, top-level keys = op_name (or Benchmark class name). Unset: replay
# reuses the replayed run's archived shapes.yaml (same shapes on both sides),
# record uses the all-ops multi-shape set (has an entry for every op).
SHAPE_FILE="${SHAPE_FILE:-}"

# Tuning space: 0 = plain autotune (default, fast); 1 = FlagTune expanded space
# (full search on first run, slow).
USE_FLAGTUNE="${USE_FLAGTUNE:-0}"

# Optional dtype whitelist, space separated (e.g. "bfloat16 float16"); empty =
# upstream's default sweep. Unsupported dtypes make upstream raise "can't be
# supported by this op", which run_batch.py uses to retry without the filter.
DTYPES="${DTYPES:-}"

# Empty = record mode: the chosen configs go to autotune_records/<op>.json.
# A past run dir = replay its records, pinning the same configs on both A/B sides.
REPLAY_FROM="${REPLAY_FROM:-}"

# always/never forces terminal color; empty = auto by tty (run.log is always plain).
FLAGGEMS_PERF_COLOR="${FLAGGEMS_PERF_COLOR:-}"

# Plugin roles: docs/design.md. Comment a line out to disable that plugin.
PLUGINS=(
  -p _parallel_warmup_plugin
  -p _seed_plugin
  -p _shape_override_plugin
  -p _autotune_record_plugin
  -p _cudagraph_plugin
  -p _pretty_report_plugin
  -p _ir_dump_plugin
)

# ============================== run ==============================

if [[ -z "$SHAPE_FILE" ]]; then
  if [[ -n "$REPLAY_FROM" && -f "$REPLAY_FROM/shapes.yaml" ]]; then
    SHAPE_FILE="$REPLAY_FROM/shapes.yaml"
  else
    SHAPE_FILE="$SCRIPT_DIR/shape/all_ops_multi.yaml"
  fi
fi
TEST_PATH="$FLAGGEMS_DIR/benchmark/test_${OP_FILE}.py"
[[ -f "$TEST_PATH" ]] || { echo "!!! no benchmark file $TEST_PATH (check OP_FILE / FLAGGEMS_DIR)" >&2; exit 2; }
[[ -f "$SHAPE_FILE" ]] || { echo "!!! no shape file $SHAPE_FILE" >&2; exit 2; }

# Output dir, overridable via OUT_DIR. The timestamp has 1 s resolution and
# concurrent launches collide, so claim it with a plain mkdir (fails if it
# exists) and fall back to -2, -3, ...
if [[ -n "${OUT_DIR:-}" ]]; then
  mkdir -p "$OUT_DIR"
else
  mkdir -p "$SCRIPT_DIR/runs"
  BASE="$SCRIPT_DIR/runs/${OP}_$(date +%Y%m%d-%H%M%S)"
  OUT_DIR="$BASE"
  n=1
  until mkdir "$OUT_DIR" 2>/dev/null; do
    n=$((n + 1))
    OUT_DIR="${BASE}-${n}"
    (( n > 99 )) && { echo "!!! cannot create an output dir ($BASE and its suffixes are taken)" >&2; exit 1; }
  done
fi
LOG_FILE="$OUT_DIR/run.log"

# The tee pipeline hides the tty from python, so decide here and pass it down.
if [[ -z "$FLAGGEMS_PERF_COLOR" && -t 1 && -z "${NO_COLOR:-}" ]]; then
  FLAGGEMS_PERF_COLOR=always
fi
export FLAGGEMS_PERF_COLOR
if [[ "$FLAGGEMS_PERF_COLOR" == "always" ]]; then
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
  PYTEST_COLOR=(--color=yes)
else
  C_RED='' C_GREEN='' C_YELLOW='' C_DIM='' C_BOLD='' C_RESET=''
  PYTEST_COLOR=()
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export FLAGGEMS_PERF_CURRENT_OP="$OP"
export PYTHONUNBUFFERED=1

# One upstream --dtypes option per dtype (it is action=append).
DTYPE_ARGS=()
for _dt in $DTYPES; do
  DTYPE_ARGS+=(--dtypes "$_dt")
done

# Record and replay are mutually exclusive, each enabled by its own variable.
if [[ -n "$REPLAY_FROM" ]]; then
  AUTOTUNE_ENV="FLAGGEMS_PERF_AUTOTUNE_REPLAY_DIR=$REPLAY_FROM/autotune_records"
  [[ -f "$REPLAY_FROM/autotune_records/$OP.json" ]] || \
    echo "${C_YELLOW}>>> warning: no $REPLAY_FROM/autotune_records/$OP.json; replay falls back to live autotune${C_RESET}" >&2
  MODE_DESC="replay($REPLAY_FROM)"
else
  AUTOTUNE_ENV="FLAGGEMS_PERF_AUTOTUNE_RECORD_DIR=$OUT_DIR/autotune_records"
  mkdir -p "$OUT_DIR/autotune_records"
  MODE_DESC=record
fi

cp -f "$SHAPE_FILE" "$OUT_DIR/shapes.yaml"  # archive the shape set actually used

status=0
{
  echo "${C_BOLD}>>> op=$OP  mode=$MODE_DESC  USE_FLAGTUNE=$USE_FLAGTUNE${C_RESET}"
  echo "${C_DIM}>>> out=$OUT_DIR  shapes=$SHAPE_FILE${C_RESET}"
  # Log the compiler under test and the switches that affect compilation, so
  # the two sides of an A/B can be reconciled. find_spec locates the package
  # without importing it.
  triton_dir="$(python -c 'import importlib.util as u, os; s = u.find_spec("triton"); print(os.path.dirname(s.origin) if s else "")')"
  triton_rev="$(git -C "$triton_dir" rev-parse --short HEAD 2>/dev/null || true)"
  echo "${C_DIM}>>> triton=${triton_dir:-not found}${triton_rev:+ (git $triton_rev)}${C_RESET}"
  compile_env="$(env | grep -E '^(FLAGTREE|TRITON|MLIR)_' | sort | tr '\n' ' ' || true)"
  [[ -n "$compile_env" ]] && echo "${C_DIM}>>> env: ${compile_env% }${C_RESET}"

  # A private Triton cache per run, deleted afterwards: compilation is
  # reproducible and the IR dump only holds this run's variants
  # (_ir_dump_plugin writes ttgir/ and ttir/ at exit).
  CACHE_DIR="$OUT_DIR/.triton_cache"
  rm -rf "$CACHE_DIR"; mkdir -p "$CACHE_DIR"
  status=0
  # -rsx: list skip/xfail reasons at the end of the log. no:cacheprovider: do
  # not write .pytest_cache into the upstream repo (concurrent batch writers clash).
  TRITON_CACHE_DIR="$CACHE_DIR" \
  FLAGGEMS_PERF_IR_DUMP_DIR="$OUT_DIR" \
  env "$AUTOTUNE_ENV" \
  USE_FLAGTUNE="$USE_FLAGTUNE" python -u -m pytest -s -rsx "$TEST_PATH::test_${OP}" \
    -p no:cacheprovider "${PLUGINS[@]}" "${PYTEST_COLOR[@]}" \
    --shape_file "$SHAPE_FILE" ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} \
    --level core --mode kernel || status=$?
  rm -rf "$CACHE_DIR"

  if (( status == 0 )); then
    echo "${C_GREEN}>>> done. outputs in $OUT_DIR${C_RESET}"
  else
    echo "${C_RED}>>> FAILED (pytest exit $status). partial outputs in $OUT_DIR${C_RESET}"
  fi
  exit "$status"
} 2>&1 | tee "$LOG_FILE" || status=$?

# Strip ANSI escapes from run.log for grep/diff (the terminal keeps its colors;
# skipped on Ctrl-C).
sed -i -E $'s/\x1b\\[[0-9;]*[A-Za-z]//g' "$LOG_FILE"
exit "$status"
