#!/bin/bash
# Generate the RLC sub-switch TTGIR matrix for one benchmark op (PPU).
#
# The four switches are the ones wired in the pass:
#   FLAGTREE_RLC_ENHANCE                          master (must be 1 for the rest to matter)
#   FLAGTREE_RLC_ENHANCE_COST_BASED               Phase 1a
#   FLAGTREE_RLC_ENHANCE_BACKWARD_PROPAGATION     Phase 1b  (needs cost_based)
#   FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING  Phase 2   (needs cost_based)
#   FLAGTREE_RLC_ENHANCE_STORE_LAYOUT_REMAT       Phase 3   (independent)
#
# 2^4 = 16 nominal combinations, but runOnOperation does
#     if (!costBased) { backwardProp = false; smallComponentSolving = false; }
# so with cost_based=0 the backward/small bits are dead: 10 distinct outcomes.
#
# Usage (from anywhere; the script cd's to the repo root):
#   bash gen_combination_ttgir.sh                        # OP=coalesce, float16, latency
#   bash gen_combination_ttgir.sh softmax                # OP=<arg 1>, OP_FILE=<arg 1>
#   bash gen_combination_ttgir.sh softmax softmax_v2     # OP=<arg 1>, benchmark file stem=<arg 2>
#   OP=add OP_FILE=add bash gen_combination_ttgir.sh     # same, via the environment
#   DTYPES=float16 METRICS=latency bash gen_combination_ttgir.sh coalesce
#   DTYPES="float16 bfloat16" bash gen_combination_ttgir.sh coalesce   # several dtypes
#   ALL16=1 bash gen_combination_ttgir.sh                # also run the 6 dead combinations
#   RESUME=1 bash gen_combination_ttgir.sh               # skip variants already dumped
#   STOP_ON_ERROR=1 bash gen_combination_ttgir.sh        # stop at the first failing variant
#   SHAPE_FILE=shape/my_coalesce.yaml bash gen_combination_ttgir.sh
#   OUT_ROOT=runs/ppu/rlc_matrix_v4 bash gen_combination_ttgir.sh coalesce
#
# One dtype is the default on purpose: the matrix compares layouts, and a second
# dtype triples the compile time without changing which switch moves which layout.
# `DTYPES` takes a space separated list when the dtype itself is what you study.
#
# Output: <OUT_ROOT>/<op>/<variant>/{run.log,shapes.yaml,ttgir,ttir,rows.tsv}, one
# directory per op, so several ops' matrices can share one root and be compared
# together or per op.
# Next step: python compare_ttgir_matrix.py runs/ppu/rlc_matrix --op <op>
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The script sits at the repo root, next to run_pytest.sh.
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT" || exit 1

# Op under test: positional (op [op_file]) wins, then the environment, then coalesce.
# OP names the test (`::test_<OP>`), OP_FILE the benchmark file (`test_<OP_FILE>.py`).
OP="${1:-${OP:-coalesce}}"
OP_FILE="${2:-${OP_FILE:-$OP}}"
OUT_PARENT="${OUT_ROOT:-runs/ppu/rlc_matrix}"   # one sub-directory per op below this
OUT_ROOT="$OUT_PARENT/$OP"
DTYPES_WANT="${DTYPES:-float16}"     # one dtype is enough to compare layouts, 3x faster
METRICS_WANT="${METRICS:-latency}"   # skip the torch baseline op
ALL16="${ALL16:-0}"
RESUME="${RESUME:-0}"                # 1: skip a variant whose ttgir/ is already populated
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"  # 1: stop at the first failing variant

# Same default as run_pytest.sh: the FlagGems checkout next to this repo.
FLAGGEMS_DIR_WANT="${FLAGGEMS_DIR:-${FLAGGEMS_MAIN_DIR:-$(dirname "$REPO_ROOT")/FlagGems}}"
BENCH_FILE="$FLAGGEMS_DIR_WANT/benchmark/test_${OP_FILE}.py"

[[ -f run_pytest.sh ]] || { echo "!! run_pytest.sh not found under $REPO_ROOT" >&2; exit 2; }
# Fail here, not ten variants later: every variant runs the same missing test.
if [[ ! -f "$BENCH_FILE" ]]; then
  echo "!! no benchmark file $BENCH_FILE" >&2
  echo "   (OP=$OP OP_FILE=$OP_FILE; set OP_FILE/FLAGGEMS_DIR for another op or checkout)" >&2
  exit 2
fi
[[ -n "${SHAPE_FILE:-}" ]] && echo ">>> SHAPE_FILE=$SHAPE_FILE (inherited by run_pytest.sh)"

FAILED=()

run_one() {
  local name="$1" master="$2" cost="$3" back="$4" small="$5" remat="$6"
  local out="$OUT_ROOT/$name"
  if [[ "$RESUME" == "1" && -d "$out/ttgir" ]] && [[ -n "$(ls -A "$out/ttgir" 2>/dev/null)" ]]; then
    echo ">>> skip $name (already dumped in $out/ttgir; RESUME=0 re-runs it)"
    return 0
  fi
  echo ""
  echo "=================================================================="
  echo ">>> $name   master=$master cost=$cost backward=$back small=$small remat=$remat"
  echo ">>> out=$out"
  echo "=================================================================="
  OUT_DIR="$out" \
  OP="$OP" OP_FILE="$OP_FILE" \
  DTYPES="$DTYPES_WANT" METRICS="$METRICS_WANT" USE_FLAGTUNE=0 \
  FLAGTREE_RLC_ENHANCE="$master" \
  FLAGTREE_RLC_ENHANCE_COST_BASED="$cost" \
  FLAGTREE_RLC_ENHANCE_BACKWARD_PROPAGATION="$back" \
  FLAGTREE_RLC_ENHANCE_SMALL_COMPONENT_SOLVING="$small" \
  FLAGTREE_RLC_ENHANCE_STORE_LAYOUT_REMAT="$remat" \
  TRITON_ALWAYS_COMPILE=1 \
  bash run_pytest.sh
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "!! $name failed (rc=$rc); see $out/run.log" >&2
    FAILED+=("$name")
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      echo "!! STOP_ON_ERROR=1: not running the remaining variants" >&2
      exit 1
    fi
  fi
}

echo ">>> op=$OP (OP_FILE=$OP_FILE)"
echo ">>> variants land in $OUT_ROOT/"
echo ">>> dtypes=$DTYPES_WANT metrics=$METRICS_WANT all16=$ALL16 resume=$RESUME"

# Reference: enhancement off entirely (what the A side of the A/B was).
run_one enhance_off 0 1 1 1 1

# cost_based=0 -> backward/small ignored by construction, so one representative.
run_one c0b0s0r0 1 0 0 0 0
run_one c0b0s0r1 1 0 0 0 1

# cost_based=1 -> the 4 (backward, small) pairs x 2 (remat).
for b in 0 1; do
  for s in 0 1; do
    for r in 0 1; do
      run_one "c1b${b}s${s}r${r}" 1 1 "$b" "$s" "$r"
    done
  done
done

if [[ "$ALL16" == "1" ]]; then
  echo ""
  echo ">>> ALL16=1: also running the combinations that are equivalent to c0b0s0r{0,1}"
  for b in 0 1; do
    for s in 0 1; do
      for r in 0 1; do
        [[ "$b" == "0" && "$s" == "0" ]] && continue   # already covered
        run_one "c0b${b}s${s}r${r}" 1 0 "$b" "$s" "$r"
      done
    done
  done
fi

echo ""
echo "=== done. variants of $OP: ==="
ls -1 "$OUT_ROOT" 2>/dev/null | sed 's/^/  /'
echo ""
echo "next: python compare_ttgir_matrix.py $OUT_PARENT --op $OP"
echo "  or just this op: python compare_ttgir_matrix.py $OUT_ROOT"

if (( ${#FAILED[@]} )); then
  echo "!! ${#FAILED[@]} variant(s) failed: ${FAILED[*]}" >&2
  exit 1
fi
