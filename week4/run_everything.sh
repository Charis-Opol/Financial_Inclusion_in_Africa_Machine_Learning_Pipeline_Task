#!/usr/bin/env bash
# Runs every Phase 4 experiment on one machine: checks, all stages, E6 (SHAP), then metrics and figures.
# Run from the repo root of a fresh clone, detached so closing the terminal doesn't stop it
# (full steps in week4/GPU_RUN.md):
#   mkdir -p week4/results/logs && nohup bash week4/run_everything.sh > week4/results/logs/run_everything.out 2>&1 &
#
# Settings (environment variables):
#   DEVICE          TabPFN device, default cuda
#   H2_ESTIMATORS   TabPFN estimators for H2 embeddings, default 8 (D3.4)
#   PARALLEL_SEEDS  1 runs one CPU lane per seed, 0 one lane for all seeds. Default: 1 if 12+ cores
#
# Two kinds of lane run at the same time. They never write to the same files:
#   GPU lane: every TabPFN arm and the H2 heads, all seeds, one after another.
#   CPU lanes: XGBoost and MLP arms. A lane holds whole seeds, so a sweep arm always runs after
#              its parent mlp_dro for the same seed.
# Finished folds are skipped, so rerunning this script resumes. Smoke and preflight rerun each time.
set -uo pipefail

if [ -x week4/.venv/bin/python ]; then PY=week4/.venv/bin/python; else PY=week4/.venv/Scripts/python.exe; fi
DEVICE=${DEVICE:-cuda}
H2_ESTIMATORS=${H2_ESTIMATORS:-8}
CORES=$(nproc 2>/dev/null || echo 8)
PARALLEL_SEEDS=${PARALLEL_SEEDS:-$([ "$CORES" -ge 12 ] && echo 1 || echo 0)}
SEEDS="0 1 2 3 4"
LOGS=week4/results/logs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGS"
RUN="$PY week4/run_all.py"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/main.log"; }

log "Device $DEVICE, H2 estimators $H2_ESTIMATORS, $CORES cores, parallel seeds $PARALLEL_SEEDS. Logs in $LOGS"

# 1. Checks. Any failure stops everything before a long job starts.
"$PY" week4/src/preflight.py --device "$DEVICE" 2>&1 | tee "$LOGS/preflight.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "Preflight failed. Nothing else ran."; exit 1; }
"$PY" -m pytest week4/tests -q -p no:cacheprovider 2>&1 | tee "$LOGS/tests.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "Tests failed. Nothing else ran."; exit 1; }
rm -rf week4/results/smoke
for stage in 4a 4b 4c; do
  $RUN --stage "$stage" --smoke --device "$DEVICE" >> "$LOGS/smoke.log" 2>&1 \
    || { log "Smoke run of stage $stage failed. See $LOGS/smoke.log"; exit 1; }
done
"$PY" week4/src/e6_shap.py --smoke >> "$LOGS/smoke.log" 2>&1 || { log "Smoke run of E6 failed. See $LOGS/smoke.log"; exit 1; }
"$PY" week4/src/analysis.py --smoke >> "$LOGS/smoke.log" 2>&1 || { log "Smoke run of the analysis failed. See $LOGS/smoke.log"; exit 1; }
log "Checks passed."

# 2. Lanes.
if [ "$PARALLEL_SEEDS" = 1 ]; then CPU_LANES=($SEEDS); else CPU_LANES=("$SEEDS"); fi
LANE_COUNT=$(( ${#CPU_LANES[@]} + 1 ))
THREADS=$(( CORES / LANE_COUNT )); [ "$THREADS" -ge 1 ] || THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
log "$LANE_COUNT lanes, $THREADS threads each."

gpu_lane() {
  set -e
  $RUN --stage e2 --arms tabpfn_v2 --device "$DEVICE"
  $RUN --stage e4 --arms tabpfn_v2 --device "$DEVICE"
  $RUN --stage e5a --device "$DEVICE"
  $RUN --stage e5b --device "$DEVICE" --h2-estimators "$H2_ESTIMATORS"
  $RUN --stage e4_hybrids --device "$DEVICE" --h2-estimators "$H2_ESTIMATORS"
}

cpu_lane() {
  set -e
  local seeds=$1
  $RUN --stage e2 --arms xgb xgb_spw mlp_erm --seeds $seeds
  $RUN --stage e3 --seeds $seeds
  $RUN --stage e4 --arms xgb mlp_erm mlp_dro --seeds $seeds
  $RUN --stage e7 --seeds $seeds
}

declare -A NAMES
gpu_lane > "$LOGS/lane_gpu.log" 2>&1 &
NAMES[$!]=gpu
for seeds in "${CPU_LANES[@]}"; do
  name="cpu_seeds_${seeds// /_}"
  cpu_lane "$seeds" > "$LOGS/lane_$name.log" 2>&1 &
  NAMES[$!]=$name
done

FAILED=0
for pid in "${!NAMES[@]}"; do
  if wait "$pid"; then log "Lane ${NAMES[$pid]} finished."; else log "Lane ${NAMES[$pid]} FAILED. See $LOGS/lane_${NAMES[$pid]}.log"; FAILED=1; fi
done

# 3. What finished. Writes week4/results/phase4_completion.md.
"$PY" week4/src/completion.py | tee -a "$LOGS/main.log"
COMPLETE=${PIPESTATUS[0]}

# 4. E6 (SHAP, CPU, seed 0; needs xgb, mlp_erm_wc and mlp_dro), then every metric, table and figure.
# The analysis skips any arm with missing folds and lists it, so it runs even after a partial failure.
export OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES
if "$PY" week4/src/e6_shap.py > "$LOGS/e6.log" 2>&1; then log "E6 finished."; else log "E6 FAILED. See $LOGS/e6.log"; FAILED=1; fi
if "$PY" week4/src/analysis.py > "$LOGS/analysis.log" 2>&1; then log "Metrics and figures written."; else log "Analysis FAILED. See $LOGS/analysis.log"; FAILED=1; fi

if [ "$FAILED" = 0 ] && [ "$COMPLETE" = 0 ]; then
  log "All Phase 4 runs finished. Commit week4/results and week4/figures, then push (week4/GPU_RUN.md, step 4)."
else
  log "Some runs failed or are missing. Rerun this script to resume; finished folds are skipped."
  exit 1
fi
