#!/usr/bin/env bash
set -euo pipefail

cd /disk/chojm
source /disk/chojm/experiments/rlm_saga_v1_20260219_122908/venv/bin/activate

: "${HUGGINGFACE_HUB_TOKEN:?HUGGINGFACE_HUB_TOKEN must be set}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/disk/chojm/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v7_5_$(date +%Y%m%d_%H%M%S)}"
MODE="${2:-full}" # smoke | full | boundary_only
CONFIG_PATH="${V75_CONFIG_PATH:-exp/config/experiment_feedback_v7_5.yaml}"
SMOKE_STAGE="${V75_SMOKE_STAGE:-stage_feedback_v7_5_smoke}"
BOUNDARY_STAGE="${V75_BOUNDARY_STAGE:-stage_feedback_v7_5_boundary_full}"
MAIN_STAGE="${V75_MAIN_STAGE:-stage_feedback_v7_5_main_regression}"
RECOVERY_STAGE="${V75_RECOVERY_STAGE:-stage_feedback_v7_5_recovery_regression}"
BOUNDARY_VARIANTS_STR="${V75_BOUNDARY_VARIANTS:-V0 V0_SPLIT_ONLY V3_PREFIX_SPLIT}"
IFS=' ' read -r -a BOUNDARY_VARIANTS <<< "$BOUNDARY_VARIANTS_STR"

mkdir -p \
  "$RUN_ROOT/runs/vllm_server_logs" \
  "$RUN_ROOT/runs/runner_logs" \
  "$RUN_ROOT/results/raw" \
  "$RUN_ROOT/results/summary" \
  "$RUN_ROOT/reports"

OLD_VLLM_PID="$(pgrep -f 'vllm.entrypoints.openai.api_server.*--port 8000' | head -n 1 || true)"
if [[ -n "$OLD_VLLM_PID" ]] && kill -0 "$OLD_VLLM_PID" 2>/dev/null; then
  kill "$OLD_VLLM_PID" || true
  sleep 5
  kill -9 "$OLD_VLLM_PID" 2>/dev/null || true
fi

export MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
export TP_SIZE="8"
export PORT="8000"
export LOG_DIR="$RUN_ROOT/runs/vllm_server_logs"
nohup bash /disk/chojm/exp/runners/setup_vllm.sh > "$RUN_ROOT/runs/runner_logs/vllm_driver.log" 2>&1 &
VLLM_DRIVER_PID=$!
echo "$VLLM_DRIVER_PID" > "$RUN_ROOT/runs/runner_logs/vllm_driver.pid"

for _ in $(seq 1 240); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -s http://localhost:8000/v1/models >/dev/null 2>&1 || {
  echo "[feedback_v7_5] vLLM failed to start" >&2
  exit 1
}

nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).txt" || true

run_stage() {
  local STAGE="$1"
  local TAG="$2"
  shift 2
  local VARIANTS=("$@")
  local BASE="$RUN_ROOT/results/raw/baseline_${TAG}.jsonl"
  local EXT="$RUN_ROOT/results/raw/extension_${TAG}.jsonl"

  set +e
  python -m exp.runners.run_paired \
    --config "$CONFIG_PATH" \
    --stage "$STAGE" \
    --variants "${VARIANTS[@]}" \
    --run-root "$RUN_ROOT" \
    --baseline-out "$BASE" \
    --extension-out "$EXT" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/run_${TAG}.log"
  local RUN_EXIT=$?
  set -e

  for V in "${VARIANTS[@]}"; do
    if [[ "$V" == "V0" ]]; then
      continue
    fi
    python -m exp.analysis.check_pair_integrity \
      --baseline "$BASE" \
      --extension "$EXT" \
      --extension-variant "$V" \
      --out "$RUN_ROOT/results/summary/integrity_${TAG}_${V,,}.json" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_${TAG}_${V,,}.log" || true
  done

  python -m exp.analysis.summarize \
    --baseline "$BASE" \
    --extension "$EXT" \
    --metrics-out "$RUN_ROOT/results/summary/metrics_${TAG}_runtime.csv" \
    --report-out "$RUN_ROOT/reports/rlm_vs_v7_5_${TAG}_runtime.md" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_runtime.log" || true

  python -m exp.analysis.summarize_dual_eval \
    --baseline "$BASE" \
    --extension "$EXT" \
    --out-dir "$RUN_ROOT/results/summary/${TAG}_dual_eval" \
    --modes runtime strict relaxed \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_dual.log" || true

  if [[ "$TAG" == boundary_* || "$TAG" == smoke* ]]; then
    python -m exp.analysis.summarize_split_failures \
      --baseline "$BASE" \
      --extension "$EXT" \
      --out-csv "$RUN_ROOT/results/summary/split_debug_${TAG}.csv" \
      --out-md "$RUN_ROOT/reports/split_debug_${TAG}.md" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/split_debug_${TAG}.log" || true

    for VV in V3_PREFIX_SPLIT V0_SPLIT_ONLY; do
      python -m exp.analysis.dump_boundary_debug_cases \
        --baseline "$BASE" \
        --extension "$EXT" \
        --variant "$VV" \
        --out-md "$RUN_ROOT/reports/boundary_debug_cases_${TAG}_${VV,,}.md" \
        --max-events 40 \
        2>&1 | tee "$RUN_ROOT/runs/runner_logs/boundary_debug_cases_${TAG}_${VV,,}.log" || true
    done
  fi

  for MODE_NAME in runtime strict; do
    local RESCORED="$RUN_ROOT/results/summary/${TAG}_dual_eval/rescored_${MODE_NAME}.jsonl"
    if [[ -f "$RESCORED" ]]; then
      python -m exp.analysis.paired_stats \
        --input-jsonl "$RESCORED" \
        --left V0 \
        --right V0_SPLIT_ONLY \
        --out-json "$RUN_ROOT/results/summary/paired_stats_${TAG}_${MODE_NAME}_v0_vs_v0_split_only.json" \
        2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_${TAG}_${MODE_NAME}_v0_vs_v0_split_only.log" || true

      python -m exp.analysis.paired_stats \
        --input-jsonl "$RESCORED" \
        --left V0 \
        --right V3_PREFIX_SPLIT \
        --out-json "$RUN_ROOT/results/summary/paired_stats_${TAG}_${MODE_NAME}_v0_vs_v3_prefix_split.json" \
        2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_${TAG}_${MODE_NAME}_v0_vs_v3_prefix_split.log" || true

      python -m exp.analysis.paired_stats \
        --input-jsonl "$RESCORED" \
        --left V0_SPLIT_ONLY \
        --right V3_PREFIX_SPLIT \
        --out-json "$RUN_ROOT/results/summary/paired_stats_${TAG}_${MODE_NAME}_v0_split_only_vs_v3_prefix_split.json" \
        2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_${TAG}_${MODE_NAME}_v0_split_only_vs_v3_prefix_split.log" || true
    fi
  done

  return "$RUN_EXIT"
}

RUN_EXIT=0

if [[ "$MODE" == "smoke" ]]; then
  run_stage "$SMOKE_STAGE" smoke_boundary "${BOUNDARY_VARIANTS[@]}" || RUN_EXIT=$?
else
  run_stage "$SMOKE_STAGE" smoke_boundary "${BOUNDARY_VARIANTS[@]}" || RUN_EXIT=$?
  if [[ "$RUN_EXIT" -eq 0 ]]; then
    run_stage "$BOUNDARY_STAGE" boundary_full "${BOUNDARY_VARIANTS[@]}" || RUN_EXIT=$?
  fi
  if [[ "$RUN_EXIT" -eq 0 && "$MODE" != "boundary_only" && "${V75_SKIP_REGRESSION:-1}" != "1" ]]; then
    run_stage "$MAIN_STAGE" main_regression V0 V3_PREFIX || RUN_EXIT=$?
    run_stage "$RECOVERY_STAGE" recovery_regression V0 V3_PREFIX || RUN_EXIT=$?
  fi
fi

VLLM_PID="$(pgrep -f 'vllm.entrypoints.openai.api_server.*--port 8000' | head -n 1 || true)"
if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
  kill "$VLLM_PID" || true
  sleep 5
  kill -9 "$VLLM_PID" 2>/dev/null || true
fi
if [[ -n "${VLLM_DRIVER_PID:-}" ]] && kill -0 "$VLLM_DRIVER_PID" 2>/dev/null; then
  kill "$VLLM_DRIVER_PID" || true
fi

nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_after_release_$(date +%Y%m%d_%H%M%S).txt" || true

exit "$RUN_EXIT"
