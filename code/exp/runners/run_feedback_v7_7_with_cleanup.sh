#!/usr/bin/env bash
set -euo pipefail

cd /disk/chojm
source /disk/chojm/experiments/rlm_saga_v1_20260219_122908/venv/bin/activate

export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-dummy}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/disk/chojm/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v7_7_full_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${2:-exp/config/experiment_feedback_v7_7.yaml}"

mkdir -p "$RUN_ROOT/runs/vllm_server_logs" "$RUN_ROOT/runs/runner_logs" "$RUN_ROOT/results/raw" "$RUN_ROOT/results/summary" "$RUN_ROOT/reports"

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
curl -s http://localhost:8000/v1/models >/dev/null 2>&1
nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_start_$(date +%Y%m%d_%H%M%S).txt" || true

sha256sum \
  /disk/chojm/exp/bench/scorer.py \
  /disk/chojm/exp/bench/evaluator_offline.py \
  /disk/chojm/exp/bench/validator_runtime.py \
  > "$RUN_ROOT/runs/runner_logs/evaluator_lock_sha256.txt"

# Pre-run alignment snapshot from latest v7.6 holdout test (definition-alignment baseline).
LATEST_V76="$(ls -dt /disk/chojm/experiments/rlm_saga_v1_feedback_v7_6_full_rerun_* /disk/chojm/experiments/rlm_saga_v1_feedback_v7_6_full_* 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_V76" ]] && [[ -f "$LATEST_V76/results/raw/extension_v7_6_boundary_test.jsonl" ]]; then
  python -m exp.analysis.immutability_alignment_report \
    --input-jsonl "$LATEST_V76/results/raw/extension_v7_6_boundary_test.jsonl" \
    --variant V3_PREFIX_SPLIT \
    --out-md "$RUN_ROOT/reports/immutability_alignment_report_v7_6_test.md" \
    --out-csv "$RUN_ROOT/results/summary/immutability_alignment_v7_6_test.csv" \
    --max-cases 20 \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/immutability_alignment_v7_6_test.log"
fi

run_stage() {
  local STAGE="$1"
  local TAG="$2"
  local BASE="$RUN_ROOT/results/raw/baseline_${TAG}.jsonl"
  local EXT="$RUN_ROOT/results/raw/extension_${TAG}.jsonl"
  local SUM_DIR="$RUN_ROOT/results/summary/${TAG}"
  mkdir -p "$SUM_DIR"

  python -m exp.runners.run_paired \
    --config "$CONFIG_PATH" \
    --stage "$STAGE" \
    --variants V0 V0_SPLIT_ONLY V3_PREFIX_SPLIT \
    --run-root "$RUN_ROOT" \
    --baseline-out "$BASE" \
    --extension-out "$EXT" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/run_${TAG}.log"

  python -m exp.analysis.check_pair_integrity \
    --baseline "$BASE" \
    --extension "$EXT" \
    --extension-variant V0_SPLIT_ONLY \
    --out "$SUM_DIR/integrity_v0_split_only.json" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_${TAG}_v0_split_only.log"

  python -m exp.analysis.check_pair_integrity \
    --baseline "$BASE" \
    --extension "$EXT" \
    --extension-variant V3_PREFIX_SPLIT \
    --out "$SUM_DIR/integrity_v3_prefix_split.json" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_${TAG}_v3_prefix_split.log"

  python -m exp.analysis.summarize \
    --baseline "$BASE" \
    --extension "$EXT" \
    --metrics-out "$SUM_DIR/metrics_runtime.csv" \
    --report-out "$RUN_ROOT/reports/rlm_vs_${TAG}_runtime.md" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_runtime.log"

  python -m exp.analysis.summarize_dual_eval \
    --baseline "$BASE" \
    --extension "$EXT" \
    --out-dir "$SUM_DIR/dual_eval" \
    --modes runtime strict relaxed \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_dual_eval.log"

  python -m exp.analysis.summarize_split_failures \
    --baseline "$BASE" \
    --extension "$EXT" \
    --out-csv "$SUM_DIR/split_debug.csv" \
    --out-md "$RUN_ROOT/reports/split_debug_${TAG}.md" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/split_debug_${TAG}.log"

  python -m exp.analysis.immutability_alignment_report \
    --input-jsonl "$EXT" \
    --variant V3_PREFIX_SPLIT \
    --out-md "$RUN_ROOT/reports/immutability_alignment_${TAG}.md" \
    --out-csv "$SUM_DIR/immutability_alignment_${TAG}.csv" \
    --max-cases 20 \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/immutability_alignment_${TAG}.log"

  python -m exp.analysis.dump_boundary_debug_cases \
    --baseline "$BASE" \
    --extension "$EXT" \
    --variant V3_PREFIX_SPLIT \
    --out-md "$RUN_ROOT/reports/boundary_debug_cases_${TAG}_v3_prefix_split.md" \
    --max-events 60 \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/boundary_debug_cases_${TAG}_v3.log"

  for MODE_NAME in runtime strict; do
    RESCORED="$SUM_DIR/dual_eval/rescored_${MODE_NAME}.jsonl"
    python -m exp.analysis.paired_stats \
      --input-jsonl "$RESCORED" \
      --left V0 \
      --right V0_SPLIT_ONLY \
      --out-json "$SUM_DIR/paired_${MODE_NAME}_v0_vs_v0_split_only.json" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_${TAG}_${MODE_NAME}_v0_vs_v0_split_only.log"

    python -m exp.analysis.paired_stats \
      --input-jsonl "$RESCORED" \
      --left V0 \
      --right V3_PREFIX_SPLIT \
      --out-json "$SUM_DIR/paired_${MODE_NAME}_v0_vs_v3_prefix_split.json" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_${TAG}_${MODE_NAME}_v0_vs_v3_prefix_split.log"

    python -m exp.analysis.paired_stats \
      --input-jsonl "$RESCORED" \
      --left V0_SPLIT_ONLY \
      --right V3_PREFIX_SPLIT \
      --out-json "$SUM_DIR/paired_${MODE_NAME}_v0_split_only_vs_v3_prefix_split.json" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_${TAG}_${MODE_NAME}_v0_split_only_vs_v3_prefix_split.log"
  done
}

run_stage "stage_feedback_v7_7_smoke" "v7_7_smoke"
run_stage "stage_feedback_v7_7_debug" "v7_7_debug"
run_stage "stage_feedback_v7_7_boundary_dev" "v7_7_boundary_dev"
run_stage "stage_feedback_v7_7_boundary_test" "v7_7_boundary_test"
run_stage "stage_feedback_v7_7_boundary_test_v2" "v7_7_boundary_test_v2"

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

echo "RUN_ROOT=$RUN_ROOT"
