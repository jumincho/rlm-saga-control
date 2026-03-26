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

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v7_1_boundary_debug_$(date +%Y%m%d_%H%M%S)}"
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
  echo "[feedback_v7_1_boundary_debug] vLLM failed to start" >&2
  exit 1
}

nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).txt" || true

BASE="$RUN_ROOT/results/raw/baseline_boundary_debug.jsonl"
EXT="$RUN_ROOT/results/raw/extension_boundary_debug.jsonl"

set +e
python -m exp.runners.run_paired \
  --config exp/config/experiment_feedback_v7_1_boundary_debug.yaml \
  --stage stage_feedback_v7_1_boundary_debug \
  --variants V0 V3_PREFIX_SPLIT \
  --run-root "$RUN_ROOT" \
  --baseline-out "$BASE" \
  --extension-out "$EXT" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/run_boundary_debug.log"
RUN_EXIT=$?
set -e

python -m exp.analysis.check_pair_integrity \
  --baseline "$BASE" \
  --extension "$EXT" \
  --extension-variant V3_PREFIX_SPLIT \
  --out "$RUN_ROOT/results/summary/integrity_boundary_debug_v3_prefix_split.json" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_boundary_debug.log" || true

python -m exp.analysis.summarize \
  --baseline "$BASE" \
  --extension "$EXT" \
  --metrics-out "$RUN_ROOT/results/summary/metrics_boundary_debug_runtime.csv" \
  --report-out "$RUN_ROOT/reports/rlm_vs_v7_1_boundary_debug_runtime.md" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_boundary_debug_runtime.log" || true

python -m exp.analysis.summarize_dual_eval \
  --baseline "$BASE" \
  --extension "$EXT" \
  --out-dir "$RUN_ROOT/results/summary/boundary_debug_dual_eval" \
  --modes runtime strict relaxed \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_boundary_debug_dual.log" || true

python -m exp.analysis.paired_stats \
  --input-jsonl "$RUN_ROOT/results/summary/boundary_debug_dual_eval/rescored_runtime.jsonl" \
  --left V0 \
  --right V3_PREFIX_SPLIT \
  --out-json "$RUN_ROOT/results/summary/paired_stats_boundary_debug_runtime_v0_vs_v3_prefix_split.json" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_stats_boundary_debug_runtime.log" || true

python -m exp.analysis.summarize_split_failures \
  --baseline "$BASE" \
  --extension "$EXT" \
  --out-csv "$RUN_ROOT/results/summary/split_debug_summary.csv" \
  --out-md "$RUN_ROOT/reports/split_debug_summary.md" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/split_debug_summary.log" || true

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
