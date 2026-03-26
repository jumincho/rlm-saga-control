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

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v2_live_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/runs/vllm_server_logs" "$RUN_ROOT/runs/runner_logs" "$RUN_ROOT/results/raw" "$RUN_ROOT/results/summary" "$RUN_ROOT/reports"
echo "$RUN_ROOT" > /tmp/rlm_saga_feedback_v2_run_root.txt

# Stop stale server on the same port if present.
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

for i in $(seq 1 240); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "[feedback_v2] vLLM failed to start" >&2
  exit 1
fi

nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).txt" || true

set +e
python -m exp.runners.run_paired \
  --config exp/config/experiment_feedback_v2.yaml \
  --stage stage_feedback_v2_main \
  --variants V0 V1 V2 V3 \
  --run-root "$RUN_ROOT" \
  --baseline-out "$RUN_ROOT/results/raw/baseline.jsonl" \
  --extension-out "$RUN_ROOT/results/raw/extension.jsonl" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/run_paired.log"
RUN_EXIT=$?
set -e

python -m exp.analysis.summarize \
  --baseline "$RUN_ROOT/results/raw/baseline.jsonl" \
  --extension "$RUN_ROOT/results/raw/extension.jsonl" \
  --metrics-out "$RUN_ROOT/results/summary/metrics.csv" \
  --report-out "$RUN_ROOT/reports/rlm_vs_rlm_saga_feedback_v2.md" \
  2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize.log" || true

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
