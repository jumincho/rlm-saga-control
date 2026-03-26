#!/usr/bin/env bash
set -euo pipefail

cd /disk/chojm
source /disk/chojm/experiments/rlm_saga_v1_20260219_122908/venv/bin/activate

: "${HUGGINGFACE_HUB_TOKEN:?HUGGINGFACE_HUB_TOKEN must be set}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_followup32_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/runs/vllm_server_logs" "$RUN_ROOT/runs/followup_logs" "$RUN_ROOT/results/raw" "$RUN_ROOT/results/summary" "$RUN_ROOT/reports"

echo "$RUN_ROOT" > /tmp/rlm_saga_followup32_run_root.txt

# 1) start vLLM 32B
export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
export TP_SIZE="8"
export PORT="8000"
export LOG_DIR="$RUN_ROOT/runs/vllm_server_logs"

nohup bash /disk/chojm/exp/runners/setup_vllm.sh > "$RUN_ROOT/runs/followup_logs/vllm_driver.log" 2>&1 &
VLLM_DRIVER_PID=$!
echo "$VLLM_DRIVER_PID" > "$RUN_ROOT/runs/followup_logs/vllm_driver.pid"

# wait for server up
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "[followup32] vLLM failed to start" >&2
  exit 1
fi

# 2) paired run (V0 vs V2 only)
set +e
python -m exp.runners.run_paired \
  --config exp/config/experiment_followup32.yaml \
  --stage stage_followup32 \
  --variants V0 V2 \
  --run-root "$RUN_ROOT" \
  --baseline-out "$RUN_ROOT/results/raw/baseline.jsonl" \
  --extension-out "$RUN_ROOT/results/raw/extension.jsonl"
RUN_EXIT=$?
set -e

# 3) summarize partial/final regardless of exit
python -m exp.analysis.summarize \
  --baseline "$RUN_ROOT/results/raw/baseline.jsonl" \
  --extension "$RUN_ROOT/results/raw/extension.jsonl" \
  --metrics-out "$RUN_ROOT/results/summary/metrics.csv" \
  --report-out "$RUN_ROOT/reports/rlm_vs_rlm_saga_followup32.md" || true

# 4) stop vLLM + free GPU
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

# 5) keep workspace tidy: remove tiny empty helper run dirs from prior aborted starts
find /disk/chojm/experiments -maxdepth 1 -type d -name 'rlm_saga_followup32_*' -empty -print -delete || true

exit "$RUN_EXIT"
