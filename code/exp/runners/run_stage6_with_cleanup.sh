#!/usr/bin/env bash
set -euo pipefail

cd /disk/chojm
source /disk/chojm/experiments/rlm_saga_v1_20260219_122908/venv/bin/activate

: "${HUGGINGFACE_HUB_TOKEN:?HUGGINGFACE_HUB_TOKEN must be set}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_stage6_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
mkdir -p "$RUN_ROOT/runs/stage6_logs" "$RUN_ROOT/runs/vllm_server_logs"

echo "$RUN_ROOT" > /tmp/rlm_saga_stage6_run_root.txt

VLLM_PID="$(pgrep -f 'vllm.entrypoints.openai.api_server.*--port 8000' | head -n 1 || true)"
echo "[stage6] run_root=$RUN_ROOT"
echo "[stage6] vllm_pid_before=$VLLM_PID"

set +e
python -m exp.runners.run_all \
  --config exp/config/experiment_6h.yaml \
  --stage stage_6h \
  --run-root "$RUN_ROOT"
EXP_EXIT=$?
set -e

echo "[stage6] experiment_exit_code=$EXP_EXIT"

if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
  echo "[stage6] stopping vllm pid=$VLLM_PID"
  kill "$VLLM_PID" || true
  sleep 5
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[stage6] vllm still alive, force kill"
    kill -9 "$VLLM_PID" || true
  fi
else
  echo "[stage6] vllm pid not found or already stopped"
fi

nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_after_release_$(date +%Y%m%d_%H%M%S).txt" || true

exit "$EXP_EXIT"
