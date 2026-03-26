#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
LOG_DIR="${LOG_DIR:-/disk/chojm/experiments/vllm_server_logs}"

mkdir -p "$LOG_DIR"

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HUGGINGFACE_HUB_TOKEN is not set"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="EMPTY"
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -lt 8 ]]; then
  echo "8GPU required, found $GPU_COUNT"
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$LOG_DIR/nvidia_smi_$(date +%Y%m%d_%H%M%S).txt"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --generation-config vllm \
  2>&1 | tee "$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"
