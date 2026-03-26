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

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v7_5_full_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${2:-exp/config/experiment_feedback_v7_5.yaml}"
STAGE="${3:-stage_feedback_v7_5_boundary_full}"

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
nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).txt" || true

BASE="$RUN_ROOT/results/raw/baseline_boundary_full.jsonl"
EXT="$RUN_ROOT/results/raw/extension_boundary_full.jsonl"

python -m exp.runners.run_paired --config "$CONFIG_PATH" --stage "$STAGE" --variants V0 V0_SPLIT_ONLY V3_PREFIX_SPLIT --run-root "$RUN_ROOT" --baseline-out "$BASE" --extension-out "$EXT" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/run_boundary_full.log"

python -m exp.analysis.check_pair_integrity --baseline "$BASE" --extension "$EXT" --extension-variant V0_SPLIT_ONLY --out "$RUN_ROOT/results/summary/integrity_boundary_full_v0_split_only.json" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_boundary_full_v0_split_only.log"
python -m exp.analysis.check_pair_integrity --baseline "$BASE" --extension "$EXT" --extension-variant V3_PREFIX_SPLIT --out "$RUN_ROOT/results/summary/integrity_boundary_full_v3_prefix_split.json" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_boundary_full_v3_prefix_split.log"

python -m exp.analysis.summarize --baseline "$BASE" --extension "$EXT" --metrics-out "$RUN_ROOT/results/summary/metrics_boundary_full_runtime.csv" --report-out "$RUN_ROOT/reports/rlm_vs_v7_5_boundary_full_runtime.md" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_boundary_full_runtime.log"
python -m exp.analysis.summarize_dual_eval --baseline "$BASE" --extension "$EXT" --out-dir "$RUN_ROOT/results/summary/boundary_full_dual_eval" --modes runtime strict relaxed 2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_boundary_full_dual.log"

python -m exp.analysis.summarize_split_failures --baseline "$BASE" --extension "$EXT" --out-csv "$RUN_ROOT/results/summary/split_debug_boundary_full.csv" --out-md "$RUN_ROOT/reports/split_debug_boundary_full.md" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/split_debug_boundary_full.log"
python -m exp.analysis.dump_boundary_debug_cases --baseline "$BASE" --extension "$EXT" --variant V3_PREFIX_SPLIT --out-md "$RUN_ROOT/reports/boundary_debug_cases_boundary_full_v3_prefix_split.md" --max-events 40 2>&1 | tee "$RUN_ROOT/runs/runner_logs/boundary_debug_cases_v3.log"
python -m exp.analysis.dump_boundary_debug_cases --baseline "$BASE" --extension "$EXT" --variant V0_SPLIT_ONLY --out-md "$RUN_ROOT/reports/boundary_debug_cases_boundary_full_v0_split_only.md" --max-events 40 2>&1 | tee "$RUN_ROOT/runs/runner_logs/boundary_debug_cases_v0_split.log"

for MODE_NAME in runtime strict; do
  RESCORED="$RUN_ROOT/results/summary/boundary_full_dual_eval/rescored_${MODE_NAME}.jsonl"
  python -m exp.analysis.paired_stats --input-jsonl "$RESCORED" --left V0 --right V0_SPLIT_ONLY --out-json "$RUN_ROOT/results/summary/paired_stats_boundary_full_${MODE_NAME}_v0_vs_v0_split_only.json" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_boundary_full_${MODE_NAME}_v0_vs_v0_split_only.log"
  python -m exp.analysis.paired_stats --input-jsonl "$RESCORED" --left V0 --right V3_PREFIX_SPLIT --out-json "$RUN_ROOT/results/summary/paired_stats_boundary_full_${MODE_NAME}_v0_vs_v3_prefix_split.json" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_boundary_full_${MODE_NAME}_v0_vs_v3_prefix_split.log"
  python -m exp.analysis.paired_stats --input-jsonl "$RESCORED" --left V0_SPLIT_ONLY --right V3_PREFIX_SPLIT --out-json "$RUN_ROOT/results/summary/paired_stats_boundary_full_${MODE_NAME}_v0_split_only_vs_v3_prefix_split.json" 2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_boundary_full_${MODE_NAME}_v0_split_only_vs_v3_prefix_split.log"
done

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
