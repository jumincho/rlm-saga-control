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

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v7_2_$(date +%Y%m%d_%H%M%S)}"
MODE="${2:-full}" # mini | full
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
  echo "[feedback_v7_2] vLLM failed to start" >&2
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
    --config exp/config/experiment_feedback_v7_2.yaml \
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
    --report-out "$RUN_ROOT/reports/rlm_vs_v7_2_${TAG}_runtime.md" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_runtime.log" || true

  python -m exp.analysis.summarize_dual_eval \
    --baseline "$BASE" \
    --extension "$EXT" \
    --out-dir "$RUN_ROOT/results/summary/${TAG}_dual_eval" \
    --modes runtime strict relaxed \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_dual.log" || true

  if [[ "$TAG" == boundary_* ]]; then
    python -m exp.analysis.summarize_split_failures \
      --baseline "$BASE" \
      --extension "$EXT" \
      --out-csv "$RUN_ROOT/results/summary/split_debug_${TAG}.csv" \
      --out-md "$RUN_ROOT/reports/split_debug_${TAG}.md" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/split_debug_${TAG}.log" || true

    python -m exp.analysis.dump_boundary_debug_cases \
      --baseline "$BASE" \
      --extension "$EXT" \
      --variant V3_PREFIX_SPLIT \
      --out-md "$RUN_ROOT/reports/boundary_debug_cases_${TAG}.md" \
      --max-events 30 \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/boundary_debug_cases_${TAG}.log" || true
  fi

  return "$RUN_EXIT"
}

RUN_EXIT=0
GATE_PASSED=0
if [[ "${V72_SKIP_MINI:-0}" != "1" ]]; then
  run_stage stage_feedback_v7_2_boundary_mini boundary_mini V0 V3_PREFIX_SPLIT || RUN_EXIT=$?
else
  GATE_PASSED=1
fi

if [[ "$RUN_EXIT" -eq 0 && "$GATE_PASSED" -eq 0 ]]; then
  python - << 'PY' "$RUN_ROOT/results/summary/metrics_boundary_mini_runtime.csv" "$RUN_ROOT/results/raw/extension_boundary_mini.jsonl" > "$RUN_ROOT/results/summary/gate_boundary_mini.json"
import csv, json, sys
metrics_path = sys.argv[1]
ext_path = sys.argv[2]
row = None
with open(metrics_path, 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r.get('variant') == 'V3_PREFIX_SPLIT':
            row = r
            break
if row is None:
    out = {'passed': False, 'reason': 'missing_variant'}
    print(json.dumps(out, ensure_ascii=False))
    raise SystemExit(0)

split_applied = float(row.get('split_applied_runtime_rate', 0.0))
split_survived = float(row.get('split_marker_survived_rate', 0.0))
blocked = {'FAILED_PARSE', 'FAILED_CONFLICT_WITH_LOCK'}
blocked_count = 0
with open(ext_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get('split_failure_reason') in blocked:
            blocked_count += 1
passed = (split_applied >= 0.9) and (split_survived >= 0.9) and (blocked_count == 0)
out = {
    'passed': passed,
    'split_applied_runtime_rate': split_applied,
    'split_marker_survived_rate': split_survived,
    'blocked_failure_count': blocked_count,
}
print(json.dumps(out, ensure_ascii=False))
PY

  cat "$RUN_ROOT/results/summary/gate_boundary_mini.json" | tee "$RUN_ROOT/runs/runner_logs/gate_boundary_mini.log"
  if python - << 'PY' "$RUN_ROOT/results/summary/gate_boundary_mini.json"
import json,sys
obj=json.load(open(sys.argv[1],'r',encoding='utf-8'))
raise SystemExit(0 if obj.get('passed') else 1)
PY
  then
    GATE_PASSED=1
  else
    GATE_PASSED=0
  fi
fi

if [[ "$RUN_EXIT" -eq 0 && "$GATE_PASSED" -eq 1 && "$MODE" != "mini" ]]; then
  run_stage stage_feedback_v7_2_boundary_full boundary_full V0 V3_PREFIX_SPLIT || RUN_EXIT=$?
  if [[ "${V72_SKIP_REGRESSION:-0}" != "1" ]]; then
    run_stage stage_feedback_v7_2_main_regression main_regression V0 V3_PREFIX || RUN_EXIT=$?
    run_stage stage_feedback_v7_2_recovery_regression recovery_regression V0 V3_PREFIX || RUN_EXIT=$?
  fi
fi

# paired stats for completed stages
for T in boundary_mini boundary_full main_regression recovery_regression; do
  if [[ -f "$RUN_ROOT/results/summary/${T}_dual_eval/rescored_runtime.jsonl" ]]; then
    if [[ "$T" == boundary_* ]]; then RIGHT="V3_PREFIX_SPLIT"; else RIGHT="V3_PREFIX"; fi
    python -m exp.analysis.paired_stats \
      --input-jsonl "$RUN_ROOT/results/summary/${T}_dual_eval/rescored_runtime.jsonl" \
      --left V0 \
      --right "$RIGHT" \
      --out-json "$RUN_ROOT/results/summary/paired_stats_${T}_runtime_v0_vs_${RIGHT,,}.json" \
      2>&1 | tee "$RUN_ROOT/runs/runner_logs/paired_stats_${T}.log" || true
  fi
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

# if gate failed, make it explicit but keep zero exit for completed debug stage
if [[ "$RUN_EXIT" -eq 0 && "$GATE_PASSED" -eq 0 ]]; then
  echo "[feedback_v7_2] boundary mini gate failed; full stages skipped" | tee -a "$RUN_ROOT/runs/runner_logs/gate_boundary_mini.log"
  exit 2
fi

exit "$RUN_EXIT"
