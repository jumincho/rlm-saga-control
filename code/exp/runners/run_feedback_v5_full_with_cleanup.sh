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

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_feedback_v5_full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p \
  "$RUN_ROOT/runs/vllm_server_logs" \
  "$RUN_ROOT/runs/runner_logs" \
  "$RUN_ROOT/results/raw" \
  "$RUN_ROOT/results/summary" \
  "$RUN_ROOT/reports"
echo "$RUN_ROOT" > /tmp/rlm_saga_feedback_v5_full_run_root.txt

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
  echo "[feedback_v5_full] vLLM failed to start" >&2
  exit 1
fi

nvidia-smi > "$RUN_ROOT/runs/vllm_server_logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).txt" || true

run_stage_variant() {
  local STAGE="$1"
  local TAG="$2"
  local VARIANT="$3"
  local LOWER
  LOWER="$(echo "$VARIANT" | tr '[:upper:]' '[:lower:]')"
  local BASE="$RUN_ROOT/results/raw/baseline_${TAG}_${LOWER}.jsonl"
  local EXT="$RUN_ROOT/results/raw/extension_${TAG}_${LOWER}.jsonl"

  set +e
  python -m exp.runners.run_paired \
    --config exp/config/experiment_feedback_v5_full.yaml \
    --stage "$STAGE" \
    --variants V0 "$VARIANT" \
    --run-root "$RUN_ROOT" \
    --baseline-out "$BASE" \
    --extension-out "$EXT" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/run_${TAG}_${LOWER}.log"
  local RUN_EXIT=$?
  set -e

  python -m exp.analysis.check_pair_integrity \
    --baseline "$BASE" \
    --extension "$EXT" \
    --out "$RUN_ROOT/results/summary/integrity_${TAG}_${LOWER}.json" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/integrity_${TAG}_${LOWER}.log" || true

  python -m exp.analysis.summarize \
    --baseline "$BASE" \
    --extension "$EXT" \
    --metrics-out "$RUN_ROOT/results/summary/metrics_${TAG}_${LOWER}_runtime.csv" \
    --report-out "$RUN_ROOT/reports/rlm_vs_${LOWER}_${TAG}_runtime.md" \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_${LOWER}_runtime.log" || true

  python -m exp.analysis.summarize_dual_eval \
    --baseline "$BASE" \
    --extension "$EXT" \
    --out-dir "$RUN_ROOT/results/summary/${TAG}_${LOWER}_dual_eval" \
    --modes runtime strict relaxed \
    2>&1 | tee "$RUN_ROOT/runs/runner_logs/summarize_${TAG}_${LOWER}_dual.log" || true

  return "$RUN_EXIT"
}

RUN_EXIT=0

run_stage_variant stage_feedback_v5_smoke smoke V3_BASE || RUN_EXIT=$?
run_stage_variant stage_feedback_v5_smoke smoke V3_PREFIX || RUN_EXIT=$?

if [[ "$RUN_EXIT" -eq 0 ]]; then
  run_stage_variant stage_feedback_v5_main main V3_BASE || RUN_EXIT=$?
  run_stage_variant stage_feedback_v5_recovery recovery V3_BASE || RUN_EXIT=$?
  run_stage_variant stage_feedback_v5_main main V3_PREFIX || RUN_EXIT=$?
  run_stage_variant stage_feedback_v5_recovery recovery V3_PREFIX || RUN_EXIT=$?
fi

python - <<'PY' > "$RUN_ROOT/reports/feedback_v5_quick_compare.md" || true
import csv
import pathlib

run_root = pathlib.Path(open('/tmp/rlm_saga_feedback_v5_full_run_root.txt').read().strip())
rows = []
for stage in ["main", "recovery"]:
    for variant in ["v3_base", "v3_prefix"]:
        p = run_root / "results" / "summary" / f"metrics_{stage}_{variant}_runtime.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["stage"] = stage
                row["exp_variant"] = variant
                rows.append(row)

def pick(exp_variant, model_variant, stage):
    for r in rows:
        if r["exp_variant"] == exp_variant and r["variant"] == model_variant and r["stage"] == stage:
            return r
    return None

print("# Feedback v5 Quick Compare\n")
print("| stage | experiment | variant | success_rate | avg_violation_count | wall_time_p50 | tokens_p50 | disruption_applied_rate | partial_compensation_rate | immutable_prefix_rate | state_at_alert_consistent_rate |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for stage in ["main", "recovery"]:
    for exp_variant in ["v3_base", "v3_prefix"]:
        for model_variant in ["V0", "V3_BASE", "V3_PREFIX"]:
            r = pick(exp_variant, model_variant, stage)
            if r is None:
                continue
            print(
                f"| {stage} | {exp_variant} | {model_variant} | "
                f"{float(r['success_rate']):.4f} | {float(r['avg_violation_count']):.4f} | "
                f"{float(r['wall_time_p50']):.2f} | {float(r['tokens_p50']):.1f} | "
                f"{float(r.get('disruption_applied_rate', 0.0)):.4f} | "
                f"{float(r.get('partial_compensation_rate', 0.0)):.4f} | "
                f"{float(r.get('immutable_prefix_rate', 0.0)):.4f} | "
                f"{float(r.get('state_at_alert_consistent_rate', 0.0)):.4f} |"
            )
PY

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
