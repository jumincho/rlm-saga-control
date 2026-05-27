#!/usr/bin/env bash
set -euo pipefail

# Defaults assume the original author's host layout. Override any of
# these via environment variables when running on a different machine.
SAGA_EXPERIMENTS_ROOT="${SAGA_EXPERIMENTS_ROOT:-/disk/chojm/experiments}"
SAGA_LLM_REQUIREMENTS="${SAGA_LLM_REQUIREMENTS:-/disk/chojm/SagaLLM/requirements.txt}"
SAGA_RLM_PACKAGE="${SAGA_RLM_PACKAGE:-/disk/chojm/rlm}"

RUN_ROOT="${1:-${SAGA_EXPERIMENTS_ROOT}/rlm_saga_v1_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RUN_ROOT"/{artifacts,reports,results/raw,results/summary,runs/vllm_server_logs}

if [[ ! -d "$RUN_ROOT/venv" ]]; then
  "$PYTHON_BIN" -m venv "$RUN_ROOT/venv"
fi

source "$RUN_ROOT/venv/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r "$SAGA_LLM_REQUIREMENTS"
python -m pip install -e "$SAGA_RLM_PACKAGE"
python -m pip install datasets pandas matplotlib pyyaml tqdm vllm
python -m pip freeze > "$RUN_ROOT/reports/requirements_freeze.txt"

echo "Bootstrap complete: $RUN_ROOT"
