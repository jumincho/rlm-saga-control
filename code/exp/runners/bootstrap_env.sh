#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/disk/chojm/experiments/rlm_saga_v1_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RUN_ROOT"/{artifacts,reports,results/raw,results/summary,runs/vllm_server_logs}

if [[ ! -d "$RUN_ROOT/venv" ]]; then
  "$PYTHON_BIN" -m venv "$RUN_ROOT/venv"
fi

source "$RUN_ROOT/venv/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r /disk/chojm/SagaLLM/requirements.txt
python -m pip install -e /disk/chojm/rlm
python -m pip install datasets pandas matplotlib pyyaml tqdm vllm
python -m pip freeze > "$RUN_ROOT/reports/requirements_freeze.txt"

echo "Bootstrap complete: $RUN_ROOT"
