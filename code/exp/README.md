# RLM+Saga Experiment Framework (v1)

## Quick start

1. Create and bootstrap env:

```bash
bash exp/runners/bootstrap_env.sh /disk/chojm/experiments/rlm_saga_v1_$(date +%Y%m%d_%H%M%S)
source /disk/chojm/experiments/<run_id>/venv/bin/activate
```

2. Export credentials (do not print them):

```bash
export HUGGINGFACE_HUB_TOKEN="***"
export OPENAI_API_KEY="EMPTY"
```

3. Start vLLM (8 GPU required):

```bash
bash exp/runners/setup_vllm.sh
```

4. Run pilot:

```bash
python -m exp.runners.run_all --config exp/config/experiment.yaml --stage stage1
```

5. Run full stage:

```bash
python -m exp.runners.run_all --config exp/config/experiment.yaml --stage stage2
```

6. Plot summary:

```bash
python -m exp.analysis.plot \
  --metrics /disk/chojm/experiments/<run_id>/results/summary/metrics.csv \
  --out-dir /disk/chojm/experiments/<run_id>/results/summary
```

## Variants

- `V0`: plain RLM (`environment=local`)
- `V1`: RLM+Tx (`environment=local_saga`, tx only)
- `V2`: RLM+Tx+RuleVal
- `V3`: RLM+Tx+RuleVal+RetryPolicy

## Security

- Tokens/keys must be provided only through env vars.
- Rotate credentials if exposed in any logs or chat transcripts.
