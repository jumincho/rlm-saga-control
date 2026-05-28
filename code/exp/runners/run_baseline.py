"""Hands-off baseline runner: V0 (plain RLM, no Saga control).

The "hands-off" regime in the three-way comparison. The runner loads the
stage's sample stream (REALM-Bench planning + LongBench-v2 + OOLONG, per
the YAML config), optionally injects failures deterministically by
sample id, and writes one JSONL row per sample under `results/raw/`.
No prefix lock, no validator-driven retry, no Saga recovery — whatever
the model emits gets scored as-is. This is the reference the other two
regimes (light branching V1/V2, full Saga control V3) get paired against
in `run_paired.py`.

Run directory layout is shared with the extension runners so that the
analysis modules can merge baseline + extension JSONL into a single
DataFrame keyed by (sample_id, seed, stage).
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from exp.runners.common import (
    append_jsonl,
    apply_failure_injection,
    load_experiment_config,
    load_stage_samples,
    run_single_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline RLM experiments (V0).")
    parser.add_argument("--config", default="exp/config/experiment.yaml")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--out", default=None, help="Path to output JSONL")
    parser.add_argument("--run-root", default=None, help="Optional run root directory")
    parser.add_argument("--model-name", default=None, help="Override model name for this run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, args.stage)
    global_cfg = cfg["global"]
    stage_cfg = cfg["stage"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (
        Path(args.run_root)
        if args.run_root
        else Path(
            os.environ.get(
                "RLM_SAGA_RUN_ROOT_BASE",
                global_cfg.get("paths", {}).get("run_root_base", "experiments"),
            )
        )
        / f"rlm_saga_v1_{timestamp}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else run_root / "results" / "raw" / "baseline.jsonl"
    log_dir = run_root / "runs" / "baseline" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = global_cfg.get("model", {})
    if args.model_name:
        model_cfg = dict(model_cfg)
        model_cfg["model_name"] = args.model_name
    run_cfg = global_cfg.get("runtime", {})
    seeds = stage_cfg.get("seeds", [0])
    failure_ratio = float(stage_cfg.get("failure_injection_ratio", 0.0))

    total = 0
    for seed in seeds:
        samples = load_stage_samples(global_cfg, stage_cfg, seed)
        total += len(samples)

        for sample in tqdm(samples, desc=f"Baseline seed={seed}"):
            injected = apply_failure_injection(sample, seed=seed, ratio=failure_ratio)
            record = run_single_sample(
                variant="V0",
                sample=injected,
                seed=seed,
                model_cfg=model_cfg,
                run_cfg=run_cfg,
                log_dir=log_dir,
            )
            record["stage"] = args.stage
            append_jsonl(out_path, record)

    print(f"[baseline] completed rows={total} -> {out_path}")


if __name__ == "__main__":
    main()
