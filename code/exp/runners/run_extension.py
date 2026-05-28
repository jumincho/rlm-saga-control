"""Extension runner: RLM+Saga variants V1 / V2 / V3.

Runs the Saga-control extensions over the same sample stream the baseline
sees. The three variants correspond to a stepped buildup of the control
story:

- V1 — RLM + transaction layer only (rollback/retry, no validator).
- V2 — V1 + rule-based runtime validator (accept / augment / reject
  decisions on the candidate plan before commit).
- V3 — V2 + retry policy + prefix-lock + boundary-split / deterministic
  recovery. "Full Saga control" in the closure report.

`paired` is the more common entry point in the v7.* rounds because it
forces V0 and these variants to see exactly the same samples; this
runner is the path for running extensions alone (e.g., to add a missing
variant to an already-completed baseline run).
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


VALID_VARIANTS = ["V1", "V2", "V3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extension RLM+Saga experiments.")
    parser.add_argument("--config", default="exp/config/experiment.yaml")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--out", default=None, help="Path to output JSONL")
    parser.add_argument("--run-root", default=None, help="Optional run root directory")
    parser.add_argument("--model-name", default=None, help="Override model name for this run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, args.stage)
    global_cfg = cfg["global"]
    stage_cfg = cfg["stage"]

    selected_variants = args.variants or stage_cfg.get("extension_variants", ["V1", "V2", "V3"])
    for variant in selected_variants:
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown extension variant: {variant}")

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

    out_path = Path(args.out) if args.out else run_root / "results" / "raw" / "extension.jsonl"
    log_dir = run_root / "runs" / "extension" / "logs"
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
        for sample in tqdm(samples, desc=f"Extension seed={seed}"):
            injected = apply_failure_injection(sample, seed=seed, ratio=failure_ratio)
            for variant in selected_variants:
                record = run_single_sample(
                    variant=variant,
                    sample=injected,
                    seed=seed,
                    model_cfg=model_cfg,
                    run_cfg=run_cfg,
                    log_dir=log_dir / variant.lower(),
                )
                record["stage"] = args.stage
                append_jsonl(out_path, record)
                total += 1

    print(f"[extension] completed rows={total} -> {out_path}")


if __name__ == "__main__":
    main()
