"""Run paired baseline + extension on exactly the same sample stream."""

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


VALID_VARIANTS = [
    "V0",
    "V0_SPLIT_ONLY",
    "V1",
    "V2",
    "V3",
    "V3_BASE",
    "V3_PREFIX",
    "V3_PREFIX_SPLIT",
    "V3_PREFIX_NO_SPLIT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired RLM vs RLM+Saga on identical samples.")
    parser.add_argument("--config", default="exp/config/experiment.yaml")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--variants", nargs="+", default=None, help="Include V0 and extension variants.")
    parser.add_argument("--baseline-out", default=None)
    parser.add_argument("--extension-out", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, args.stage)
    global_cfg = cfg["global"]
    stage_cfg = cfg["stage"]

    selected_variants = args.variants or (["V0"] + stage_cfg.get("extension_variants", ["V2"]))
    selected_variants = list(dict.fromkeys(selected_variants))
    for variant in selected_variants:
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
    if "V0" not in selected_variants:
        raise ValueError("Paired run requires V0 in --variants.")

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
        / f"rlm_saga_paired_{timestamp}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_out = Path(args.baseline_out) if args.baseline_out else run_root / "results" / "raw" / "baseline.jsonl"
    extension_out = (
        Path(args.extension_out) if args.extension_out else run_root / "results" / "raw" / "extension.jsonl"
    )

    log_dir = run_root / "runs" / "paired" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = global_cfg.get("model", {})
    if args.model_name:
        model_cfg = dict(model_cfg)
        model_cfg["model_name"] = args.model_name
    run_cfg = global_cfg.get("runtime", {})

    seeds = stage_cfg.get("seeds", [0])
    failure_ratio = float(stage_cfg.get("failure_injection_ratio", 0.0))

    total_samples = 0
    total_records = 0
    ext_variants = [v for v in selected_variants if v != "V0"]

    for seed in seeds:
        samples = load_stage_samples(global_cfg, stage_cfg, seed)
        total_samples += len(samples)

        for sample in tqdm(samples, desc=f"Paired seed={seed}"):
            injected = apply_failure_injection(sample, seed=seed, ratio=failure_ratio)

            # Always run baseline first, then extension variants on the same sample object.
            base_record = run_single_sample(
                variant="V0",
                sample=injected,
                seed=seed,
                model_cfg=model_cfg,
                run_cfg=run_cfg,
                log_dir=log_dir / "v0",
            )
            base_record["stage"] = args.stage
            append_jsonl(baseline_out, base_record)
            total_records += 1

            for variant in ext_variants:
                ext_record = run_single_sample(
                    variant=variant,
                    sample=injected,
                    seed=seed,
                    model_cfg=model_cfg,
                    run_cfg=run_cfg,
                    log_dir=log_dir / variant.lower(),
                )
                ext_record["stage"] = args.stage
                append_jsonl(extension_out, ext_record)
                total_records += 1

    print(
        f"[paired] samples={total_samples} records={total_records} variants={selected_variants} "
        f"-> baseline={baseline_out} extension={extension_out}"
    )


if __name__ == "__main__":
    main()
