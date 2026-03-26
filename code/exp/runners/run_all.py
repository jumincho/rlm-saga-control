"""Orchestrate baseline + extension + summarization."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full RLM+Saga experiment stage.")
    parser.add_argument("--config", default="exp/config/experiment.yaml")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-extension", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def ensure_8gpu(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    gpu_log = run_root / "runs" / "vllm_server_logs"
    gpu_log.mkdir(parents=True, exist_ok=True)

    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv",
    ]
    output = subprocess.check_output(query_cmd, text=True)
    with open(gpu_log / f"nvidia_smi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(output)

    gpu_lines = [line for line in output.strip().splitlines()[1:] if line.strip()]
    if len(gpu_lines) < 8:
        raise RuntimeError(f"8GPU required, found {len(gpu_lines)}")


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_root:
        run_root = Path(args.run_root)
    else:
        run_root = Path("experiments") / f"rlm_saga_v1_{timestamp}"

    ensure_8gpu(run_root)

    baseline_out = run_root / "results" / "raw" / "baseline.jsonl"
    extension_out = run_root / "results" / "raw" / "extension.jsonl"
    summary_out = run_root / "results" / "summary" / "metrics.csv"
    report_out = run_root / "reports" / "rlm_vs_rlm_saga_v1.md"

    env = os.environ.copy()

    if not args.skip_baseline:
        subprocess.run(
            [
                "python",
                "-m",
                "exp.runners.run_baseline",
                "--config",
                args.config,
                "--stage",
                args.stage,
                "--out",
                str(baseline_out),
                "--run-root",
                str(run_root),
            ],
            check=True,
            env=env,
        )

    if not args.skip_extension:
        subprocess.run(
            [
                "python",
                "-m",
                "exp.runners.run_extension",
                "--config",
                args.config,
                "--stage",
                args.stage,
                "--out",
                str(extension_out),
                "--run-root",
                str(run_root),
            ],
            check=True,
            env=env,
        )

    if not args.skip_summary:
        subprocess.run(
            [
                "python",
                "-m",
                "exp.analysis.summarize",
                "--baseline",
                str(baseline_out),
                "--extension",
                str(extension_out),
                "--metrics-out",
                str(summary_out),
                "--report-out",
                str(report_out),
            ],
            check=True,
            env=env,
        )

    print(f"[run_all] completed: {run_root}")


if __name__ == "__main__":
    main()
