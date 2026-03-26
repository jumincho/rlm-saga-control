"""Plot helper for summarized metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot summary metrics.")
    parser.add_argument("--metrics", required=True, help="CSV from summarize.py")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Success rate plot.
    plt.figure(figsize=(8, 4))
    plt.bar(df["variant"], df["success_rate"])
    plt.ylim(0, 1)
    plt.title("Success Rate by Variant")
    plt.ylabel("Success Rate")
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate.png", dpi=160)
    plt.close()

    # Tail latency plot.
    plt.figure(figsize=(8, 4))
    plt.plot(df["variant"], df["wall_time_p95"], marker="o", label="P95")
    plt.plot(df["variant"], df["wall_time_p99"], marker="o", label="P99")
    plt.title("Wall Time Tail by Variant")
    plt.ylabel("Seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wall_time_tail.png", dpi=160)
    plt.close()

    # Token tail plot.
    plt.figure(figsize=(8, 4))
    plt.plot(df["variant"], df["tokens_p95"], marker="o", label="P95")
    plt.plot(df["variant"], df["tokens_p99"], marker="o", label="P99")
    plt.title("Token Tail by Variant")
    plt.ylabel("Tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "token_tail.png", dpi=160)
    plt.close()

    print(f"[plot] wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
