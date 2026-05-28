"""Aggregate the runtime split-attempt bookkeeping by variant.

A focused debug summary. For each variant, reports: how often a
boundary split was attempted (`split_attempted_rate`), how many
attempts on average, the rate at which the attempt actually produced
a `boundary_split_pre` / `boundary_split_post` pair (`split_applied
_runtime_rate`), whether those markers survived into the final commit
(`split_marker_survived_rate`), and the distribution of
`split_apply_mode` values:

- `REAL_CROSSING_FOUND`                       — a real travel segment
  crossed the alert boundary and was split.
- `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`    — no real crossing, but
  a synthetic crossing was reconstructed from the actor state.
- `SYNTHETIC_INSERTED_NO_TRAVEL_FOUND`        — no travel data
  available; a synthetic pre/post pair was inserted.

Plus a top-N table of `split_failure_reason`. Inputs are the raw
JSONL (no re-scoring needed); used in the v7.1 — v7.3 boundary debug
rounds to chase `split_applied_runtime_rate` up to 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize split failure reasons")
    p.add_argument("--baseline", required=True)
    p.add_argument("--extension", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.baseline) + read_jsonl(args.extension)
    if not rows:
        raise RuntimeError("No rows found")
    df = pd.DataFrame(rows)

    for col in [
        "split_attempted",
        "split_attempt_count",
        "split_applied_runtime",
        "split_marker_survived",
    ]:
        if col not in df.columns:
            df[col] = 0

    if "split_failure_reason" not in df.columns:
        df["split_failure_reason"] = "NOT_RECORDED"
    if "split_apply_mode" not in df.columns:
        df["split_apply_mode"] = "NOT_RECORDED"

    summary = (
        df.groupby("variant", dropna=False)
        .agg(
            runs=("variant", "size"),
            split_attempted_rate=("split_attempted", "mean"),
            split_attempt_count_mean=("split_attempt_count", "mean"),
            split_applied_runtime_rate=("split_applied_runtime", "mean"),
            split_marker_survived_rate=("split_marker_survived", "mean"),
            split_applied_real_rate=("split_apply_mode", lambda s: float((s == "REAL_CROSSING_FOUND").mean())),
            split_applied_system_constructed_rate=(
                "split_apply_mode",
                lambda s: float((s == "SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE").mean()),
            ),
            split_applied_synthetic_missing_travel_rate=(
                "split_apply_mode",
                lambda s: float((s == "SYNTHETIC_INSERTED_NO_TRAVEL_FOUND").mean()),
            ),
        )
        .reset_index()
        .sort_values("variant")
    )

    reason_counts = (
        df.groupby(["variant", "split_failure_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["variant", "count"], ascending=[True, False])
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Split Debug Summary\n\n")
        f.write("## Aggregate\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Failure Reasons\n\n")
        if reason_counts.empty:
            f.write("- none\n")
        else:
            f.write(reason_counts.to_markdown(index=False))

    print(f"[split_debug] csv={out_csv} md={out_md}")


if __name__ == "__main__":
    main()
