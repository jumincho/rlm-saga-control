"""Analyze paired loss-cases where baseline wins over extension."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired loss-cases for a target variant.")
    parser.add_argument("--input-jsonl", required=True, help="rescored_*.jsonl from summarize_dual_eval")
    parser.add_argument("--left", default="V0", help="baseline variant")
    parser.add_argument("--right", default="V3_PREFIX", help="target extension variant")
    parser.add_argument("--stage", default="stage_feedback_v6_recovery")
    parser.add_argument("--problem-prefix", default="P9")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _safe_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.input_jsonl)
    df = _to_df(rows)
    if df.empty:
        raise RuntimeError("No rows found")

    key_cols = ["sample_id", "seed"]
    if "stage" in df.columns:
        key_cols.append("stage")

    sub = df[df["variant"].isin([args.left, args.right])].copy()
    if "stage" in sub.columns:
        sub = sub[sub["stage"] == args.stage]
    sub = sub[sub["sample_id"].astype(str).str.startswith(f"{args.problem_prefix}:")]
    if sub.empty:
        raise RuntimeError("No rows after filtering stage/problem/variants")

    left_df = sub[sub["variant"] == args.left].copy()
    right_df = sub[sub["variant"] == args.right].copy()

    left_cols = key_cols + ["success", "violation_count", "violations"]
    right_cols = key_cols + ["success", "violation_count", "violations", "repair_modes", "tx_rollbacks", "tx_retries"]
    for col in left_cols:
        if col not in left_df.columns:
            left_df[col] = None
    for col in right_cols:
        if col not in right_df.columns:
            right_df[col] = None

    merged = left_df[left_cols].merge(
        right_df[right_cols],
        on=key_cols,
        suffixes=("_left", "_right"),
        how="inner",
    )

    losses = merged[(merged["success_left"] == 1) & (merged["success_right"] == 0)].copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if losses.empty:
        losses.to_csv(out_csv, index=False)
        Path(args.out_md).write_text(
            f"# Loss-case Analysis ({args.right} vs {args.left})\n\n"
            f"- stage: `{args.stage}`\n"
            f"- problem: `{args.problem_prefix}`\n"
            "- no loss-cases found\n",
            encoding="utf-8",
        )
        return

    losses.to_csv(out_csv, index=False)

    viol_counter: Counter[str] = Counter()
    repair_counter: Counter[str] = Counter()
    rollback_vals: list[int] = []
    retry_vals: list[int] = []
    for _, row in losses.iterrows():
        for v in _safe_list(row.get("violations_right")):
            viol_counter[str(v)] += 1
        for mode in _safe_list(row.get("repair_modes")):
            repair_counter[str(mode)] += 1
        try:
            rollback_vals.append(int(row.get("tx_rollbacks", 0)))
        except Exception:
            pass
        try:
            retry_vals.append(int(row.get("tx_retries", 0)))
        except Exception:
            pass

    top_viol = viol_counter.most_common(10)
    top_repairs = repair_counter.most_common(10)
    avg_rb = sum(rollback_vals) / len(rollback_vals) if rollback_vals else 0.0
    avg_rt = sum(retry_vals) / len(retry_vals) if retry_vals else 0.0

    md = []
    md.append(f"# Loss-case Analysis ({args.right} vs {args.left})")
    md.append("")
    md.append(f"- stage: `{args.stage}`")
    md.append(f"- problem: `{args.problem_prefix}`")
    md.append(f"- paired rows: `{len(merged)}`")
    md.append(f"- loss cases (`{args.left}=1`, `{args.right}=0`): `{len(losses)}`")
    md.append(f"- avg right rollbacks (loss-only): `{avg_rb:.3f}`")
    md.append(f"- avg right retries (loss-only): `{avg_rt:.3f}`")
    md.append("")
    md.append("## Top Violations (Right)")
    md.append("")
    if top_viol:
        for k, c in top_viol:
            md.append(f"- {k}: {c}")
    else:
        md.append("- none")
    md.append("")
    md.append("## Top Repair Modes (Right)")
    md.append("")
    if top_repairs:
        for k, c in top_repairs:
            md.append(f"- {k}: {c}")
    else:
        md.append("- none")
    md.append("")
    md.append("## Loss Cases")
    md.append("")
    cols = key_cols + ["violation_count_left", "violation_count_right", "violations_right", "repair_modes", "tx_rollbacks", "tx_retries"]
    table_df = losses[cols].copy()
    md.append(table_df.to_markdown(index=False))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[loss_case] rows={len(losses)} md={out_md} csv={out_csv}")


if __name__ == "__main__":
    main()

