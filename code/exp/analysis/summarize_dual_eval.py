"""Re-score completed runs with runtime/strict/relaxed evaluators and summarize all modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from exp.analysis.summarize import extract_total_tokens, read_jsonl, summarize
from exp.bench.evaluator_offline import evaluate_prediction, get_evaluator_version
from exp.bench.types import BenchmarkSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-evaluator summarization (runtime/strict/relaxed).")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=["runtime", "strict", "relaxed"])
    return parser.parse_args()


def _row_to_sample(row: dict[str, Any]) -> BenchmarkSample:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return BenchmarkSample(
        sample_id=str(row.get("sample_id", "")),
        track=str(row.get("track", "")),
        source=str(row.get("source", "")),
        task_type=str(row.get("task_type", "planning")),
        prompt="",
        answer=row.get("ground_truth"),
        metadata=metadata,
    )


def _paired_win_loss(df: pd.DataFrame, left: str = "V0", right: str = "V3") -> tuple[int, int, int]:
    if df.empty:
        return (0, 0, 0)
    sub = df[df["variant"].isin([left, right])]
    if sub.empty:
        return (0, 0, 0)
    pair_keys = ["sample_id", "seed"]
    if "stage" in sub.columns:
        pair_keys.append("stage")
    pivot = sub.pivot_table(index=pair_keys, columns="variant", values="success", aggfunc="first")
    if left not in pivot.columns or right not in pivot.columns:
        return (0, 0, 0)
    paired = pivot.dropna()
    if paired.empty:
        return (0, 0, 0)
    right_wins = int((paired[right] > paired[left]).sum())
    left_wins = int((paired[right] < paired[left]).sum())
    ties = int((paired[right] == paired[left]).sum())
    return right_wins, left_wins, ties


def _integrity_checks(df: pd.DataFrame, left: str = "V0", right: str = "V3") -> list[str]:
    issues: list[str] = []
    if df.empty:
        return ["empty dataframe"]
    pair_keys = ["sample_id", "seed"]
    if "stage" in df.columns:
        pair_keys.append("stage")

    dup = df.duplicated(subset=pair_keys + ["variant"]).sum()
    if dup:
        issues.append(f"duplicate records by key+variant: {int(dup)}")

    if "variant" in df.columns:
        counts = df["variant"].value_counts().to_dict()
        lv = counts.get(left, 0)
        rv = counts.get(right, 0)
        if lv != rv:
            issues.append(f"variant row count mismatch {left}={lv} {right}={rv}")

    sub = df[df["variant"].isin([left, right])]
    pivot = sub.pivot_table(index=pair_keys, columns="variant", values="success", aggfunc="first")
    paired = pivot.dropna()
    if left not in pivot.columns or right not in pivot.columns:
        issues.append("missing one of paired variants in pivot")
    else:
        if len(paired) != len(pivot):
            issues.append(f"incomplete pairs {len(paired)}/{len(pivot)}")
    return issues


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_jsonl(args.baseline)
    ext_rows = read_jsonl(args.extension)
    rows = base_rows + ext_rows
    if not rows:
        raise RuntimeError("No rows found")

    for mode in args.modes:
        rescored: list[dict[str, Any]] = []
        version = get_evaluator_version(mode)
        for row in rows:
            prediction = str(row.get("prediction", "") or "")
            sample = _row_to_sample(row)
            scored = evaluate_prediction(sample, prediction, mode=mode)

            out_row = dict(row)
            out_row["success"] = int(scored.get("success", 0))
            out_row["score"] = float(scored.get("score", 0.0))
            out_row["violation_count"] = int(scored.get("violation_count", 0))
            out_row["violations"] = list(scored.get("violations", []))
            diag = scored.get("diagnostics", {}) if isinstance(scored.get("diagnostics"), dict) else {}
            out_row["valid_json"] = int(diag.get("valid_json", 0))
            out_row["events_count"] = int(diag.get("events_count", 0))
            out_row["non_empty_events"] = int(diag.get("non_empty_events", 0))
            out_row["disruption_required"] = int(diag.get("disruption_required", 0))
            out_row["disruption_applicable"] = int(diag.get("disruption_applicable", 0))
            out_row["disruption_applied"] = int(diag.get("disruption_applied", 0))
            out_row["partial_compensation_applicable"] = int(
                diag.get("partial_compensation_applicable", 0)
            )
            out_row["partial_compensation_ok"] = int(diag.get("partial_compensation_ok", 0))
            out_row["crossing_split_applicable"] = int(diag.get("crossing_split_applicable", 0))
            out_row["crossing_split_applied"] = int(diag.get("crossing_split_applied", 0))
            out_row["immutable_check_applicable"] = int(diag.get("immutable_check_applicable", 0))
            out_row["immutable_prefix_ok"] = int(diag.get("immutable_prefix_ok", 0))
            out_row["immutable_prefix_after_split_ok"] = int(
                diag.get("immutable_prefix_after_split_ok", 0)
            )
            out_row["state_check_applicable"] = int(diag.get("state_check_applicable", 0))
            out_row["state_at_alert_consistent"] = int(diag.get("state_at_alert_consistent", 0))
            out_row["offline_eval_mode"] = mode
            out_row["offline_eval_version"] = version
            rescored.append(out_row)

        for row in rescored:
            meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            row["failure_injection"] = int(meta.get("failure_injection", "none") != "none")
            row["total_tokens"] = extract_total_tokens(row.get("usage", {}))

        df = pd.DataFrame(rescored)
        summary_df = summarize(df)
        left_variant = "V0"
        right_variants = []
        if "variant" in df.columns:
            right_variants = sorted(
                [str(v) for v in df["variant"].dropna().unique().tolist() if str(v) != left_variant]
            )

        raw_out = out_dir / f"rescored_{mode}.jsonl"
        metrics_out = out_dir / f"metrics_{mode}.csv"
        report_out = out_dir / f"report_{mode}.md"
        _write_jsonl(raw_out, rescored)
        summary_df.to_csv(metrics_out, index=False)

        with open(report_out, "w", encoding="utf-8") as f:
            f.write(f"# Dual Evaluation Report ({mode})\n\n")
            f.write(f"- evaluator_version: `{version}`\n")
            f.write(f"- rows: {len(rescored)}\n\n")
            f.write("## Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
            if right_variants:
                for right_variant in right_variants:
                    right_wins, left_wins, ties = _paired_win_loss(df, left_variant, right_variant)
                    integrity = _integrity_checks(df, left_variant, right_variant)
                    f.write(f"\n\n## Paired {right_variant} vs {left_variant} (Success)\n\n")
                    f.write(f"- {right_variant} wins: {right_wins}\n")
                    f.write(f"- {left_variant} wins: {left_wins}\n")
                    f.write(f"- ties: {ties}\n")
                    f.write("\n### Integrity Checks\n\n")
                    if integrity:
                        for item in integrity:
                            f.write(f"- {item}\n")
                    else:
                        f.write("- passed\n")
            else:
                f.write("\n\n## Paired Comparison\n\n- skipped: no extension variants found\n")

        print(f"[dual_eval:{mode}] raw={raw_out} metrics={metrics_out} report={report_out}")


if __name__ == "__main__":
    main()
