"""Summarize baseline vs extension JSONL results into CSV and Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--report-out", required=True)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def extract_total_tokens(usage: dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    summaries = usage.get("model_usage_summaries", {})
    total = 0
    if isinstance(summaries, dict):
        for model_summary in summaries.values():
            if isinstance(model_summary, dict):
                total += int(model_summary.get("total_input_tokens", 0))
                total += int(model_summary.get("total_output_tokens", 0))
    return total


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if "V0" in df["variant"].values:
        base = df[df["variant"] == "V0"]
        budget_cap = float(base["total_tokens"].median())
        time_cap = float(base["wall_time_sec"].median())
    else:
        budget_cap = float(df["total_tokens"].median())
        time_cap = float(df["wall_time_sec"].median())

    grouped = []

    def _mean(sub_df: pd.DataFrame, column: str) -> float:
        if column not in sub_df.columns:
            return 0.0
        return float(sub_df[column].mean())

    def _mean_given_applicable(sub_df: pd.DataFrame, value_col: str, applicable_col: str) -> float:
        if value_col not in sub_df.columns or applicable_col not in sub_df.columns:
            return 0.0
        app = sub_df[sub_df[applicable_col] == 1]
        if app.empty:
            return 0.0
        return float(app[value_col].mean())

    def _rate_equals(sub_df: pd.DataFrame, column: str, value: str) -> float:
        if column not in sub_df.columns or len(sub_df) == 0:
            return 0.0
        return float((sub_df[column].astype(str) == value).mean())

    def _ratio_mean(sub_df: pd.DataFrame, num_col: str, den_col: str) -> float:
        if num_col not in sub_df.columns or den_col not in sub_df.columns or len(sub_df) == 0:
            return 0.0
        den = sub_df[den_col].replace(0, pd.NA)
        ratio = sub_df[num_col] / den
        return float(ratio.fillna(0.0).mean())

    for variant, sub in df.groupby("variant"):
        in_budget = sub[sub["total_tokens"] <= budget_cap]
        in_time = sub[sub["wall_time_sec"] <= time_cap]
        disruption_sub = (
            sub[sub.get("disruption_required", 0) == 1]
            if "disruption_required" in sub.columns
            else sub.iloc[0:0]
        )
        row = {
            "variant": variant,
            "runs": len(sub),
            "success_rate": sub["success"].mean(),
            "success_at_equal_budget": in_budget["success"].mean() if len(in_budget) else 0.0,
            "success_at_equal_time": in_time["success"].mean() if len(in_time) else 0.0,
            "avg_violation_count": sub["violation_count"].mean(),
            "recovery_success_rate": sub[sub["failure_injection"] == 1]["success"].mean()
            if (sub["failure_injection"] == 1).any()
            else 0.0,
            "invalid_commit_rate": sub["invalid_commit_rate"].mean(),
            "state_corruption_rate": sub["state_corruption"].mean(),
            "avg_rollbacks": sub["tx_rollbacks"].mean(),
            "avg_retries": sub["tx_retries"].mean(),
            "best_plan_selected_rate": _mean(sub, "best_plan_selected"),
            "best_plan_violation_improvement_over_last_mean": _mean(
                sub, "best_plan_violation_improvement_over_last"
            ),
            "best_plan_score_best_mean": _mean(sub, "best_plan_score_best"),
            "best_plan_score_last_mean": _mean(sub, "best_plan_score_last"),
            "best_plan_score_improvement_over_last_mean": _mean(
                sub, "best_plan_score_improvement_over_last"
            ),
            "suffix_only_output_ok_rate": _mean(sub, "suffix_only_output_ok"),
            "prefix_edit_attempt_detected_rate": _mean(
                sub, "prefix_edit_attempt_detected"
            ),
            "prefix_edit_ignored_count_mean": _mean(
                sub, "prefix_edit_ignored_count"
            ),
            "immutability_guard_triggered_rate": _mean(
                sub, "immutability_guard_triggered"
            ),
            "immutability_guard_photo_edit_rate": _rate_equals(
                sub, "immutability_guard_failure_stage", "PHOTO_EDIT"
            ),
            "immutability_guard_bestof_rate": _rate_equals(
                sub, "immutability_guard_failure_stage", "BESTOF_SELECT"
            ),
            "guard_view_hash_match_rate": _mean(sub, "guard_view_hash_match"),
            "scorer_view_hash_match_rate": _mean(sub, "scorer_view_hash_match"),
            "scorer_view_locked_monotonic_rate": _mean(sub, "scorer_view_locked_monotonic"),
            "scorer_view_final_monotonic_rate": _mean(sub, "scorer_view_final_monotonic"),
            "num_candidates_generated_mean": _mean(sub, "num_candidates_generated"),
            "num_candidates_scored_mean": _mean(sub, "num_candidates_scored"),
            "num_candidates_discarded_by_guard_mean": _mean(
                sub, "num_candidates_discarded_by_guard"
            ),
            "guard_discard_ratio_mean": _ratio_mean(
                sub, "num_candidates_discarded_by_guard", "num_candidates_generated"
            ),
            "state_fail_time_early_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "TIME_EARLY"
            ),
            "state_fail_time_non_monotonic_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "TIME_NON_MONOTONIC"
            ),
            "state_fail_location_mismatch_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "LOCATION_MISMATCH"
            ),
            "state_fail_missing_suffix_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "MISSING_SUFFIX"
            ),
            "state_fail_missing_boundary_event_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "MISSING_BOUNDARY_EVENT"
            ),
            "state_fail_in_transit_mismatch_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "IN_TRANSIT_MISMATCH"
            ),
            "state_fail_time_parse_rate": _rate_equals(
                sub, "state_consistency_failure_reason", "TIME_PARSE_FAIL"
            ),
            "split_attempted_rate": _mean(sub, "split_attempted"),
            "split_attempt_count_mean": _mean(sub, "split_attempt_count"),
            "split_applied_runtime_rate": _mean(sub, "split_applied_runtime"),
            "split_marker_survived_rate": _mean(sub, "split_marker_survived"),
            "split_candidate_found_rate": _mean(sub, "split_candidate_found"),
            "split_candidate_count_mean": _mean(sub, "split_candidate_count"),
            "timeline_norm_applied_rate": _mean(sub, "timeline_norm_applied"),
            "timeline_norm_total_shift_minutes_mean": _mean(
                sub, "timeline_norm_total_shift_minutes"
            ),
            "timeline_norm_overlap_fixes_count_mean": _mean(
                sub, "timeline_norm_overlap_fixes_count"
            ),
            "timeline_norm_location_fix_applied_rate": _mean(
                sub, "timeline_norm_location_fix_applied"
            ),
            "boundary_canonicalization_applied_rate": _mean(
                sub, "boundary_canonicalization_applied"
            ),
            "post_boundary_monotonic_fix_applied_rate": _mean(
                sub, "post_boundary_monotonic_fix_applied"
            ),
            "post_boundary_monotonic_fix_count_mean": _mean(
                sub, "post_boundary_monotonic_fix_count"
            ),
            "post_boundary_total_shift_minutes_mean": _mean(
                sub, "post_boundary_total_shift_minutes"
            ),
            "missing_boundary_event_detected_rate": _mean(
                sub, "missing_boundary_event_detected"
            ),
            "missing_boundary_event_autofixed_rate": _mean(
                sub, "missing_boundary_event_autofixed"
            ),
            "missing_boundary_event_autofix_minutes_pre_mean": _mean(
                sub, "missing_boundary_event_autofix_minutes_pre"
            ),
            "missing_boundary_event_autofix_minutes_post_mean": _mean(
                sub, "missing_boundary_event_autofix_minutes_post"
            ),
            "immutable_scope_sanitize_applied_rate": _mean(
                sub, "immutable_scope_sanitize_applied"
            ),
            "immutable_scope_terms_removed_count_mean": _mean(
                sub, "immutable_scope_terms_removed_count"
            ),
            "immutable_scope_contains_disruption_terms_before_rate": _mean(
                sub, "immutable_scope_contains_disruption_terms_before"
            ),
            "immutable_scope_contains_disruption_terms_after_rate": _mean(
                sub, "immutable_scope_contains_disruption_terms_after"
            ),
            "boundary_pre_end_minus_alert_min_mean": _mean(
                sub, "boundary_pre_end_minus_alert_min"
            ),
            "boundary_post_start_minus_alert_min_mean": _mean(
                sub, "boundary_post_start_minus_alert_min"
            ),
            "early_exit_taken_rate": _mean(sub, "early_exit_taken"),
            "llm_call_count_capped_rate": _mean(sub, "llm_call_count_capped"),
            "llm_call_reason_plan_mean": _mean(sub, "llm_call_reason_plan"),
            "llm_call_reason_format_mean": _mean(sub, "llm_call_reason_format"),
            "llm_call_reason_fill_events_mean": _mean(
                sub, "llm_call_reason_fill_events"
            ),
            "llm_call_reason_boundary_crossing_mean": _mean(
                sub, "llm_call_reason_boundary_crossing"
            ),
            "llm_call_reason_constraint_mean": _mean(
                sub, "llm_call_reason_constraint"
            ),
            "llm_call_reason_photo_time_mean": _mean(
                sub, "llm_call_reason_photo_time"
            ),
            "llm_call_reason_tailor_hours_mean": _mean(
                sub, "llm_call_reason_tailor_hours"
            ),
            "boundary_gate_passed_rate": _mean(sub, "boundary_gate_passed"),
            "boundary_gate_pass_after_fix_iter_mean": _mean(
                sub, "boundary_gate_pass_after_fix_iter"
            ),
            "boundary_gate_strict_success_last_rate": _mean(
                sub, "boundary_gate_strict_success_last"
            ),
            "boundary_gate_strict_violation_count_last_mean": _mean(
                sub, "boundary_gate_strict_violation_count_last"
            ),
            "boundary_gate_failed_reason_non_monotonic_mean": _mean(
                sub, "boundary_gate_failed_reason_non_monotonic"
            ),
            "boundary_gate_failed_reason_missing_boundary_mean": _mean(
                sub, "boundary_gate_failed_reason_missing_boundary"
            ),
            "boundary_gate_failed_reason_state_mismatch_mean": _mean(
                sub, "boundary_gate_failed_reason_state_mismatch"
            ),
            "boundary_gate_failed_reason_immutable_terms_mean": _mean(
                sub, "boundary_gate_failed_reason_immutable_terms"
            ),
            "boundary_gate_failed_reason_marker_lost_mean": _mean(
                sub, "boundary_gate_failed_reason_marker_lost"
            ),
            "boundary_gate_fix_split_or_comp_mean": _mean(
                sub, "boundary_gate_fix_split_or_comp"
            ),
            "boundary_gate_fix_boundary_autofix_mean": _mean(
                sub, "boundary_gate_fix_boundary_autofix"
            ),
            "boundary_gate_fix_state_shift_mean": _mean(
                sub, "boundary_gate_fix_state_shift"
            ),
            "boundary_gate_fix_canonicalize_mean": _mean(
                sub, "boundary_gate_fix_canonicalize"
            ),
            "boundary_gate_fix_monotonic_mean": _mean(
                sub, "boundary_gate_fix_monotonic"
            ),
            "boundary_gate_fix_immutable_sanitize_mean": _mean(
                sub, "boundary_gate_fix_immutable_sanitize"
            ),
            "split_retry_triggered_rate": _mean(sub, "split_retry_triggered"),
            "split_marker_lost_stage_nonempty_rate": float(
                (sub.get("split_marker_lost_stage", "").astype(str).str.len() > 0).mean()
            )
            if "split_marker_lost_stage" in sub.columns and len(sub) > 0
            else 0.0,
            "immutable_terms_detected_rate": _mean(sub, "immutable_terms_detected"),
            "v3_fallback_to_split_only_rate": _mean(sub, "v3_fallback_to_split_only"),
            "prefix_contains_disruption_terms_rate": _mean(
                sub, "prefix_contains_disruption_terms"
            ),
            "split_pre_contains_disruption_terms_rate": _mean(
                sub, "split_pre_contains_disruption_terms"
            ),
            "split_applied_real_rate": _rate_equals(
                sub, "split_apply_mode", "REAL_CROSSING_FOUND"
            ),
            "split_applied_system_constructed_rate": _rate_equals(
                sub, "split_apply_mode", "SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE"
            ),
            "split_applied_synthetic_missing_travel_rate": _rate_equals(
                sub, "split_apply_mode", "SYNTHETIC_INSERTED_NO_TRAVEL_FOUND"
            ),
            "split_applied_synthetic_rate": _rate_equals(
                sub, "split_apply_mode", "SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE"
            )
            + _rate_equals(
                sub, "split_apply_mode", "SYNTHETIC_INSERTED_NO_TRAVEL_FOUND"
            ),
            "split_failed_parse_rate": _rate_equals(sub, "split_apply_mode", "FAILED_PARSE"),
            "split_failed_lock_conflict_rate": _rate_equals(
                sub, "split_apply_mode", "FAILED_CONFLICT_WITH_LOCK"
            ),
            "valid_json_rate": _mean(sub, "valid_json"),
            "non_empty_events_rate": _mean(sub, "non_empty_events"),
            "disruption_applicable_rate": _mean(disruption_sub, "disruption_applicable"),
            "disruption_applied_given_applicable": _mean_given_applicable(
                disruption_sub, "disruption_applied", "disruption_applicable"
            ),
            "partial_compensation_applicable_rate": _mean(
                disruption_sub, "partial_compensation_applicable"
            ),
            "partial_compensation_correct_given_applicable": _mean_given_applicable(
                disruption_sub, "partial_compensation_ok", "partial_compensation_applicable"
            ),
            "crossing_split_applicable_rate": _mean(
                disruption_sub, "crossing_split_applicable"
            ),
            "crossing_split_applied_given_applicable": _mean_given_applicable(
                disruption_sub, "crossing_split_applied", "crossing_split_applicable"
            ),
            "immutable_check_applicable_rate": _mean(disruption_sub, "immutable_check_applicable"),
            "immutable_prefix_given_applicable": _mean_given_applicable(
                disruption_sub, "immutable_prefix_ok", "immutable_check_applicable"
            ),
            "immutable_prefix_after_split_given_applicable": _mean_given_applicable(
                disruption_sub, "immutable_prefix_after_split_ok", "crossing_split_applicable"
            ),
            "state_check_applicable_rate": _mean(disruption_sub, "state_check_applicable"),
            "state_consistent_given_applicable": _mean_given_applicable(
                disruption_sub, "state_at_alert_consistent", "state_check_applicable"
            ),
            # Backward-compatible aliases.
            "disruption_applied_rate": _mean_given_applicable(
                disruption_sub, "disruption_applied", "disruption_applicable"
            ),
            "partial_compensation_rate": _mean_given_applicable(
                disruption_sub, "partial_compensation_ok", "partial_compensation_applicable"
            ),
            "crossing_split_applied_rate": _mean_given_applicable(
                disruption_sub, "crossing_split_applied", "crossing_split_applicable"
            ),
            "immutable_prefix_rate": _mean_given_applicable(
                disruption_sub, "immutable_prefix_ok", "immutable_check_applicable"
            ),
            "immutable_prefix_after_split_rate": _mean_given_applicable(
                disruption_sub, "immutable_prefix_after_split_ok", "crossing_split_applicable"
            ),
            "state_at_alert_consistent_rate": _mean_given_applicable(
                disruption_sub, "state_at_alert_consistent", "state_check_applicable"
            ),
            "wall_time_p50": sub["wall_time_sec"].quantile(0.50),
            "wall_time_p90": sub["wall_time_sec"].quantile(0.90),
            "wall_time_p95": sub["wall_time_sec"].quantile(0.95),
            "wall_time_p99": sub["wall_time_sec"].quantile(0.99),
            "tokens_p50": sub["total_tokens"].quantile(0.50),
            "tokens_p90": sub["total_tokens"].quantile(0.90),
            "tokens_p95": sub["total_tokens"].quantile(0.95),
            "tokens_p99": sub["total_tokens"].quantile(0.99),
            "llm_call_count_mean": _mean(sub, "llm_call_count"),
            "llm_time_total_sec_mean": _mean(sub, "llm_time_total_sec"),
            "validator_time_total_sec_mean": _mean(sub, "validator_time_total_sec"),
            "postproc_time_total_sec_mean": _mean(sub, "postproc_time_total_sec"),
            "timeline_norm_time_total_sec_mean": _mean(
                sub, "timeline_norm_time_total_sec"
            ),
            "llm_time_total_sec_p50": sub["llm_time_total_sec"].quantile(0.50)
            if "llm_time_total_sec" in sub.columns
            else 0.0,
            "llm_time_total_sec_p90": sub["llm_time_total_sec"].quantile(0.90)
            if "llm_time_total_sec" in sub.columns
            else 0.0,
            "llm_time_total_sec_p95": sub["llm_time_total_sec"].quantile(0.95)
            if "llm_time_total_sec" in sub.columns
            else 0.0,
            "validator_time_total_sec_p50": sub["validator_time_total_sec"].quantile(0.50)
            if "validator_time_total_sec" in sub.columns
            else 0.0,
            "validator_time_total_sec_p90": sub["validator_time_total_sec"].quantile(0.90)
            if "validator_time_total_sec" in sub.columns
            else 0.0,
            "validator_time_total_sec_p95": sub["validator_time_total_sec"].quantile(0.95)
            if "validator_time_total_sec" in sub.columns
            else 0.0,
            "postproc_time_total_sec_p50": sub["postproc_time_total_sec"].quantile(0.50)
            if "postproc_time_total_sec" in sub.columns
            else 0.0,
            "postproc_time_total_sec_p90": sub["postproc_time_total_sec"].quantile(0.90)
            if "postproc_time_total_sec" in sub.columns
            else 0.0,
            "postproc_time_total_sec_p95": sub["postproc_time_total_sec"].quantile(0.95)
            if "postproc_time_total_sec" in sub.columns
            else 0.0,
            "violation_p50": sub["violation_count"].quantile(0.50),
            "violation_p90": sub["violation_count"].quantile(0.90),
            "violation_p95": sub["violation_count"].quantile(0.95),
        }
        grouped.append(row)

    out = pd.DataFrame(grouped).sort_values("variant")
    return out


def success_criteria_line(summary_df: pd.DataFrame) -> list[str]:
    lines = []
    if "V0" not in summary_df["variant"].values or "V2" not in summary_df["variant"].values:
        lines.append("- Success criteria check skipped: V0 or V2 missing.")
        return lines

    v0 = summary_df[summary_df["variant"] == "V0"].iloc[0]
    v2 = summary_df[summary_df["variant"] == "V2"].iloc[0]

    def reduction_ratio(baseline: float, candidate: float) -> float | None:
        # If baseline is zero, percent reduction is undefined.
        if baseline <= 0:
            return None
        return (baseline - candidate) / baseline

    reduce_invalid = reduction_ratio(
        float(v0["invalid_commit_rate"]), float(v2["invalid_commit_rate"])
    )
    reduce_corruption = reduction_ratio(
        float(v0["state_corruption_rate"]), float(v2["state_corruption_rate"])
    )
    success_drop = v0["success_rate"] - v2["success_rate"]
    p95_improved = (v2["wall_time_p95"] < v0["wall_time_p95"]) or (
        v2["tokens_p95"] < v0["tokens_p95"]
    )

    invalid_str = f"{reduce_invalid:.2%}" if reduce_invalid is not None else "N/A (baseline=0)"
    corruption_str = (
        f"{reduce_corruption:.2%}" if reduce_corruption is not None else "N/A (baseline=0)"
    )
    lines.append(f"- Criterion1 (>=25% reduction): invalid={invalid_str}, corruption={corruption_str}")
    lines.append(f"- Criterion2 (success drop <=5pp): drop={success_drop:.2%}")
    lines.append(f"- Criterion3 (P95 improved): {p95_improved}")
    return lines


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.baseline) + read_jsonl(args.extension)
    if not rows:
        raise RuntimeError("No result rows found")

    for row in rows:
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        row["failure_injection"] = int(meta.get("failure_injection", "none") != "none")
        row["total_tokens"] = extract_total_tokens(row.get("usage", {}))
        for key in [
            "valid_json",
            "events_count",
            "non_empty_events",
            "disruption_required",
            "disruption_applicable",
            "disruption_applied",
            "partial_compensation_applicable",
            "partial_compensation_ok",
            "crossing_split_applicable",
            "crossing_split_applied",
            "immutable_check_applicable",
            "immutable_prefix_ok",
            "immutable_prefix_after_split_ok",
            "state_check_applicable",
            "state_at_alert_consistent",
            "split_attempted",
            "split_attempt_count",
            "split_applied_runtime",
            "split_marker_survived",
            "split_candidate_found",
            "split_candidate_count",
            "best_plan_selected",
            "suffix_only_output_ok",
            "prefix_edit_attempt_detected",
            "immutability_guard_triggered",
            "guard_view_hash_match",
            "scorer_view_hash_match",
            "scorer_view_locked_monotonic",
            "scorer_view_final_monotonic",
            "num_candidates_generated",
            "num_candidates_scored",
            "num_candidates_discarded_by_guard",
            "timeline_norm_applied",
            "timeline_norm_total_shift_minutes",
            "timeline_norm_overlap_fixes_count",
            "timeline_norm_location_fix_applied",
            "boundary_canonicalization_applied",
            "post_boundary_monotonic_fix_applied",
            "post_boundary_monotonic_fix_count",
            "post_boundary_total_shift_minutes",
            "missing_boundary_event_detected",
            "missing_boundary_event_autofixed",
            "missing_boundary_event_autofix_minutes_pre",
            "missing_boundary_event_autofix_minutes_post",
            "immutable_scope_sanitize_applied",
            "immutable_scope_terms_removed_count",
            "immutable_scope_contains_disruption_terms_before",
            "immutable_scope_contains_disruption_terms_after",
            "boundary_pre_end_minus_alert_min",
            "boundary_post_start_minus_alert_min",
            "prefix_contains_disruption_terms",
            "split_pre_contains_disruption_terms",
            "llm_call_count",
            "llm_call_count_capped",
            "llm_call_reason_plan",
            "llm_call_reason_format",
            "llm_call_reason_fill_events",
            "llm_call_reason_boundary_crossing",
            "llm_call_reason_constraint",
            "llm_call_reason_photo_time",
            "llm_call_reason_tailor_hours",
            "early_exit_taken",
            "boundary_gate_passed",
            "boundary_gate_pass_after_fix_iter",
            "boundary_gate_strict_success_last",
            "boundary_gate_strict_violation_count_last",
            "boundary_gate_failed_reason_non_monotonic",
            "boundary_gate_failed_reason_missing_boundary",
            "boundary_gate_failed_reason_state_mismatch",
            "boundary_gate_failed_reason_immutable_terms",
            "boundary_gate_failed_reason_marker_lost",
            "boundary_gate_fix_split_or_comp",
            "boundary_gate_fix_boundary_autofix",
            "boundary_gate_fix_state_shift",
            "boundary_gate_fix_canonicalize",
            "boundary_gate_fix_monotonic",
            "boundary_gate_fix_immutable_sanitize",
            "v3_fallback_to_split_only",
            "split_retry_triggered",
            "immutable_terms_detected",
        ]:
            row[key] = int(row.get(key, 0))
        row["best_plan_violation_improvement_over_last"] = float(
            row.get("best_plan_violation_improvement_over_last", 0.0)
        )
        row["best_plan_score_best"] = float(row.get("best_plan_score_best", 0.0))
        row["best_plan_score_last"] = float(row.get("best_plan_score_last", 0.0))
        row["best_plan_score_improvement_over_last"] = float(
            row.get("best_plan_score_improvement_over_last", 0.0)
        )
        row["prefix_edit_ignored_count"] = float(row.get("prefix_edit_ignored_count", 0.0))
        row["llm_time_total_sec"] = float(row.get("llm_time_total_sec", 0.0))
        row["validator_time_total_sec"] = float(row.get("validator_time_total_sec", 0.0))
        row["postproc_time_total_sec"] = float(row.get("postproc_time_total_sec", 0.0))
        row["timeline_norm_time_total_sec"] = float(
            row.get("timeline_norm_time_total_sec", 0.0)
        )
        row["split_apply_mode"] = str(row.get("split_apply_mode", ""))
        row["immutability_guard_failure_stage"] = str(
            row.get("immutability_guard_failure_stage", "")
        )
        row["state_consistency_failure_reason"] = str(
            row.get("state_consistency_failure_reason", "")
        )

    df = pd.DataFrame(rows)

    summary_df = summarize(df)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(metrics_out, index=False)

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write("# RLM vs RLM+Saga v1 Report\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Cost-sensitive Comparison\n\n")
        f.write(
            "- `success_at_equal_budget`: success rate under V0 median token budget cap.\n"
            "- `success_at_equal_time`: success rate under V0 median wall-time cap.\n"
        )
        f.write("\n## Disruption / Format Diagnostics\n\n")
        f.write(
            "- `valid_json_rate`: valid JSON parse ratio.\n"
            "- `non_empty_events_rate`: non-empty schedule ratio.\n"
            "- `best_plan_selected_rate`: fraction of runs where best-of commit replaced the last candidate.\n"
            "- `best_plan_violation_improvement_over_last_mean`: mean violation improvement from best-of commit.\n"
            "- `best_plan_score_improvement_over_last_mean`: weighted score improvement from best-of commit.\n"
            "- `suffix_only_output_ok_rate`: rate where boundary output passed suffix-only structural gate.\n"
            "- `prefix_edit_attempt_detected_rate`: rate where model attempted prefix edits (then ignored).\n"
            "- `prefix_edit_ignored_count_mean`: average ignored prefix-edit event count.\n"
            "- `immutability_guard_triggered_rate`: rate where prefix hash guard rejected a candidate.\n"
            "- `disruption_applicable_rate`: samples where disruption checks are applicable.\n"
            "- `disruption_applied_given_applicable`: disruption handling rate conditioned on applicability.\n"
            "- `partial_compensation_correct_given_applicable`: boundary compensation correctness when applicable.\n"
            "- `crossing_split_applied_given_applicable`: split-based boundary handling rate when applicable.\n"
            "- `split_applied_runtime_rate`: split application rate recorded at runtime repair stage.\n"
            "- `split_marker_survived_rate`: rate where split markers survived in final candidate.\n"
            "- `split_applied_real_rate`: runtime split mode = REAL_CROSSING_FOUND.\n"
            "- `split_applied_system_constructed_rate`: runtime split mode = SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE.\n"
            "- `split_applied_synthetic_missing_travel_rate`: runtime split mode = SYNTHETIC_INSERTED_NO_TRAVEL_FOUND.\n"
            "- `split_applied_synthetic_rate`: aggregate synthetic split rate (system-constructed + missing-travel).\n"
            "- `immutable_prefix_given_applicable`: immutable-prefix consistency when prefix check is applicable.\n"
            "- `immutable_prefix_after_split_given_applicable`: immutable-prefix consistency after split application.\n"
            "- `state_consistent_given_applicable`: boundary-state consistency when state check is applicable.\n"
        )
        f.write("\n\n## Success Criteria\n\n")
        for line in success_criteria_line(summary_df):
            f.write(line + "\n")
        f.write("\n## Security Note\n\n")
        f.write("- Rotate previously exposed tokens/keys after experiment completion.\n")

    print(f"[summarize] metrics={metrics_out} report={report_out}")


if __name__ == "__main__":
    main()
