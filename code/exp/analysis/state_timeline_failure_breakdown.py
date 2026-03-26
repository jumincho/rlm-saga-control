from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_VIOLATION = "State timeline is inconsistent at disruption boundary"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Break down strict state-timeline inconsistency failures."
    )
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--variant", default="V3_PREFIX_SPLIT")
    p.add_argument("--out-md", required=True)
    p.add_argument("--max-cases", type=int, default=31)
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input_jsonl))
    target = []
    for row in rows:
        if row.get("variant") != args.variant:
            continue
        violations = row.get("violations", []) or []
        if TARGET_VIOLATION in violations:
            target.append(row)

    by_reason = Counter()
    by_stage = Counter()
    by_mode = Counter()
    by_split_mode = Counter()
    examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in target:
        reason = str(row.get("state_consistency_failure_reason", "") or "UNKNOWN")
        by_reason[reason] += 1
        by_stage[str(row.get("stage", ""))] += 1
        by_mode[str(row.get("runtime_validator_mode", ""))] += 1
        by_split_mode[str(row.get("split_apply_mode", ""))] += 1
        if len(examples_by_reason[reason]) < max(1, args.max_cases // 4):
            examples_by_reason[reason].append(row)

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# State Timeline Failure Breakdown\n\n")
        f.write(f"- input: `{args.input_jsonl}`\n")
        f.write(f"- variant: `{args.variant}`\n")
        f.write(f"- target violation: `{TARGET_VIOLATION}`\n")
        f.write(f"- matched rows: {len(target)}\n\n")

        f.write("## Reason Counts\n\n")
        for reason, count in by_reason.most_common():
            f.write(f"- {reason}: {count}\n")

        f.write("\n## Split Mode Counts\n\n")
        for mode, count in by_split_mode.most_common():
            f.write(f"- {mode}: {count}\n")

        f.write("\n## Stage Counts\n\n")
        for stage, count in by_stage.most_common():
            f.write(f"- {stage}: {count}\n")

        f.write("\n## Validator Mode Counts\n\n")
        for mode, count in by_mode.most_common():
            f.write(f"- {mode}: {count}\n")

        f.write("\n## Sample Cases\n\n")
        dumped = 0
        for reason, cases in examples_by_reason.items():
            f.write(f"### {reason}\n\n")
            for row in cases:
                if dumped >= args.max_cases:
                    break
                dumped += 1
                candidate = row.get("split_candidate_summary", {}) or {}
                f.write(f"- sample_id: `{row.get('sample_id','')}` seed={row.get('seed','')}\n")
                f.write(f"  - split_apply_mode: `{row.get('split_apply_mode','')}`\n")
                f.write(
                    f"  - next_actionable_time: `{row.get('next_actionable_time','')}` "
                    f"/ suffix_first_event_time_before: `{row.get('suffix_first_event_time_before','')}` "
                    f"/ suffix_first_event_time_after: `{row.get('suffix_first_event_time_after','')}`\n"
                )
                f.write(
                    f"  - timeline_norm_applied: `{row.get('timeline_norm_applied',0)}` "
                    f"/ shift_min: `{row.get('timeline_norm_total_shift_minutes',0)}` "
                    f"/ overlap_fix: `{row.get('timeline_norm_overlap_fixes_count',0)}`\n"
                )
                f.write(
                    f"  - boundary_pre_end_minus_alert_min: `{row.get('boundary_pre_end_minus_alert_min',0)}` "
                    f"/ boundary_post_start_minus_alert_min: `{row.get('boundary_post_start_minus_alert_min',0)}`\n"
                )
                f.write(
                    f"  - candidate(route={candidate.get('route','')}, "
                    f"seg_start={candidate.get('seg_start','')}, seg_end={candidate.get('seg_end','')})\n"
                )
                f.write(
                    f"  - violations: `{'; '.join((row.get('violations') or [])[:5])}`\n"
                )
            f.write("\n")
            if dumped >= args.max_cases:
                break

    print(f"[state_timeline_failure_breakdown] out={out} matched={len(target)}")


if __name__ == "__main__":
    main()

