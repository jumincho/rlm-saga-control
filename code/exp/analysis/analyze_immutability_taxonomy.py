"""Analyze immutable-prefix failures and generate taxonomy report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Immutability failure taxonomy")
    p.add_argument("--baseline", required=True)
    p.add_argument("--extension", required=True)
    p.add_argument("--variant", default="V3_PREFIX_SPLIT")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--max-cases-per-type", type=int, default=20)
    return p.parse_args()


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


def _to_minutes(hhmm: str) -> int | None:
    try:
        parts = str(hhmm).strip().split(":")
        if len(parts) != 2:
            return None
        h = int(parts[0])
        m = int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return h * 60 + m
    except Exception:
        return None


def _event_tuple(event: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(event.get("start", "")),
        str(event.get("end", "")),
        str(event.get("who", "")),
        str(event.get("what", "")),
        str(event.get("location", "")),
        str(event.get("notes", "")),
    )


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _canonical_core(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        out.append(
            (
                _to_minutes(str(event.get("start", ""))),
                _to_minutes(str(event.get("end", ""))),
                _norm(event.get("who", "")),
                _norm(event.get("what", "")),
                _norm(event.get("location", "")),
            )
        )
    out.sort(
        key=lambda x: (
            x[0] if x[0] is not None else 10**9,
            x[1] if x[1] is not None else 10**9,
            x[2],
            x[3],
            x[4],
        )
    )
    return out


def _split_pre_subset(events: list[dict[str, Any]]) -> list[tuple[str, str, str, str, str, str]]:
    out: list[tuple[str, str, str, str, str, str]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
        if "boundary_split_pre" in blob:
            out.append(_event_tuple(event))
    out.sort()
    return out


def _classify(locked_prefix: list[dict[str, Any]], final_prefix: list[dict[str, Any]]) -> str:
    lock_split = _split_pre_subset(locked_prefix)
    final_split = _split_pre_subset(final_prefix)
    if lock_split != final_split:
        return "SPLIT_PRE_REBUILT"

    if len(locked_prefix) != len(final_prefix):
        return "COUNT_OR_ORDER_CHANGED"

    lock_core = _canonical_core(locked_prefix)
    final_core = _canonical_core(final_prefix)
    if lock_core == final_core:
        lock_full = [_event_tuple(ev) for ev in locked_prefix]
        final_full = [_event_tuple(ev) for ev in final_prefix]
        if lock_full != final_full:
            return "COUNT_OR_ORDER_CHANGED"
        return "UNKNOWN"

    lock_no_time = [(x[2], x[3], x[4]) for x in lock_core]
    final_no_time = [(x[2], x[3], x[4]) for x in final_core]
    if lock_no_time == final_no_time:
        return "TIME_ONLY_CHANGED"
    return "LOCATION_OR_STATE_CHANGED"


def _safe_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [ev for ev in value if isinstance(ev, dict)]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("sample_id", "")), int(row.get("seed", 0)), str(row.get("stage", "")))


def main() -> None:
    args = parse_args()
    baseline = read_jsonl(args.baseline)
    extension = [r for r in read_jsonl(args.extension) if str(r.get("variant", "")) == args.variant]

    base_map = {_key(r): r for r in baseline}
    ext_map = {_key(r): r for r in extension}

    failed_rows: list[dict[str, Any]] = []
    for row in extension:
        if int(row.get("crossing_split_applicable", 0)) != 1:
            continue
        if int(row.get("immutable_prefix_after_split_ok", 0)) == 1:
            continue
        failed_rows.append(row)

    counter: Counter[str] = Counter()
    grouped_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in failed_rows:
        locked = _safe_events(row.get("locked_prefix_snapshot", []))
        final = _safe_events(row.get("final_prefix_extracted", []))
        diff = str(row.get("immutable_diff_type", "")).strip()
        if not diff:
            diff = _classify(locked, final)
        if not diff:
            diff = "UNKNOWN"
        counter[diff] += 1

        key = _key(row)
        base = base_map.get(key, {})
        grouped_cases[diff].append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "seed": int(row.get("seed", 0)),
                "stage": str(row.get("stage", "")),
                "violation_count_v3": int(row.get("violation_count", 0)),
                "violation_count_v0": int(base.get("violation_count", 0)),
                "violations_v3": list(row.get("violations", [])),
                "immutability_guard_failure_stage": str(row.get("immutability_guard_failure_stage", "")),
                "split_apply_mode": str(row.get("split_apply_mode", "")),
                "locked_prefix_snapshot": locked,
                "final_prefix_extracted": final,
            }
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Immutability Failure Taxonomy\n\n")
        f.write(f"- baseline rows: {len(baseline)}\n")
        f.write(f"- extension rows ({args.variant}): {len(extension)}\n")
        f.write(f"- failed rows (immutable_prefix_after_split_ok=0): {len(failed_rows)}\n\n")

        if not failed_rows:
            f.write("No immutable-prefix failures found.\n")
        else:
            f.write("## Type Counts\n\n")
            f.write("| diff_type | count |\n|---|---:|\n")
            for k, v in counter.most_common():
                f.write(f"| {k} | {v} |\n")

            for diff_type, cases in sorted(grouped_cases.items(), key=lambda x: (-len(x[1]), x[0])):
                f.write(f"\n\n## {diff_type}\n\n")
                for case in cases[: max(1, args.max_cases_per_type)]:
                    f.write(
                        f"### {case['sample_id']} | seed={case['seed']} | stage={case['stage']}\n\n"
                    )
                    f.write(f"- violation_count_v0: {case['violation_count_v0']}\n")
                    f.write(f"- violation_count_v3: {case['violation_count_v3']}\n")
                    f.write(
                        f"- immutability_guard_failure_stage: {case['immutability_guard_failure_stage']}\n"
                    )
                    f.write(f"- split_apply_mode: {case['split_apply_mode']}\n")
                    f.write("- violations_v3:\n")
                    if case["violations_v3"]:
                        for v in case["violations_v3"]:
                            f.write(f"  - {v}\n")
                    else:
                        f.write("  - (none)\n")
                    f.write("- locked_prefix_snapshot:\n")
                    f.write(
                        f"```json\n{json.dumps(case['locked_prefix_snapshot'], ensure_ascii=False, indent=2)}\n```\n"
                    )
                    f.write("- final_prefix_extracted:\n")
                    f.write(
                        f"```json\n{json.dumps(case['final_prefix_extracted'], ensure_ascii=False, indent=2)}\n```\n"
                    )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("diff_type,count\n")
            for k, v in counter.most_common():
                f.write(f"{k},{v}\n")

    print(
        f"[immutability_taxonomy] failed={len(failed_rows)} out_md={out_md}"
        + (f" out_csv={args.out_csv}" if args.out_csv else "")
    )


if __name__ == "__main__":
    main()

