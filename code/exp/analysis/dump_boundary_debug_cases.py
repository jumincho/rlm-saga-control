"""Dump boundary debug cases with violations, split debug fields, and event previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exp.bench.schema import parse_json_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boundary debug case dump")
    p.add_argument("--baseline", required=True)
    p.add_argument("--extension", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--variant", default="V3_PREFIX_SPLIT")
    p.add_argument("--max-events", type=int, default=20)
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


def _events_from_prediction(prediction: str) -> list[dict[str, Any]]:
    parsed, err = parse_json_response(prediction)
    if err or parsed is None:
        return []
    if isinstance(parsed, dict) and "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"events": parsed.get("schedule", [])}
    if not isinstance(parsed, dict):
        return []
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for ev in events:
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _fmt_list(items: list[Any]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {x}" for x in items)


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("sample_id", "")), int(row.get("seed", 0)), str(row.get("stage", "")))


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.baseline)
    ext_rows = [r for r in read_jsonl(args.extension) if str(r.get("variant", "")) == args.variant]

    base_map = {_key(r): r for r in base_rows}
    ext_map = {_key(r): r for r in ext_rows}
    keys = sorted(set(base_map.keys()) | set(ext_map.keys()))

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("# Boundary Debug Failure Breakdown\n\n")
        f.write(f"- baseline rows: {len(base_rows)}\n")
        f.write(f"- extension rows ({args.variant}): {len(ext_rows)}\n")
        f.write(f"- merged keys: {len(keys)}\n\n")

        for sample_id, seed, stage in keys:
            f.write(f"## {sample_id} | seed={seed} | stage={stage}\n\n")
            for label, row in [("V0", base_map.get((sample_id, seed, stage))), (args.variant, ext_map.get((sample_id, seed, stage)))]:
                if row is None:
                    f.write(f"### {label}\n- missing row\n\n")
                    continue
                pred = str(row.get("prediction", "") or "")
                events = _events_from_prediction(pred)
                split_summary = row.get("split_candidate_summary", {})
                if not isinstance(split_summary, dict):
                    split_summary = {}
                before_count = split_summary.get("events_before", "?")
                after_count = split_summary.get("events_after", "?")
                f.write(f"### {label}\n")
                f.write(f"- success: {row.get('success')}\n")
                f.write(f"- violation_count: {row.get('violation_count')}\n")
                f.write(f"- violations:\n{_fmt_list(list(row.get('violations', [])))}\n")
                f.write(f"- split_apply_mode: {row.get('split_apply_mode', '')}\n")
                f.write(f"- split_attempted: {row.get('split_attempted', 0)}\n")
                f.write(f"- split_attempt_count: {row.get('split_attempt_count', 0)}\n")
                f.write(f"- split_applied_runtime: {row.get('split_applied_runtime', 0)}\n")
                f.write(f"- split_marker_survived: {row.get('split_marker_survived', 0)}\n")
                f.write(f"- split_candidate_found/count: {row.get('split_candidate_found', 0)}/{row.get('split_candidate_count', 0)}\n")
                f.write(f"- split_failure_reason: {row.get('split_failure_reason', '')}\n")
                f.write(f"- split_failure_reasons:\n{_fmt_list(list(row.get('split_failure_reasons', [])))}\n")
                f.write("- split_candidate_summary:\n")
                f.write(f"```json\n{json.dumps(split_summary, ensure_ascii=False, indent=2)}\n```\n")
                f.write(f"- split events before/after: {before_count} -> {after_count}\n")
                f.write(f"- parsed events count: {len(events)}\n")
                preview = events[: max(0, int(args.max_events))]
                f.write("- events preview:\n")
                f.write(f"```json\n{json.dumps(preview, ensure_ascii=False, indent=2)}\n```\n\n")

    print(f"[boundary_debug_dump] out={out}")


if __name__ == "__main__":
    main()
