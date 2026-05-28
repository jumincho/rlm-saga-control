"""Confirm the runner's immutability guard and the offline scorer agree.

The runtime immutability guard hashes the locked prefix (events ending
at or before the alert minute) and rejects candidates whose prefix
hash drifts. The offline scorer canonicalizes the same prefix and also
hashes it; if the two views ever disagree, "immutability preserved"
becomes a meaningless claim. This script recomputes the scorer's view
hash for every row where `crossing_split_applicable=1 and
immutable_prefix_after_split_ok=0`, joins with the guard view hash
already stamped on the row, and labels each failure as:

- `BOTH_MATCH`               — neither view sees a problem (rare here).
- `GUARD_MATCH_SCORER_MISMATCH` — guard accepted, strict scorer rejects.
- `GUARD_MISMATCH_SCORER_MATCH` — guard rejected, strict scorer accepts.
- `BOTH_MISMATCH`            — both views see drift.
- `SCORER_NON_MONOTONIC`     — the final plan isn't even per-actor
  monotonic, so the scorer view question is moot.

The closure-report's "guard vs strict view alignment" check is the
match-rate this script reports on `BOTH_MATCH` + `BOTH_MISMATCH`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from exp.bench.schema import parse_json_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Immutability alignment report")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--variant", default="V3_PREFIX_SPLIT")
    p.add_argument("--stage", default="")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv", default="")
    p.add_argument("--max-cases", type=int, default=20)
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
        h, m = str(hhmm).strip().split(":")
        hh = int(h)
        mm = int(m)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh * 60 + mm
    except Exception:
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _actors(who: str) -> list[str]:
    out = [a.strip().lower() for a in str(who).replace("&", ",").replace("/", ",").split(",") if a.strip()]
    return out or ["_unknown"]


def _events_time_monotonic(events: list[dict[str, Any]]) -> bool:
    by_actor: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None or end_m < start_m:
            return False
        for actor in _actors(str(event.get("who", ""))):
            by_actor.setdefault(actor, []).append((start_m, end_m))
    for windows in by_actor.values():
        windows.sort(key=lambda x: (x[0], x[1]))
        prev_end = None
        for start_m, end_m in windows:
            if prev_end is not None and start_m < prev_end:
                return False
            prev_end = end_m
    return True


def _canonical(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for event in events:
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


def _scorer_view_hash(events: list[dict[str, Any]], alert_min: int | None) -> tuple[str, int, list[tuple[Any, ...]]]:
    if alert_min is None:
        return "", 0, []
    pre_events: list[dict[str, Any]] = []
    for event in events:
        end_m = _to_minutes(str(event.get("end", "")))
        if end_m is not None and end_m <= alert_min:
            pre_events.append(
                {
                    "start": str(event.get("start", "")),
                    "end": str(event.get("end", "")),
                    "who": str(event.get("who", "")),
                    "what": str(event.get("what", "")),
                    "location": str(event.get("location", "")),
                    "notes": str(event.get("notes", "")),
                }
            )
    canon = _canonical(pre_events)
    monotonic = int(_events_time_monotonic(pre_events))
    payload = {"canonical": canon, "time_monotonic": monotonic}
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), monotonic, canon


def _extract_events(prediction: str) -> list[dict[str, Any]]:
    parsed, err = parse_json_response(prediction)
    if err or not isinstance(parsed, dict):
        return []
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"events": parsed.get("schedule", [])}
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        out.append(
            {
                "start": str(e.get("start", "")),
                "end": str(e.get("end", "")),
                "who": str(e.get("who", "")),
                "what": str(e.get("what", "")),
                "location": str(e.get("location", "")),
                "notes": str(e.get("notes", "")),
            }
        )
    return out


def _alert_min(row: dict[str, Any]) -> int | None:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    disruptions = metadata.get("disruptions", [])
    if not isinstance(disruptions, list):
        return None
    candidates: list[int] = []
    for d in disruptions:
        if not isinstance(d, dict):
            continue
        for k in ("start_time", "new_arrival_time", "arrival_time"):
            mm = _to_minutes(str(d.get(k, "")))
            if mm is not None:
                candidates.append(mm)
                break
    return min(candidates) if candidates else None


def _cmp_category(guard_match: int, scorer_match: int, scorer_mono: int) -> str:
    if scorer_mono == 0:
        return "SCORER_NON_MONOTONIC"
    if guard_match and scorer_match:
        return "BOTH_MATCH"
    if guard_match and not scorer_match:
        return "GUARD_MATCH_SCORER_MISMATCH"
    if (not guard_match) and scorer_match:
        return "GUARD_MISMATCH_SCORER_MATCH"
    return "BOTH_MISMATCH"


def main() -> None:
    args = parse_args()
    rows = [r for r in read_jsonl(args.input_jsonl) if str(r.get("variant", "")) == args.variant]
    if args.stage:
        rows = [r for r in rows if str(r.get("stage", "")) == args.stage]

    fail_rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    case_rows: list[dict[str, Any]] = []

    for row in rows:
        if int(row.get("crossing_split_applicable", 0)) != 1:
            continue
        if int(row.get("immutable_prefix_after_split_ok", 1)) == 1:
            continue
        fail_rows.append(row)

        guard_match = int(row.get("guard_view_hash_match", 0))
        scorer_match = int(row.get("scorer_view_hash_match", 0))

        alert_min = _alert_min(row)
        locked = row.get("locked_prefix_snapshot", []) if isinstance(row.get("locked_prefix_snapshot", []), list) else []
        final_events = _extract_events(str(row.get("prediction", "")))

        if not row.get("scorer_view_locked_hash"):
            locked_h, locked_m, locked_c = _scorer_view_hash(locked, alert_min)
        else:
            locked_h = str(row.get("scorer_view_locked_hash", ""))
            locked_m = int(row.get("scorer_view_locked_monotonic", 0))
            _, _, locked_c = _scorer_view_hash(locked, alert_min)

        if not row.get("scorer_view_final_hash"):
            final_h, final_m, final_c = _scorer_view_hash(final_events, alert_min)
        else:
            final_h = str(row.get("scorer_view_final_hash", ""))
            final_m = int(row.get("scorer_view_final_monotonic", 0))
            _, _, final_c = _scorer_view_hash(final_events, alert_min)

        scorer_match = int(bool(locked_h) and bool(final_h) and locked_h == final_h)
        category = _cmp_category(guard_match, scorer_match, final_m)
        counter[category] += 1

        case_rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "seed": int(row.get("seed", 0)),
                "stage": str(row.get("stage", "")),
                "guard_view_hash_match": guard_match,
                "scorer_view_hash_match": scorer_match,
                "scorer_view_final_monotonic": final_m,
                "category": category,
                "locked_prefix_hash": str(row.get("locked_prefix_hash", "")),
                "final_prefix_hash": str(row.get("final_prefix_hash", "")),
                "scorer_view_locked_hash": locked_h,
                "scorer_view_final_hash": final_h,
                "locked_prefix_events": locked,
                "final_prefix_extracted": row.get("final_prefix_extracted", []),
                "scorer_view_locked_extract": locked_c,
                "scorer_view_final_extract": final_c,
                "violations": row.get("violations", []),
                "immutable_diff_type": str(row.get("immutable_diff_type", "")),
            }
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    fails = len(fail_rows)
    guard_rate = sum(int(r.get("guard_view_hash_match", 0)) for r in fail_rows) / fails if fails else 0.0
    scorer_rate = sum(int(r.get("scorer_view_hash_match", 0)) for r in case_rows) / fails if fails else 0.0

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Immutability Alignment Report\n\n")
        f.write(f"- input_rows({args.variant}): {total}\n")
        f.write(f"- immutable_fail_rows: {fails}\n")
        f.write(f"- guard_view_hash_match_rate_on_fail: {guard_rate:.4f}\n")
        f.write(f"- scorer_view_hash_match_rate_on_fail: {scorer_rate:.4f}\n\n")

        f.write("## Category Counts\n\n")
        f.write("| category | count |\n|---|---:|\n")
        for k, v in counter.most_common():
            f.write(f"| {k} | {v} |\n")

        for case in case_rows[: max(1, args.max_cases)]:
            f.write(
                f"\n\n## {case['sample_id']} | seed={case['seed']} | stage={case['stage']}\n\n"
            )
            f.write(f"- category: {case['category']}\n")
            f.write(f"- guard_view_hash_match: {case['guard_view_hash_match']}\n")
            f.write(f"- scorer_view_hash_match: {case['scorer_view_hash_match']}\n")
            f.write(f"- scorer_view_final_monotonic: {case['scorer_view_final_monotonic']}\n")
            f.write(f"- immutable_diff_type: {case['immutable_diff_type']}\n")
            f.write(f"- locked_prefix_hash: `{case['locked_prefix_hash']}`\n")
            f.write(f"- final_prefix_hash: `{case['final_prefix_hash']}`\n")
            f.write(f"- scorer_view_locked_hash: `{case['scorer_view_locked_hash']}`\n")
            f.write(f"- scorer_view_final_hash: `{case['scorer_view_final_hash']}`\n")
            f.write("- violations:\n")
            if case["violations"]:
                for v in case["violations"][:10]:
                    f.write(f"  - {v}\n")
            else:
                f.write("  - (none)\n")
            f.write("- scorer_view_locked_extract:\n")
            f.write("```json\n" + json.dumps(case["scorer_view_locked_extract"], ensure_ascii=False, indent=2) + "\n```\n")
            f.write("- scorer_view_final_extract:\n")
            f.write("```json\n" + json.dumps(case["scorer_view_final_extract"], ensure_ascii=False, indent=2) + "\n```\n")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("category,count\n")
            for k, v in counter.most_common():
                f.write(f"{k},{v}\n")

    print(f"[immutability_alignment] rows={total} fails={fails} out={out_md}")


if __name__ == "__main__":
    main()
