"""Decompose the gate's false-positive cases — "gate-pass but strict-fail".

The companion to `gate_alignment_report.py`. Once you know there are
false positives (the gate let the plan through but strict scoring
rejects it), this script breaks them down by *what* the strict
evaluator complained about: which violation strings appear most often
on FP rows, which combinations co-occur, and which
`state_consistency_failure_reason` values dominate. The output is the
direct input to the closure-report's "remaining FP causes" line and
the v7.14 round was specifically aimed at clearing the top of this
table (`PHOTO_TIME_EXCEEDED` and timeline-monotonicity failures).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sample_id", "")),
        int(row.get("seed", 0)),
        str(row.get("variant", "")),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate FP breakdown")
    p.add_argument("--runtime-jsonl", required=True)
    p.add_argument("--strict-jsonl", required=True)
    p.add_argument("--variant", default="V3_PREFIX_SPLIT")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runtime_rows = [
        r
        for r in _read_jsonl(Path(args.runtime_jsonl))
        if str(r.get("variant", "")) == args.variant
    ]
    strict_rows = [
        r
        for r in _read_jsonl(Path(args.strict_jsonl))
        if str(r.get("variant", "")) == args.variant
    ]

    runtime_map = {_key(r): r for r in runtime_rows}
    strict_map = {_key(r): r for r in strict_rows}
    keys = sorted(set(runtime_map.keys()) & set(strict_map.keys()))

    joined: list[dict[str, Any]] = []
    for k in keys:
        rr = runtime_map[k]
        sr = strict_map[k]
        gate_pass = int(rr.get("boundary_gate_passed", 0))
        strict_pass = int(sr.get("success", 0))
        strict_violations = sr.get("violations", [])
        if not isinstance(strict_violations, list):
            strict_violations = []
        joined.append(
            {
                "sample_id": k[0],
                "seed": k[1],
                "variant": k[2],
                "gate_pass": gate_pass,
                "strict_pass": strict_pass,
                "strict_violation_count": int(sr.get("violation_count", 0)),
                "strict_violations": [str(v) for v in strict_violations],
                "gate_fail_reasons": rr.get("boundary_gate_failed_reasons", []),
                "state_consistency_failure_reason": rr.get(
                    "state_consistency_failure_reason", ""
                ),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if not joined:
        pd.DataFrame().to_csv(out_csv, index=False)
        out_md.write_text("# Gate FP Breakdown\n\nNo matched rows.\n", encoding="utf-8")
        print("[gate_fp_breakdown] matched=0")
        return

    df = pd.DataFrame(joined)
    fp = df[(df["gate_pass"] == 1) & (df["strict_pass"] == 0)].copy()
    fp.to_csv(out_csv, index=False)

    violation_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    state_reason_counter: Counter[str] = Counter()
    for _, row in fp.iterrows():
        violations = [str(v) for v in row.get("strict_violations", [])]
        for v in violations:
            violation_counter[v] += 1
        combo_key = " | ".join(sorted(violations)) if violations else "(none)"
        combo_counter[combo_key] += 1
        state_reason_counter[str(row.get("state_consistency_failure_reason", ""))] += 1

    top_violations = violation_counter.most_common(max(1, int(args.top_k)))
    top_combos = combo_counter.most_common(max(1, min(int(args.top_k), 10)))
    top_state = state_reason_counter.most_common(max(1, min(int(args.top_k), 10)))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Gate FP Breakdown\n\n")
        f.write(f"- variant: `{args.variant}`\n")
        f.write(f"- matched rows: `{len(df)}`\n")
        f.write(f"- gate_pass & strict_fail (FP): `{len(fp)}`\n\n")

        f.write("## Top Strict Violations On FP Cases\n\n")
        if not top_violations:
            f.write("- none\n")
        else:
            for v, c in top_violations:
                f.write(f"- `{v}`: {c}\n")

        f.write("\n## Top Violation Combos On FP Cases\n\n")
        if not top_combos:
            f.write("- none\n")
        else:
            for combo, c in top_combos:
                f.write(f"- `{combo}`: {c}\n")

        f.write("\n## State Failure Reasons On FP Cases\n\n")
        if not top_state:
            f.write("- none\n")
        else:
            for reason, c in top_state:
                f.write(f"- `{reason}`: {c}\n")

        f.write("\n## Sample FP Cases\n\n")
        show = fp.head(max(1, int(args.max_cases)))
        if show.empty:
            f.write("- none\n")
        else:
            for _, row in show.iterrows():
                f.write(
                    f"- sample `{row['sample_id']}` seed `{int(row['seed'])}`\n"
                    f"  - strict_violations: `{row['strict_violations']}`\n"
                    f"  - state_reason: `{row['state_consistency_failure_reason']}`\n"
                    f"  - gate_fail_reasons(runtime): `{row['gate_fail_reasons']}`\n"
                )

    print(
        f"[gate_fp_breakdown] matched={len(df)} fp={len(fp)} "
        f"out_md={out_md} out_csv={out_csv}"
    )


if __name__ == "__main__":
    main()
