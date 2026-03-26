"""Report boundary gate alignment against strict offline evaluator."""

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


def _as_bool_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row.get("sample_id", "")), int(row.get("seed", 0)), str(row.get("variant", "")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate alignment report")
    p.add_argument("--runtime-jsonl", required=True)
    p.add_argument("--strict-jsonl", required=True)
    p.add_argument("--variant", default="V3_PREFIX_SPLIT")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--max-cases", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runtime_rows = [r for r in _read_jsonl(Path(args.runtime_jsonl)) if str(r.get("variant", "")) == args.variant]
    strict_rows = [r for r in _read_jsonl(Path(args.strict_jsonl)) if str(r.get("variant", "")) == args.variant]

    runtime_map = {_key(r): r for r in runtime_rows}
    strict_map = {_key(r): r for r in strict_rows}
    keys = sorted(set(runtime_map.keys()) & set(strict_map.keys()))

    joined: list[dict[str, Any]] = []
    for k in keys:
        rr = runtime_map[k]
        sr = strict_map[k]
        strict_pass = _as_bool_int(sr.get("success", 0))
        gate_pass = _as_bool_int(rr.get("boundary_gate_passed", 0))
        gate_fail_reasons = rr.get("boundary_gate_failed_reasons", [])
        if not isinstance(gate_fail_reasons, list):
            gate_fail_reasons = []
        joined.append(
            {
                "sample_id": k[0],
                "seed": k[1],
                "variant": k[2],
                "strict_pass": strict_pass,
                "gate_pass": gate_pass,
                "strict_violation_count": int(sr.get("violation_count", 0)),
                "gate_fail_reasons": gate_fail_reasons,
                "strict_violations": sr.get("violations", []),
            }
        )

    if not joined:
        Path(args.out_md).write_text("# Gate Alignment Report\n\nNo matched rows.\n", encoding="utf-8")
        pd.DataFrame().to_csv(args.out_csv, index=False)
        print(f"[gate_alignment] matched=0 out={args.out_md}")
        return

    df = pd.DataFrame(joined)
    strict_pass = df[df["strict_pass"] == 1]
    gate_pass = df[df["gate_pass"] == 1]
    strict_and_gate = df[(df["strict_pass"] == 1) & (df["gate_pass"] == 1)]

    tp = int(((df["strict_pass"] == 1) & (df["gate_pass"] == 1)).sum())
    fn = int(((df["strict_pass"] == 1) & (df["gate_pass"] == 0)).sum())
    fp = int(((df["strict_pass"] == 0) & (df["gate_pass"] == 1)).sum())
    tn = int(((df["strict_pass"] == 0) & (df["gate_pass"] == 0)).sum())

    gate_recall = float(tp / len(strict_pass)) if len(strict_pass) else 0.0
    gate_precision = float(tp / len(gate_pass)) if len(gate_pass) else 0.0

    fail_on_strict_pass = df[(df["strict_pass"] == 1) & (df["gate_pass"] == 0)]
    reason_counter: Counter[str] = Counter()
    for reasons in fail_on_strict_pass["gate_fail_reasons"].tolist():
        for reason in reasons:
            reason_counter[str(reason)] += 1
    top_reasons = reason_counter.most_common(10)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Gate Alignment Report\n\n")
        f.write(f"- variant: `{args.variant}`\n")
        f.write(f"- matched rows: `{len(df)}`\n")
        f.write(f"- strict_pass & gate_pass (TP): `{tp}`\n")
        f.write(f"- strict_pass & gate_fail (FN): `{fn}`\n")
        f.write(f"- strict_fail & gate_pass (FP): `{fp}`\n")
        f.write(f"- strict_fail & gate_fail (TN): `{tn}`\n")
        f.write(f"- gate_recall_on_strict_pass: `{gate_recall:.4f}`\n")
        f.write(f"- gate_precision_on_gate_pass: `{gate_precision:.4f}`\n\n")

        f.write("## Gate-Fail Reasons On Strict-Pass Cases\n\n")
        if not top_reasons:
            f.write("- none\n")
        else:
            for reason, count in top_reasons:
                f.write(f"- `{reason}`: {count}\n")

        f.write("\n## Strict-Pass But Gate-Fail Samples\n\n")
        show = fail_on_strict_pass.head(max(1, int(args.max_cases)))
        if show.empty:
            f.write("- none\n")
        else:
            for _, row in show.iterrows():
                f.write(
                    f"- sample `{row['sample_id']}` seed `{int(row['seed'])}`\n"
                    f"  - gate_fail_reasons: `{row['gate_fail_reasons']}`\n"
                    f"  - strict_violations: `{row['strict_violations']}`\n"
                )

        f.write("\n## Strict-Pass And Gate-Pass Samples\n\n")
        show_ok = strict_and_gate.head(max(1, int(args.max_cases)))
        if show_ok.empty:
            f.write("- none\n")
        else:
            for _, row in show_ok.iterrows():
                f.write(f"- sample `{row['sample_id']}` seed `{int(row['seed'])}`\n")

    print(
        f"[gate_alignment] matched={len(df)} tp={tp} fn={fn} fp={fp} tn={tn} "
        f"recall={gate_recall:.4f} precision={gate_precision:.4f} out={out_md}"
    )


if __name__ == "__main__":
    main()

