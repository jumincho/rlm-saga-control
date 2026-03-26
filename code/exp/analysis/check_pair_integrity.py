"""Validate paired-run integrity between baseline and extension jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check paired-run key integrity.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-variant", default=None)
    parser.add_argument("--extension-variant", default=None)
    return parser.parse_args()


def _read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    bdf = _read_jsonl(args.baseline)
    edf = _read_jsonl(args.extension)

    if args.baseline_variant and "variant" in bdf.columns:
        bdf = bdf[bdf["variant"] == args.baseline_variant]
    if args.extension_variant and "variant" in edf.columns:
        edf = edf[edf["variant"] == args.extension_variant]

    keys = ["sample_id", "seed"]
    if "stage" in bdf.columns and "stage" in edf.columns:
        keys.append("stage")

    out: dict[str, object] = {
        "baseline_rows": int(len(bdf)),
        "extension_rows": int(len(edf)),
        "pair_keys": keys,
        "baseline_variant_filter": args.baseline_variant,
        "extension_variant_filter": args.extension_variant,
    }

    b_dup = int(bdf.duplicated(subset=keys).sum()) if len(bdf) else 0
    e_dup = int(edf.duplicated(subset=keys).sum()) if len(edf) else 0
    out["baseline_duplicates"] = b_dup
    out["extension_duplicates"] = e_dup

    b_keys = set(tuple(x) for x in bdf[keys].to_records(index=False)) if len(bdf) else set()
    e_keys = set(tuple(x) for x in edf[keys].to_records(index=False)) if len(edf) else set()
    out["baseline_unique_keys"] = len(b_keys)
    out["extension_unique_keys"] = len(e_keys)
    out["matched_pairs"] = len(b_keys.intersection(e_keys))
    out["baseline_only_keys"] = len(b_keys - e_keys)
    out["extension_only_keys"] = len(e_keys - b_keys)
    out["passed"] = bool(
        b_dup == 0 and e_dup == 0 and len(b_keys) == len(e_keys) == len(b_keys.intersection(e_keys))
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
