"""Sign test + bootstrap CI on paired success diffs (the headline number).

The arithmetic that turns paired baseline + extension JSONL into the
closure-report's headline claim ("V3 wins more samples than V0 with
p < 0.05"). For every (sample_id, seed, stage) that appears for both
the `left` (default `V0`) and `right` (default `V3_PREFIX`) variants
this computes `success_right - success_left`, then reports:

- `wins / losses / ties` counts.
- `mean_success_diff` and a non-parametric bootstrap 95% CI on the
  paired diff.
- A two-sided sign-test p-value on (wins, losses).

The argparse defaults (`left=V0, right=V3_PREFIX`) match the
configuration used in the last completed paired comparison (v7.13)
described in the closure report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired stats from rescored JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--left", default="V0")
    parser.add_argument("--right", default="V3_PREFIX")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pair_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sample_id", "")),
        int(row.get("seed", 0)),
        str(row.get("stage", "")),
    )


def sign_test_two_sided_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = max(wins, losses)
    tail = 0.0
    for i in range(k, n + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci_mean(values: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(max(100, n_boot)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / float(n))
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return means[lo_idx], means[hi_idx]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)

    by_key: dict[tuple[str, int, str], dict[str, int]] = {}
    for row in rows:
        key = _pair_key(row)
        v = str(row.get("variant", ""))
        if v not in {args.left, args.right}:
            continue
        by_key.setdefault(key, {})[v] = int(row.get("success", 0))

    diffs: list[float] = []
    wins = losses = ties = 0
    for d in by_key.values():
        if args.left not in d or args.right not in d:
            continue
        diff = float(d[args.right] - d[args.left])
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            ties += 1

    n = len(diffs)
    mean_diff = (sum(diffs) / float(n)) if n > 0 else 0.0
    ci_lo, ci_hi = bootstrap_ci_mean(diffs, n_boot=args.bootstrap, seed=0)
    p_value = sign_test_two_sided_p(wins, losses)

    out = {
        "left": args.left,
        "right": args.right,
        "paired_n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_success_diff": mean_diff,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "sign_test_p_two_sided": p_value,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
