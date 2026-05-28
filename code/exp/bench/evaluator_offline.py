"""Offline evaluator: re-score completed runs under runtime/strict/relaxed.

`scorer.score_prediction` is a single function but its behavior pivots on
a *policy* string. This module exposes those policies under the three
reporting modes the closure report cares about:

- `runtime`  — what the Saga layer used live during the run; lenient on
  immutability / state hard-fails so the agent doesn't get penalized for
  things it could plausibly still recover from.
- `strict`   — the offline "is this actually correct" judge; enforces all
  three hard constraint families (disruption / immutable prefix / state
  consistency).
- `relaxed`  — sits between the two: hard on disruption handling, soft on
  immutable prefix and state consistency, used to sanity-check that the
  strict failures are not just the strict policy being severe.

`summarize_dual_eval` calls this module in all three modes against the same
raw JSONL to produce side-by-side runtime / strict / relaxed metrics.
"""

from __future__ import annotations

from exp.bench.scorer import (
    RELAXED_POLICY,
    RUNTIME_POLICY,
    STRICT_POLICY,
    get_policy_version,
    score_prediction,
)
from exp.bench.types import BenchmarkSample

EVALUATOR_MODES = {
    "runtime": RUNTIME_POLICY,
    "strict": STRICT_POLICY,
    "relaxed": RELAXED_POLICY,
}


def get_evaluator_version(mode: str) -> str:
    policy = EVALUATOR_MODES.get(mode, RUNTIME_POLICY)
    return get_policy_version(policy)


def evaluate_prediction(
    sample: BenchmarkSample,
    prediction: str,
    mode: str = "runtime",
) -> dict:
    policy = EVALUATOR_MODES.get(mode, RUNTIME_POLICY)
    return score_prediction(sample=sample, prediction=prediction, policy=policy)
