"""Offline evaluator interface for strict/relaxed reporting."""

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
