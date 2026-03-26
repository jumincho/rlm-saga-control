"""Runtime validator interface (used during agent execution)."""

from __future__ import annotations

from exp.bench.scorer import (
    RUNTIME_P8_HARD_POLICY,
    RUNTIME_POLICY,
    build_rule_validator,
    get_policy_version,
)
from exp.bench.types import BenchmarkSample

VALIDATOR_RUNTIME_POLICY = RUNTIME_POLICY
VALIDATOR_RUNTIME_VERSION = get_policy_version(VALIDATOR_RUNTIME_POLICY)


def resolve_runtime_validator_policy(sample: BenchmarkSample, mode: str = "auto") -> str:
    if mode == "p8_hard":
        return RUNTIME_P8_HARD_POLICY
    if mode == "runtime":
        return RUNTIME_POLICY

    problem = str(sample.metadata.get("problem", "")).strip().upper()
    if problem == "P8":
        return RUNTIME_P8_HARD_POLICY
    return RUNTIME_POLICY


def build_runtime_validator(sample: BenchmarkSample, mode: str = "auto"):
    policy = resolve_runtime_validator_policy(sample, mode=mode)
    return build_rule_validator(sample, policy=policy)


def get_runtime_validator_version(sample: BenchmarkSample, mode: str = "auto") -> str:
    policy = resolve_runtime_validator_policy(sample, mode=mode)
    return get_policy_version(policy)
