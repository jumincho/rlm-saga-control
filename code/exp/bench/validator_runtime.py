"""Runtime validator the Saga layer calls during agent execution.

When the agent emits a candidate plan, the Saga layer hands it to this
validator before committing — accept, augment (give back feedback and
let the model retry), or reject. The wrapped scorer policy is normally
`runtime_v3`, which is deliberately softer than the offline strict
judge so the agent doesn't get knocked out by recoverable issues. P8
(boundary-crossing) samples get the harder `runtime_p8_hard_v1` policy
because the partial-compensation invariant is the whole point there;
`resolve_runtime_validator_policy` picks the policy based on the sample
metadata's `problem` field.

The "auto" mode is what the runner uses by default; "runtime" and
"p8_hard" overrides exist for debug and for the hard-mode follow-up
runs (v7.x). `get_runtime_validator_version` is recorded in the per-row
output so re-scoring under a different evaluator never gets confused
about which policy a row was committed under.
"""

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
