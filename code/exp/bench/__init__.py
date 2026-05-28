"""Benchmark-side code: sample loaders, output schema, and scorer.

Holds the per-benchmark sample loaders (REALM-Bench planning tasks,
LongBench-v2 long-context QA, OOLONG synthetic long-context), the strict
output JSON schema and parser, the scoring policy family (runtime / strict
/ relaxed), and the runtime validator that the Saga layer calls during
agent execution. `scorer.py` is the load-bearing piece — its strict policy
defines what counts as a successful repair under disturbance.
"""
