"""Typed sample containers shared across loaders, runners, and scorers.

Every benchmark loader (REALM-Bench planning, LongBench-v2 long-context
MCQ, OOLONG long-context QA) returns a list of `BenchmarkSample` records,
and every downstream module — runners, scorer, validators, analysis —
keys off the same dataclass. The `track` field separates planning runs
from long-context runs; `task_type` ("planning" / "mcq" / "qa") routes
into the right branch of `scorer.score_prediction`; `metadata` is the
free-form per-sample payload that carries scenario-specific information
(disruptions, locations, travel times, locked-prefix events, boundary-
crossing hints) the scorer and the Saga layer need to reason about
immutability and recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchmarkSample:
    sample_id: str
    track: str
    source: str
    task_type: str
    prompt: str
    answer: str | list[str] | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
