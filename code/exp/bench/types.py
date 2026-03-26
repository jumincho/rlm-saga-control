"""Typed sample containers for benchmark runners."""

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
