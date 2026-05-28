"""Loader for OOLONG synthetic long-context QA samples.

OOLONG is the second long-context benchmark, complementing LongBench-v2.
Items here are open-form QA rather than MCQ; the scorer falls back to
exact-match plus a tolerant span match. Loader can stratify selection
by context-length bucket (short / medium / long / xlong) so a small
sample still covers the long tail. Like LongBench, this track does not
exercise the Saga repair path — it's there to confirm the extension
variants aren't trading away long-context QA accuracy in exchange for
the planning-track gains.
"""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from exp.bench.schema import render_json_instruction
from exp.bench.types import BenchmarkSample


def _to_answer_text(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return str(value)


def _build_oolong_prompt(context: str, question: str) -> str:
    instruction = render_json_instruction("qa")
    return (
        "Solve the question using the long context below.\n"
        "You may use REPL decomposition and sub-calls.\n"
        f"{instruction}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}"
    )


def _bucket_index(context_len: int) -> str:
    if context_len < 8_000:
        return "short"
    if context_len < 32_000:
        return "medium"
    if context_len < 128_000:
        return "long"
    return "xlong"


def load_oolong_samples(
    limit: int,
    seed: int,
    dataset_id: str = "oolongbench/oolong-synth",
    split: str = "validation",
    stratified: bool = False,
) -> list[BenchmarkSample]:
    """Load OOLONG synth samples."""
    ds = load_dataset(dataset_id, split=split)

    records = list(ds)
    rng = random.Random(seed)

    if not stratified:
        records = records[:limit]
    else:
        buckets: dict[str, list[dict[str, Any]]] = {"short": [], "medium": [], "long": [], "xlong": []}
        for row in records:
            buckets[_bucket_index(int(row.get("context_len", 0)))].append(row)
        per_bucket = max(1, limit // max(1, len(buckets)))
        selected: list[dict[str, Any]] = []
        for key in ["short", "medium", "long", "xlong"]:
            rows = buckets[key]
            rng.shuffle(rows)
            selected.extend(rows[:per_bucket])
        if len(selected) < limit:
            remaining = [r for r in records if r not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: limit - len(selected)])
        records = selected[:limit]

    samples: list[BenchmarkSample] = []
    for row in records:
        sample_id = f"oolong:{row['id']}"
        prompt = _build_oolong_prompt(row["context_window_text"], row["question"])
        answer = _to_answer_text(row.get("answer"))
        metadata = {
            "task_group": row.get("task_group"),
            "task": row.get("task"),
            "context_len": int(row.get("context_len", 0)),
            "answer_type": row.get("answer_type"),
            "dataset": row.get("dataset"),
        }
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                track="long_context",
                source=dataset_id,
                task_type="qa",
                prompt=prompt,
                answer=answer,
                metadata=metadata,
            )
        )

    return samples
