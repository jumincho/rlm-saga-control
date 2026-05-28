"""Loader for LongBench-v2 multiple-choice long-context items.

LongBench-v2 is one of the two long-context benchmarks the experiment
covers (the other is OOLONG). Each item becomes a `BenchmarkSample` with
`task_type="mcq"`: the scorer will extract a single A/B/C/D letter from
the model's response and compare to the gold letter. The loader can
stratify selection by (domain, length) so a small budget still gets
coverage across domains and context-length buckets — that's the path
the staged configs take. Track is `long_context`; this track does not
exercise the Saga repair path, it's there as a sanity-check that the
extension variants don't *hurt* on tasks where disturbance recovery is
not the point.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from datasets import load_dataset

from exp.bench.schema import render_json_instruction
from exp.bench.types import BenchmarkSample


def _build_longbench_prompt(context: str, question: str, choices: dict[str, str]) -> str:
    instruction = render_json_instruction("qa")
    formatted_choices = "\n".join([f"{k}: {v}" for k, v in choices.items()])
    return (
        "Answer the multiple-choice question using the context.\n"
        f"{instruction}\n"
        "Return only the choice letter (A/B/C/D).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{formatted_choices}"
    )


def load_longbench_samples(
    limit: int,
    seed: int,
    dataset_id: str = "zai-org/LongBench-v2",
    split: str = "train",
    stratified: bool = True,
) -> list[BenchmarkSample]:
    """Load LongBench-v2 with optional domain x length stratification."""
    ds = load_dataset(dataset_id, split=split)
    rows = list(ds)
    rng = random.Random(seed)

    if stratified:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row.get("domain", "unknown"), row.get("length", "unknown"))].append(row)

        group_keys = sorted(groups.keys())
        per_group = max(1, limit // max(1, len(group_keys)))
        selected: list[dict[str, Any]] = []
        for key in group_keys:
            bucket = groups[key]
            rng.shuffle(bucket)
            selected.extend(bucket[:per_group])

        if len(selected) < limit:
            remaining = [r for r in rows if r not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: limit - len(selected)])
        rows = selected[:limit]
    else:
        rng.shuffle(rows)
        rows = rows[:limit]

    samples: list[BenchmarkSample] = []
    for row in rows:
        choices = {
            "A": row.get("choice_A", ""),
            "B": row.get("choice_B", ""),
            "C": row.get("choice_C", ""),
            "D": row.get("choice_D", ""),
        }
        prompt = _build_longbench_prompt(row.get("context", ""), row.get("question", ""), choices)
        answer = str(row.get("answer", "")).strip()
        sample_id = f"longbench:{row.get('_id', 'unknown')}"
        metadata = {
            "domain": row.get("domain"),
            "sub_domain": row.get("sub_domain"),
            "difficulty": row.get("difficulty"),
            "length": row.get("length"),
        }
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                track="long_context",
                source=dataset_id,
                task_type="mcq",
                prompt=prompt,
                answer=answer,
                metadata=metadata,
            )
        )
    return samples
