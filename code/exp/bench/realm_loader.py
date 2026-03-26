"""Loader for REALM-Bench planning tasks (P5/P6/P8/P9)."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

from exp.bench.schema import render_json_instruction
from exp.bench.types import BenchmarkSample

DEFAULT_REALM_REPO = "https://github.com/genglongling/REALM-Bench.git"


def ensure_realm_bench(cache_dir: Path, repo_url: str = DEFAULT_REALM_REPO) -> Path:
    """Clone REALM-Bench into cache_dir if needed and return repo path."""
    repo_path = cache_dir / "REALM-Bench"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not repo_path.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return repo_path

    # Best-effort refresh.
    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "pull", "--ff-only"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass
    return repo_path


def _format_realm_prompt(problem: str, payload: dict[str, Any]) -> str:
    instruction = render_json_instruction("planning")
    payload_str = json.dumps(payload, ensure_ascii=True, indent=2)
    return (
        f"You are solving REALM-Bench {problem}.\n"
        "Use REPL reasoning if needed, but final output must follow the required JSON schema.\n"
        "Output must be a single JSON object only, no markdown fences and no extra commentary.\n"
        "Example shape: {\"plan_summary\":\"...\",\"events\":[{\"start\":\"09:00\",\"end\":\"09:30\",\"who\":\"A\",\"what\":\"task\",\"location\":\"X\"}]}\n"
        f"{instruction}\n\n"
        "Instance data:\n"
        f"{payload_str}"
    )


def _to_minutes(hhmm: str) -> int | None:
    try:
        h, m = str(hhmm).split(":")
        hh = int(h)
        mm = int(m)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh * 60 + mm
    except Exception:
        return None


def _to_hhmm(minutes: int) -> str:
    minutes = max(0, minutes)
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _build_boundary_crossing_addendum(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    disruptions = payload.get("disruptions", []) or []
    target = disruptions[0] if disruptions else {}
    route = str(target.get("route", "hotel-church"))
    alert = str(target.get("start_time", "13:00"))
    alert_min = _to_minutes(alert) or 13 * 60
    depart = _to_hhmm(alert_min - 10)
    planned_end = _to_hhmm(alert_min + 20)
    addendum = (
        "\n\nAdditional boundary-crossing requirement (for partial-compensation evaluation):\n"
        f"- Include at least one travel segment on route `{route}` that starts at {depart} "
        f"and crosses the disruption boundary at {alert} (planned end around {planned_end}).\n"
        "- Keep pre-alert elapsed progress immutable and compensate only the post-alert remainder.\n"
        "- In notes, explicitly mention that only the remaining segment after alert was compensated.\n"
    )
    boundary_meta = {
        "require_boundary_crossing": True,
        "boundary_route": route,
        "boundary_alert_time": alert,
        "boundary_departure_hint": depart,
        "boundary_planned_end_hint": planned_end,
        "scenario_disruption_applicable": True,
        "scenario_partial_compensation_required": True,
        "scenario_immutable_check_applicable": True,
        "scenario_state_check_applicable": True,
    }
    return addendum, boundary_meta


def load_realm_samples(
    realm_repo_path: Path,
    problems: list[str],
    limit_per_problem: int,
    seed: int,
    specific_ids: dict[str, list[str]] | None = None,
    boundary_crossing_only: bool = False,
    scenario_applicability_mode: str = "auto",
) -> list[BenchmarkSample]:
    """Load REALM samples from JSON files under datasets/P*/ directories."""
    rng = random.Random(seed)
    samples: list[BenchmarkSample] = []

    for problem in problems:
        if boundary_crossing_only and problem.upper() != "P8":
            continue
        folder = realm_repo_path / "datasets" / problem
        if not folder.exists():
            continue

        json_files = sorted(folder.glob("**/*.json"))
        if specific_ids and problem in specific_ids:
            wanted = {sid.lower() for sid in specific_ids[problem]}
            selected = [f for f in json_files if f.stem.lower() in wanted]
        else:
            selected = json_files[:]
            rng.shuffle(selected)
            selected = selected[:limit_per_problem]

        for file_path in selected:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            sample_id = payload.get("instance_id", file_path.stem)
            disruptions = payload.get("disruptions", [])
            metadata = {
                "problem": problem,
                "file_path": str(file_path),
                "constraints": payload.get("constraints", {}),
                "disruptions": disruptions,
                "locations": payload.get("locations", {}),
                "travel_times": payload.get("travel_times", {}),
            }
            prompt = _format_realm_prompt(problem, payload)

            # Scenario-based applicability flags to prevent output-dependent denominator drift.
            if scenario_applicability_mode == "p8_fixed" and problem.upper() == "P8":
                metadata["scenario_disruption_applicable"] = True
                metadata["scenario_immutable_check_applicable"] = True
                metadata["scenario_state_check_applicable"] = True

            # Boundary-crossing focused track for partial-compensation verification.
            if boundary_crossing_only and problem.upper() == "P8":
                addendum, boundary_meta = _build_boundary_crossing_addendum(payload)
                prompt += addendum
                metadata.update(boundary_meta)

            samples.append(
                BenchmarkSample(
                    sample_id=f"{problem}:{sample_id}",
                    track="planning",
                    source="realm-bench",
                    task_type="planning",
                    prompt=prompt,
                    answer=None,
                    metadata=metadata,
                )
            )

    return samples
