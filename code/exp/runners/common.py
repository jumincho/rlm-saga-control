"""Shared utilities for baseline/extension experiment runners."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from exp.bench.longbench_loader import load_longbench_samples
from exp.bench.oolong_loader import load_oolong_samples
from exp.bench.realm_loader import ensure_realm_bench, load_realm_samples
from exp.bench.evaluator_offline import evaluate_prediction, get_evaluator_version
from exp.bench.schema import parse_json_response
from exp.bench.types import BenchmarkSample
from exp.bench.validator_runtime import (
    build_runtime_validator,
    get_runtime_validator_version,
)

# Ensure local repository is importable when running from workspace root.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RLM_REPO_ROOT = WORKSPACE_ROOT / "rlm"
if str(RLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RLM_REPO_ROOT))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(config_path: str | Path, stage: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    if "run" not in config or stage not in config["run"]:
        raise ValueError(f"Missing run stage '{stage}' in config {config_path}")
    return {
        "global": config,
        "stage": config["run"][stage],
    }


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def should_inject_failure(sample_id: str, seed: int, ratio: float) -> bool:
    if ratio <= 0:
        return False
    bucket = _stable_hash(f"{sample_id}|{seed}") % 10_000
    return bucket < int(ratio * 10_000)


def apply_failure_injection(sample: BenchmarkSample, seed: int, ratio: float) -> BenchmarkSample:
    """Deterministically inject failures based on sample_id + seed."""
    if not should_inject_failure(sample.sample_id, seed, ratio):
        return sample

    mode = _stable_hash(f"inject|{sample.sample_id}|{seed}") % 3
    metadata = copy.deepcopy(sample.metadata)
    prompt = sample.prompt

    if mode == 0:
        prompt += (
            "\n\nBefore solving, intentionally execute a failing REPL statement "
            "(`raise RuntimeError('INJECTED_FAILURE')`) once, then recover and proceed."
        )
        metadata["failure_injection"] = "code_exception"
    elif mode == 1:
        prompt += (
            "\n\nBefore solving, intentionally try reading `/nonexistent/path.txt` "
            "once in REPL, then recover and continue."
        )
        metadata["failure_injection"] = "file_io_failure"
    else:
        if sample.task_type == "planning" and metadata.get("disruptions"):
            disruptions = copy.deepcopy(metadata["disruptions"])
            for disruption in disruptions:
                if "delay" in disruption and isinstance(disruption["delay"], int):
                    disruption["delay"] += 30
                if "duration" in disruption and isinstance(disruption["duration"], int):
                    disruption["duration"] += 30
            metadata["disruptions"] = disruptions
            prompt += (
                "\n\nUse a stricter disruption condition: all listed disruptions are now stronger "
                "(+30 minutes impact). Replan while preserving immutable past events."
            )
            metadata["failure_injection"] = "disruption_escalation"
        else:
            metadata["failure_injection"] = "none"

    return replace(sample, prompt=prompt, metadata=metadata)


def load_stage_samples(global_cfg: dict[str, Any], stage_cfg: dict[str, Any], seed: int) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []

    paths_cfg = global_cfg.get("paths", {})
    realm_cfg = stage_cfg.get("realm", {})
    if realm_cfg.get("enabled", False):
        cache_dir = Path(paths_cfg.get("realm_cache", str(WORKSPACE_ROOT / ".cache" / "realm")))
        realm_repo = ensure_realm_bench(cache_dir)
        specific_ids = realm_cfg.get("specific_ids")
        samples.extend(
            load_realm_samples(
                realm_repo_path=realm_repo,
                problems=realm_cfg.get("problems", ["P5", "P6", "P8", "P9"]),
                limit_per_problem=int(realm_cfg.get("limit_per_problem", 10)),
                seed=seed,
                specific_ids=specific_ids,
                boundary_crossing_only=bool(realm_cfg.get("boundary_crossing_only", False)),
                scenario_applicability_mode=str(
                    realm_cfg.get("scenario_applicability_mode", "auto")
                ),
            )
        )

    oolong_cfg = stage_cfg.get("oolong", {})
    if oolong_cfg.get("enabled", False):
        samples.extend(
            load_oolong_samples(
                limit=int(oolong_cfg.get("limit", 50)),
                seed=seed,
                dataset_id=oolong_cfg.get("dataset_id", "oolongbench/oolong-synth"),
                split=oolong_cfg.get("split", "validation"),
                stratified=bool(oolong_cfg.get("stratified", False)),
            )
        )

    longbench_cfg = stage_cfg.get("longbench", {})
    if longbench_cfg.get("enabled", False):
        samples.extend(
            load_longbench_samples(
                limit=int(longbench_cfg.get("limit", 30)),
                seed=seed,
                dataset_id=longbench_cfg.get("dataset_id", "zai-org/LongBench-v2"),
                split=longbench_cfg.get("split", "train"),
                stratified=bool(longbench_cfg.get("stratified", True)),
            )
        )

    return samples


def build_rlm_instance(
    variant: str,
    sample: BenchmarkSample,
    model_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    log_dir: Path,
) -> RLM:
    from rlm import RLM
    from rlm.logger import RLMLogger

    backend = model_cfg.get("backend", "vllm")
    model_name = model_cfg.get("model_name", "Qwen/Qwen2.5-14B-Instruct")
    base_url = model_cfg.get("base_url", "http://localhost:8000/v1")
    api_key = os.environ.get(model_cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")

    max_new_tokens = int(model_cfg.get("max_new_tokens", 2048))
    max_iterations = int(run_cfg.get("max_iterations", 30))
    max_timeout = run_cfg.get("max_timeout")

    # Keep long-context runs bounded so they do not dominate wall-clock time.
    if sample.track == "long_context":
        max_new_tokens = min(max_new_tokens, 512)
        max_iterations = min(max_iterations, 3)
        if max_timeout is not None:
            max_timeout = min(float(max_timeout), 120.0)

    backend_kwargs = {
        "model_name": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": model_cfg.get("temperature", 0.2),
        "top_p": model_cfg.get("top_p", 0.95),
        "max_new_tokens": max_new_tokens,
    }

    environment = "local"
    environment_kwargs: dict[str, Any] = {}

    saga_variants = {"V1", "V2", "V3", "V3_BASE", "V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"}
    if variant in saga_variants:
        environment = "local_saga"
        runtime_validator_mode = str(run_cfg.get("runtime_validator_mode", "auto"))
        if variant in {"V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"}:
            runtime_validator_mode = str(run_cfg.get("runtime_validator_mode_prefix", runtime_validator_mode))
        environment_kwargs = {
            "saga_enabled": True,
            "saga_mode": run_cfg.get("saga_mode", "iteration"),
            "saga_file_snapshot_mode": run_cfg.get("saga_file_snapshot_mode", "diff"),
            "saga_rule_validation": variant in {"V2", "V3", "V3_BASE", "V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"},
            "saga_max_retries": int(run_cfg.get("saga_max_retries", 0))
            if variant in {"V3", "V3_BASE", "V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"}
            else 0,
        }
        if variant in {"V2", "V3", "V3_BASE", "V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"}:
            environment_kwargs["saga_validator"] = build_runtime_validator(
                sample, mode=runtime_validator_mode
            )

    logger = RLMLogger(log_dir=str(log_dir), file_name=f"{variant.lower()}_{sample.sample_id.replace(':', '_')}")

    return RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment=environment,
        environment_kwargs=environment_kwargs,
        max_depth=int(run_cfg.get("max_depth", 1)),
        max_iterations=max_iterations,
        max_timeout=max_timeout,
        max_errors=run_cfg.get("max_errors"),
        logger=logger,
        verbose=bool(run_cfg.get("verbose", False)),
    )


def _usage_to_dict(usage_summary: Any) -> dict[str, Any]:
    if usage_summary is None:
        return {}
    try:
        return usage_summary.to_dict()
    except Exception:
        return {}


def _collect_tx_metrics(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {
            "tx_count": 0,
            "tx_rollbacks": 0,
            "tx_retries": 0,
            "invalid_commit_rate": 0.0,
        }

    tx = metadata.get("transactions", [])
    if not isinstance(tx, list):
        tx = []

    commits = [event for event in tx if event.get("phase") == "commit"]
    rollbacks = [event for event in tx if event.get("phase") == "rollback"]
    retries = [event for event in tx if event.get("phase") == "retry"]
    bad_commits = [event for event in commits if event.get("validation_result") == "Rejection"]

    invalid_commit_rate = len(bad_commits) / len(commits) if commits else 0.0

    return {
        "tx_count": len(tx),
        "tx_rollbacks": len(rollbacks),
        "tx_retries": len(retries),
        "invalid_commit_rate": invalid_commit_rate,
    }


def _to_minutes(hhmm: str) -> int | None:
    try:
        parts = str(hhmm).strip().split(":")
        if len(parts) != 2:
            return None
        h = int(parts[0])
        m = int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return h * 60 + m
    except Exception:
        return None


def _resolve_alert_min(disruptions: list[dict[str, Any]]) -> int | None:
    candidates: list[int] = []
    for disruption in disruptions:
        for key in ("start_time", "new_arrival_time", "arrival_time"):
            minute = _to_minutes(str(disruption.get(key, "")))
            if minute is not None:
                candidates.append(minute)
                break
    return min(candidates) if candidates else None


def _extract_events_from_prediction(prediction: str) -> list[dict[str, Any]]:
    parsed, err = parse_json_response(prediction)
    if err or parsed is None:
        return []
    if isinstance(parsed, dict) and "suffix_events" in parsed and "events" not in parsed and isinstance(parsed.get("suffix_events"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("suffix_events", [])}
    if isinstance(parsed, dict) and "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
    events = parsed.get("events", []) if isinstance(parsed, dict) else []
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        out.append(
            {
                "start": str(event.get("start", "")),
                "end": str(event.get("end", "")),
                "who": str(event.get("who", "")),
                "what": str(event.get("what", "")),
                "location": str(event.get("location", "")),
                "notes": str(event.get("notes", "")),
            }
        )
    return out


def _event_signature(event: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(event.get("start", "")),
        str(event.get("end", "")),
        str(event.get("who", "")),
        str(event.get("what", "")),
        str(event.get("location", "")),
        str(event.get("notes", "")),
    )


def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _split_actors_for_monotonic(who: str) -> list[str]:
    actors = [a.strip().lower() for a in re.split(r"[,&/]+", str(who).strip()) if a.strip()]
    return actors or ["_unknown"]


def _events_time_monotonic_like_scorer(events: list[dict[str, Any]]) -> bool:
    by_actor: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None or end_m < start_m:
            return False
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            by_actor.setdefault(actor, []).append((start_m, end_m))

    for windows in by_actor.values():
        windows.sort(key=lambda x: (x[0], x[1]))
        prev_end: int | None = None
        for start_m, end_m in windows:
            if prev_end is not None and start_m < prev_end:
                return False
            prev_end = end_m
    return True


def _scorer_canonical_prefix_view(
    events: list[dict[str, Any]], alert_min: int | None
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]], int]:
    if alert_min is None:
        return [], [], 0
    pre_events: list[dict[str, Any]] = []
    for event in events:
        end_min = _to_minutes(str(event.get("end", "")))
        if end_min is not None and end_min <= alert_min:
            pre_events.append(
                {
                    "start": str(event.get("start", "")),
                    "end": str(event.get("end", "")),
                    "who": str(event.get("who", "")),
                    "what": str(event.get("what", "")),
                    "location": str(event.get("location", "")),
                    "notes": str(event.get("notes", "")),
                }
            )
    canonical: list[tuple[Any, ...]] = []
    for event in pre_events:
        canonical.append(
            (
                _to_minutes(str(event.get("start", ""))),
                _to_minutes(str(event.get("end", ""))),
                _norm_text(event.get("who", "")),
                _norm_text(event.get("what", "")),
                _norm_text(event.get("location", "")),
            )
        )
    canonical.sort(
        key=lambda x: (
            x[0] if x[0] is not None else 10**9,
            x[1] if x[1] is not None else 10**9,
            x[2],
            x[3],
            x[4],
        )
    )
    monotonic = int(_events_time_monotonic_like_scorer(pre_events))
    return pre_events, canonical, monotonic


def _scorer_view_hash(events: list[dict[str, Any]], alert_min: int | None) -> tuple[str, int]:
    _pre, canonical, monotonic = _scorer_canonical_prefix_view(events, alert_min)
    payload = {"canonical": canonical, "time_monotonic": monotonic}
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), monotonic


def _repair_prefix_actor_overlaps(
    events: list[dict[str, Any]], alert_min: int | None
) -> list[dict[str, Any]]:
    if not events:
        return []
    out = _dedup_sort_events(copy.deepcopy(events))
    by_actor_end: dict[str, int] = {}
    for event in out:
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None:
            continue
        actors = _split_actors_for_monotonic(str(event.get("who", "")))
        required_start = start_m
        for actor in actors:
            prev_end = by_actor_end.get(actor)
            if prev_end is not None and prev_end > required_start:
                required_start = prev_end
        if required_start > start_m:
            duration = max(1, end_m - start_m)
            new_start = required_start
            new_end = new_start + duration
            if alert_min is not None and new_end > alert_min:
                new_end = alert_min
                if new_start >= new_end:
                    new_start = max(0, new_end - 1)
            event["start"] = _to_hhmm(new_start)
            event["end"] = _to_hhmm(new_end)
            event["notes"] = (
                str(event.get("notes", "")) + " [prefix_monotonic_fix=1]"
            ).strip()
            start_m, end_m = new_start, new_end
        for actor in actors:
            by_actor_end[actor] = max(by_actor_end.get(actor, 0), end_m)
    return _dedup_sort_events(out)


def _canonical_prefix_events_for_hash(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    canonical: list[tuple[Any, ...]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        canonical.append(
            (
                _to_minutes(str(event.get("start", ""))),
                _to_minutes(str(event.get("end", ""))),
                _norm_text(event.get("who", "")),
                _norm_text(event.get("what", "")),
                _norm_text(event.get("location", "")),
            )
        )
    canonical.sort(
        key=lambda x: (
            x[0] if x[0] is not None else 10**9,
            x[1] if x[1] is not None else 10**9,
            x[2],
            x[3],
            x[4],
        )
    )
    return canonical


def _prefix_hash(events: list[dict[str, Any]]) -> str:
    canonical = _canonical_prefix_events_for_hash(events)
    blob = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _classify_immutable_diff_type(
    locked_prefix_snapshot: list[dict[str, Any]],
    final_prefix_extracted: list[dict[str, Any]],
) -> str:
    if not locked_prefix_snapshot and not final_prefix_extracted:
        return ""

    if not _events_time_monotonic_like_scorer(locked_prefix_snapshot) or not _events_time_monotonic_like_scorer(
        final_prefix_extracted
    ):
        return "NON_MONOTONIC_TIMELINE"

    def _split_pre_subset(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        subset: list[tuple[Any, ...]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
            if "boundary_split_pre" not in blob:
                continue
            subset.append(_event_signature(event))
        subset.sort()
        return subset

    lock_split_pre = _split_pre_subset(locked_prefix_snapshot)
    final_split_pre = _split_pre_subset(final_prefix_extracted)
    if lock_split_pre != final_split_pre:
        return "SPLIT_PRE_REBUILT"

    if len(locked_prefix_snapshot) != len(final_prefix_extracted):
        return "COUNT_OR_ORDER_CHANGED"

    lock_core = _canonical_prefix_events_for_hash(locked_prefix_snapshot)
    final_core = _canonical_prefix_events_for_hash(final_prefix_extracted)

    if lock_core == final_core:
        lock_full = [_event_signature(ev) for ev in locked_prefix_snapshot]
        final_full = [_event_signature(ev) for ev in final_prefix_extracted]
        if lock_full != final_full:
            return "COUNT_OR_ORDER_CHANGED"
        return ""

    lock_no_time = [(x[2], x[3], x[4]) for x in lock_core]
    final_no_time = [(x[2], x[3], x[4]) for x in final_core]
    if lock_no_time == final_no_time:
        return "TIME_ONLY_CHANGED"
    return "LOCATION_OR_STATE_CHANGED"


def _dedup_sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for event in events:
        unique[_event_signature(event)] = event
    out = list(unique.values())
    out.sort(key=lambda ev: (_to_minutes(str(ev.get("start", ""))) or 10**9, _to_minutes(str(ev.get("end", ""))) or 10**9, str(ev.get("who", ""))))
    return out


def _extract_prefix_events(events: list[dict[str, Any]], alert_min: int | None) -> list[dict[str, Any]]:
    if alert_min is None:
        return []
    prefix = []
    for event in events:
        end_min = _to_minutes(str(event.get("end", "")))
        if end_min is not None and end_min <= alert_min:
            prefix.append(_sanitize_prefix_like_event(dict(event), alert_min))
    return _dedup_sort_events(prefix)


def _apply_prefix_lock_json(
    raw_prediction: str,
    locked_prefix_events: list[dict[str, Any]],
    alert_min: int | None,
) -> str | None:
    if not locked_prefix_events or alert_min is None:
        return None
    parsed, err = parse_json_response(raw_prediction)
    if err or parsed is None:
        return None
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
    if not isinstance(parsed, dict):
        return None
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return None
    suffix: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start_min = _to_minutes(str(event.get("start", "")))
        if start_min is not None and start_min >= alert_min:
            suffix.append(
                {
                    "start": str(event.get("start", "")),
                    "end": str(event.get("end", "")),
                    "who": str(event.get("who", "")),
                    "what": str(event.get("what", "")),
                    "location": str(event.get("location", "")),
                    "notes": str(event.get("notes", "")),
                }
            )
    merged = _dedup_sort_events(list(locked_prefix_events) + suffix)
    if not merged:
        return None
    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": merged,
    }
    return json.dumps(payload, ensure_ascii=False)


def _to_hhmm(minutes: int) -> str:
    minutes = max(0, int(minutes))
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _normalize_token_set(text: str) -> set[str]:
    tokens = [tok for tok in text.lower().replace("->", " ").replace("-", " ").split() if len(tok) >= 2]
    return set(tokens)


_DISRUPTION_TERM_RE = re.compile(
    r"\b(disruption|disruptions|traffic|delay|delayed|road[_ ]closure|alert)\b",
    re.IGNORECASE,
)


def _contains_disruption_terms(text: str) -> bool:
    return bool(_DISRUPTION_TERM_RE.search(str(text)))


def _strip_disruption_terms(text: str) -> str:
    cleaned = _DISRUPTION_TERM_RE.sub(" ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_prefix_like_event(event: dict[str, Any], alert_min: int | None) -> bool:
    blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
    if "boundary_split_pre" in blob:
        return True
    end_min = _to_minutes(str(event.get("end", "")))
    return alert_min is not None and end_min is not None and end_min <= alert_min


def _sanitize_prefix_like_event(
    event: dict[str, Any], alert_min: int | None
) -> dict[str, Any]:
    out = dict(event)
    if not _is_prefix_like_event(out, alert_min):
        return out
    out["what"] = _strip_disruption_terms(str(out.get("what", "")))
    out["notes"] = _strip_disruption_terms(str(out.get("notes", "")))
    return out


def _prefix_disruption_flags(
    events: list[dict[str, Any]], alert_min: int | None
) -> tuple[int, int]:
    prefix_contains = 0
    split_pre_contains = 0
    for event in events:
        blob = f"{event.get('what', '')} {event.get('notes', '')}"
        if _is_prefix_like_event(event, alert_min) and _contains_disruption_terms(blob):
            prefix_contains = 1
        marker_blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
        if "boundary_split_pre" in marker_blob and _contains_disruption_terms(blob):
            split_pre_contains = 1
    return prefix_contains, split_pre_contains


def _route_matches(from_loc: str, to_loc: str, metadata: dict[str, Any]) -> bool:
    route = str(metadata.get("boundary_route", "")).strip().lower()
    if not route:
        return True
    route_tokens = _normalize_token_set(route)
    seg_tokens = _normalize_token_set(from_loc) | _normalize_token_set(to_loc)
    return bool(route_tokens.intersection(seg_tokens))


def _lookup_travel_baseline(from_loc: str, to_loc: str, metadata: dict[str, Any]) -> int | None:
    travel = metadata.get("travel_times", {})
    if not isinstance(travel, dict) or not travel:
        return None

    from_n = from_loc.strip().lower()
    to_n = to_loc.strip().lower()
    candidates = [
        f"{from_n}-{to_n}",
        f"{to_n}-{from_n}",
        f"{from_n}->{to_n}",
        f"{to_n}->{from_n}",
    ]
    for key in candidates:
        v = travel.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)

    from_toks = _normalize_token_set(from_n)
    to_toks = _normalize_token_set(to_n)
    for key, value in travel.items():
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        kt = _normalize_token_set(str(key))
        if from_toks and to_toks and from_toks.intersection(kt) and to_toks.intersection(kt):
            return int(value)
    return None


def _resolve_disruption_delay(metadata: dict[str, Any]) -> int:
    disruptions = metadata.get("disruptions", [])
    if not isinstance(disruptions, list):
        return 30
    for disruption in disruptions:
        if not isinstance(disruption, dict):
            continue
        for key in ("delay", "duration"):
            v = disruption.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return 30


def _split_marker_state(events: list[dict[str, Any]]) -> tuple[int, int]:
    has_pre = 0
    has_post = 0
    for event in events:
        notes = str(event.get("notes", "")).lower()
        what = str(event.get("what", "")).lower()
        blob = f"{notes} {what}"
        if "boundary_split_pre" in blob:
            has_pre = 1
        if "boundary_split_post" in blob:
            has_post = 1
    return has_pre, has_post


def _is_split_marker_event(event: dict[str, Any]) -> bool:
    notes = str(event.get("notes", "")).lower()
    what = str(event.get("what", "")).lower()
    blob = f"{notes} {what}"
    return ("boundary_split_pre" in blob) or ("boundary_split_post" in blob)


def _extract_immutable_anchor_events(
    events: list[dict[str, Any]], alert_min: int | None
) -> list[dict[str, Any]]:
    if alert_min is None:
        return []
    anchors: list[dict[str, Any]] = []
    for event in events:
        normalized = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        if _is_split_marker_event(normalized):
            anchors.append(_sanitize_prefix_like_event(normalized, alert_min))
            continue
        end_min = _to_minutes(normalized.get("end", ""))
        if end_min is not None and end_min <= alert_min:
            anchors.append(_sanitize_prefix_like_event(normalized, alert_min))
    return _dedup_sort_events(anchors)


def _apply_immutable_anchor_synthesis_json(
    raw_prediction: str,
    immutable_anchor_events: list[dict[str, Any]],
    alert_min: int | None,
) -> str | None:
    if not immutable_anchor_events or alert_min is None:
        return None
    parsed, err = parse_json_response(raw_prediction)
    if err or parsed is None or not isinstance(parsed, dict):
        return None
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return None
    suffix: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        normalized = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        if _is_split_marker_event(normalized):
            continue
        start_min = _to_minutes(normalized.get("start", ""))
        if start_min is not None and start_min >= alert_min:
            suffix.append(normalized)
    merged = _dedup_sort_events(list(immutable_anchor_events) + suffix)
    if not merged:
        return None
    return json.dumps({"plan_summary": str(parsed.get("plan_summary", "")), "events": merged}, ensure_ascii=False)


def _compute_alert_state_summary(
    immutable_anchor_events: list[dict[str, Any]], alert_min: int | None
) -> dict[str, Any]:
    if not immutable_anchor_events or alert_min is None:
        return {}
    pre, post = 0, 0
    last_loc = ""
    last_actor = ""
    split_post_end_min: int | None = None
    split_post_location = ""
    split_post_actor = ""
    for event in immutable_anchor_events:
        if _is_split_marker_event(event):
            marker_blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
            if "boundary_split_pre" in marker_blob:
                pre = 1
            if "boundary_split_post" in marker_blob:
                post = 1
                post_end = _to_minutes(str(event.get("end", "")))
                if post_end is not None and (
                    split_post_end_min is None or post_end >= split_post_end_min
                ):
                    split_post_end_min = post_end
                    split_post_location = str(event.get("location", "")).strip()
                    split_post_actor = str(event.get("who", "")).strip()
        end_min = _to_minutes(str(event.get("end", "")))
        if end_min is not None and end_min <= alert_min:
            if str(event.get("location", "")).strip():
                last_loc = str(event.get("location", "")).strip()
            if str(event.get("who", "")).strip():
                last_actor = str(event.get("who", "")).strip()
    next_actionable_min = split_post_end_min if split_post_end_min is not None else alert_min
    next_actionable_location = (
        split_post_location if split_post_location else (last_loc or "")
    )
    next_actionable_actor = split_post_actor if split_post_actor else (last_actor or "")
    return {
        "alert_time": _to_hhmm(alert_min),
        "last_location_before_alert": last_loc,
        "last_actor_before_alert": last_actor,
        "split_pre_present": pre,
        "split_post_present": post,
        "anchor_count": len(immutable_anchor_events),
        "split_post_end_time": _to_hhmm(split_post_end_min) if split_post_end_min is not None else "",
        "split_post_destination": split_post_location,
        "split_post_actor": split_post_actor,
        "next_actionable_time": _to_hhmm(next_actionable_min),
        "next_actionable_location": next_actionable_location,
        "next_actionable_actor": next_actionable_actor,
    }


def _resolve_next_actionable_min(alert_state: dict[str, Any], alert_min: int | None) -> int | None:
    if isinstance(alert_state, dict):
        next_time = _to_minutes(str(alert_state.get("next_actionable_time", "")))
        if next_time is not None:
            return next_time
    return alert_min


def _apply_suffix_only_synthesis_json(
    raw_prediction: str,
    immutable_anchor_events: list[dict[str, Any]],
    next_actionable_min: int | None,
    require_suffix_key: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction": None,
        "suffix_only_output_ok": 0,
        "prefix_edit_attempt_detected": 0,
        "prefix_edit_ignored_count": 0,
        "suffix_first_event_time": "",
    }
    if not immutable_anchor_events:
        return result

    parsed, err = parse_json_response(raw_prediction)
    if err or parsed is None or not isinstance(parsed, dict):
        return result

    suffix_events = parsed.get("suffix_events")
    events = parsed.get("events")
    schedule = parsed.get("schedule")
    has_suffix_key = isinstance(suffix_events, list)
    has_full_events_key = isinstance(events, list) or isinstance(schedule, list)

    if "prefix_events" in parsed and isinstance(parsed.get("prefix_events"), list):
        result["prefix_edit_attempt_detected"] = 1
        result["prefix_edit_ignored_count"] = len(parsed.get("prefix_events", []))

    if require_suffix_key and not has_suffix_key:
        if has_full_events_key:
            result["prefix_edit_attempt_detected"] = 1
            result["prefix_edit_ignored_count"] = max(
                int(result.get("prefix_edit_ignored_count", 0)),
                len(events) if isinstance(events, list) else len(schedule) if isinstance(schedule, list) else 1,
            )
        return result

    if has_suffix_key:
        parsed = {
            "plan_summary": str(parsed.get("plan_summary", "")),
            "events": suffix_events,
        }
    elif isinstance(schedule, list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": schedule}

    events = parsed.get("events", [])
    if not isinstance(events, list):
        return result

    suffix: list[dict[str, Any]] = []
    prefix_edits_ignored = int(result.get("prefix_edit_ignored_count", 0))
    for event in events:
        if not isinstance(event, dict):
            continue
        normalized = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        if _is_split_marker_event(normalized):
            prefix_edits_ignored += 1
            continue
        start_min = _to_minutes(normalized.get("start", ""))
        if next_actionable_min is not None:
            if start_min is None or start_min < next_actionable_min:
                prefix_edits_ignored += 1
                continue
        suffix.append(normalized)

    merged = _dedup_sort_events(list(immutable_anchor_events) + suffix)
    if not merged:
        return result

    suffix_start_candidates = []
    for event in suffix:
        start_min = _to_minutes(str(event.get("start", "")))
        if start_min is not None:
            suffix_start_candidates.append(start_min)
    suffix_first = ""
    if suffix_start_candidates:
        suffix_first = _to_hhmm(min(suffix_start_candidates))

    result["prediction"] = json.dumps(
        {"plan_summary": str(parsed.get("plan_summary", "")), "events": merged},
        ensure_ascii=False,
    )
    result["suffix_only_output_ok"] = 1
    result["prefix_edit_attempt_detected"] = int(
        int(result.get("prefix_edit_attempt_detected", 0)) == 1 or prefix_edits_ignored > 0
    )
    result["prefix_edit_ignored_count"] = prefix_edits_ignored
    result["suffix_first_event_time"] = suffix_first
    return result


def _deterministic_suffix_state_fix_json(
    raw_prediction: str,
    *,
    immutable_anchor_events: list[dict[str, Any]],
    next_actionable_min: int | None,
    next_actionable_location: str,
    next_actionable_actor: str,
    run_cfg: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    metrics: dict[str, Any] = {
        "timeline_norm_applied": 0,
        "timeline_norm_total_shift_minutes": 0,
        "timeline_norm_overlap_fixes_count": 0,
        "suffix_first_event_time_before": "",
        "suffix_first_event_time_after": "",
        "timeline_norm_runtime_sec": 0.0,
    }
    norm_started = time.perf_counter()
    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return None, metrics
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
    events = parsed.get("events", [])
    if not isinstance(events, list) or not events:
        return None, metrics

    norm_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        norm_events.append(
            {
                "start": str(event.get("start", "")),
                "end": str(event.get("end", "")),
                "who": str(event.get("who", "")),
                "what": str(event.get("what", "")),
                "location": str(event.get("location", "")),
                "notes": str(event.get("notes", "")),
            }
        )
    if not norm_events:
        return None, metrics

    anchor_signatures = {_event_signature(event) for event in immutable_anchor_events}
    suffix_idx: list[int] = []
    for idx, event in enumerate(norm_events):
        if _is_split_marker_event(event):
            continue
        if _event_signature(event) in anchor_signatures:
            continue
        suffix_idx.append(idx)

    if not suffix_idx:
        metrics["timeline_norm_runtime_sec"] = time.perf_counter() - norm_started
        return None, metrics

    def _suffix_first_time(indices: list[int]) -> int | None:
        starts = []
        for i in indices:
            s = _to_minutes(str(norm_events[i].get("start", "")))
            if s is not None:
                starts.append(s)
        return min(starts) if starts else None

    first_before = _suffix_first_time(suffix_idx)
    if first_before is not None:
        metrics["suffix_first_event_time_before"] = _to_hhmm(first_before)

    total_shift = 0
    overlap_fixes = 0
    location_fix_applied = 0
    changed = False

    # Keep suffix in a stable deterministic order.
    suffix_order = sorted(
        suffix_idx,
        key=lambda i: (
            _to_minutes(str(norm_events[i].get("start", ""))) or 10**9,
            _to_minutes(str(norm_events[i].get("end", ""))) or 10**9,
            i,
        ),
    )

    # Step 1: ensure suffix begins at/after next actionable boundary.
    first_start = _suffix_first_time(suffix_order)
    if (
        bool(run_cfg.get("planning_boundary_state_shift_fix", True))
        and next_actionable_min is not None
        and first_start is not None
        and first_start < next_actionable_min
    ):
        delta = next_actionable_min - first_start
        for idx in suffix_order:
            s = _to_minutes(str(norm_events[idx].get("start", "")))
            e = _to_minutes(str(norm_events[idx].get("end", "")))
            if s is None or e is None:
                continue
            norm_events[idx]["start"] = _to_hhmm(s + delta)
            norm_events[idx]["end"] = _to_hhmm(e + delta)
            norm_events[idx]["notes"] = (
                str(norm_events[idx].get("notes", "")) + f" [state_shift_fix={delta}]"
            ).strip()
        total_shift += delta
        changed = True

    # Step 2: enforce scorer-like actor monotonic timeline for suffix by deterministic shifts.
    actor_prev_end: dict[str, int] = {}
    for event in immutable_anchor_events:
        end_m = _to_minutes(str(event.get("end", "")))
        if end_m is None:
            continue
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            actor_prev_end[actor] = max(actor_prev_end.get(actor, 0), end_m)

    for pos, idx in enumerate(suffix_order):
        event = norm_events[idx]
        s = _to_minutes(str(event.get("start", "")))
        e = _to_minutes(str(event.get("end", "")))
        if s is None or e is None:
            continue
        required_start = s
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            prev_end = actor_prev_end.get(actor)
            if prev_end is not None and prev_end > required_start:
                required_start = prev_end
        if required_start > s:
            delta = required_start - s
            for shifted_idx in suffix_order[pos:]:
                ss = _to_minutes(str(norm_events[shifted_idx].get("start", "")))
                ee = _to_minutes(str(norm_events[shifted_idx].get("end", "")))
                if ss is None or ee is None:
                    continue
                norm_events[shifted_idx]["start"] = _to_hhmm(ss + delta)
                norm_events[shifted_idx]["end"] = _to_hhmm(ee + delta)
                norm_events[shifted_idx]["notes"] = (
                    str(norm_events[shifted_idx].get("notes", "")) + f" [timeline_overlap_fix={delta}]"
                ).strip()
            total_shift += delta
            overlap_fixes += 1
            changed = True
            s += delta
            e += delta
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            actor_prev_end[actor] = max(actor_prev_end.get(actor, 0), e)

    # Step 3: if boundary location mismatches, insert deterministic travel bridge and shift suffix.
    if bool(run_cfg.get("planning_boundary_location_fix", True)):
        first_idx = min(
            suffix_order,
            key=lambda i: (
                _to_minutes(str(norm_events[i].get("start", ""))) or 10**9,
                _to_minutes(str(norm_events[i].get("end", ""))) or 10**9,
            ),
        )
        expected = _normalize_token_set(next_actionable_location)
        observed_loc = str(norm_events[first_idx].get("location", ""))
        observed = _normalize_token_set(observed_loc)
        if expected and observed and not expected.intersection(observed):
            travel_minutes = max(
                5, int(run_cfg.get("planning_boundary_location_fix_travel_minutes", 15))
            )
            first_start = _to_minutes(str(norm_events[first_idx].get("start", "")))
            if first_start is not None:
                travel_start = (
                    next_actionable_min
                    if next_actionable_min is not None
                    else max(0, first_start - travel_minutes)
                )
                travel_end = travel_start + travel_minutes
                if travel_end > first_start:
                    delta = travel_end - first_start
                    for shifted_idx in suffix_order:
                        ss = _to_minutes(str(norm_events[shifted_idx].get("start", "")))
                        ee = _to_minutes(str(norm_events[shifted_idx].get("end", "")))
                        if ss is None or ee is None:
                            continue
                        norm_events[shifted_idx]["start"] = _to_hhmm(ss + delta)
                        norm_events[shifted_idx]["end"] = _to_hhmm(ee + delta)
                        norm_events[shifted_idx]["notes"] = (
                            str(norm_events[shifted_idx].get("notes", ""))
                            + f" [state_location_shift={delta}]"
                        ).strip()
                    total_shift += delta
                    changed = True
                bridge_actor = (
                    next_actionable_actor.strip()
                    or str(norm_events[first_idx].get("who", "")).strip()
                    or "Team"
                )
                travel_event = {
                    "start": _to_hhmm(travel_start),
                    "end": _to_hhmm(travel_end),
                    "who": bridge_actor,
                    "what": "Travel to next suffix location",
                    "location": f"{next_actionable_location}->{observed_loc}",
                    "notes": "state_location_bridge=1 boundary_transition=1",
                }
                norm_events.append(travel_event)
            location_fix_applied = 1
            changed = True

    merged = _dedup_sort_events(norm_events)
    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": merged,
    }
    first_after = _suffix_first_time(suffix_order)
    if first_after is not None:
        metrics["suffix_first_event_time_after"] = _to_hhmm(first_after)
    metrics["timeline_norm_total_shift_minutes"] = total_shift
    metrics["timeline_norm_overlap_fixes_count"] = overlap_fixes
    metrics["timeline_norm_location_fix_applied"] = location_fix_applied
    metrics["timeline_norm_applied"] = int(changed)
    metrics["timeline_norm_runtime_sec"] = time.perf_counter() - norm_started
    return json.dumps(payload, ensure_ascii=False), metrics


def _apply_boundary_structural_guards(
    *,
    raw_prediction: str,
    immutable_anchor_events: list[dict[str, Any]],
    alert_state: dict[str, Any],
    alert_min: int | None,
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    next_actionable_min = _resolve_next_actionable_min(alert_state, alert_min)
    out: dict[str, Any] = {
        "prediction": raw_prediction,
        "suffix_only_output_ok": 0,
        "prefix_edit_attempt_detected": 0,
        "prefix_edit_ignored_count": 0,
        "suffix_first_event_time": "",
        "suffix_first_event_time_before": "",
        "suffix_first_event_time_after": "",
        "next_actionable_time": _to_hhmm(next_actionable_min)
        if next_actionable_min is not None
        else "",
        "timeline_norm_applied": 0,
        "timeline_norm_total_shift_minutes": 0,
        "timeline_norm_overlap_fixes_count": 0,
        "timeline_norm_location_fix_applied": 0,
        "timeline_norm_runtime_sec": 0.0,
        "boundary_canonicalization_applied": 0,
        "boundary_pre_end_minus_alert_min": 0,
        "boundary_post_start_minus_alert_min": 0,
        "post_boundary_monotonic_fix_applied": 0,
        "post_boundary_monotonic_fix_count": 0,
        "post_boundary_total_shift_minutes": 0,
        "post_boundary_first_start_before": "",
        "post_boundary_first_start_after": "",
        "missing_boundary_event_detected": 0,
        "missing_boundary_event_autofixed": 0,
        "missing_boundary_event_autofix_minutes_pre": 0,
        "missing_boundary_event_autofix_minutes_post": 0,
        "immutable_scope_sanitize_applied": 0,
        "immutable_scope_terms_removed_count": 0,
        "immutable_scope_contains_disruption_terms_before": 0,
        "immutable_scope_contains_disruption_terms_after": 0,
    }
    if not immutable_anchor_events:
        return out
    if bool(run_cfg.get("planning_boundary_suffix_only_output", True)):
        suffix_fix = _apply_suffix_only_synthesis_json(
            raw_prediction,
            immutable_anchor_events,
            next_actionable_min,
            require_suffix_key=bool(
                run_cfg.get("planning_boundary_suffix_schema_strict", True)
            ),
        )
        if suffix_fix.get("prediction"):
            out.update(suffix_fix)
            deterministic_fixed, norm_metrics = _deterministic_suffix_state_fix_json(
                str(out.get("prediction", "")),
                immutable_anchor_events=immutable_anchor_events,
                next_actionable_min=next_actionable_min,
                next_actionable_location=str(
                    alert_state.get("next_actionable_location", "")
                )
                if isinstance(alert_state, dict)
                else "",
                next_actionable_actor=str(
                    alert_state.get("next_actionable_actor", "")
                )
                if isinstance(alert_state, dict)
                else "",
                run_cfg=run_cfg,
            )
            if deterministic_fixed:
                out["prediction"] = deterministic_fixed
            out.update(norm_metrics)
            if bool(run_cfg.get("planning_boundary_canonicalize_split", True)):
                canon_fixed, canon_metrics = _canonicalize_boundary_split_markers_json(
                    str(out.get("prediction", "")),
                    alert_min,
                )
                out.update(canon_metrics)
                if canon_fixed:
                    out["prediction"] = canon_fixed
            boundary_fixed, boundary_metrics = _ensure_boundary_event_json(
                str(out.get("prediction", "")),
                alert_min=alert_min,
                next_actionable_min=next_actionable_min,
                next_actionable_location=str(
                    alert_state.get("next_actionable_location", "")
                )
                if isinstance(alert_state, dict)
                else "",
                next_actionable_actor=str(
                    alert_state.get("next_actionable_actor", "")
                )
                if isinstance(alert_state, dict)
                else "",
                run_cfg=run_cfg,
            )
            out.update(boundary_metrics)
            if boundary_fixed:
                out["prediction"] = boundary_fixed
            post_fixed, post_metrics = _post_boundary_monotonic_clamp_json(
                str(out.get("prediction", "")),
                alert_min=alert_min,
                next_actionable_min=next_actionable_min,
            )
            out.update(post_metrics)
            if post_fixed:
                out["prediction"] = post_fixed
            imm_fixed, imm_metrics = _sanitize_immutable_scope_json(
                str(out.get("prediction", "")),
                alert_min,
            )
            out.update(imm_metrics)
            # Immutable scope is read-only; keep metrics for diagnostics only.
            _ = imm_fixed
            if norm_metrics.get("suffix_first_event_time_after"):
                out["suffix_first_event_time"] = str(
                    norm_metrics.get("suffix_first_event_time_after", "")
                )
            out["next_actionable_time"] = (
                _to_hhmm(next_actionable_min) if next_actionable_min is not None else ""
            )
            return out
    fixed = _apply_immutable_anchor_synthesis_json(
        raw_prediction, immutable_anchor_events, alert_min
    )
    if fixed and bool(run_cfg.get("planning_boundary_canonicalize_split", True)):
        canon_fixed, canon_metrics = _canonicalize_boundary_split_markers_json(
            fixed, alert_min
        )
        out.update(canon_metrics)
        if canon_fixed:
            fixed = canon_fixed
    if fixed:
        out["prediction"] = fixed
    return out


def _infer_state_consistency_failure_reason(
    *,
    events: list[dict[str, Any]],
    next_actionable_min: int | None,
    next_actionable_location: str,
    state_check_applicable: int,
    state_at_alert_consistent: int,
) -> str:
    if not state_check_applicable or state_at_alert_consistent:
        return ""
    if not events:
        return "MISSING_EVENTS"

    non_marker = [ev for ev in events if not _is_split_marker_event(ev)]
    if not non_marker:
        return "MISSING_SUFFIX"

    # Parse/format issues are still a frequent root cause in state failures.
    for ev in non_marker:
        s = _to_minutes(str(ev.get("start", "")))
        e = _to_minutes(str(ev.get("end", "")))
        if s is None or e is None or e < s:
            return "TIME_PARSE_FAIL"

    if not _events_time_monotonic_like_scorer(non_marker):
        return "TIME_NON_MONOTONIC"

    suffix = non_marker
    if next_actionable_min is not None:
        suffix = []
        for ev in non_marker:
            start_m = _to_minutes(str(ev.get("start", "")))
            if start_m is None:
                continue
            if start_m >= next_actionable_min:
                suffix.append(ev)
        if not suffix:
            return "MISSING_BOUNDARY_EVENT"

    if next_actionable_min is not None:
        early = []
        for ev in suffix:
            start_m = _to_minutes(str(ev.get("start", "")))
            if start_m is not None and start_m < next_actionable_min:
                early.append(ev)
        if early:
            return "TIME_EARLY"

    if next_actionable_location.strip():
        expected = _normalize_token_set(next_actionable_location)
        if expected:
            if not suffix:
                return "MISSING_SUFFIX"
            first = sorted(
                suffix,
                key=lambda ev: (
                    _to_minutes(str(ev.get("start", ""))) or 10**9,
                    _to_minutes(str(ev.get("end", ""))) or 10**9,
                ),
            )[0]
            observed = _normalize_token_set(str(first.get("location", "")))
            if expected and observed and not expected.intersection(observed):
                return "LOCATION_MISMATCH"
            if expected and not observed:
                return "IN_TRANSIT_MISMATCH"
    return "UNKNOWN"


def _route_endpoints(
    metadata: dict[str, Any], events: list[dict[str, Any]], alert_min: int
) -> tuple[str, str, bool]:
    route = str(metadata.get("boundary_route", "")).strip()
    for sep in ("->", "-", " to ", "/"):
        if sep in route:
            parts = [p.strip() for p in route.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[-1], True
    before_loc = ""
    after_loc = ""
    for event in events:
        end_min = _to_minutes(str(event.get("end", "")))
        if end_min is not None and end_min <= alert_min:
            before_loc = str(event.get("location", "")).strip() or before_loc
        start_min = _to_minutes(str(event.get("start", "")))
        if start_min is not None and start_min >= alert_min and not after_loc:
            after_loc = str(event.get("location", "")).strip()
    inferred = bool(before_loc or after_loc)
    return before_loc or "A", after_loc or "B", inferred


def _resolve_boundary_depart_plan(metadata: dict[str, Any], alert_min: int) -> tuple[int, int]:
    depart_hint = _to_minutes(str(metadata.get("boundary_departure_hint", "")))
    end_hint = _to_minutes(str(metadata.get("boundary_planned_end_hint", "")))
    depart = depart_hint if depart_hint is not None else max(0, alert_min - 10)
    planned_end = end_hint if end_hint is not None else alert_min + 20
    if not (depart < alert_min < planned_end):
        depart = max(0, alert_min - 10)
        planned_end = alert_min + 20
    return depart, planned_end


def _boundary_alignment_deltas(
    events: list[dict[str, Any]], alert_min: int | None
) -> tuple[int, int]:
    if alert_min is None:
        return 0, 0
    pre_end: int | None = None
    post_start: int | None = None
    for event in events:
        blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
        if "boundary_split_pre" in blob:
            end_m = _to_minutes(str(event.get("end", "")))
            if end_m is not None and (pre_end is None or end_m > pre_end):
                pre_end = end_m
        if "boundary_split_post" in blob:
            start_m = _to_minutes(str(event.get("start", "")))
            if start_m is not None and (post_start is None or start_m < post_start):
                post_start = start_m
    pre_delta = (pre_end - alert_min) if pre_end is not None else 0
    post_delta = (post_start - alert_min) if post_start is not None else 0
    return pre_delta, post_delta


def _canonicalize_boundary_split_markers_json(
    raw_prediction: str,
    alert_min: int | None,
) -> tuple[str | None, dict[str, Any]]:
    metrics = {
        "boundary_canonicalization_applied": 0,
        "boundary_pre_end_minus_alert_min": 0,
        "boundary_post_start_minus_alert_min": 0,
    }
    if alert_min is None:
        return None, metrics
    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return None, metrics
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {
            "plan_summary": str(parsed.get("plan_summary", "")),
            "events": parsed.get("schedule", []),
        }
    events = parsed.get("events", [])
    if not isinstance(events, list) or not events:
        return None, metrics

    norm_events: list[dict[str, Any]] = []
    pre_indices: list[int] = []
    post_indices: list[int] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        norm = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        idx = len(norm_events)
        marker_blob = f"{norm.get('what', '')} {norm.get('notes', '')}".lower()
        if "boundary_split_pre" in marker_blob:
            pre_indices.append(idx)
        if "boundary_split_post" in marker_blob:
            post_indices.append(idx)
        norm_events.append(norm)

    changed = False
    for idx in pre_indices:
        start_m = _to_minutes(str(norm_events[idx].get("start", "")))
        end_m = _to_minutes(str(norm_events[idx].get("end", "")))
        if end_m is None or end_m != alert_min:
            norm_events[idx]["end"] = _to_hhmm(alert_min)
            changed = True
        if start_m is None or start_m >= alert_min:
            norm_events[idx]["start"] = _to_hhmm(max(0, alert_min - 1))
            changed = True

    for idx in post_indices:
        start_m = _to_minutes(str(norm_events[idx].get("start", "")))
        end_m = _to_minutes(str(norm_events[idx].get("end", "")))
        if start_m is None or start_m != alert_min:
            norm_events[idx]["start"] = _to_hhmm(alert_min)
            changed = True
        if end_m is None or end_m <= alert_min:
            norm_events[idx]["end"] = _to_hhmm(alert_min + 1)
            changed = True

    merged = _dedup_sort_events(norm_events)
    pre_delta, post_delta = _boundary_alignment_deltas(merged, alert_min)
    metrics["boundary_pre_end_minus_alert_min"] = int(pre_delta)
    metrics["boundary_post_start_minus_alert_min"] = int(post_delta)
    metrics["boundary_canonicalization_applied"] = int(changed)
    if not changed:
        return None, metrics
    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": merged,
    }
    return json.dumps(payload, ensure_ascii=False), metrics


def _sanitize_immutable_scope_json(
    raw_prediction: str,
    alert_min: int | None,
) -> tuple[str | None, dict[str, Any]]:
    metrics = {
        "immutable_scope_sanitize_applied": 0,
        "immutable_scope_terms_removed_count": 0,
        "immutable_scope_contains_disruption_terms_before": 0,
        "immutable_scope_contains_disruption_terms_after": 0,
    }
    if alert_min is None:
        return None, metrics

    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return None, metrics
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {
            "plan_summary": str(parsed.get("plan_summary", "")),
            "events": parsed.get("schedule", []),
        }
    events = parsed.get("events", [])
    if not isinstance(events, list) or not events:
        return None, metrics

    changed = False
    removed_terms = 0
    out_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        norm = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        blob_before = f"{norm.get('what', '')} {norm.get('notes', '')} {norm.get('location', '')}"
        if _is_prefix_like_event(norm, alert_min) and _contains_disruption_terms(blob_before):
            metrics["immutable_scope_contains_disruption_terms_before"] = 1
            prev_what = norm["what"]
            prev_notes = norm["notes"]
            prev_loc = norm["location"]
            norm["what"] = _strip_disruption_terms(norm["what"])
            norm["notes"] = _strip_disruption_terms(norm["notes"])
            norm["location"] = _strip_disruption_terms(norm["location"])
            if (
                norm["what"] != prev_what
                or norm["notes"] != prev_notes
                or norm["location"] != prev_loc
            ):
                changed = True
                removed_terms += 1
        blob_after = f"{norm.get('what', '')} {norm.get('notes', '')} {norm.get('location', '')}"
        if _is_prefix_like_event(norm, alert_min) and _contains_disruption_terms(blob_after):
            metrics["immutable_scope_contains_disruption_terms_after"] = 1
        out_events.append(norm)

    if not changed:
        return None, metrics

    metrics["immutable_scope_sanitize_applied"] = 1
    metrics["immutable_scope_terms_removed_count"] = int(removed_terms)
    payload = {"plan_summary": str(parsed.get("plan_summary", "")), "events": _dedup_sort_events(out_events)}
    return json.dumps(payload, ensure_ascii=False), metrics


def _ensure_boundary_event_json(
    raw_prediction: str,
    *,
    alert_min: int | None,
    next_actionable_min: int | None,
    next_actionable_location: str,
    next_actionable_actor: str,
    run_cfg: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    metrics = {
        "missing_boundary_event_detected": 0,
        "missing_boundary_event_autofixed": 0,
        "missing_boundary_event_autofix_minutes_pre": 0,
        "missing_boundary_event_autofix_minutes_post": 0,
    }
    if alert_min is None:
        return None, metrics

    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return None, metrics
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {
            "plan_summary": str(parsed.get("plan_summary", "")),
            "events": parsed.get("schedule", []),
        }
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return None, metrics

    norm_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        norm_events.append(
            {
                "start": str(event.get("start", "")),
                "end": str(event.get("end", "")),
                "who": str(event.get("who", "")),
                "what": str(event.get("what", "")),
                "location": str(event.get("location", "")),
                "notes": str(event.get("notes", "")),
            }
        )

    split_pre, split_post = _split_marker_state(norm_events)
    has_non_marker_post = False
    start_threshold = (
        next_actionable_min if next_actionable_min is not None else alert_min
    )
    for event in norm_events:
        if _is_split_marker_event(event):
            continue
        start_m = _to_minutes(str(event.get("start", "")))
        if start_m is not None and start_m >= start_threshold:
            has_non_marker_post = True
            break

    if split_pre == 1 and split_post == 1 and has_non_marker_post:
        return None, metrics

    metrics["missing_boundary_event_detected"] = 1
    changed = False
    if split_pre == 0 or split_post == 0:
        pre_minutes = max(5, int(run_cfg.get("planning_boundary_autofix_pre_minutes", 10)))
        post_minutes = max(5, int(run_cfg.get("planning_boundary_autofix_post_minutes", 10)))
        route_loc = next_actionable_location.strip() or "B"
        actor = next_actionable_actor.strip() or "Team"
        pre_event = {
            "start": _to_hhmm(max(0, alert_min - pre_minutes)),
            "end": _to_hhmm(alert_min),
            "who": actor,
            "what": "Travel (pre-alert) auto boundary",
            "location": route_loc,
            "notes": "boundary_split_pre boundary_crossing=true source=boundary_autofix",
        }
        post_event = {
            "start": _to_hhmm(alert_min),
            "end": _to_hhmm(alert_min + post_minutes),
            "who": actor,
            "what": "Travel (post-alert compensated) auto boundary",
            "location": route_loc,
            "notes": "boundary_split_post partial_compensation=1 boundary_crossing=true source=boundary_autofix",
        }
        norm_events.extend([pre_event, post_event])
        metrics["missing_boundary_event_autofix_minutes_pre"] = int(pre_minutes)
        metrics["missing_boundary_event_autofix_minutes_post"] = int(post_minutes)
        changed = True

    if not has_non_marker_post:
        post_start = start_threshold
        post_dur = max(5, int(run_cfg.get("planning_boundary_autofix_post_event_minutes", 10)))
        handoff = {
            "start": _to_hhmm(post_start),
            "end": _to_hhmm(post_start + post_dur),
            "who": next_actionable_actor.strip() or "Team",
            "what": "Boundary handoff after disruption",
            "location": next_actionable_location.strip() or "B",
            "notes": "boundary_handoff_autofix=1",
        }
        norm_events.append(handoff)
        changed = True

    if not changed:
        return None, metrics
    metrics["missing_boundary_event_autofixed"] = 1
    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": _dedup_sort_events(norm_events),
    }
    return json.dumps(payload, ensure_ascii=False), metrics


def _post_boundary_monotonic_clamp_json(
    raw_prediction: str,
    *,
    alert_min: int | None,
    next_actionable_min: int | None,
) -> tuple[str | None, dict[str, Any]]:
    metrics = {
        "post_boundary_monotonic_fix_applied": 0,
        "post_boundary_monotonic_fix_count": 0,
        "post_boundary_total_shift_minutes": 0,
        "post_boundary_first_start_before": "",
        "post_boundary_first_start_after": "",
    }
    if alert_min is None:
        return None, metrics

    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return None, metrics
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {
            "plan_summary": str(parsed.get("plan_summary", "")),
            "events": parsed.get("schedule", []),
        }
    events = parsed.get("events", [])
    if not isinstance(events, list) or not events:
        return None, metrics

    norm_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        norm_events.append(
            {
                "start": str(event.get("start", "")),
                "end": str(event.get("end", "")),
                "who": str(event.get("who", "")),
                "what": str(event.get("what", "")),
                "location": str(event.get("location", "")),
                "notes": str(event.get("notes", "")),
            }
        )
    if not norm_events:
        return None, metrics

    post_indices: list[int] = []
    adjustable_indices: list[int] = []
    for idx, event in enumerate(norm_events):
        marker = _is_split_marker_event(event)
        start_m = _to_minutes(str(event.get("start", "")))
        if marker:
            blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
            if "boundary_split_post" in blob:
                post_indices.append(idx)
            continue
        if start_m is not None and start_m >= alert_min:
            post_indices.append(idx)
            adjustable_indices.append(idx)

    if not post_indices or not adjustable_indices:
        return None, metrics

    def _first_start(indices: list[int]) -> int | None:
        vals = []
        for i in indices:
            s = _to_minutes(str(norm_events[i].get("start", "")))
            if s is not None:
                vals.append(s)
        return min(vals) if vals else None

    first_before = _first_start(adjustable_indices)
    if first_before is not None:
        metrics["post_boundary_first_start_before"] = _to_hhmm(first_before)

    actor_prev_end: dict[str, int] = {}
    for idx, event in enumerate(norm_events):
        if idx in adjustable_indices:
            continue
        e = _to_minutes(str(event.get("end", "")))
        if e is None:
            continue
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            actor_prev_end[actor] = max(actor_prev_end.get(actor, 0), e)

    adjustable_indices = sorted(
        adjustable_indices,
        key=lambda i: (
            _to_minutes(str(norm_events[i].get("start", ""))) or 10**9,
            _to_minutes(str(norm_events[i].get("end", ""))) or 10**9,
            i,
        ),
    )

    required_floor = next_actionable_min if next_actionable_min is not None else alert_min
    total_shift = 0
    fix_count = 0
    changed = False
    for pos, idx in enumerate(adjustable_indices):
        event = norm_events[idx]
        s = _to_minutes(str(event.get("start", "")))
        e = _to_minutes(str(event.get("end", "")))
        if s is None or e is None:
            continue
        required_start = max(s, required_floor)
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            prev_end = actor_prev_end.get(actor)
            if prev_end is not None and prev_end > required_start:
                required_start = prev_end
        if required_start > s:
            delta = required_start - s
            for shifted_idx in adjustable_indices[pos:]:
                ss = _to_minutes(str(norm_events[shifted_idx].get("start", "")))
                ee = _to_minutes(str(norm_events[shifted_idx].get("end", "")))
                if ss is None or ee is None:
                    continue
                norm_events[shifted_idx]["start"] = _to_hhmm(ss + delta)
                norm_events[shifted_idx]["end"] = _to_hhmm(ee + delta)
            total_shift += delta
            fix_count += 1
            changed = True
            s += delta
            e += delta
        for actor in _split_actors_for_monotonic(str(event.get("who", ""))):
            actor_prev_end[actor] = max(actor_prev_end.get(actor, 0), e)

    if not changed:
        return None, metrics

    first_after = _first_start(adjustable_indices)
    if first_after is not None:
        metrics["post_boundary_first_start_after"] = _to_hhmm(first_after)
    metrics["post_boundary_monotonic_fix_applied"] = 1
    metrics["post_boundary_monotonic_fix_count"] = int(fix_count)
    metrics["post_boundary_total_shift_minutes"] = int(total_shift)
    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": _dedup_sort_events(norm_events),
    }
    return json.dumps(payload, ensure_ascii=False), metrics


def _has_boundary_crossing_segment(
    events: list[dict[str, Any]], alert_min: int | None
) -> bool:
    if alert_min is None:
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None:
            continue
        if start_m < alert_min < end_m:
            return True
    return False


def _post_boundary_monotonic_ok(
    events: list[dict[str, Any]],
    alert_min: int | None,
    next_actionable_min: int | None,
) -> bool:
    if alert_min is None:
        return True
    floor = next_actionable_min if next_actionable_min is not None else alert_min
    post_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start_m = _to_minutes(str(event.get("start", "")))
        if _is_split_marker_event(event):
            blob = f"{event.get('what', '')} {event.get('notes', '')}".lower()
            if "boundary_split_post" in blob:
                post_events.append(event)
            continue
        if start_m is not None and start_m >= floor:
            post_events.append(event)
    if not post_events:
        return True
    return _events_time_monotonic_like_scorer(_dedup_sort_events(post_events))


def _immutable_scope_has_disruption_terms(
    events: list[dict[str, Any]], alert_min: int | None
) -> bool:
    if alert_min is None:
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        if not _is_prefix_like_event(event, alert_min):
            continue
        blob = " ".join(
            [
                str(event.get("what", "")),
                str(event.get("notes", "")),
                str(event.get("location", "")),
            ]
        )
        if _contains_disruption_terms(blob):
            return True
    return False


def _immutable_scope_term_locations(
    events: list[dict[str, Any]], alert_min: int | None
) -> list[str]:
    if alert_min is None:
        return []
    locations: list[str] = []
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if not _is_prefix_like_event(event, alert_min):
            continue
        for field in ("what", "notes", "location"):
            value = str(event.get(field, ""))
            if _contains_disruption_terms(value):
                locations.append(f"{idx}:{field}")
    return locations


STRICT_VIOLATION_TO_GATE_REASON = {
    "Boundary crossing split not applied": ["SPLIT_MISSING", "MARKER_LOST"],
    "Required disruption-boundary crossing segment is missing": ["BOUNDARY_MISSING"],
    "Partial journey compensation is missing at disruption boundary": ["COMP_MISSING"],
    "State timeline is inconsistent at disruption boundary": ["STATE_MISMATCH", "NON_MONOTONIC"],
    "Immutable past appears modified by disruption terms": ["IMMUTABLE_TERMS"],
    "PHOTO_TIME_EXCEEDED": ["PHOTO_TIME_EXCEEDED"],
}


def _gate_reasons_from_strict_violations(violations: list[str]) -> list[str]:
    reasons: list[str] = []
    for violation in violations:
        mapped = STRICT_VIOLATION_TO_GATE_REASON.get(str(violation), [])
        reasons.extend(mapped)
    # Fallback mapping for close variants of strict message strings.
    text = " | ".join(str(v) for v in violations).lower()
    if "boundary crossing split not applied" in text:
        reasons.extend(["SPLIT_MISSING", "MARKER_LOST"])
    if "required disruption-boundary crossing segment is missing" in text:
        reasons.append("BOUNDARY_MISSING")
    if "partial journey compensation is missing" in text:
        reasons.append("COMP_MISSING")
    if "state timeline is inconsistent at disruption boundary" in text:
        reasons.extend(["STATE_MISMATCH", "NON_MONOTONIC"])
    if "immutable past appears modified by disruption terms" in text:
        reasons.append("IMMUTABLE_TERMS")
    if "exceeds photo_time" in text:
        reasons.append("PHOTO_TIME_EXCEEDED")

    seen: set[str] = set()
    uniq: list[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        uniq.append(reason)
    return uniq


def _extract_primary_actor(prev_event: dict[str, Any], next_event: dict[str, Any]) -> str:
    who = str(next_event.get("who", "")).strip() or str(prev_event.get("who", "")).strip()
    return who if who else "Team"


def _shift_event_time(event: dict[str, Any], shift_min: int, threshold_min: int) -> dict[str, Any]:
    shifted = dict(event)
    start_min = _to_minutes(str(shifted.get("start", "")))
    end_min = _to_minutes(str(shifted.get("end", "")))
    if start_min is None or end_min is None:
        return shifted
    if start_min >= threshold_min:
        shifted["start"] = _to_hhmm(start_min + shift_min)
        shifted["end"] = _to_hhmm(end_min + shift_min)
    return shifted


def _apply_boundary_split_compensation_json(
    raw_prediction: str,
    metadata: dict[str, Any],
    alert_min: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction": None,
        "split_attempted": 0,
        "split_applied": 0,
        "split_apply_mode": "SKIPPED_NOT_APPLICABLE",
        "split_failure_reason": "SKIPPED_NOT_APPLICABLE",
        "split_candidate_found": 0,
        "split_candidate_count": 0,
        "crossing_candidate_summary": {},
        "split_marker_survived": 0,
    }
    require_boundary = bool(metadata.get("require_boundary_crossing", False))
    if not require_boundary:
        return result
    result["split_attempted"] = 1
    if alert_min is None:
        result["split_apply_mode"] = "FAILED_NO_CANDIDATE"
        result["split_failure_reason"] = "FAILED_NO_CANDIDATE"
        return result

    parsed, err = parse_json_response(raw_prediction)
    if err or parsed is None or not isinstance(parsed, dict):
        result["split_apply_mode"] = "FAILED_PARSE"
        result["split_failure_reason"] = "FAILED_PARSE"
        return result
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}

    events = _extract_events_from_prediction(raw_prediction)
    if len(events) < 2:
        result["split_apply_mode"] = "FAILED_NO_CANDIDATE"
        result["split_failure_reason"] = "FAILED_NO_CANDIDATE"
        return result
    events = _dedup_sort_events(events)

    strategy = "observed_gap"
    synthetic_apply_mode = "SYNTHETIC_INSERTED_NO_TRAVEL_FOUND"
    route_tokens = _normalize_token_set(str(metadata.get("boundary_route", "")))
    explicit_crossing_idx = -1
    for i, event in enumerate(events):
        start_min = _to_minutes(str(event.get("start", "")))
        end_min = _to_minutes(str(event.get("end", "")))
        if start_min is None or end_min is None or end_min <= start_min:
            continue
        if not (start_min < alert_min < end_min):
            continue
        what = str(event.get("what", ""))
        notes = str(event.get("notes", ""))
        loc = str(event.get("location", ""))
        blob = f"{what} {notes} {loc}".lower()
        blob_tokens = _normalize_token_set(blob)
        has_marker = "boundary_crossing=true" in blob
        has_route = bool(route_tokens and route_tokens.intersection(blob_tokens))
        is_travel = "travel" in what.lower()
        if has_marker or has_route or is_travel:
            explicit_crossing_idx = i
            break

    crossing_idx = -1
    if explicit_crossing_idx < 0:
        for i in range(len(events) - 1):
            prev_event = events[i]
            next_event = events[i + 1]
            seg_start = _to_minutes(str(prev_event.get("end", "")))
            seg_end = _to_minutes(str(next_event.get("start", "")))
            if seg_start is None or seg_end is None or seg_end <= seg_start:
                continue
            if not (seg_start < alert_min < seg_end):
                continue
            if not _route_matches(str(prev_event.get("location", "")), str(next_event.get("location", "")), metadata):
                continue
            crossing_idx = i
            break

    synthetic_mode = (explicit_crossing_idx < 0) and (crossing_idx < 0)
    result["split_candidate_found"] = int((explicit_crossing_idx >= 0) or (crossing_idx >= 0))
    result["split_candidate_count"] = int((explicit_crossing_idx >= 0) or (crossing_idx >= 0))
    if synthetic_mode:
        strategy = "synthetic"
        seg_start, seg_end = _resolve_boundary_depart_plan(metadata, alert_min)
        if not (seg_start < alert_min < seg_end):
            result["split_apply_mode"] = "FAILED_NO_CANDIDATE"
            result["split_failure_reason"] = "FAILED_NO_CANDIDATE"
            return result
        from_loc, to_loc, constructed_from_state = _route_endpoints(metadata, events, alert_min)
        synthetic_apply_mode = (
            "SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE"
            if constructed_from_state
            else "SYNTHETIC_INSERTED_NO_TRAVEL_FOUND"
        )
        actor = "Team"
        for event in events:
            end_m = _to_minutes(str(event.get("end", "")))
            start_m = _to_minutes(str(event.get("start", "")))
            if end_m is not None and end_m <= alert_min and str(event.get("who", "")).strip():
                actor = str(event.get("who", "")).strip()
            if start_m is not None and start_m >= alert_min and str(event.get("who", "")).strip():
                actor = str(event.get("who", "")).strip()
                break
        observed = seg_end - seg_start
        baseline = _lookup_travel_baseline(from_loc, to_loc, metadata)
        if baseline is None or baseline <= 0:
            baseline = observed
        delay = _resolve_disruption_delay(metadata)
        elapsed = max(0, alert_min - seg_start)
        remaining_baseline = max(1, baseline - elapsed)
        post_duration = remaining_baseline + max(1, delay)
        new_segment_duration = elapsed + post_duration
        shift_min = max(0, new_segment_duration - observed)
        new_post_end = seg_end + shift_min
        route_desc = f"{from_loc}->{to_loc}"
        pre_split = {
            "start": _to_hhmm(seg_start),
            "end": _to_hhmm(alert_min),
            "who": actor,
            "what": f"Travel (pre-alert) {route_desc}",
            "location": from_loc,
            "notes": (
                f"boundary_split_pre route={route_desc} source=synthetic "
                f"split_mode={synthetic_apply_mode.lower()} boundary_crossing=true"
            ),
        }
        post_split = {
            "start": _to_hhmm(alert_min),
            "end": _to_hhmm(new_post_end),
            "who": actor,
            "what": f"Travel (post-alert compensated) {route_desc}",
            "location": to_loc,
            "notes": (
                "boundary_split_post partial_compensation=1 synthetic=1 boundary_crossing=true "
                f"delay_plus={delay} baseline={baseline} observed={observed}"
            ),
        }
        shifted_events: list[dict[str, Any]] = []
        for event in events:
            shifted_events.append(_shift_event_time(event, shift_min, seg_end))
        merged = shifted_events + [pre_split, post_split]
        result["crossing_candidate_summary"] = {
            "strategy": strategy,
            "route": route_desc,
            "seg_start": _to_hhmm(seg_start),
            "seg_end": _to_hhmm(seg_end),
            "shift_min": shift_min,
            "events_before": len(events),
        }
    else:
        if explicit_crossing_idx >= 0:
            strategy = "real_event"
            crossing_event = events[explicit_crossing_idx]
            seg_start = _to_minutes(str(crossing_event.get("start", "")))
            seg_end = _to_minutes(str(crossing_event.get("end", "")))
            if seg_start is None or seg_end is None or seg_end <= seg_start:
                result["split_apply_mode"] = "FAILED_PARSE"
                result["split_failure_reason"] = "FAILED_PARSE"
                return result
            from_loc, to_loc, _ = _route_endpoints(metadata, events, alert_min)
            observed = seg_end - seg_start
            baseline = _lookup_travel_baseline(from_loc, to_loc, metadata)
            if baseline is None or baseline <= 0:
                baseline = observed
            delay = _resolve_disruption_delay(metadata)
            elapsed = max(0, alert_min - seg_start)
            remaining_baseline = max(1, baseline - elapsed)
            post_duration = remaining_baseline + max(1, delay)
            new_segment_duration = elapsed + post_duration
            shift_min = max(0, new_segment_duration - observed)
            new_cross_end = seg_end + shift_min
            actor = str(crossing_event.get("who", "")).strip() or "Team"
            route_desc = f"{from_loc}->{to_loc}"
            pre_split = {
                "start": _to_hhmm(seg_start),
                "end": _to_hhmm(alert_min),
                "who": actor,
                "what": f"Travel (pre-alert) {route_desc}",
                "location": from_loc,
                "notes": (
                    f"boundary_split_pre route={route_desc} "
                    "boundary_crossing=true source=real_event"
                ),
            }
            post_split = {
                "start": _to_hhmm(alert_min),
                "end": _to_hhmm(new_cross_end),
                "who": actor,
                "what": f"Travel (post-alert compensated) {route_desc}",
                "location": to_loc,
                "notes": (
                    "boundary_split_post partial_compensation=1 boundary_crossing=true source=real_event "
                    f"delay_plus={delay} baseline={baseline} observed={observed}"
                ),
            }
            shifted_events: list[dict[str, Any]] = []
            for idx, event in enumerate(events):
                if idx < explicit_crossing_idx:
                    shifted_events.append(dict(event))
                elif idx > explicit_crossing_idx:
                    shifted_events.append(_shift_event_time(event, shift_min, seg_end))
            merged = shifted_events[:explicit_crossing_idx] + [pre_split, post_split] + shifted_events[explicit_crossing_idx:]
            result["crossing_candidate_summary"] = {
                "strategy": strategy,
                "crossing_idx": explicit_crossing_idx,
                "route": route_desc,
                "seg_start": _to_hhmm(seg_start),
                "seg_end": _to_hhmm(seg_end),
                "shift_min": shift_min,
                "events_before": len(events),
            }
        else:
            prev_event = events[crossing_idx]
            next_event = events[crossing_idx + 1]
            seg_start = _to_minutes(str(prev_event.get("end", "")))
            seg_end = _to_minutes(str(next_event.get("start", "")))
            if seg_start is None or seg_end is None or seg_end <= seg_start:
                result["split_apply_mode"] = "FAILED_PARSE"
                result["split_failure_reason"] = "FAILED_PARSE"
                return result

            observed = seg_end - seg_start
            baseline = _lookup_travel_baseline(
                str(prev_event.get("location", "")),
                str(next_event.get("location", "")),
                metadata,
            )
            if baseline is None or baseline <= 0:
                baseline = observed

            delay = _resolve_disruption_delay(metadata)
            elapsed = max(0, alert_min - seg_start)
            remaining_baseline = max(1, baseline - elapsed)
            post_duration = remaining_baseline + max(1, delay)
            new_segment_duration = elapsed + post_duration
            shift_min = max(0, new_segment_duration - observed)
            new_next_start = seg_end + shift_min

            actor = _extract_primary_actor(prev_event, next_event)
            split_actor = f"{actor}_split"
            route_desc = f"{prev_event.get('location', '')}->{next_event.get('location', '')}"

            pre_split = {
                "start": _to_hhmm(seg_start),
                "end": _to_hhmm(alert_min),
                "who": split_actor,
                "what": f"Travel (pre-alert) {route_desc}",
                "location": str(prev_event.get("location", "")),
                "notes": (
                    f"boundary_split_pre route={route_desc} "
                    "boundary_crossing=true"
                ),
            }
            post_split = {
                "start": _to_hhmm(alert_min),
                "end": _to_hhmm(new_next_start),
                "who": split_actor,
                "what": f"Travel (post-alert compensated) {route_desc}",
                "location": str(next_event.get("location", "")),
                "notes": (
                    "boundary_split_post partial_compensation=1 boundary_crossing=true "
                    f"delay_plus={delay} baseline={baseline} observed={observed}"
                ),
            }

            shifted_events: list[dict[str, Any]] = []
            for idx, event in enumerate(events):
                if idx <= crossing_idx:
                    shifted_events.append(dict(event))
                else:
                    shifted_events.append(_shift_event_time(event, shift_min, seg_end))

            merged = shifted_events[: crossing_idx + 1] + [pre_split, post_split] + shifted_events[crossing_idx + 1 :]
            result["crossing_candidate_summary"] = {
                "strategy": strategy,
                "crossing_idx": crossing_idx,
                "route": route_desc,
                "seg_start": _to_hhmm(seg_start),
                "seg_end": _to_hhmm(seg_end),
                "shift_min": shift_min,
                "events_before": len(events),
            }

    merged = _dedup_sort_events(merged)
    if not merged:
        result["split_apply_mode"] = "FAILED_NO_CANDIDATE"
        result["split_failure_reason"] = "FAILED_NO_CANDIDATE"
        return result

    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": merged,
    }
    prediction = json.dumps(payload, ensure_ascii=False)
    pre_mark, post_mark = _split_marker_state(merged)
    applied = int(pre_mark == 1 and post_mark == 1)
    result["prediction"] = prediction
    result["split_applied"] = applied
    result["split_marker_survived"] = applied
    result["split_apply_mode"] = synthetic_apply_mode if synthetic_mode else "REAL_CROSSING_FOUND"
    result["split_failure_reason"] = result["split_apply_mode"]
    result["crossing_candidate_summary"]["events_after"] = len(merged)
    return result


def _evaluate_with_optional_prefix_lock(
    sample: BenchmarkSample,
    prediction: str,
    mode: str,
    locked_prefix_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if locked_prefix_events:
        metadata = copy.deepcopy(sample.metadata)
        metadata["locked_prefix_events"] = locked_prefix_events
        eval_sample = replace(sample, metadata=metadata)
        return evaluate_prediction(eval_sample, prediction, mode=mode)
    return evaluate_prediction(sample, prediction, mode=mode)


def _disruption_invariant_score(scored: dict[str, Any]) -> int:
    diag = scored.get("diagnostics", {}) if isinstance(scored, dict) else {}
    if not isinstance(diag, dict):
        return 0
    score = 0
    score += int(diag.get("disruption_applied", 0)) if int(diag.get("disruption_applicable", 0)) else 0
    score += int(diag.get("partial_compensation_ok", 0)) if int(diag.get("partial_compensation_applicable", 0)) else 0
    score += int(diag.get("crossing_split_applied", 0)) if int(diag.get("crossing_split_applicable", 0)) else 0
    score += int(diag.get("immutable_prefix_ok", 0)) if int(diag.get("immutable_check_applicable", 0)) else 0
    score += int(diag.get("immutable_prefix_after_split_ok", 0)) if int(diag.get("crossing_split_applicable", 0)) else 0
    score += int(diag.get("state_at_alert_consistent", 0)) if int(diag.get("state_check_applicable", 0)) else 0
    return score


def _prefer_scored(
    candidate: dict[str, Any],
    current: dict[str, Any],
    selection_policy: str = "default",
) -> bool:
    candidate_diag = candidate.get("diagnostics", {}) if isinstance(candidate, dict) else {}
    current_diag = current.get("diagnostics", {}) if isinstance(current, dict) else {}
    cand_non_empty = int(candidate_diag.get("non_empty_events", 0))
    cur_non_empty = int(current_diag.get("non_empty_events", 0))
    cand_valid_json = int(candidate_diag.get("valid_json", 0))
    cur_valid_json = int(current_diag.get("valid_json", 0))

    # P0 priority: non-empty events are more valuable than another empty schedule,
    # even when temporary constraint violations remain.
    if cur_non_empty == 0 and cand_non_empty == 1 and cand_valid_json == 1:
        return True
    if cur_valid_json == 0 and cand_valid_json == 1:
        return True

    if selection_policy == "invariant_first":
        cand_inv = _disruption_invariant_score(candidate)
        curr_inv = _disruption_invariant_score(current)
        if cand_inv != curr_inv:
            return cand_inv > curr_inv

    if int(candidate.get("success", 0)) > int(current.get("success", 0)):
        return True
    if int(candidate.get("success", 0)) < int(current.get("success", 0)):
        return False
    c_v = int(candidate.get("violation_count", 10**9))
    cur_v = int(current.get("violation_count", 10**9))
    if c_v != cur_v:
        return c_v < cur_v
    return len(candidate.get("violations", [])) < len(current.get("violations", []))


def _is_parse_or_format_failure(scored: dict[str, Any]) -> bool:
    violations = " | ".join(scored.get("violations", []))
    needles = [
        "JSONDecodeError",
        "No JSON object found",
        "Parsed JSON is not an object",
    ]
    return any(k in violations for k in needles)


def _is_empty_events_failure(scored: dict[str, Any]) -> bool:
    violations = " | ".join(scored.get("violations", []))
    return "Missing or empty events list" in violations


def _is_disruption_failure(scored: dict[str, Any]) -> bool:
    violations = " | ".join(scored.get("violations", []))
    needles = [
        "disruption",
        "Immutable past appears modified",
        "No events scheduled after disruption start",
        "State timeline is inconsistent at disruption boundary",
    ]
    return any(k in violations for k in needles)


def _has_missing_boundary_crossing_failure(scored: dict[str, Any]) -> bool:
    violations = " | ".join(scored.get("violations", []))
    needles = [
        "Required disruption-boundary crossing segment is missing",
        "Boundary crossing split not applied",
    ]
    return any(k in violations for k in needles)


def _has_constraint_failures(scored: dict[str, Any]) -> bool:
    if int(scored.get("violation_count", 0)) <= 0:
        return False
    if _is_parse_or_format_failure(scored) or _is_empty_events_failure(scored):
        return False
    return True


def _is_photo_time_failure(scored: dict[str, Any]) -> bool:
    violations = " | ".join(scored.get("violations", []))
    return "photo_time" in violations.lower()


def _is_tailor_close_failure(scored: dict[str, Any]) -> bool:
    violations = " | ".join(scored.get("violations", []))
    return "tailor task after close" in violations.lower()


def _weighted_commit_score(scored: dict[str, Any]) -> int:
    diagnostics = scored.get("diagnostics", {}) if isinstance(scored, dict) else {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    violations = [str(v).lower() for v in scored.get("violations", [])]
    score = 0

    valid_json = int(diagnostics.get("valid_json", 0))
    non_empty = int(diagnostics.get("non_empty_events", 0))
    if valid_json == 0:
        score += 1000
    if non_empty == 0:
        score += 1000

    for violation in violations:
        if "immutable past appears modified" in violation:
            score += 200
        elif "state timeline is inconsistent" in violation:
            score += 150
        elif "photo_time" in violation:
            score += 80
        else:
            score += 10

    if int(scored.get("success", 0)) == 1:
        score -= 50
    return score


def _prefer_commit_candidate(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    cand_score = _weighted_commit_score(candidate)
    curr_score = _weighted_commit_score(current)
    if cand_score != curr_score:
        return cand_score < curr_score
    if int(candidate.get("success", 0)) != int(current.get("success", 0)):
        return int(candidate.get("success", 0)) > int(current.get("success", 0))
    c_v = int(candidate.get("violation_count", 10**9))
    cur_v = int(current.get("violation_count", 10**9))
    if c_v != cur_v:
        return c_v < cur_v
    cand_inv = _disruption_invariant_score(candidate)
    curr_inv = _disruption_invariant_score(current)
    if cand_inv != curr_inv:
        return cand_inv > curr_inv
    return len(candidate.get("violations", [])) < len(current.get("violations", []))


def _deterministic_photo_time_suffix_repair_json(
    *,
    raw_prediction: str,
    constraints: dict[str, Any],
    alert_state: dict[str, Any] | None,
    run_cfg: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    metrics: dict[str, Any] = {
        "photo_time_violation_detected": 0,
        "photo_time_repair_triggered": 0,
        "photo_time_repair_success": 0,
        "photo_time_repair_method": "",
        "photo_time_repair_search_nodes": 0,
        "photo_time_repair_wall_ms": 0.0,
    }
    started = time.perf_counter()

    def _ret(prediction: str | None, *, method: str = "") -> tuple[str | None, dict[str, Any]]:
        metrics["photo_time_repair_method"] = method
        metrics["photo_time_repair_wall_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        return prediction, metrics

    if not bool(run_cfg.get("planning_photo_time_deterministic_repair", True)):
        return _ret(None, method="DISABLED")
    photo_limit = _to_minutes(str(constraints.get("photo_time", "")))
    if photo_limit is None:
        return _ret(None, method="NO_LIMIT")
    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return _ret(None, method="PARSE_FAIL")
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
    events = parsed.get("events", [])
    if not isinstance(events, list) or not events:
        return _ret(None, method="NO_EVENTS")

    metrics["photo_time_repair_triggered"] = 1

    next_actionable_min = _resolve_next_actionable_min(alert_state or {}, None)
    floor_min = max(0, int(next_actionable_min if next_actionable_min is not None else 0))

    norm_events: list[dict[str, Any]] = []
    mutable: list[dict[str, Any]] = []
    photo_violation = 0
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        norm = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        norm_events.append(norm)
        if _is_split_marker_event(norm):
            continue
        s = _to_minutes(norm["start"])
        e = _to_minutes(norm["end"])
        if s is None or e is None or e <= s:
            continue
        if e > photo_limit:
            photo_violation = 1
        if s < floor_min:
            continue
        mutable.append(
            {
                "idx": idx,
                "start": s,
                "end": e,
                "dur": max(1, e - s),
            }
        )

    metrics["photo_time_violation_detected"] = int(photo_violation)
    if photo_violation == 0:
        return _ret(None, method="NO_PHOTO_VIOLATION")
    if not mutable:
        return _ret(None, method="NO_MUTABLE_SUFFIX")

    def _build_schedule(order: list[dict[str, Any]]) -> tuple[bool, dict[int, tuple[int, int]], int, int]:
        # Pack post-boundary tasks in order, then compress durations (>=1m) to fit photo_time.
        starts: list[int] = []
        durs: list[int] = []
        cursor = floor_min
        for item in order:
            starts.append(cursor)
            durs.append(int(item["dur"]))
            cursor += int(item["dur"])
        overflow = max(0, cursor - photo_limit)
        if overflow > 0:
            reducible = sum(max(0, d - 1) for d in durs)
            if reducible < overflow:
                return False, {}, overflow, 0
            rem = overflow
            # shrink from tail first to preserve early-event anchoring
            for i in range(len(durs) - 1, -1, -1):
                if rem <= 0:
                    break
                can = max(0, durs[i] - 1)
                if can <= 0:
                    continue
                take = min(can, rem)
                durs[i] -= take
                rem -= take
        schedule: dict[int, tuple[int, int]] = {}
        shift_cost = 0
        cursor = floor_min
        for item, dur in zip(order, durs):
            ns = cursor
            ne = ns + max(1, dur)
            schedule[int(item["idx"])] = (ns, ne)
            shift_cost += abs(ns - int(item["start"]))
            cursor = ne
        feasible = cursor <= photo_limit
        return feasible, schedule, max(0, cursor - photo_limit), shift_cost

    # Candidate orders: original, earliest-end, shortest-duration, (optionally) permutations.
    mutable_sorted = sorted(mutable, key=lambda x: (x["start"], x["end"], x["idx"]))
    candidate_orders: list[tuple[str, list[dict[str, Any]]]] = [
        ("SHIFT", mutable_sorted),
        ("REORDER_END", sorted(mutable_sorted, key=lambda x: (x["end"], x["start"], x["idx"]))),
        ("REORDER_DUR", sorted(mutable_sorted, key=lambda x: (x["dur"], x["end"], x["idx"]))),
    ]
    if bool(run_cfg.get("planning_photo_time_permutation_search", True)) and len(mutable_sorted) <= 6:
        perm_limit = max(1, int(run_cfg.get("planning_photo_time_permutation_limit", 720)))
        count = 0
        for perm in itertools.permutations(mutable_sorted, len(mutable_sorted)):
            candidate_orders.append(("REORDER_PERM", list(perm)))
            count += 1
            if count >= perm_limit:
                break

    best_method = ""
    best_schedule: dict[int, tuple[int, int]] = {}
    best_score: tuple[int, int] | None = None
    search_nodes = 0
    for method, order in candidate_orders:
        search_nodes += 1
        feasible, schedule, overflow, shift_cost = _build_schedule(order)
        score = (0 if feasible else 1, overflow * 1000 + shift_cost)
        if best_score is None or score < best_score:
            best_score = score
            best_schedule = schedule
            best_method = method
        if feasible and method == "SHIFT":
            break

    metrics["photo_time_repair_search_nodes"] = int(search_nodes)
    if not best_schedule:
        return _ret(None, method="UNSAT")

    changed = 0
    for idx, (ns, ne) in best_schedule.items():
        if idx < 0 or idx >= len(norm_events):
            continue
        event = norm_events[idx]
        os = _to_minutes(str(event.get("start", "")))
        oe = _to_minutes(str(event.get("end", "")))
        if os is None or oe is None:
            continue
        if os != ns or oe != ne:
            changed += 1
        event["start"] = _to_hhmm(ns)
        event["end"] = _to_hhmm(ne)
        event["notes"] = (str(event.get("notes", "")) + " [photo_time_fix=deterministic]").strip()

    if changed == 0:
        return _ret(None, method="NO_CHANGE")
    merged = _dedup_sort_events(norm_events)
    payload = {"plan_summary": str(parsed.get("plan_summary", "")), "events": merged}
    metrics["photo_time_repair_success"] = 1
    return _ret(json.dumps(payload, ensure_ascii=False), method=best_method)


def _deterministic_tailor_hours_suffix_repair_json(
    *,
    raw_prediction: str,
    constraints: dict[str, Any],
    alert_state: dict[str, Any] | None,
    run_cfg: dict[str, Any],
) -> str | None:
    if not bool(run_cfg.get("planning_tailor_hours_deterministic_repair", True)):
        return None
    tailor_close = _to_minutes(str(constraints.get("tailor_closes", "")))
    if tailor_close is None:
        return None
    parsed, err = parse_json_response(raw_prediction)
    if err or not isinstance(parsed, dict):
        return None
    if "events" not in parsed and isinstance(parsed.get("schedule"), list):
        parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
    events = parsed.get("events", [])
    if not isinstance(events, list) or not events:
        return None

    next_actionable_min = _resolve_next_actionable_min(alert_state or {}, None)
    changed = 0
    out_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        norm = {
            "start": str(event.get("start", "")),
            "end": str(event.get("end", "")),
            "who": str(event.get("who", "")),
            "what": str(event.get("what", "")),
            "location": str(event.get("location", "")),
            "notes": str(event.get("notes", "")),
        }
        if _is_split_marker_event(norm):
            out_events.append(norm)
            continue
        start_m = _to_minutes(norm["start"])
        end_m = _to_minutes(norm["end"])
        if start_m is None or end_m is None:
            out_events.append(norm)
            continue
        if next_actionable_min is not None and start_m < next_actionable_min:
            out_events.append(norm)
            continue
        what_blob = str(norm.get("what", "")).lower()
        if "tailor" not in what_blob or end_m <= tailor_close:
            out_events.append(norm)
            continue

        duration = max(1, end_m - start_m)
        new_end = tailor_close
        new_start = new_end - duration
        if next_actionable_min is not None and new_start < next_actionable_min:
            new_start = next_actionable_min
            new_end = new_start + duration
        if new_start >= new_end:
            new_start = max(0, new_end - 1)
        norm["start"] = _to_hhmm(new_start)
        norm["end"] = _to_hhmm(new_end)
        norm["notes"] = (norm["notes"] + " [tailor_hours_fix=deterministic]").strip()
        changed += 1
        out_events.append(norm)

    if changed == 0:
        return None
    payload = {
        "plan_summary": str(parsed.get("plan_summary", "")),
        "events": _dedup_sort_events(out_events),
    }
    return json.dumps(payload, ensure_ascii=False)


def _backend_chat_completion(
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str | None:
    base_url = str(model_cfg.get("base_url", "http://localhost:8000/v1")).rstrip("/")
    if base_url.endswith("/chat/completions"):
        url = base_url
    else:
        url = f"{base_url}/chat/completions"
    api_key = os.environ.get(model_cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
    payload = {
        "model": model_cfg.get("model_name", "Qwen/Qwen2.5-14B-Instruct"),
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _repair_planning_output(
    *,
    sample: BenchmarkSample,
    raw_prediction: str,
    model_cfg: dict[str, Any],
    mode: str,
    violations: list[str] | None = None,
    immutable_anchor_events: list[dict[str, Any]] | None = None,
    alert_state: dict[str, Any] | None = None,
) -> str | None:
    constraints = sample.metadata.get("constraints", {})
    disruptions = sample.metadata.get("disruptions", [])
    problem = str(sample.metadata.get("problem", "")).strip().upper()
    base_system = (
        "You are a strict output normalizer. Return ONLY valid JSON with keys "
        "`plan_summary` and `events`."
    )

    if mode == "format":
        user = (
            "Convert the following model output to strict JSON. "
            "Do not add markdown or explanations. "
            "If events can be recovered, include them as `events` list with keys "
            "start,end,who,what,location,notes.\n\n"
            f"Output to normalize:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1200,
            temperature=0.0,
        )

    if mode == "fill_events":
        user = (
            "The current output has missing or empty events. "
            "Create a minimally complete plan JSON with at least 6 events. "
            "Keep as much existing information as possible. "
            "Every event must contain start,end,who,what,location,notes.\n\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Disruptions: {json.dumps(disruptions, ensure_ascii=False)}\n\n"
            f"Current output:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1400,
            temperature=0.0,
        )

    if mode == "disruption":
        violation_text = "; ".join(violations or [])
        p9_guard = ""
        if problem == "P9":
            p9_guard = (
                "4) For P9, keep cooking/safety/deadline constraints as invariant-first priority "
                "while applying disruption changes.\n"
            )
        user = (
            "Repair this planning JSON for disruption handling with two-phase logic.\n"
            "Rules:\n"
            "1) Preserve immutable past events before disruption start as much as possible.\n"
            "2) Update post-disruption events so disruptions are explicitly reflected in notes/what.\n"
            "3) Return only strict JSON with plan_summary/events.\n\n"
            f"{p9_guard}"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Disruptions: {json.dumps(disruptions, ensure_ascii=False)}\n"
            f"Violations to fix: {violation_text}\n\n"
            f"Current output:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1600,
            temperature=0.0,
        )

    if mode == "boundary_crossing":
        alert = str(sample.metadata.get("boundary_alert_time", "13:00"))
        depart = str(sample.metadata.get("boundary_departure_hint", "12:50"))
        planned_end = str(sample.metadata.get("boundary_planned_end_hint", "13:20"))
        route = str(sample.metadata.get("boundary_route", "A-B"))
        violation_text = "; ".join(violations or [])
        user = (
            "Repair this planning JSON with suffix-only replanning after boundary split.\n"
            "Required:\n"
            "1) Output ONLY `suffix_events` in JSON. Do not output full `events`.\n"
            "2) Keep pre-alert prefix immutable; do not rewrite pre-alert finished events.\n"
            f"2) Include one travel event on route `{route}` with start < {alert} < end.\n"
            "   Add `boundary_crossing=true` in notes for that crossing travel event.\n"
            f"3) Prefer boundary hints: depart near {depart}, planned end near {planned_end}.\n"
            "4) Regenerate only post-alert suffix events to satisfy constraints.\n"
            "5) Do not schedule any suffix event before `next_actionable_time`.\n"
            "6) Return strict JSON with keys: plan_summary, suffix_events.\n\n"
            f"Violations to fix: {violation_text}\n\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Disruptions: {json.dumps(disruptions, ensure_ascii=False)}\n\n"
            f"Immutable anchors (must stay unchanged): {json.dumps(immutable_anchor_events or [], ensure_ascii=False)}\n"
            f"Alert boundary state: {json.dumps(alert_state or {}, ensure_ascii=False)}\n\n"
            f"Current output:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1600,
            temperature=0.0,
        )

    if mode == "constraint":
        violation_text = "; ".join(violations or [])
        user = (
            "Repair this planning JSON so it satisfies constraints while changing as little as possible. "
            "Keep immutable past events when possible and adjust post-disruption events.\n\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Disruptions: {json.dumps(disruptions, ensure_ascii=False)}\n"
            f"Violations to fix: {violation_text}\n\n"
            f"Current output:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1400,
            temperature=0.0,
        )

    if mode == "photo_time":
        violation_text = "; ".join(violations or [])
        user = (
            "Repair this planning JSON with a photo_time-focused suffix edit.\n"
            "Hard rules:\n"
            "1) Output ONLY `suffix_events` in JSON. Do not output full `events`.\n"
            "2) Do not change immutable anchors.\n"
            "3) Keep split crossing events unchanged.\n"
            "4) Regenerate only post-alert suffix so that no event exceeds photo_time.\n"
            "5) Do not schedule any suffix event before `next_actionable_time`.\n"
            "6) Return only strict JSON with keys plan_summary,suffix_events.\n\n"
            f"Violations to fix: {violation_text}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Disruptions: {json.dumps(disruptions, ensure_ascii=False)}\n"
            f"Immutable anchors: {json.dumps(immutable_anchor_events or [], ensure_ascii=False)}\n"
            f"Alert boundary state: {json.dumps(alert_state or {}, ensure_ascii=False)}\n\n"
            f"Current output:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1400,
            temperature=0.0,
        )

    if mode == "tailor_hours":
        violation_text = "; ".join(violations or [])
        user = (
            "Repair this planning JSON with a tailor-hours-focused suffix edit.\n"
            "Hard rules:\n"
            "1) Output ONLY `suffix_events` in JSON. Do not output full `events`.\n"
            "2) Keep immutable prefix/split anchors unchanged.\n"
            "3) Regenerate only post-alert suffix.\n"
            "4) Ensure no tailor task ends after `tailor_closes`.\n"
            "5) Do not schedule any suffix event before `next_actionable_time`.\n"
            "6) Return strict JSON with keys: plan_summary, suffix_events.\n\n"
            f"Violations to fix: {violation_text}\n"
            f"Constraints: {json.dumps(constraints, ensure_ascii=False)}\n"
            f"Disruptions: {json.dumps(disruptions, ensure_ascii=False)}\n"
            f"Immutable anchors: {json.dumps(immutable_anchor_events or [], ensure_ascii=False)}\n"
            f"Alert boundary state: {json.dumps(alert_state or {}, ensure_ascii=False)}\n\n"
            f"Current output:\n{raw_prediction}"
        )
        return _backend_chat_completion(
            model_cfg=model_cfg,
            messages=[{"role": "system", "content": base_system}, {"role": "user", "content": user}],
            max_tokens=1400,
            temperature=0.0,
        )

    return None


def run_single_sample(
    variant: str,
    sample: BenchmarkSample,
    seed: int,
    model_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    log_dir: Path,
) -> dict[str, Any]:
    """Run one sample and return normalized result record."""
    rlm = build_rlm_instance(variant, sample, model_cfg, run_cfg, log_dir)

    started = time.perf_counter()
    llm_call_count = 0
    llm_call_count_capped = 0
    llm_call_reason_counts: dict[str, int] = {}
    llm_time_total = 0.0
    validator_time_total = 0.0
    postproc_time_total = 0.0
    timeline_norm_time_total = 0.0
    status = "ok"
    raw_response = ""
    usage = {}
    metadata = None
    error_message = None
    runtime_eval_mode = str(run_cfg.get("runtime_eval_mode", "runtime"))
    runtime_validator_mode = str(run_cfg.get("runtime_validator_mode", "auto"))
    if variant in {"V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"}:
        runtime_validator_mode = str(run_cfg.get("runtime_validator_mode_prefix", runtime_validator_mode))
    runtime_eval_version = get_evaluator_version(runtime_eval_mode)
    saga_variants = {"V2", "V3", "V3_BASE", "V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"}
    validator_version = (
        get_runtime_validator_version(sample, mode=runtime_validator_mode)
        if variant in saga_variants
        else "none"
    )
    effective_prompt = sample.prompt
    if bool(run_cfg.get("planning_boundary_real_candidate_hint", True)) and bool(
        sample.metadata.get("require_boundary_crossing", False)
    ):
        alert = str(sample.metadata.get("boundary_alert_time", "13:00"))
        route = str(sample.metadata.get("boundary_route", "A-B"))
        effective_prompt = (
            f"{sample.prompt}\n\n"
            "Boundary output contract (must satisfy):\n"
            f"- Include at least one explicit travel event on route `{route}` with start < {alert} < end.\n"
            "- For that event, include `boundary_crossing=true` in notes.\n"
            "- Keep this event in final JSON; do not omit it.\n"
        )

    try:
        llm_call_count += 1
        llm_call_reason_counts["PLAN"] = llm_call_reason_counts.get("PLAN", 0) + 1
        llm_started = time.perf_counter()
        completion = rlm.completion(effective_prompt)
        llm_time_total += time.perf_counter() - llm_started
        raw_response = completion.response or ""
        usage = _usage_to_dict(completion.usage_summary)
        metadata = completion.metadata
    except Exception as exc:
        status = "error"
        error_message = str(exc)
    finally:
        rlm.close()

    root_completion_time_sec = time.perf_counter() - started

    prefix_lock_variants = set(
        run_cfg.get("planning_prefix_lock_variants", ["V3_PREFIX", "V3_PREFIX_SPLIT", "V3_PREFIX_NO_SPLIT"])
    )
    prefix_lock_enabled = bool(run_cfg.get("planning_prefix_lock", True)) and variant in prefix_lock_variants
    selection_policy = str(
        run_cfg.get(
            "planning_selection_policy_prefix_lock"
            if prefix_lock_enabled
            else "planning_selection_policy",
            "default",
        )
    )
    alert_min = _resolve_alert_min(sample.metadata.get("disruptions", []))
    next_actionable_min: int | None = alert_min
    next_actionable_location = ""
    locked_prefix_events: list[dict[str, Any]] = []

    if status == "ok":
        validator_started = time.perf_counter()
        scored = _evaluate_with_optional_prefix_lock(
            sample=sample,
            prediction=raw_response,
            mode=runtime_eval_mode,
            locked_prefix_events=locked_prefix_events,
        )
        validator_time_total += time.perf_counter() - validator_started
    else:
        scored = {
            "success": 0,
            "score": 0.0,
            "violation_count": 1,
            "violations": [error_message or "unknown error"],
            "parsed_output": None,
            "diagnostics": {
                "valid_json": 0,
                "events_count": 0,
                "non_empty_events": 0,
                "disruption_required": int(bool(sample.metadata.get("disruptions"))),
                "disruption_applicable": 0,
                "disruption_applied": 0,
                "partial_compensation_applicable": 0,
                "partial_compensation_ok": 0,
                "crossing_split_applicable": 0,
                "crossing_split_applied": 0,
                "immutable_check_applicable": 0,
                "immutable_prefix_ok": 0,
                "immutable_prefix_after_split_ok": 0,
                "state_check_applicable": 0,
                "state_at_alert_consistent": 0,
                "prefix_lock_used": 0,
            },
        }

    repair_applied = 0
    repair_modes: list[str] = []
    split_attempted = 0
    split_attempt_count = 0
    split_applied_runtime = 0
    split_marker_survived = 0
    split_apply_mode = "SKIPPED_NOT_APPLICABLE"
    split_failure_reason = "NOT_ATTEMPTED"
    split_failure_reasons: list[str] = []
    boundary_gate_passed = 0
    boundary_gate_pass_after_fix_iter = -1
    boundary_gate_failed_reasons: list[str] = []
    boundary_gate_fix_counts: dict[str, int] = {}
    gate_strict_success_last = 0
    gate_strict_violation_count_last = 0
    early_exit_taken = 0
    early_exit_stage = ""
    v3_fallback_to_split_only = 0
    v3_fallback_reason = ""
    split_retry_triggered = 0
    split_marker_lost_stage = ""
    split_candidate_found = 0
    split_candidate_count = 0
    split_candidate_summary: dict[str, Any] = {}
    best_plan_selected = 0
    best_plan_source = "final"
    best_plan_violation_improvement_over_last = 0
    best_plan_score_best = 0
    best_plan_score_last = 0
    best_plan_score_improvement_over_last = 0
    suffix_only_output_ok = 0
    prefix_edit_attempt_detected = 0
    prefix_edit_ignored_count = 0
    next_actionable_time = ""
    suffix_first_event_time = ""
    suffix_first_event_time_before = ""
    suffix_first_event_time_after = ""
    timeline_norm_applied = 0
    timeline_norm_total_shift_minutes = 0
    timeline_norm_overlap_fixes_count = 0
    timeline_norm_location_fix_applied = 0
    boundary_canonicalization_applied = 0
    post_boundary_monotonic_fix_applied = 0
    post_boundary_monotonic_fix_count = 0
    post_boundary_total_shift_minutes = 0
    post_boundary_first_start_before = ""
    post_boundary_first_start_after = ""
    missing_boundary_event_detected = 0
    missing_boundary_event_autofixed = 0
    missing_boundary_event_autofix_minutes_pre = 0
    missing_boundary_event_autofix_minutes_post = 0
    immutable_scope_sanitize_applied = 0
    immutable_scope_terms_removed_count = 0
    immutable_scope_contains_disruption_terms_before = 0
    immutable_scope_contains_disruption_terms_after = 0
    boundary_pre_end_minus_alert_min = 0
    boundary_post_start_minus_alert_min = 0
    state_consistency_failure_reason = ""
    immutability_guard_triggered = 0
    immutability_guard_failure_stage = ""
    immutability_guard_failure_stages: list[str] = []
    num_candidates_generated = 0
    num_candidates_scored = 0
    num_candidates_discarded_by_guard = 0
    locked_prefix_snapshot: list[dict[str, Any]] = []
    locked_prefix_hash = ""
    scorer_view_locked_hash = ""
    scorer_view_locked_monotonic = 0
    final_prefix_extracted: list[dict[str, Any]] = []
    final_prefix_hash = ""
    scorer_view_final_hash = ""
    scorer_view_final_monotonic = 0
    guard_view_hash_match = 0
    scorer_view_hash_match = 0
    immutable_diff_type = ""
    prefix_contains_disruption_terms = 0
    split_pre_contains_disruption_terms = 0
    immutable_terms_detected = 0
    immutable_terms_locations: list[str] = []
    immutable_terms_fix_action = ""
    photo_time_violation_detected = 0
    photo_time_repair_triggered = 0
    photo_time_repair_success = 0
    photo_time_repair_method = ""
    photo_time_repair_search_nodes = 0
    photo_time_repair_wall_ms = 0.0

    # P0: format-stability repair path (applied to all variants, for fair comparison).
    if status == "ok" and sample.task_type == "planning":
        current_scored = scored
        current_prediction = raw_response
        immutable_anchor_events: list[dict[str, Any]] = []
        alert_state_summary: dict[str, Any] = {}
        next_actionable_min: int | None = alert_min
        next_actionable_location = ""
        best_commit_scored = current_scored
        best_commit_prediction = current_prediction
        best_commit_source = "initial"

        def _update_best_commit(prediction: str, scored_obj: dict[str, Any], source: str) -> None:
            nonlocal best_commit_scored, best_commit_prediction, best_commit_source
            if _prefer_commit_candidate(scored_obj, best_commit_scored):
                best_commit_scored = scored_obj
                best_commit_prediction = prediction
                best_commit_source = source

        split_variants = set(run_cfg.get("planning_boundary_split_variants", ["V3_PREFIX_SPLIT", "V0_SPLIT_ONLY"]))
        split_enabled = bool(run_cfg.get("planning_boundary_split_enabled", True)) and (
            variant in split_variants
        )
        guard_variants = set(
            run_cfg.get("planning_immutability_prefix_hash_guard_variants", ["V3_PREFIX_SPLIT"])
        )
        immutability_guard_enabled = bool(
            run_cfg.get("planning_immutability_prefix_hash_guard", True)
        ) and (variant in guard_variants)
        suffix_regen_variants = set(
            run_cfg.get("planning_boundary_suffix_regen_variants", ["V3_PREFIX_SPLIT"])
        )
        suffix_regen_always = bool(run_cfg.get("planning_boundary_suffix_regen_always", True)) and (
            split_enabled and variant in suffix_regen_variants
        )

        def _refresh_locked_prefix_snapshot() -> None:
            nonlocal locked_prefix_events
            nonlocal locked_prefix_snapshot, locked_prefix_hash
            nonlocal scorer_view_locked_hash, scorer_view_locked_monotonic
            if alert_min is None or not locked_prefix_events:
                return
            if bool(run_cfg.get("planning_prefix_monotonic_repair", True)):
                locked_prefix_events = _repair_prefix_actor_overlaps(
                    locked_prefix_events, alert_min
                )
            locked_prefix_snapshot = _dedup_sort_events(copy.deepcopy(locked_prefix_events))
            locked_prefix_hash = _prefix_hash(locked_prefix_snapshot)
            scorer_view_locked_hash, scorer_view_locked_monotonic = _scorer_view_hash(
                locked_prefix_snapshot, alert_min
            )

        def _guard_prediction(
            prediction: str,
            stage: str,
        ) -> tuple[str | None, list[dict[str, Any]], str]:
            nonlocal immutability_guard_triggered, immutability_guard_failure_stage
            nonlocal num_candidates_generated, num_candidates_discarded_by_guard
            num_candidates_generated += 1
            if (
                bool(run_cfg.get("planning_prefix_lock_force_synthesis", True))
                and alert_min is not None
                and locked_prefix_events
            ):
                fixed_prediction = _apply_prefix_lock_json(
                    prediction, locked_prefix_events, alert_min
                )
                if fixed_prediction:
                    prediction = fixed_prediction
            if alert_min is None:
                return prediction, [], ""
            candidate_prefix = _extract_prefix_events(
                _extract_events_from_prediction(prediction), alert_min
            )
            candidate_hash = _prefix_hash(candidate_prefix) if candidate_prefix else ""
            if (
                immutability_guard_enabled
                and split_enabled
                and locked_prefix_hash
                and candidate_hash != locked_prefix_hash
            ):
                immutability_guard_triggered = 1
                immutability_guard_failure_stages.append(stage)
                if not immutability_guard_failure_stage:
                    immutability_guard_failure_stage = stage
                num_candidates_discarded_by_guard += 1
                return None, candidate_prefix, candidate_hash
            return prediction, candidate_prefix, candidate_hash

        def _evaluate_candidate(
            prediction: str, lock_events: list[dict[str, Any]]
        ) -> dict[str, Any]:
            nonlocal num_candidates_scored, validator_time_total
            num_candidates_scored += 1
            validator_started = time.perf_counter()
            result = _evaluate_with_optional_prefix_lock(
                sample=sample,
                prediction=prediction,
                mode=runtime_eval_mode,
                locked_prefix_events=lock_events,
            )
            validator_time_total += time.perf_counter() - validator_started
            return result

        def _evaluate_candidate_strict(
            prediction: str, lock_events: list[dict[str, Any]]
        ) -> dict[str, Any]:
            nonlocal validator_time_total
            validator_started = time.perf_counter()
            result = _evaluate_with_optional_prefix_lock(
                sample=sample,
                prediction=prediction,
                mode="strict",
                locked_prefix_events=lock_events,
            )
            validator_time_total += time.perf_counter() - validator_started
            return result

        def _call_repair(*, reason: str = "REPAIR", **kwargs: Any) -> str | None:
            nonlocal llm_call_count, llm_time_total, llm_call_count_capped
            llm_call_cap = int(run_cfg.get("planning_llm_call_cap", 0))
            if llm_call_cap > 0 and llm_call_count >= llm_call_cap:
                llm_call_count_capped = 1
                return None
            llm_call_count += 1
            llm_call_reason_counts[reason] = llm_call_reason_counts.get(reason, 0) + 1
            llm_started = time.perf_counter()
            result = _repair_planning_output(**kwargs)
            llm_time_total += time.perf_counter() - llm_started
            return result

        def _sync_locked_prefix_from_prediction(prediction: str) -> None:
            nonlocal locked_prefix_events
            if not prefix_lock_enabled or alert_min is None:
                return
            refreshed = _extract_prefix_events(
                _extract_events_from_prediction(prediction), alert_min
            )
            if refreshed:
                locked_prefix_events = refreshed
                _refresh_locked_prefix_snapshot()

        def _is_state_immutable_ready(scored_obj: dict[str, Any]) -> bool:
            diag = scored_obj.get("diagnostics", {}) if isinstance(scored_obj, dict) else {}
            if not isinstance(diag, dict):
                return False
            if int(diag.get("valid_json", 0)) != 1 or int(diag.get("non_empty_events", 0)) != 1:
                return False
            if split_enabled:
                return (
                    int(diag.get("crossing_split_applied", 0)) == 1
                    and int(diag.get("immutable_prefix_after_split_ok", 0)) == 1
                    and int(diag.get("state_at_alert_consistent", 0)) == 1
                )
            return int(scored_obj.get("success", 0)) == 1

        def _maybe_take_early_exit(stage: str) -> bool:
            nonlocal early_exit_taken, early_exit_stage, boundary_gate_passed
            if early_exit_taken:
                return True
            if not bool(run_cfg.get("planning_early_exit_enabled", True)):
                return False
            if split_enabled and not boundary_gate_passed:
                return False
            if int(current_scored.get("success", 0)) != 1:
                return False
            if not _is_state_immutable_ready(current_scored):
                return False
            early_exit_taken = 1
            early_exit_stage = stage
            return True

        def _consume_structural(structural: dict[str, Any]) -> None:
            nonlocal suffix_first_event_time_before, suffix_first_event_time_after
            nonlocal timeline_norm_applied, timeline_norm_total_shift_minutes
            nonlocal timeline_norm_overlap_fixes_count, timeline_norm_location_fix_applied
            nonlocal timeline_norm_time_total
            nonlocal boundary_canonicalization_applied
            nonlocal post_boundary_monotonic_fix_applied, post_boundary_monotonic_fix_count
            nonlocal post_boundary_total_shift_minutes
            nonlocal post_boundary_first_start_before, post_boundary_first_start_after
            nonlocal missing_boundary_event_detected, missing_boundary_event_autofixed
            nonlocal missing_boundary_event_autofix_minutes_pre, missing_boundary_event_autofix_minutes_post
            nonlocal immutable_scope_sanitize_applied, immutable_scope_terms_removed_count
            nonlocal immutable_scope_contains_disruption_terms_before, immutable_scope_contains_disruption_terms_after
            nonlocal boundary_pre_end_minus_alert_min, boundary_post_start_minus_alert_min
            if str(structural.get("suffix_first_event_time_before", "")):
                suffix_first_event_time_before = str(
                    structural.get("suffix_first_event_time_before", "")
                )
            if str(structural.get("suffix_first_event_time_after", "")):
                suffix_first_event_time_after = str(
                    structural.get("suffix_first_event_time_after", "")
                )
            timeline_norm_applied = max(
                timeline_norm_applied, int(structural.get("timeline_norm_applied", 0))
            )
            timeline_norm_total_shift_minutes += int(
                structural.get("timeline_norm_total_shift_minutes", 0)
            )
            timeline_norm_overlap_fixes_count += int(
                structural.get("timeline_norm_overlap_fixes_count", 0)
            )
            timeline_norm_location_fix_applied += int(
                structural.get("timeline_norm_location_fix_applied", 0)
            )
            timeline_norm_time_total += float(
                structural.get("timeline_norm_runtime_sec", 0.0)
            )
            boundary_canonicalization_applied = max(
                boundary_canonicalization_applied,
                int(structural.get("boundary_canonicalization_applied", 0)),
            )
            post_boundary_monotonic_fix_applied = max(
                post_boundary_monotonic_fix_applied,
                int(structural.get("post_boundary_monotonic_fix_applied", 0)),
            )
            post_boundary_monotonic_fix_count += int(
                structural.get("post_boundary_monotonic_fix_count", 0)
            )
            post_boundary_total_shift_minutes += int(
                structural.get("post_boundary_total_shift_minutes", 0)
            )
            if str(structural.get("post_boundary_first_start_before", "")):
                post_boundary_first_start_before = str(
                    structural.get("post_boundary_first_start_before", "")
                )
            if str(structural.get("post_boundary_first_start_after", "")):
                post_boundary_first_start_after = str(
                    structural.get("post_boundary_first_start_after", "")
                )
            missing_boundary_event_detected = max(
                missing_boundary_event_detected,
                int(structural.get("missing_boundary_event_detected", 0)),
            )
            missing_boundary_event_autofixed = max(
                missing_boundary_event_autofixed,
                int(structural.get("missing_boundary_event_autofixed", 0)),
            )
            missing_boundary_event_autofix_minutes_pre = max(
                missing_boundary_event_autofix_minutes_pre,
                int(structural.get("missing_boundary_event_autofix_minutes_pre", 0)),
            )
            missing_boundary_event_autofix_minutes_post = max(
                missing_boundary_event_autofix_minutes_post,
                int(structural.get("missing_boundary_event_autofix_minutes_post", 0)),
            )
            immutable_scope_sanitize_applied = max(
                immutable_scope_sanitize_applied,
                int(structural.get("immutable_scope_sanitize_applied", 0)),
            )
            immutable_scope_terms_removed_count += int(
                structural.get("immutable_scope_terms_removed_count", 0)
            )
            immutable_scope_contains_disruption_terms_before = max(
                immutable_scope_contains_disruption_terms_before,
                int(
                    structural.get(
                        "immutable_scope_contains_disruption_terms_before", 0
                    )
                ),
            )
            immutable_scope_contains_disruption_terms_after = max(
                immutable_scope_contains_disruption_terms_after,
                int(
                    structural.get(
                        "immutable_scope_contains_disruption_terms_after", 0
                    )
                ),
            )
            boundary_pre_end_minus_alert_min = int(
                structural.get("boundary_pre_end_minus_alert_min", boundary_pre_end_minus_alert_min)
            )
            boundary_post_start_minus_alert_min = int(
                structural.get(
                    "boundary_post_start_minus_alert_min",
                    boundary_post_start_minus_alert_min,
                )
            )

        def _consume_photo_repair_metrics(metrics: dict[str, Any]) -> None:
            nonlocal photo_time_violation_detected, photo_time_repair_triggered
            nonlocal photo_time_repair_success, photo_time_repair_method
            nonlocal photo_time_repair_search_nodes, photo_time_repair_wall_ms
            if not metrics:
                return
            photo_time_violation_detected = max(
                photo_time_violation_detected,
                int(metrics.get("photo_time_violation_detected", 0)),
            )
            photo_time_repair_triggered = max(
                photo_time_repair_triggered,
                int(metrics.get("photo_time_repair_triggered", 0)),
            )
            photo_time_repair_success = max(
                photo_time_repair_success,
                int(metrics.get("photo_time_repair_success", 0)),
            )
            method = str(metrics.get("photo_time_repair_method", ""))
            if method:
                photo_time_repair_method = method
            photo_time_repair_search_nodes = max(
                photo_time_repair_search_nodes,
                int(metrics.get("photo_time_repair_search_nodes", 0)),
            )
            photo_time_repair_wall_ms = max(
                photo_time_repair_wall_ms,
                float(metrics.get("photo_time_repair_wall_ms", 0.0)),
            )

        def _inc_gate_fix(name: str) -> None:
            boundary_gate_fix_counts[name] = boundary_gate_fix_counts.get(name, 0) + 1

        def _collect_boundary_gate_reasons(
            prediction: str, lock_events: list[dict[str, Any]]
        ) -> tuple[list[str], dict[str, Any]]:
            strict_scored = _evaluate_candidate_strict(prediction, lock_events)
            strict_violations = strict_scored.get("violations", [])
            if not isinstance(strict_violations, list):
                strict_violations = []
            reasons = _gate_reasons_from_strict_violations(
                [str(v) for v in strict_violations]
            )

            # Keep marker-loss explicitly tracked; this helps pinpoint pipeline corruption.
            events = _extract_events_from_prediction(prediction)
            split_pre, split_post = _split_marker_state(events)
            if split_pre == 0 and split_post == 0:
                reasons.extend(["SPLIT_MISSING", "MARKER_LOST"])
            elif split_pre == 0 or split_post == 0:
                reasons.append("MARKER_LOST")

            seen: set[str] = set()
            uniq: list[str] = []
            for reason in reasons:
                if reason in seen:
                    continue
                seen.add(reason)
                uniq.append(reason)
            return uniq, strict_scored

        def _apply_boundary_invariant_gate() -> None:
            nonlocal current_prediction, current_scored
            nonlocal split_attempted, split_attempt_count, split_applied_runtime
            nonlocal split_marker_survived, split_apply_mode, split_failure_reason
            nonlocal split_candidate_found, split_candidate_count, split_candidate_summary
            nonlocal split_failure_reasons
            nonlocal suffix_only_output_ok, prefix_edit_attempt_detected, prefix_edit_ignored_count
            nonlocal next_actionable_time, suffix_first_event_time
            nonlocal boundary_gate_passed, boundary_gate_pass_after_fix_iter
            nonlocal gate_strict_success_last, gate_strict_violation_count_last
            nonlocal immutable_terms_detected, immutable_terms_locations, immutable_terms_fix_action
            nonlocal split_marker_lost_stage

            if not split_enabled or alert_min is None:
                return

            gate_iters = max(1, int(run_cfg.get("planning_boundary_gate_fix_iters", 2)))
            gate_iters = min(gate_iters, 2)
            for fix_iter in range(gate_iters + 1):
                reasons, strict_gate_scored = _collect_boundary_gate_reasons(
                    current_prediction, locked_prefix_events
                )
                gate_strict_success_last = int(strict_gate_scored.get("success", 0))
                gate_strict_violation_count_last = int(
                    strict_gate_scored.get("violation_count", 0)
                )
                if not reasons:
                    boundary_gate_passed = 1
                    boundary_gate_pass_after_fix_iter = fix_iter
                    _maybe_take_early_exit(f"BOUNDARY_GATE_ITER_{fix_iter}")
                    return
                boundary_gate_failed_reasons.extend(reasons)
                if fix_iter >= gate_iters:
                    break

                # 1) Ensure split/marker + boundary crossing segment and compensation.
                if any(
                    r in reasons
                    for r in ("SPLIT_MISSING", "MARKER_LOST", "BOUNDARY_MISSING", "COMP_MISSING")
                ):
                    split_result = _apply_boundary_split_compensation_json(
                        current_prediction, sample.metadata, alert_min
                    )
                    split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
                    split_attempt_count += int(split_result.get("split_attempted", 0))
                    split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
                    split_failure_reason = str(split_result.get("split_failure_reason", split_failure_reason))
                    split_failure_reasons.append(split_failure_reason)
                    split_candidate_found = max(
                        split_candidate_found, int(split_result.get("split_candidate_found", 0))
                    )
                    split_candidate_count += int(split_result.get("split_candidate_count", 0))
                    cand_summary = split_result.get("crossing_candidate_summary")
                    if isinstance(cand_summary, dict) and cand_summary:
                        split_candidate_summary = cand_summary
                    if split_result.get("prediction"):
                        current_prediction = str(split_result["prediction"])
                        split_applied_runtime = max(
                            split_applied_runtime, int(split_result.get("split_applied", 0))
                        )
                        split_marker_survived = max(
                            split_marker_survived,
                            int(split_result.get("split_marker_survived", 0)),
                        )
                        if int(split_result.get("split_marker_survived", 0)) == 0:
                            split_marker_lost_stage = split_marker_lost_stage or "GATE_SPLIT_APPLY"
                    _inc_gate_fix("SPLIT_OR_COMPENSATION")

                boundary_fixed, boundary_metrics = _ensure_boundary_event_json(
                    current_prediction,
                    alert_min=alert_min,
                    next_actionable_min=next_actionable_min,
                    next_actionable_location=next_actionable_location,
                    next_actionable_actor=str(alert_state_summary.get("next_actionable_actor", "")),
                    run_cfg=run_cfg,
                )
                if boundary_fixed:
                    current_prediction = boundary_fixed
                    split_applied_runtime = max(split_applied_runtime, 1)
                    split_marker_survived = max(split_marker_survived, 1)
                    _inc_gate_fix("BOUNDARY_AUTOFIX")
                _consume_structural(boundary_metrics)

                # 2) Deterministic state/timeline repair on suffix with fixed prefix.
                if immutable_anchor_events:
                    deterministic_fixed, norm_metrics = _deterministic_suffix_state_fix_json(
                        current_prediction,
                        immutable_anchor_events=immutable_anchor_events,
                        next_actionable_min=next_actionable_min,
                        next_actionable_location=next_actionable_location,
                        next_actionable_actor=str(alert_state_summary.get("next_actionable_actor", "")),
                        run_cfg=run_cfg,
                    )
                    if deterministic_fixed:
                        current_prediction = deterministic_fixed
                        _inc_gate_fix("STATE_SHIFT_FIX")
                    _consume_structural(norm_metrics)
                    if str(norm_metrics.get("next_actionable_time", "")):
                        next_actionable_time = str(norm_metrics.get("next_actionable_time", ""))
                    if str(norm_metrics.get("suffix_first_event_time_after", "")):
                        suffix_first_event_time = str(
                            norm_metrics.get("suffix_first_event_time_after", "")
                        )

                # 3) Canonicalize boundary marker edges around alert time.
                canon_fixed, canon_metrics = _canonicalize_boundary_split_markers_json(
                    current_prediction, alert_min
                )
                if canon_fixed:
                    current_prediction = canon_fixed
                    _inc_gate_fix("BOUNDARY_CANONICALIZE")
                _consume_structural(canon_metrics)

                # 4) Clamp full post-boundary timeline monotonicity at commit tail.
                post_fixed, post_metrics = _post_boundary_monotonic_clamp_json(
                    current_prediction,
                    alert_min=alert_min,
                    next_actionable_min=next_actionable_min,
                )
                if post_fixed:
                    current_prediction = post_fixed
                    _inc_gate_fix("POST_BOUNDARY_MONOTONIC")
                _consume_structural(post_metrics)

                # 5) Deterministic photo_time repair before immutable handling.
                if "PHOTO_TIME_EXCEEDED" in reasons:
                    deterministic_photo, photo_metrics = (
                        _deterministic_photo_time_suffix_repair_json(
                            raw_prediction=current_prediction,
                            constraints=sample.metadata.get("constraints", {}),
                            alert_state=alert_state_summary,
                            run_cfg=run_cfg,
                        )
                    )
                    _consume_photo_repair_metrics(photo_metrics)
                    if deterministic_photo:
                        current_prediction = deterministic_photo
                        _inc_gate_fix("PHOTO_TIME_REPAIR")

                # 6) Immutable scope handling: never sanitize in-place.
                #    Instead restore immutable prefix from lock snapshot, then optionally
                #    run one suffix-only regen if immutable terms still remain.
                imm_events = _extract_events_from_prediction(current_prediction)
                imm_locs_before = _immutable_scope_term_locations(imm_events, alert_min)
                if imm_locs_before:
                    immutable_terms_detected = 1
                    immutable_terms_locations = imm_locs_before
                    immutable_terms_fix_action = "RESTORE_FROM_SNAPSHOT"
                    restore_source = (
                        locked_prefix_snapshot if locked_prefix_snapshot else locked_prefix_events
                    )
                    restored = _apply_prefix_lock_json(
                        current_prediction, restore_source, alert_min
                    )
                    if restored:
                        current_prediction = restored
                        _inc_gate_fix("IMMUTABLE_RESTORE")

                    # Optional single suffix-only regen path if immutable terms still persist.
                    imm_events_after_restore = _extract_events_from_prediction(current_prediction)
                    imm_locs_after_restore = _immutable_scope_term_locations(
                        imm_events_after_restore, alert_min
                    )
                    if (
                        imm_locs_after_restore
                        and immutable_anchor_events
                        and bool(
                            run_cfg.get("planning_boundary_immutable_terms_regen", True)
                        )
                    ):
                        regenerated = _call_repair(
                            reason="IMMUTABLE_TERMS",
                            sample=sample,
                            raw_prediction=current_prediction,
                            model_cfg=model_cfg,
                            mode="boundary_crossing",
                            violations=["Immutable past appears modified by disruption terms"],
                            immutable_anchor_events=immutable_anchor_events,
                            alert_state=alert_state_summary,
                        )
                        if regenerated:
                            regenerated = (
                                _apply_prefix_lock_json(
                                    regenerated, restore_source, alert_min
                                )
                                or regenerated
                            )
                            current_prediction = regenerated
                            immutable_terms_fix_action = "REGEN_SUFFIX"
                            _inc_gate_fix("IMMUTABLE_REGEN")
                        else:
                            immutable_terms_fix_action = "DISCARD"

                _, imm_metrics = _sanitize_immutable_scope_json(current_prediction, alert_min)
                _consume_structural(imm_metrics)

                # 7) Re-evaluate and refresh lock snapshot before next gate check.
                _sync_locked_prefix_from_prediction(current_prediction)
                guarded_prediction, _, _ = _guard_prediction(
                    current_prediction, f"BOUNDARY_GATE_{fix_iter}"
                )
                if guarded_prediction:
                    current_prediction = guarded_prediction
                    current_scored = _evaluate_candidate(
                        current_prediction, locked_prefix_events
                    )
                    _update_best_commit(
                        current_prediction, current_scored, f"boundary_gate_{fix_iter}"
                    )

            boundary_gate_passed = 0
            boundary_gate_pass_after_fix_iter = -1

        def _build_split_only_fallback(
            prediction: str,
        ) -> tuple[str | None, dict[str, Any] | None]:
            nonlocal split_attempted, split_attempt_count, split_applied_runtime
            nonlocal split_marker_survived, split_apply_mode, split_failure_reason
            nonlocal split_candidate_found, split_candidate_count, split_candidate_summary
            nonlocal split_failure_reasons
            nonlocal next_actionable_time, suffix_first_event_time
            nonlocal suffix_only_output_ok, prefix_edit_attempt_detected, prefix_edit_ignored_count

            fallback_prediction = prediction
            split_result = _apply_boundary_split_compensation_json(
                fallback_prediction, sample.metadata, alert_min
            )
            split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
            split_attempt_count += int(split_result.get("split_attempted", 0))
            split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
            split_failure_reason = str(split_result.get("split_failure_reason", split_failure_reason))
            split_failure_reasons.append(split_failure_reason)
            split_candidate_found = max(
                split_candidate_found, int(split_result.get("split_candidate_found", 0))
            )
            split_candidate_count += int(split_result.get("split_candidate_count", 0))
            cand_summary = split_result.get("crossing_candidate_summary")
            if isinstance(cand_summary, dict) and cand_summary:
                split_candidate_summary = cand_summary
            if split_result.get("prediction"):
                fallback_prediction = str(split_result["prediction"])
                split_applied_runtime = max(
                    split_applied_runtime, int(split_result.get("split_applied", 0))
                )
                split_marker_survived = max(
                    split_marker_survived, int(split_result.get("split_marker_survived", 0))
                )

            boundary_fixed, boundary_metrics = _ensure_boundary_event_json(
                fallback_prediction,
                alert_min=alert_min,
                next_actionable_min=next_actionable_min,
                next_actionable_location=next_actionable_location,
                next_actionable_actor=str(alert_state_summary.get("next_actionable_actor", "")),
                run_cfg=run_cfg,
            )
            _consume_structural(boundary_metrics)
            if boundary_fixed:
                fallback_prediction = boundary_fixed
                split_applied_runtime = max(split_applied_runtime, 1)
                split_marker_survived = max(split_marker_survived, 1)

            canon_fixed, canon_metrics = _canonicalize_boundary_split_markers_json(
                fallback_prediction, alert_min
            )
            _consume_structural(canon_metrics)
            if canon_fixed:
                fallback_prediction = canon_fixed

            if immutable_anchor_events:
                deterministic_fixed, norm_metrics = _deterministic_suffix_state_fix_json(
                    fallback_prediction,
                    immutable_anchor_events=immutable_anchor_events,
                    next_actionable_min=next_actionable_min,
                    next_actionable_location=next_actionable_location,
                    next_actionable_actor=str(alert_state_summary.get("next_actionable_actor", "")),
                    run_cfg=run_cfg,
                )
                _consume_structural(norm_metrics)
                if deterministic_fixed:
                    fallback_prediction = deterministic_fixed
                if str(norm_metrics.get("next_actionable_time", "")):
                    next_actionable_time = str(norm_metrics.get("next_actionable_time", ""))
                if str(norm_metrics.get("suffix_first_event_time_after", "")):
                    suffix_first_event_time = str(
                        norm_metrics.get("suffix_first_event_time_after", "")
                    )

            post_fixed, post_metrics = _post_boundary_monotonic_clamp_json(
                fallback_prediction,
                alert_min=alert_min,
                next_actionable_min=next_actionable_min,
            )
            _consume_structural(post_metrics)
            if post_fixed:
                fallback_prediction = post_fixed

            if (
                split_enabled
                and immutable_anchor_events
                and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
            ):
                structural = _apply_boundary_structural_guards(
                    raw_prediction=fallback_prediction,
                    immutable_anchor_events=immutable_anchor_events,
                    alert_state=alert_state_summary,
                    alert_min=alert_min,
                    run_cfg=run_cfg,
                )
                _consume_structural(structural)
                fallback_prediction = str(structural.get("prediction") or fallback_prediction)
                suffix_only_output_ok = max(
                    suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                )
                prefix_edit_attempt_detected = max(
                    prefix_edit_attempt_detected,
                    int(structural.get("prefix_edit_attempt_detected", 0)),
                )
                prefix_edit_ignored_count += int(
                    structural.get("prefix_edit_ignored_count", 0)
                )

            _sync_locked_prefix_from_prediction(fallback_prediction)
            guarded_prediction, _, _ = _guard_prediction(
                fallback_prediction, "BOUNDARY_GATE_FALLBACK_SPLIT_ONLY"
            )
            if not guarded_prediction:
                return None, None
            candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
            return guarded_prediction, candidate

        if prefix_lock_enabled and alert_min is not None:
            current_events = _extract_events_from_prediction(current_prediction)
            locked_prefix_events = _extract_prefix_events(current_events, alert_min)
            _refresh_locked_prefix_snapshot()
            current_scored = _evaluate_candidate(current_prediction, locked_prefix_events)
            _update_best_commit(current_prediction, current_scored, "initial_prefix_lock")

        if bool(run_cfg.get("planning_json_repair", True)) and _is_parse_or_format_failure(current_scored):
            repaired = _call_repair(
                reason="FORMAT",
                sample=sample,
                raw_prediction=current_prediction,
                model_cfg=model_cfg,
                mode="format",
            )
            if repaired:
                if prefix_lock_enabled and locked_prefix_events:
                    repaired = _apply_prefix_lock_json(repaired, locked_prefix_events, alert_min) or repaired
                repair_applied = 1
                repair_modes.append("format")
                guarded_prediction, _, _ = _guard_prediction(repaired, "FORMAT_REPAIR")
                if guarded_prediction:
                    candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                    _update_best_commit(guarded_prediction, candidate, "format_repair")
                    if _prefer_scored(candidate, current_scored, selection_policy=selection_policy):
                        current_scored = candidate
                        current_prediction = guarded_prediction
                        _maybe_take_early_exit("FORMAT_REPAIR")

        if (
            bool(run_cfg.get("planning_fill_events_repair", True))
            and _is_empty_events_failure(current_scored)
            and not early_exit_taken
        ):
            max_fill_attempts = max(1, int(run_cfg.get("planning_fill_events_max_attempts", 2)))
            for _ in range(max_fill_attempts):
                repaired = _call_repair(
                    reason="FILL_EVENTS",
                    sample=sample,
                    raw_prediction=current_prediction,
                    model_cfg=model_cfg,
                    mode="fill_events",
                )
                if not repaired:
                    break
                if prefix_lock_enabled and locked_prefix_events:
                    repaired = _apply_prefix_lock_json(repaired, locked_prefix_events, alert_min) or repaired
                repair_applied = 1
                repair_modes.append("fill_events")
                guarded_prediction, _, _ = _guard_prediction(repaired, "FILL_EVENTS")
                if not guarded_prediction:
                    continue
                candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                _update_best_commit(guarded_prediction, candidate, "fill_events_repair")
                if _prefer_scored(candidate, current_scored, selection_policy=selection_policy):
                    current_scored = candidate
                    current_prediction = guarded_prediction
                    _maybe_take_early_exit("FILL_EVENTS")
                    if prefix_lock_enabled and alert_min is not None and not locked_prefix_events:
                        locked_prefix_events = _extract_prefix_events(
                            _extract_events_from_prediction(current_prediction), alert_min
                        )
                        _refresh_locked_prefix_snapshot()
                if not _is_empty_events_failure(current_scored):
                    break

        if split_enabled and not early_exit_taken:
            split_result = _apply_boundary_split_compensation_json(
                current_prediction,
                sample.metadata,
                alert_min,
            )
            split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
            split_attempt_count += int(split_result.get("split_attempted", 0))
            split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
            split_failure_reason = str(split_result.get("split_failure_reason", split_failure_reason))
            split_failure_reasons.append(split_failure_reason)
            split_candidate_found = max(
                split_candidate_found, int(split_result.get("split_candidate_found", 0))
            )
            split_candidate_count += int(split_result.get("split_candidate_count", 0))
            cand_summary = split_result.get("crossing_candidate_summary")
            if isinstance(cand_summary, dict) and cand_summary:
                split_candidate_summary = cand_summary
            if split_result.get("prediction"):
                split_prediction = str(split_result["prediction"])
                split_applied_runtime = max(
                    split_applied_runtime, int(split_result.get("split_applied", 0))
                )
                split_marker_survived = max(
                    split_marker_survived, int(split_result.get("split_marker_survived", 0))
                )
                if alert_min is not None:
                    immutable_anchor_events = _extract_immutable_anchor_events(
                        _extract_events_from_prediction(split_prediction), alert_min
                    )
                    alert_state_summary = _compute_alert_state_summary(
                        immutable_anchor_events, alert_min
                    )
                    next_actionable_min = _resolve_next_actionable_min(
                        alert_state_summary, alert_min
                    )
                    next_actionable_location = str(
                        alert_state_summary.get("next_actionable_location", "")
                    )
                split_locked_prefix = locked_prefix_events
                if prefix_lock_enabled and alert_min is not None:
                    split_locked_prefix = _extract_prefix_events(
                        _extract_events_from_prediction(split_prediction), alert_min
                    )
                    if split_locked_prefix:
                        locked_prefix_events = split_locked_prefix
                        _refresh_locked_prefix_snapshot()
                if (
                    split_enabled
                    and immutable_anchor_events
                    and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                ):
                    structural = _apply_boundary_structural_guards(
                        raw_prediction=split_prediction,
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                        alert_min=alert_min,
                        run_cfg=run_cfg,
                    )
                    _consume_structural(structural)
                    split_prediction = str(structural.get("prediction") or split_prediction)
                    suffix_only_output_ok = max(
                        suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                    )
                    prefix_edit_attempt_detected = max(
                        prefix_edit_attempt_detected,
                        int(structural.get("prefix_edit_attempt_detected", 0)),
                    )
                    prefix_edit_ignored_count += int(
                        structural.get("prefix_edit_ignored_count", 0)
                    )
                    if str(structural.get("next_actionable_time", "")):
                        next_actionable_time = str(structural.get("next_actionable_time", ""))
                    if str(structural.get("suffix_first_event_time", "")):
                        suffix_first_event_time = str(
                            structural.get("suffix_first_event_time", "")
                        )
                _sync_locked_prefix_from_prediction(split_prediction)
                guarded_prediction, _, _ = _guard_prediction(
                    split_prediction, "BOUNDARY_SPLIT_APPLY"
                )
                if guarded_prediction:
                    candidate = _evaluate_candidate(guarded_prediction, split_locked_prefix)
                    _update_best_commit(guarded_prediction, candidate, "boundary_split_apply")
                    repair_applied = 1
                    repair_modes.append("boundary_split")
                    force_split = bool(run_cfg.get("planning_boundary_split_force", True))
                    if force_split or _prefer_scored(
                        candidate, current_scored, selection_policy=selection_policy
                    ):
                        current_prediction = guarded_prediction
                        current_scored = candidate
                        _maybe_take_early_exit("BOUNDARY_SPLIT_APPLY")
                        if prefix_lock_enabled and alert_min is not None:
                            locked_prefix_events = split_locked_prefix
                            _refresh_locked_prefix_snapshot()

        # Boundary-specific structure: after split, run suffix-only regeneration at least once
        # so post-alert timeline is explicitly repaired under fixed prefix.
        if split_enabled and suffix_regen_always and not early_exit_taken:
            regen_attempts = max(1, int(run_cfg.get("planning_boundary_suffix_regen_attempts", 1)))
            regen_attempts = min(regen_attempts, 2)
            for _ in range(regen_attempts):
                repaired = _call_repair(
                    reason="BOUNDARY_CROSSING",
                    sample=sample,
                    raw_prediction=current_prediction,
                    model_cfg=model_cfg,
                    mode="boundary_crossing",
                    violations=current_scored.get("violations", []),
                    immutable_anchor_events=immutable_anchor_events,
                    alert_state=alert_state_summary,
                )
                if not repaired:
                    break
                split_result = _apply_boundary_split_compensation_json(
                    repaired,
                    sample.metadata,
                    alert_min,
                )
                split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
                split_attempt_count += int(split_result.get("split_attempted", 0))
                split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
                split_failure_reason = str(split_result.get("split_failure_reason", split_failure_reason))
                split_failure_reasons.append(split_failure_reason)
                split_candidate_found = max(
                    split_candidate_found, int(split_result.get("split_candidate_found", 0))
                )
                split_candidate_count += int(split_result.get("split_candidate_count", 0))
                cand_summary = split_result.get("crossing_candidate_summary")
                if isinstance(cand_summary, dict) and cand_summary:
                    split_candidate_summary = cand_summary
                if not split_result.get("prediction"):
                    continue
                repaired = str(split_result["prediction"])
                split_applied_runtime = max(
                    split_applied_runtime, int(split_result.get("split_applied", 0))
                )
                split_marker_survived = max(
                    split_marker_survived, int(split_result.get("split_marker_survived", 0))
                )
                if (
                    split_enabled
                    and immutable_anchor_events
                    and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                ):
                    structural = _apply_boundary_structural_guards(
                        raw_prediction=repaired,
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                        alert_min=alert_min,
                        run_cfg=run_cfg,
                    )
                    _consume_structural(structural)
                    repaired = str(structural.get("prediction") or repaired)
                    suffix_only_output_ok = max(
                        suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                    )
                    prefix_edit_attempt_detected = max(
                        prefix_edit_attempt_detected,
                        int(structural.get("prefix_edit_attempt_detected", 0)),
                    )
                    prefix_edit_ignored_count += int(
                        structural.get("prefix_edit_ignored_count", 0)
                    )
                    if str(structural.get("next_actionable_time", "")):
                        next_actionable_time = str(structural.get("next_actionable_time", ""))
                    if str(structural.get("suffix_first_event_time", "")):
                        suffix_first_event_time = str(
                            structural.get("suffix_first_event_time", "")
                        )
                elif prefix_lock_enabled and locked_prefix_events:
                    repaired = _apply_prefix_lock_json(repaired, locked_prefix_events, alert_min) or repaired
                _sync_locked_prefix_from_prediction(repaired)
                guarded_prediction, _, _ = _guard_prediction(
                    repaired, "BOUNDARY_SUFFIX_REGEN"
                )
                if not guarded_prediction:
                    continue
                candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                _update_best_commit(guarded_prediction, candidate, "boundary_suffix_regen")
                repair_applied = 1
                repair_modes.extend(["boundary_split", "boundary_crossing"])
                if _prefer_scored(candidate, current_scored, selection_policy=selection_policy):
                    current_scored = candidate
                    current_prediction = guarded_prediction
                    _maybe_take_early_exit("BOUNDARY_SUFFIX_REGEN")
                    if prefix_lock_enabled and alert_min is not None and not locked_prefix_events:
                        locked_prefix_events = _extract_prefix_events(
                            _extract_events_from_prediction(current_prediction), alert_min
                        )
                        _refresh_locked_prefix_snapshot()
                if int(current_scored.get("success", 0)) == 1:
                    break

        # P2: targeted retries for Saga variants (constraint vs disruption).
        if (
            variant in saga_variants
            and bool(run_cfg.get("planning_targeted_retry", True))
            and int(current_scored.get("success", 0)) == 0
            and int(current_scored.get("violation_count", 0)) > 0
            and not early_exit_taken
        ):
            max_repairs = 1 if variant == "V2" else max(1, int(run_cfg.get("saga_max_retries", 2)))
            max_repairs = min(max_repairs, 2)
            for _ in range(max_repairs):
                # Priority order for v7.8:
                # 1) state / immutable consistency (boundary structure)
                # 2) photo_time / store-hours domain constraints
                if split_enabled and _is_disruption_failure(current_scored):
                    mode = "boundary_crossing"
                elif _is_photo_time_failure(current_scored):
                    mode = "photo_time"
                elif _is_tailor_close_failure(current_scored):
                    mode = "tailor_hours"
                elif split_enabled and _has_missing_boundary_crossing_failure(current_scored):
                    mode = "boundary_crossing"
                elif _has_constraint_failures(current_scored):
                    mode = "constraint"
                else:
                    break
                if mode == "photo_time":
                    deterministic, photo_metrics = _deterministic_photo_time_suffix_repair_json(
                        raw_prediction=current_prediction,
                        constraints=sample.metadata.get("constraints", {}),
                        alert_state=alert_state_summary,
                        run_cfg=run_cfg,
                    )
                    _consume_photo_repair_metrics(photo_metrics)
                    if deterministic:
                        guarded_prediction, _, _ = _guard_prediction(
                            deterministic, "PHOTO_DETERMINISTIC"
                        )
                        if guarded_prediction:
                            candidate = _evaluate_candidate(
                                guarded_prediction, locked_prefix_events
                            )
                            _update_best_commit(
                                guarded_prediction, candidate, "photo_time_deterministic"
                            )
                            repair_applied = 1
                            repair_modes.append("photo_time_deterministic")
                            if _prefer_scored(
                                candidate, current_scored, selection_policy=selection_policy
                            ):
                                current_scored = candidate
                                current_prediction = guarded_prediction
                                _maybe_take_early_exit("PHOTO_TIME_DETERMINISTIC")
                            if int(current_scored.get("success", 0)) == 1 or not _is_photo_time_failure(
                                current_scored
                            ):
                                continue
                if mode == "tailor_hours":
                    deterministic_tailor = _deterministic_tailor_hours_suffix_repair_json(
                        raw_prediction=current_prediction,
                        constraints=sample.metadata.get("constraints", {}),
                        alert_state=alert_state_summary,
                        run_cfg=run_cfg,
                    )
                    if deterministic_tailor:
                        guarded_prediction, _, _ = _guard_prediction(
                            deterministic_tailor, "TAILOR_DETERMINISTIC"
                        )
                        if guarded_prediction:
                            candidate = _evaluate_candidate(
                                guarded_prediction, locked_prefix_events
                            )
                            _update_best_commit(
                                guarded_prediction, candidate, "tailor_hours_deterministic"
                            )
                            repair_applied = 1
                            repair_modes.append("tailor_hours_deterministic")
                            if _prefer_scored(
                                candidate, current_scored, selection_policy=selection_policy
                            ):
                                current_scored = candidate
                                current_prediction = guarded_prediction
                                _maybe_take_early_exit("TAILOR_DETERMINISTIC")
                            if int(current_scored.get("success", 0)) == 1 or not _is_tailor_close_failure(
                                current_scored
                            ):
                                continue
                repaired = _call_repair(
                    reason=mode.upper(),
                    sample=sample,
                    raw_prediction=current_prediction,
                    model_cfg=model_cfg,
                    mode=mode,
                    violations=current_scored.get("violations", []),
                    immutable_anchor_events=immutable_anchor_events,
                    alert_state=alert_state_summary,
                )
                if not repaired:
                    break
                if split_enabled:
                    split_result = _apply_boundary_split_compensation_json(
                        repaired,
                        sample.metadata,
                        alert_min,
                    )
                    split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
                    split_attempt_count += int(split_result.get("split_attempted", 0))
                    split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
                    split_failure_reason = str(
                        split_result.get("split_failure_reason", split_failure_reason)
                    )
                    split_failure_reasons.append(split_failure_reason)
                    split_candidate_found = max(
                        split_candidate_found, int(split_result.get("split_candidate_found", 0))
                    )
                    split_candidate_count += int(split_result.get("split_candidate_count", 0))
                    cand_summary = split_result.get("crossing_candidate_summary")
                    if isinstance(cand_summary, dict) and cand_summary:
                        split_candidate_summary = cand_summary
                    if split_result.get("prediction"):
                        repaired = str(split_result["prediction"])
                        split_applied_runtime = max(
                            split_applied_runtime, int(split_result.get("split_applied", 0))
                        )
                        split_marker_survived = max(
                            split_marker_survived,
                            int(split_result.get("split_marker_survived", 0)),
                        )
                        repair_modes.append("boundary_split")
                if (
                    split_enabled
                    and immutable_anchor_events
                    and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                ):
                    structural = _apply_boundary_structural_guards(
                        raw_prediction=repaired,
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                        alert_min=alert_min,
                        run_cfg=run_cfg,
                    )
                    _consume_structural(structural)
                    repaired = str(structural.get("prediction") or repaired)
                    suffix_only_output_ok = max(
                        suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                    )
                    prefix_edit_attempt_detected = max(
                        prefix_edit_attempt_detected,
                        int(structural.get("prefix_edit_attempt_detected", 0)),
                    )
                    prefix_edit_ignored_count += int(
                        structural.get("prefix_edit_ignored_count", 0)
                    )
                    if str(structural.get("next_actionable_time", "")):
                        next_actionable_time = str(structural.get("next_actionable_time", ""))
                    if str(structural.get("suffix_first_event_time", "")):
                        suffix_first_event_time = str(
                            structural.get("suffix_first_event_time", "")
                        )
                elif prefix_lock_enabled and locked_prefix_events:
                    repaired = _apply_prefix_lock_json(repaired, locked_prefix_events, alert_min) or repaired
                repair_applied = 1
                repair_modes.append(mode)
                _sync_locked_prefix_from_prediction(repaired)
                guarded_prediction, _, _ = _guard_prediction(
                    repaired, f"TARGETED_{mode.upper()}"
                )
                if not guarded_prediction:
                    continue
                candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                _update_best_commit(guarded_prediction, candidate, f"targeted_{mode}")
                if _prefer_scored(candidate, current_scored, selection_policy=selection_policy):
                    current_scored = candidate
                    current_prediction = guarded_prediction
                    _maybe_take_early_exit(f"TARGETED_{mode.upper()}")
                    if prefix_lock_enabled and alert_min is not None and not locked_prefix_events:
                        locked_prefix_events = _extract_prefix_events(
                            _extract_events_from_prediction(current_prediction), alert_min
                        )
                        _refresh_locked_prefix_snapshot()
                if int(current_scored.get("success", 0)) == 1:
                    break

        if (
            split_enabled
            and bool(run_cfg.get("planning_photo_time_hard_gate", True))
            and _is_photo_time_failure(current_scored)
            and not early_exit_taken
        ):
            photo_loops = max(1, int(run_cfg.get("planning_photo_time_hard_loops", 2)))
            photo_loops = min(photo_loops, 2)
            for _ in range(photo_loops):
                deterministic, photo_metrics = _deterministic_photo_time_suffix_repair_json(
                    raw_prediction=current_prediction,
                    constraints=sample.metadata.get("constraints", {}),
                    alert_state=alert_state_summary,
                    run_cfg=run_cfg,
                )
                _consume_photo_repair_metrics(photo_metrics)
                if deterministic:
                    guarded_prediction, _, _ = _guard_prediction(
                        deterministic, "PHOTO_DETERMINISTIC"
                    )
                    if guarded_prediction:
                        candidate = _evaluate_candidate(
                            guarded_prediction, locked_prefix_events
                        )
                        _update_best_commit(
                            guarded_prediction, candidate, "photo_time_deterministic"
                        )
                        repair_applied = 1
                        repair_modes.append("photo_time_deterministic")
                        if _prefer_scored(
                            candidate, current_scored, selection_policy=selection_policy
                        ):
                            current_scored = candidate
                            current_prediction = guarded_prediction
                            _maybe_take_early_exit("PHOTO_DETERMINISTIC")
                        if not _is_photo_time_failure(current_scored):
                            break
                repaired = _call_repair(
                    reason="PHOTO_TIME",
                    sample=sample,
                    raw_prediction=current_prediction,
                    model_cfg=model_cfg,
                    mode="photo_time",
                    violations=current_scored.get("violations", []),
                    immutable_anchor_events=immutable_anchor_events,
                    alert_state=alert_state_summary,
                )
                if not repaired:
                    break
                split_result = _apply_boundary_split_compensation_json(
                    repaired,
                    sample.metadata,
                    alert_min,
                )
                split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
                split_attempt_count += int(split_result.get("split_attempted", 0))
                split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
                split_failure_reason = str(split_result.get("split_failure_reason", split_failure_reason))
                split_failure_reasons.append(split_failure_reason)
                split_candidate_found = max(
                    split_candidate_found, int(split_result.get("split_candidate_found", 0))
                )
                split_candidate_count += int(split_result.get("split_candidate_count", 0))
                cand_summary = split_result.get("crossing_candidate_summary")
                if isinstance(cand_summary, dict) and cand_summary:
                    split_candidate_summary = cand_summary
                if not split_result.get("prediction"):
                    continue
                repaired = str(split_result["prediction"])
                split_applied_runtime = max(
                    split_applied_runtime, int(split_result.get("split_applied", 0))
                )
                split_marker_survived = max(
                    split_marker_survived, int(split_result.get("split_marker_survived", 0))
                )
                if (
                    split_enabled
                    and immutable_anchor_events
                    and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                ):
                    structural = _apply_boundary_structural_guards(
                        raw_prediction=repaired,
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                        alert_min=alert_min,
                        run_cfg=run_cfg,
                    )
                    _consume_structural(structural)
                    repaired = str(structural.get("prediction") or repaired)
                    suffix_only_output_ok = max(
                        suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                    )
                    prefix_edit_attempt_detected = max(
                        prefix_edit_attempt_detected,
                        int(structural.get("prefix_edit_attempt_detected", 0)),
                    )
                    prefix_edit_ignored_count += int(
                        structural.get("prefix_edit_ignored_count", 0)
                    )
                    if str(structural.get("next_actionable_time", "")):
                        next_actionable_time = str(structural.get("next_actionable_time", ""))
                    if str(structural.get("suffix_first_event_time", "")):
                        suffix_first_event_time = str(
                            structural.get("suffix_first_event_time", "")
                        )
                _sync_locked_prefix_from_prediction(repaired)
                guarded_prediction, _, _ = _guard_prediction(repaired, "PHOTO_EDIT")
                if not guarded_prediction:
                    continue
                candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                _update_best_commit(guarded_prediction, candidate, "photo_time_hard_gate")
                repair_applied = 1
                repair_modes.append("photo_time")
                if _prefer_scored(candidate, current_scored, selection_policy=selection_policy):
                    current_scored = candidate
                    current_prediction = guarded_prediction
                    _maybe_take_early_exit("PHOTO_EDIT")
                if not _is_photo_time_failure(current_scored):
                    break

        if (
            split_enabled
            and bool(run_cfg.get("planning_tailor_hard_gate", True))
            and _is_tailor_close_failure(current_scored)
            and not early_exit_taken
        ):
            tailor_loops = max(1, int(run_cfg.get("planning_tailor_hard_loops", 1)))
            tailor_loops = min(tailor_loops, 2)
            for _ in range(tailor_loops):
                deterministic_tailor = _deterministic_tailor_hours_suffix_repair_json(
                    raw_prediction=current_prediction,
                    constraints=sample.metadata.get("constraints", {}),
                    alert_state=alert_state_summary,
                    run_cfg=run_cfg,
                )
                if deterministic_tailor:
                    guarded_prediction, _, _ = _guard_prediction(
                        deterministic_tailor, "TAILOR_EDIT"
                    )
                    if guarded_prediction:
                        candidate = _evaluate_candidate(
                            guarded_prediction, locked_prefix_events
                        )
                        _update_best_commit(guarded_prediction, candidate, "tailor_hours_hard_gate")
                        repair_applied = 1
                        repair_modes.append("tailor_hours")
                        if _prefer_scored(
                            candidate, current_scored, selection_policy=selection_policy
                        ):
                            current_scored = candidate
                            current_prediction = guarded_prediction
                            _maybe_take_early_exit("TAILOR_EDIT")
                        if not _is_tailor_close_failure(current_scored):
                            break
                repaired = _call_repair(
                    reason="TAILOR_HOURS",
                    sample=sample,
                    raw_prediction=current_prediction,
                    model_cfg=model_cfg,
                    mode="tailor_hours",
                    violations=current_scored.get("violations", []),
                    immutable_anchor_events=immutable_anchor_events,
                    alert_state=alert_state_summary,
                )
                if not repaired:
                    break
                if (
                    split_enabled
                    and immutable_anchor_events
                    and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                ):
                    structural = _apply_boundary_structural_guards(
                        raw_prediction=repaired,
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                        alert_min=alert_min,
                        run_cfg=run_cfg,
                    )
                    _consume_structural(structural)
                    repaired = str(structural.get("prediction") or repaired)
                    suffix_only_output_ok = max(
                        suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                    )
                    prefix_edit_attempt_detected = max(
                        prefix_edit_attempt_detected,
                        int(structural.get("prefix_edit_attempt_detected", 0)),
                    )
                    prefix_edit_ignored_count += int(
                        structural.get("prefix_edit_ignored_count", 0)
                    )
                    if str(structural.get("next_actionable_time", "")):
                        next_actionable_time = str(structural.get("next_actionable_time", ""))
                    if str(structural.get("suffix_first_event_time", "")):
                        suffix_first_event_time = str(
                            structural.get("suffix_first_event_time", "")
                        )
                _sync_locked_prefix_from_prediction(repaired)
                guarded_prediction, _, _ = _guard_prediction(repaired, "TAILOR_REPAIR")
                if not guarded_prediction:
                    continue
                candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                _update_best_commit(guarded_prediction, candidate, "tailor_hours_repair")
                repair_applied = 1
                repair_modes.append("tailor_hours")
                if _prefer_scored(candidate, current_scored, selection_policy=selection_policy):
                    current_scored = candidate
                    current_prediction = guarded_prediction
                    _maybe_take_early_exit("TAILOR_REPAIR")
                if not _is_tailor_close_failure(current_scored):
                    break

        if split_enabled and split_applied_runtime == 0 and not early_exit_taken:
            split_result = _apply_boundary_split_compensation_json(
                current_prediction,
                sample.metadata,
                alert_min,
            )
            split_attempted = max(split_attempted, int(split_result.get("split_attempted", 0)))
            split_attempt_count += int(split_result.get("split_attempted", 0))
            split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
            split_failure_reason = str(split_result.get("split_failure_reason", split_failure_reason))
            split_failure_reasons.append(split_failure_reason)
            split_candidate_found = max(
                split_candidate_found, int(split_result.get("split_candidate_found", 0))
            )
            split_candidate_count += int(split_result.get("split_candidate_count", 0))
            cand_summary = split_result.get("crossing_candidate_summary")
            if isinstance(cand_summary, dict) and cand_summary:
                split_candidate_summary = cand_summary
            if split_result.get("prediction"):
                split_prediction = str(split_result["prediction"])
                split_applied_runtime = max(
                    split_applied_runtime, int(split_result.get("split_applied", 0))
                )
                split_marker_survived = max(
                    split_marker_survived, int(split_result.get("split_marker_survived", 0))
                )
                if alert_min is not None:
                    immutable_anchor_events = _extract_immutable_anchor_events(
                        _extract_events_from_prediction(split_prediction), alert_min
                    )
                    alert_state_summary = _compute_alert_state_summary(
                        immutable_anchor_events, alert_min
                    )
                    next_actionable_min = _resolve_next_actionable_min(
                        alert_state_summary, alert_min
                    )
                    next_actionable_location = str(
                        alert_state_summary.get("next_actionable_location", "")
                    )
                split_locked_prefix = locked_prefix_events
                if prefix_lock_enabled and alert_min is not None:
                    split_locked_prefix = _extract_prefix_events(
                        _extract_events_from_prediction(split_prediction), alert_min
                    )
                    if split_locked_prefix:
                        locked_prefix_events = split_locked_prefix
                        _refresh_locked_prefix_snapshot()
                if (
                    split_enabled
                    and immutable_anchor_events
                    and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                ):
                    structural = _apply_boundary_structural_guards(
                        raw_prediction=split_prediction,
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                        alert_min=alert_min,
                        run_cfg=run_cfg,
                    )
                    _consume_structural(structural)
                    split_prediction = str(structural.get("prediction") or split_prediction)
                    suffix_only_output_ok = max(
                        suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                    )
                    prefix_edit_attempt_detected = max(
                        prefix_edit_attempt_detected,
                        int(structural.get("prefix_edit_attempt_detected", 0)),
                    )
                    prefix_edit_ignored_count += int(
                        structural.get("prefix_edit_ignored_count", 0)
                    )
                    if str(structural.get("next_actionable_time", "")):
                        next_actionable_time = str(structural.get("next_actionable_time", ""))
                    if str(structural.get("suffix_first_event_time", "")):
                        suffix_first_event_time = str(
                            structural.get("suffix_first_event_time", "")
                        )
                _sync_locked_prefix_from_prediction(split_prediction)
                guarded_prediction, _, _ = _guard_prediction(
                    split_prediction, "SPLIT_FALLBACK"
                )
                if guarded_prediction:
                    candidate = _evaluate_candidate(guarded_prediction, split_locked_prefix)
                    _update_best_commit(guarded_prediction, candidate, "boundary_split_fallback")
                    repair_applied = 1
                    repair_modes.append("boundary_split")
                    force_split = bool(run_cfg.get("planning_boundary_split_force", True))
                    if force_split or _prefer_scored(
                        candidate, current_scored, selection_policy=selection_policy
                    ):
                        current_prediction = guarded_prediction
                        current_scored = candidate
                        _maybe_take_early_exit("SPLIT_FALLBACK")
                        if prefix_lock_enabled and alert_min is not None:
                            locked_prefix_events = split_locked_prefix
                            _refresh_locked_prefix_snapshot()
            if split_applied_runtime == 0 and bool(run_cfg.get("planning_boundary_split_force", True)):
                force_fill_attempts = max(
                    1, int(run_cfg.get("planning_force_split_fill_fallback_attempts", 2))
                )
                force_fill_attempts = min(force_fill_attempts, 2)
                for _ in range(force_fill_attempts):
                    forced_fill = _call_repair(
                        reason="FILL_EVENTS",
                        sample=sample,
                        raw_prediction=current_prediction,
                        model_cfg=model_cfg,
                        mode="fill_events",
                        violations=current_scored.get("violations", []),
                        immutable_anchor_events=immutable_anchor_events,
                        alert_state=alert_state_summary,
                    )
                    if not forced_fill:
                        break
                    split_result = _apply_boundary_split_compensation_json(
                        forced_fill,
                        sample.metadata,
                        alert_min,
                    )
                    split_attempted = max(
                        split_attempted, int(split_result.get("split_attempted", 0))
                    )
                    split_attempt_count += int(split_result.get("split_attempted", 0))
                    split_apply_mode = str(split_result.get("split_apply_mode", split_apply_mode))
                    split_failure_reason = str(
                        split_result.get("split_failure_reason", split_failure_reason)
                    )
                    split_failure_reasons.append(split_failure_reason)
                    split_candidate_found = max(
                        split_candidate_found, int(split_result.get("split_candidate_found", 0))
                    )
                    split_candidate_count += int(
                        split_result.get("split_candidate_count", 0)
                    )
                    cand_summary = split_result.get("crossing_candidate_summary")
                    if isinstance(cand_summary, dict) and cand_summary:
                        split_candidate_summary = cand_summary
                    if not split_result.get("prediction"):
                        continue
                    split_prediction = str(split_result["prediction"])
                    split_applied_runtime = max(
                        split_applied_runtime, int(split_result.get("split_applied", 0))
                    )
                    split_marker_survived = max(
                        split_marker_survived, int(split_result.get("split_marker_survived", 0))
                    )
                    if alert_min is not None:
                        immutable_anchor_events = _extract_immutable_anchor_events(
                            _extract_events_from_prediction(split_prediction), alert_min
                        )
                        alert_state_summary = _compute_alert_state_summary(
                            immutable_anchor_events, alert_min
                        )
                        next_actionable_min = _resolve_next_actionable_min(
                            alert_state_summary, alert_min
                        )
                        next_actionable_location = str(
                            alert_state_summary.get("next_actionable_location", "")
                        )
                    if (
                        split_enabled
                        and immutable_anchor_events
                        and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
                    ):
                        structural = _apply_boundary_structural_guards(
                            raw_prediction=split_prediction,
                            immutable_anchor_events=immutable_anchor_events,
                            alert_state=alert_state_summary,
                            alert_min=alert_min,
                            run_cfg=run_cfg,
                        )
                        _consume_structural(structural)
                        split_prediction = str(structural.get("prediction") or split_prediction)
                        suffix_only_output_ok = max(
                            suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                        )
                        prefix_edit_attempt_detected = max(
                            prefix_edit_attempt_detected,
                            int(structural.get("prefix_edit_attempt_detected", 0)),
                        )
                        prefix_edit_ignored_count += int(
                            structural.get("prefix_edit_ignored_count", 0)
                        )
                        if str(structural.get("next_actionable_time", "")):
                            next_actionable_time = str(structural.get("next_actionable_time", ""))
                        if str(structural.get("suffix_first_event_time", "")):
                            suffix_first_event_time = str(
                                structural.get("suffix_first_event_time", "")
                            )
                    _sync_locked_prefix_from_prediction(split_prediction)
                    guarded_prediction, _, _ = _guard_prediction(
                        split_prediction, "SPLIT_FORCE_FILL"
                    )
                    if not guarded_prediction:
                        continue
                    candidate = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                    _update_best_commit(
                        guarded_prediction, candidate, "boundary_split_force_fill"
                    )
                    repair_applied = 1
                    repair_modes.extend(["fill_events", "boundary_split"])
                    current_prediction = guarded_prediction
                    current_scored = candidate
                    _maybe_take_early_exit("SPLIT_FORCE_FILL")
                    if split_applied_runtime:
                        break

        if (prefix_lock_enabled and locked_prefix_events) or (
            split_enabled
            and immutable_anchor_events
            and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
        ):
            fixed = None
            if (
                split_enabled
                and immutable_anchor_events
                and bool(run_cfg.get("planning_boundary_structural_synthesis", True))
            ):
                structural = _apply_boundary_structural_guards(
                    raw_prediction=current_prediction,
                    immutable_anchor_events=immutable_anchor_events,
                    alert_state=alert_state_summary,
                    alert_min=alert_min,
                    run_cfg=run_cfg,
                )
                _consume_structural(structural)
                fixed = str(structural.get("prediction") or "")
                suffix_only_output_ok = max(
                    suffix_only_output_ok, int(structural.get("suffix_only_output_ok", 0))
                )
                prefix_edit_attempt_detected = max(
                    prefix_edit_attempt_detected,
                    int(structural.get("prefix_edit_attempt_detected", 0)),
                )
                prefix_edit_ignored_count += int(
                    structural.get("prefix_edit_ignored_count", 0)
                )
                if str(structural.get("next_actionable_time", "")):
                    next_actionable_time = str(structural.get("next_actionable_time", ""))
                if str(structural.get("suffix_first_event_time", "")):
                    suffix_first_event_time = str(
                        structural.get("suffix_first_event_time", "")
                    )
                if not fixed:
                    fixed = None
            if fixed is None:
                fixed = _apply_prefix_lock_json(current_prediction, locked_prefix_events, alert_min)
            if fixed:
                _sync_locked_prefix_from_prediction(fixed)
                guarded_prediction, _, _ = _guard_prediction(fixed, "FINAL_LOCK_SYNTHESIS")
                if not guarded_prediction:
                    guarded_prediction = None
                if not guarded_prediction:
                    fixed = None
            if fixed:
                fixed_scored = _evaluate_candidate(guarded_prediction, locked_prefix_events)
                _update_best_commit(guarded_prediction, fixed_scored, "final_lock_synthesis")
                if _prefer_scored(fixed_scored, current_scored, selection_policy=selection_policy):
                    current_prediction = guarded_prediction
                    current_scored = fixed_scored

        if split_enabled:
            _apply_boundary_invariant_gate()
            if (
                variant == "V3_PREFIX_SPLIT"
                and boundary_gate_passed == 0
                and bool(run_cfg.get("planning_boundary_gate_fallback_split_only", True))
            ):
                fallback_prediction, fallback_scored = _build_split_only_fallback(
                    current_prediction
                )
                if fallback_prediction and fallback_scored:
                    v3_fallback_to_split_only = 1
                    v3_fallback_reason = "|".join(boundary_gate_failed_reasons[-4:])
                    _update_best_commit(
                        fallback_prediction,
                        fallback_scored,
                        "gate_fallback_split_only",
                    )
                    if _prefer_scored(
                        fallback_scored, current_scored, selection_policy=selection_policy
                    ):
                        current_prediction = fallback_prediction
                        current_scored = fallback_scored
                        _maybe_take_early_exit("BOUNDARY_GATE_FALLBACK_SPLIT_ONLY")

        if bool(run_cfg.get("planning_commit_best_of", True)):
            current_violation = int(current_scored.get("violation_count", 0))
            best_plan_score_best = _weighted_commit_score(best_commit_scored)
            best_plan_score_last = _weighted_commit_score(current_scored)
            best_candidate_prediction = best_commit_prediction
            best_candidate_scored = best_commit_scored
            guarded_prediction, _, _ = _guard_prediction(
                best_candidate_prediction, "BESTOF_SELECT"
            )
            if not guarded_prediction:
                best_candidate_scored = current_scored
                best_candidate_prediction = current_prediction
            else:
                best_candidate_prediction = guarded_prediction
                best_candidate_scored = _evaluate_candidate(
                    best_candidate_prediction, locked_prefix_events
                )
                _update_best_commit(best_candidate_prediction, best_candidate_scored, "bestof_guarded")

            best_plan_score_best = _weighted_commit_score(best_candidate_scored)
            if _prefer_commit_candidate(best_candidate_scored, current_scored):
                best_plan_selected = 1
                best_plan_source = best_commit_source
                best_plan_violation_improvement_over_last = max(
                    0, current_violation - int(best_candidate_scored.get("violation_count", 0))
                )
                best_plan_score_improvement_over_last = max(
                    0, best_plan_score_last - best_plan_score_best
                )
                current_scored = best_candidate_scored
                current_prediction = best_candidate_prediction
            else:
                best_plan_selected = 0
                best_plan_source = "final"
                best_plan_violation_improvement_over_last = 0
                best_plan_score_improvement_over_last = 0
                best_plan_score_best = best_plan_score_last

        if split_enabled and bool(
            run_cfg.get("planning_boundary_gate_recheck_after_bestof", True)
        ):
            boundary_gate_passed = 0
            boundary_gate_pass_after_fix_iter = -1
            _apply_boundary_invariant_gate()
            if (
                variant == "V3_PREFIX_SPLIT"
                and boundary_gate_passed == 0
                and bool(run_cfg.get("planning_boundary_gate_fallback_split_only", True))
            ):
                fallback_prediction, fallback_scored = _build_split_only_fallback(
                    current_prediction
                )
                if fallback_prediction and fallback_scored:
                    v3_fallback_to_split_only = 1
                    v3_fallback_reason = "|".join(boundary_gate_failed_reasons[-4:])
                    _update_best_commit(
                        fallback_prediction,
                        fallback_scored,
                        "gate_fallback_split_only_post_bestof",
                    )
                    if _prefer_scored(
                        fallback_scored, current_scored, selection_policy=selection_policy
                    ):
                        current_prediction = fallback_prediction
                        current_scored = fallback_scored

        if split_enabled:
            final_events = _extract_events_from_prediction(current_prediction)
            pre_mark, post_mark = _split_marker_state(final_events)
            has_split_markers = int(pre_mark == 1 and post_mark == 1)
            if (
                has_split_markers == 0
                and bool(run_cfg.get("planning_split_marker_retry_on_finalize", True))
                and alert_min is not None
            ):
                split_retry_triggered = 1
                split_marker_lost_stage = split_marker_lost_stage or "FINAL_RECONCILE"
                retry_split = _apply_boundary_split_compensation_json(
                    current_prediction, sample.metadata, alert_min
                )
                if retry_split.get("prediction"):
                    retried_prediction = str(retry_split["prediction"])
                    boundary_fixed, boundary_metrics = _ensure_boundary_event_json(
                        retried_prediction,
                        alert_min=alert_min,
                        next_actionable_min=next_actionable_min,
                        next_actionable_location=next_actionable_location,
                        next_actionable_actor=str(
                            alert_state_summary.get("next_actionable_actor", "")
                        ),
                        run_cfg=run_cfg,
                    )
                    _consume_structural(boundary_metrics)
                    if boundary_fixed:
                        retried_prediction = boundary_fixed
                    canon_fixed, canon_metrics = _canonicalize_boundary_split_markers_json(
                        retried_prediction, alert_min
                    )
                    _consume_structural(canon_metrics)
                    if canon_fixed:
                        retried_prediction = canon_fixed
                    post_fixed, post_metrics = _post_boundary_monotonic_clamp_json(
                        retried_prediction,
                        alert_min=alert_min,
                        next_actionable_min=next_actionable_min,
                    )
                    _consume_structural(post_metrics)
                    if post_fixed:
                        retried_prediction = post_fixed
                    _sync_locked_prefix_from_prediction(retried_prediction)
                    guarded_prediction, _, _ = _guard_prediction(
                        retried_prediction, "FINAL_SPLIT_RETRY"
                    )
                    if guarded_prediction:
                        candidate = _evaluate_candidate(
                            guarded_prediction, locked_prefix_events
                        )
                        _update_best_commit(
                            guarded_prediction, candidate, "final_split_retry"
                        )
                        if _prefer_scored(
                            candidate, current_scored, selection_policy=selection_policy
                        ):
                            current_prediction = guarded_prediction
                            current_scored = candidate
                            final_events = _extract_events_from_prediction(
                                current_prediction
                            )
                            pre_mark, post_mark = _split_marker_state(final_events)
                            has_split_markers = int(pre_mark == 1 and post_mark == 1)
                            if has_split_markers == 1:
                                split_marker_lost_stage = ""

            split_applied_runtime = max(split_applied_runtime, has_split_markers)
            split_marker_survived = has_split_markers
            if not has_split_markers:
                split_apply_mode = "FAILED_CONFLICT_WITH_LOCK"
                split_failure_reason = "FAILED_CONFLICT_WITH_LOCK"
                split_failure_reasons.append("FAILED_CONFLICT_WITH_LOCK")

        scored = current_scored
        raw_response = current_prediction

    final_events = _extract_events_from_prediction(raw_response) if status == "ok" else []
    if status == "ok" and not suffix_first_event_time and final_events:
        first_suffix: int | None = None
        for event in final_events:
            if _is_split_marker_event(event):
                continue
            start_m = _to_minutes(str(event.get("start", "")))
            if start_m is None:
                continue
            if next_actionable_min is None or start_m >= next_actionable_min:
                if first_suffix is None or start_m < first_suffix:
                    first_suffix = start_m
        if first_suffix is not None:
            suffix_first_event_time = _to_hhmm(first_suffix)

    if status == "ok" and not next_actionable_time and next_actionable_min is not None:
        next_actionable_time = _to_hhmm(next_actionable_min)

    if status == "ok" and alert_min is not None:
        pre_delta, post_delta = _boundary_alignment_deltas(final_events, alert_min)
        boundary_pre_end_minus_alert_min = int(pre_delta)
        boundary_post_start_minus_alert_min = int(post_delta)
        final_prefix_extracted = _extract_prefix_events(final_events, alert_min)
        final_prefix_hash = _prefix_hash(final_prefix_extracted) if final_prefix_extracted else ""
        if not locked_prefix_snapshot and locked_prefix_events:
            locked_prefix_snapshot = _dedup_sort_events(copy.deepcopy(locked_prefix_events))
            locked_prefix_hash = _prefix_hash(locked_prefix_snapshot)
        immutable_diff_type = _classify_immutable_diff_type(
            locked_prefix_snapshot, final_prefix_extracted
        )
        if locked_prefix_events:
            scorer_view_locked_hash, scorer_view_locked_monotonic = _scorer_view_hash(
                locked_prefix_events, alert_min
            )
        scorer_view_final_hash, scorer_view_final_monotonic = _scorer_view_hash(
            final_events, alert_min
        )
        guard_view_hash_match = int(
            bool(locked_prefix_hash) and bool(final_prefix_hash) and locked_prefix_hash == final_prefix_hash
        )
        scorer_view_hash_match = int(
            bool(scorer_view_locked_hash)
            and bool(scorer_view_final_hash)
            and scorer_view_locked_hash == scorer_view_final_hash
        )
        (
            prefix_contains_disruption_terms,
            split_pre_contains_disruption_terms,
        ) = _prefix_disruption_flags(final_events, alert_min)

    state_consistency_failure_reason = _infer_state_consistency_failure_reason(
        events=final_events,
        next_actionable_min=next_actionable_min,
        next_actionable_location=next_actionable_location,
        state_check_applicable=int(scored.get("diagnostics", {}).get("state_check_applicable", 0)),
        state_at_alert_consistent=int(
            scored.get("diagnostics", {}).get("state_at_alert_consistent", 0)
        ),
    )

    duration = time.perf_counter() - started
    tx_metrics = _collect_tx_metrics(metadata)
    postproc_time_total = max(0.0, duration - llm_time_total - validator_time_total)

    state_corruption = int(
        tx_metrics["tx_rollbacks"] > 0 and scored["success"] == 0 and sample.task_type == "planning"
    )
    gate_failed_reason_counts: dict[str, int] = {}
    for reason in boundary_gate_failed_reasons:
        gate_failed_reason_counts[reason] = gate_failed_reason_counts.get(reason, 0) + 1

    return {
        "variant": variant,
        "seed": seed,
        "sample_id": sample.sample_id,
        "track": sample.track,
        "source": sample.source,
        "task_type": sample.task_type,
        "status": status,
        "error": error_message,
        "success": scored["success"],
        "score": scored["score"],
        "violation_count": scored["violation_count"],
        "violations": scored["violations"],
        "prediction": raw_response,
        "ground_truth": sample.answer,
        "metadata": sample.metadata,
        "usage": usage,
        "wall_time_sec": duration,
        "root_completion_time_sec": root_completion_time_sec,
        "state_corruption": state_corruption,
        "repair_applied": repair_applied,
        "repair_count": len(repair_modes),
        "repair_modes": repair_modes,
        "best_plan_selected": best_plan_selected,
        "best_plan_source": best_plan_source,
        "best_plan_violation_improvement_over_last": best_plan_violation_improvement_over_last,
        "best_plan_score_best": best_plan_score_best,
        "best_plan_score_last": best_plan_score_last,
        "best_plan_score_improvement_over_last": best_plan_score_improvement_over_last,
        "suffix_only_output_ok": suffix_only_output_ok,
        "prefix_edit_attempt_detected": prefix_edit_attempt_detected,
        "prefix_edit_ignored_count": prefix_edit_ignored_count,
        "next_actionable_time": next_actionable_time,
        "suffix_first_event_time": suffix_first_event_time,
        "suffix_first_event_time_before": suffix_first_event_time_before,
        "suffix_first_event_time_after": suffix_first_event_time_after,
        "timeline_norm_applied": timeline_norm_applied,
        "timeline_norm_total_shift_minutes": timeline_norm_total_shift_minutes,
        "timeline_norm_overlap_fixes_count": timeline_norm_overlap_fixes_count,
        "timeline_norm_location_fix_applied": timeline_norm_location_fix_applied,
        "boundary_canonicalization_applied": boundary_canonicalization_applied,
        "post_boundary_monotonic_fix_applied": post_boundary_monotonic_fix_applied,
        "post_boundary_monotonic_fix_count": post_boundary_monotonic_fix_count,
        "post_boundary_total_shift_minutes": post_boundary_total_shift_minutes,
        "post_boundary_first_start_before": post_boundary_first_start_before,
        "post_boundary_first_start_after": post_boundary_first_start_after,
        "missing_boundary_event_detected": missing_boundary_event_detected,
        "missing_boundary_event_autofixed": missing_boundary_event_autofixed,
        "missing_boundary_event_autofix_minutes_pre": missing_boundary_event_autofix_minutes_pre,
        "missing_boundary_event_autofix_minutes_post": missing_boundary_event_autofix_minutes_post,
        "immutable_scope_sanitize_applied": immutable_scope_sanitize_applied,
        "immutable_scope_terms_removed_count": immutable_scope_terms_removed_count,
        "immutable_scope_contains_disruption_terms_before": immutable_scope_contains_disruption_terms_before,
        "immutable_scope_contains_disruption_terms_after": immutable_scope_contains_disruption_terms_after,
        "boundary_pre_end_minus_alert_min": boundary_pre_end_minus_alert_min,
        "boundary_post_start_minus_alert_min": boundary_post_start_minus_alert_min,
        "llm_call_count": llm_call_count,
        "llm_call_count_capped": llm_call_count_capped,
        "llm_call_reason_counts": llm_call_reason_counts,
        "llm_call_reason_plan": int(llm_call_reason_counts.get("PLAN", 0)),
        "llm_call_reason_format": int(llm_call_reason_counts.get("FORMAT", 0)),
        "llm_call_reason_fill_events": int(llm_call_reason_counts.get("FILL_EVENTS", 0)),
        "llm_call_reason_boundary_crossing": int(llm_call_reason_counts.get("BOUNDARY_CROSSING", 0)),
        "llm_call_reason_constraint": int(llm_call_reason_counts.get("CONSTRAINT", 0)),
        "llm_call_reason_photo_time": int(llm_call_reason_counts.get("PHOTO_TIME", 0)),
        "llm_call_reason_tailor_hours": int(llm_call_reason_counts.get("TAILOR_HOURS", 0)),
        "llm_time_total_sec": llm_time_total,
        "validator_time_total_sec": validator_time_total,
        "postproc_time_total_sec": postproc_time_total,
        "timeline_norm_time_total_sec": timeline_norm_time_total,
        "boundary_gate_passed": boundary_gate_passed,
        "boundary_gate_pass_after_fix_iter": boundary_gate_pass_after_fix_iter,
        "boundary_gate_strict_success_last": gate_strict_success_last,
        "boundary_gate_strict_violation_count_last": gate_strict_violation_count_last,
        "boundary_gate_failed_reasons": boundary_gate_failed_reasons,
        "boundary_gate_failed_reason_counts": gate_failed_reason_counts,
        "boundary_gate_failed_reason_non_monotonic": int(
            boundary_gate_failed_reasons.count("NON_MONOTONIC")
        ),
        "boundary_gate_failed_reason_missing_boundary": int(
            boundary_gate_failed_reasons.count("BOUNDARY_MISSING")
        ),
        "boundary_gate_failed_reason_state_mismatch": int(
            boundary_gate_failed_reasons.count("STATE_MISMATCH")
        ),
        "boundary_gate_failed_reason_immutable_terms": int(
            boundary_gate_failed_reasons.count("IMMUTABLE_TERMS")
        ),
        "boundary_gate_failed_reason_photo_time": int(
            boundary_gate_failed_reasons.count("PHOTO_TIME_EXCEEDED")
        ),
        "boundary_gate_failed_reason_marker_lost": int(
            boundary_gate_failed_reasons.count("MARKER_LOST")
        ),
        "boundary_gate_fix_counts": boundary_gate_fix_counts,
        "boundary_gate_fix_split_or_comp": int(
            boundary_gate_fix_counts.get("SPLIT_OR_COMPENSATION", 0)
        ),
        "boundary_gate_fix_boundary_autofix": int(
            boundary_gate_fix_counts.get("BOUNDARY_AUTOFIX", 0)
        ),
        "boundary_gate_fix_state_shift": int(
            boundary_gate_fix_counts.get("STATE_SHIFT_FIX", 0)
        ),
        "boundary_gate_fix_canonicalize": int(
            boundary_gate_fix_counts.get("BOUNDARY_CANONICALIZE", 0)
        ),
        "boundary_gate_fix_monotonic": int(
            boundary_gate_fix_counts.get("POST_BOUNDARY_MONOTONIC", 0)
        ),
        "boundary_gate_fix_immutable_sanitize": int(
            boundary_gate_fix_counts.get("IMMUTABLE_SANITIZE", 0)
        ),
        "boundary_gate_fix_photo_time_repair": int(
            boundary_gate_fix_counts.get("PHOTO_TIME_REPAIR", 0)
        ),
        "photo_time_violation_detected": photo_time_violation_detected,
        "photo_time_repair_triggered": photo_time_repair_triggered,
        "photo_time_repair_success": photo_time_repair_success,
        "photo_time_repair_method": photo_time_repair_method,
        "photo_time_repair_search_nodes": photo_time_repair_search_nodes,
        "photo_time_repair_wall_ms": photo_time_repair_wall_ms,
        "early_exit_taken": early_exit_taken,
        "early_exit_stage": early_exit_stage,
        "split_retry_triggered": split_retry_triggered,
        "split_marker_lost_stage": split_marker_lost_stage,
        "immutable_terms_detected": immutable_terms_detected,
        "immutable_terms_locations": immutable_terms_locations,
        "immutable_terms_fix_action": immutable_terms_fix_action,
        "v3_fallback_to_split_only": v3_fallback_to_split_only,
        "v3_fallback_reason": v3_fallback_reason,
        "state_consistency_failure_reason": state_consistency_failure_reason,
        "immutability_guard_triggered": immutability_guard_triggered,
        "immutability_guard_failure_stage": immutability_guard_failure_stage,
        "immutability_guard_failure_stages": immutability_guard_failure_stages,
        "num_candidates_generated": num_candidates_generated,
        "num_candidates_scored": num_candidates_scored,
        "num_candidates_discarded_by_guard": num_candidates_discarded_by_guard,
        "locked_prefix_snapshot": locked_prefix_snapshot,
        "locked_prefix_hash": locked_prefix_hash,
        "final_prefix_extracted": final_prefix_extracted,
        "final_prefix_hash": final_prefix_hash,
        "guard_view_hash_match": guard_view_hash_match,
        "scorer_view_locked_hash": scorer_view_locked_hash,
        "scorer_view_final_hash": scorer_view_final_hash,
        "scorer_view_locked_monotonic": scorer_view_locked_monotonic,
        "scorer_view_final_monotonic": scorer_view_final_monotonic,
        "scorer_view_hash_match": scorer_view_hash_match,
        "immutable_diff_type": immutable_diff_type,
        "prefix_contains_disruption_terms": prefix_contains_disruption_terms,
        "split_pre_contains_disruption_terms": split_pre_contains_disruption_terms,
        "split_attempted": split_attempted,
        "split_attempt_count": split_attempt_count,
        "split_applied_runtime": split_applied_runtime,
        "split_marker_survived": split_marker_survived,
        "split_apply_mode": split_apply_mode,
        "split_failure_reason": split_failure_reason,
        "split_failure_reasons": split_failure_reasons,
        "split_candidate_found": split_candidate_found,
        "split_candidate_count": split_candidate_count,
        "split_candidate_summary": split_candidate_summary,
        "runtime_evaluator_mode": runtime_eval_mode,
        "runtime_evaluator_version": runtime_eval_version,
        "runtime_validator_mode": runtime_validator_mode,
        "validator_version": validator_version,
        "valid_json": int(scored.get("diagnostics", {}).get("valid_json", 0)),
        "events_count": int(scored.get("diagnostics", {}).get("events_count", 0)),
        "non_empty_events": int(scored.get("diagnostics", {}).get("non_empty_events", 0)),
        "disruption_required": int(scored.get("diagnostics", {}).get("disruption_required", 0)),
        "disruption_applicable": int(scored.get("diagnostics", {}).get("disruption_applicable", 0)),
        "disruption_applied": int(scored.get("diagnostics", {}).get("disruption_applied", 0)),
        "partial_compensation_applicable": int(
            scored.get("diagnostics", {}).get("partial_compensation_applicable", 0)
        ),
        "partial_compensation_ok": int(
            scored.get("diagnostics", {}).get("partial_compensation_ok", 0)
        ),
        "crossing_split_applicable": int(
            scored.get("diagnostics", {}).get("crossing_split_applicable", 0)
        ),
        "crossing_split_applied": int(
            scored.get("diagnostics", {}).get("crossing_split_applied", 0)
        ),
        "immutable_check_applicable": int(
            scored.get("diagnostics", {}).get("immutable_check_applicable", 0)
        ),
        "immutable_prefix_ok": int(scored.get("diagnostics", {}).get("immutable_prefix_ok", 0)),
        "immutable_prefix_after_split_ok": int(
            scored.get("diagnostics", {}).get("immutable_prefix_after_split_ok", 0)
        ),
        "state_check_applicable": int(
            scored.get("diagnostics", {}).get("state_check_applicable", 0)
        ),
        "state_at_alert_consistent": int(
            scored.get("diagnostics", {}).get("state_at_alert_consistent", 0)
        ),
        "prefix_lock_used": int(scored.get("diagnostics", {}).get("prefix_lock_used", 0)),
        **tx_metrics,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
