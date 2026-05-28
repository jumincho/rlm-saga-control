"""Scoring and validation utilities — the load-bearing arbitration layer.

Single source of truth for what counts as a successful plan repair, used
both live by the Saga validator (`build_rule_validator`) and offline by
the closure-report re-scorer (`score_prediction`). Behavior pivots on a
*policy* string:

- `runtime_v3` — what the Saga layer uses live; lenient on hard-fails so
  recoverable issues do not knock out the agent.
- `runtime_p8_hard_v1` — runtime policy with all three hard constraints
  enforced; used for P8 (boundary-crossing) samples where the partial-
  compensation invariant is the whole point.
- `strict_v1` — the offline "is it actually correct" judge; enforces
  disruption handling, immutable prefix, and state consistency hard.
- `relaxed_v1` — strict on disruption handling, soft on immutable prefix
  and state, used to sanity-check that strict failures are not just the
  strict policy being severe.

Loose vs strict scoring refers to whether those three constraint families
are hard-failed. For planning tasks the scorer:

1. Parses the model's JSON (with REPL-output recovery fallbacks); empty /
   unparseable plans short-circuit to a single violation.
2. Checks per-event well-formedness (required fields, valid times).
3. Checks deadline / open-close constraints from `constraints`.
4. If there are `disruptions`: builds per-actor timelines, decides which
   segments are "post-disruption" or "crossing the disruption boundary",
   checks whether the disruption was actually reflected (`disruption_applied`),
   whether the boundary-crossing segment was split into a pre/post pair
   (`crossing_split_applied`), whether partial compensation matches the
   pre-immutable / post-extended pattern, whether the immutable prefix
   (events ending at or before the alert minute) survived unchanged, and
   whether per-actor windows are monotonic across the alert boundary.

The diagnostics dict produced here is what the runner stamps onto every
output row; analysis modules then pivot off those fields.

For MCQ tasks the scorer extracts a single A/B/C/D letter; for generic
QA it does exact-match plus tolerant substring match against `|`-
separated multi-answer targets.

`build_rule_validator` wraps the same machinery for the Saga layer's
in-loop accept / augment / reject decision: minor violations (≤ 2) come
back as `augment` so the model can revise; more get a `reject`.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from exp.bench.schema import parse_json_response
from exp.bench.types import BenchmarkSample


TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
SCORER_VERSION = "scorer_v7_20260226"
RUNTIME_POLICY = "runtime_v3"
RUNTIME_P8_HARD_POLICY = "runtime_p8_hard_v1"
STRICT_POLICY = "strict_v1"
RELAXED_POLICY = "relaxed_v1"


def _resolve_policy(policy: str) -> dict[str, bool]:
    if policy == RUNTIME_P8_HARD_POLICY:
        return {
            "enforce_disruption_hard": True,
            "enforce_immutable_hard": True,
            "enforce_state_hard": True,
        }
    if policy == STRICT_POLICY:
        return {
            "enforce_disruption_hard": True,
            "enforce_immutable_hard": True,
            "enforce_state_hard": True,
        }
    if policy == RELAXED_POLICY:
        return {
            "enforce_disruption_hard": True,
            "enforce_immutable_hard": False,
            "enforce_state_hard": False,
        }
    # Runtime validator policy should avoid brittle hard-fails.
    return {
        "enforce_disruption_hard": False,
        "enforce_immutable_hard": False,
        "enforce_state_hard": False,
    }


def get_policy_version(policy: str) -> str:
    return f"{SCORER_VERSION}:{policy}"


def _to_minutes(hhmm: str) -> int | None:
    if not isinstance(hhmm, str) or not TIME_RE.match(hhmm.strip()):
        return None
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _normalize_qa_span(value: str) -> str:
    value = _normalize_text(value)
    value = re.sub(r"^(final answer|answer|prediction)\s*:\s*", "", value)
    return value.strip(" .,:;\"'")


def _extract_mcq_choice(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    return text.strip().upper()[:1]


def _planning_diagnostics_template(disruptions: list[dict[str, Any]] | None) -> dict[str, Any]:
    has_disruption = bool(disruptions)
    return {
        "valid_json": 1,
        "events_count": 0,
        "non_empty_events": 0,
        "disruption_required": int(has_disruption),
        "disruption_applicable": 0,
        "disruption_applied": 0,
        "partial_compensation_applicable": 0,
        "partial_compensation_ok": 0 if has_disruption else 1,
        "crossing_split_applicable": 0,
        "crossing_split_applied": 0,
        "immutable_check_applicable": 0,
        "immutable_prefix_ok": 0 if has_disruption else 1,
        "immutable_prefix_after_split_ok": 0 if has_disruption else 1,
        "state_check_applicable": 0,
        "state_at_alert_consistent": 0 if has_disruption else 1,
        "prefix_lock_used": 0,
    }


def _disruption_keywords(disruptions: list[dict[str, Any]]) -> list[str]:
    keywords = {
        "delay",
        "delayed",
        "closure",
        "closed",
        "disruption",
        "reroute",
        "re-route",
        "traffic",
        "detour",
        "reschedule",
        "late",
    }
    for disruption in disruptions:
        for key in ("type", "flight"):
            value = str(disruption.get(key, "")).strip().lower()
            if not value:
                continue
            keywords.add(value)
            for token in re.split(r"[-_/:\s]+", value):
                if len(token) >= 3:
                    keywords.add(token)
    return sorted(keywords)


def _tokenize(value: str) -> list[str]:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", value.lower()) if len(tok) >= 3]
    return tokens


def _event_blob(event: dict[str, Any]) -> str:
    return _normalize_text(
        f"{event.get('who', '')} {event.get('what', '')} {event.get('location', '')} {event.get('notes', '')}"
    )


def _canonical_event(event: dict[str, Any]) -> tuple[Any, ...]:
    who = _normalize_text(str(event.get("who", "")))
    what = _normalize_text(str(event.get("what", "")))
    location = _normalize_text(str(event.get("location", "")))
    start = _to_minutes(str(event.get("start", "")))
    end = _to_minutes(str(event.get("end", "")))
    return (start, end, who, what, location)


def _canonicalize_events(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    canon = [_canonical_event(event) for event in events if isinstance(event, dict)]
    return sorted(canon, key=lambda x: (x[0] if x[0] is not None else 10**9, x[1] if x[1] is not None else 10**9, x[2], x[3], x[4]))


def _resolve_alert_min(disruptions: list[dict[str, Any]]) -> int | None:
    candidates: list[int] = []
    for disruption in disruptions:
        for key in ("start_time", "new_arrival_time", "arrival_time"):
            minute = _to_minutes(str(disruption.get(key, "")))
            if minute is not None:
                candidates.append(minute)
                break
    return min(candidates) if candidates else None


def _event_tokens(event: dict[str, Any]) -> set[str]:
    return set(_tokenize(_event_blob(event)))


def _route_tokens(route: str) -> set[str]:
    return set(_tokenize(route))


def _resolve_location_code(location: str, metadata: dict[str, Any] | None) -> str | None:
    if not location:
        return None
    loc = _normalize_text(location)
    locations = (metadata or {}).get("locations", {})
    if not isinstance(locations, dict):
        return None

    for code, name in locations.items():
        code_norm = _normalize_text(str(code))
        name_norm = _normalize_text(str(name))
        if loc == code_norm or loc == name_norm:
            return code_norm
    for code, name in locations.items():
        code_norm = _normalize_text(str(code))
        name_norm = _normalize_text(str(name))
        if loc in name_norm or name_norm in loc:
            return code_norm
    return None


def _lookup_travel_time(
    from_location: str,
    to_location: str,
    metadata: dict[str, Any] | None,
) -> int | None:
    travel = (metadata or {}).get("travel_times", {})
    if not isinstance(travel, dict):
        return None

    from_code = _resolve_location_code(from_location, metadata) or _normalize_text(from_location)
    to_code = _resolve_location_code(to_location, metadata) or _normalize_text(to_location)
    direct = [f"{from_code}-{to_code}", f"{to_code}-{from_code}"]
    for key in direct:
        if key in travel and isinstance(travel[key], (int, float)):
            return int(travel[key])

    from_tokens = set(_tokenize(from_location))
    to_tokens = set(_tokenize(to_location))
    for key, value in travel.items():
        if not isinstance(value, (int, float)):
            continue
        kt = set(_tokenize(str(key)))
        if from_tokens and to_tokens and from_tokens.intersection(kt) and to_tokens.intersection(kt):
            return int(value)
    return None


def _build_actor_timelines(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    timelines: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None:
            continue
        who_raw = str(event.get("who", "")).strip()
        actors = [a.strip().lower() for a in re.split(r"[,&/]+", who_raw) if a.strip()]
        if not actors:
            actors = ["_unknown"]
        for actor in actors:
            timelines.setdefault(actor, []).append(event)

    for actor in timelines:
        timelines[actor] = sorted(
            timelines[actor],
            key=lambda ev: (
                _to_minutes(str(ev.get("start", ""))) or 10**9,
                _to_minutes(str(ev.get("end", ""))) or 10**9,
            ),
        )
    return timelines


def _analyze_disruption_segments(
    events: list[dict[str, Any]],
    disruptions: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    alert_min: int | None,
) -> dict[str, int]:
    if not disruptions or alert_min is None:
        return {
            "relevant_segments": 0,
            "applied_segments": 0,
            "crossing_segments": 0,
            "crossing_compensated": 0,
        }

    timelines = _build_actor_timelines(events)
    relevant_segments = 0
    applied_segments = 0
    crossing_segments = 0
    crossing_compensated = 0

    disruptions_with_time: list[tuple[dict[str, Any], int, int]] = []
    for disruption in disruptions:
        d_start = _to_minutes(str(disruption.get("start_time", "")))
        if d_start is None:
            continue
        delta = disruption.get("delay", disruption.get("duration", 0))
        if not isinstance(delta, (int, float)):
            delta = 0
        disruptions_with_time.append((disruption, d_start, int(delta)))

    boundary_route = _normalize_text(str((metadata or {}).get("boundary_route", "")))
    boundary_depart_min = _to_minutes(str((metadata or {}).get("boundary_departure_hint", "")))
    boundary_end_min = _to_minutes(str((metadata or {}).get("boundary_planned_end_hint", "")))
    boundary_baseline = None
    if boundary_depart_min is not None and boundary_end_min is not None and boundary_end_min > boundary_depart_min:
        boundary_baseline = boundary_end_min - boundary_depart_min
    require_boundary = bool((metadata or {}).get("require_boundary_crossing", False))

    for actor_events in timelines.values():
        for idx in range(len(actor_events) - 1):
            prev_ev = actor_events[idx]
            next_ev = actor_events[idx + 1]
            seg_start = _to_minutes(str(prev_ev.get("end", "")))
            seg_end = _to_minutes(str(next_ev.get("start", "")))
            if seg_start is None or seg_end is None or seg_end <= seg_start:
                continue

            from_loc = str(prev_ev.get("location", ""))
            to_loc = str(next_ev.get("location", ""))
            if _normalize_text(from_loc) == _normalize_text(to_loc):
                continue

            segment_tokens = set()
            segment_tokens.update(_event_tokens(prev_ev))
            segment_tokens.update(_event_tokens(next_ev))
            segment_tokens.update(_tokenize(from_loc))
            segment_tokens.update(_tokenize(to_loc))

            for disruption, d_start, delta in disruptions_with_time:
                route = str(disruption.get("route", ""))
                r_tokens = _route_tokens(route)
                if r_tokens and not segment_tokens.intersection(r_tokens):
                    continue

                is_post = seg_start >= d_start
                is_crossing = seg_start < d_start < seg_end
                if not (is_post or is_crossing):
                    continue

                relevant_segments += 1
                if is_crossing:
                    crossing_segments += 1

                baseline = _lookup_travel_time(from_loc, to_loc, metadata)
                if baseline is None and boundary_baseline is not None:
                    dis_route = _normalize_text(str(disruption.get("route", "")))
                    if boundary_route and dis_route == boundary_route:
                        baseline = boundary_baseline
                observed = seg_end - seg_start
                kw_ok = int(
                    any(k in _event_blob(prev_ev) for k in _disruption_keywords([disruption]))
                    or any(k in _event_blob(next_ev) for k in _disruption_keywords([disruption]))
                )
                duration_ok = 0
                if baseline is not None and delta > 0:
                    required_extra = delta
                    if is_crossing and seg_end > seg_start:
                        affected = seg_end - d_start
                        span = seg_end - seg_start
                        required_extra = max(1, int(round(delta * (affected / span))))
                    duration_ok = int((observed - baseline) >= max(1, required_extra))

                applied_ok = duration_ok if require_boundary else (kw_ok or duration_ok)
                if applied_ok:
                    applied_segments += 1
                    if is_crossing:
                        crossing_compensated += 1
                break

    return {
        "relevant_segments": relevant_segments,
        "applied_segments": applied_segments,
        "crossing_segments": crossing_segments,
        "crossing_compensated": crossing_compensated,
    }


def _disruption_route_terms(disruptions: list[dict[str, Any]], metadata: dict[str, Any] | None) -> list[str]:
    terms: set[str] = set()
    for disruption in disruptions:
        route = str(disruption.get("route", "")).strip().lower()
        for tok in _tokenize(route):
            terms.add(tok)

    locations = (metadata or {}).get("locations", {})
    if isinstance(locations, dict):
        for key, value in locations.items():
            for tok in _tokenize(str(key)):
                terms.add(tok)
            for tok in _tokenize(str(value)):
                terms.add(tok)
    return sorted(terms)


def _is_disruption_applied(
    post_events: list[dict[str, Any]],
    disruptions: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    segment_analysis: dict[str, int] | None = None,
) -> int:
    if segment_analysis and segment_analysis.get("relevant_segments", 0) > 0:
        return int(segment_analysis.get("applied_segments", 0) > 0)
    if not post_events:
        return 0

    route_terms = _disruption_route_terms(disruptions, metadata)
    keywords = _disruption_keywords(disruptions)

    for event in post_events:
        blob = _event_blob(event)
        has_disruption_kw = any(keyword in blob for keyword in keywords)
        has_route_kw = any(term in blob for term in route_terms) if route_terms else True
        if has_disruption_kw and has_route_kw:
            return 1

    # Fallback: any explicit disruption marker in post-disruption notes.
    for event in post_events:
        blob = _event_blob(event)
        if any(marker in blob for marker in ["traffic", "delay", "closure", "reroute", "detour", "reschedule"]):
            return 1

    return 0


def _immutable_prefix_ok(
    pre_events: list[dict[str, Any]],
    disruptions: list[dict[str, Any]],
    locked_prefix_events: list[dict[str, Any]] | None = None,
) -> int:
    if not pre_events:
        return 0

    if not _events_time_monotonic(pre_events):
        return 0

    if isinstance(locked_prefix_events, list) and locked_prefix_events:
        expected = _canonicalize_events(locked_prefix_events)
        actual = _canonicalize_events(pre_events)
        return int(actual == expected)

    keywords = _disruption_keywords(disruptions)
    for event in pre_events:
        if any(keyword in _event_blob(event) for keyword in keywords):
            return 0
    return 1


def _state_consistent_at_alert(
    events: list[dict[str, Any]],
    alert_min: int | None,
    locked_prefix_events: list[dict[str, Any]] | None = None,
) -> int:
    if alert_min is None:
        return int(_events_time_monotonic(events))

    if not _events_time_monotonic(events):
        return 0

    if isinstance(locked_prefix_events, list) and locked_prefix_events:
        pre_events = [
            event
            for event in events
            if _to_minutes(str(event.get("end", ""))) is not None
            and _to_minutes(str(event.get("end", ""))) <= alert_min
        ]
        if _canonicalize_events(pre_events) != _canonicalize_events(locked_prefix_events):
            return 0

    by_actor: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None:
            continue
        who_raw = str(event.get("who", "")).strip()
        actors = [a.strip().lower() for a in re.split(r"[,&/]+", who_raw) if a.strip()]
        if not actors:
            actors = ["_unknown"]
        for actor in actors:
            by_actor.setdefault(actor, []).append((start_m, end_m))

    for windows in by_actor.values():
        windows.sort(key=lambda x: (x[0], x[1]))
        crossing = 0
        prev_end = None
        for start_m, end_m in windows:
            if prev_end is not None and start_m < prev_end:
                return 0
            if start_m < alert_min < end_m:
                crossing += 1
                if crossing > 1:
                    return 0
            prev_end = end_m
    return 1


def _crossing_split_markers(events: list[dict[str, Any]]) -> tuple[int, int]:
    has_pre = 0
    has_post = 0
    for event in events:
        notes = _normalize_text(str(event.get("notes", "")))
        what = _normalize_text(str(event.get("what", "")))
        blob = f"{notes} {what}"
        if "boundary_split_pre" in blob:
            has_pre = 1
        if "boundary_split_post" in blob:
            has_post = 1
    return has_pre, has_post


def _split_partial_compensation_correct(events: list[dict[str, Any]], alert_min: int | None) -> int:
    if alert_min is None:
        return 0
    pre_candidates: list[dict[str, Any]] = []
    post_candidates: list[dict[str, Any]] = []
    for event in events:
        notes = _normalize_text(str(event.get("notes", "")))
        what = _normalize_text(str(event.get("what", "")))
        blob = f"{notes} {what}"
        if "boundary_split_pre" in blob:
            pre_candidates.append(event)
        if "boundary_split_post" in blob:
            post_candidates.append(event)
    if not pre_candidates or not post_candidates:
        return 0

    for pre in pre_candidates:
        pre_end = _to_minutes(str(pre.get("end", "")))
        if pre_end != alert_min:
            continue
        for post in post_candidates:
            post_start = _to_minutes(str(post.get("start", "")))
            post_end = _to_minutes(str(post.get("end", "")))
            if post_start != alert_min or post_end is None or post_end <= post_start:
                continue
            notes = _normalize_text(str(post.get("notes", "")))
            if "partial_compensation=1" in notes or "compensated" in notes:
                return 1
    return 0


def _events_time_monotonic(events: list[dict[str, Any]]) -> bool:
    # Evaluate timeline consistency per participant instead of globally.
    by_actor: dict[str, list[tuple[int, int]]] = {}
    for event in events:
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None or end_m < start_m:
            return False
        who_raw = str(event.get("who", "")).strip()
        actors = [a.strip().lower() for a in re.split(r"[,&/]+", who_raw) if a.strip()]
        if not actors:
            actors = ["_unknown"]
        for actor in actors:
            by_actor.setdefault(actor, []).append((start_m, end_m))

    for windows in by_actor.values():
        windows.sort(key=lambda x: (x[0], x[1]))
        prev_end = None
        for start_m, end_m in windows:
            if prev_end is not None and start_m < prev_end:
                return False
            prev_end = end_m
    return True


def _score_planning_json(
    parsed: dict[str, Any],
    constraints: dict[str, Any],
    disruptions: list[dict[str, Any]] | None,
    metadata: dict[str, Any] | None,
    enforce_disruption_hard: bool,
    enforce_immutable_hard: bool,
    enforce_state_hard: bool,
) -> tuple[int, list[str], dict[str, Any]]:
    violations: list[str] = []
    disruptions = disruptions or []
    diagnostics = _planning_diagnostics_template(disruptions)

    events = parsed.get("events")
    if not isinstance(events, list) or not events:
        diagnostics["events_count"] = len(events) if isinstance(events, list) else 0
        return 1, ["Missing or empty events list"], diagnostics
    diagnostics["events_count"] = len(events)
    diagnostics["non_empty_events"] = 1

    # Required fields and time format.
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            violations.append(f"event[{idx}] is not an object")
            continue
        for key in ["start", "end", "who", "what", "location"]:
            if key not in event:
                violations.append(f"event[{idx}] missing key '{key}'")
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if start_m is None or end_m is None:
            violations.append(f"event[{idx}] has invalid time format")
        elif end_m < start_m:
            violations.append(f"event[{idx}] end before start")

    # Deadline constraints.
    deadline_keys = ["wedding_deadline", "dinner_deadline", "photo_time"]
    for key in deadline_keys:
        if key in constraints:
            limit = _to_minutes(str(constraints[key]))
            if limit is None:
                continue
            for idx, event in enumerate(events):
                end_m = _to_minutes(str(event.get("end", "")))
                if end_m is not None and end_m > limit:
                    violations.append(f"event[{idx}] exceeds {key}")

    # Store open / close constraints.
    gift_open = _to_minutes(str(constraints.get("gift_store_opens", "")))
    tailor_close = _to_minutes(str(constraints.get("tailor_closes", "")))
    for idx, event in enumerate(events):
        what = _normalize_text(str(event.get("what", "")))
        start_m = _to_minutes(str(event.get("start", "")))
        end_m = _to_minutes(str(event.get("end", "")))
        if gift_open is not None and "gift" in what and start_m is not None and start_m < gift_open:
            violations.append(f"event[{idx}] gift task before store opens")
        if tailor_close is not None and "tailor" in what and end_m is not None and end_m > tailor_close:
            violations.append(f"event[{idx}] tailor task after close")

    # Disruption awareness checks for reactive planning tasks.
    if disruptions:
        metadata = metadata or {}
        scenario_disruption_applicable = bool(metadata.get("scenario_disruption_applicable", False))
        scenario_partial_compensation_required = bool(
            metadata.get("scenario_partial_compensation_required", False)
        )
        scenario_immutable_check_applicable = bool(
            metadata.get("scenario_immutable_check_applicable", False)
        )
        scenario_state_check_applicable = bool(
            metadata.get("scenario_state_check_applicable", False)
        )
        require_boundary_crossing = bool(metadata.get("require_boundary_crossing", False))

        earliest = _resolve_alert_min(disruptions)
        pre_events = events
        post_events = events
        if earliest is not None:
            pre_events = [
                event for event in events if _to_minutes(str(event.get("end", ""))) is not None and _to_minutes(str(event.get("end", ""))) <= earliest
            ]
            post_events = [
                event for event in events if _to_minutes(str(event.get("start", ""))) is not None and _to_minutes(str(event.get("start", ""))) >= earliest
            ]

        locked_prefix = metadata.get("locked_prefix_events", []) if isinstance(metadata, dict) else []
        if isinstance(locked_prefix, list) and locked_prefix:
            diagnostics["prefix_lock_used"] = 1

        seg = _analyze_disruption_segments(events, disruptions, metadata, earliest)
        disruption_applied = _is_disruption_applied(post_events, disruptions, metadata, seg)
        split_pre, split_post = _crossing_split_markers(events)
        split_applied = int(split_pre == 1 and split_post == 1)
        disruption_applicable = int(
            scenario_disruption_applicable or seg.get("relevant_segments", 0) > 0 or bool(post_events)
        )
        if require_boundary_crossing and split_applied:
            disruption_applied = 1
        diagnostics["disruption_applicable"] = disruption_applicable
        diagnostics["disruption_applied"] = disruption_applied
        diagnostics["crossing_split_applicable"] = int(require_boundary_crossing)
        diagnostics["crossing_split_applied"] = split_applied
        diagnostics["partial_compensation_applicable"] = int(
            scenario_partial_compensation_required or seg.get("crossing_segments", 0) > 0
        )
        if diagnostics["partial_compensation_applicable"] == 1:
            if seg.get("crossing_segments", 0) > 0:
                diagnostics["partial_compensation_ok"] = int(
                    seg.get("crossing_compensated", 0) >= seg.get("crossing_segments", 0)
                )
            elif split_applied:
                diagnostics["partial_compensation_ok"] = _split_partial_compensation_correct(
                    events, earliest
                )
            else:
                diagnostics["partial_compensation_ok"] = 0
        else:
            diagnostics["partial_compensation_ok"] = 1

        diagnostics["immutable_check_applicable"] = int(
            scenario_immutable_check_applicable or bool(pre_events)
        )
        diagnostics["immutable_prefix_ok"] = _immutable_prefix_ok(
            pre_events,
            disruptions,
            locked_prefix_events=locked_prefix if isinstance(locked_prefix, list) else None,
        )
        diagnostics["immutable_prefix_after_split_ok"] = int(
            diagnostics["immutable_prefix_ok"] if split_applied else 0
        )
        diagnostics["state_check_applicable"] = int(
            scenario_state_check_applicable or (earliest is not None and bool(post_events))
        )
        diagnostics["state_at_alert_consistent"] = _state_consistent_at_alert(
            events,
            earliest,
            locked_prefix_events=locked_prefix if isinstance(locked_prefix, list) else None,
        )

        if require_boundary_crossing and seg.get("crossing_segments", 0) <= 0 and not split_applied:
            violations.append("Required disruption-boundary crossing segment is missing")

        if diagnostics["disruption_applicable"] and not post_events:
            violations.append("No events scheduled after disruption start")
        if enforce_disruption_hard and diagnostics["crossing_split_applicable"] and not split_applied:
            violations.append("Boundary crossing split not applied")
        if enforce_disruption_hard and diagnostics["disruption_applicable"] and not disruption_applied:
            violations.append("No post-disruption event reflects disruption handling (hard)")
        if (
            enforce_disruption_hard
            and diagnostics["partial_compensation_applicable"]
            and not diagnostics["partial_compensation_ok"]
        ):
            violations.append("Partial journey compensation is missing at disruption boundary")
        if (
            enforce_immutable_hard
            and diagnostics["immutable_check_applicable"]
            and not diagnostics["immutable_prefix_ok"]
        ):
            violations.append("Immutable past appears modified by disruption terms")
        if (
            enforce_state_hard
            and diagnostics["state_check_applicable"]
            and not diagnostics["state_at_alert_consistent"]
        ):
            violations.append("State timeline is inconsistent at disruption boundary")

    return len(violations), violations, diagnostics


def _extract_event_dicts_from_repl(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def _extract_balanced_object(src: str, brace_start: int) -> str | None:
        depth = 0
        in_single = False
        in_double = False
        escape = False
        for idx in range(brace_start, len(src)):
            ch = src[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if not in_double and ch == "'":
                in_single = not in_single
                continue
            if not in_single and ch == '"':
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return src[brace_start : idx + 1]
        return None

    for match in re.finditer(r"(?:events|schedule)\.append\(\s*{", text):
        brace_start = text.find("{", match.start())
        if brace_start == -1:
            continue
        chunk = _extract_balanced_object(text, brace_start)
        if not chunk:
            continue

        cleaned = re.sub(r",\s*([}\]])", r"\1", chunk)
        cleaned = re.sub(r"""f"[^"]*\"""", '"<expr>"', cleaned)
        cleaned = re.sub(r"""f'[^']*'""", '"<expr>"', cleaned)
        try:
            parsed = parse_json_response(cleaned)[0]
            if not isinstance(parsed, dict):
                continue
        except Exception:
            continue

        required = {"start", "end", "who", "what", "location"}
        if required.issubset(parsed.keys()):
            event = {k: str(parsed.get(k, "")) for k in ["start", "end", "who", "what", "location", "notes"]}
            events.append(event)

    return events


def _extract_event_dicts_from_list_assignment(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for key in ["events", "schedule", "plan_events"]:
        # Non-greedy capture of list assignment; tolerate markdown/code wrappers.
        for match in re.finditer(rf"{key}\s*=\s*(\[[\s\S]*?\])", text):
            list_text = match.group(1)
            if not list_text:
                continue
            try_candidates = [list_text, re.sub(r",\s*([}\]])", r"\1", list_text)]
            parsed_list = None
            for cand in try_candidates:
                try:
                    parsed_list = json.loads(cand)
                    break
                except Exception:
                    pass
                try:
                    parsed_list = ast.literal_eval(cand)
                    break
                except Exception:
                    pass
            if not isinstance(parsed_list, list):
                continue
            for item in parsed_list:
                if not isinstance(item, dict):
                    continue
                required = {"start", "end", "who", "what", "location"}
                if not required.issubset(item.keys()):
                    continue
                events.append(
                    {
                        k: str(item.get(k, ""))
                        for k in ["start", "end", "who", "what", "location", "notes"]
                    }
                )

    return events


def score_prediction(
    sample: BenchmarkSample,
    prediction: str,
    policy: str = RUNTIME_POLICY,
) -> dict[str, Any]:
    """Score a single sample prediction."""
    policy_cfg = _resolve_policy(policy)
    if sample.task_type == "planning":
        parsed, parse_error = parse_json_response(prediction)
        if parsed is not None and "events" not in parsed and isinstance(parsed.get("schedule"), list):
            parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}

        if parse_error or parsed is None:
            recovered_events = _extract_event_dicts_from_repl(prediction)
            recovered_events.extend(_extract_event_dicts_from_list_assignment(prediction))
            # Deduplicate stable event tuples.
            dedup = []
            seen = set()
            for e in recovered_events:
                sig = tuple(e.get(k, "") for k in ["start", "end", "who", "what", "location", "notes"])
                if sig in seen:
                    continue
                seen.add(sig)
                dedup.append(e)
            recovered_events = dedup
            if recovered_events:
                parsed = {
                    "plan_summary": "Recovered from REPL append events",
                    "events": recovered_events,
                }
                parse_error = None
            else:
                diagnostics = _planning_diagnostics_template(sample.metadata.get("disruptions", []))
                diagnostics["valid_json"] = 0
                return {
                    "success": 0,
                    "score": 0.0,
                    "violation_count": 1,
                    "violations": [parse_error or "Invalid JSON output"],
                    "parsed_output": None,
                    "diagnostics": diagnostics,
                }

        if parsed is None:
            diagnostics = _planning_diagnostics_template(sample.metadata.get("disruptions", []))
            diagnostics["valid_json"] = 0
            return {
                "success": 0,
                "score": 0.0,
                "violation_count": 1,
                "violations": [parse_error or "Invalid JSON output"],
                "parsed_output": None,
                "diagnostics": diagnostics,
            }

        violation_count, violations, diagnostics = _score_planning_json(
            parsed=parsed,
            constraints=sample.metadata.get("constraints", {}),
            disruptions=sample.metadata.get("disruptions", []),
            metadata=sample.metadata,
            enforce_disruption_hard=policy_cfg["enforce_disruption_hard"],
            enforce_immutable_hard=policy_cfg["enforce_immutable_hard"],
            enforce_state_hard=policy_cfg["enforce_state_hard"],
        )
        success = 1 if violation_count == 0 else 0
        return {
            "success": success,
            "score": float(success),
            "violation_count": violation_count,
            "violations": violations,
            "parsed_output": parsed,
            "diagnostics": diagnostics,
        }

    if sample.task_type == "mcq":
        pred = _extract_mcq_choice(prediction)
        gold = _extract_mcq_choice(str(sample.answer or ""))
        success = int(pred == gold)
        return {
            "success": success,
            "score": float(success),
            "violation_count": 0,
            "violations": [],
            "parsed_output": pred,
            "diagnostics": {},
        }

    # Generic QA: exact match first, then tolerant span match.
    gold_raw = str(sample.answer or "")
    gold = _normalize_qa_span(gold_raw)
    pred = _normalize_qa_span(prediction)
    success = int(pred == gold)
    if not success and gold:
        # Support multi-answer targets encoded as "a | b | c".
        candidates = [_normalize_qa_span(x) for x in gold_raw.split("|")]
        candidates = [c for c in candidates if c]
        if not candidates:
            candidates = [gold]

        for cand in candidates:
            if len(cand) >= 3 and cand in pred:
                success = 1
                break
            if len(pred) >= 3 and pred in cand:
                success = 1
                break

    return {
        "success": success,
        "score": float(success),
        "violation_count": 0,
        "violations": [],
        "parsed_output": pred,
        "diagnostics": {},
    }


def build_rule_validator(sample: BenchmarkSample, policy: str = RUNTIME_POLICY):
    """Build saga validator callable for a specific sample."""
    policy_cfg = _resolve_policy(policy)

    def _validator(payload: dict[str, Any]) -> dict[str, Any]:
        # Only validate planning tasks aggressively.
        if sample.task_type != "planning":
            return {
                "status": "accept",
                "validation_result": "Feedback",
                "reason": "non_planning_task",
                "feedback": "",
            }

        locals_dict = payload.get("locals", {}) or {}
        candidate_values: list[str] = []
        for key in ["Final", "final_answer", "answer", "result", "output", "plan", "schedule"]:
            value = locals_dict.get(key)
            if isinstance(value, str) and value.strip():
                candidate_values.append(value)
            elif isinstance(value, (dict, list)):
                try:
                    candidate_values.append(json.dumps(value, ensure_ascii=False))
                except Exception:
                    pass

        result = payload.get("result")
        if result is not None:
            stdout = getattr(result, "stdout", "")
            if isinstance(stdout, str) and "{" in stdout and "}" in stdout:
                candidate_values.append(stdout)

        if not candidate_values:
            return {
                "status": "accept",
                "validation_result": "Feedback",
                "reason": "no_structured_candidate",
                "feedback": "",
            }

        parsed, parse_error = parse_json_response(candidate_values[-1])
        if parsed is not None and "events" not in parsed and isinstance(parsed.get("schedule"), list):
            parsed = {"plan_summary": str(parsed.get("plan_summary", "")), "events": parsed.get("schedule", [])}
        if parse_error or parsed is None:
            recovered = _extract_event_dicts_from_repl(candidate_values[-1])
            recovered.extend(_extract_event_dicts_from_list_assignment(candidate_values[-1]))
            if recovered:
                parsed = {"plan_summary": "Recovered in validator", "events": recovered}
                parse_error = None
            else:
                return {
                    "status": "reject",
                    "validation_result": "Rejection",
                    "reason": "json_parse_failed",
                    "feedback": f"JSON validation failed: {parse_error}",
                }

        if parsed is None:
            return {
                "status": "reject",
                "validation_result": "Rejection",
                "reason": "json_parse_failed",
                "feedback": f"JSON validation failed: {parse_error}",
            }

        violation_count, violations, diagnostics = _score_planning_json(
            parsed,
            constraints=sample.metadata.get("constraints", {}),
            disruptions=sample.metadata.get("disruptions", []),
            metadata=sample.metadata,
            enforce_disruption_hard=policy_cfg["enforce_disruption_hard"],
            enforce_immutable_hard=policy_cfg["enforce_immutable_hard"],
            enforce_state_hard=policy_cfg["enforce_state_hard"],
        )
        if violation_count == 0:
            return {
                "status": "accept",
                "validation_result": "Feedback",
                "reason": "valid_plan",
                "feedback": "",
            }

        if diagnostics.get("non_empty_events", 0) == 0:
            return {
                "status": "reject",
                "validation_result": "Rejection",
                "reason": "empty_events",
                "feedback": "events list is empty; fill minimal valid schedule before commit",
            }

        if violation_count <= 2:
            return {
                "status": "augment",
                "validation_result": "Augmentation",
                "reason": "minor_violations",
                "feedback": "; ".join(violations[:2]),
            }

        return {
            "status": "reject",
            "validation_result": "Rejection",
            "reason": "constraint_violations",
            "feedback": "; ".join(violations[:3]),
        }

    return _validator
