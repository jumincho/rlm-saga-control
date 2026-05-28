"""Output schema and tolerant JSON parser for model responses.

The planning track forces the model to emit a single JSON object with an
`events` list of `{start, end, who, what, location, notes}` records.
`render_json_instruction` is the user-facing instruction snippet that goes
into the prompt. `parse_json_response` is the corresponding parser, and it
is deliberately permissive: code-fence wrappers (```json ...```) are
stripped, balanced `{...}` blocks are scanned out of mixed REPL output,
trailing commas are tolerated, and Python-literal dicts (`{'key': ...}`)
are accepted as a last-resort fallback. The parser prefers candidates
that already match the planning schema shape (i.e., have an `events`
list) so that mixed REPL output with multiple JSON-ish objects still
resolves to the intended plan.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

PLANNING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "who": {"type": "string"},
                    "what": {"type": "string"},
                    "location": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["start", "end", "who", "what", "location"],
            },
        },
    },
    "required": ["events"],
}


def render_json_instruction(task_type: str) -> str:
    """Return strict output format instructions for the model."""
    if task_type == "planning":
        return (
            "Return ONLY valid JSON with keys: plan_summary (string) and events (list). "
            "Each event must include start, end, who, what, location, and optional notes. "
            "Use 24-hour HH:MM time format. Do not include markdown code fences."
        )
    return "Return only the final answer string on a single line without labels or explanations."


def extract_json_block(text: str) -> str | None:
    """Extract JSON object candidate from raw text."""
    if not text:
        return None

    text = text.strip()

    if text.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _extract_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    primary = extract_json_block(text)
    if primary:
        candidates.append(primary)

    # Also scan for balanced {...} objects inside mixed outputs (e.g., REPL/code blocks).
    stack: list[int] = []
    seen: set[str] = set()
    for idx, ch in enumerate(text):
        if ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            start = stack.pop()
            chunk = text[start : idx + 1].strip()
            if len(chunk) >= 2 and chunk not in seen:
                seen.add(chunk)
                candidates.append(chunk)

    # Prefer larger candidates first.
    candidates.sort(key=len, reverse=True)
    return candidates


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse model output as JSON object and return (obj, error)."""
    candidates = _extract_object_candidates(text)
    if not candidates:
        return None, "No JSON object found in response"

    def _strip_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    attempts: list[str] = []
    for candidate in candidates:
        attempts.extend([candidate, _strip_trailing_commas(candidate)])

    last_error = None
    parsed_objects: list[dict[str, Any]] = []
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                parsed_objects.append(parsed)
                continue
            last_error = "Parsed JSON is not an object"
        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError: {exc}"

    # Best-effort fallback for Python-literal style dict output.
    for attempt in attempts:
        try:
            parsed = ast.literal_eval(attempt)
            if isinstance(parsed, dict):
                parsed_objects.append(parsed)
        except Exception:
            continue

    if parsed_objects:
        # If available, prioritize the candidate that matches planning schema shape.
        for obj in parsed_objects:
            if isinstance(obj.get("events"), list):
                return obj, None
        return parsed_objects[0], None

    return None, last_error or "Invalid JSON output"
