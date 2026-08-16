"""Shared JSON parsing, extraction, and schema instruction utilities.

Used by both OpenAICompatibleProvider (direct HTTP) and BackendRoleProvider
(CLI episodes) to enforce identical schema extraction and validate-reprompt
semantics.
"""

from __future__ import annotations

import json
from typing import Any

from ..v1.json_schema import describe_schema, validate


def build_json_instruction(schema: dict[str, Any]) -> str:
    """Build prose schema instructions matching the standard fallback format."""
    return (
        "Respond with a single JSON object only — no prose, no "
        f"code fences — matching this schema:\n{describe_schema(schema)}"
    )


def _parse_json_object(content: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a single JSON object from string content, stripping code fences if present."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response was not a JSON object"
    return parsed, ""


_HARNESS_METADATA_TYPES = {"usage", "logdir", "heartbeat", "thinking", "system", "session_captured"}


def extract_last_json_object(
    text: str, schema: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str]:
    """Extract the last valid JSON object from arbitrary text or log traces.

    Attempts standard parsing first; if that fails, scans backwards for balanced
    curly braces `{ ... }` and attempts JSON decoding, skipping harness metadata lines.
    """
    if not text or not text.strip():
        return None, "empty response"

    parsed, err = _parse_json_object(text)
    if parsed is not None and parsed.get("type") not in _HARNESS_METADATA_TYPES:
        if schema is None or not validate(parsed, schema):
            return parsed, ""

    first_fallback: dict[str, Any] | None = None

    # Scan backwards for JSON object substrings
    last_brace_idx = text.rfind("}")
    while last_brace_idx != -1:
        # Search backwards for matching '{'
        depth = 0
        found_start = -1
        for i in range(last_brace_idx, -1, -1):
            char = text[i]
            if char == "}":
                depth += 1
            elif char == "{":
                depth -= 1
                if depth == 0:
                    found_start = i
                    break
        if found_start != -1:
            candidate = text[found_start : last_brace_idx + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and obj.get("type") not in _HARNESS_METADATA_TYPES:
                    if schema is not None:
                        if not validate(obj, schema):
                            return obj, ""
                        if first_fallback is None:
                            first_fallback = obj
                    else:
                        return obj, ""
            except json.JSONDecodeError:
                pass
        last_brace_idx = text.rfind("}", 0, last_brace_idx)

    if first_fallback is not None:
        return first_fallback, ""

    return None, err
