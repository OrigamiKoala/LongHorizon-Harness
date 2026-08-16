"""Shared JSON parsing, extraction, and schema instruction utilities.

Used by both OpenAICompatibleProvider (direct HTTP) and BackendRoleProvider
(CLI episodes) to enforce identical schema extraction and validate-reprompt
semantics.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..v1.json_schema import describe_schema, validate


def build_json_instruction(schema: dict[str, Any]) -> str:
    """Build prose schema instructions matching the standard fallback format."""
    return (
        "Respond with a single JSON object only — no prose, no "
        f"code fences — matching this schema:\n{describe_schema(schema)}"
    )


_HARNESS_METADATA_TYPES = {"usage", "logdir", "heartbeat", "thinking", "system", "session_captured"}


def _parse_json_object(content: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a single JSON object from string content, stripping code fences or trailing data if present."""
    if not content or not content.strip():
        return None, "empty response"
    text = content.strip()

    # Fast path 1: direct parse of full string
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "response was not a JSON object"
    except json.JSONDecodeError:
        pass

    # Fast path 2: markdown code fences ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = fence_pattern.findall(text)
    for block in matches:
        block_text = block.strip()
        try:
            parsed = json.loads(block_text)
            if isinstance(parsed, dict):
                return parsed, ""
        except json.JSONDecodeError:
            pass

    # Fallback: scan for JSON objects using raw_decode (handles trailing prose/extra data)
    decoder = json.JSONDecoder()
    pos = 0
    last_dict: dict[str, Any] | None = None
    first_err: str = ""

    while pos < len(text):
        brace_idx = text.find("{", pos)
        if brace_idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, brace_idx)
            if isinstance(obj, dict):
                last_dict = obj
            pos = max(end_idx, brace_idx + 1)
        except json.JSONDecodeError as exc:
            if not first_err:
                first_err = f"invalid JSON: {exc}"
            pos = brace_idx + 1

    if last_dict is not None:
        return last_dict, ""

    # If no dict was found, produce standard decode error message
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None, "response was not a JSON object"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"

    return None, first_err or "response was not a JSON object"


def extract_last_json_object(
    text: str, schema: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str]:
    """Extract the last valid JSON object from arbitrary text or log traces.

    Attempts standard parsing first; if that fails, scans for JSON objects using
    JSONDecoder, skipping harness metadata lines, and validating against schema if given.
    """
    if not text or not text.strip():
        return None, "empty response"

    # Fast path: standard parse if valid single JSON object
    parsed, err = _parse_json_object(text)
    if parsed is not None and parsed.get("type") not in _HARNESS_METADATA_TYPES:
        if schema is None or not validate(parsed, schema):
            return parsed, ""

    # Scan for all JSON objects in text
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    # Also scan inside fenced code blocks
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    for block in fence_pattern.findall(text):
        b_obj, _ = _parse_json_object(block)
        if b_obj is not None and b_obj.get("type") not in _HARNESS_METADATA_TYPES:
            candidates.append(b_obj)

    pos = 0
    last_err = err
    while pos < len(text):
        brace_idx = text.find("{", pos)
        if brace_idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, brace_idx)
            if isinstance(obj, dict) and obj.get("type") not in _HARNESS_METADATA_TYPES:
                candidates.append(obj)
            pos = max(end_idx, brace_idx + 1)
        except json.JSONDecodeError as exc:
            last_err = f"invalid JSON: {exc}"
            pos = brace_idx + 1

    if not candidates:
        return None, last_err or "could not extract JSON object from output"

    # Search in reverse (last valid object)
    first_fallback: dict[str, Any] | None = None
    for cand in reversed(candidates):
        if schema is not None:
            if not validate(cand, schema):
                return cand, ""
            if first_fallback is None:
                first_fallback = cand
        else:
            return cand, ""

    if first_fallback is not None:
        return first_fallback, ""

    return None, last_err or "could not extract valid JSON object matching schema"
