from __future__ import annotations

import pytest
from typing import Any

from kusudaemon.roles.protocol import RoleProvider, RoleProviderBase
from kusudaemon.roles.json_io import _parse_json_object, build_json_instruction, extract_last_json_object
from kusudaemon.roles.factory import LazyRoleProvider
from kusudaemon.roles.backend_provider import BackendRoleProvider
from kusudaemon.v1.provider import OpenAICompatibleProvider


class _DummyScriptedProvider:
    model: str = "dummy-model"

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        retries: int = 2,
        on_reasoning: Any = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        return {"action": "dispatch", "reason": "ok"}


def test_role_provider_protocol_runtime_check():
    scripted = _DummyScriptedProvider()
    assert isinstance(scripted, RoleProvider)

    openai_p = OpenAICompatibleProvider(model="test-model")
    assert isinstance(openai_p, RoleProvider)
    assert isinstance(openai_p, RoleProviderBase)

    lazy_p = LazyRoleProvider(lambda: openai_p)
    assert isinstance(lazy_p, RoleProvider)
    assert isinstance(lazy_p, RoleProviderBase)


def test_role_provider_base_hooks():
    base = RoleProviderBase()
    halted = False
    base.set_abort_hook(lambda: halted)
    assert base._should_abort is not None
    assert base._should_abort() is False

    events: list[dict[str, Any]] = []
    base.set_event_hook(lambda from_m, to_m, r: events.append({"from": from_m, "to": to_m, "reason": r}))
    assert base._on_model_fallback is not None
    base._on_model_fallback("a", "b", "rate limit")
    assert len(events) == 1
    assert events[0] == {"from": "a", "to": "b", "reason": "rate limit"}


def test_json_io_helpers():
    schema = {
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    instruction = build_json_instruction(schema)
    assert "JSON object" in instruction
    assert '"status"' in instruction

    raw = 'Some preface text\n```json\n{"status": "ok", "count": 42}\n```\nTrailing text'
    parsed, err = extract_last_json_object(raw)
    assert parsed == {"status": "ok", "count": 42}
    assert err == ""

    obj, err = _parse_json_object('{"status": "ok", "count": 42}')
    assert obj == {"status": "ok", "count": 42}
    assert err == ""

    bad_obj, bad_err = _parse_json_object("[1, 2, 3]")
    assert bad_obj is None
    assert "not a JSON object" in bad_err

    # Test trailing usage records are skipped
    log_with_usage = (
        '{"items": ["a", "b"], "verdict": "pass"}\n'
        '{"type": "usage", "prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0, "total_tokens": 120}'
    )
    test_schema = {"type": "object", "required": ["items", "verdict"], "properties": {"items": {"type": "array"}, "verdict": {"type": "string"}}}
    parsed_usage, err_usage = extract_last_json_object(log_with_usage, schema=test_schema)
    assert parsed_usage == {"items": ["a", "b"], "verdict": "pass"}
    assert err_usage == ""
