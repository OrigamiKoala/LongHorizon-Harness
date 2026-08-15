from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from kusudaemon.adapters.base import AgentAdapter
from kusudaemon.environment.base import Environment
from kusudaemon.roles.backend_provider import BackendRoleProvider, _flatten_messages
from kusudaemon.roles.protocol import RoleProvider
from kusudaemon.types import EpisodeBudget, EpisodeResult
from kusudaemon.v1.provider import ProviderError


class _FakeRoleAdapter(AgentAdapter):
    def __init__(self, responses: list[str | Exception]):
        self.responses = list(responses)
        self.prompts_received: list[str] = []

    async def run_episode(
        self,
        prompt: str,
        env: Environment,
        budget: EpisodeBudget,
        **kwargs: Any,
    ) -> EpisodeResult:
        self.prompts_received.append(prompt)
        if not self.responses:
            return EpisodeResult(
                status="error",
                error="no more responses",
            )
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return EpisodeResult(
            status="done",
            actions_log=resp,
        )


class _DummyEnv(Environment):
    def read_file(self, path: str) -> str:
        return ""

    def write_file(self, path: str, content: str) -> None:
        pass

    def run_command(self, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
        return 0, "", ""


def test_flatten_messages():
    messages = [
        {"role": "system", "content": "You are a classifier."},
        {"role": "user", "content": "Hello, classify this."},
        {"role": "assistant", "content": "Sure, what is it?"},
        {"role": "user", "content": "Task A."},
    ]
    prompt = _flatten_messages(messages, {"type": "object"})
    assert "JSON object" in prompt
    assert "You are a classifier." in prompt
    assert "Hello, classify this." in prompt
    assert "[Assistant]" in prompt
    assert "Sure, what is it?" in prompt
    assert "Task A." in prompt


def test_backend_role_provider_success(tmp_path: Path):
    valid_json = json.dumps({"action": "dispatch", "reason": "single ready node"})
    adapter = _FakeRoleAdapter([valid_json])
    provider = BackendRoleProvider(
        backend="opencode",
        run_dir=tmp_path,
        env=_DummyEnv(),
        adapter_factory=lambda phase: adapter,
        model="test-model",
    )
    assert isinstance(provider, RoleProvider)
    assert provider.model == "test-model"

    schema = {
        "type": "object",
        "required": ["action", "reason"],
        "properties": {
            "action": {"type": "string"},
            "reason": {"type": "string"},
        },
    }
    result = provider.complete_json(
        [{"role": "user", "content": "What next?"}],
        schema=schema,
    )
    assert result == {"action": "dispatch", "reason": "single ready node"}
    assert len(adapter.prompts_received) == 1


def test_backend_role_provider_retry_on_invalid_json(tmp_path: Path):
    invalid = "I cannot output JSON directly, but here is my thought..."
    valid_json = json.dumps({"action": "dispatch", "reason": "corrected"})
    adapter = _FakeRoleAdapter([invalid, valid_json])
    provider = BackendRoleProvider(
        backend="opencode",
        run_dir=tmp_path,
        env=_DummyEnv(),
        adapter_factory=lambda phase: adapter,
        model="test-model",
    )

    schema = {
        "type": "object",
        "required": ["action", "reason"],
    }
    result = provider.complete_json(
        [{"role": "user", "content": "What next?"}],
        schema=schema,
        retries=2,
    )
    assert result == {"action": "dispatch", "reason": "corrected"}
    assert len(adapter.prompts_received) == 2
    assert "did not validate" in adapter.prompts_received[1]


def test_backend_role_provider_exhausted_retries(tmp_path: Path):
    invalid1 = "Not json 1"
    invalid2 = "Not json 2"
    adapter = _FakeRoleAdapter([invalid1, invalid2])
    provider = BackendRoleProvider(
        backend="opencode",
        run_dir=tmp_path,
        env=_DummyEnv(),
        adapter_factory=lambda phase: adapter,
        model="test-model",
    )

    schema = {"type": "object", "required": ["action"]}
    with pytest.raises(ProviderError) as exc_info:
        provider.complete_json(
            [{"role": "user", "content": "What next?"}],
            schema=schema,
            retries=1,
        )
    assert "structured output failed after" in str(exc_info.value)


def test_backend_role_provider_abort(tmp_path: Path):
    adapter = _FakeRoleAdapter([json.dumps({"action": "dispatch"})])
    provider = BackendRoleProvider(
        backend="opencode",
        run_dir=tmp_path,
        env=_DummyEnv(),
        adapter_factory=lambda phase: adapter,
    )
    provider.set_abort_hook(lambda: True)

    with pytest.raises(ProviderError, match="aborted"):
        provider.complete_json(
            [{"role": "user", "content": "What next?"}],
            schema={"type": "object"},
        )
