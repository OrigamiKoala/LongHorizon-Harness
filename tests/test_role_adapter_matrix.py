from __future__ import annotations

from pathlib import Path
import pytest

from kusudaemon.adapters.antigravity import AntigravityAdapter
from kusudaemon.adapters.claude_code import ClaudeCodeAdapter
from kusudaemon.adapters.codex import CodexAdapter
from kusudaemon.adapters.opencode import OpenCodeAdapter
from kusudaemon.pipeline.backends import build_role_adapter
from kusudaemon.pipeline.driver import RunOptions
from kusudaemon.roles.backend_provider import BackendRoleProvider
from kusudaemon.roles.factory import make_role_provider, _resolve_role_transport
from kusudaemon.v1.provider import OpenAICompatibleProvider


def test_build_role_adapter_matrix(tmp_path: Path):
    run_dir = tmp_path / "run1"

    # opencode
    opencode_adapter = build_role_adapter("opencode", run_dir=run_dir, phase="classify")
    assert isinstance(opencode_adapter, OpenCodeAdapter)
    assert opencode_adapter.workspace_path == str(run_dir / "tmp" / "roles" / "classify")
    assert "OPENCODE_PERMISSION" in opencode_adapter._env_prefix
    assert "deny" in opencode_adapter._env_prefix

    # antigravity
    antigravity_adapter = build_role_adapter("antigravity", run_dir=run_dir, phase="classify")
    assert isinstance(antigravity_adapter, AntigravityAdapter)
    assert antigravity_adapter.workspace_path == str(run_dir / "tmp" / "roles" / "classify")
    assert "--sandbox" in antigravity_adapter.command_template

    # claude
    claude_adapter = build_role_adapter("claude", run_dir=run_dir, phase="plan")
    assert isinstance(claude_adapter, ClaudeCodeAdapter)
    assert claude_adapter.workspace_path == str(run_dir / "tmp" / "roles" / "plan")
    assert "--disallowedTools" in claude_adapter._claude_parts

    # codex
    codex_adapter = build_role_adapter("codex", run_dir=run_dir, phase="survey")
    assert isinstance(codex_adapter, CodexAdapter)
    assert codex_adapter.workspace_path == str(run_dir / "tmp" / "roles" / "survey")
    assert "--sandbox" in codex_adapter.command_template
    assert "read-only" in codex_adapter.command_template


def test_resolve_role_transport(tmp_path: Path, monkeypatch):
    # Default auto with gptme backend -> http
    assert _resolve_role_transport("gptme") == ("gptme", "http")

    # Default auto with opencode backend -> backend
    assert _resolve_role_transport("opencode") == ("opencode", "backend")

    # Env override
    monkeypatch.setenv("KUSUDAEMON_ROLE_TRANSPORT", "backend")
    assert _resolve_role_transport("gptme") == ("gptme", "backend")

    monkeypatch.setenv("KUSUDAEMON_ROLE_TRANSPORT", "http")
    assert _resolve_role_transport("opencode") == ("opencode", "http")


def test_make_role_provider(tmp_path: Path):
    run_dir = tmp_path / "run_make"

    # Direct: with backend='opencode', make_role_provider produces BackendRoleProvider
    options = RunOptions(backend="opencode")
    provider = make_role_provider(options=options, run_dir=run_dir)
    assert isinstance(provider, BackendRoleProvider)
    assert provider.backend == "opencode"

    # Direct: with backend='gptme', make_role_provider produces OpenAICompatibleProvider
    options_gptme = RunOptions(backend="gptme")
    provider_gptme = make_role_provider(options=options_gptme, run_dir=run_dir)
    assert isinstance(provider_gptme, OpenAICompatibleProvider)

    # Lazy: with lazy=True, make_role_provider produces LazyRoleProvider
    lazy_provider = make_role_provider(options=options, run_dir=run_dir, lazy=True)
    underlying = lazy_provider._resolve()
    assert isinstance(underlying, BackendRoleProvider)
    assert underlying.backend == "opencode"
