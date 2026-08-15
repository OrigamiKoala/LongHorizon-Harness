"""Google Antigravity Writer adapter (``agy -p - --output-format stream-json``, Google AGY CLI).

Drives the Antigravity CLI (agy) as an agent backend.

Features:
- Runs ``agy --print - --output-format stream-json --dangerously-skip-permissions`` in non-interactive mode.
- Streams structured JSON events translated via ``_agent_worker.py`` into
  Kusudaemon's unified trace format.
- Supports session resume: ``supports_session_resume = True`` via
  ``agy --conversation <conversation_id>`` or ``run_episode(..., resume_session_id=...)``.
- Supports tool restriction / sandboxing: ``supports_tool_restriction = True``
  via ``--sandbox`` or permission parameters.
- Full parameter validation and clean error handling.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from ..environment.base import Environment
from ..types import DEFAULT_TMP_DIR, DEFAULT_WORKSPACE_PATH, EpisodeBudget, EpisodeResult
from .cli_agent import CommandAgentAdapter
from .trace_output import extract_visible_output

_WORKER_SCRIPT = Path(__file__).with_name("_agent_worker.py")
_PYTHON = sys.executable

_VALID_FORMATS = ("stream-json", "json", "text")
_VALID_EFFORTS = ("low", "medium", "high")


class AntigravityAdapter(CommandAgentAdapter):
    supports_session_resume = True
    supports_tool_restriction = True
    has_file_tools = True

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        conversation_id: str | None = None,
        effort: str | None = None,
        sandbox: bool = False,
        agent: str | None = None,
        project: str | None = None,
        add_dirs: list[str] | None = None,
        format: str = "stream-json",
        auto_approve: bool = True,
        disable_slash_commands: bool = True,
        log_file: str | None = None,
        workspace_path: str = DEFAULT_WORKSPACE_PATH,
        prompt_dir: str = f"{DEFAULT_TMP_DIR}/prompts",
        mcp_config: str | None = None,
        hidden_paths: tuple[str, ...] = (),
        hidden_path_exceptions: tuple[str, ...] = (),
    ) -> None:
        if format not in _VALID_FORMATS:
            raise ValueError(
                f"invalid format {format!r}; choices are {_VALID_FORMATS}"
            )
        if effort is not None:
            normalized_effort = effort.lower()
            if normalized_effort not in _VALID_EFFORTS:
                raise ValueError(
                    f"invalid effort {effort!r}; choices are {_VALID_EFFORTS}"
                )
            effort = normalized_effort

        env_parts: list[str] = []

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("ANTIGRAVITY_API_KEY")
        if key:
            quoted_key = shlex.quote(key)
            env_parts.append(f"GEMINI_API_KEY={quoted_key}")

        command_parts = ["agy", "--print", "-"]

        if format:
            command_parts.extend(["--output-format", shlex.quote(format)])
        if auto_approve:
            command_parts.append("--dangerously-skip-permissions")
        if disable_slash_commands:
            command_parts.append("--disable-slash-commands")
        if model:
            command_parts.extend(["--model", shlex.quote(model)])
        if conversation_id:
            command_parts.extend(["--conversation", shlex.quote(conversation_id)])
        if effort:
            command_parts.extend(["--effort", shlex.quote(effort)])
        if sandbox:
            command_parts.append("--sandbox")
        if agent:
            command_parts.extend(["--agent", shlex.quote(agent)])
        if project:
            command_parts.extend(["--project", shlex.quote(project)])
        if log_file:
            command_parts.extend(["--log-file", shlex.quote(log_file)])

        resolved_add_dirs = list(add_dirs or [])
        env_add_dirs = os.getenv("KUSUDAEMON_ANTIGRAVITY_ADD_DIRS") or os.getenv(
            "KUSUDAEMON_AGY_ADD_DIRS"
        )
        if env_add_dirs:
            resolved_add_dirs.extend(part for part in env_add_dirs.split(os.pathsep) if part)
        for add_dir in resolved_add_dirs:
            command_parts.extend(["--add-dir", shlex.quote(add_dir)])

        self._env_prefix = (" ".join(env_parts) + " ") if env_parts else ""
        self._agy_parts = command_parts
        self.model = model
        self.agent = agent
        self.mcp_config = mcp_config

        super().__init__(
            command_template=self._template(self._env_prefix, command_parts),
            prompt_dir=prompt_dir,
            workspace_path=workspace_path,
            visible_output_parser=extract_visible_output,
            hidden_paths=hidden_paths,
            hidden_path_exceptions=hidden_path_exceptions,
        )

    @staticmethod
    def _template(env_prefix: str, parts: list[str]) -> str:
        quoted_worker = shlex.quote(str(_WORKER_SCRIPT))
        return (
            f"{env_prefix}{shlex.quote(_PYTHON)} {quoted_worker} --format antigravity -- "
            f"{' '.join(parts)} < {{prompt_path}}"
        )

    async def run_episode(
        self,
        prompt: str,
        env: Environment,
        budget: EpisodeBudget,
        live_trajectory_path: str | None = None,
        *,
        resume_session_id: str | None = None,
    ) -> EpisodeResult:
        if resume_session_id:
            parts = [*self._agy_parts]
            if "--conversation" in parts:
                idx = parts.index("--conversation")
                parts[idx + 1] = shlex.quote(str(resume_session_id))
            else:
                parts.extend(["--conversation", shlex.quote(str(resume_session_id))])
            override = self._template(self._env_prefix, parts)
            return await super().run_episode(
                prompt,
                env,
                budget,
                live_trajectory_path,
                command_override=override,
            )
        return await super().run_episode(prompt, env, budget, live_trajectory_path)
