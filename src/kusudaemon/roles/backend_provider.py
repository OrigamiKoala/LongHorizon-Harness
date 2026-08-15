"""Agent backend-driven role provider (ROLE-CALLS-VIA-BACKENDS-PLAN.md §1.2).

Executes reasoning and structured-output role calls as one-shot, tool-less CLI
episodes on the configured agent backend (opencode, claude, or codex) instead of
raw HTTP calls.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

from ..adapters.base import EpisodeBudget
from ..environment.base import Environment
from ..environment.local import LocalEnvironment
from ..v0.events import EventLog
from ..v0.run_dir import events_path
from ..v1.json_schema import validate
from ..v1.provider import ProviderError
from .episode_loop import get_episode_loop
from .json_io import _parse_json_object, build_json_instruction, extract_last_json_object
from .protocol import RoleProviderBase


def _flatten_messages(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
) -> str:
    """Flatten structured chat messages and JSON schema into a single prompt string."""
    system_parts: list[str] = [build_json_instruction(schema)]
    conversation_parts: list[dict[str, str]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            conversation_parts.append({"role": role, "content": content})

    system_text = "\n\n".join(part.strip() for part in system_parts if part.strip())

    if not conversation_parts:
        return system_text

    if len(conversation_parts) == 1 and conversation_parts[0]["role"] == "user":
        user_content = conversation_parts[0]["content"].strip()
        return f"{system_text}\n\n{user_content}"

    formatted_convo: list[str] = []
    for msg in conversation_parts:
        role_label = msg["role"].capitalize()
        formatted_convo.append(f"[{role_label}]:\n{msg['content'].strip()}")

    convo_text = "\n\n".join(formatted_convo)
    return f"{system_text}\n\n---\n\n{convo_text}"


class BackendRoleProvider(RoleProviderBase):
    """Executes role calls as one-shot CLI episodes via the backend adapter."""

    def __init__(
        self,
        *,
        backend: str,
        run_dir: Path | str,
        env: Environment | None = None,
        model: str | None = None,
        budget: EpisodeBudget | None = None,
        max_episode_retries: int = 2,
        log: EventLog | None = None,
        concurrency: int = 4,
        adapter_factory: Callable[[str], AgentAdapter] | None = None,
    ) -> None:
        self.backend = str(backend).strip().lower()
        self.run_dir = Path(run_dir)
        self.env = env
        self.model = model or ""
        self.budget = budget or EpisodeBudget(max_duration_seconds=600)
        self.max_episode_retries = max_episode_retries
        self._concurrency = concurrency
        self._adapter_factory = adapter_factory
        if log is not None:
            self.log: EventLog | None = log
        else:
            p = events_path(self.run_dir)
            self.log = EventLog(p) if p.parent.exists() else None

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        retries: int = 2,
        on_reasoning: Callable[[str], None] | None = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Execute a schema-constrained call via one-shot backend episode(s)."""
        from ..pipeline.backends import build_role_adapter

        base_messages = list(messages)
        last_error = "empty response"
        episode_loop = get_episode_loop(concurrency=self._concurrency)

        for attempt in range(retries + 1):
            if self._should_abort is not None and self._should_abort():
                raise ProviderError("Execution aborted by driver")

            prompt = _flatten_messages(base_messages, schema)
            call_id = f"call_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            roles_dir = self.run_dir / "roles" / "role"
            roles_dir.mkdir(parents=True, exist_ok=True)
            live_trajectory = roles_dir / f"{call_id}_raw_trajectory.jsonl"

            if self._adapter_factory is not None:
                adapter = self._adapter_factory("role")
            else:
                adapter = build_role_adapter(
                    backend=self.backend,
                    run_dir=self.run_dir,
                    phase="role",
                    model=self.model or None,
                    env=self.env,
                )
            env = self.env or LocalEnvironment(tmp_dir=str(self.run_dir / "tmp"))

            # Run episode with retries for episode-level failures (timeout/error)
            episode_result = None
            for ep_try in range(self.max_episode_retries + 1):
                if self._should_abort is not None and self._should_abort():
                    raise ProviderError("Execution aborted by driver")

                res = episode_loop.run_coroutine(
                    adapter.run_episode(
                        prompt,
                        env,
                        self.budget,
                        live_trajectory_path=str(live_trajectory),
                    )
                )

                # Observability: forward thinking/reasoning if recorded
                if on_reasoning is not None and live_trajectory.is_file():
                    try:
                        for line in live_trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
                            line = line.strip()
                            if line:
                                rec = json.loads(line)
                                if rec.get("type") in ("thinking", "reasoning") and rec.get("content"):
                                    on_reasoning(str(rec["content"]))
                    except Exception:
                        pass

                if res.status in ("error", "timeout"):
                    if self.log is not None:
                        try:
                            self.log.append(
                                {
                                    "node_id": "-",
                                    "role": "role",
                                    "type": "role_episode_failed",
                                    "backend": self.backend,
                                    "status": res.status,
                                    "error": res.error or "",
                                    "attempt": ep_try + 1,
                                }
                            )
                        except Exception:
                            pass
                    if ep_try < self.max_episode_retries:
                        continue
                    raise ProviderError(
                        f"role episode failed ({res.status}) after {self.max_episode_retries + 1} attempts: {res.error or 'unknown error'}"
                    )
                episode_result = res
                break

            if episode_result is None:
                raise ProviderError(f"role episode failed on backend {self.backend}")

            # Extract output
            content = (episode_result.metadata or {}).get("assistant_visible_output") or ""
            if not content.strip():
                # Fallback to scanning actions_log for valid JSON
                parsed, parse_err = extract_last_json_object(episode_result.actions_log)
            else:
                parsed, parse_err = _parse_json_object(content)

            if parsed is not None:
                schema_errors = validate(parsed, schema)
                if not schema_errors:
                    return parsed
                last_error = "; ".join(schema_errors)
            else:
                last_error = parse_err or "could not extract JSON object from output"

            # Reprompt on failure
            base_messages = [
                *base_messages,
                {"role": "assistant", "content": content or episode_result.actions_log},
                {
                    "role": "user",
                    "content": f"That did not validate: {last_error}. Return corrected JSON only.",
                },
            ]

        raise ProviderError(
            f"structured output failed after {retries + 1} attempts: {last_error}"
        )
