from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from kusudaemon.adapters.base import AgentAdapter
from kusudaemon.environment.base import Environment
from kusudaemon.pipeline.driver import RunOptions, RecursiveDriver
from kusudaemon.roles.backend_provider import BackendRoleProvider
from kusudaemon.types import EpisodeBudget, EpisodeResult
from kusudaemon.v1.tree import TaskTree


class _ScriptedCLIAdapter(AgentAdapter):
    """Simulates a CLI backend handling both role queries and writer tasks."""
    has_file_tools = True

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    async def run_episode(
        self,
        prompt: str,
        env: Environment,
        budget: EpisodeBudget,
        **kwargs: Any,
    ) -> EpisodeResult:
        if "files_touched" in prompt or "scope estimator" in prompt or "Work object digest" in prompt:
            data = {
                "files_touched": "1",
                "artifacts": 1,
                "answerable_without_exploration": True,
                "questions": [],
                "objections": [],
            }
            content = json.dumps(data)
        elif "boundary_after" in prompt:
            content = json.dumps({"boundary_after": [], "rationale": "one chunk"})
        elif "children" in prompt and "unit_start" in prompt:
            data = {
                "children": [
                    {
                        "id": "node-1",
                        "brief": "write section 1",
                        "unit_start": 0,
                        "unit_end": 0,
                        "estimated_calls": 1,
                        "shape": "prose-dominant",
                    }
                ],
                "probes": [],
            }
            content = json.dumps(data)
        elif "You are the Orchestrator" in prompt:
            data = {
                "action": "dispatch",
                "node_id": "single",
                "reason": "first ready node",
            }
            content = json.dumps(data)
        elif "You are the Reviewer" in prompt or "Rubric:" in prompt:
            data = {
                "verdict": "pass",
                "items": [{"id": "item-1", "pass": True}],
            }
            content = json.dumps(data)
        elif "deriving authoring rules" in prompt or "DERIVE" in prompt:
            content = json.dumps({"rules": []})
        else:
            out_file = self.run_dir / "out" / "single.md"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("# Section 1\nContent written by backend.\n", encoding="utf-8")
            content = "Done writing artifact."

        return EpisodeResult(
            status="done",
            actions_log=content,
        )


class _LocalTestEnv(Environment):
    def read_file(self, path: str) -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def write_file(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def run_command(self, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
        return 0, "", ""


def test_keyless_run_end_to_end(tmp_path: Path, monkeypatch):
    import asyncio
    # Strip any OpenAI/external API keys from the environment to test keyless run
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    run_dir = tmp_path / "test_keyless"
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = _ScriptedCLIAdapter(run_dir)
    env = _LocalTestEnv()

    role_provider = BackendRoleProvider(
        backend="opencode",
        run_dir=run_dir,
        env=env,
        adapter_factory=lambda phase: adapter,
        model="opencode/deepseek-v4-flash-free",
    )

    options = RunOptions(
        goal="Write a brief doc",
        source_text="Sample input text for keyless run.",
        backend="opencode",
        tier_override="T1",
        no_intake=True,
    )

    driver = RecursiveDriver(
        run_dir,
        provider=role_provider,
        options=options,
        env=env,
        writer_adapter_factory=lambda node: adapter,
    )

    report = asyncio.run(driver.run())
    assert report.status == "done"

    # Verify artifact output
    output_path = run_dir / "out" / "single.md"
    assert output_path.exists()
    assert "Section 1" in output_path.read_text(encoding="utf-8")
