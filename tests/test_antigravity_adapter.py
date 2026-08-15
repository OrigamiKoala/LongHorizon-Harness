"""Tests for AntigravityAdapter, worker translation, and CLI/backend registration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from kusudaemon.adapters._agent_worker import AGY, ANTIGRAVITY, translate_antigravity, translate_line
from kusudaemon.adapters.antigravity import AntigravityAdapter
from kusudaemon.pipeline import cli as pipeline_cli
from kusudaemon.pipeline import run as pipeline_run
from kusudaemon.pipeline.backends import WRITER_BACKENDS, build_research_adapter, build_role_adapter, build_writer_adapter
from kusudaemon.types import EpisodeBudget
from kusudaemon.v1.tree import TaskNode
from kusudaemon.v4.research import ResearchQuery


class _FakeEnv:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, timeout: int = 300, tee_path: str | None = None):
        self.commands.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="", termination_reason=None)

    async def upload(self, local_path: str, remote_path: str) -> None:
        self.commands.append(f"upload {remote_path}")

    async def download(self, local_path: str, remote_path: str) -> None:
        pass


def _run_episode(adapter, **kwargs):
    return asyncio.run(
        adapter.run_episode(
            "prompt",
            _FakeEnv() if "env" not in kwargs else kwargs.pop("env"),
            EpisodeBudget(max_duration_seconds=60),
            **kwargs,
        )
    )


class AntigravityAdapterFlagsAndTemplateTest(unittest.TestCase):
    def test_adapter_invariants(self) -> None:
        adapter = AntigravityAdapter(workspace_path="/tmp/ws")
        self.assertTrue(adapter.has_file_tools)
        self.assertTrue(adapter.supports_session_resume)
        self.assertTrue(adapter.supports_tool_restriction)

    def test_default_command_template(self) -> None:
        adapter = AntigravityAdapter(workspace_path="/tmp/ws")
        cmd = adapter.command_template
        self.assertIn("--format antigravity -- agy --print - --output-format stream-json --dangerously-skip-permissions --disable-slash-commands", cmd)
        self.assertIn("< {prompt_path}", cmd)

    def test_model_and_effort_flags(self) -> None:
        adapter = AntigravityAdapter(
            workspace_path="/tmp/ws",
            model="gemini-3.7-flash",
            effort="high",
        )
        cmd = adapter.command_template
        self.assertIn("--model gemini-3.7-flash", cmd)
        self.assertIn("--effort high", cmd)

    def test_sandbox_flag(self) -> None:
        adapter = AntigravityAdapter(
            workspace_path="/tmp/ws",
            sandbox=True,
        )
        self.assertIn("--sandbox", adapter.command_template)

    def test_agent_and_project_flags(self) -> None:
        adapter = AntigravityAdapter(
            workspace_path="/tmp/ws",
            agent="build",
            project="test-proj",
        )
        cmd = adapter.command_template
        self.assertIn("--agent build", cmd)
        self.assertIn("--project test-proj", cmd)

    def test_add_dirs_flags(self) -> None:
        adapter = AntigravityAdapter(
            workspace_path="/tmp/ws",
            add_dirs=["/dir/a", "/dir/b"],
        )
        cmd = adapter.command_template
        self.assertIn("--add-dir /dir/a", cmd)
        self.assertIn("--add-dir /dir/b", cmd)

    def test_api_key_env(self) -> None:
        adapter = AntigravityAdapter(
            workspace_path="/tmp/ws",
            api_key="test-gemini-key",
        )
        cmd = adapter.command_template
        self.assertIn("GEMINI_API_KEY=test-gemini-key", cmd)

    def test_invalid_format_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AntigravityAdapter(workspace_path="/tmp/ws", format="xml")  # type: ignore[arg-type]
        self.assertIn("invalid format", str(ctx.exception))

    def test_invalid_effort_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AntigravityAdapter(workspace_path="/tmp/ws", effort="max")  # type: ignore[arg-type]
        self.assertIn("invalid effort", str(ctx.exception))


class AntigravitySessionResumeTest(unittest.TestCase):
    def test_resume_injects_conversation_flag(self) -> None:
        adapter = AntigravityAdapter(workspace_path="/tmp/ws")
        env = _FakeEnv()
        _run_episode(adapter, env=env, resume_session_id="conv_target_456")
        cmd = next(c for c in env.commands if "agy" in c)
        self.assertIn("--conversation conv_target_456", cmd)

    def test_resume_overrides_existing_conversation_id(self) -> None:
        adapter = AntigravityAdapter(workspace_path="/tmp/ws", conversation_id="conv_old")
        env = _FakeEnv()
        _run_episode(adapter, env=env, resume_session_id="conv_new")
        cmd = next(c for c in env.commands if "agy" in c)
        self.assertIn("--conversation conv_new", cmd)
        self.assertNotIn("--conversation conv_old", cmd)

    def test_fresh_episode_no_resume(self) -> None:
        adapter = AntigravityAdapter(workspace_path="/tmp/ws")
        env = _FakeEnv()
        _run_episode(adapter, env=env)
        cmd = next(c for c in env.commands if "agy" in c)
        self.assertNotIn("--conversation", cmd)


class AntigravityWorkerTranslationTest(unittest.TestCase):
    def test_init_emits_logdir_with_session_id(self) -> None:
        record = {
            "event": "init",
            "conversation_id": "conv_12345",
            "init": {"model": "gemini-3.7-flash"},
        }
        res = translate_antigravity(record, session_dir="/tmp/session")
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 1)
        data = json.loads(res[0])
        self.assertEqual(data["type"], "logdir")
        self.assertEqual(data["session_id"], "conv_12345")
        self.assertEqual(data["logdir"], "/tmp/session")
        self.assertEqual(data["model"], "gemini-3.7-flash")

    def test_agent_response_text_delta(self) -> None:
        record = {
            "event": "step_update",
            "step_update": {
                "step_type": "agent_response",
                "state": "DONE",
                "text_delta": "Hello from Antigravity!",
            },
        }
        res = translate_antigravity(record, session_dir="/tmp/session")
        self.assertIsNotNone(res)
        data = json.loads(res[0])
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], "Hello from Antigravity!")

    def test_thinking_translates(self) -> None:
        record = {
            "event": "step_update",
            "step_update": {
                "step_type": "thinking",
                "thought": "Formulating the solution...",
            },
        }
        res = translate_antigravity(record, session_dir="/tmp/session")
        self.assertIsNotNone(res)
        data = json.loads(res[0])
        self.assertEqual(data["type"], "thinking")
        self.assertEqual(data["content"], "Formulating the solution...")

    def test_tool_active_and_done(self) -> None:
        active_rec = {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "state": "ACTIVE",
                "tool_name": "run_command",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "ls -la"},
                },
            },
        }
        res_active = translate_antigravity(active_rec, session_dir="/tmp/session")
        self.assertIsNotNone(res_active)
        call_entry = json.loads(res_active[0])
        self.assertEqual(call_entry["type"], "message")
        self.assertEqual(call_entry["role"], "tool")
        self.assertIn("tool_use run_command", call_entry["content"])
        self.assertIn("ls -la", call_entry["content"])

        done_rec = {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "state": "DONE",
                "tool_name": "run_command",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "ls -la"},
                    "output": "file1.txt\nfile2.txt\n",
                },
            },
        }
        res_done = translate_antigravity(done_rec, session_dir="/tmp/session")
        self.assertIsNotNone(res_done)
        res_entry = json.loads(res_done[0])
        self.assertEqual(res_entry["type"], "message")
        self.assertEqual(res_entry["role"], "tool")
        self.assertIn("tool_result: file1.txt", res_entry["content"])

    def test_error_event(self) -> None:
        record = {
            "event": "error",
            "error": "Authentication failed",
        }
        res = translate_antigravity(record, session_dir="/tmp/session")
        self.assertIsNotNone(res)
        data = json.loads(res[0])
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "system")
        self.assertEqual(data["content"], "Error: Authentication failed")

    def test_translate_line_antigravity_and_agy(self) -> None:
        line = json.dumps({
            "event": "step_update",
            "step_update": {
                "step_type": "agent_response",
                "text_delta": "Task completed.",
            },
        })
        res1 = translate_line(line, fmt=ANTIGRAVITY, session_dir="/tmp/session")
        self.assertIsNotNone(res1)
        self.assertEqual(json.loads(res1[0])["content"], "Task completed.")

        res2 = translate_line(line, fmt=AGY, session_dir="/tmp/session")
        self.assertIsNotNone(res2)
        self.assertEqual(json.loads(res2[0])["content"], "Task completed.")


class BackendRegistrationTest(unittest.TestCase):
    def test_antigravity_in_writer_backends(self) -> None:
        self.assertIn("antigravity", WRITER_BACKENDS)

    def test_build_writer_adapter_antigravity(self) -> None:
        node = TaskNode(id="ch01", brief="Write intro", artifact="out/ch01.md", gates=["nonempty"])
        adapter = build_writer_adapter(
            "antigravity",
            workspace_path="/tmp/ws",
            prompt_dir="/tmp/prompts",
            node=node,
            model="gemini-3.7-flash",
        )
        self.assertIsInstance(adapter, AntigravityAdapter)
        self.assertEqual(adapter.model, "gemini-3.7-flash")
        self.assertEqual(adapter.hidden_path_exceptions, ("out/ch01.md", "scratch/ch01"))

    def test_build_research_adapter_antigravity(self) -> None:
        query = ResearchQuery(slug="q1", kind="web", question="What is X?")
        adapter = build_research_adapter(
            "antigravity",
            workspace_path="/tmp/ws",
            prompt_dir="/tmp/prompts",
            query=query,
            model="gemini-3.7-flash",
        )
        self.assertIsInstance(adapter, AntigravityAdapter)
        self.assertEqual(adapter.model, "gemini-3.7-flash")

    def test_build_role_adapter_antigravity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            adapter = build_role_adapter(
                "antigravity",
                run_dir=td,
                phase="plan",
                model="gemini-3.7-flash",
            )
            self.assertIsInstance(adapter, AntigravityAdapter)
            self.assertIn("--sandbox", adapter.command_template)

    def test_cli_and_run_accept_antigravity_backend(self) -> None:
        run_p = pipeline_run.build_parser()
        args = run_p.parse_args(["--backend", "antigravity", "--goal", "g"])
        self.assertEqual(args.backend, "antigravity")

        cli_p = pipeline_cli.build_pipeline_parser()
        args_cli = cli_p.parse_args(["run", "--backend", "antigravity", "--goal", "g"])
        self.assertEqual(args_cli.backend, "antigravity")


if __name__ == "__main__":
    unittest.main()
