import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from kusudaemon.environment.local import LocalEnvironment
from kusudaemon.pipeline.bypass import (
    clear_node_bypass,
    is_node_bypassed,
    set_node_bypass,
)
from kusudaemon.types import EpisodeBudget, EpisodeResult
from kusudaemon.v0.events import EventLog
from kusudaemon.v0.runner import run_node
from kusudaemon.v1.reviewer import ReviewVerdict
from kusudaemon.v1.round_loop import review_and_transition_node
from kusudaemon.v1.tree import TaskNode, TaskTree
from kusudaemon.v4.research import Probe, run_research_query
from kusudaemon.v4.research_loop import run_research_loop


class FakeAdapter:
    def __init__(self, actions_log: str = "done", delay: float = 0.0):
        self.actions_log = actions_log
        self.delay = delay
        self.called = False

    async def run_episode(self, prompt, env, budget, **kwargs):
        self.called = True
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return EpisodeResult(status="done", actions_log=self.actions_log)


class HangingProvider:
    def complete_json(self, *args, **kwargs):
        import time
        time.sleep(10)
        return {"verdict": "fail", "items": [{"id": "item1", "pass": False, "defect": "failed"}]}


class TestBypassStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_and_check_node_bypass(self):
        self.assertFalse(is_node_bypassed(self.run_dir, "node_01"))
        set_node_bypass(self.run_dir, "node_01")
        self.assertTrue(is_node_bypassed(self.run_dir, "node_01"))
        self.assertTrue(is_node_bypassed(self.run_dir, "node_01", "review"))
        self.assertFalse(is_node_bypassed(self.run_dir, "node_02"))

        clear_node_bypass(self.run_dir, "node_01")
        self.assertFalse(is_node_bypassed(self.run_dir, "node_01"))

    def test_process_specific_bypass(self):
        set_node_bypass(self.run_dir, "node_01", process="review")
        self.assertTrue(is_node_bypassed(self.run_dir, "node_01", "review"))
        self.assertFalse(is_node_bypassed(self.run_dir, "node_01", "writer"))

    def test_wildcard_bypass(self):
        set_node_bypass(self.run_dir, "*")
        self.assertTrue(is_node_bypassed(self.run_dir, "any_node"))
        self.assertTrue(is_node_bypassed(self.run_dir, "another_node", "review"))


class TestReviewBypass(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / "out").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "events.jsonl").touch()
        self.tree_path = self.run_dir / "tree.json"
        self.tree = TaskTree(nodes={})
        self.node = TaskNode(
            id="node_01",
            brief="test node",
            artifact="out/node_01.md",
            gates=["nonempty"],
            shape="prose-dominant",
            rubric={"style": "clear"},
            judgment=["style"],
            status="awaiting_review",
        )
        self.tree.nodes[self.node.id] = self.node
        self.tree.save(self.tree_path)
        (self.run_dir / "out" / "node_01.md").write_text("Hello world", encoding="utf-8")
        self.log = EventLog(self.run_dir / "events.jsonl")

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_pre_bypassed_review_passes_node(self):
        set_node_bypass(self.run_dir, "node_01", "review")
        provider = HangingProvider()

        await review_and_transition_node(
            self.run_dir,
            self.node,
            self.tree,
            self.tree_path,
            provider=provider,
            max_attempts=3,
            log=self.log,
        )

        self.assertEqual(self.node.status, "passed")
        events = self.log.read_all()
        types = [e.get("type") for e in events]
        self.assertIn("node_review_bypassed", types)

    async def test_mid_flight_review_bypass(self):
        provider = HangingProvider()

        async def _trigger_bypass_later():
            await asyncio.sleep(0.2)
            set_node_bypass(self.run_dir, "node_01", "review")

        asyncio.create_task(_trigger_bypass_later())

        await review_and_transition_node(
            self.run_dir,
            self.node,
            self.tree,
            self.tree_path,
            provider=provider,
            max_attempts=3,
            log=self.log,
        )

        self.assertEqual(self.node.status, "passed")
        events = self.log.read_all()
        types = [e.get("type") for e in events]
        self.assertIn("node_review_bypassed", types)


class TestResearchAndExploreBypass(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / "events.jsonl").touch()
        self.env = LocalEnvironment(tmp_dir=self.tmp.name)
        self.budget = EpisodeBudget()

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_research_query_bypassed(self):
        set_node_bypass(self.run_dir, "node_01", "research")
        adapter = FakeAdapter()
        query = Probe(slug="query_1", kind="web", question="What is Python?")

        finding = await run_research_query(
            self.run_dir,
            "node_01",
            query,
            adapter,
            self.env,
            self.budget,
        )

        self.assertFalse(adapter.called)
        self.assertEqual(finding.text, "")

    async def test_research_loop_with_bypassed_probe(self):
        tree = TaskTree(nodes={})
        node = TaskNode(id="node_01", brief="test", artifact="out/node_01.md", gates=["nonempty"], status="pending")
        tree.nodes[node.id] = node
        tree_path = self.run_dir / "tree.json"
        tree.save(tree_path)

        set_node_bypass(self.run_dir, "node_01")
        adapter = FakeAdapter()
        plan = {"node_01": [Probe(slug="q1", kind="web", question="test")]}

        results = await run_research_loop(
            self.run_dir,
            tree_path,
            plan,
            lambda n, q: adapter,
            self.env,
            self.budget,
        )

        self.assertIn("node_01", results)
        self.assertEqual(len(results["node_01"]), 1)
        self.assertEqual(results["node_01"][0].text, "")


class TestRunnerExecutionBypass(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / "events.jsonl").touch()
        (self.run_dir / "out").mkdir(parents=True, exist_ok=True)
        self.env = LocalEnvironment(tmp_dir=self.tmp.name)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_run_node_pre_bypassed(self):
        set_node_bypass(self.run_dir, "node_01", "writer")
        adapter = FakeAdapter()

        result = await run_node(
            self.run_dir,
            "node_01",
            "prompt",
            adapter,
            self.env,
            EpisodeBudget(),
        )

        self.assertFalse(adapter.called)
        self.assertEqual(result.status, "done")
        self.assertTrue(result.metadata.get("bypassed"))

    async def test_run_node_mid_flight_bypassed(self):
        adapter = FakeAdapter(delay=5.0)

        async def _trigger_bypass():
            await asyncio.sleep(0.15)
            set_node_bypass(self.run_dir, "node_01", "writer")

        asyncio.create_task(_trigger_bypass())

        result = await run_node(
            self.run_dir,
            "node_01",
            "prompt",
            adapter,
            self.env,
            EpisodeBudget(),
        )

        self.assertEqual(result.status, "done")
        self.assertTrue(result.metadata.get("bypassed"))


class TestDashboardStateAndServerBypass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name)
        self.run_dir = self.runs_root / "run_123"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "events.jsonl").touch()

    def tearDown(self):
        self.tmp.cleanup()

    def test_dashboard_state_bypass(self):
        from kusudaemon.dashboard.state import RunState
        state = RunState(self.runs_root)
        state.attach("run_123")

        ok = state.bypass_node("node_01", "review")
        self.assertTrue(ok)
        self.assertTrue(is_node_bypassed(self.run_dir, "node_01", "review"))

    def test_cli_cmd_bypass(self):
        import argparse
        from kusudaemon.pipeline.cli import cmd_bypass
        argv = argparse.Namespace(
            runs_root=str(self.runs_root),
            run_id="run_123",
            node_id="node_02",
            process="",
        )
        code = cmd_bypass(argv)
        self.assertEqual(code, 0)
        self.assertTrue(is_node_bypassed(self.run_dir, "node_02"))


if __name__ == "__main__":
    unittest.main()
