from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from kusudaemon.adapters.cli_agent import extract_tokens_from_actions_log
from kusudaemon.dashboard.state import RunState, _scan_trace_usage
from kusudaemon.v0.run_dir import create_run_dir, node_scratch_dir


class TokenAndSubagentLabelsTest(TestCase):
    def test_extract_tokens_from_actions_log_with_usage_records(self) -> None:
        actions_log = "\n".join([
            json.dumps({"type": "message", "role": "assistant", "content": "hello"}),
            json.dumps({
                "type": "usage",
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "reasoning_tokens": 15,
                "total_tokens": 165,
                "cost_usd": 0.0012,
            }),
            json.dumps({
                "type": "usage",
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "reasoning_tokens": 10,
                "total_tokens": 260,
                "cost_usd": 0.0020,
            }),
        ])
        info = extract_tokens_from_actions_log(actions_log)
        self.assertEqual(info["prompt_tokens"], 320)
        self.assertEqual(info["completion_tokens"], 80)
        self.assertEqual(info["reasoning_tokens"], 25)
        self.assertEqual(info["total_tokens"], 425)
        self.assertAlmostEqual(info["cost_usd"], 0.0032)

    def test_extract_tokens_from_actions_log_fallback_estimate(self) -> None:
        actions_log = "Just plain text output without usage json"
        prompt = "Short prompt text"
        info = extract_tokens_from_actions_log(actions_log, prompt=prompt, visible_output="Just plain text output")
        self.assertGreater(info["prompt_tokens"], 0)
        self.assertGreater(info["completion_tokens"], 0)
        self.assertEqual(info["total_tokens"], info["prompt_tokens"] + info["completion_tokens"])

    def test_scan_trace_usage_and_cached_cost_totals_scratch_fallback(self) -> None:
        tmp_root = Path("/tmp/test_kusudaemon_cost_fallback")
        tmp_root.mkdir(parents=True, exist_ok=True)
        run_id = "test_run_tokens"
        run_dir = tmp_root / run_id
        create_run_dir(tmp_root, run_id)

        # Write trace file in scratch
        scratch = node_scratch_dir(run_dir, "1.1")
        scratch.mkdir(parents=True, exist_ok=True)
        trace_file = scratch / "trace.jsonl"
        trace_file.write_text(
            json.dumps({
                "type": "usage",
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "reasoning_tokens": 50,
                "total_tokens": 650,
                "cost_usd": 0.005,
            }) + "\n",
            encoding="utf-8",
        )

        tu = _scan_trace_usage(trace_file)
        self.assertEqual(tu["total_tokens"], 650)
        self.assertEqual(tu["prompt_tokens"], 500)
        self.assertEqual(tu["completion_tokens"], 100)
        self.assertEqual(tu["reasoning_tokens"], 50)

        # RunState._cached_cost_totals should pick it up even without cost.jsonl
        state = RunState(runs_root=tmp_root)
        totals = state._cached_cost_totals(run_dir)
        self.assertEqual(totals["total_tokens"], 650)
        self.assertEqual(totals["prompt_tokens"], 500)
        self.assertEqual(totals["completion_tokens"], 100)
        self.assertEqual(totals["reasoning_tokens"], 50)
        self.assertAlmostEqual(totals["cost_usd"], 0.005)
