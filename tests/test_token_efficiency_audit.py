from __future__ import annotations

import json
from pathlib import Path
import pytest

from kusudaemon.eval.measure import tokens_by_role
from kusudaemon.pipeline.driver import RunOptions
from kusudaemon.pipeline.prompts import build_node_prompt, segments
from kusudaemon.roles.backend_provider import BackendRoleProvider
from kusudaemon.roles.factory import _resolve_role_transport
from kusudaemon.types import EpisodeBudget
from kusudaemon.v0.cost import CostLedger, CostRecord
from kusudaemon.v1.reviewer import VERDICT_SCHEMA
from kusudaemon.v1.tree import TaskNode
from kusudaemon.v2.retrieval import top_k_for_budget
from kusudaemon.v2.survey import Chunk, SpineUnit, _split_oversized_units, assemble_spine
from kusudaemon.v3.document_review import DOC_REVIEW_SCHEMA


def test_cost_record_token_estimation_fallback(tmp_path: Path):
    ledger_path = tmp_path / "cost.jsonl"
    ledger = CostLedger(ledger_path)

    # When explicit tokens are provided
    rec1 = ledger.record(
        role="writer",
        phase="execute",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert not rec1.estimated
    assert rec1.prompt_tokens == 100
    assert rec1.completion_tokens == 50

    # When tokens are 0 but text is provided -> estimated tokens
    rec2 = ledger.record(
        role="role",
        phase="role",
        prompt_text="Hello world, this is a test prompt for cost estimation.",
        completion_text="Here is the output text.",
    )
    assert rec2.estimated
    assert rec2.prompt_tokens > 0
    assert rec2.completion_tokens > 0

    all_records = ledger.read_all()
    assert len(all_records) == 2
    assert all_records[0]["estimated"] is False
    assert all_records[1]["estimated"] is True


def test_schema_max_items_constraints():
    assert VERDICT_SCHEMA["properties"]["items"]["maxItems"] == 12
    assert DOC_REVIEW_SCHEMA["properties"]["items"]["maxItems"] == 12


def test_episode_budget_max_output_tokens():
    budget = EpisodeBudget(max_duration_seconds=300, max_output_tokens=2048)
    assert budget.max_output_tokens == 2048

    with pytest.raises(ValueError, match="max_output_tokens"):
        EpisodeBudget(max_output_tokens=0)


def test_split_oversized_spine_units():
    # 5 chunks of 8,000 tokens each = 40,000 tokens
    chunks = [
        Chunk(index=i, text=f"Chunk {i}", tokens=8000)
        for i in range(5)
    ]
    raw_units = [[0, 4, "Mega Chapter", 40000]]
    # Split with max_tokens = 16000
    split = _split_oversized_units(raw_units, chunks, max_tokens=16000)
    assert len(split) == 3
    # First unit: chunks 0, 1 = 16k tokens
    assert split[0][0] == 0
    assert split[0][1] == 1
    assert split[0][3] == 16000
    # Second unit: chunks 2, 3 = 16k tokens
    assert split[1][0] == 2
    assert split[1][1] == 3
    assert split[1][3] == 16000
    # Third unit: chunk 4 = 8k tokens
    assert split[2][0] == 4
    assert split[2][1] == 4
    assert split[2][3] == 8000


def test_top_k_for_budget():
    # Low budget (2,000 tokens) -> minimum floor of 4
    assert top_k_for_budget(2000, avg_chunk_tokens=800) == 5
    # High budget (50,000 tokens) -> maximum ceiling of 32
    assert top_k_for_budget(50000, avg_chunk_tokens=800) == 32
    # Zero or negative budget -> DEFAULT_TOP_K (8)
    assert top_k_for_budget(0) == 8


def test_prompt_retry_resuming_skips_inlining(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    node = TaskNode(
        id="n1",
        brief="write n1",
        artifact="out/n1.md",
        gates=["nonempty"],
        type="generic",
        last_defect="gate failure: line repetition",
    )
    (run_dir / node.artifact).write_text("Previous long artifact content here...", encoding="utf-8")

    # Not resuming -> inlines prior artifact
    prompt_fresh = build_node_prompt(node, run_dir, resuming=False)
    assert "Your previous artifact" in prompt_fresh
    assert "Previous long artifact content here" in prompt_fresh

    # Resuming -> does NOT inline prior artifact
    prompt_resuming = build_node_prompt(node, run_dir, resuming=True)
    assert "Your previous artifact" not in prompt_resuming
    assert "Previous long artifact content here" not in prompt_resuming


def test_tokens_by_role_rollup():
    calls = [
        ([{"role": "user", "content": "hello world"}], VERDICT_SCHEMA),
        ([{"role": "user", "content": "another prompt"}], VERDICT_SCHEMA),
    ]
    summary = tokens_by_role(calls)
    assert "reviewer" in summary
    assert summary["reviewer"]["calls"] == 2
    assert summary["reviewer"]["input_tokens"] > 0


def test_run_options_defaults():
    options = RunOptions()
    assert options.episode_cache is True
    assert options.inline_spans is True


def test_resolve_role_transport_with_keys(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("KUSUDAEMON_ROLE_TRANSPORT", raising=False)
    monkeypatch.delenv("KUSUDAEMON_ROLE_BACKEND", raising=False)

    # Without api key -> opencode defaults to backend
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("KUSUDAEMON_PROVIDER_API_KEY", raising=False)
    assert _resolve_role_transport("opencode") == ("opencode", "backend")

    # With API key configured -> opencode resolves to http
    monkeypatch.setenv("KUSUDAEMON_PROVIDER_API_KEY", "sk-test-key-12345")
    assert _resolve_role_transport("opencode") == ("opencode", "http")
