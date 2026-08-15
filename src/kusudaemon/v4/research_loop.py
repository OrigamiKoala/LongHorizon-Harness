"""The v4 entrypoint: run every planned research query before the round
loop (``v1/round_loop.py``) ever starts dispatching Writers — the same
"composable library module, wiring is later" posture v2's four modules and
v3's ``assembly_loop.py`` already take (nothing here is called from
``cli.py``).

**Why research queries are not a new ``tree.json``/``TaskNode`` field.**
PLAN.md §6's ``inputs`` is already "what this node's prompt should
include" (the example: ``["source.pdf#pp.184-211"]``). A research finding
is just another input — a path to a small file the prompt-builder inlines
— so this module folds each finding's path straight into the target
node's ``inputs`` list rather than inventing a parallel field on
``TaskNode``. That keeps v1's schema (and every module built on it)
untouched; the research *plan* itself (which node needs which questions
answered) lives outside ``tree.json``, supplied by the caller the same way
v1 trees were "hand- or script-authored" before v2's planner existed.

**Ordering rationale**: like the pilot/contract freeze happening before
execution (§4.5), resolving research before dispatch means a Writer node's
prompt is fully assembled — brief, contract, inputs including any research
findings — *before* its own bounded episode starts, rather than a Writer
doing open-ended search inside its own loop and blowing its turn budget on
tool dumps (§8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..adapters.base import AgentAdapter
from ..environment.base import Environment
from ..types import EpisodeBudget
from ..v1.gates import all_passed
from ..v1.tree import TaskNode, TaskTree
from .research import ResearchFinding, ResearchQuery, run_research_query

AdapterFactory = Callable[[TaskNode, ResearchQuery], AgentAdapter]
ResearchPlan = dict[str, list[ResearchQuery]]


def attach_finding(node: TaskNode, finding: ResearchFinding) -> bool:
    """Fold a finding's path into ``node.inputs``, harness-side.

    Only when the finding cleared its own gate: an empty retrieval is a
    soft miss, not a reason to poison a node's inputs with a pointer to
    nothing (§13 v4: "these are retrieval fixes," the lowest-priority tier
    — a missing citation degrades gracefully, it doesn't block a run).
    Idempotent — a path already present is never duplicated, which is what
    makes re-running the loop after a crash safe (PLAN.md §10: resume must
    never double-apply a harness-side mutation).

    Returns whether the finding is (now, or already) attached.
    """
    if not all_passed(finding.gate_results):
        return False
    input_ref = str(finding.finding_path)
    if input_ref not in node.inputs:
        node.inputs.append(input_ref)
    return True


import asyncio


async def run_research_loop(
    run_dir: str | Path,
    tree_path: str | Path,
    plan: ResearchPlan,
    adapter_factory: AdapterFactory,
    env: Environment,
    budget: EpisodeBudget | None = None,
    *,
    max_parallel: int = 1,
) -> dict[str, list[ResearchFinding]]:
    run_dir = Path(run_dir)
    tree = TaskTree.load(tree_path)
    budget = budget or EpisodeBudget()
    results: dict[str, list[ResearchFinding]] = {}

    for node_id in plan:
        if node_id not in tree.nodes:
            raise KeyError(f"research plan references unknown node {node_id!r}")

    if max_parallel <= 1:
        for node_id, queries in plan.items():
            node = tree.nodes[node_id]
            findings: list[ResearchFinding] = []
            for query in queries:
                adapter = adapter_factory(node, query)
                finding = await run_research_query(run_dir, node_id, query, adapter, env, budget)
                findings.append(finding)
                attach_finding(node, finding)
            results[node_id] = findings
    else:
        sem = asyncio.Semaphore(max_parallel)

        async def _run_one(node_id: str, query: ResearchQuery) -> tuple[str, ResearchFinding]:
            node = tree.nodes[node_id]
            adapter = adapter_factory(node, query)
            async with sem:
                finding = await run_research_query(run_dir, node_id, query, adapter, env, budget)
            return node_id, finding

        tasks = [
            _run_one(node_id, query)
            for node_id, queries in plan.items()
            for query in queries
        ]
        gathered = await asyncio.gather(*tasks)
        for node_id, finding in gathered:
            node = tree.nodes[node_id]
            results.setdefault(node_id, []).append(finding)
            attach_finding(node, finding)

    tree.save(tree_path)
    return results
