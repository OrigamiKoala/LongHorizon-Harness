"""cost.jsonl: append-only, fsync'd cost ledger (PLAN-EFFICIENCY-AND-HORIZON.md §M1).

Records token spend and estimated USD cost for every provider call and agent episode:
{ts, role, phase, node, model, prompt_tokens, completion_tokens, reasoning_tokens, cost_usd, cached}
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Standard pricing table per 1M tokens (input, output)
MODEL_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.0, 60.0),
    "o3-mini": (1.10, 4.40),
}


def estimate_cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    if not model:
        return 0.0
    model_lower = model.lower()
    for name, (in_rate, out_rate) in MODEL_PRICING_PER_M.items():
        if name in model_lower:
            return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate
    # Default fallback: $1.00 / $3.00 per M tokens
    return (prompt_tokens / 1_000_000.0) * 1.00 + (completion_tokens / 1_000_000.0) * 3.00


from ..v1.gates import estimate_tokens


@dataclass(frozen=True)
class CostRecord:
    ts: float
    role: str
    phase: str
    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    cached: bool = False
    estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "role": self.role,
            "phase": self.phase,
            "node": self.node,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cached": self.cached,
            "estimated": self.estimated,
        }


class CostLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        role: str = "",
        phase: str = "",
        node: str = "-",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float | None = None,
        cached: bool = False,
        estimated: bool = False,
        prompt_text: str = "",
        completion_text: str = "",
        ts: float | None = None,
    ) -> CostRecord:
        if prompt_tokens == 0 and prompt_text:
            prompt_tokens = max(1, estimate_tokens(prompt_text))
            estimated = True
        if completion_tokens == 0 and completion_text:
            completion_tokens = max(1, estimate_tokens(completion_text))
            estimated = True
        if cost_usd is None:
            cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        rec = CostRecord(
            ts=ts if ts is not None else time.time(),
            role=role,
            phase=phase,
            node=node,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost_usd,
            cached=cached,
            estimated=estimated,
        )
        line = json.dumps(rec.to_dict(), sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return rec

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def totals(self) -> dict[str, Any]:
        all_recs = self.read_all()
        total_prompt = sum(r.get("prompt_tokens", 0) for r in all_recs)
        total_completion = sum(r.get("completion_tokens", 0) for r in all_recs)
        total_reasoning = sum(r.get("reasoning_tokens", 0) for r in all_recs)
        total_tokens = total_prompt + total_completion + total_reasoning
        total_cost = sum(r.get("cost_usd", 0.0) for r in all_recs)
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "reasoning_tokens": total_reasoning,
            "total_tokens": total_tokens,
            "cost_usd": round(total_cost, 6),
            "records_count": len(all_recs),
        }
