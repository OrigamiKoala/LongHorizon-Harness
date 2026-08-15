"""Optional embedding backend (``pip install "kusudaemon[retrieval]"``).

Isolated here so the rest of the harness never imports
``sentence_transformers`` at module scope, and so the test suite — which
per CLAUDE.md must run with no optional extras installed — can check
availability and skip. Mirrors the pattern
``adapters/tools/searxng_search.py`` uses for ``gptme``.

``cosine`` is pure stdlib and separately unit-testable with hand-written
vectors and no model installed — which is what lets the §3.7 algorithm
tests in ``test_v2_survey_deterministic.py`` drive
``survey_chunks_deterministic`` with injected fake vectors.
"""

from __future__ import annotations

import math
from typing import Callable

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"  # the paper's dense encoder

_model_cache: dict[str, Callable[[list[str]], list[list[float]]]] = {}


class EmbeddingsUnavailable(RuntimeError):
    """Raised when a caller demanded embeddings without the extra installed."""


def embeddings_available() -> bool:
    """Always False — local models and sentence-transformers are not used."""
    return False


def embed_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    raise EmbeddingsUnavailable(
        "Local model embeddings are disabled. Use zero-token structural survey."
    )


def cosine(a: list[float], b: list[float]) -> float:
    """Plain dot product — inputs from ``embed_texts`` are already
    normalized. Pure stdlib, unit-testable with hand-written vectors and
    no model installed."""
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        or 1.0
    )