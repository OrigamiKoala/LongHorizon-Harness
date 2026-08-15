"""Tests for zero-token deterministic / structural survey.

Model-free, dependency-free tests driving survey_chunks_deterministic /
survey_chunks_structural using structural signals (headings, page breaks,
and token limits). No external libraries, local models, or embeddings.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from kusudaemon.v2.embeddings import cosine  # noqa: E402
from kusudaemon.v2.survey import (  # noqa: E402
    DEFAULT_CONFIDENCE_FLOOR,
    Chunk,
    assemble_spine,
    chunk_text,
    survey_chunks_deterministic,
    survey_chunks_structural,
)


def _chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(index=i, text=text, tokens=max(10, len(text.split()))) for i, text in enumerate(texts)]


class SurveyDeterministicTest(unittest.TestCase):
    def test_fewer_than_two_chunks_returns_empty(self) -> None:
        self.assertEqual(survey_chunks_deterministic(_chunks(["only"])), [])
        self.assertEqual(survey_chunks_deterministic(_chunks([])), [])

    def test_heading_emits_boundary(self) -> None:
        chunks = _chunks(
            ["Intro content", "## Chapter 2: Methods\nDetails here", "More details"]
        )
        votes = survey_chunks_deterministic(chunks, min_unit_tokens=5)
        self.assertEqual([v.boundary_after for v in votes], [0])
        self.assertEqual(votes[0].label, "Methods")

    def test_label_from_heading(self) -> None:
        from kusudaemon.v2.survey import _label_for_chunk

        label = _label_for_chunk(Chunk(index=0, text="## Photosynthesis\nbody text", tokens=3))
        self.assertEqual(label, "Photosynthesis")

    def test_label_fallback_to_first_words(self) -> None:
        from kusudaemon.v2.survey import _label_for_chunk

        chunk = Chunk(
            index=0, text="these are the first eight words used as the label here", tokens=12
        )
        self.assertEqual(
            _label_for_chunk(chunk), "these are the first eight words used as"
        )

    def test_label_truncated_to_120_chars(self) -> None:
        from kusudaemon.v2.survey import _label_for_chunk

        long_heading = "# " + "word." * 60  # > 120 chars after stripping "# "
        chunk = Chunk(index=0, text=long_heading + "\nbody text", tokens=62)
        label = _label_for_chunk(chunk)
        self.assertLessEqual(len(label), 120)
        self.assertEqual(label, ("word." * 60)[:120])

    def test_cosine_is_pure_stdlib(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()