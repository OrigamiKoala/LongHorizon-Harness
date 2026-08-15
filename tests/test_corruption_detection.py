"""Tests for pipeline/corruption.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kusudaemon.pipeline.corruption import (
    check_artifact_text_corruption,
    is_artifact_corrupted,
)
from kusudaemon.v1.tree import TaskNode


class CorruptionDetectionTest(unittest.TestCase):
    def test_empty_artifact_is_corrupted(self) -> None:
        is_corrupt, reason = check_artifact_text_corruption("")
        self.assertTrue(is_corrupt)
        self.assertIn("empty", reason)

        is_corrupt, reason = check_artifact_text_corruption("   \n\t  ")
        self.assertTrue(is_corrupt)
        self.assertIn("empty", reason)

    def test_null_bytes_is_corrupted(self) -> None:
        text = "Valid text with some words to pass length check but has null byte \x00 in the middle."
        is_corrupt, reason = check_artifact_text_corruption(text)
        self.assertTrue(is_corrupt)
        self.assertIn("null bytes", reason)

    def test_stub_artifact_is_corrupted(self) -> None:
        text = "Too short stub with only six words."
        is_corrupt, reason = check_artifact_text_corruption(text)
        self.assertTrue(is_corrupt)
        self.assertIn("stub", reason)

    def test_degenerate_repetition_loop_is_corrupted(self) -> None:
        text = "This is introductory content to set the stage.\n" + ("Repeated identical sentence line.\n" * 8)
        is_corrupt, reason = check_artifact_text_corruption(text)
        self.assertTrue(is_corrupt)
        self.assertIn("degenerate", reason)

    def test_explicit_rewrite_defect_marker_is_corrupted(self) -> None:
        text = "# Section 1\n\n" + ("This is a long substantive paragraph explaining chemistry concepts in detail. " * 5)
        is_corrupt, reason = check_artifact_text_corruption(text, defect="redispatch requested by operator [rewrite]: start fresh")
        self.assertTrue(is_corrupt)
        self.assertIn("explicit rewrite", reason)

    def test_unrecoverable_defect_is_corrupted(self) -> None:
        text = "# Section 1\n\n" + ("This is a long substantive paragraph explaining chemistry concepts in detail. " * 5)
        is_corrupt, reason = check_artifact_text_corruption(text, defect="unrecoverable structural corruption in section tree")
        self.assertTrue(is_corrupt)
        self.assertIn("unrecoverable", reason)

    def test_healthy_artifact_is_not_corrupted(self) -> None:
        text = (
            "# Introduction\n\n"
            "This is a substantive, high quality chapter explaining thermodynamics principles.\n\n"
            "## Section 1: First Law\n\n"
            "Energy is conserved in all isolated thermodynamic systems.\n"
            "Heat added to the system equals increase in internal energy plus work done.\n"
        )
        is_corrupt, reason = check_artifact_text_corruption(text)
        self.assertFalse(is_corrupt)
        self.assertEqual(reason, "healthy")

    def test_is_artifact_corrupted_file_missing(self) -> None:
        node = TaskNode(id="a", brief="Write intro", artifact="out/a.md", gates=["nonempty"])
        with tempfile.TemporaryDirectory() as run_dir:
            is_corrupt, reason = is_artifact_corrupted(run_dir, node)
            self.assertTrue(is_corrupt)
            self.assertIn("missing", reason)

    def test_is_artifact_corrupted_file_healthy(self) -> None:
        node = TaskNode(id="a", brief="Write intro", artifact="out/a.md", gates=["nonempty"])
        with tempfile.TemporaryDirectory() as run_dir:
            out_dir = Path(run_dir) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "a.md").write_text(
                "# Title\n\nSubstantive draft content that has plenty of words and paragraphs to explain the complete concept clearly without any issues.\n",
                encoding="utf-8",
            )
            is_corrupt, reason = is_artifact_corrupted(run_dir, node)
            self.assertFalse(is_corrupt)
            self.assertEqual(reason, "healthy")
