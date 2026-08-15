"""PLAN-EFFICIENCY-AND-HORIZON.md §M3: Content-addressed episode memoization.

Caches successful writer episodes by cryptographic digest of their complete
input set (prompt text, model, inputs' sha256 bytes, contract sha256, and
tool allowlist). A hit short-circuits episode execution while still
evaluating gates and review passes fresh from scratch.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


def compute_episode_cache_key(
    prompt_text: str,
    model: str,
    input_hashes: Iterable[tuple[str, str]],
    contract_hash: str,
    tool_allowlist: Iterable[str],
) -> str:
    """Compute SHA256 digest over the canonical representation of episode inputs."""
    h = hashlib.sha256()
    h.update(prompt_text.encode("utf-8"))
    h.update(b"||model:")
    h.update(model.encode("utf-8"))
    h.update(b"||inputs:")
    for path, sha in sorted(input_hashes):
        h.update(f"{path}:{sha};".encode("utf-8"))
    h.update(b"||contract:")
    h.update(contract_hash.encode("utf-8"))
    h.update(b"||tools:")
    for tool in sorted(tool_allowlist):
        h.update(f"{tool},".encode("utf-8"))
    return h.hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Return hex sha256 of file contents or empty hash if non-existent."""
    if not path.is_file():
        return hashlib.sha256(b"").hexdigest()
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha256(b"").hexdigest()


class EpisodeCache:
    """Disk-backed content-addressed episode cache stored in runs_root/cache."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(
        self,
        key: str,
        artifact_text: str,
        promotion: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "artifact_text": artifact_text,
            "promotion": promotion,
            "metadata": metadata or {},
        }
        path = self.cache_dir / f"{key}.json"
        temp_path = self.cache_dir / f"{key}.tmp.{hashlib.sha256(key.encode()).hexdigest()[:8]}"
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def clear(self) -> int:
        """Remove all cached entries. Returns number of removed files."""
        if not self.cache_dir.exists():
            return 0
        count = len(list(self.cache_dir.glob("*.json")))
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        return count
