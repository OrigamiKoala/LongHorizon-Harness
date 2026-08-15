"""Sync-over-async execution bridge for role episodes (ROLE-CALLS-VIA-BACKENDS-PLAN.md §1.4).

Provides a dedicated worker thread with its own asyncio event loop so that
synchronous `complete_json` callers can await `CommandAgentAdapter.run_episode()`
without raising `RuntimeError: cannot be called from a running event loop`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class EpisodeLoop:
    """Manages a background thread running an asyncio event loop."""

    def __init__(self, concurrency: int = 4) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max(1, concurrency))
        self._started = threading.Event()

    def _ensure_running(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                self._started.clear()
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="kusudaemon-role-episode-loop",
                    daemon=True,
                )
                self._thread.start()
                self._started.wait()
            return self._loop  # type: ignore[return-value]

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    def run_coroutine(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        timeout: float | None = None,
    ) -> T:
        """Run a coroutine on the background thread loop and block until completion."""
        loop = self._ensure_running()
        with self._semaphore:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"Episode execution timed out after {timeout}s") from exc
            except Exception:
                future.cancel()
                raise


_GLOBAL_EPISODE_LOOP: EpisodeLoop | None = None
_GLOBAL_LOCK = threading.Lock()


def get_episode_loop(concurrency: int = 4) -> EpisodeLoop:
    """Return the shared global EpisodeLoop instance."""
    global _GLOBAL_EPISODE_LOOP
    with _GLOBAL_LOCK:
        if _GLOBAL_EPISODE_LOOP is None:
            _GLOBAL_EPISODE_LOOP = EpisodeLoop(concurrency=concurrency)
        return _GLOBAL_EPISODE_LOOP
