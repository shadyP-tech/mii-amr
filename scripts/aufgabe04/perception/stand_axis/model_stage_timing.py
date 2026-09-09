"""Monotonic per-stage timing for metric image diagnostics."""

from __future__ import annotations

import time


class ModelStageTiming:
    def __init__(self) -> None:
        self._started = self._previous = time.perf_counter()
        self._stages: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._stages[name] = (now - self._previous) * 1000.0
        self._previous = now

    def snapshot(self) -> dict[str, float]:
        return {**self._stages, "total": (time.perf_counter() - self._started) * 1000.0}
