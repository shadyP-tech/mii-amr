"""Late-bound compatibility access to ROS symbols exported by ``runtime``.

The legacy follower module aliases the runtime module, and focused offline
tests patch its ROS bindings. Components resolve those bindings at call time so
the refactor preserves that established test seam without importing runtime
and creating a cycle.
"""

from __future__ import annotations

import sys


RUNTIME_MODULE = "scripts.aufgabe04.navigation.waypoint_follower.runtime"


class RuntimeBindingProxy:
    """Delegate calls and attributes to the runtime's current binding."""

    def __init__(self, name: str, fallback: object) -> None:
        self._name = name
        self._fallback = fallback

    def _binding(self):
        runtime = sys.modules.get(RUNTIME_MODULE)
        if runtime is None:
            return self._fallback
        return getattr(runtime, self._name, self._fallback)

    def __call__(self, *args, **kwargs):
        return self._binding()(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._binding(), name)
