"""Compatibility entrypoint for the modular Aufgabe 04 waypoint follower.

The operational implementation lives in
``scripts.aufgabe04.navigation.waypoint_follower.runtime`` so the driving
pipeline remains split into testable contracts and one ROS motion edge.  This
module aliases that runtime to preserve historical imports and monkeypatch
targets without introducing a second motion implementation.
"""

from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module


_runtime = _import_module(
    "scripts.aufgabe04.navigation.waypoint_follower.runtime"
)

_sys.modules[__name__] = _runtime
