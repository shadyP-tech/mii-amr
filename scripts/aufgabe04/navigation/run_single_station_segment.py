"""Compatibility entry point for one validated station-route segment.

The implementation lives in :mod:`scripts.aufgabe04.navigation.station_segment`.
This module deliberately aliases that canonical runtime module so existing
imports and test monkeypatches continue to target the globals used by
``main``.  Direct script execution remains supported as before.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.station_segment import runtime as _runtime


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(_runtime.main())

# A module alias, rather than copied names, is required for backwards-
# compatible monkeypatching of effects such as ``run_ros_preflight`` and
# ``run_simple_waypoint_follower``.
sys.modules[__name__] = _runtime
