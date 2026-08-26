"""Typed, string-compatible decisions used by the follower control loop.

The enums deliberately inherit from ``str`` so existing logs, JSON payloads,
tests, callbacks, and monkeypatch seams keep seeing the same stable values.
They add type-safe comparisons without introducing ROS or runtime side effects.
"""

from __future__ import annotations

from enum import Enum


class StringDirective(str, Enum):
    """Python 3.10-compatible string enum with stable ``str()`` output."""

    def __str__(self) -> str:
        return self.value


class RouteRefreshAction(StringDirective):
    CONTINUE = ""
    ADOPTED = "adopted"
    STOPPED = "stopped"
    COMPLETED = "completed"


class BlockageRecoveryAction(StringDirective):
    NOT_ATTEMPTED = ""
    ADOPTED = "adopted"
    CLEARED = "cleared"
    STOPPED = "stopped"
    COMPLETED = "completed"


class StartupJoinAction(StringDirective):
    STOP = "stop"
    ZERO = "zero"
    ANCHOR = "anchor"


class AcquisitionGoalAction(StringDirective):
    COMPLETE = "complete"
    HOLD_FOR_PHYSICAL_FACE = "hold_for_physical_face"
    MISSING_DYNAMIC_ROUTE_PROVIDER = "missing_dynamic_route_provider"
    AXIS_ACQUISITION_TIMEOUT = "axis_acquisition_timeout"
