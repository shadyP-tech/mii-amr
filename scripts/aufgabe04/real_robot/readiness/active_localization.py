"""Pure orchestration for pre-mission startup active localization.

This module owns only the fail-closed sequence around initial route planning.
All effects that can touch ROS, start a subprocess, read operator input, or
write evidence are injected by the autonomous composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, Mapping

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    StartupActiveLocalizationConfig,
    startup_active_localization_attempt_dir,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    StartupRouteUncertaintySelectionRejected,
)


@dataclass(frozen=True)
class InitialCoveragePlanningAttempt:
    """Inputs for one immutable initial-route planning attempt."""

    planning_attempt_index: int
    start: Pose2D
    preflight_path: Path
    selection_evidence_path: Path | None
    propagate_route_selection_rejection: bool

    def __post_init__(self) -> None:
        if (
            type(self.planning_attempt_index) is not int
            or isinstance(self.planning_attempt_index, bool)
            or self.planning_attempt_index < 0
        ):
            raise ValueError("planning_attempt_index must be non-negative")
        if not isinstance(self.start, Pose2D):
            raise ValueError("start must be a Pose2D")
        if not isinstance(self.preflight_path, Path):
            raise ValueError("preflight_path must be a Path")
        if self.selection_evidence_path is not None and not isinstance(
            self.selection_evidence_path,
            Path,
        ):
            raise ValueError("selection_evidence_path must be a Path or None")
        if type(self.propagate_route_selection_rejection) is not bool:
            raise ValueError(
                "propagate_route_selection_rejection must be boolean"
            )


@dataclass(frozen=True)
class StartupActiveLocalizationPlanningConfig:
    """Bound one optional localization phase to one new mission session."""

    session_root: Path
    motion: StartupActiveLocalizationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.session_root, Path):
            raise ValueError("session_root must be a Path")
        if self.session_root == Path(".") or ".." in self.session_root.parts:
            raise ValueError("session_root must identify one safe session path")
        if not isinstance(self.motion, StartupActiveLocalizationConfig):
            raise ValueError("motion must be StartupActiveLocalizationConfig")

    @property
    def event_log_path(self) -> Path:
        return self.session_root / "adaptive_replans.jsonl"

    @property
    def initial_preflight_path(self) -> Path:
        return self.session_root / "preflight/preplanning_localization.json"

    def selection_evidence_path(self, planning_attempt_index: int) -> Path:
        return (
            startup_active_localization_attempt_dir(
                self.session_root,
                attempt_index=planning_attempt_index,
            )
            / "startup_route_uncertainty_selection.json"
        )

    def post_motion_preflight_path(self, attempt_index: int) -> Path:
        return (
            startup_active_localization_attempt_dir(
                self.session_root,
                attempt_index=attempt_index,
            )
            / "post_motion_preplanning_localization.json"
        )


@dataclass(frozen=True)
class StartupActiveLocalizationPlanningEffects:
    """Injected effects for the startup planning/localization state machine."""

    plan_initial_route: Callable[[InitialCoveragePlanningAttempt], int]
    run_active_localization: Callable[
        [int, StartupRouteUncertaintySelectionRejected],
        Mapping[str, object],
    ]
    admit_stationary_localization: Callable[[Path], Pose2D]
    append_event: Callable[[Path, dict[str, object]], None]
    wall_clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        for field in (
            "plan_initial_route",
            "run_active_localization",
            "admit_stationary_localization",
            "append_event",
            "wall_clock",
        ):
            if not callable(getattr(self, field)):
                raise ValueError(f"{field} must be callable")


@dataclass(frozen=True)
class InitialCoveragePlanningOutcome:
    """Successful planner status and the last freshly admitted map pose."""

    planning_status: int
    start: Pose2D
    planning_attempt_count: int
    active_localization_attempt_count: int

    def __post_init__(self) -> None:
        if type(self.planning_status) is not int:
            raise ValueError("planning_status must be an integer")
        if not isinstance(self.start, Pose2D):
            raise ValueError("start must be a Pose2D")
        for field in (
            "planning_attempt_count",
            "active_localization_attempt_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")


def _timestamp(
    effects: StartupActiveLocalizationPlanningEffects,
) -> float:
    value = effects.wall_clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError("active-localization event clock is invalid")
    return float(value)


def _validate_motion_result(result: Mapping[str, object]) -> None:
    if not isinstance(result, Mapping):
        raise RuntimeError("active-localization result is not a mapping")
    expected = {
        "status": "completed",
        "motion_published": True,
        "route_authorized": False,
        "mission_run_authorized": False,
        "requires_fresh_stationary_localization": True,
        "requires_separate_mission_run": True,
    }
    mismatches = {
        field: {"expected": wanted, "actual": result.get(field)}
        for field, wanted in expected.items()
        if result.get(field) != wanted
    }
    if mismatches:
        raise RuntimeError(
            "active-localization result violates the readiness contract: "
            f"{mismatches}"
        )


def plan_with_optional_startup_active_localization(
    config: StartupActiveLocalizationPlanningConfig,
    effects: StartupActiveLocalizationPlanningEffects,
    *,
    initial_start: Pose2D,
) -> InitialCoveragePlanningOutcome:
    """Plan, optionally localize on exact uncertainty rejection, and retry.

    Only :class:`StartupRouteUncertaintySelectionRejected` is recoverable.
    All other planner, motion, evidence, and localization failures propagate
    immediately.  A successful active-localization child still requires a
    fresh stationary localization admission and never authorizes the later
    mission ``RUN``.
    """

    if not isinstance(config, StartupActiveLocalizationPlanningConfig):
        raise ValueError(
            "config must be StartupActiveLocalizationPlanningConfig"
        )
    if not isinstance(effects, StartupActiveLocalizationPlanningEffects):
        raise ValueError(
            "effects must be StartupActiveLocalizationPlanningEffects"
        )
    if not isinstance(initial_start, Pose2D):
        raise ValueError("initial_start must be a Pose2D")

    motion = config.motion
    current_start = initial_start
    current_preflight = config.initial_preflight_path
    planning_attempt_index = 0
    active_attempt_count = 0

    if motion.enabled:
        effects.append_event(
            config.event_log_path,
            {
                "schema_version": 1,
                "event": "startup_active_localization_enabled",
                "timestamp": _timestamp(effects),
                "config": motion.to_evidence_dict(),
                "route_limits_unchanged": True,
                "motion_authorized": False,
                "mission_run_authorized": False,
            },
        )

    while True:
        attempt = InitialCoveragePlanningAttempt(
            planning_attempt_index=planning_attempt_index,
            start=current_start,
            preflight_path=current_preflight,
            selection_evidence_path=(
                config.selection_evidence_path(planning_attempt_index)
                if motion.enabled
                else None
            ),
            propagate_route_selection_rejection=motion.enabled,
        )
        if not motion.enabled:
            status = effects.plan_initial_route(attempt)
            return InitialCoveragePlanningOutcome(status, current_start, 1, 0)

        try:
            status = effects.plan_initial_route(attempt)
        except StartupRouteUncertaintySelectionRejected as rejection:
            if rejection.evidence_path != attempt.selection_evidence_path:
                raise RuntimeError(
                    "startup route-selection rejection is not bound to the "
                    "current immutable planning attempt"
                ) from rejection
            if active_attempt_count >= motion.max_attempts:
                raise RuntimeError(
                    "startup route uncertainty selection failed before "
                    "mission motion and the active-localization budget is "
                    f"exhausted: {rejection}"
                ) from rejection

            attempt_index = active_attempt_count
            effects.append_event(
                config.event_log_path,
                {
                    "schema_version": 1,
                    "event": "startup_active_localization_scheduled",
                    "timestamp": _timestamp(effects),
                    "attempt_index": attempt_index,
                    "planning_attempt_index": planning_attempt_index,
                    "rejected_selection_json": str(rejection.evidence_path),
                    "rejected_selection_sha256": rejection.evidence_sha256,
                    "rejection_reason": rejection.reason,
                    "translation_authorized": False,
                    "route_limits_unchanged": True,
                    "motion_authorized": "LOCALIZE_only",
                    "mission_run_authorized": False,
                },
            )
            result = effects.run_active_localization(
                attempt_index,
                rejection,
            )
            _validate_motion_result(result)
            preflight_path = config.post_motion_preflight_path(attempt_index)
            current_start = effects.admit_stationary_localization(
                preflight_path
            )
            if not isinstance(current_start, Pose2D):
                raise RuntimeError(
                    "fresh active-localization admission returned no Pose2D"
                )
            effects.append_event(
                config.event_log_path,
                {
                    "schema_version": 1,
                    "event": "startup_active_localization_completed",
                    "timestamp": _timestamp(effects),
                    "attempt_index": attempt_index,
                    "planning_attempt_index": planning_attempt_index,
                    "post_motion_preplanning_localization_json": str(
                        preflight_path
                    ),
                    "motion_published": True,
                    "translation_commanded": False,
                    "route_limits_unchanged": True,
                    "route_authorized": False,
                    "mission_run_authorized": False,
                    "next_planner_start": {
                        "x_m": current_start.x_m,
                        "y_m": current_start.y_m,
                        "yaw_rad": current_start.yaw_rad,
                    },
                },
            )
            active_attempt_count += 1
            planning_attempt_index += 1
            current_preflight = preflight_path
            continue

        return InitialCoveragePlanningOutcome(
            status,
            current_start,
            planning_attempt_index + 1,
            active_attempt_count,
        )


__all__ = [
    "InitialCoveragePlanningAttempt",
    "InitialCoveragePlanningOutcome",
    "StartupActiveLocalizationPlanningConfig",
    "StartupActiveLocalizationPlanningEffects",
    "plan_with_optional_startup_active_localization",
]
