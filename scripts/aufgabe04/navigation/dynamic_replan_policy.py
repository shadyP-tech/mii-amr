"""Pure hysteresis and refresh policy for simulation dynamic route updates."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    MaterialTarget,
    angular_distance,
)


@dataclass(frozen=True)
class DynamicReplanState:
    target_revision: int = 0
    current_target: MaterialTarget | None = None
    last_observed_time_sec: float | None = None
    last_route_plan_time_sec: float | None = None
    last_planned_start: Pose2D | None = None
    last_planned_target_revision: int = 0
    clock_invalid: bool = False


@dataclass(frozen=True)
class DynamicReplanDecision:
    should_replan: bool
    target_revision: int
    target_changed: bool
    fail_closed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DynamicReplanPolicy:
    target_position_threshold_m: float = 0.06
    target_yaw_threshold_rad: float = math.radians(10.0)
    start_deviation_threshold_m: float = 0.15
    refresh_timeout_sec: float = 3.0
    terminal_route_lock_distance_m: float = 0.42
    replan_on_start_deviation: bool = True
    clock_rollback_tolerance_sec: float = 1e-6

    def __post_init__(self) -> None:
        if type(self.replan_on_start_deviation) is not bool:
            raise ValueError("replan_on_start_deviation must be boolean")
        for name in (
            "target_position_threshold_m",
            "target_yaw_threshold_rad",
            "start_deviation_threshold_m",
            "refresh_timeout_sec",
            "terminal_route_lock_distance_m",
            "clock_rollback_tolerance_sec",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def evaluate(
        self,
        state: DynamicReplanState,
        *,
        target: MaterialTarget | None,
        robot_pose: Pose2D,
        now_sec: float,
    ) -> tuple[DynamicReplanState, DynamicReplanDecision]:
        """Return updated observation state and a side-effect-free plan decision."""

        self._validate_state(state)
        self._validate_pose(robot_pose, "robot_pose")
        now = self._finite_nonnegative(now_sec, "now_sec")
        if state.clock_invalid:
            return state, DynamicReplanDecision(
                should_replan=False,
                target_revision=state.target_revision,
                target_changed=False,
                fail_closed=True,
                reasons=("clock_reset_required",),
            )
        if (
            state.last_observed_time_sec is not None
            and now + self.clock_rollback_tolerance_sec < state.last_observed_time_sec
        ):
            invalid_state = replace(state, clock_invalid=True)
            return invalid_state, DynamicReplanDecision(
                should_replan=False,
                target_revision=state.target_revision,
                target_changed=False,
                fail_closed=True,
                reasons=("time_moved_backwards",),
            )

        updated = replace(state, last_observed_time_sec=now)
        if target is None:
            return updated, DynamicReplanDecision(
                should_replan=False,
                target_revision=updated.target_revision,
                target_changed=False,
                fail_closed=True,
                reasons=("target_unavailable",),
            )
        self._validate_target(target)

        target_changed = self._is_material_target_change(updated.current_target, target)
        if target_changed:
            updated = replace(
                updated,
                target_revision=updated.target_revision + 1,
                current_target=target,
            )

        reasons: list[str] = []
        if updated.last_route_plan_time_sec is None:
            reasons.append("route_missing")
        if updated.target_revision != updated.last_planned_target_revision:
            reasons.append("target_revision_changed")
        # Once the robot is inside the already collision-checked terminal
        # corridor, replanning from its live pose would route it back to the
        # corridor entrance.  Keep following the installed route unless the
        # material target itself changes; the follower still enforces LiDAR
        # stops and all normal route/cmd_vel safety checks.
        terminal_route_locked = (
            updated.current_target is not None
            and updated.last_route_plan_time_sec is not None
            and updated.target_revision == updated.last_planned_target_revision
            and math.hypot(
                robot_pose.x_m - updated.current_target.pose.x_m,
                robot_pose.y_m - updated.current_target.pose.y_m,
            ) <= self.terminal_route_lock_distance_m
        )
        if (
            self.replan_on_start_deviation
            and updated.last_planned_start is not None
            and not terminal_route_locked
        ):
            deviation = math.hypot(
                robot_pose.x_m - updated.last_planned_start.x_m,
                robot_pose.y_m - updated.last_planned_start.y_m,
            )
            if deviation > 0.0 and deviation >= self.start_deviation_threshold_m:
                reasons.append("material_start_deviation")
        # Terminal locking protects immutable route geometry; it must not stop
        # the planner from publishing a freshness heartbeat for that geometry.
        # The planner recognizes a lone refresh_timeout as heartbeat-only and
        # therefore never runs live-start A* for this decision.
        if (
            updated.last_route_plan_time_sec is not None
            and now - updated.last_route_plan_time_sec >= self.refresh_timeout_sec
        ):
            reasons.append("refresh_timeout")

        return updated, DynamicReplanDecision(
            should_replan=bool(reasons),
            target_revision=updated.target_revision,
            target_changed=target_changed,
            fail_closed=False,
            reasons=tuple(reasons),
        )

    def mark_route_planned(
        self,
        state: DynamicReplanState,
        *,
        planned_start: Pose2D,
        now_sec: float,
        target_revision: int | None = None,
    ) -> DynamicReplanState:
        """Record a successful plan; failed planning must not call this method."""

        self._validate_state(state)
        if state.clock_invalid:
            raise ValueError("cannot mark a route while clock reset is unresolved")
        self._validate_pose(planned_start, "planned_start")
        now = self._finite_nonnegative(now_sec, "now_sec")
        if (
            state.last_observed_time_sec is not None
            and now + self.clock_rollback_tolerance_sec < state.last_observed_time_sec
        ):
            raise ValueError("route plan timestamp moved backwards")
        revision = state.target_revision if target_revision is None else target_revision
        if type(revision) is not int or revision != state.target_revision:
            raise ValueError("planned route revision does not match current target revision")
        return replace(
            state,
            last_observed_time_sec=max(now, state.last_observed_time_sec or 0.0),
            last_route_plan_time_sec=now,
            last_planned_start=planned_start,
            last_planned_target_revision=revision,
        )

    def reset_after_clock_change(
        self, state: DynamicReplanState, *, now_sec: float | None = None
    ) -> DynamicReplanState:
        """Explicitly clear route ownership after a simulation clock reset."""

        self._validate_state(state)
        observed = (
            None
            if now_sec is None
            else self._finite_nonnegative(now_sec, "now_sec")
        )
        return DynamicReplanState(
            target_revision=state.target_revision,
            current_target=state.current_target,
            last_observed_time_sec=observed,
            last_route_plan_time_sec=None,
            last_planned_start=None,
            last_planned_target_revision=0,
            clock_invalid=False,
        )

    def _is_material_target_change(
        self, current: MaterialTarget | None, observed: MaterialTarget
    ) -> bool:
        if current is None:
            return True
        if current.face_id != observed.face_id:
            return True
        if current.evidence_state != observed.evidence_state:
            return True
        translation = math.hypot(
            current.pose.x_m - observed.pose.x_m,
            current.pose.y_m - observed.pose.y_m,
        )
        if translation > 0.0 and translation >= self.target_position_threshold_m:
            return True
        yaw_change = angular_distance(current.pose.yaw_rad, observed.pose.yaw_rad)
        return yaw_change > 0.0 and yaw_change >= self.target_yaw_threshold_rad

    @staticmethod
    def _finite_nonnegative(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    @staticmethod
    def _validate_pose(pose: Pose2D, name: str) -> None:
        if not all(
            math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
        ):
            raise ValueError(f"{name} must be finite")

    def _validate_target(self, target: MaterialTarget) -> None:
        if not target.face_id or not target.evidence_state:
            raise ValueError("material target face_id and evidence_state are required")
        self._validate_pose(target.pose, "target.pose")

    def _validate_state(self, state: DynamicReplanState) -> None:
        if type(state.target_revision) is not int or state.target_revision < 0:
            raise ValueError("target_revision must be a non-negative integer")
        if (
            type(state.last_planned_target_revision) is not int
            or state.last_planned_target_revision < 0
            or state.last_planned_target_revision > state.target_revision
        ):
            raise ValueError("last planned target revision is invalid")
        if type(state.clock_invalid) is not bool:
            raise ValueError("clock_invalid must be boolean")
        for name in ("last_observed_time_sec", "last_route_plan_time_sec"):
            value = getattr(state, name)
            if value is not None:
                self._finite_nonnegative(value, name)
        if state.current_target is not None:
            self._validate_target(state.current_target)
        elif state.target_revision != 0:
            raise ValueError("a positive target revision requires a current target")
        if state.last_planned_start is not None:
            self._validate_pose(state.last_planned_start, "last_planned_start")
