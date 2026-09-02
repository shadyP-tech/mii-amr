"""Pose-to-admitted-step orchestration for one follower control cycle.

This component owns startup target admission, route-phase step resolution,
route-tube admission, and waypoint lifecycle effects.  It may publish zero
commands and write stop traces, but it never admits or publishes motion,
refreshes routes, drains callbacks, or controls the normal loop cadence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time
from typing import Mapping

from scripts.aufgabe04.navigation.control.driving_behavior import (
    PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.control.follower_safety import (
    initial_pose_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep,
    VelocityCommand,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    AcquisitionGoalAction,
    StringDirective,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    ExecutionRouteAdmissionDecision,
    ExecutionRouteAdmissionStatus,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    ControlStepAction,
    ControlStepStopKind,
    RouteCommandPhase,
    WaypointLifecycleAction,
    WaypointLifecycleDecision,
    acquisition_goal_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import (
    CertifiedCornerStopEvidence,
    acquisition_goal_stop_details,
    certified_corner_stop_details,
    certified_static_start_stop_details,
    terminal_heading_timeout_stop_details,
    waypoint_timeout_stop_details,
    with_controller_trace_failure,
    with_route_check_error,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    CertifiedStaticStartupDecision,
    StartupPoseAdmissionAction,
    StartupPoseAdmissionDecision,
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading_budget import (
    TerminalHeadingBudgetState,
    reset_terminal_heading_budget,
    terminal_heading_budget_decision,
)


MonotonicClock = Callable[[], float]
StaticStartupDecision = Callable[..., CertifiedStaticStartupDecision]


class StepCycleGuardAction(StringDirective):
    """Next loop action after pose-to-step admission."""

    PROCEED = "proceed"
    RETRY = "retry"
    STOP = "stop"
    COMPLETE = "complete"


@dataclass(frozen=True)
class StepCycleGuardDecision:
    """An admitted step or the zero/terminal effect already applied for it."""

    action: StepCycleGuardAction
    step: ControllerStep | None = None
    route_check: ExecutionRouteCheck | None = None
    stop_reason: str = ""
    stop_details: Mapping[str, object] | None = None
    evaluated_at: float | None = None


class StepCycleGuardRuntimeMixin:
    """Turn one accepted pose into a route- and lifecycle-admitted step."""

    def _reset_terminal_heading_budget(
        self,
        *,
        target_index: int | None = None,
    ) -> None:
        """Reset the final-heading clock at an explicit lifecycle boundary."""

        self.terminal_heading_budget_state = reset_terminal_heading_budget(
            target_index=target_index,
        )

    def _startup_pose_admission_decision(
        self,
        pose: Pose2D,
        *,
        static_startup_decision_fn: StaticStartupDecision = (
            certified_static_startup_decision
        ),
        monotonic_fn: MonotonicClock = time.monotonic,
    ) -> StartupPoseAdmissionDecision:
        """Apply startup gates without publishing or directing loop control."""

        if self.target_index != 0:
            return StartupPoseAdmissionDecision(
                StartupPoseAdmissionAction.PROCEED,
                selected_target_index=self.target_index,
            )

        initial_failure = initial_pose_failure(
            pose,
            self.waypoints[0],
            self.follower_config.initial_distance_limit_m,
        )
        if initial_failure:
            return StartupPoseAdmissionDecision(
                StartupPoseAdmissionAction.STOP,
                stop_reason=initial_failure,
            )

        if not self.certified_static_start_pending:
            return StartupPoseAdmissionDecision(
                StartupPoseAdmissionAction.PROCEED,
                selected_target_index=0,
            )

        startup_decision = static_startup_decision_fn(
            pose,
            self.waypoints,
            tracking_tube_radius_m=(
                self.follower_config.certified_route_tube_radius_m
            ),
            chord_sample_spacing_m=(
                self.follower_config.certified_route_chord_sample_spacing_m
            ),
        )
        self.certified_static_start_pending = False
        if not startup_decision.ok:
            stop_details = certified_static_start_stop_details(
                startup_decision.route_check.to_log_dict(),
                certificate_reason=startup_decision.route_check.reason,
            )
            return StartupPoseAdmissionDecision(
                StartupPoseAdmissionAction.STOP,
                static_start_consumed=True,
                stop_reason="pose outside certified startup segment",
                stop_details=stop_details,
            )

        assert startup_decision.target_index is not None
        selected_target_index = startup_decision.target_index
        if selected_target_index == 1:
            self.target_index = 1
            self._reset_terminal_heading_budget(target_index=1)
            self.target_started_at = monotonic_fn()
            self._reset_progress_watchdog(monotonic_fn())
            return StartupPoseAdmissionDecision(
                StartupPoseAdmissionAction.ZERO_HOLD,
                selected_target_index=1,
                static_start_consumed=True,
            )
        return StartupPoseAdmissionDecision(
            StartupPoseAdmissionAction.PROCEED,
            selected_target_index=selected_target_index,
            static_start_consumed=True,
        )

    def _prepare_certified_corner_stop_evidence(
        self,
        pose: Pose2D,
        failed_step: ControllerStep,
        reason: str,
    ) -> CertifiedCornerStopEvidence:
        """Build corner-stop evidence after motion has been revoked."""

        stop_details = certified_corner_stop_details(
            reason=reason,
            route_kind=self.current_route_kind,
            target_index=failed_step.target_index,
            pursuit_index=failed_step.pursuit_index,
            distance_to_vertex_m=failed_step.distance_to_target_m,
            release_tolerance_m=(
                self.follower_config.certified_corner_release_tolerance_m
            ),
            hold_tolerance_m=(
                self.follower_config.certified_corner_hold_tolerance_m
            ),
            tracking_tube_radius_m=(
                self.follower_config.certified_route_tube_radius_m
            ),
            reacquire_attempts=(
                0
                if self.certified_corner_latch is None
                else self.certified_corner_latch.reacquire_attempts
            ),
            max_reacquire_attempts=(
                self.follower_config.certified_corner_max_reacquire_attempts
            ),
        )
        failure_route_check: ExecutionRouteCheck | None = None
        if self.current_route_kind in PHYSICAL_ROUTE_KINDS:
            try:
                failure_route_check = self._execution_route_check(
                    pose,
                    failed_step,
                )
            except (ValueError, OverflowError) as exc:
                stop_details = with_route_check_error(stop_details, exc)
        return CertifiedCornerStopEvidence(
            step=failed_step,
            stop_details=stop_details,
            route_check=failure_route_check,
        )

    def _execution_route_admission_decision(
        self,
        pose: Pose2D,
        step: ControllerStep,
    ) -> ExecutionRouteAdmissionDecision:
        """Check the active route tube without applying stop effects."""

        check_required = (
            self.current_route_kind in PHYSICAL_ROUTE_KINDS
            and (
                not self.dynamic_join_pending
                or (
                    self.dynamic_join_limit_m is not None
                    and self.dynamic_join_limit_m
                    <= self.follower_config.certified_route_tube_radius_m
                    + 1.0e-9
                )
            )
        )
        if not check_required:
            return ExecutionRouteAdmissionDecision(
                ExecutionRouteAdmissionStatus.SKIPPED
            )

        route_check = self._execution_route_check(pose, step)
        if route_check.ok:
            return ExecutionRouteAdmissionDecision(
                ExecutionRouteAdmissionStatus.ADMITTED,
                route_check=route_check,
            )
        return ExecutionRouteAdmissionDecision(
            ExecutionRouteAdmissionStatus.STOP,
            route_check=route_check,
            stop_details=route_check.to_log_dict(),
        )

    def _waypoint_lifecycle_decision(
        self,
        step: ControllerStep,
        pose: Pose2D,
        *,
        monotonic_fn: MonotonicClock = time.monotonic,
    ) -> WaypointLifecycleDecision:
        """Apply waypoint state transitions and classify the next loop effect."""

        if step.target_index != self.target_index:
            self._clear_intermediate_terminal_heading_latch(
                target_changed=True,
            )
            self.target_index = step.target_index
            self._reset_terminal_heading_budget(
                target_index=step.target_index,
            )
            self.certified_corner_latch = None
            self.target_started_at = monotonic_fn()
            self._reset_progress_watchdog(monotonic_fn())
        if step.reached_goal:
            self._reset_terminal_heading_budget()
            now_monotonic = monotonic_fn()
            goal_decision = acquisition_goal_decision(
                route_kind=self.current_route_kind,
                provider_available=self.waypoint_provider is not None,
                hold_started_at=self.axis_acquisition_hold_started_at,
                now_monotonic=now_monotonic,
                timeout_sec=(
                    self.follower_config.axis_acquisition_wait_timeout_sec
                ),
            )
            self.axis_acquisition_hold_started_at = (
                goal_decision.hold_started_at
            )
            goal_action = goal_decision.action
            if goal_action == AcquisitionGoalAction.HOLD_FOR_PHYSICAL_FACE:
                return WaypointLifecycleDecision(
                    WaypointLifecycleAction.HOLD
                )
            if goal_action == AcquisitionGoalAction.COMPLETE:
                return WaypointLifecycleDecision(
                    WaypointLifecycleAction.COMPLETE
                )
            stop_details = acquisition_goal_stop_details(
                reason=goal_action,
                route_kind=self.current_route_kind,
                hold_elapsed_sec=goal_decision.hold_elapsed_sec,
                timeout_sec=(
                    self.follower_config.axis_acquisition_wait_timeout_sec
                ),
            )
            return WaypointLifecycleDecision(
                WaypointLifecycleAction.STOP,
                stop_reason=goal_action.replace("_", " "),
                stop_details=stop_details,
            )

        timeout_now = monotonic_fn()
        timeout_elapsed = timeout_now - self.target_started_at
        timeout_failure = waypoint_timeout_failure(
            timeout_elapsed,
            self.follower_config.waypoint_timeout_sec,
        )
        terminal_heading_decision = terminal_heading_budget_decision(
            getattr(
                self,
                "terminal_heading_budget_state",
                TerminalHeadingBudgetState(target_index=self.target_index),
            ),
            target_index=step.target_index,
            final_target_index=len(self.waypoints) - 1,
            progress_mode=step.progress_mode,
            now_monotonic=timeout_now,
            timeout_sec=self.follower_config.terminal_heading_timeout_sec,
            entry_allowed=not timeout_failure,
        )
        self.terminal_heading_budget_state = terminal_heading_decision.state
        if terminal_heading_decision.active:
            if not terminal_heading_decision.failure:
                return WaypointLifecycleDecision(
                    WaypointLifecycleAction.PROCEED
                )
            terminal_heading_started_at = (
                terminal_heading_decision.state.started_at
            )
            assert terminal_heading_started_at is not None
            terminal_heading_elapsed_sec = (
                terminal_heading_decision.elapsed_sec
            )
            assert terminal_heading_elapsed_sec is not None
            stop_details = terminal_heading_timeout_stop_details(
                reason=terminal_heading_decision.failure,
                route_kind=self.current_route_kind,
                waypoint_elapsed_sec=timeout_elapsed,
                waypoint_timeout_sec=(
                    self.follower_config.waypoint_timeout_sec
                ),
                terminal_heading_elapsed_sec=(
                    terminal_heading_elapsed_sec
                ),
                terminal_heading_timeout_sec=(
                    self.follower_config.terminal_heading_timeout_sec
                ),
                terminal_heading_entry_waypoint_elapsed_sec=(
                    terminal_heading_started_at - self.target_started_at
                ),
                target_index=step.target_index,
                pursuit_index=step.pursuit_index,
                distance_to_target_m=step.distance_to_target_m,
                progress_mode=step.progress_mode,
                controlled_heading_error_rad=(
                    step.controlled_heading_error_rad
                ),
                robot_x_m=pose.x_m,
                robot_y_m=pose.y_m,
                robot_yaw_rad=pose.yaw_rad,
            )
            return WaypointLifecycleDecision(
                WaypointLifecycleAction.STOP,
                stop_reason=terminal_heading_decision.failure,
                stop_details=stop_details,
                evaluated_at=timeout_now,
            )
        if not timeout_failure:
            return WaypointLifecycleDecision(
                WaypointLifecycleAction.PROCEED
            )
        stop_details = waypoint_timeout_stop_details(
            reason=timeout_failure,
            route_kind=self.current_route_kind,
            elapsed_sec=timeout_elapsed,
            timeout_sec=self.follower_config.waypoint_timeout_sec,
            target_index=step.target_index,
            pursuit_index=step.pursuit_index,
            distance_to_target_m=step.distance_to_target_m,
            progress_mode=step.progress_mode,
            axis_acquisition_target_revision=(
                self.axis_acquisition_target_revision
            ),
            viewpoint_sampling_target_revision=(
                self.viewpoint_sampling_target_revision
            ),
            robot_x_m=pose.x_m,
            robot_y_m=pose.y_m,
            robot_yaw_rad=pose.yaw_rad,
        )
        return WaypointLifecycleDecision(
            WaypointLifecycleAction.STOP,
            stop_reason=timeout_failure,
            stop_details=stop_details,
            evaluated_at=timeout_now,
        )

    def _step_cycle_guard_decision(
        self,
        pose: Pose2D,
        loop_period_sec: float,
    ) -> StepCycleGuardDecision:
        """Produce one admitted step or apply its required zero/stop effect."""

        if self.last_pose is not None:
            self.distance_estimate_m += math.hypot(
                pose.x_m - self.last_pose.x_m,
                pose.y_m - self.last_pose.y_m,
            )
        self.last_pose = pose

        startup_admission = self._startup_pose_admission_decision(pose)
        if startup_admission.action == StartupPoseAdmissionAction.STOP:
            if startup_admission.stop_details is not None:
                self.latest_stop_details = startup_admission.stop_details
            self.publish_repeated_zero()
            return StepCycleGuardDecision(
                StepCycleGuardAction.STOP,
                stop_reason=startup_admission.stop_reason,
                stop_details=startup_admission.stop_details,
            )
        if startup_admission.action == StartupPoseAdmissionAction.ZERO_HOLD:
            # A bounded startup handoff must consume a full zero period before
            # the next cycle rechecks every runtime safety input and route tube.
            self.publish_zero()
            self._hold_zero_control_period(loop_period_sec)
            return StepCycleGuardDecision(StepCycleGuardAction.RETRY)

        step_resolution = self._resolve_control_step(pose)
        if step_resolution.stop_kind == ControlStepStopKind.CERTIFIED_CORNER:
            # Revoke the preceding command before logging or evidence I/O can
            # extend an in-progress turn.
            self.publish_zero()
        if step_resolution.command_phase == RouteCommandPhase.CERTIFIED_ROUTE:
            self._log_certified_corner_phase(step_resolution.corner_step)
        if step_resolution.action == ControlStepAction.ZERO_HOLD:
            self.publish_zero()
            self._hold_zero_control_period(loop_period_sec)
            return StepCycleGuardDecision(StepCycleGuardAction.RETRY)
        if step_resolution.action == ControlStepAction.STOP:
            self.latest_stop_details = step_resolution.stop_details
            if step_resolution.stop_kind == ControlStepStopKind.CERTIFIED_CORNER:
                failed_step = step_resolution.step
                assert failed_step is not None
                corner_stop_evidence = (
                    self._prepare_certified_corner_stop_evidence(
                        pose,
                        failed_step,
                        step_resolution.stop_reason,
                    )
                )
                self.latest_stop_details = corner_stop_evidence.stop_details
                trace_failure = self._append_controller_trace(
                    event="certified_corner_stop",
                    pose=pose,
                    step=corner_stop_evidence.step,
                    route_check=corner_stop_evidence.route_check,
                    nominal_command=corner_stop_evidence.step.command,
                    effective_command=VelocityCommand(0.0, 0.0),
                    reason=step_resolution.stop_reason,
                    fail_closed=True,
                )
                if trace_failure:
                    self.latest_stop_details = with_controller_trace_failure(
                        self.latest_stop_details,
                        trace_failure,
                    )
            self.publish_repeated_zero()
            return StepCycleGuardDecision(
                StepCycleGuardAction.STOP,
                stop_reason=step_resolution.stop_reason,
                stop_details=self.latest_stop_details,
            )

        step = step_resolution.step
        assert step is not None
        route_admission = self._execution_route_admission_decision(pose, step)
        route_check = route_admission.route_check
        if route_admission.status == ExecutionRouteAdmissionStatus.STOP:
            assert route_check is not None
            assert route_admission.stop_details is not None
            route_stop_details = route_admission.stop_details
            self.latest_stop_details = route_stop_details
            # Revoke the preceding command before trace I/O. A slow evidence
            # sink must never extend the prior nonzero Twist after departure.
            self.publish_repeated_zero()
            trace_failure = self._append_controller_trace(
                event="route_tube_stop",
                pose=pose,
                step=step,
                route_check=route_check,
                nominal_command=step.command,
                effective_command=VelocityCommand(0.0, 0.0),
                reason=route_check.reason,
                fail_closed=True,
            )
            if trace_failure:
                self.latest_stop_details = with_controller_trace_failure(
                    route_stop_details,
                    trace_failure,
                    fail_closed=True,
                )
            return StepCycleGuardDecision(
                StepCycleGuardAction.STOP,
                stop_reason=route_check.reason,
                stop_details=self.latest_stop_details,
            )

        lifecycle = self._waypoint_lifecycle_decision(step, pose)
        if lifecycle.action == WaypointLifecycleAction.HOLD:
            self.publish_zero()
            self._hold_zero_control_period(loop_period_sec)
            return StepCycleGuardDecision(StepCycleGuardAction.RETRY)
        if lifecycle.action == WaypointLifecycleAction.COMPLETE:
            self.publish_repeated_zero()
            return StepCycleGuardDecision(StepCycleGuardAction.COMPLETE)
        if lifecycle.action == WaypointLifecycleAction.STOP:
            self.latest_stop_details = lifecycle.stop_details
            self.publish_repeated_zero()
            return StepCycleGuardDecision(
                StepCycleGuardAction.STOP,
                stop_reason=lifecycle.stop_reason,
                stop_details=lifecycle.stop_details,
                evaluated_at=lifecycle.evaluated_at,
            )

        return StepCycleGuardDecision(
            StepCycleGuardAction.PROCEED,
            step=step,
            route_check=route_check,
        )
