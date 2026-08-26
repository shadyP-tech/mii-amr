"""Safety-gated control state machine for the sole follower node."""

from __future__ import annotations

import math
import time
from typing import Mapping

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None

from scripts.aufgabe04.navigation.control.driving_behavior import (
    PHYSICAL_ROUTE_KINDS,
    controller_config_for_route_kind,
    next_control_loop_timing,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.control.follower_safety import (
    OBSTACLE_TOO_CLOSE,
    initial_pose_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    CLEARANCE_LIMITED_MOTION_FLOOR,
)
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep,
    VelocityCommand,
    compute_join_anchor_command,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    ExecutionRouteAdmissionDecision,
    ExecutionRouteAdmissionStatus,
    certified_startup_join_action,
)
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    AcquisitionGoalAction,
    RouteRefreshAction,
    StartupJoinAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    ControlStepAction,
    ControlStepResolution,
    ControlStepStopKind,
    RouteCommandPhase,
    WaypointLifecycleAction,
    WaypointLifecycleDecision,
    acquisition_goal_decision,
    route_command_phase,
    viewpoint_sampling_deadline_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.command_admission import (
    CommandAdmissionDecision,
    PreparedCommandDecision,
    command_shape_interval_sec,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import (
    CertifiedCornerStopEvidence,
    acquisition_goal_stop_details,
    certified_corner_stop_details,
    certified_static_start_stop_details,
    control_result,
    initial_runtime_input_stop_details,
    intermediate_terminal_heading_stop_details,
    noop_result,
    nonfinite_velocity_stop_details,
    ros_shutdown_stop_details,
    viewpoint_sampling_timeout_stop_details,
    waypoint_timeout_stop_details,
    with_controller_trace_failure,
    with_route_check_error,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.recovery_dispatch import (
    BlockageRecoveryTrigger,
    RecoveryLoopAction,
    front_sector_recovery_evidence,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    StartupPoseAdmissionAction,
    StartupPoseAdmissionDecision,
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading import (
    compute_intermediate_terminal_heading_command,
    intermediate_terminal_heading_entry_tolerance_m,
    intermediate_terminal_heading_hold_diagnostics,
)

rclpy = RuntimeBindingProxy("rclpy", rclpy)


class ControlLoopRuntimeMixin:
    """Control-loop behavior mixed into the sole follower node."""

    def _startup_pose_admission_decision(
        self,
        pose: Pose2D,
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

        startup_decision = certified_static_startup_decision(
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
            self.target_started_at = time.monotonic()
            self._reset_progress_watchdog(time.monotonic())
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

    def _resolve_control_step(
        self,
        pose: Pose2D,
    ) -> ControlStepResolution:
        """Resolve one controller step without publishing or loop control."""

        command_phase = route_command_phase(
            dynamic_join_pending=self.dynamic_join_pending,
            start_egress_lock_index=self.start_egress_lock_index,
            start_egress_forward_alignment_index=(
                self.start_egress_forward_alignment_index
            ),
        )
        if command_phase == RouteCommandPhase.DYNAMIC_JOIN:
            join_action, join_failure = certified_startup_join_action(
                pose,
                self.waypoints[0],
                self.dynamic_join_limit_m,
                self.follower_config.dynamic_join_tolerance_m,
            )
            if join_action == StartupJoinAction.STOP:
                assert join_failure is not None
                return ControlStepResolution(
                    ControlStepAction.STOP,
                    command_phase,
                    stop_kind=ControlStepStopKind.DYNAMIC_JOIN,
                    stop_reason=str(join_failure["reason"]),
                    stop_details=join_failure,
                )
            if join_action == StartupJoinAction.ZERO:
                self.dynamic_join_pending = False
                self.dynamic_join_limit_m = None
                if self.target_index != 0:
                    self._clear_intermediate_terminal_heading_latch(
                        target_changed=True,
                    )
                self.target_index = 0
                self.target_started_at = time.monotonic()
                self._reset_progress_watchdog(time.monotonic())
                return ControlStepResolution(
                    ControlStepAction.ZERO_HOLD,
                    command_phase,
                )

        route_controller_config = controller_config_for_route_kind(
            self.follower_config.controller,
            self.current_route_kind,
            reverse_staging=self.reverse_staging,
            viewpoint_sampling_goal_tolerance_m=(
                self.follower_config.viewpoint_sampling_goal_tolerance_m
            ),
            viewpoint_sampling_heading_tolerance_rad=(
                self.follower_config
                .viewpoint_sampling_heading_tolerance_rad
            ),
            physical_goal_tolerance_m=(
                self.follower_config.physical_goal_tolerance_m
            ),
            physical_waypoint_tolerance_m=(
                self.follower_config.physical_waypoint_tolerance_m
            ),
        )
        if command_phase == RouteCommandPhase.DYNAMIC_JOIN:
            # During handoff, pursue only the collision-certified route start.
            # Normal lookahead would form an unchecked chord to waypoint 1.
            step = compute_join_anchor_command(
                pose,
                self.waypoints[0],
                route_controller_config,
                join_tolerance_m=(
                    self.follower_config.dynamic_join_tolerance_m
                ),
            )
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=step,
            )
        if command_phase == RouteCommandPhase.START_EGRESS:
            step = self._start_egress_command(
                pose,
                route_controller_config,
            )
            return ControlStepResolution(
                (
                    ControlStepAction.ZERO_HOLD
                    if step is None
                    else ControlStepAction.PROCEED
                ),
                command_phase,
                step=step,
            )
        if command_phase == RouteCommandPhase.REVERSE_EGRESS_ALIGNMENT:
            step = self._reverse_egress_forward_alignment_command(
                pose,
                route_controller_config,
            )
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=step,
            )

        corner_decision = self._certified_corner_decision(
            pose,
            route_controller_config,
        )
        if corner_decision.failure:
            failed_step = corner_decision.step
            assert failed_step is not None
            return ControlStepResolution(
                ControlStepAction.STOP,
                command_phase,
                step=failed_step,
                corner_step=failed_step,
                stop_kind=ControlStepStopKind.CERTIFIED_CORNER,
                stop_reason=corner_decision.failure,
            )
        if corner_decision.step is not None:
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=corner_decision.step,
                corner_step=corner_decision.step,
            )

        terminal_heading_decision = (
            compute_intermediate_terminal_heading_command(
                pose,
                self.waypoints,
                self.target_index,
                route_controller_config,
                self.current_route_kind,
                self.intermediate_terminal_heading_latch,
                hold_tolerance_m=(
                    self.follower_config
                    .viewpoint_sampling_terminal_heading_hold_tolerance_m
                ),
                viewpoint_sampling_target_distance_m=(
                    self.follower_config
                    .viewpoint_sampling_target_distance_m
                ),
                viewpoint_sampling_target_envelope_radius_m=(
                    self.follower_config
                    .viewpoint_sampling_terminal_heading_target_envelope_radius_m
                ),
            )
        )
        self.intermediate_terminal_heading_latch = (
            terminal_heading_decision.latch
        )
        step = terminal_heading_decision.step
        if not terminal_heading_decision.failure:
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=step,
            )

        hold_diagnostics = (
            intermediate_terminal_heading_hold_diagnostics(
                pose,
                terminal_heading_decision.latch,
                hold_tolerance_m=(
                    self.follower_config
                    .viewpoint_sampling_terminal_heading_hold_tolerance_m
                ),
                viewpoint_sampling_target_distance_m=(
                    self.follower_config
                    .viewpoint_sampling_target_distance_m
                ),
                viewpoint_sampling_target_envelope_radius_m=(
                    self.follower_config
                    .viewpoint_sampling_terminal_heading_target_envelope_radius_m
                ),
            )
            if terminal_heading_decision.latch is not None
            else {}
        )
        stop_details = intermediate_terminal_heading_stop_details(
            reason=terminal_heading_decision.failure,
            route_kind=self.current_route_kind,
            target_index=step.target_index,
            distance_to_target_m=step.distance_to_target_m,
            entry_tolerance_m=(
                intermediate_terminal_heading_entry_tolerance_m(
                    route_controller_config
                )
            ),
            hold_tolerance_m=(
                self.follower_config
                .viewpoint_sampling_terminal_heading_hold_tolerance_m
            ),
            distance_comparison_epsilon_m=(
                INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
            ),
            hold_diagnostics=hold_diagnostics,
        )
        return ControlStepResolution(
            ControlStepAction.STOP,
            command_phase,
            step=step,
            stop_kind=(
                ControlStepStopKind.INTERMEDIATE_TERMINAL_HEADING
            ),
            stop_reason=terminal_heading_decision.failure.replace("_", " "),
            stop_details=stop_details,
        )

    def _prepare_command_for_publication(
        self,
        command_admission: CommandAdmissionDecision,
        *,
        now_monotonic: float,
        loop_period_sec: float,
    ) -> PreparedCommandDecision:
        """Validate and shape one admitted command without publishing it."""

        raw_effective_command = command_admission.effective_command
        if not command_admission.finite:
            return PreparedCommandDecision(
                raw_effective_command=raw_effective_command,
                shaped_command=None,
                shape_dt_sec=None,
                stop_details=nonfinite_velocity_stop_details(
                    linear_x_mps=raw_effective_command.linear_x_mps,
                    angular_z_radps=raw_effective_command.angular_z_radps,
                ),
            )

        shape_dt_sec = command_shape_interval_sec(
            loop_period_sec=loop_period_sec,
            now_monotonic=now_monotonic,
            last_shape_at=self.last_command_shape_at,
        )
        shaped_command = self.command_smoother.apply(
            raw_effective_command,
            dt_sec=shape_dt_sec,
        )
        self.last_command_shape_at = now_monotonic
        return PreparedCommandDecision(
            raw_effective_command=raw_effective_command,
            shaped_command=shaped_command,
            shape_dt_sec=shape_dt_sec,
            trace_diagnostics={
                "driving_behavior": {
                    "command_smoothing_enabled": (
                        self.follower_config.command_smoothing.enabled
                    ),
                    "unshaped_effective_command": {
                        "linear_x_mps": raw_effective_command.linear_x_mps,
                        "angular_z_radps": (
                            raw_effective_command.angular_z_radps
                        ),
                    },
                    "shape_dt_sec": shape_dt_sec,
                }
            },
        )

    def _waypoint_lifecycle_decision(
        self,
        step: ControllerStep,
        pose: Pose2D,
    ) -> WaypointLifecycleDecision:
        """Apply waypoint state transitions and classify the next loop effect."""

        if step.target_index != self.target_index:
            self._clear_intermediate_terminal_heading_latch(
                target_changed=True,
            )
            self.target_index = step.target_index
            self.certified_corner_latch = None
            self.target_started_at = time.monotonic()
            self._reset_progress_watchdog(time.monotonic())
        if step.reached_goal:
            now_monotonic = time.monotonic()
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

        timeout_now = time.monotonic()
        timeout_elapsed = timeout_now - self.target_started_at
        timeout_failure = waypoint_timeout_failure(
            timeout_elapsed,
            self.follower_config.waypoint_timeout_sec,
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

    def run(self) -> FollowerResult:
        if len(self.waypoints) < 2:
            return noop_result("fewer than two waypoints")
        started_at = time.monotonic()

        def finish(
            status: str,
            stop_reason: str,
            *,
            stop_details: Mapping[str, object] | None = None,
            now_monotonic: float | None = None,
        ) -> FollowerResult:
            """Snapshot mutable runtime counters into one result contract."""

            return control_result(
                status,
                stop_reason,
                started_at=started_at,
                now_monotonic=(
                    time.monotonic()
                    if now_monotonic is None
                    else now_monotonic
                ),
                distance_estimate_m=self.distance_estimate_m,
                motion_published=self.motion_published,
                stop_details=stop_details,
            )

        if self.current_route_kind == "viewpoint_sampling":
            self.viewpoint_sampling_started_at = started_at
            self.viewpoint_sampling_target_started_at = started_at
        self.publish_repeated_zero()
        startup_failure = self._wait_for_initial_runtime_inputs(started_at)
        if startup_failure:
            self.publish_repeated_zero()
            # ``_wait_for_initial_runtime_inputs`` leaves the most recent
            # fail-closed sensor/TF evidence in ``latest_stop_details``.  In
            # particular, the global-consistency monitor records the complete
            # map<-odom continuity decision there.  Preserve that top-level
            # contract for the semantic safety_stop instead of returning the
            # legacy five-field result that silently discarded it.
            stop_details = initial_runtime_input_stop_details(
                self.latest_stop_details,
                reason=startup_failure,
                motion_published=self.motion_published,
            )
            trace_failure = self._append_controller_trace(
                event="initial_runtime_input_stop",
                reason=startup_failure,
                fail_closed=True,
                effective_command=VelocityCommand(0.0, 0.0),
                diagnostics=stop_details,
            )
            if trace_failure:
                # The original runtime-input stop remains primary.  A
                # secondary evidence-write fault must not replace the
                # localization/sensor evidence needed by bounded recovery.
                stop_details = with_controller_trace_failure(
                    stop_details,
                    trace_failure,
                )
            self.latest_stop_details = stop_details
            return finish(
                "stopped",
                startup_failure,
                stop_details=stop_details,
            )
        loop_sleep_sec = 1.0 / max(self.follower_config.control_rate_hz, 1.0)
        self.control_loop_deadline_sec = time.monotonic() + loop_sleep_sec
        try:
            while rclpy.ok():
                self._drain_runtime_callbacks()
                safety_failure = self._safety_failure()
                if safety_failure:
                    front_evidence = front_sector_recovery_evidence(
                        self.latest_stop_details
                    )
                    if (
                        safety_failure == OBSTACLE_TOO_CLOSE
                        and self.blockage_recovery_provider is not None
                        and front_evidence is not None
                    ):
                        self.publish_repeated_zero()
                        recovery_pose = (
                            self._current_pose_lookup_with_stale_recovery().pose
                        )
                        recovery_disposition = self._blockage_recovery_outcome(
                            trigger=(
                                BlockageRecoveryTrigger.OBSTACLE_SAFETY_STOP
                            ),
                            pose=recovery_pose,
                            stop_reason=safety_failure,
                            stop_details=self.latest_stop_details,
                            front_evidence=front_evidence,
                        )
                        if (
                            recovery_disposition.action
                            == RecoveryLoopAction.HOLD_AND_RETRY
                        ):
                            self._hold_zero_control_period(loop_sleep_sec)
                            continue
                        if (
                            recovery_disposition.action
                            == RecoveryLoopAction.ZERO_HOLD_AND_RETRY
                        ):
                            # A separately confirmed clear front sector may
                            # resume only on the next full safety cycle.
                            self.publish_zero()
                            self._hold_zero_control_period(loop_sleep_sec)
                            continue
                        safety_failure = recovery_disposition.stop_reason
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        safety_failure,
                        stop_details=self.latest_stop_details,
                    )
                localization_failure = (
                    self._global_consistency_monitor_failure()
                )
                if localization_failure:
                    # LiDAR and ordinary runtime safety have already run for
                    # this cycle. Revoke the prior Twist before any monitor
                    # evidence/logging and terminate this authorization; the
                    # monitor is not permitted to steer or mutate the route.
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        localization_failure,
                        stop_details=self.latest_stop_details,
                    )
                # Endpoint graph discovery in _safety_failure can briefly
                # delay TF listener callbacks.  Recover only that resulting
                # stale-transform case, while holding an explicit zero command.
                pose_lookup = self._current_pose_lookup_with_stale_recovery()
                pose = pose_lookup.pose
                if pose is None:
                    self.publish_repeated_zero()
                    stop_reason = str(
                        (pose_lookup.details or {}).get(
                            "stop_reason",
                            "map-to-base transform unavailable",
                        )
                    )
                    stop_details = dict(pose_lookup.details or {})
                    if not stop_details.get("pose_lookup_trace_recorded"):
                        trace_failure = self._append_controller_trace(
                            event="pose_lookup_stop",
                            reason=stop_reason,
                            fail_closed=True,
                            diagnostics=stop_details,
                        )
                        if trace_failure:
                            stop_details = with_controller_trace_failure(
                                stop_details,
                                trace_failure,
                            )
                    return finish(
                        "stopped",
                        stop_reason,
                        stop_details=stop_details,
                    )
                route_refresh = self._refresh_dynamic_route(pose)
                if route_refresh == RouteRefreshAction.STOPPED:
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        str(
                            (self.latest_stop_details or {}).get(
                                "reason",
                                "dynamic route withdrawn",
                            )
                        ),
                        stop_details=self.latest_stop_details,
                    )
                if route_refresh == RouteRefreshAction.COMPLETED:
                    self.publish_repeated_zero()
                    return finish(
                        "completed",
                        "",
                        stop_details=self.latest_stop_details,
                    )
                sampling_now = time.monotonic()
                sampling_deadline = viewpoint_sampling_deadline_decision(
                    route_kind=self.current_route_kind,
                    phase_started_at=self.viewpoint_sampling_started_at,
                    target_started_at=(
                        self.viewpoint_sampling_target_started_at
                    ),
                    now_monotonic=sampling_now,
                    phase_timeout_sec=(
                        self.follower_config.viewpoint_sampling_timeout_sec
                    ),
                    target_timeout_sec=(
                        self.follower_config
                        .viewpoint_sampling_target_timeout_sec
                    ),
                )
                if sampling_deadline.failure:
                    self.latest_stop_details = (
                        viewpoint_sampling_timeout_stop_details(
                            reason=sampling_deadline.failure,
                            route_kind=self.current_route_kind,
                            phase_elapsed_sec=(
                                sampling_deadline.phase_elapsed_sec
                            ),
                            target_elapsed_sec=(
                                sampling_deadline.target_elapsed_sec
                            ),
                            phase_timeout_sec=(
                                self.follower_config
                                .viewpoint_sampling_timeout_sec
                            ),
                            target_timeout_sec=(
                                self.follower_config
                                .viewpoint_sampling_target_timeout_sec
                            ),
                        )
                    )
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        sampling_deadline.failure.replace("_", " "),
                        now_monotonic=sampling_now,
                        stop_details=self.latest_stop_details,
                    )
                if route_refresh == RouteRefreshAction.ADOPTED:
                    # A verified handoff still gets one complete zero-command
                    # control period before the new route may command motion.
                    self.publish_zero()
                    self._hold_zero_control_period(loop_sleep_sec)
                    continue
                if self.last_pose is not None:
                    self.distance_estimate_m += math.hypot(
                        pose.x_m - self.last_pose.x_m,
                        pose.y_m - self.last_pose.y_m,
                    )
                self.last_pose = pose
                startup_admission = (
                    self._startup_pose_admission_decision(pose)
                )
                if (
                    startup_admission.action
                    == StartupPoseAdmissionAction.STOP
                ):
                    if startup_admission.stop_details is not None:
                        self.latest_stop_details = (
                            startup_admission.stop_details
                        )
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        startup_admission.stop_reason,
                        stop_details=startup_admission.stop_details,
                    )
                if (
                    startup_admission.action
                    == StartupPoseAdmissionAction.ZERO_HOLD
                ):
                    # Make the bounded startup handoff observable as a
                    # complete zero-command control period.  No motion is
                    # published until the next loop rechecks every runtime
                    # safety input and the route tube.
                    self.publish_zero()
                    self._hold_zero_control_period(loop_sleep_sec)
                    continue
                step_resolution = self._resolve_control_step(pose)
                if (
                    step_resolution.stop_kind
                    == ControlStepStopKind.CERTIFIED_CORNER
                ):
                    # Revoke the preceding command before logging, route-check
                    # evidence, or trace I/O can extend an in-progress turn.
                    self.publish_zero()
                if (
                    step_resolution.command_phase
                    == RouteCommandPhase.CERTIFIED_ROUTE
                ):
                    self._log_certified_corner_phase(
                        step_resolution.corner_step
                    )
                if step_resolution.action == ControlStepAction.ZERO_HOLD:
                    self.publish_zero()
                    self._hold_zero_control_period(loop_sleep_sec)
                    continue
                if step_resolution.action == ControlStepAction.STOP:
                    self.latest_stop_details = step_resolution.stop_details
                    if (
                        step_resolution.stop_kind
                        == ControlStepStopKind.CERTIFIED_CORNER
                    ):
                        failed_step = step_resolution.step
                        assert failed_step is not None
                        corner_stop_evidence = (
                            self._prepare_certified_corner_stop_evidence(
                                pose,
                                failed_step,
                                step_resolution.stop_reason,
                            )
                        )
                        self.latest_stop_details = (
                            corner_stop_evidence.stop_details
                        )
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
                            self.latest_stop_details = (
                                with_controller_trace_failure(
                                    self.latest_stop_details,
                                    trace_failure,
                                )
                            )
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        step_resolution.stop_reason,
                        stop_details=self.latest_stop_details,
                    )
                step = step_resolution.step
                assert step is not None
                route_admission = (
                    self._execution_route_admission_decision(pose, step)
                )
                route_check = route_admission.route_check
                if (
                    route_admission.status
                    == ExecutionRouteAdmissionStatus.STOP
                ):
                    assert route_check is not None
                    assert route_admission.stop_details is not None
                    route_stop_details = route_admission.stop_details
                    self.latest_stop_details = route_stop_details
                    # Revoke the preceding command before trace I/O.  A slow
                    # or failing evidence sink must never extend the lifetime
                    # of the last nonzero Twist after route-tube departure.
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
                        # Route departure remains the primary terminal reason
                        # even if secondary evidence storage also fails.
                        self.latest_stop_details = (
                            with_controller_trace_failure(
                                route_stop_details,
                                trace_failure,
                                fail_closed=True,
                            )
                        )
                    return finish(
                        "stopped",
                        route_check.reason,
                        stop_details=self.latest_stop_details,
                    )
                lifecycle = self._waypoint_lifecycle_decision(step, pose)
                if lifecycle.action == WaypointLifecycleAction.HOLD:
                    # Remain stationary but keep spinning sensor callbacks
                    # and polling the manifest. A physical-face revision will
                    # be adopted at the top of a subsequent cycle.
                    self.publish_zero()
                    self._hold_zero_control_period(loop_sleep_sec)
                    continue
                if lifecycle.action == WaypointLifecycleAction.COMPLETE:
                    self.publish_repeated_zero()
                    return finish(
                        "completed",
                        "",
                    )
                if lifecycle.action == WaypointLifecycleAction.STOP:
                    self.latest_stop_details = lifecycle.stop_details
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        lifecycle.stop_reason,
                        now_monotonic=lifecycle.evaluated_at,
                        stop_details=lifecycle.stop_details,
                    )
                now_monotonic = time.monotonic()
                motion_admission = self._motion_command_admission_decision(step)
                command_admission = motion_admission.command_admission
                front_clearance_scale = motion_admission.front_clearance_scale
                effective_linear_x_mps = (
                    command_admission.effective_command.linear_x_mps
                )
                if motion_admission.stop_details is not None:
                    self.latest_stop_details = motion_admission.stop_details
                    self.publish_repeated_zero()
                    trace_failure = self._append_controller_trace(
                        event="motion_floor_zero_hold",
                        pose=pose,
                        step=step,
                        route_check=route_check,
                        nominal_command=step.command,
                        effective_command=VelocityCommand(0.0, 0.0),
                        reason=CLEARANCE_LIMITED_MOTION_FLOOR,
                        fail_closed=False,
                    )
                    if trace_failure:
                        return finish(
                            "stopped",
                            trace_failure,
                            stop_details=self.latest_stop_details,
                    )
                    front_evidence = self.latest_front_clearance_details or {}
                    recovery_disposition = self._blockage_recovery_outcome(
                        trigger=BlockageRecoveryTrigger.CLEARANCE_FLOOR,
                        pose=pose,
                        stop_reason=CLEARANCE_LIMITED_MOTION_FLOOR,
                        stop_details=self.latest_stop_details,
                        front_evidence=front_evidence,
                        nominal_linear_x_mps=step.command.linear_x_mps,
                    )
                    if (
                        recovery_disposition.action
                        == RecoveryLoopAction.ZERO_HOLD_AND_RETRY
                    ):
                        self.publish_zero()
                        self._hold_zero_control_period(loop_sleep_sec)
                        continue
                    return finish(
                        "stopped",
                        recovery_disposition.stop_reason,
                        stop_details=self.latest_stop_details,
                    )
                progress_decision = self._progress_watchdog_decision(
                    step,
                    now_monotonic=now_monotonic,
                    front_clearance_scale=front_clearance_scale,
                    effective_linear_x_mps=effective_linear_x_mps,
                )
                if progress_decision.failure:
                    self.latest_stop_details = progress_decision.stop_details
                    self.publish_repeated_zero()
                    front_evidence = self.latest_front_clearance_details or {}
                    recovery_disposition = self._blockage_recovery_outcome(
                        trigger=BlockageRecoveryTrigger.STUCK_WATCHDOG,
                        pose=pose,
                        stop_reason=progress_decision.failure,
                        stop_details=self.latest_stop_details,
                        front_evidence=front_evidence,
                        nominal_linear_x_mps=step.command.linear_x_mps,
                    )
                    if (
                        recovery_disposition.action
                        == RecoveryLoopAction.HOLD_AND_RETRY
                    ):
                        self._hold_zero_control_period(loop_sleep_sec)
                        continue
                    return finish(
                        "stopped",
                        recovery_disposition.stop_reason,
                        stop_details=self.latest_stop_details,
                    )
                prepared_command = self._prepare_command_for_publication(
                    command_admission,
                    now_monotonic=now_monotonic,
                    loop_period_sec=loop_sleep_sec,
                )
                if prepared_command.stop_details is not None:
                    self.latest_stop_details = prepared_command.stop_details
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        str(prepared_command.stop_details["reason"]),
                        stop_details=prepared_command.stop_details,
                    )
                shaped_command = prepared_command.shaped_command
                assert shaped_command is not None
                trace_failure = self._append_controller_trace(
                    event="control_cycle",
                    pose=pose,
                    step=step,
                    route_check=route_check,
                    nominal_command=step.command,
                    effective_command=shaped_command,
                    diagnostics=prepared_command.trace_diagnostics,
                    fail_closed=False,
                )
                if trace_failure:
                    self.publish_repeated_zero()
                    return finish(
                        "stopped",
                        trace_failure,
                        stop_details=self.latest_stop_details,
                    )
                self._publish_velocity_command(shaped_command)
                timing = next_control_loop_timing(
                    previous_deadline_sec=self.control_loop_deadline_sec,
                    now_sec=time.monotonic(),
                    control_rate_hz=self.follower_config.control_rate_hz,
                )
                self.control_loop_deadline_sec = timing.next_deadline_sec
                time.sleep(timing.sleep_sec)
        finally:
            self.publish_repeated_zero()
        shutdown_details = ros_shutdown_stop_details()
        self.latest_stop_details = shutdown_details
        return finish(
            "stopped",
            "ROS shutdown",
            stop_details=shutdown_details,
        )
