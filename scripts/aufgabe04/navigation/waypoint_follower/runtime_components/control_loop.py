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
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.control.follower_safety import (
    OBSTACLE_TOO_CLOSE,
    initial_pose_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    CLEARANCE_LIMITED_MOTION_FLOOR,
    classify_linear_command,
    reachable_distance_progress_epsilon,
)
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    VelocityCommand,
    compute_join_anchor_command,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    certified_startup_join_action,
    stuck_progress_details,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    acquisition_goal_action,
    viewpoint_sampling_target_timeout_failure,
    viewpoint_sampling_timeout_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.commands import (
    finite_velocity_command as _finite_velocity_command,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
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

    def run(self) -> FollowerResult:
        if len(self.waypoints) < 2:
            return FollowerResult("noop", "fewer than two waypoints", 0.0, 0.0, False)
        started_at = time.monotonic()
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
            stop_details = dict(self.latest_stop_details or {})
            if not stop_details:
                stop_details = {
                    "reason": startup_failure,
                    "source": "initial_runtime_input_wait",
                    "fail_closed": True,
                }
            # Recovery policy must be able to distinguish this zero-motion
            # startup stop from a monitor stop after motion.  Never overwrite
            # contradictory upstream evidence: retaining it makes the later
            # classifier reject the malformed/conflicting contract.
            stop_details.setdefault("execution_phase", "before_motion")
            stop_details.setdefault("phase", "initial_runtime_input_wait")
            stop_details.setdefault(
                "motion_published",
                bool(self.motion_published),
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
                stop_details = {
                    **stop_details,
                    "controller_trace_error": trace_failure,
                    "controller_trace_fault_code": (
                        "controller_trace_write_failed"
                    ),
                }
            self.latest_stop_details = stop_details
            return FollowerResult(
                "stopped",
                startup_failure,
                time.monotonic() - started_at,
                self.distance_estimate_m,
                self.motion_published,
                stop_details,
            )
        loop_sleep_sec = 1.0 / max(self.follower_config.control_rate_hz, 1.0)
        self.control_loop_deadline_sec = time.monotonic() + loop_sleep_sec
        try:
            while rclpy.ok():
                self._drain_runtime_callbacks()
                safety_failure = self._safety_failure()
                if safety_failure:
                    if (
                        safety_failure == OBSTACLE_TOO_CLOSE
                        and self.blockage_recovery_provider is not None
                        and isinstance(
                            (self.latest_stop_details or {}).get(
                                "front_clearance"
                            ),
                            Mapping,
                        )
                        and (self.latest_stop_details or {})[
                            "front_clearance"
                        ].get("source")
                        == "front_sector"
                    ):
                        self.publish_repeated_zero()
                        recovery_pose = (
                            self._current_pose_lookup_with_stale_recovery().pose
                        )
                        if recovery_pose is not None:
                            recovery = self._attempt_blockage_recovery(
                                recovery_pose,
                                safety_failure,
                                self.latest_stop_details or {},
                            )
                            if recovery == "adopted":
                                self._hold_zero_control_period(loop_sleep_sec)
                                continue
                            if recovery == "cleared":
                                # A separately confirmed clear front sector may
                                # resume only on the next full safety cycle.
                                self.publish_zero()
                                self._hold_zero_control_period(loop_sleep_sec)
                                continue
                            if recovery == "stopped":
                                safety_failure = str(
                                    (self.latest_stop_details or {}).get(
                                        "reason",
                                        safety_failure,
                                    )
                                )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        safety_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
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
                    return FollowerResult(
                        "stopped",
                        localization_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
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
                            stop_details["controller_trace_error"] = (
                                trace_failure
                            )
                            stop_details["controller_trace_fault_code"] = (
                                "controller_trace_write_failed"
                            )
                    return FollowerResult(
                        "stopped",
                        stop_reason,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        stop_details,
                    )
                route_refresh = self._refresh_dynamic_route(pose)
                if route_refresh == "stopped":
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        str((self.latest_stop_details or {}).get("reason", "dynamic route withdrawn")),
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                if route_refresh == "completed":
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                sampling_now = time.monotonic()
                sampling_timeout = viewpoint_sampling_timeout_failure(
                    route_kind=self.current_route_kind,
                    phase_started_at=self.viewpoint_sampling_started_at,
                    now_monotonic=sampling_now,
                    timeout_sec=self.follower_config.viewpoint_sampling_timeout_sec,
                )
                if not sampling_timeout:
                    sampling_timeout = viewpoint_sampling_target_timeout_failure(
                        route_kind=self.current_route_kind,
                        target_started_at=(
                            self.viewpoint_sampling_target_started_at
                        ),
                        now_monotonic=sampling_now,
                        timeout_sec=(
                            self.follower_config.viewpoint_sampling_target_timeout_sec
                        ),
                    )
                if sampling_timeout:
                    self.latest_stop_details = {
                        "reason": sampling_timeout,
                        "route_kind": self.current_route_kind,
                        "phase_elapsed_sec": (
                            None
                            if self.viewpoint_sampling_started_at is None
                            else time.monotonic()
                            - self.viewpoint_sampling_started_at
                        ),
                        "target_elapsed_sec": (
                            None
                            if self.viewpoint_sampling_target_started_at is None
                            else time.monotonic()
                            - self.viewpoint_sampling_target_started_at
                        ),
                        "phase_timeout_sec": (
                            self.follower_config.viewpoint_sampling_timeout_sec
                        ),
                        "target_timeout_sec": (
                            self.follower_config.viewpoint_sampling_target_timeout_sec
                        ),
                        "fail_closed": True,
                    }
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        sampling_timeout.replace("_", " "),
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                if route_refresh == "adopted":
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
                if self.target_index == 0:
                    initial_failure = initial_pose_failure(
                        pose,
                        self.waypoints[0],
                        self.follower_config.initial_distance_limit_m,
                    )
                    if initial_failure:
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            initial_failure,
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                        )
                    if self.certified_static_start_pending:
                        startup_decision = certified_static_startup_decision(
                            pose,
                            self.waypoints,
                            tracking_tube_radius_m=(
                                self.follower_config.certified_route_tube_radius_m
                            ),
                            chord_sample_spacing_m=(
                                self.follower_config
                                .certified_route_chord_sample_spacing_m
                            ),
                        )
                        self.certified_static_start_pending = False
                        if not startup_decision.ok:
                            self.latest_stop_details = {
                                **startup_decision.route_check.to_log_dict(),
                                "reason": "pose outside certified startup segment",
                                "certificate_reason": (
                                    startup_decision.route_check.reason
                                ),
                                "startup_target_candidates": [0, 1],
                                "source": "execution_route_certificate",
                                "fail_closed": True,
                            }
                            self.publish_repeated_zero()
                            return FollowerResult(
                                "stopped",
                                "pose outside certified startup segment",
                                time.monotonic() - started_at,
                                self.distance_estimate_m,
                                self.motion_published,
                                self.latest_stop_details,
                            )
                        if startup_decision.target_index == 1:
                            self.target_index = 1
                            self.target_started_at = time.monotonic()
                            self._reset_progress_watchdog(time.monotonic())
                            # Make the bounded startup handoff observable as a
                            # complete zero-command control period.  No motion
                            # is published until the next loop rechecks all
                            # runtime safety inputs and the route tube.
                            self.publish_zero()
                            self._hold_zero_control_period(loop_sleep_sec)
                            continue
                if self.dynamic_join_pending:
                    join_action, join_failure = certified_startup_join_action(
                        pose,
                        self.waypoints[0],
                        self.dynamic_join_limit_m,
                        self.follower_config.dynamic_join_tolerance_m,
                    )
                    if join_action == "stop":
                        assert join_failure is not None
                        self.latest_stop_details = join_failure
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            str(join_failure["reason"]),
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    if join_action == "zero":
                        self.dynamic_join_pending = False
                        self.dynamic_join_limit_m = None
                        if self.target_index != 0:
                            self._clear_intermediate_terminal_heading_latch(
                                target_changed=True,
                            )
                        self.target_index = 0
                        self.target_started_at = time.monotonic()
                        self._reset_progress_watchdog(time.monotonic())
                        self.publish_zero()
                        self._hold_zero_control_period(loop_sleep_sec)
                        continue
                    # During handoff, pursue only the collision-certified route
                    # start.  Normal progress advancement/lookahead would form
                    # an unchecked chord from the live pose to waypoint 1.
                    step = compute_join_anchor_command(
                        pose,
                        self.waypoints[0],
                        controller_config_for_route_kind(
                            self.follower_config.controller,
                            self.current_route_kind,
                            reverse_staging=self.reverse_staging,
                            viewpoint_sampling_goal_tolerance_m=(
                                self.follower_config.viewpoint_sampling_goal_tolerance_m
                            ),
                            viewpoint_sampling_heading_tolerance_rad=(
                                self.follower_config.viewpoint_sampling_heading_tolerance_rad
                            ),
                            physical_goal_tolerance_m=(
                                self.follower_config.physical_goal_tolerance_m
                            ),
                            physical_waypoint_tolerance_m=(
                                self.follower_config.physical_waypoint_tolerance_m
                            ),
                        ),
                        join_tolerance_m=self.follower_config.dynamic_join_tolerance_m,
                    )
                else:
                    route_controller_config = controller_config_for_route_kind(
                        self.follower_config.controller,
                        self.current_route_kind,
                        reverse_staging=self.reverse_staging,
                        viewpoint_sampling_goal_tolerance_m=(
                            self.follower_config.viewpoint_sampling_goal_tolerance_m
                        ),
                        viewpoint_sampling_heading_tolerance_rad=(
                            self.follower_config.viewpoint_sampling_heading_tolerance_rad
                        ),
                        physical_goal_tolerance_m=(
                            self.follower_config.physical_goal_tolerance_m
                        ),
                        physical_waypoint_tolerance_m=(
                            self.follower_config.physical_waypoint_tolerance_m
                        ),
                    )
                    if self.start_egress_lock_index is not None:
                        step = self._start_egress_command(
                            pose,
                            route_controller_config,
                        )
                        if step is None:
                            # Make the lock-to-normal transition explicit; the
                            # next control tick may resume ordinary lookahead.
                            self.publish_zero()
                            self._hold_zero_control_period(loop_sleep_sec)
                            continue
                    elif self.start_egress_forward_alignment_index is not None:
                        step = self._reverse_egress_forward_alignment_command(
                            pose,
                            route_controller_config,
                        )
                    else:
                        corner_decision = self._certified_corner_decision(
                            pose,
                            route_controller_config,
                        )
                        if corner_decision.failure:
                            # Revoke the preceding command before logging or
                            # trace I/O can extend an in-progress rotation.
                            self.publish_zero()
                        self._log_certified_corner_phase(corner_decision.step)
                        if corner_decision.failure:
                            failed_step = corner_decision.step
                            assert failed_step is not None
                            self.latest_stop_details = {
                                "reason": corner_decision.failure,
                                "source": "execution_route_certificate",
                                "route_kind": self.current_route_kind,
                                "target_index": failed_step.target_index,
                                "pursuit_index": failed_step.pursuit_index,
                                "distance_to_vertex_m": (
                                    failed_step.distance_to_target_m
                                ),
                                "release_tolerance_m": (
                                    self.follower_config
                                    .certified_corner_release_tolerance_m
                                ),
                                "hold_tolerance_m": (
                                    self.follower_config
                                    .certified_corner_hold_tolerance_m
                                ),
                                "tracking_tube_radius_m": (
                                    self.follower_config
                                    .certified_route_tube_radius_m
                                ),
                                "reacquire_attempts": (
                                    0
                                    if self.certified_corner_latch is None
                                    else self.certified_corner_latch.reacquire_attempts
                                ),
                                "max_reacquire_attempts": (
                                    self.follower_config
                                    .certified_corner_max_reacquire_attempts
                                ),
                                "fail_closed": True,
                            }
                            failure_route_check: ExecutionRouteCheck | None = None
                            if self.current_route_kind in PHYSICAL_ROUTE_KINDS:
                                try:
                                    failure_route_check = self._execution_route_check(
                                        pose,
                                        failed_step,
                                    )
                                except (ValueError, OverflowError) as exc:
                                    self.latest_stop_details = {
                                        **self.latest_stop_details,
                                        "route_check_error": str(exc),
                                        "route_check_error_type": (
                                            exc.__class__.__name__
                                        ),
                                    }
                            trace_failure = self._append_controller_trace(
                                event="certified_corner_stop",
                                pose=pose,
                                step=failed_step,
                                route_check=failure_route_check,
                                nominal_command=failed_step.command,
                                effective_command=VelocityCommand(0.0, 0.0),
                                reason=corner_decision.failure,
                                fail_closed=True,
                            )
                            if trace_failure:
                                self.latest_stop_details = {
                                    **self.latest_stop_details,
                                    "controller_trace_error": trace_failure,
                                    "controller_trace_fault_code": (
                                        "controller_trace_write_failed"
                                    ),
                                }
                            self.publish_repeated_zero()
                            return FollowerResult(
                                "stopped",
                                corner_decision.failure,
                                time.monotonic() - started_at,
                                self.distance_estimate_m,
                                self.motion_published,
                                self.latest_stop_details,
                            )
                        if corner_decision.step is not None:
                            step = corner_decision.step
                        else:
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
                        if (
                            corner_decision.step is None
                            and terminal_heading_decision.failure
                        ):
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
                            self.latest_stop_details = {
                                "reason": terminal_heading_decision.failure,
                                "fault_code": terminal_heading_decision.failure,
                                "route_kind": self.current_route_kind,
                                "target_index": step.target_index,
                                "distance_to_target_m": step.distance_to_target_m,
                                "entry_tolerance_m": (
                                    intermediate_terminal_heading_entry_tolerance_m(
                                        route_controller_config
                                    )
                                ),
                                "hold_tolerance_m": (
                                    self.follower_config
                                    .viewpoint_sampling_terminal_heading_hold_tolerance_m
                                ),
                                "distance_comparison_epsilon_m": (
                                    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
                                ),
                                "effective_hold_limit_m": (
                                    self.follower_config
                                    .viewpoint_sampling_terminal_heading_hold_tolerance_m
                                    + INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
                                ),
                                **hold_diagnostics,
                                "fail_closed": True,
                            }
                            self.publish_repeated_zero()
                            return FollowerResult(
                                "stopped",
                                terminal_heading_decision.failure.replace("_", " "),
                                time.monotonic() - started_at,
                                self.distance_estimate_m,
                                self.motion_published,
                                self.latest_stop_details,
                            )
                route_check: ExecutionRouteCheck | None = None
                if (
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
                ):
                    route_check = self._execution_route_check(pose, step)
                    if not route_check.ok:
                        route_stop_details = route_check.to_log_dict()
                        self.latest_stop_details = route_stop_details
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
                            # Route departure remains the primary terminal
                            # safety reason even if secondary evidence storage
                            # also fails.
                            self.latest_stop_details = {
                                **route_stop_details,
                                "controller_trace_error": trace_failure,
                                "controller_trace_fault_code": (
                                    "controller_trace_write_failed"
                                ),
                                "fail_closed": True,
                            }
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            route_check.reason,
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
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
                    if self.axis_acquisition_hold_started_at is None:
                        self.axis_acquisition_hold_started_at = now_monotonic
                    hold_elapsed = (
                        now_monotonic - self.axis_acquisition_hold_started_at
                    )
                    goal_action = acquisition_goal_action(
                        route_kind=self.current_route_kind,
                        provider_available=self.waypoint_provider is not None,
                        hold_elapsed_sec=hold_elapsed,
                        timeout_sec=(
                            self.follower_config.axis_acquisition_wait_timeout_sec
                        ),
                    )
                    if goal_action == "hold_for_physical_face":
                        # Remain stationary but keep spinning sensor callbacks
                        # and polling the manifest. A physical-face revision
                        # will be adopted at the top of a subsequent cycle.
                        self.publish_zero()
                        self._hold_zero_control_period(loop_sleep_sec)
                        continue
                    if goal_action != "complete":
                        self.latest_stop_details = {
                            "reason": goal_action,
                            "route_kind": self.current_route_kind,
                            "hold_elapsed_sec": hold_elapsed,
                            "timeout_sec": (
                                self.follower_config.axis_acquisition_wait_timeout_sec
                            ),
                            "fail_closed": True,
                        }
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            goal_action.replace("_", " "),
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                timeout_now = time.monotonic()
                timeout_elapsed = timeout_now - self.target_started_at
                timeout_failure = waypoint_timeout_failure(
                    timeout_elapsed,
                    self.follower_config.waypoint_timeout_sec,
                )
                if timeout_failure:
                    self.latest_stop_details = {
                        "reason": timeout_failure,
                        "route_kind": self.current_route_kind,
                        "elapsed_sec": timeout_elapsed,
                        "timeout_sec": self.follower_config.waypoint_timeout_sec,
                        "target_index": step.target_index,
                        "pursuit_index": step.pursuit_index,
                        "distance_to_target_m": step.distance_to_target_m,
                        "progress_mode": step.progress_mode,
                        "axis_acquisition_target_revision": (
                            self.axis_acquisition_target_revision
                        ),
                        "viewpoint_sampling_target_revision": (
                            self.viewpoint_sampling_target_revision
                        ),
                        "robot_pose": {
                            "x_m": pose.x_m,
                            "y_m": pose.y_m,
                            "yaw_rad": pose.yaw_rad,
                        },
                        "fail_closed": True,
                    }
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        timeout_failure,
                        timeout_now - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                now_monotonic = time.monotonic()
                front_clearance_scale = self._motion_clearance_linear_scale(
                    step.command.linear_x_mps
                )
                effective_linear_x_mps = step.command.linear_x_mps * front_clearance_scale
                command_floor = classify_linear_command(
                    step.command.linear_x_mps,
                    effective_linear_x_mps,
                    linear_motion_floor_mps=(
                        self.follower_config.linear_motion_floor_mps
                    ),
                )
                clearance_limited_below_floor = (
                    self.current_route_kind in PHYSICAL_ROUTE_KINDS
                    and front_clearance_scale < 1.0 - 1.0e-12
                    and command_floor.zero_hold_required
                )
                if clearance_limited_below_floor:
                    self.latest_stop_details = {
                        "reason": CLEARANCE_LIMITED_MOTION_FLOOR,
                        "source": "linear_motion_floor",
                        **command_floor.to_log_dict(),
                        "front_clearance_scale": front_clearance_scale,
                        "front_clearance": dict(
                            self.latest_front_clearance_details or {}
                        ),
                        "target_index": step.target_index,
                        "pursuit_index": step.pursuit_index,
                        "distance_to_target_m": step.distance_to_target_m,
                        "progress_mode": step.progress_mode,
                        "fail_closed": True,
                    }
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
                        return FollowerResult(
                            "stopped",
                            trace_failure,
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    front_evidence = self.latest_front_clearance_details or {}
                    recovery = ""
                    if (
                        self.blockage_recovery_provider is not None
                        and step.command.linear_x_mps > 0.0
                        and front_evidence.get("source") == "front_sector"
                    ):
                        recovery = self._attempt_blockage_recovery(
                            pose,
                            CLEARANCE_LIMITED_MOTION_FLOOR,
                            self.latest_stop_details,
                        )
                    if recovery in {"adopted", "cleared"}:
                        self.publish_zero()
                        self._hold_zero_control_period(loop_sleep_sec)
                        continue
                    stop_reason = (
                        str(
                            (self.latest_stop_details or {}).get(
                                "reason",
                                CLEARANCE_LIMITED_MOTION_FLOOR,
                            )
                        )
                        if recovery == "stopped"
                        else CLEARANCE_LIMITED_MOTION_FLOOR
                    )
                    return FollowerResult(
                        "stopped",
                        stop_reason,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                distance_progress_epsilon_m = (
                    self.follower_config.stuck_progress_epsilon_m
                )
                if self.current_route_kind in PHYSICAL_ROUTE_KINDS:
                    bounded_progress_epsilon_m = (
                        reachable_distance_progress_epsilon(
                            self.follower_config.stuck_progress_epsilon_m,
                            remaining_distance_m=step.distance_to_target_m,
                            waypoint_tolerance_m=(
                                self.follower_config.physical_waypoint_tolerance_m
                            ),
                            expected_effective_travel_m=(
                                abs(effective_linear_x_mps)
                                * self.follower_config.stuck_timeout_sec
                            ),
                        )
                    )
                    if (
                        bounded_progress_epsilon_m
                        < self.follower_config.stuck_progress_epsilon_m
                    ):
                        # The comparison is strict. Half of the reachable
                        # headroom remains attainable before vertex capture.
                        distance_progress_epsilon_m = (
                            0.5 * bounded_progress_epsilon_m
                        )
                progress_failure = self._progress_failure(
                    step.distance_to_target_m,
                    step.controlled_heading_error_rad,
                    step.target_index,
                    step.pursuit_index,
                    now_monotonic,
                    (
                        abs(step.command.linear_x_mps) > 0.0
                        or abs(step.command.angular_z_radps) > 0.0
                    ),
                    step.progress_mode,
                    distance_progress_epsilon_m,
                )
                if progress_failure:
                    self.latest_stop_details = stuck_progress_details(
                        target_index=self.target_index,
                        distance_to_target_m=step.distance_to_target_m,
                        last_progress_distance_m=self.last_progress_distance_m,
                        elapsed_without_progress_sec=now_monotonic - self.last_progress_at,
                        max_without_progress_sec=self.follower_config.stuck_timeout_sec,
                        progress_epsilon_m=distance_progress_epsilon_m,
                        commanded_linear_x_mps=step.command.linear_x_mps,
                        commanded_angular_z_radps=step.command.angular_z_radps,
                        front_clearance_scale=front_clearance_scale,
                        effective_linear_x_mps=effective_linear_x_mps,
                        front_clearance_details=self.latest_front_clearance_details,
                        pursuit_index=step.pursuit_index,
                        controlled_heading_error_rad=(
                            step.controlled_heading_error_rad
                        ),
                        last_progress_heading_error_rad=(
                            self.last_progress_heading_error_rad
                        ),
                        heading_progress_epsilon_rad=(
                            self.follower_config.stuck_heading_progress_epsilon_rad
                        ),
                        last_progress_target_index=(
                            self.last_progress_target_index
                        ),
                        last_progress_pursuit_index=(
                            self.last_progress_pursuit_index
                        ),
                    )
                    self.publish_repeated_zero()
                    front_evidence = self.latest_front_clearance_details or {}
                    recovery = ""
                    if (
                        self.blockage_recovery_provider is not None
                        and step.command.linear_x_mps > 0.0
                        and front_evidence.get("source") == "front_sector"
                    ):
                        recovery = self._attempt_blockage_recovery(
                            pose,
                            progress_failure,
                            self.latest_stop_details,
                        )
                    if recovery == "adopted":
                        self._hold_zero_control_period(loop_sleep_sec)
                        continue
                    if recovery == "stopped":
                        progress_failure = str(
                            (self.latest_stop_details or {}).get(
                                "reason",
                                progress_failure,
                            )
                        )
                    if recovery == "cleared":
                        # A stuck watchdog is not discharged by clear LiDAR;
                        # _attempt_blockage_recovery converts that case into a
                        # fail-closed controller/localization diagnosis.
                        progress_failure = str(
                            (self.latest_stop_details or {}).get(
                                "reason",
                                progress_failure,
                            )
                        )
                    return FollowerResult(
                        "stopped",
                        progress_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                if not _finite_velocity_command(
                    effective_linear_x_mps,
                    step.command.angular_z_radps,
                ):
                    self.publish_repeated_zero()
                    self.latest_stop_details = {
                        "reason": "controller produced a non-finite velocity command",
                        "fault_code": "nonfinite_velocity_command",
                        "linear_x_mps": effective_linear_x_mps,
                        "angular_z_radps": step.command.angular_z_radps,
                        "fail_closed": True,
                    }
                    return FollowerResult(
                        "stopped",
                        self.latest_stop_details["reason"],
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                raw_effective_command = VelocityCommand(
                    effective_linear_x_mps,
                    step.command.angular_z_radps,
                )
                command_shape_dt_sec = (
                    loop_sleep_sec
                    if self.last_command_shape_at is None
                    else now_monotonic - self.last_command_shape_at
                )
                shaped_command = self.command_smoother.apply(
                    raw_effective_command,
                    dt_sec=command_shape_dt_sec,
                )
                self.last_command_shape_at = now_monotonic
                trace_failure = self._append_controller_trace(
                    event="control_cycle",
                    pose=pose,
                    step=step,
                    route_check=route_check,
                    nominal_command=step.command,
                    effective_command=shaped_command,
                    diagnostics={
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
                            "shape_dt_sec": command_shape_dt_sec,
                        }
                    },
                    fail_closed=False,
                )
                if trace_failure:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        trace_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
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
