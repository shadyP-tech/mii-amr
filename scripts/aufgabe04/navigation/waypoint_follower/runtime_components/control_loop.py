"""Safety-gated control state machine for the sole follower node."""

from __future__ import annotations

import time
from typing import Mapping

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None

from scripts.aufgabe04.navigation.control.driving_behavior import (
    next_control_loop_timing,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep,
    VelocityCommand,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    certified_startup_join_action,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    ControlStepResolution,
    WaypointLifecycleDecision,
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
    control_result,
    initial_runtime_input_stop_details,
    noop_result,
    nonfinite_velocity_stop_details,
    ros_shutdown_stop_details,
    with_controller_trace_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.cycle_guard import (
    ControlCycleGuardAction,
    ControlCycleGuardRuntimeMixin,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.motion_cycle_guard import (
    MotionCycleGuardAction,
    MotionCycleGuardDecision,
    MotionCycleGuardRuntimeMixin,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.route_step_resolution import (
    RouteStepResolutionRuntimeMixin,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.route_cycle_guard import (
    RouteCycleGuardAction,
    RouteCycleGuardRuntimeMixin,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.step_cycle_guard import (
    StepCycleGuardAction,
    StepCycleGuardRuntimeMixin,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    StartupPoseAdmissionDecision,
    certified_static_startup_decision,
)

rclpy = RuntimeBindingProxy("rclpy", rclpy)


class ControlLoopRuntimeMixin(
    StepCycleGuardRuntimeMixin,
    MotionCycleGuardRuntimeMixin,
    ControlCycleGuardRuntimeMixin,
    RouteCycleGuardRuntimeMixin,
    RouteStepResolutionRuntimeMixin,
):
    """Control-loop behavior mixed into the sole follower node."""

    def _startup_pose_admission_decision(
        self,
        pose: Pose2D,
    ) -> StartupPoseAdmissionDecision:
        """Delegate startup admission through the stable node seam."""

        return super()._startup_pose_admission_decision(
            pose,
            static_startup_decision_fn=certified_static_startup_decision,
            monotonic_fn=time.monotonic,
        )

    def _resolve_control_step(
        self,
        pose: Pose2D,
    ) -> ControlStepResolution:
        """Delegate route-phase resolution through the stable node seam."""

        return super()._resolve_control_step(
            pose,
            startup_join_action_fn=certified_startup_join_action,
            monotonic_fn=time.monotonic,
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

    def _motion_cycle_guard_decision(
        self,
        pose: Pose2D,
        step: ControllerStep,
        route_check: ExecutionRouteCheck | None,
        loop_period_sec: float,
    ) -> MotionCycleGuardDecision:
        """Delegate motion admission through the stable monotonic-clock seam."""

        return super()._motion_cycle_guard_decision(
            pose,
            step,
            route_check,
            loop_period_sec,
            monotonic_fn=time.monotonic,
        )

    def _waypoint_lifecycle_decision(
        self,
        step: ControllerStep,
        pose: Pose2D,
    ) -> WaypointLifecycleDecision:
        """Delegate lifecycle decisions through the stable node seam."""

        return super()._waypoint_lifecycle_decision(
            step,
            pose,
            monotonic_fn=time.monotonic,
        )

    def run(self) -> FollowerResult:
        """Run the ordered fail-closed pipeline until completion or shutdown.

        Every cycle must pass runtime, route, step, and motion guards before the
        command is shaped, traced, and published.  The final publication and
        deadline update stay visible here because this method is the sole
        normal-motion orchestration edge.
        """

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
                cycle_guard = self._control_cycle_guard_decision(
                    loop_sleep_sec
                )
                if cycle_guard.action == ControlCycleGuardAction.RETRY:
                    continue
                if cycle_guard.action == ControlCycleGuardAction.STOP:
                    return finish(
                        "stopped",
                        cycle_guard.stop_reason,
                        stop_details=cycle_guard.stop_details,
                    )
                pose = cycle_guard.pose
                assert pose is not None
                route_guard = self._route_cycle_guard_decision(
                    pose,
                    loop_sleep_sec,
                )
                if route_guard.action == RouteCycleGuardAction.RETRY:
                    continue
                if route_guard.action == RouteCycleGuardAction.STOP:
                    return finish(
                        "stopped",
                        route_guard.stop_reason,
                        now_monotonic=route_guard.evaluated_at,
                        stop_details=route_guard.stop_details,
                    )
                if route_guard.action == RouteCycleGuardAction.COMPLETE:
                    return finish(
                        "completed",
                        "",
                        stop_details=route_guard.stop_details,
                    )
                step_guard = self._step_cycle_guard_decision(
                    pose,
                    loop_sleep_sec,
                )
                if step_guard.action == StepCycleGuardAction.RETRY:
                    continue
                if step_guard.action == StepCycleGuardAction.STOP:
                    return finish(
                        "stopped",
                        step_guard.stop_reason,
                        now_monotonic=step_guard.evaluated_at,
                        stop_details=step_guard.stop_details,
                    )
                if step_guard.action == StepCycleGuardAction.COMPLETE:
                    return finish(
                        "completed",
                        "",
                    )
                step = step_guard.step
                assert step is not None
                route_check = step_guard.route_check
                motion_guard = self._motion_cycle_guard_decision(
                    pose,
                    step,
                    route_check,
                    loop_sleep_sec,
                )
                if motion_guard.action == MotionCycleGuardAction.RETRY:
                    continue
                if motion_guard.action == MotionCycleGuardAction.STOP:
                    return finish(
                        "stopped",
                        motion_guard.stop_reason,
                        stop_details=motion_guard.stop_details,
                    )
                command_admission = motion_guard.command_admission
                now_monotonic = motion_guard.evaluated_at
                assert command_admission is not None
                assert now_monotonic is not None
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
