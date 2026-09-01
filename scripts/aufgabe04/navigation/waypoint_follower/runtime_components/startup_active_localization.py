"""Bounded startup active-localization behavior for the sole motion node."""

from __future__ import annotations

import math
import time

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None

from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    RotationProgress,
    StartupActiveLocalizationConfig,
    StartupActiveLocalizationMotionResult,
    advance_rotation_progress,
    translation_from_start_m,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)


rclpy = RuntimeBindingProxy("rclpy", rclpy)


class StartupActiveLocalizationRuntimeMixin:
    """Execute one odom-controlled, scan-guarded in-place rotation."""

    def _wait_for_active_localization_inputs(self, *, deadline: float) -> str:
        """Wait under zero command for fresh scan/odom and exclusive ownership."""

        last_failure = "active localization inputs unavailable"
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_zero()
            self._service_or_wait_for_callbacks(0.05)
            scan_failure = self._freshness_failure(
                "scan",
                self.latest_scan,
                self.latest_scan_receipt,
                self.follower_config.max_scan_age_sec,
            )
            if scan_failure:
                last_failure = scan_failure
                continue
            odom_failure = self._freshness_failure(
                "odom",
                self.latest_odom,
                self.latest_odom_receipt,
                self.follower_config.max_odom_age_sec,
            )
            if odom_failure:
                last_failure = odom_failure
                continue
            pose = self._latest_odom_pose()
            if pose is None:
                last_failure = "finite odometry pose unavailable"
                continue
            safety_failure = self._safety_failure()
            if safety_failure:
                last_failure = safety_failure
                continue
            return ""
        return "ROS shutdown" if not rclpy.ok() else last_failure

    def run_startup_active_localization(
        self,
        config: StartupActiveLocalizationConfig,
        *,
        attempt_index: int,
    ) -> StartupActiveLocalizationMotionResult:
        """Run one bounded rotation and prove a stopped odometry pair afterward."""

        if not isinstance(config, StartupActiveLocalizationConfig):
            raise ValueError("config must be a StartupActiveLocalizationConfig")
        if not config.enabled:
            raise ValueError("startup active localization is not enabled")
        direction = config.direction_for_attempt(attempt_index)
        if (
            config.angular_speed_radps
            > self.follower_config.controller.max_angular_radps + 1.0e-12
        ):
            raise ValueError(
                "active-localization angular speed exceeds follower maximum"
            )
        if (
            self.follower_config.min_obstacle_distance_m + 1.0e-12
            < config.minimum_clearance_m
        ):
            raise ValueError(
                "follower obstacle distance is below active-localization clearance"
            )

        started_at = time.monotonic()
        initial_zero_count = int(getattr(self, "zero_command_publish_count", 0))
        maximum_translation_m = 0.0
        progress = RotationProgress(previous_yaw_rad=0.0)

        def finish(
            status: str,
            reason: str,
            *,
            details: dict[str, object] | None = None,
            confirm_stationarity: bool = True,
        ) -> StartupActiveLocalizationMotionResult:
            self.publish_repeated_zero(count=config.stop_command_count)
            stop_details = dict(details or {})
            stationarity_evidence: dict[str, object] | None = None
            if confirm_stationarity and rclpy.ok():
                stationarity, stationarity_evidence = (
                    self._wait_for_stationary_odom_pair(
                        deadline_monotonic=time.monotonic() + 2.0,
                    )
                )
                stop_details["stationary_odom"] = stationarity_evidence
                if status == "completed" and stationarity is None:
                    status = "stopped"
                    reason = "post-rotation odometry stationarity not confirmed"
            stop_details.update(
                {
                    "phase": "startup_active_localization",
                    "attempt_index": attempt_index,
                    "direction": "ccw" if direction > 0 else "cw",
                    "translation_commanded": False,
                    "fail_closed": status != "completed",
                }
            )
            zero_count = (
                int(getattr(self, "zero_command_publish_count", 0))
                - initial_zero_count
            )
            return StartupActiveLocalizationMotionResult(
                status=status,
                stop_reason=reason,
                duration_sec=max(0.0, time.monotonic() - started_at),
                requested_rotation_rad=config.rotation_rad,
                accumulated_progress_rad=progress.accumulated_progress_rad,
                accumulated_reverse_rad=progress.accumulated_reverse_rad,
                maximum_translation_m=maximum_translation_m,
                motion_published=bool(self.motion_published),
                zero_command_count=max(1, zero_count),
                stop_details=stop_details,
            )

        self.publish_repeated_zero(count=config.stop_command_count)
        startup_failure = self._wait_for_active_localization_inputs(
            deadline=(started_at + self.follower_config.initial_sensor_wait_sec),
        )
        if startup_failure:
            return finish(
                "stopped",
                startup_failure,
                details=dict(self.latest_stop_details or {}),
                confirm_stationarity=False,
            )

        start_pose = self._latest_odom_pose()
        if start_pose is None:
            return finish(
                "stopped",
                "finite odometry pose unavailable",
                confirm_stationarity=False,
            )
        progress = RotationProgress(previous_yaw_rad=start_pose.yaw_rad)
        start_xy_m = (start_pose.x_m, start_pose.y_m)
        last_progress_at = time.monotonic()
        last_progress_rad = 0.0
        period_sec = 1.0 / config.control_rate_hz

        while rclpy.ok():
            now = time.monotonic()
            if now - started_at > config.timeout_sec:
                return finish(
                    "stopped",
                    "startup active localization timeout",
                    details={
                        "timeout_sec": config.timeout_sec,
                        "accumulated_progress_rad": (
                            progress.accumulated_progress_rad
                        ),
                    },
                )

            self._service_or_wait_for_callbacks(min(0.02, period_sec))
            safety_failure = self._safety_failure()
            if safety_failure:
                return finish(
                    "stopped",
                    safety_failure,
                    details=dict(self.latest_stop_details or {}),
                )
            pose = self._latest_odom_pose()
            if pose is None:
                return finish(
                    "stopped",
                    "finite odometry pose unavailable during active localization",
                )

            progress = advance_rotation_progress(
                progress,
                current_yaw_rad=pose.yaw_rad,
                direction=direction,
            )
            displacement_m = translation_from_start_m(
                start_xy_m,
                (pose.x_m, pose.y_m),
            )
            maximum_translation_m = max(maximum_translation_m, displacement_m)
            if displacement_m > config.maximum_translation_m:
                return finish(
                    "stopped",
                    "active-localization in-place translation bound exceeded",
                    details={
                        "translation_from_start_m": displacement_m,
                        "maximum_translation_m": config.maximum_translation_m,
                    },
                )
            if progress.accumulated_progress_rad >= config.target_progress_rad:
                return finish(
                    "completed",
                    "",
                    details={
                        "target_progress_rad": config.target_progress_rad,
                        "accumulated_progress_rad": (
                            progress.accumulated_progress_rad
                        ),
                    },
                )

            if now - last_progress_at >= config.progress_window_sec:
                progress_delta = (
                    progress.accumulated_progress_rad - last_progress_rad
                )
                if progress_delta < config.minimum_progress_rad:
                    return finish(
                        "stopped",
                        "insufficient active-localization angular progress",
                        details={
                            "progress_window_sec": config.progress_window_sec,
                            "progress_delta_rad": progress_delta,
                            "minimum_progress_rad": config.minimum_progress_rad,
                        },
                    )
                last_progress_at = now
                last_progress_rad = progress.accumulated_progress_rad

            nominal = VelocityCommand(
                linear_x_mps=0.0,
                angular_z_radps=(
                    direction * config.angular_speed_radps
                ),
            )
            command = self.command_smoother.apply(nominal, dt_sec=period_sec)
            if (
                not math.isfinite(command.angular_z_radps)
                or abs(command.linear_x_mps) > 1.0e-12
                or abs(command.angular_z_radps)
                > config.angular_speed_radps + 1.0e-12
            ):
                return finish(
                    "stopped",
                    "invalid shaped active-localization command",
                )
            trace_failure = self._append_controller_trace(
                event="startup_active_localization_cycle",
                nominal_command=nominal,
                effective_command=command,
                diagnostics={
                    "attempt_index": attempt_index,
                    "direction": "ccw" if direction > 0 else "cw",
                    "accumulated_progress_rad": (
                        progress.accumulated_progress_rad
                    ),
                    "accumulated_reverse_rad": (
                        progress.accumulated_reverse_rad
                    ),
                    "translation_from_start_m": displacement_m,
                    "translation_commanded": False,
                },
                fail_closed=False,
            )
            if trace_failure:
                return finish(
                    "stopped",
                    trace_failure,
                    details=dict(self.latest_stop_details or {}),
                )
            self._publish_velocity_command(command)
            time.sleep(period_sec)

        return finish(
            "stopped",
            "ROS shutdown during startup active localization",
            confirm_stationarity=False,
        )


__all__ = ["StartupActiveLocalizationRuntimeMixin"]
