"""TF, AMCL, odometry, and stale-localization recovery orchestration."""

from __future__ import annotations

import math
import time
from typing import Mapping

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from std_srvs.srv import Empty
    from tf2_ros import TransformException
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Duration = None
    Time = None
    Empty = None
    TransformException = Exception

from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.localization.localization_ownership import (
    MONITOR_ACTION_FORCE_ZERO_RESEAL,
    evaluate_global_consistency_monitor,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    evaluate_map_odom_continuity,
)
from scripts.aufgabe04.navigation.localization.tf_stale_recovery_policy import (
    OdomStationaritySample,
    StationarityDecision,
    TfEdgeSample,
    evaluate_recovery_acceptance,
    evaluate_recovery_eligibility,
    evaluate_stationarity,
)
from scripts.aufgabe04.navigation.waypoint_follower.pose_lookup import (
    SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE,
    PoseLookupResult,
    _yaw_from_quaternion,
    pose_lookup_diagnostics as _pose_lookup_diagnostics,
    ros_stamp_sec as _ros_stamp_sec,
    stale_tf_recovery_failure_details as _stale_tf_recovery_failure_details,
    tf_lookup_failure_details,
    validated_planar_pose_from_tf as _validated_planar_pose_from_tf,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.constants import (
    AMCL_STALE_TF_RECOVERY_POLL_SEC,
    SIMULATION_ODOM_FALLBACK_SOURCE,
    STALE_TF_RECOVERY_MAX_CALLBACKS,
    STALE_TF_RECOVERY_MAX_DURATION_SEC,
    STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)

rclpy = RuntimeBindingProxy("rclpy", rclpy)
Duration = RuntimeBindingProxy("Duration", Duration)
Time = RuntimeBindingProxy("Time", Time)
Empty = RuntimeBindingProxy("Empty", Empty)


class LocalizationSamplingMixin:
    """Focused localization sampling behavior."""

    def _ros_now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000.0
    def _is_real_amcl_runtime(self) -> bool:
        runtime = getattr(self, "runtime_config", None)
        return (
            getattr(self, "odom_execution_context", None) is None
            and
            getattr(runtime, "localization_source", "") == "amcl"
            and getattr(runtime, "use_sim_time", True) is False
        )
    def _tf_edge_sample(self, parent_frame: str, child_frame: str) -> TfEdgeSample:
        """Read one configured TF edge for diagnosis, never as a control pose."""

        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame,
                child_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            stamp_sec = (
                Time.from_msg(transform.header.stamp).nanoseconds
                / 1_000_000_000.0
            )
        except (TransformException, AttributeError, TypeError, ValueError):
            stamp_sec = None
        return TfEdgeSample(parent_frame, child_frame, stamp_sec)
    def _composed_tf_sample(self, lookup: PoseLookupResult) -> TfEdgeSample:
        return TfEdgeSample(
            self.runtime_config.map_frame,
            self.runtime_config.base_frame,
            lookup.stamp_sec,
        )
    def _odom_stationarity_sample(self) -> OdomStationaritySample | None:
        """Capture finite odom pose/twist evidence from one distinct callback."""

        msg = getattr(self, "latest_odom", None)
        pose = self._latest_odom_pose()
        if msg is None or pose is None:
            return None
        try:
            stamp_sec = _ros_stamp_sec(msg.header.stamp)
            linear_x_mps = float(msg.twist.twist.linear.x)
            angular_z_radps = float(msg.twist.twist.angular.z)
            callback_count = int(
                getattr(self, "latest_odom_callback_count", 0)
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if stamp_sec is None or not all(
            math.isfinite(value)
            for value in (linear_x_mps, angular_z_radps)
        ):
            return None
        try:
            return OdomStationaritySample(
                callback_count=callback_count,
                stamp_sec=stamp_sec,
                x_m=pose.x_m,
                y_m=pose.y_m,
                yaw_rad=pose.yaw_rad,
                linear_x_mps=linear_x_mps,
                angular_z_radps=angular_z_radps,
            )
        except ValueError:
            return None
    def _wait_for_stationary_odom_pair(
        self,
        *,
        deadline_monotonic: float,
    ) -> tuple[StationarityDecision | None, dict[str, object]]:
        """Prove stationarity from two fresh advancing samples under zero hold."""

        first = self._odom_stationarity_sample()
        attempts: list[dict[str, object]] = []
        if first is None:
            return None, {
                "accepted": False,
                "reason": "initial_odom_stationarity_sample_unavailable",
            }
        while rclpy.ok() and time.monotonic() < deadline_monotonic:
            self.publish_zero()
            self._service_or_wait_for_callbacks(
                min(
                    AMCL_STALE_TF_RECOVERY_POLL_SEC,
                    max(0.0, deadline_monotonic - time.monotonic()),
                )
            )
            second = self._odom_stationarity_sample()
            if second is None:
                continue
            if second.callback_count <= first.callback_count:
                continue
            decision = evaluate_stationarity(
                first,
                second,
                now_sec=self._ros_now_sec(),
            )
            attempts.append(decision.to_log_dict())
            # Retain only bounded recent evidence in a physical run artifact.
            if len(attempts) > 4:
                attempts.pop(0)
            if decision.accepted:
                return decision, {
                    "accepted": True,
                    "reason": decision.reason,
                    "first_sample": first.to_log_dict(),
                    "second_sample": second.to_log_dict(),
                    "decision": decision.to_log_dict(),
                    "attempts": attempts,
                }
            if decision.reasons == ("odom_sample_separation_too_short",):
                # TurtleBot odometry commonly arrives around 20 Hz.  Keep the
                # older sample until the pair spans the required 80 ms instead
                # of sliding forever over adjacent 50 ms callbacks.
                continue
            first = second
        return None, {
            "accepted": False,
            "reason": "stationarity_confirmation_timeout",
            "last_sample": first.to_log_dict(),
            "attempts": attempts,
        }
