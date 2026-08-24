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

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.localization_ownership import (
    MONITOR_ACTION_FORCE_ZERO_RESEAL,
    evaluate_global_consistency_monitor,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.odom_route_adapter import (
    evaluate_map_odom_continuity,
)
from scripts.aufgabe04.navigation.tf_stale_recovery_policy import (
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


class SimulationOdomRecoveryMixin:
    """Focused simulation odom recovery behavior."""

    def _simulation_odom_fallback_after_stale_retry(
        self,
        *,
        first_lookup: PoseLookupResult,
        retry_lookup: PoseLookupResult,
        callback_drain: dict[str, object],
        odom_callback_count_before: int,
        odom_callback_count_after: int,
        odom_msg,
        odom_receipt,
        scan_msg,
        scan_receipt,
    ) -> PoseLookupResult:
        """Validate one explicit Gazebo-only direct-odometry recovery."""

        runtime = getattr(self, "runtime_config", None)
        follower_config = getattr(self, "follower_config", None)
        enabled = (
            getattr(
                follower_config,
                "allow_simulation_odom_after_stale_tf",
                False,
            )
            is True
        )
        if not enabled:
            disabled_details = {
                "source": SIMULATION_ODOM_FALLBACK_SOURCE,
                "pose_source": SIMULATION_ODOM_FALLBACK_SOURCE,
                "attempted": False,
                "accepted": False,
                "fail_closed": True,
                "zero_published_before_fallback": True,
                "not_real_robot_migration_evidence": True,
                "first_lookup": _pose_lookup_diagnostics(first_lookup),
                "retry_lookup": _pose_lookup_diagnostics(retry_lookup),
                "callback_drain": callback_drain,
                "predicates": {
                    "explicitly_enabled": False,
                },
                "rejection_reasons": ["explicitly_enabled"],
            }
            original_failure = _stale_tf_recovery_failure_details(
                dict(retry_lookup.details or {}),
                first_lookup=first_lookup,
                retry_lookup=retry_lookup,
                callback_drain=callback_drain,
            )
            original_failure["simulation_odom_fallback"] = disabled_details
            return PoseLookupResult(
                None,
                original_failure,
                retry_lookup.stamp_sec,
            )
        use_sim_time = getattr(runtime, "use_sim_time", False) is True
        localization_is_tf = getattr(runtime, "localization_source", "") == "tf"
        map_frame = str(getattr(runtime, "map_frame", ""))
        odom_frame = str(getattr(runtime, "odom_frame", ""))
        base_frame = str(getattr(runtime, "base_frame", ""))
        map_frame_is_odom_frame = bool(map_frame) and map_frame == odom_frame

        odom_freshness = self._fallback_message_freshness_evidence(
            "odom",
            odom_msg,
            odom_receipt,
            getattr(follower_config, "max_odom_age_sec", 1.0),
        )
        scan_freshness = self._fallback_message_freshness_evidence(
            "scan",
            scan_msg,
            scan_receipt,
            getattr(follower_config, "max_scan_age_sec", 1.0),
        )

        header = getattr(odom_msg, "header", None)
        odom_message_frame = str(getattr(header, "frame_id", ""))
        odom_child_frame = str(getattr(odom_msg, "child_frame_id", ""))
        odom_stamp_sec = _ros_stamp_sec(getattr(header, "stamp", None))
        pose_message = getattr(getattr(odom_msg, "pose", None), "pose", None)
        position = getattr(pose_message, "position", None)
        orientation = getattr(pose_message, "orientation", None)

        def numeric(attribute_owner, attribute_name: str) -> float | None:
            try:
                value = float(getattr(attribute_owner, attribute_name))
            except (AttributeError, TypeError, ValueError, OverflowError):
                return None
            return value if math.isfinite(value) else None

        x_m = numeric(position, "x")
        y_m = numeric(position, "y")
        quaternion = {
            field_name: numeric(orientation, field_name)
            for field_name in ("x", "y", "z", "w")
        }
        quaternion_finite = all(
            value is not None for value in quaternion.values()
        )
        quaternion_norm = (
            math.sqrt(
                sum(float(value) ** 2 for value in quaternion.values())
            )
            if quaternion_finite
            else None
        )
        quaternion_norm_valid = (
            quaternion_norm is not None
            and abs(quaternion_norm - 1.0)
            <= SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE
        )
        yaw_rad = None
        if quaternion_finite:
            yaw_candidate = _yaw_from_quaternion(
                type(
                    "_QuaternionValue",
                    (),
                    quaternion,
                )()
            )
            if math.isfinite(yaw_candidate):
                yaw_rad = yaw_candidate

        retry_reason = str((retry_lookup.details or {}).get("reason", ""))
        predicates = {
            "explicitly_enabled": enabled,
            "use_sim_time": use_sim_time,
            "localization_source_is_tf": localization_is_tf,
            "map_frame_is_odom_frame": map_frame_is_odom_frame,
            "retry_is_stale_transform": retry_reason == "stale_transform",
            "odom_callback_advanced_during_recovery": (
                odom_callback_count_after > odom_callback_count_before
            ),
            "odom_message_available": odom_msg is not None,
            "odom_parent_frame_exact": (
                bool(odom_message_frame)
                and odom_message_frame == map_frame == odom_frame
            ),
            "odom_child_frame_exact": (
                bool(odom_child_frame)
                and odom_child_frame == base_frame
            ),
            "odom_fresh": bool(odom_freshness["fresh"]),
            "scan_fresh_after_recovery": bool(scan_freshness["fresh"]),
            "odom_stamp_available": odom_stamp_sec is not None,
            "retry_tf_stamp_available": retry_lookup.stamp_sec is not None,
            "odom_stamp_newer_than_tf_retry": (
                odom_stamp_sec is not None
                and retry_lookup.stamp_sec is not None
                and odom_stamp_sec > retry_lookup.stamp_sec
            ),
            "position_xy_finite": x_m is not None and y_m is not None,
            "quaternion_finite": quaternion_finite,
            "quaternion_norm_valid": quaternion_norm_valid,
            "yaw_finite": yaw_rad is not None,
        }
        rejection_reasons = [
            predicate
            for predicate, passed in predicates.items()
            if not passed
        ]
        fallback_episode = getattr(
            self,
            "_simulation_odom_fallback_episode",
            0,
        ) + (
            0
            if getattr(
                self,
                "_simulation_odom_fallback_active",
                False,
            )
            else 1
        )
        details: dict[str, object] = {
            "source": SIMULATION_ODOM_FALLBACK_SOURCE,
            "pose_source": SIMULATION_ODOM_FALLBACK_SOURCE,
            "attempted": True,
            "accepted": not rejection_reasons,
            "fail_closed": bool(rejection_reasons),
            "zero_published_before_fallback": True,
            "not_real_robot_migration_evidence": True,
            "fallback_episode": fallback_episode,
            "first_lookup_age_sec": (first_lookup.details or {}).get(
                "age_sec"
            ),
            "retry_lookup_age_sec": (retry_lookup.details or {}).get(
                "age_sec"
            ),
            "first_lookup_stamp_sec": first_lookup.stamp_sec,
            "retry_lookup_stamp_sec": retry_lookup.stamp_sec,
            "first_lookup": _pose_lookup_diagnostics(first_lookup),
            "retry_lookup": _pose_lookup_diagnostics(retry_lookup),
            "callback_drain": callback_drain,
            "runtime": {
                "use_sim_time": getattr(runtime, "use_sim_time", None),
                "localization_source": getattr(
                    runtime,
                    "localization_source",
                    None,
                ),
                "map_frame": map_frame,
                "odom_frame": odom_frame,
                "base_frame": base_frame,
            },
            "odom": {
                "frame_id": odom_message_frame,
                "child_frame_id": odom_child_frame,
                "header_stamp_sec": odom_stamp_sec,
                "receipt_monotonic_sec": odom_receipt,
                "callback_count_before_recovery": (
                    odom_callback_count_before
                ),
                "callback_count_after_recovery": (
                    odom_callback_count_after
                ),
                "freshness": odom_freshness,
                "pose": {
                    "x_m": x_m,
                    "y_m": y_m,
                    "yaw_rad": yaw_rad,
                    "quaternion": quaternion,
                    "quaternion_norm": quaternion_norm,
                    "quaternion_norm_tolerance": (
                        SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE
                    ),
                },
            },
            "scan": {
                "receipt_monotonic_sec": scan_receipt,
                "freshness": scan_freshness,
            },
            "predicates": predicates,
            "rejection_reasons": rejection_reasons,
        }

        if rejection_reasons:
            original_failure = _stale_tf_recovery_failure_details(
                dict(retry_lookup.details or {}),
                first_lookup=first_lookup,
                retry_lookup=retry_lookup,
                callback_drain=callback_drain,
            )
            original_failure["simulation_odom_fallback"] = details
            return PoseLookupResult(
                None,
                original_failure,
                retry_lookup.stamp_sec,
            )

        event_name = "simulation_odom_pose_fallback_started"
        if not getattr(
            self,
            "_simulation_odom_fallback_active",
            False,
        ):
            if not self._emit_route_update(
                RouteUpdate(
                    kind=RouteUpdateKind.UNCHANGED,
                    event_name=event_name,
                    event_fields=details,
                )
            ):
                return self._semantic_event_failure_lookup(
                    event_name=event_name,
                    stamp_sec=odom_stamp_sec,
                )
            self._simulation_odom_fallback_episode = fallback_episode
            self._simulation_odom_fallback_active = True
        return PoseLookupResult(
            Pose2D(float(x_m), float(y_m), float(yaw_rad)),
            details,
            odom_stamp_sec,
        )
