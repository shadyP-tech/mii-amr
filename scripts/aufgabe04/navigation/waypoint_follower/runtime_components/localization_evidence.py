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


from .tf_sampling import map_tf_monitor_warning, sample_tf_pose


class LocalizationEvidenceMixin:
    """Focused localization evidence behavior."""

    def _map_from_odom_lookup(self) -> PoseLookupResult:
        context = self.odom_execution_context
        return sample_tf_pose(
            self, target_frame=context.map_frame, source_frame=context.odom_frame,
            max_future_sec=self.follower_config.amcl_edge_future_tolerance_sec,
        )

    def _global_consistency_monitor_failure(self, map_lookup: PoseLookupResult | None = None) -> str:
        """Apply unchanged live continuity gates to one structured TF sample."""

        context = getattr(self, "odom_execution_context", None)
        if context is None:
            return ""
        lookup = self._map_from_odom_lookup() if map_lookup is None else map_lookup
        pose = lookup.pose
        live_transform = None if pose is None else PlanarTransform2D(
            pose.x_m, pose.y_m, pose.yaw_rad,
        )
        continuity = evaluate_map_odom_continuity(context, live_transform)
        monitor = evaluate_global_consistency_monitor(
            reseal_required=not continuity.accepted,
            diagnostic_warning=map_tf_monitor_warning(lookup),
        )
        if monitor.action != MONITOR_ACTION_FORCE_ZERO_RESEAL:
            return ""
        reason = "global localization consistency requires zero and reseal"
        self.latest_stop_details = {
            "reason": reason,
            "fault_code": "localization_reseal_required",
            "source": "global_consistency_monitor",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "monitor_action": monitor.action,
            "monitor_reason": monitor.reason,
            "monitor_warning": monitor.diagnostic_warning,
            "continuity": continuity.to_evidence(),
            "tf_sample": _pose_lookup_diagnostics(lookup),
            "fail_closed": True,
        }
        return reason

    def _current_pose_lookup(self) -> PoseLookupResult:
        context = getattr(self, "odom_execution_context", None)
        return sample_tf_pose(
            self,
            target_frame=self.runtime_config.map_frame if context is None else context.odom_frame,
            source_frame=self.runtime_config.base_frame,
            max_future_sec=self.follower_config.max_future_timestamp_sec,
        )

    def _post_stale_tf_recovery_freshness_failure(self) -> str:
        scan_failure = self._freshness_failure(
            "scan",
            self.latest_scan,
            self.latest_scan_receipt,
            self.follower_config.max_scan_age_sec,
        )
        if scan_failure:
            return scan_failure
        return self._freshness_failure(
            "odom",
            self.latest_odom,
            self.latest_odom_receipt,
            self.follower_config.max_odom_age_sec,
        )
    def _fallback_message_freshness_evidence(
        self,
        name: str,
        msg,
        receipt,
        max_age_sec: float,
    ) -> dict[str, object]:
        """Apply the ordinary freshness gate and retain its predicate evidence."""

        try:
            failure = self._freshness_failure(
                name,
                msg,
                receipt,
                max_age_sec,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return {
                "sensor": name,
                "fresh": False,
                "failure": f"malformed {name} freshness data",
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "max_future_sec": getattr(
                    self.follower_config,
                    "max_future_timestamp_sec",
                    None,
                ),
            }

        if failure:
            stop_details = dict(self.latest_stop_details or {})
            return {
                "sensor": name,
                "fresh": False,
                "failure": failure,
                "receipt_age_sec": stop_details.get("receipt_age_sec"),
                "header_age_sec": stop_details.get("header_age_sec"),
                "max_age_sec": max_age_sec,
                "max_future_sec": stop_details.get(
                    "max_future_sec",
                    getattr(
                        self.follower_config,
                        "max_future_timestamp_sec",
                        None,
                    ),
                ),
                "receipt_stale": stop_details.get("receipt_stale"),
                "header_stale": stop_details.get("header_stale"),
                "receipt_future": stop_details.get("receipt_future"),
                "header_future": stop_details.get("header_future"),
            }

        try:
            receipt_age_sec = time.monotonic() - float(receipt)
            header_age_sec = (
                self.get_clock().now() - Time.from_msg(msg.header.stamp)
            ).nanoseconds / 1_000_000_000.0
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            # This branch is conservative: the ordinary gate just passed, but
            # evidence extraction itself was not trustworthy.
            return {
                "sensor": name,
                "fresh": False,
                "failure": f"malformed {name} timing evidence",
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "max_future_sec": getattr(
                    self.follower_config,
                    "max_future_timestamp_sec",
                    None,
                ),
            }
        return {
            "sensor": name,
            "fresh": True,
            "failure": "",
            "receipt_age_sec": receipt_age_sec,
            "header_age_sec": header_age_sec,
            "max_age_sec": max_age_sec,
            "max_future_sec": self.follower_config.max_future_timestamp_sec,
            "receipt_stale": False,
            "header_stale": False,
            "receipt_future": False,
            "header_future": False,
        }
    def _semantic_event_failure_lookup(
        self,
        *,
        event_name: str,
        stamp_sec: float | None,
    ) -> PoseLookupResult:
        callback_failure = dict(self.latest_stop_details or {})
        return PoseLookupResult(
            None,
            {
                "stop_reason": callback_failure.get(
                    "reason",
                    "semantic event callback failed",
                ),
                "source": "semantic_event_callback",
                "event_name": event_name,
                "semantic_event_failure": callback_failure,
                "fail_closed": True,
            },
            stamp_sec,
        )
    def _real_amcl_recovery_failure(
        self,
        *,
        reason: str,
        evidence: Mapping[str, object],
        stamp_sec: float | None,
    ) -> PoseLookupResult:
        """Persist one terminal AMCL recovery result without masking its cause."""

        details: dict[str, object] = {
            "stop_reason": "map-to-base transform unavailable",
            "source": "real_amcl_stale_tf_recovery",
            "reason": reason,
            "fail_closed": True,
            **dict(evidence),
        }
        event_name = "real_amcl_stale_tf_recovery_failed"
        if not self._emit_route_update(
            RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                event_name=event_name,
                event_fields=details,
            )
        ):
            details["semantic_event_failure"] = dict(
                self.latest_stop_details or {}
            )
        trace_failure = self._append_controller_trace(
            event="pose_lookup_stop",
            reason=reason,
            fail_closed=True,
            diagnostics=details,
        )
        if trace_failure:
            # The transform/recovery stop remains the primary safety reason.
            details["controller_trace_error"] = trace_failure
            details["controller_trace_fault_code"] = (
                "controller_trace_write_failed"
            )
        else:
            details["pose_lookup_trace_recorded"] = True
        self.latest_stop_details = details
        return PoseLookupResult(None, details, stamp_sec)
