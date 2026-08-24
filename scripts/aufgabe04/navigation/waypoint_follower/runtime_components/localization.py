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


from .localization_sampling import LocalizationSamplingMixin
from .localization_evidence import LocalizationEvidenceMixin
from .amcl_recovery import AmclRecoveryMixin
from .simulation_odom_recovery import SimulationOdomRecoveryMixin

class LocalizationRuntimeMixin(LocalizationSamplingMixin, LocalizationEvidenceMixin, AmclRecoveryMixin, SimulationOdomRecoveryMixin):
    """Localization behavior mixed into the sole follower node."""














    def _primary_tf_result_with_restore_event(
        self,
        result: PoseLookupResult,
        *,
        recovered_after_retry: bool,
    ) -> PoseLookupResult:
        if (
            result.pose is None
            or not getattr(
                self,
                "_simulation_odom_fallback_active",
                False,
            )
        ):
            return result

        # The prior command may have been nonzero.  Hold zero while the
        # semantic source transition is synchronously committed.
        self.publish_zero()
        event_name = "simulation_odom_pose_fallback_restored"
        event_fields = {
            "source": "tf_lookup",
            "pose_source": "tf_lookup",
            "primary_tf_stamp_sec": result.stamp_sec,
            "recovered_after_retry": recovered_after_retry,
            "fallback_episode": getattr(
                self,
                "_simulation_odom_fallback_episode",
                0,
            ),
            "not_real_robot_migration_evidence": True,
        }
        if not self._emit_route_update(
            RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                event_name=event_name,
                event_fields=event_fields,
            )
        ):
            return self._semantic_event_failure_lookup(
                event_name=event_name,
                stamp_sec=result.stamp_sec,
            )
        self._simulation_odom_fallback_active = False
        return result


    def _current_pose_lookup_with_stale_recovery(self) -> PoseLookupResult:
        first_lookup = self._current_pose_lookup()
        first_details = dict(first_lookup.details or {})
        if first_lookup.pose is not None:
            return self._primary_tf_result_with_restore_event(
                first_lookup,
                recovered_after_retry=False,
            )
        if first_details.get("reason") != "stale_transform":
            return first_lookup

        # Override any preceding nonzero Twist before servicing queued
        # scan/odom/TF work.  No motion command is published during recovery.
        odom_callback_count_before = getattr(
            self,
            "latest_odom_callback_count",
            0,
        )
        self.publish_zero()
        real_amcl_runtime = self._is_real_amcl_runtime()
        map_to_odom_before = None
        if real_amcl_runtime:
            map_to_odom_before = self._tf_edge_sample(
                self.runtime_config.map_frame,
                self.runtime_config.odom_frame,
            )
        drain_details = self._drain_runtime_callbacks(
            max_callbacks=STALE_TF_RECOVERY_MAX_CALLBACKS,
            max_duration_sec=STALE_TF_RECOVERY_MAX_DURATION_SEC,
            spin_timeout_sec=STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
        )
        odom_callback_count_after = getattr(
            self,
            "latest_odom_callback_count",
            0,
        )
        odom_msg = getattr(self, "latest_odom", None)
        odom_receipt = getattr(self, "latest_odom_receipt", None)
        scan_msg = getattr(self, "latest_scan", None)
        scan_receipt = getattr(self, "latest_scan_receipt", None)
        retry_lookup = self._current_pose_lookup()
        retry_details = dict(retry_lookup.details or {})
        if retry_lookup.pose is None:
            if retry_details.get("reason") == "stale_transform":
                if real_amcl_runtime:
                    assert map_to_odom_before is not None
                    return self._real_amcl_stale_tf_recovery(
                        first_lookup=first_lookup,
                        retry_lookup=retry_lookup,
                        callback_drain=drain_details,
                        map_to_odom_before=map_to_odom_before,
                        map_to_odom_retry=self._tf_edge_sample(
                            self.runtime_config.map_frame,
                            self.runtime_config.odom_frame,
                        ),
                        odom_to_base_retry=self._tf_edge_sample(
                            self.runtime_config.odom_frame,
                            self.runtime_config.base_frame,
                        ),
                    )
                return self._simulation_odom_fallback_after_stale_retry(
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                    odom_callback_count_before=(
                        odom_callback_count_before
                    ),
                    odom_callback_count_after=(
                        odom_callback_count_after
                    ),
                    odom_msg=odom_msg,
                    odom_receipt=odom_receipt,
                    scan_msg=scan_msg,
                    scan_receipt=scan_receipt,
                )
            # Preserve the retry failure as the top-level stop diagnostic.
            # Persistent stale_transform therefore stops exactly as before,
            # with its retry age in the legacy age_sec field.
            return PoseLookupResult(
                None,
                _stale_tf_recovery_failure_details(
                    retry_details,
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                ),
                retry_lookup.stamp_sec,
            )
        if (
            first_lookup.stamp_sec is None
            or retry_lookup.stamp_sec is None
            or retry_lookup.stamp_sec <= first_lookup.stamp_sec
        ):
            nonadvancing_details = tf_lookup_failure_details(
                reason="nonadvancing_transform",
                target_frame=self.runtime_config.map_frame,
                source_frame=self.runtime_config.base_frame,
                max_age_sec=self.follower_config.max_tf_age_sec,
            )
            return PoseLookupResult(
                None,
                _stale_tf_recovery_failure_details(
                    nonadvancing_details,
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                ),
                retry_lookup.stamp_sec,
            )
        freshness_failure = self._post_stale_tf_recovery_freshness_failure()
        if freshness_failure:
            freshness_details = {
                "stop_reason": freshness_failure,
                "source": "stale_tf_recovery",
                "reason": "post_recovery_sensor_freshness_failure",
                "sensor_failure": dict(self.latest_stop_details or {}),
            }
            return PoseLookupResult(
                None,
                _stale_tf_recovery_failure_details(
                    freshness_details,
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                ),
                retry_lookup.stamp_sec,
            )
        return self._primary_tf_result_with_restore_event(
            retry_lookup,
            recovered_after_retry=True,
        )

    def _current_pose(self) -> Pose2D | None:
        return self._current_pose_lookup().pose
