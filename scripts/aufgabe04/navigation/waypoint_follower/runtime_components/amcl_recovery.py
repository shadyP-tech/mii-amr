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


class AmclRecoveryMixin:
    """Focused amcl recovery behavior."""

    def _real_amcl_stale_tf_recovery(
        self,
        *,
        first_lookup: PoseLookupResult,
        retry_lookup: PoseLookupResult,
        callback_drain: Mapping[str, object],
        map_to_odom_before: TfEdgeSample,
        map_to_odom_retry: TfEdgeSample,
        odom_to_base_retry: TfEdgeSample,
    ) -> PoseLookupResult:
        """Perform one bounded, zero-held AMCL no-motion refresh episode."""

        timeout_sec = float(
            getattr(
                self.follower_config,
                "runtime_nomotion_update_timeout_sec",
                2.0,
            )
        )
        deadline = time.monotonic() + timeout_sec
        now_sec = self._ros_now_sec()
        composed_before = self._composed_tf_sample(first_lookup)
        composed_retry = self._composed_tf_sample(retry_lookup)
        eligibility = evaluate_recovery_eligibility(
            localization_source=getattr(
                self.runtime_config,
                "localization_source",
                "",
            ),
            use_sim_time=getattr(
                self.runtime_config,
                "use_sim_time",
                True,
            ),
            composed_before=composed_before,
            composed_retry=composed_retry,
            map_to_odom_before=map_to_odom_before,
            map_to_odom_retry=map_to_odom_retry,
            odom_to_base_retry=odom_to_base_retry,
            now_sec=now_sec,
            max_tf_age_sec=self.follower_config.max_tf_age_sec,
            composed_future_tolerance_sec=(
                self.follower_config.max_future_timestamp_sec
            ),
            map_to_odom_future_tolerance_sec=(
                self.follower_config.amcl_edge_future_tolerance_sec
            ),
        )
        base_evidence: dict[str, object] = {
            "service_name": getattr(
                self,
                "runtime_nomotion_update_service",
                getattr(
                    self.follower_config,
                    "runtime_nomotion_update_service",
                    "request_nomotion_update",
                ),
            ),
            "timeout_sec": timeout_sec,
            "service_requested": False,
            "service_completed": False,
            "zero_held": True,
            "motion_authorized": False,
            "requires_route_tube_readmission": True,
            "callback_drain": dict(callback_drain),
            "eligibility": eligibility.to_log_dict(),
            "tf_edges": {
                "composed_before": composed_before.to_log_dict(
                    now_sec=now_sec
                ),
                "composed_retry": composed_retry.to_log_dict(
                    now_sec=now_sec
                ),
                "map_to_odom_before": map_to_odom_before.to_log_dict(
                    now_sec=now_sec
                ),
                "map_to_odom_retry": map_to_odom_retry.to_log_dict(
                    now_sec=now_sec
                ),
                "odom_to_base_retry": odom_to_base_retry.to_log_dict(
                    now_sec=now_sec
                ),
            },
        }
        if not eligibility.accepted:
            return self._real_amcl_recovery_failure(
                reason=eligibility.reason,
                evidence=base_evidence,
                stamp_sec=retry_lookup.stamp_sec,
            )

        sensor_failure = self._post_stale_tf_recovery_freshness_failure()
        if sensor_failure:
            return self._real_amcl_recovery_failure(
                reason="pre_request_sensor_freshness_failure",
                evidence={
                    **base_evidence,
                    "sensor_failure": dict(self.latest_stop_details or {}),
                },
                stamp_sec=retry_lookup.stamp_sec,
            )
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return self._real_amcl_recovery_failure(
                reason="pre_request_cmd_vel_ownership_failure",
                evidence={
                    **base_evidence,
                    "ownership_failure": ownership_failure,
                },
                stamp_sec=retry_lookup.stamp_sec,
            )

        stationarity, stationarity_evidence = (
            self._wait_for_stationary_odom_pair(
                deadline_monotonic=deadline,
            )
        )
        if stationarity is None:
            return self._real_amcl_recovery_failure(
                reason=str(stationarity_evidence["reason"]),
                evidence={
                    **base_evidence,
                    "stationarity_before_request": stationarity_evidence,
                },
                stamp_sec=retry_lookup.stamp_sec,
            )

        started_evidence = {
            **base_evidence,
            "stationarity_before_request": stationarity_evidence,
        }
        if not self._emit_route_update(
            RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                event_name="real_amcl_stale_tf_recovery_started",
                event_fields=started_evidence,
            )
        ):
            return self._real_amcl_recovery_failure(
                reason="recovery_start_event_failed",
                evidence={
                    **started_evidence,
                    "semantic_event_failure": dict(
                        self.latest_stop_details or {}
                    ),
                },
                stamp_sec=retry_lookup.stamp_sec,
            )
        trace_failure = self._append_controller_trace(
            event="real_amcl_stale_tf_recovery_started",
            reason="persistent_stale_localization_edge",
            fail_closed=False,
            diagnostics=started_evidence,
        )
        if trace_failure:
            return self._real_amcl_recovery_failure(
                reason="controller_trace_write_failed",
                evidence={
                    **started_evidence,
                    "controller_trace_error": trace_failure,
                },
                stamp_sec=retry_lookup.stamp_sec,
            )

        client = getattr(self, "runtime_nomotion_update_client", None)
        if client is None or not client.service_is_ready():
            return self._real_amcl_recovery_failure(
                reason="nomotion_update_service_unavailable",
                evidence=started_evidence,
                stamp_sec=retry_lookup.stamp_sec,
            )
        self.publish_zero()
        try:
            future = client.call_async(Empty.Request())
        except Exception as exc:
            return self._real_amcl_recovery_failure(
                reason="nomotion_update_service_request_failed",
                evidence={
                    **started_evidence,
                    "service_exception_type": exc.__class__.__name__,
                    "service_exception": str(exc),
                },
                stamp_sec=retry_lookup.stamp_sec,
            )

        request_evidence = {
            **started_evidence,
            "service_requested": True,
        }
        candidate_lookup = retry_lookup
        candidate_map_to_odom = map_to_odom_retry
        candidate_odom_to_base = odom_to_base_retry
        service_completed = False
        service_error: BaseException | None = None
        probe = None
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_zero()
            self._service_or_wait_for_callbacks(
                min(
                    AMCL_STALE_TF_RECOVERY_POLL_SEC,
                    max(0.0, deadline - time.monotonic()),
                )
            )
            if future.done():
                service_completed = True
                try:
                    service_error = future.exception()
                except Exception as exc:
                    service_error = exc
                if service_error is not None:
                    break
                candidate_lookup = self._current_pose_lookup()
                candidate_map_to_odom = self._tf_edge_sample(
                    self.runtime_config.map_frame,
                    self.runtime_config.odom_frame,
                )
                candidate_odom_to_base = self._tf_edge_sample(
                    self.runtime_config.odom_frame,
                    self.runtime_config.base_frame,
                )
                scan_evidence = self._fallback_message_freshness_evidence(
                    "scan",
                    self.latest_scan,
                    self.latest_scan_receipt,
                    self.follower_config.max_scan_age_sec,
                )
                odom_evidence = self._fallback_message_freshness_evidence(
                    "odom",
                    self.latest_odom,
                    self.latest_odom_receipt,
                    self.follower_config.max_odom_age_sec,
                )
                owner_ok = not self._cmd_vel_ownership_failure()
                probe = evaluate_recovery_acceptance(
                    eligibility=eligibility,
                    composed_before=composed_before,
                    composed_recovered=self._composed_tf_sample(
                        candidate_lookup
                    ),
                    map_to_odom_before=map_to_odom_before,
                    map_to_odom_recovered=candidate_map_to_odom,
                    odom_to_base_recovered=candidate_odom_to_base,
                    stationarity=stationarity,
                    scan_fresh=bool(scan_evidence["fresh"]),
                    odom_fresh=bool(odom_evidence["fresh"]),
                    exclusive_cmd_vel_owner=owner_ok,
                    now_sec=self._ros_now_sec(),
                    max_tf_age_sec=self.follower_config.max_tf_age_sec,
                    composed_future_tolerance_sec=(
                        self.follower_config.max_future_timestamp_sec
                    ),
                    map_to_odom_future_tolerance_sec=(
                        self.follower_config.amcl_edge_future_tolerance_sec
                    ),
                )
                if probe.accepted:
                    break
                terminal_probe_reasons = tuple(
                    reason
                    for reason in probe.reasons
                    if reason
                    in {
                        "scan_not_fresh",
                        "odom_not_fresh",
                        "cmd_vel_owner_not_exclusive",
                    }
                    or reason.startswith(
                        "odom_to_base_recovered_not_fresh:"
                    )
                )
                if terminal_probe_reasons:
                    return self._real_amcl_recovery_failure(
                        reason=terminal_probe_reasons[0],
                        evidence={
                            **request_evidence,
                            "service_completed": True,
                            "acceptance_probe": probe.to_log_dict(),
                            "scan_freshness": scan_evidence,
                            "odom_freshness": odom_evidence,
                        },
                        stamp_sec=candidate_lookup.stamp_sec,
                    )

        if service_error is not None:
            return self._real_amcl_recovery_failure(
                reason="nomotion_update_service_failed",
                evidence={
                    **request_evidence,
                    "service_completed": True,
                    "service_exception_type": (
                        service_error.__class__.__name__
                    ),
                    "service_exception": str(service_error),
                },
                stamp_sec=candidate_lookup.stamp_sec,
            )
        if not service_completed:
            return self._real_amcl_recovery_failure(
                reason="nomotion_update_service_timeout",
                evidence=request_evidence,
                stamp_sec=candidate_lookup.stamp_sec,
            )
        if probe is None or not probe.accepted:
            return self._real_amcl_recovery_failure(
                reason="stale_tf_recovery_timeout",
                evidence={
                    **request_evidence,
                    "service_completed": True,
                    "acceptance_probe": (
                        None if probe is None else probe.to_log_dict()
                    ),
                    "tf_edges_after_request": {
                        "composed": self._composed_tf_sample(
                            candidate_lookup
                        ).to_log_dict(now_sec=self._ros_now_sec()),
                        "map_to_odom": (
                            candidate_map_to_odom.to_log_dict(
                                now_sec=self._ros_now_sec()
                            )
                        ),
                        "odom_to_base": (
                            candidate_odom_to_base.to_log_dict(
                                now_sec=self._ros_now_sec()
                            )
                        ),
                    },
                },
                stamp_sec=candidate_lookup.stamp_sec,
            )

        # Complete a whole controller-period zero handoff before the final
        # stationarity and admission samples.  If the bounded episode cannot
        # fit that handoff, recovery remains terminal.
        zero_cycle_sec = 1.0 / max(
            self.follower_config.control_rate_hz,
            1.0,
        )
        if deadline - time.monotonic() < zero_cycle_sec:
            return self._real_amcl_recovery_failure(
                reason="zero_cycle_handoff_timeout",
                evidence={
                    **request_evidence,
                    "service_completed": True,
                    "acceptance_probe": probe.to_log_dict(),
                },
                stamp_sec=candidate_lookup.stamp_sec,
            )
        self.publish_zero()
        self._service_or_wait_for_callbacks(zero_cycle_sec)

        final_stationarity, final_stationarity_evidence = (
            self._wait_for_stationary_odom_pair(
                deadline_monotonic=deadline,
            )
        )
        if final_stationarity is None:
            return self._real_amcl_recovery_failure(
                reason=str(final_stationarity_evidence["reason"]),
                evidence={
                    **request_evidence,
                    "service_completed": True,
                    "stationarity_after_request": (
                        final_stationarity_evidence
                    ),
                },
                stamp_sec=candidate_lookup.stamp_sec,
            )

        final_lookup = self._current_pose_lookup()
        final_map_to_odom = self._tf_edge_sample(
            self.runtime_config.map_frame,
            self.runtime_config.odom_frame,
        )
        final_odom_to_base = self._tf_edge_sample(
            self.runtime_config.odom_frame,
            self.runtime_config.base_frame,
        )
        scan_evidence = self._fallback_message_freshness_evidence(
            "scan",
            self.latest_scan,
            self.latest_scan_receipt,
            self.follower_config.max_scan_age_sec,
        )
        odom_evidence = self._fallback_message_freshness_evidence(
            "odom",
            self.latest_odom,
            self.latest_odom_receipt,
            self.follower_config.max_odom_age_sec,
        )
        ownership_failure = self._cmd_vel_ownership_failure()
        final_now_sec = self._ros_now_sec()
        acceptance = evaluate_recovery_acceptance(
            eligibility=eligibility,
            composed_before=composed_before,
            composed_recovered=self._composed_tf_sample(final_lookup),
            map_to_odom_before=map_to_odom_before,
            map_to_odom_recovered=final_map_to_odom,
            odom_to_base_recovered=final_odom_to_base,
            stationarity=final_stationarity,
            scan_fresh=bool(scan_evidence["fresh"]),
            odom_fresh=bool(odom_evidence["fresh"]),
            exclusive_cmd_vel_owner=not ownership_failure,
            now_sec=final_now_sec,
            max_tf_age_sec=self.follower_config.max_tf_age_sec,
            composed_future_tolerance_sec=(
                self.follower_config.max_future_timestamp_sec
            ),
            map_to_odom_future_tolerance_sec=(
                self.follower_config.amcl_edge_future_tolerance_sec
            ),
        )
        final_evidence = {
            **request_evidence,
            "service_completed": True,
            "stationarity_after_request": final_stationarity_evidence,
            "scan_freshness": scan_evidence,
            "odom_freshness": odom_evidence,
            "ownership_failure": ownership_failure,
            "acceptance": acceptance.to_log_dict(),
            "tf_edges_after_request": {
                "composed": self._composed_tf_sample(
                    final_lookup
                ).to_log_dict(now_sec=final_now_sec),
                "map_to_odom": final_map_to_odom.to_log_dict(
                    now_sec=final_now_sec
                ),
                "odom_to_base": final_odom_to_base.to_log_dict(
                    now_sec=final_now_sec
                ),
            },
            "zero_cycle_handoff_completed": True,
        }
        if not acceptance.accepted or final_lookup.pose is None:
            return self._real_amcl_recovery_failure(
                reason=acceptance.reason,
                evidence=final_evidence,
                stamp_sec=final_lookup.stamp_sec,
            )

        if not self._emit_route_update(
            RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                event_name="real_amcl_stale_tf_recovery_recovered",
                event_fields=final_evidence,
            )
        ):
            return self._real_amcl_recovery_failure(
                reason="recovery_event_failed",
                evidence={
                    **final_evidence,
                    "semantic_event_failure": dict(
                        self.latest_stop_details or {}
                    ),
                },
                stamp_sec=final_lookup.stamp_sec,
            )
        trace_failure = self._append_controller_trace(
            event="real_amcl_stale_tf_recovery_recovered",
            pose=final_lookup.pose,
            reason=acceptance.reason,
            fail_closed=False,
            diagnostics=final_evidence,
        )
        if trace_failure:
            return self._real_amcl_recovery_failure(
                reason="controller_trace_write_failed",
                evidence={
                    **final_evidence,
                    "controller_trace_error": trace_failure,
                },
                stamp_sec=final_lookup.stamp_sec,
            )
        return PoseLookupResult(
            final_lookup.pose,
            {
                "source": "real_amcl_stale_tf_recovery",
                "accepted": True,
                **final_evidence,
            },
            final_lookup.stamp_sec,
        )
