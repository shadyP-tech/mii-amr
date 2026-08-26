"""Stationary blockage confirmation and replacement-route recovery."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Mapping

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None

from scripts.aufgabe04.navigation.control.follower_safety import (
    NO_VALID_FRONT_SECTOR_SCAN_RANGES,
    front_sector_decision,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.coverage.transient_blockage_admission import (
    StationaryBlockageAdmission,
    collect_stationary_blockage_admission,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    PersistentObstacleConfig,
    StationaryFrontSectorSample,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand
from scripts.aufgabe04.navigation.waypoint_follower.pose_lookup import PoseLookupResult
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    BlockageRecoveryAction,
    RouteRefreshAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.recovery_dispatch import (
    BlockageRecoveryDisposition,
    BlockageRecoveryTrigger,
    blockage_recovery_disposition,
    blockage_recovery_eligible,
)

rclpy = RuntimeBindingProxy("rclpy", rclpy)


class BlockageRecoveryRuntimeMixin:
    """Blockage recovery behavior mixed into the sole follower node."""

    def _stationary_front_sample(
        self,
    ) -> tuple[StationaryFrontSectorSample | None, dict[str, object]]:
        """Capture one fresh front ray with simultaneous map/odom poses."""

        scan_failure = self._freshness_failure(
            "scan",
            self.latest_scan,
            self.latest_scan_receipt,
            self.follower_config.max_scan_age_sec,
        )
        if scan_failure:
            return None, dict(self.latest_stop_details or {})
        odom_failure = self._freshness_failure(
            "odom",
            self.latest_odom,
            self.latest_odom_receipt,
            self.follower_config.max_odom_age_sec,
        )
        if odom_failure:
            return None, dict(self.latest_stop_details or {})
        decision = front_sector_decision(
            getattr(self.latest_scan, "ranges", None),
            float(getattr(self.latest_scan, "angle_min", 0.0)),
            float(getattr(self.latest_scan, "angle_increment", 0.0)),
            0.0,
            self.follower_config.front_obstacle_sector_rad,
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
            source="front_sector",
        )
        front_details = decision.to_log_dict()
        self.latest_front_clearance_details = front_details
        if (
            decision.nearest_valid_range_m is None
            or decision.nearest_valid_bearing_rad is None
        ):
            return None, {
                **front_details,
                "reason": decision.stop_reason
                or NO_VALID_FRONT_SECTOR_SCAN_RANGES,
                "source": "stationary_blockage_confirmation",
                "fail_closed": True,
            }
        execution_lookup = self._current_pose_lookup_with_stale_recovery()
        if execution_lookup.pose is None:
            return None, {
                **dict(execution_lookup.details or {}),
                "source": "stationary_blockage_confirmation",
                "fail_closed": True,
            }
        odom_pose = self._latest_odom_pose()
        if odom_pose is None:
            return None, {
                "reason": "odometry pose is invalid during blockage confirmation",
                "source": "stationary_blockage_confirmation",
                "fail_closed": True,
            }
        context = getattr(self, "odom_execution_context", None)
        map_pose = (
            execution_lookup.pose
            if context is None
            else context.odom_pose_to_map(execution_lookup.pose)
        )
        try:
            sample = StationaryFrontSectorSample(
                timestamp_sec=float(self.latest_scan_receipt),
                front_range_m=decision.nearest_valid_range_m,
                front_bearing_rad=decision.nearest_valid_bearing_rad,
                map_pose=map_pose,
                odom_pose=odom_pose,
            )
        except (TypeError, ValueError) as exc:
            return None, {
                "reason": f"stationary blockage sample is invalid: {exc}",
                "source": "stationary_blockage_confirmation",
                "fail_closed": True,
            }
        return sample, front_details

    def _confirm_stationary_blockage(self) -> StationaryBlockageAdmission:
        """Hold zero until a coherent obstacle or coherent clearance is proven."""

        config = self.follower_config.persistent_obstacle_config
        assert isinstance(config, PersistentObstacleConfig)
        return collect_stationary_blockage_admission(
            config=config,
            timeout_sec=(
                self.follower_config.blockage_confirmation_timeout_sec
            ),
            clearance_threshold_m=(
                self.follower_config.front_obstacle_slow_distance_m
            ),
            initial_scan_receipt=getattr(self, "latest_scan_receipt", None),
            runtime_ok=rclpy.ok,
            publish_zero=self.publish_zero,
            service_callbacks=self._service_or_wait_for_callbacks,
            current_scan_receipt=lambda: getattr(
                self,
                "latest_scan_receipt",
                None,
            ),
            capture_sample=self._stationary_front_sample,
            monotonic=time.monotonic,
        )

    def _blockage_recovery_outcome(
        self,
        *,
        trigger: BlockageRecoveryTrigger,
        pose: Pose2D | None,
        stop_reason: str,
        stop_details: Mapping[str, object] | None,
        front_evidence: Mapping[str, object] | None,
        nominal_linear_x_mps: float | None = None,
    ) -> BlockageRecoveryDisposition:
        """Attempt an admitted recovery and reduce it to one loop outcome."""

        provider_available = self.blockage_recovery_provider is not None
        if trigger == BlockageRecoveryTrigger.OBSTACLE_SAFETY_STOP:
            # A front-sector safety stop occurs before this cycle computes a
            # controller command. Its recovery admission therefore depends on
            # validated front evidence, not a fabricated nominal velocity.
            eligible = (
                provider_available
                and (front_evidence or {}).get("source") == "front_sector"
            )
        else:
            eligible = (
                nominal_linear_x_mps is not None
                and blockage_recovery_eligible(
                    provider_available=provider_available,
                    nominal_linear_x_mps=nominal_linear_x_mps,
                    front_evidence=front_evidence,
                )
            )
        recovery = BlockageRecoveryAction.NOT_ATTEMPTED
        if eligible and pose is not None:
            recovery = self._attempt_blockage_recovery(
                pose,
                stop_reason,
                stop_details or {},
            )
        return blockage_recovery_disposition(
            trigger=trigger,
            recovery_action=recovery,
            fallback_reason=stop_reason,
            latest_reason=(
                (self.latest_stop_details or {}).get("reason", stop_reason)
            ),
        )

    def _attempt_blockage_recovery(
        self,
        pose: Pose2D,
        stop_reason: str,
        stop_details: Mapping[str, object],
    ) -> BlockageRecoveryAction:
        """Plan and atomically adopt one physical coverage route revision."""

        provider = self.blockage_recovery_provider
        if provider is None:
            return BlockageRecoveryAction.NOT_ATTEMPTED
        # Motion must already be zero before synchronous planning, artifact
        # sealing, or event logging begins. Sensor callbacks continue on the
        # background executor while the planner runs.
        self.publish_repeated_zero()
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            self.latest_stop_details = {
                **dict(stop_details),
                "reason": ownership_failure,
                "fault_code": "cmd_vel_ownership_ambiguous_before_replan",
                "source": "blockage_recovery_admission",
                "original_stop_reason": stop_reason,
                "fail_closed": True,
            }
            return BlockageRecoveryAction.STOPPED
        confirmation = self._confirm_stationary_blockage()
        trace_failure = self._append_controller_trace(
            event=f"blockage_{confirmation.status}",
            # Controller traces are always in the execution frame. The
            # independently named map pose remains in confirmation evidence.
            pose=pose,
            reason=stop_reason,
            fail_closed=confirmation.status == "failed",
            effective_command=VelocityCommand(0.0, 0.0),
            front_cluster_summary=confirmation.evidence,
        )
        if trace_failure:
            return BlockageRecoveryAction.STOPPED
        if confirmation.status == "cleared":
            if stop_reason == "stuck no progress":
                self.latest_stop_details = {
                    **dict(stop_details),
                    **confirmation.evidence,
                    "reason": (
                        "stuck no progress without a confirmed persistent "
                        "front obstacle"
                    ),
                    "fault_code": "stuck_without_persistent_front_obstacle",
                    "original_stop_reason": stop_reason,
                    "fail_closed": True,
                }
                return BlockageRecoveryAction.STOPPED
            self.latest_stop_details = {
                **dict(stop_details),
                **confirmation.evidence,
                "reason": "stationary front clearance confirmed",
                "original_stop_reason": stop_reason,
                "fail_closed": False,
            }
            self._reset_progress_watchdog(time.monotonic())
            return BlockageRecoveryAction.CLEARED
        if confirmation.status != "confirmed" or confirmation.pose is None:
            self.latest_stop_details = {
                **dict(stop_details),
                **confirmation.evidence,
                "reason": "stationary blockage confirmation failed",
                "fault_code": "stationary_blockage_unconfirmed",
                "original_stop_reason": stop_reason,
                "fail_closed": True,
            }
            return BlockageRecoveryAction.STOPPED
        confirmed_pose = confirmation.pose
        context = getattr(self, "odom_execution_context", None)
        runtime = getattr(self, "runtime_config", None)
        planning_frame = getattr(runtime, "map_frame", "map")
        execution_frame = (
            planning_frame
            if context is None
            else context.odom_frame
        )
        confirmed_stop_details = {
            **dict(stop_details),
            **confirmation.evidence,
            "front_clearance": dict(confirmation.front_clearance or {}),
            "trigger_pose": {
                "frame_id": execution_frame,
                "x_m": pose.x_m,
                "y_m": pose.y_m,
                "yaw_rad": pose.yaw_rad,
            },
            "fail_closed": False,
        }
        try:
            update = provider(
                confirmed_pose,
                stop_reason,
                confirmed_stop_details,
            )
        except Exception as exc:
            self.latest_stop_details = {
                **confirmed_stop_details,
                "reason": f"blockage recovery provider failed: {exc}",
                "fault_code": "blockage_recovery_provider_exception",
                "original_stop_reason": stop_reason,
                "fail_closed": True,
            }
            return BlockageRecoveryAction.STOPPED
        if update is None:
            return BlockageRecoveryAction.NOT_ATTEMPTED
        # Planning and artifact sealing are synchronous.  TF/AMCL and sensor
        # callbacks continue on the background executors while that work runs,
        # so the pose that triggered recovery is no longer authoritative for
        # route admission.  Recheck every live input and bind adoption to a
        # fresh post-plan execution pose instead of silently reusing ``pose``.
        admission = self._post_replan_admission_pose()
        if admission.pose is None:
            self.latest_stop_details = {
                **confirmed_stop_details,
                **dict(admission.details or {}),
                "reason": "post-replan runtime admission failed",
                "fault_code": "post_replan_admission_failed",
                "original_stop_reason": stop_reason,
                "fail_closed": True,
            }
            return BlockageRecoveryAction.STOPPED
        fresh_pose = admission.pose
        fresh_planning_pose = (
            fresh_pose
            if context is None
            else context.odom_pose_to_map(fresh_pose)
        )
        update = replace(
            update,
            event_fields={
                **dict(update.event_fields),
                "planning_stop_pose": {
                    "frame_id": planning_frame,
                    "x_m": confirmed_pose.x_m,
                    "y_m": confirmed_pose.y_m,
                    "yaw_rad": confirmed_pose.yaw_rad,
                },
                "post_plan_admission_pose": {
                    "frame_id": planning_frame,
                    "x_m": fresh_planning_pose.x_m,
                    "y_m": fresh_planning_pose.y_m,
                    "yaw_rad": fresh_planning_pose.yaw_rad,
                },
                "post_plan_execution_pose": {
                    "frame_id": execution_frame,
                    "x_m": fresh_pose.x_m,
                    "y_m": fresh_pose.y_m,
                    "yaw_rad": fresh_pose.yaw_rad,
                },
                "post_plan_pose_delta_m": math.hypot(
                    fresh_planning_pose.x_m - confirmed_pose.x_m,
                    fresh_planning_pose.y_m - confirmed_pose.y_m,
                ),
                "post_plan_runtime_revalidated": True,
                "stationary_obstacle_confirmation": (
                    confirmation.evidence.get(
                        "stationary_obstacle_confirmation",
                        {},
                    )
                ),
            },
        )
        self.queued_route_update = update
        refresh = self._refresh_dynamic_route(fresh_pose)
        if refresh == RouteRefreshAction.ADOPTED:
            trace_failure = self._append_controller_trace(
                event="replacement_route_adopted",
                pose=fresh_pose,
                reason=stop_reason,
                fail_closed=False,
                effective_command=VelocityCommand(0.0, 0.0),
                front_cluster_summary=confirmation.evidence,
            )
            if trace_failure:
                return BlockageRecoveryAction.STOPPED
        return BlockageRecoveryAction(str(refresh))

    def _post_replan_admission_pose(self) -> PoseLookupResult:
        """Return a fresh stopped pose after synchronous replacement planning.

        Obstacle proximity is intentionally not re-evaluated here: the sealed
        escape route exists precisely because the robot may still be close to
        the confirmed blocker.  Scan/odom freshness, TF availability, and the
        later exact-start join check remain mandatory before adoption.
        """

        self.publish_zero()
        self._drain_runtime_callbacks()
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return PoseLookupResult(
                None,
                {
                    "stop_reason": ownership_failure,
                    "reason": ownership_failure,
                    "fault_code": (
                        "cmd_vel_ownership_ambiguous_after_replan"
                    ),
                    "source": "post_replan_admission",
                    "fail_closed": True,
                },
            )
        scan_failure = self._freshness_failure(
            "scan",
            self.latest_scan,
            self.latest_scan_receipt,
            self.follower_config.max_scan_age_sec,
        )
        if scan_failure:
            return PoseLookupResult(
                None,
                {
                    **dict(self.latest_stop_details or {}),
                    "stop_reason": scan_failure,
                    "source": "post_replan_admission",
                    "fail_closed": True,
                },
            )
        odom_failure = self._freshness_failure(
            "odom",
            self.latest_odom,
            self.latest_odom_receipt,
            self.follower_config.max_odom_age_sec,
        )
        if odom_failure:
            return PoseLookupResult(
                None,
                {
                    **dict(self.latest_stop_details or {}),
                    "stop_reason": odom_failure,
                    "source": "post_replan_admission",
                    "fail_closed": True,
                },
            )
        localization_failure = self._global_consistency_monitor_failure()
        if localization_failure:
            return PoseLookupResult(
                None,
                {
                    **dict(self.latest_stop_details or {}),
                    "stop_reason": localization_failure,
                    "source": "post_replan_admission",
                    "fail_closed": True,
                },
            )
        pose_lookup = self._current_pose_lookup_with_stale_recovery()
        if pose_lookup.pose is None:
            return PoseLookupResult(
                None,
                {
                    **dict(pose_lookup.details or {}),
                    "stop_reason": str(
                        (pose_lookup.details or {}).get(
                            "stop_reason",
                            "execution-frame transform unavailable",
                        )
                    ),
                    "source": "post_replan_admission",
                    "fail_closed": True,
                },
                pose_lookup.stamp_sec,
            )
        return pose_lookup
