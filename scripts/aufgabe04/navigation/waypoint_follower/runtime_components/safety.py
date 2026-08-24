"""Live sensor, obstacle, progress, and command-ownership checks."""

from __future__ import annotations

import math
import time

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from rclpy.time import Time
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Time = None

from scripts.aufgabe04.navigation.control.follower_safety import (
    NO_VALID_FRONT_SECTOR_SCAN_RANGES,
    OBSTACLE_TOO_CLOSE,
    cmd_vel_ownership_failure,
    front_sector_decision,
    linear_scale_for_front_clearance,
    message_freshness_failure,
    obstacle_decision,
    stuck_progress_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import (
    RuntimeBindingProxy,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.node_identity import (
    format_node_identity,
    node_identity,
)

rclpy = RuntimeBindingProxy("rclpy", rclpy)
Time = RuntimeBindingProxy("Time", Time)


class SafetyRuntimeMixin:
    """Live safety behavior mixed into the sole follower node."""

    def _wait_for_initial_runtime_inputs(self, started_at: float) -> str:
        deadline = started_at + self.follower_config.initial_sensor_wait_sec
        last_failure = "missing scan"
        while rclpy.ok():
            self._service_or_wait_for_callbacks(0.05)
            scan_failure = self._freshness_failure(
                "scan",
                self.latest_scan,
                self.latest_scan_receipt,
                self.follower_config.max_scan_age_sec,
            )
            if scan_failure:
                last_failure = scan_failure
            else:
                odom_failure = self._freshness_failure(
                    "odom",
                    self.latest_odom,
                    self.latest_odom_receipt,
                    self.follower_config.max_odom_age_sec,
                )
                if odom_failure:
                    last_failure = odom_failure
                else:
                    pose_lookup = self._current_pose_lookup()
                    if pose_lookup.pose is None:
                        self.latest_stop_details = pose_lookup.details
                        last_failure = "map-to-base transform unavailable"
                    else:
                        # Odom-owned execution still depends on the live
                        # map<-odom edge as a read-only global-consistency
                        # monitor.  A newly constructed child TF buffer can
                        # receive odom<-base before map<-odom, so warm and
                        # validate that second edge while motion remains zero
                        # and inside the existing bounded startup wait.
                        localization_failure = (
                            self._global_consistency_monitor_failure()
                        )
                        if localization_failure:
                            last_failure = localization_failure
                        else:
                            self.latest_stop_details = None
                            return ""
            if time.monotonic() >= deadline:
                return last_failure
            self.publish_zero()
        return "ROS shutdown"

    def _safety_failure(self) -> str:
        self.latest_stop_details = None
        scan_failure = self._freshness_failure("scan", self.latest_scan, self.latest_scan_receipt, self.follower_config.max_scan_age_sec)
        if scan_failure:
            return scan_failure
        odom_failure = self._freshness_failure("odom", self.latest_odom, self.latest_odom_receipt, self.follower_config.max_odom_age_sec)
        if odom_failure:
            return odom_failure
        obstacle_failure = self._obstacle_failure()
        if obstacle_failure:
            return obstacle_failure
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return ownership_failure
        return ""

    def _freshness_failure(self, name: str, msg, receipt, max_age_sec: float) -> str:
        if msg is None or receipt is None:
            failure = message_freshness_failure(
                name,
                has_message=False,
                receipt_age_sec=None,
                header_age_sec=None,
                max_age_sec=max_age_sec,
            )
            self.latest_stop_details = {
                "reason": failure,
                "source": "message_freshness",
                "sensor": name,
                "has_message": False,
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "fail_closed": True,
            }
            return failure
        now = self.get_clock().now()
        receipt_age = time.monotonic() - float(receipt)
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        failure = message_freshness_failure(
            name,
            has_message=True,
            receipt_age_sec=receipt_age,
            header_age_sec=header_age,
            max_age_sec=max_age_sec,
            max_future_sec=self.follower_config.max_future_timestamp_sec,
        )
        if failure:
            self.latest_stop_details = {
                "reason": failure,
                "source": "message_freshness",
                "sensor": name,
                "has_message": True,
                "receipt_age_sec": receipt_age,
                "header_age_sec": header_age,
                "max_age_sec": max_age_sec,
                "max_future_sec": self.follower_config.max_future_timestamp_sec,
                "receipt_stale": receipt_age > max_age_sec,
                "header_stale": header_age > max_age_sec,
                "receipt_future": (
                    receipt_age < -self.follower_config.max_future_timestamp_sec
                ),
                "header_future": (
                    header_age < -self.follower_config.max_future_timestamp_sec
                ),
                "fail_closed": True,
            }
        return failure

    def _scan_range_min(self) -> float | None:
        return float(getattr(self.latest_scan, "range_min")) if hasattr(self.latest_scan, "range_min") else None

    def _scan_range_max(self) -> float | None:
        return float(getattr(self.latest_scan, "range_max")) if hasattr(self.latest_scan, "range_max") else None

    def _obstacle_failure(self) -> str:
        if getattr(self, "blockage_recovery_provider", None) is not None:
            hard = obstacle_decision(
                getattr(self.latest_scan, "ranges", None),
                self.follower_config.omnidirectional_hard_stop_distance_m,
                range_min_m=self._scan_range_min(),
                range_max_m=self._scan_range_max(),
                source="global_hard_scan",
            )
            if hard.stop_reason:
                self.latest_stop_details = hard.to_log_dict()
                return hard.stop_reason
            reversing = self.start_egress_reverse
            directional = front_sector_decision(
                getattr(self.latest_scan, "ranges", None),
                float(getattr(self.latest_scan, "angle_min", 0.0)),
                float(getattr(self.latest_scan, "angle_increment", 0.0)),
                math.pi if reversing else 0.0,
                self.follower_config.front_obstacle_sector_rad,
                self.follower_config.min_obstacle_distance_m,
                range_min_m=self._scan_range_min(),
                range_max_m=self._scan_range_max(),
                source="rear_sector" if reversing else "front_sector",
            )
            if directional.stop_reason:
                directional_details = directional.to_log_dict()
                self.latest_stop_details = {
                    **directional_details,
                    # The transient planner accepts only explicitly bounded
                    # front evidence. Rear blockage during a reverse escape is
                    # an unrecoverable safety stop, never a new forward keepout.
                    **(
                        {"front_clearance": directional_details}
                        if not reversing
                        else {}
                    ),
                }
                return directional.stop_reason
            return ""
        decision = obstacle_decision(
            getattr(self.latest_scan, "ranges", None),
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
        )
        if decision.stop_reason:
            self.latest_stop_details = decision.to_log_dict()
        return decision.stop_reason

    def _motion_clearance_linear_scale(self, linear_x_mps: float) -> float:
        if self.latest_scan is None:
            return 1.0
        if abs(linear_x_mps) <= 1.0e-12:
            self.latest_front_clearance_details = None
            return 1.0
        reversing = linear_x_mps < 0.0
        decision = front_sector_decision(
            getattr(self.latest_scan, "ranges", None),
            float(getattr(self.latest_scan, "angle_min", 0.0)),
            float(getattr(self.latest_scan, "angle_increment", 0.0)),
            math.pi if reversing else 0.0,
            self.follower_config.front_obstacle_sector_rad,
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
            source="rear_sector" if reversing else "front_sector",
        )
        self.latest_front_clearance_details = decision.to_log_dict()
        if decision.stop_reason in (NO_VALID_FRONT_SECTOR_SCAN_RANGES, OBSTACLE_TOO_CLOSE):
            return 0.0
        return linear_scale_for_front_clearance(
            decision.nearest_valid_range_m,
            self.follower_config.min_obstacle_distance_m,
            self.follower_config.front_obstacle_slow_distance_m,
        )

    def _progress_failure(
        self,
        distance_to_target_m: float,
        controlled_heading_error_rad: float,
        target_index: int,
        pursuit_index: int,
        now_monotonic: float,
        motion_commanded: bool,
        progress_mode: str = "path_tracking",
        distance_progress_epsilon_m: float | None = None,
    ) -> str:
        if distance_progress_epsilon_m is None:
            distance_progress_epsilon_m = (
                self.follower_config.stuck_progress_epsilon_m
            )
        if (
            not math.isfinite(distance_progress_epsilon_m)
            or distance_progress_epsilon_m < 0.0
        ):
            raise ValueError(
                "distance_progress_epsilon_m must be finite and non-negative"
            )
        heading_error_abs = abs(controlled_heading_error_rad)
        target_changed = target_index != self.last_progress_target_index
        pursuit_advanced = (
            not target_changed
            and pursuit_index > self.last_progress_pursuit_index
        )
        # Pure-pursuit progress is monotonic. A pursuit-index regression or
        # same-target chatter must not renew the stuck watchdog indefinitely.
        indices_changed = target_changed or pursuit_advanced
        heading_mode_first_entry = (
            progress_mode
            in {
                "exact_vertex_alignment",
                "heading_corridor",
                "terminal_heading",
            }
            and progress_mode not in self.progress_heading_modes_seen
        )
        if indices_changed:
            self.progress_heading_modes_seen.clear()
            self.progress_heading_error_by_mode.clear()
            heading_mode_first_entry = progress_mode in {
                "exact_vertex_alignment",
                "heading_corridor",
                "terminal_heading",
            }
        if heading_mode_first_entry:
            self.progress_heading_modes_seen.add(progress_mode)
        if indices_changed or heading_mode_first_entry:
            self.last_progress_distance_m = distance_to_target_m
            self.last_progress_heading_error_rad = (
                heading_error_abs if math.isfinite(heading_error_abs) else math.inf
            )
            self.progress_heading_error_by_mode[progress_mode] = (
                self.last_progress_heading_error_rad
            )
            self.last_progress_target_index = target_index
            self.last_progress_pursuit_index = pursuit_index
            self.last_progress_mode = progress_mode
            self.last_progress_at = now_monotonic
            return ""
        self.last_progress_mode = progress_mode
        mode_heading_baseline = self.progress_heading_error_by_mode.get(
            progress_mode
        )
        if mode_heading_baseline is None:
            # Path-bearing and terminal-yaw errors are different metrics.  A
            # tolerance-boundary chatter may enter a mode for the first time
            # without constituting progress, so establish its own baseline
            # without renewing the watchdog deadline.
            mode_heading_baseline = (
                heading_error_abs if math.isfinite(heading_error_abs) else math.inf
            )
            self.progress_heading_error_by_mode[progress_mode] = (
                mode_heading_baseline
            )
        self.last_progress_heading_error_rad = mode_heading_baseline
        distance_improved = (
            distance_to_target_m + distance_progress_epsilon_m
            < self.last_progress_distance_m
        )
        heading_improved = (
            math.isfinite(heading_error_abs)
            and heading_error_abs
            + self.follower_config.stuck_heading_progress_epsilon_rad
            < mode_heading_baseline
        )
        if distance_improved:
            self.last_progress_distance_m = distance_to_target_m
        if heading_improved:
            self.progress_heading_error_by_mode[progress_mode] = heading_error_abs
            self.last_progress_heading_error_rad = heading_error_abs
        if distance_improved or heading_improved:
            self.last_progress_at = now_monotonic
            return ""
        return stuck_progress_failure(
            now_monotonic - self.last_progress_at,
            self.follower_config.stuck_timeout_sec,
            motion_commanded,
        )

    def _cmd_vel_ownership_failure(self) -> str:
        publishers = self.get_publishers_info_by_topic(self.runtime_config.cmd_vel_topic)
        publisher_identities = sorted({_node_identity(publisher) for publisher in publishers})
        self_identity = _format_node_identity(self.get_namespace(), self.get_name())
        return cmd_vel_ownership_failure(
            publisher_identities,
            self_identity,
            # Allow-lists are useful for preflight discovery, but publishing
            # begins only when this process is the sole cmd_vel owner.
            (),
        )
