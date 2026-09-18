"""Small inspection turns through the existing sole velocity-publishing node."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict

from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand
from scripts.aufgabe04.navigation.execution.candidate_centering_permit import (
    MAX_ADVISORY_AGE_SEC, STOP_TOLERANCE_RAD,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.bindings import RuntimeBindingProxy


rclpy = RuntimeBindingProxy("rclpy", None)


def fresh_centering_target(scan, advisory, now_sec):
    """Recheck the same narrow measured bearing before starting the turn."""
    from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
    from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
    from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
    model = advisory["intrinsics"]
    u, v = advisory["measured_center_px"]
    ray = ((u-model["cx_px"])/model["fx_px"], (v-model["cy_px"])/model["fy_px"], 1.0)
    direction = rotate_vector(ray, advisory["scan_from_camera"]["rotation_xyzw"])
    if scan.header.frame_id != advisory["scan_from_camera"]["parent_frame"]:
        return None
    distance = advisory["associated_range_m"]
    association = associate_candidate_lidar_target(
        PlainLaserScan(ranges=tuple(scan.ranges), angle_min=scan.angle_min,
            angle_increment=scan.angle_increment, range_min=scan.range_min,
            range_max=scan.range_max, scan_frame_id=scan.header.frame_id,
            scan_stamp_sec=scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9,
            angle_max=scan.angle_max, scan_topology_profile="full_rotation"),
        map_bearing_rad=math.atan2(direction[1], direction[0]),
        cone_half_angle_rad=math.radians(3.0),
        accepted_range_m=(max(0.001, distance-0.04), distance+0.04),
        now_sec=now_sec, max_scan_age_sec=0.5,
    )
    return asdict(association) if association.associated and association.eligible_cluster_count == 1 else None


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def centering_angular_command(error_rad: float) -> float:
    """Proportional slowdown for small turns; never reverse to chase overshoot."""
    if abs(error_rad) <= STOP_TOLERANCE_RAD:
        return 0.0
    return math.copysign(min(0.12, 1.5 * abs(error_rad)), error_rad)


class CenteringOdomTravel:
    """Count every odometry callback, including reversing motion while stopping."""

    def __init__(self, anchor):
        self._lock = threading.Lock()
        self.anchor = anchor
        self.previous = anchor
        self.travel = 0.0
        self.signed_travel = 0.0
        self.translation = 0.0

    def record(self, pose):
        if pose is None:
            return
        with self._lock:
            delta = _wrap(pose.yaw_rad - self.previous.yaw_rad)
            self.travel += abs(delta)
            self.signed_travel += delta
            self.translation = max(self.translation, math.hypot(pose.x_m - self.anchor.x_m, pose.y_m - self.anchor.y_m))
            self.previous = pose

    def snapshot(self):
        with self._lock:
            return self.travel, self.signed_travel, self.translation, self.previous


class CandidateCenteringRuntimeMixin:
    def run_candidate_centering(self, permit):
        """Execute an already sealed turn after fresh, stationary live preflight."""
        started = time.monotonic()
        initial_zero_count = self.zero_command_publish_count
        monitor = None
        turn = float(permit["signed_turn_rad"])
        remaining = float(permit["remaining_travel_rad"])
        advisory = permit["advisory"]
        requested_anchor = advisory["anchor_odom_pose"]
        direction = math.copysign(1.0, turn)
        live_target = None

        def finish(status, reason):
            # Callback accounting remains active throughout zero publication
            # and stop confirmation, so braking and reverse motion spend budget.
            self.publish_repeated_zero(count=10)
            stationary, stop_evidence = self._wait_for_stationary_odom_pair(deadline_monotonic=time.monotonic() + 2.0)
            travel, signed, translation, pose = (monitor.snapshot() if monitor else (0.0, 0.0, 0.0, self._latest_odom_pose()))
            self._candidate_centering_odom_monitor = None
            final_error = _wrap(turn - signed)
            if status == "completed":
                if stationary is None:
                    status, reason = "stopped", "centering stationary stop not confirmed"
                elif travel > remaining + 1e-12 or translation > 0.01:
                    status, reason = "stopped", "centering stop exceeded motion budget"
                elif abs(final_error) > STOP_TOLERANCE_RAD:
                    status, reason = "stopped", "centering stopped outside yaw tolerance"
            return {
                "status": status, "stop_reason": reason,
                "translation_commanded": False,
                "motion_published": bool(self.motion_published),
                "actual_angular_travel_rad": travel,
                "total_angular_travel_rad": float(permit["previous_angular_travel_rad"]) + travel,
                "maximum_translation_m": translation,
                "signed_angular_travel_rad": signed,
                "final_yaw_error_rad": final_error,
                "final_odom_pose": None if pose is None else {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad},
                "stopped_at_sec": self._ros_now_sec(),
                "stationary_odom": stop_evidence,
                "preflight_current_target": live_target,
                "zero_command_count": self.zero_command_publish_count - initial_zero_count,
                "duration_sec": time.monotonic() - started,
            }

        self.publish_repeated_zero(count=10)
        last_failure = "centering fresh inputs unavailable"
        while rclpy.ok() and time.monotonic() - started < 2.0:
            self._service_or_wait_for_callbacks(0.05)
            last_failure = self._safety_failure()
            if not last_failure and self._latest_odom_pose() is not None:
                break
        else:
            return finish("stopped", last_failure)
        stationary, _ = self._wait_for_stationary_odom_pair(deadline_monotonic=time.monotonic() + 1.0)
        if stationary is None:
            return finish("stopped", "centering initial pose is not stationary")
        pose = self._latest_odom_pose()
        age = self._ros_now_sec() - float(advisory["created_at_sec"])
        if not 0.0 <= age <= MAX_ADVISORY_AGE_SEC:
            return finish("stopped", "centering advisory expired before motion")
        if (pose is None or math.hypot(pose.x_m - requested_anchor["x_m"], pose.y_m - requested_anchor["y_m"]) > 0.01
                or abs(_wrap(pose.yaw_rad - requested_anchor["yaw_rad"])) > math.radians(1.0)):
            return finish("stopped", "centering odometry anchor changed")
        # The child must actually have newer scan and odometry messages than
        # the observation that asked it to move, in the same frame contract.
        def stamp(msg):
            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        if (self.latest_odom.header.frame_id != self.runtime_config.odom_frame
                or self.latest_odom.child_frame_id != self.runtime_config.base_frame
                or stamp(self.latest_scan) <= float(advisory["scan_stamp_sec"])
                or stamp(self.latest_odom) <= float(advisory["odom_stamp_sec"])):
            return finish("stopped", "centering live sensor frame or timestamp mismatch")
        live_target = fresh_centering_target(self.latest_scan, advisory, self._ros_now_sec())
        if live_target is None:
            return finish("stopped", "centering current candidate scan association changed")
        if self._safety_failure():
            return finish("stopped", "centering live preflight safety failed")
        monitor = CenteringOdomTravel(pose)
        self._candidate_centering_odom_monitor = monitor
        period = 0.05
        while rclpy.ok():
            self._service_or_wait_for_callbacks(period)
            failure = self._safety_failure()
            if failure:
                return finish("stopped", failure)
            if time.monotonic() - started > float(permit["timeout_sec"]) - 2.5:
                return finish("stopped", "centering turn timeout")
            # record() is idempotent for an unchanged current pose and supports
            # offline harnesses without ROS subscription callbacks.
            monitor.record(self._latest_odom_pose())
            travel, signed, translation, _ = monitor.snapshot()
            if translation > 0.01:
                return finish("stopped", "centering translation bound exceeded")
            if travel >= remaining - STOP_TOLERANCE_RAD:
                return finish("stopped", "centering angular travel budget exhausted")
            error = turn - signed
            if abs(error) <= STOP_TOLERANCE_RAD:
                return finish("completed", "")
            if error * direction < 0:
                return finish("stopped", "centering overshot the requested angle")
            speed = centering_angular_command(error)
            speed = math.copysign(min(abs(speed), self.follower_config.controller.max_angular_radps), speed)
            # Reserve one control cycle plus the tight stop margin before the
            # absolute travel boundary. Actual stop motion is checked again.
            if travel + abs(speed) * period + STOP_TOLERANCE_RAD >= remaining:
                return finish("stopped", "centering braking travel reserve exhausted")
            command = VelocityCommand(linear_x_mps=0.0, angular_z_radps=speed)
            failure = self._append_controller_trace(
                event="candidate_centering_cycle", nominal_command=command,
                effective_command=command,
                diagnostics={"candidate_id": permit["candidate_id"], "view_id": permit["view_id"], "turn_index": permit["turn_index"], "angular_travel_rad": travel, "yaw_error_rad": error, "translation_commanded": False},
                fail_closed=False,
            )
            if failure:
                return finish("stopped", failure)
            self._publish_velocity_command(command)
        return finish("stopped", "ROS shutdown during centering")
