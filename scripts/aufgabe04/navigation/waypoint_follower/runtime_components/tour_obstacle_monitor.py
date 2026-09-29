"""Tour-only forward route obstruction detection before command publication.

The monitor never steers, changes a route, or changes LiDAR safety distances.
It holds zero on the first blocked scan and exits on a second source-distinct
scan. The parent then obtains a separate stopped cohort and fresh localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import (
    FUTURE_TOLERANCE_SEC, MAX_SCAN_AGE_SEC, exact_transform_pose,
    finite, scan_geometry, stamp_seconds,
)

try:  # pragma: no cover - ROS adapter only.
    from rclpy.duration import Duration
    from rclpy.time import Time
except ImportError:  # pragma: no cover
    Duration = Time = None


TOUR_ROUTE_BLOCKED = "stored pose tour route blocked"
TOUR_MONITOR_INVALID = "stored pose tour obstacle monitor unavailable"
LOOKAHEAD_M = .8
EXTRA_CORRIDOR_RADIUS_M = .02 + .03 + .075


@dataclass(frozen=True)
class TourMonitorDecision:
    action: str
    details: Mapping[str, object] | None = None


def scan_points_odom(geometry: Mapping[str, object], pose: Mapping[str, float]):
    """Project only actual finite returns; None supplies no occupancy evidence."""
    x, y, yaw = (finite(pose[name], name) for name in ("x_m", "y_m", "yaw_rad"))
    for index, value in enumerate(geometry["ranges"]):
        if value is None:
            continue
        angle = yaw + geometry["angle_min"] + index * geometry["angle_increment"]
        yield (x + value * math.cos(angle), y + value * math.sin(angle))


def forward_corridor_hit(points: Sequence[tuple[float, float]], *, pose: Pose2D,
                         waypoints: Sequence[Pose2D], target_index: int,
                         corridor_radius_m: float, lookahead_m: float = LOOKAHEAD_M):
    """Find an observed hit in the next bounded swept route centerline.

    Interior corners have round joins. The starting and final cross-sections
    are clipped, so returns behind the robot or beyond a goal are not obstacles
    on this route solely because they are near an endpoint.
    """
    radius, budget = finite(corridor_radius_m, "corridor_radius_m"), finite(lookahead_m, "lookahead_m")
    if radius <= 0 or budget <= 0 or not 0 <= target_index < len(waypoints):
        raise ValueError("invalid tour route corridor")
    start = (finite(pose.x_m, "pose.x_m"), finite(pose.y_m, "pose.y_m"))
    segments = []
    for waypoint in waypoints[target_index:]:
        target = (finite(waypoint.x_m, "waypoint.x_m"), finite(waypoint.y_m, "waypoint.y_m"))
        dx, dy = target[0]-start[0], target[1]-start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        used = min(length, budget)
        end = (start[0]+dx*used/length, start[1]+dy*used/length)
        segments.append((start, end))
        budget -= used
        if budget <= 1e-9:
            break
        start = target
    for point in points:
        px, py = (finite(value, "scan point") for value in point)
        for index, (a, b) in enumerate(segments):
            dx, dy = b[0]-a[0], b[1]-a[1]
            fraction = ((px-a[0])*dx + (py-a[1])*dy)/(dx*dx+dy*dy)
            if (index == 0 and fraction < 0) or (index == len(segments)-1 and fraction > 1):
                continue
            along = max(0., min(1., fraction))
            if math.hypot(px-a[0]-along*dx, py-a[1]-along*dy) <= radius:
                return {"x_m": px, "y_m": py}
    return None


def classify_monitor_scan(*, previous_blocked_stamp: float | None,
                          stamp_sec: float, blocked_point: Mapping[str, float] | None,
                          evidence: Mapping[str, object]):
    """Source-stamp confirmation policy; returns decision and next pending stamp."""
    if blocked_point is None:
        return TourMonitorDecision("clear"), None
    common = {**evidence, "blocked_point_odom": dict(blocked_point),
              "source": "stored_pose_tour_obstacle_monitor", "fail_closed": True}
    delta = None if previous_blocked_stamp is None else stamp_sec-previous_blocked_stamp
    if delta is not None and .08-1e-6 <= delta <= .5+1e-6:
        return TourMonitorDecision("stop", {**common,
            "reason": TOUR_ROUTE_BLOCKED, "fault_code": "stored_pose_tour_route_blocked",
            "previous_scan_stamp_sec": previous_blocked_stamp,
            "confirmed_distinct_scans": 2}), stamp_sec
    pending = previous_blocked_stamp if delta is not None and 0 <= delta < .08 else stamp_sec
    return TourMonitorDecision("hold", {**common,
        "reason": "stored pose tour route obstruction awaiting confirmation",
        "confirmed_distinct_scans": 1}), pending


class TourObstacleMonitorRuntimeMixin:
    def _tour_obstacle_monitor_decision(self, pose, step):
        if not self.follower_config.stored_pose_tour_obstacle_monitor:
            return TourMonitorDecision("clear")
        if step.command.linear_x_mps <= 0:
            self._tour_pending_blocked_stamp = None
            return TourMonitorDecision("clear")
        try:
            if self.current_route_kind != "admitted_candidate_pose" or self.odom_execution_context is None:
                raise ValueError("tour monitor requires the admitted odom route")
            scan = self.latest_scan
            if scan is None or scan.header.frame_id != self.follower_config.stored_pose_tour_scan_frame:
                raise ValueError("tour monitor scan frame mismatch")
            stamp = stamp_seconds(scan.header.stamp)
            now = self.get_clock().now().nanoseconds / 1e9
            if not -FUTURE_TOLERANCE_SEC <= now-stamp <= MAX_SCAN_AGE_SEC:
                raise ValueError("tour monitor scan is stale or future-dated")
            query = Time.from_msg(scan.header.stamp)
            odom_frame, base_frame = self.runtime_config.odom_frame, self.runtime_config.base_frame
            scan_tf = self.tf_buffer.lookup_transform(odom_frame, scan.header.frame_id,
                query, timeout=Duration(seconds=0))
            base_tf = self.tf_buffer.lookup_transform(odom_frame, base_frame,
                query, timeout=Duration(seconds=0))
            scan_pose = exact_transform_pose(scan_tf, target_frame=odom_frame,
                source_frame=scan.header.frame_id, stamp_sec=stamp)
            base_pose = exact_transform_pose(base_tf, target_frame=odom_frame,
                source_frame=base_frame, stamp_sec=stamp)
            radius = self.follower_config.stored_pose_tour_robot_radius_m + EXTRA_CORRIDOR_RADIUS_M
            hit = forward_corridor_hit(tuple(scan_points_odom(scan_geometry(scan), scan_pose)),
                pose=pose, waypoints=self.waypoints, target_index=step.target_index,
                corridor_radius_m=radius)
            decision, self._tour_pending_blocked_stamp = classify_monitor_scan(
                previous_blocked_stamp=getattr(self, "_tour_pending_blocked_stamp", None),
                stamp_sec=stamp, blocked_point=hit, evidence={
                    "execution_frame": odom_frame, "scan_frame": scan.header.frame_id,
                    "scan_stamp_sec": stamp, "scan_pose_odom": scan_pose, "base_pose_odom": base_pose,
                    "lookahead_m": LOOKAHEAD_M, "corridor_radius_m": radius})
            return decision
        except Exception as exc:
            # This is a terminal sensor/TF/geometry fault, never replan evidence.
            return TourMonitorDecision("stop", {
                "reason": TOUR_MONITOR_INVALID, "source": "stored_pose_tour_obstacle_monitor",
                "fault_code": "stored_pose_tour_monitor_invalid", "error": str(exc), "fail_closed": True})
