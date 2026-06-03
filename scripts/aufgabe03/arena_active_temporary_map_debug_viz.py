#!/usr/bin/env python3
"""
Publish the arena-active temporary exploration maps for RViz.

This node is read-only: it subscribes to /scan and /odom, rebuilds the same
odom-frame LocalGrid used by active-explore localization recovery, publishes an
observed OccupancyGrid for free-space inspection, and publishes the inflated
planning OccupancyGrid on a second topic. It does not publish /cmd_vel or
interact with Nav2.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

try:
    import rclpy
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import OccupancyGrid, Odometry
    from nav_msgs.msg import Path as NavPath
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    rclpy = None
    Point = None
    PoseStamped = None
    OccupancyGrid = None
    Odometry = object
    NavPath = None
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    qos_profile_sensor_data = None
    LaserScan = object
    Marker = None
    MarkerArray = None

from arena_active_explore import (
    ActiveExploreConfig,
    ActiveExplorePlan,
    RawCandidate,
    empty_local_grid,
    finalize_grid,
    grid_cell_counts,
    mark_scan_on_grid,
    min_scan_range_in_sector,
    normalize_angle_rad,
    plan_candidate,
    point_from_heading,
)
from arena_active_spin_core.curve_following import active_explore_curve_path
from arena_active_spin_core.models import ArenaActiveSpinConfig
from arena_active_spin_core.scan_safety import odom_pose_from_msg, scan_sample_from_msg
from arena_active_spin_core.temporary_map import temporary_map_occupancy_data
from two_stage_waypoint.model import (
    DEFAULT_ARENA_ACTIVE_EXPLORE_CANDIDATE_MARKER_TOPIC,
    DEFAULT_ARENA_ACTIVE_EXPLORE_PATH_TOPIC,
    DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_FRAME,
    DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_PUBLISH_PERIOD_SEC,
    DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_TOPIC,
    DEFAULT_ARENA_ACTIVE_TEMPORARY_PLANNING_MAP_TOPIC,
)


DEFAULT_SCAN_TOPIC = "/scan"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_STATUS_PERIOD_SEC = 2.0
DEFAULT_ARENA_ACTIVE_EXPLORE_DECISION_MARKER_TOPIC = (
    "/mii_amr/arena_active/explore_decisions"
)


def arena_active_default(field_name):
    return ArenaActiveSpinConfig.__dataclass_fields__[field_name].default


DEFAULT_MAX_ODOM_SCAN_AGE_SEC = arena_active_default("max_odom_scan_age_sec")
DEFAULT_MAP_MAX_SAMPLES = arena_active_default("active_explore_map_max_samples")


@dataclass(frozen=True)
class Rgba:
    r: float
    g: float
    b: float
    a: float = 1.0


COLOR_SELECTED_PATH = Rgba(0.0, 0.95, 0.25, 1.0)
COLOR_SELECTED_ASTAR = Rgba(0.0, 0.65, 1.0, 0.70)
COLOR_ACCEPTED_CANDIDATE = Rgba(0.0, 0.65, 1.0, 0.90)
COLOR_ACCEPTED_HIGH_SCORE = Rgba(0.0, 0.85, 0.95, 0.85)
COLOR_ACCEPTED_MEDIUM_SCORE = Rgba(0.95, 0.78, 0.05, 0.85)
COLOR_ACCEPTED_LOW_SCORE = Rgba(1.0, 0.45, 0.0, 0.80)
COLOR_REJECTED_CANDIDATE = Rgba(1.0, 0.25, 0.15, 0.65)
COLOR_SELECTED_CANDIDATE = Rgba(0.0, 0.95, 0.25, 1.0)
COLOR_ALTERNATIVE_ASTAR = Rgba(0.85, 0.85, 0.85, 0.45)


@dataclass(frozen=True)
class DebugGridPair:
    display_grid: object | None = None
    planning_grid: object | None = None


def temporary_map_config_from_args(args):
    return ActiveExploreConfig(
        max_attempts=arena_active_default("active_explore_max_attempts"),
        max_single_move_m=getattr(
            args,
            "max_single_move_m",
            arena_active_default("active_explore_max_single_move_m"),
        ),
        max_total_distance_m=getattr(
            args,
            "max_total_distance_m",
            arena_active_default("active_explore_max_total_distance_m"),
        ),
        max_candidate_path_m=getattr(
            args,
            "max_candidate_path_m",
            arena_active_default("active_explore_max_candidate_path_m"),
        ),
        grid_resolution_m=args.grid_resolution_m,
        grid_size_m=args.grid_size_m,
        inflation_radius_m=args.inflation_radius_m,
        soft_clearance_radius_m=getattr(
            args,
            "soft_clearance_radius_m",
            arena_active_default("active_explore_soft_clearance_radius_m"),
        ),
        soft_clearance_weight=getattr(
            args,
            "soft_clearance_weight",
            arena_active_default("active_explore_soft_clearance_weight"),
        ),
        unknown_blocked=getattr(
            args,
            "unknown_blocked",
            arena_active_default("active_explore_unknown_blocked"),
        ),
        max_path_segments=arena_active_default("active_explore_max_path_segments"),
        target_nearest_short_wall_range_m=arena_active_default(
            "center_reposition_target_nearest_short_wall_range_m"
        ),
        center_min_step_m=arena_active_default("center_reposition_min_step_m"),
        lateral_offset_threshold_m=arena_active_default(
            "center_reposition_lateral_offset_threshold_m"
        ),
        lateral_target_offset_m=arena_active_default(
            "center_reposition_lateral_target_offset_m"
        ),
        heater_approach_target_range_m=arena_active_default(
            "center_reposition_heater_approach_target_range_m"
        ),
        heater_approach_min_selected_score=arena_active_default(
            "center_reposition_heater_approach_min_selected_score"
        ),
        heater_approach_max_opposite_score=arena_active_default(
            "center_reposition_heater_approach_max_opposite_score"
        ),
        heater_approach_min_delta=arena_active_default(
            "center_reposition_heater_approach_min_delta"
        ),
    )


def trim_scan_samples(samples, max_samples):
    max_samples = max(1, int(max_samples))
    if len(samples) <= max_samples:
        return samples
    return samples[-max_samples:]


def require_debug_grid_inputs(scan_samples, latest_odom_pose):
    if not scan_samples:
        raise RuntimeError("No scan samples are available for the temporary map")
    if latest_odom_pose is None:
        raise RuntimeError("No odom pose is available for the temporary map")


def build_debug_grids(
    scan_samples,
    latest_odom_pose,
    config,
    need_display_grid=True,
    need_planning_grid=True,
):
    if not need_display_grid and not need_planning_grid:
        return DebugGridPair()
    require_debug_grid_inputs(scan_samples, latest_odom_pose)
    grid, mutable, robot_cell = empty_local_grid(latest_odom_pose, config)
    occupied_cells = set()
    for sample in scan_samples:
        scan_pose = getattr(sample, "odom_pose", None)
        if scan_pose is None:
            continue
        occupied_cells.update(
            mark_scan_on_grid(
                mutable,
                grid,
                sample,
                scan_pose,
                config,
                preserve_occupied=True,
            )
        )

    display_grid = None
    planning_grid = None
    if need_display_grid:
        display_grid = finalize_grid(
            latest_odom_pose,
            config,
            mutable,
            robot_cell,
            occupied_cells,
            inflation_radius_m=0.0,
        )
    if need_planning_grid:
        planning_grid = finalize_grid(
            latest_odom_pose,
            config,
            mutable,
            robot_cell,
            occupied_cells,
        )
    return DebugGridPair(display_grid, planning_grid)


def build_debug_grid(scan_samples, latest_odom_pose, config):
    return build_debug_grids(
        scan_samples,
        latest_odom_pose,
        config,
        need_display_grid=False,
        need_planning_grid=True,
    ).planning_grid


def build_debug_display_grid(scan_samples, latest_odom_pose, config):
    return build_debug_grids(
        scan_samples,
        latest_odom_pose,
        config,
        need_display_grid=True,
        need_planning_grid=False,
    ).display_grid


def open_corridor_raw_candidates(scan, robot_pose, config):
    yaw = math.radians(float(robot_pose.yaw_deg))
    raw = []
    for angle_deg in (-90, -60, -30, 0, 30, 60, 90):
        sector_min = min_scan_range_in_sector(
            scan,
            angle_deg - 8.0,
            angle_deg + 8.0,
        )
        if sector_min is None:
            continue
        usable_distance = sector_min - config.inflation_radius_m
        if usable_distance < config.center_min_step_m:
            continue
        distance = min(config.max_single_move_m, usable_distance)
        heading = normalize_angle_rad(yaw + math.radians(angle_deg))
        x, y = point_from_heading(robot_pose, heading, distance)
        raw.append(
            RawCandidate(
                "open_corridor",
                x,
                y,
                heading,
                geometry_progress=0.25 * distance / config.max_single_move_m,
                metadata={
                    "sector_center_deg": angle_deg,
                    "sector_min_range_m": sector_min,
                    "preview_source": "latest_scan_open_corridor",
                },
            )
        )
    return tuple(raw)


def build_debug_active_explore_plan(scan, robot_pose, grid, config):
    raw_candidates = open_corridor_raw_candidates(scan, robot_pose, config)
    candidates = tuple(plan_candidate(raw, grid, config) for raw in raw_candidates)
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return ActiveExplorePlan(
            False,
            "no_reachable_open_corridor_candidate",
            None,
            candidates,
            grid,
        )
    selected = max(
        accepted,
        key=lambda candidate: (
            candidate.score if candidate.score is not None else float("-inf")
        ),
    )
    return ActiveExplorePlan(True, "selected_open_corridor_preview", selected, candidates, grid)


def rviz_occupancy_grid_qos_profile():
    if QoSProfile is None or DurabilityPolicy is None:
        return 1
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def build_occupancy_grid_message(grid, frame_id, stamp):
    if OccupancyGrid is None:
        raise RuntimeError(
            "nav_msgs.msg.OccupancyGrid is unavailable; source ROS 2 Humble first"
        )
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution_m)
    msg.info.width = int(grid.width)
    msg.info.height = int(grid.height)
    msg.info.origin.position.x = float(grid.origin_x)
    msg.info.origin.position.y = float(grid.origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.x = 0.0
    msg.info.origin.orientation.y = 0.0
    msg.info.origin.orientation.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = temporary_map_occupancy_data(grid)
    return msg


def point_msg(x, y, z=0.0):
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def pose_stamped_msg(x, y, frame_id, stamp):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0
    pose.pose.orientation.w = 1.0
    return pose


def build_path_message(points, frame_id, stamp):
    if NavPath is None or PoseStamped is None:
        raise RuntimeError("nav_msgs/geometry_msgs path types are unavailable")
    msg = NavPath()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.poses = [pose_stamped_msg(x, y, frame_id, stamp) for x, y in points]
    return msg


def executable_path_points(candidate, current_pose, max_distance_m):
    if candidate is None or current_pose is None:
        return ()
    return active_explore_curve_path(candidate, current_pose, max_distance_m)


def apply_marker_common(marker, frame_id, stamp, namespace, marker_id, color):
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.color.r = color.r
    marker.color.g = color.g
    marker.color.b = color.b
    marker.color.a = color.a


def delete_all_marker():
    marker = Marker()
    marker.action = Marker.DELETEALL
    return marker


def line_strip_marker(frame_id, stamp, namespace, marker_id, points, color, width_m):
    if not points:
        return None
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, color)
    marker.type = Marker.LINE_STRIP
    marker.scale.x = width_m
    marker.points = [point_msg(x, y, 0.04) for x, y in points]
    return marker


def sphere_list_marker(frame_id, stamp, namespace, marker_id, points, color, scale_m):
    if not points:
        return None
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, color)
    marker.type = Marker.SPHERE_LIST
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    marker.points = [point_msg(x, y, 0.06) for x, y in points]
    return marker


def append_marker(markers, marker):
    if marker is not None:
        markers.append(marker)


def marker_namespace_suffix(value):
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    suffix = "_".join(part for part in "".join(chars).split("_") if part)
    return suffix or "unknown"


def accepted_score_bounds(candidates):
    scores = [
        float(candidate.score)
        for candidate in candidates
        if candidate.accepted and candidate.score is not None
    ]
    if not scores:
        return None, None
    return min(scores), max(scores)


def accepted_candidate_score_bucket(candidate, min_score, max_score):
    if candidate.score is None:
        return "unscored", COLOR_ACCEPTED_CANDIDATE
    if min_score is None or max_score is None or max_score <= min_score:
        return "high", COLOR_ACCEPTED_HIGH_SCORE
    fraction = (float(candidate.score) - min_score) / (max_score - min_score)
    if fraction >= 0.67:
        return "high", COLOR_ACCEPTED_HIGH_SCORE
    if fraction >= 0.34:
        return "medium", COLOR_ACCEPTED_MEDIUM_SCORE
    return "low", COLOR_ACCEPTED_LOW_SCORE


def build_candidate_marker_array(plan, current_pose, max_distance_m, frame_id, stamp):
    if Marker is None or MarkerArray is None or Point is None:
        raise RuntimeError("visualization_msgs marker types are unavailable")
    markers = [delete_all_marker()]
    selected = plan.selected if plan is not None else None
    accepted_endpoints = []
    rejected_endpoints = []
    for candidate in plan.candidates if plan is not None else ():
        point = (candidate.target_x, candidate.target_y)
        if candidate.accepted:
            accepted_endpoints.append(point)
        else:
            rejected_endpoints.append(point)

    append_marker(
        markers,
        sphere_list_marker(
            frame_id,
            stamp,
            "active_explore_accepted_candidates",
            1,
            accepted_endpoints,
            COLOR_ACCEPTED_CANDIDATE,
            0.06,
        ),
    )
    append_marker(
        markers,
        sphere_list_marker(
            frame_id,
            stamp,
            "active_explore_rejected_candidates",
            2,
            rejected_endpoints,
            COLOR_REJECTED_CANDIDATE,
            0.045,
        ),
    )
    if selected is not None:
        append_marker(
            markers,
            line_strip_marker(
                frame_id,
                stamp,
                "active_explore_selected_astar_path",
                3,
                selected.path_world,
                COLOR_SELECTED_ASTAR,
                0.012,
            ),
        )
        curve_points = executable_path_points(selected, current_pose, max_distance_m)
        append_marker(
            markers,
            line_strip_marker(
                frame_id,
                stamp,
                "active_explore_selected_curve_path",
                4,
                curve_points,
                COLOR_SELECTED_PATH,
                0.025,
            ),
        )
        append_marker(
            markers,
            sphere_list_marker(
                frame_id,
                stamp,
                "active_explore_selected_candidate",
                5,
                [(selected.target_x, selected.target_y)],
                COLOR_SELECTED_CANDIDATE,
                0.08,
            ),
        )
    return MarkerArray(markers=markers)


def build_candidate_decision_marker_array(plan, frame_id, stamp, max_alternative_paths=3):
    if Marker is None or MarkerArray is None or Point is None:
        raise RuntimeError("visualization_msgs marker types are unavailable")
    markers = [delete_all_marker()]
    if plan is None:
        return MarkerArray(markers=markers)

    selected = plan.selected
    candidates = tuple(plan.candidates)
    min_score, max_score = accepted_score_bounds(candidates)
    accepted_points_by_bucket = {}
    rejected_points_by_reason = {}
    accepted_alternatives = []

    for candidate in candidates:
        point = (candidate.target_x, candidate.target_y)
        if candidate.accepted:
            if candidate != selected:
                bucket, _color = accepted_candidate_score_bucket(
                    candidate,
                    min_score,
                    max_score,
                )
                accepted_points_by_bucket.setdefault(bucket, []).append(point)
                accepted_alternatives.append(candidate)
            continue
        reason = marker_namespace_suffix(candidate.rejection_reason or "rejected")
        rejected_points_by_reason.setdefault(reason, []).append(point)

    bucket_colors = {
        "high": COLOR_ACCEPTED_HIGH_SCORE,
        "medium": COLOR_ACCEPTED_MEDIUM_SCORE,
        "low": COLOR_ACCEPTED_LOW_SCORE,
        "unscored": COLOR_ACCEPTED_CANDIDATE,
    }
    marker_id = 10
    for bucket in ("high", "medium", "low", "unscored"):
        points = accepted_points_by_bucket.get(bucket, ())
        append_marker(
            markers,
            sphere_list_marker(
                frame_id,
                stamp,
                f"active_explore_decision_accepted_{bucket}",
                marker_id,
                points,
                bucket_colors[bucket],
                0.055,
            ),
        )
        marker_id += 1

    for reason in sorted(rejected_points_by_reason):
        append_marker(
            markers,
            sphere_list_marker(
                frame_id,
                stamp,
                f"active_explore_decision_rejected_{reason}",
                marker_id,
                rejected_points_by_reason[reason],
                COLOR_REJECTED_CANDIDATE,
                0.04,
            ),
        )
        marker_id += 1

    if selected is not None:
        append_marker(
            markers,
            sphere_list_marker(
                frame_id,
                stamp,
                "active_explore_decision_selected",
                marker_id,
                [(selected.target_x, selected.target_y)],
                COLOR_SELECTED_CANDIDATE,
                0.09,
            ),
        )
        marker_id += 1

    accepted_alternatives.sort(
        key=lambda candidate: (
            candidate.score if candidate.score is not None else float("-inf")
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(
        accepted_alternatives[:max(0, int(max_alternative_paths))],
        start=1,
    ):
        append_marker(
            markers,
            line_strip_marker(
                frame_id,
                stamp,
                f"active_explore_decision_alternative_astar_rank_{rank}",
                marker_id,
                candidate.path_world,
                COLOR_ALTERNATIVE_ASTAR,
                0.007,
            ),
        )
        marker_id += 1

    return MarkerArray(markers=markers)


def pose_cache_signature(pose):
    if pose is None:
        return None
    return (
        float(getattr(pose, "x", 0.0)),
        float(getattr(pose, "y", 0.0)),
        float(getattr(pose, "yaw_deg", 0.0)),
    )


def config_cache_signature(config):
    return (
        float(config.grid_resolution_m),
        float(config.grid_size_m),
        float(config.inflation_radius_m),
        float(config.soft_clearance_radius_m),
        float(config.soft_clearance_weight),
        bool(config.unknown_blocked),
        float(config.max_single_move_m),
        float(config.max_total_distance_m),
        None
        if config.max_candidate_path_m is None
        else float(config.max_candidate_path_m),
    )


def publisher_subscription_count(publisher):
    get_count = getattr(publisher, "get_subscription_count", None)
    if get_count is None:
        return None
    try:
        return int(get_count())
    except Exception:
        return None


class ArenaActiveTemporaryMapDebugViz(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run this RViz publisher.")
        if OccupancyGrid is None:
            raise RuntimeError(
                "nav_msgs.msg.OccupancyGrid is unavailable. Source ROS 2 Humble before "
                "running the temporary map debug publisher."
            )
        super().__init__("mii_amr_arena_active_temporary_map_debug_viz")
        self.args = args
        self.config = temporary_map_config_from_args(args)
        self.latest_odom_pose = None
        self.latest_odom_received_sec = None
        self.latest_scan = None
        self.scan_samples = []
        self.rejected_scan_count = 0
        self.last_status_sec = 0.0
        self.accepted_scan_revision = 0
        self.odom_revision = 0
        self.grid_cache_key = None
        self.cached_display_grid = None
        self.cached_planning_grid = None

        self.map_pub = None
        if args.publish_observed_map_viz:
            self.map_pub = self.create_publisher(
                OccupancyGrid,
                args.map_topic,
                rviz_occupancy_grid_qos_profile(),
            )
        self.planning_map_pub = None
        if args.publish_planning_map_viz:
            self.planning_map_pub = self.create_publisher(
                OccupancyGrid,
                args.planning_map_topic,
                rviz_occupancy_grid_qos_profile(),
            )
        self.path_pub = None
        self.marker_pub = None
        self.decision_marker_pub = None
        if args.publish_path_viz:
            if NavPath is None or PoseStamped is None or MarkerArray is None:
                raise RuntimeError(
                    "ROS RViz path/marker message types are unavailable. Source ROS 2 "
                    "Humble before running active-explore path visualization."
                )
            self.path_pub = self.create_publisher(
                NavPath,
                args.path_topic,
                rviz_occupancy_grid_qos_profile(),
            )
            self.marker_pub = self.create_publisher(
                MarkerArray,
                args.candidate_marker_topic,
                rviz_occupancy_grid_qos_profile(),
            )
            if args.publish_decision_viz:
                self.decision_marker_pub = self.create_publisher(
                    MarkerArray,
                    args.decision_marker_topic,
                    rviz_occupancy_grid_qos_profile(),
                )
        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            10,
        )
        self.map_timer = None
        if args.publish_observed_map_viz or args.publish_planning_map_viz:
            self.map_timer = self.create_timer(
                args.map_publish_period_sec,
                self.map_timer_callback,
            )
        self.path_timer = None
        if args.publish_path_viz:
            self.path_timer = self.create_timer(
                args.path_publish_period_sec,
                self.path_timer_callback,
            )
        self.get_logger().info(
            "Publishing arena-active temporary map debug visualization: "
            f"observed_map={args.map_topic if args.publish_observed_map_viz else 'disabled'}, "
            f"planning_map={args.planning_map_topic if args.publish_planning_map_viz else 'disabled'}, "
            f"frame={args.map_frame}, "
            f"scan={args.scan_topic}, odom={args.odom_topic}, "
            f"path={args.path_topic if args.publish_path_viz else 'disabled'}, "
            f"decisions={args.decision_marker_topic if args.publish_path_viz and args.publish_decision_viz else 'disabled'}, "
            f"subscriber_gating={'enabled' if args.subscriber_gating else 'disabled'}"
        )

    def now_sec(self):
        return time.time()

    def clear_grid_cache(self):
        self.grid_cache_key = None
        self.cached_display_grid = None
        self.cached_planning_grid = None

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_received_sec = self.now_sec()
        self.odom_revision += 1
        self.clear_grid_cache()

    def scan_callback(self, msg):
        self.latest_scan = msg
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            self.rejected_scan_count += 1
            return
        if self.now_sec() - self.latest_odom_received_sec > self.args.max_odom_scan_age_sec:
            self.rejected_scan_count += 1
            return
        self.scan_samples.append(scan_sample_from_msg(msg, self.latest_odom_pose))
        self.scan_samples = trim_scan_samples(
            self.scan_samples,
            self.args.map_max_samples,
        )
        self.accepted_scan_revision += 1
        self.clear_grid_cache()

    def publisher_should_publish(self, publisher):
        if publisher is None:
            return False
        if not self.args.subscriber_gating:
            return True
        count = publisher_subscription_count(publisher)
        if count is None:
            return True
        return count > 0

    def output_skip_reason(self, publisher, enabled=True):
        if not enabled or publisher is None:
            return "disabled"
        if not self.args.subscriber_gating:
            return None
        count = publisher_subscription_count(publisher)
        if count is None:
            return None
        if count <= 0:
            return "no_subscribers"
        return None

    def path_skip_reason(self):
        if not self.args.publish_path_viz:
            return "disabled"
        reasons = [
            ("path", self.output_skip_reason(self.path_pub, enabled=True)),
            ("markers", self.output_skip_reason(self.marker_pub, enabled=True)),
            (
                "decisions",
                self.output_skip_reason(
                    self.decision_marker_pub,
                    enabled=self.args.publish_decision_viz,
                ),
            ),
        ]
        if any(reason is None for _name, reason in reasons):
            return None
        unique_reasons = {reason for _name, reason in reasons}
        if len(unique_reasons) == 1:
            return unique_reasons.pop()
        return ",".join(f"{name}={reason}" for name, reason in reasons)

    def current_grid_cache_key(self):
        return (
            self.accepted_scan_revision,
            self.odom_revision,
            self.latest_odom_received_sec,
            pose_cache_signature(self.latest_odom_pose),
            len(self.scan_samples),
            config_cache_signature(self.config),
        )

    def cached_debug_grids(self, need_display_grid=True, need_planning_grid=True):
        if not need_display_grid and not need_planning_grid:
            return DebugGridPair()
        key = self.current_grid_cache_key()
        if key != self.grid_cache_key:
            self.grid_cache_key = key
            self.cached_display_grid = None
            self.cached_planning_grid = None
        missing_display = need_display_grid and self.cached_display_grid is None
        missing_planning = need_planning_grid and self.cached_planning_grid is None
        if missing_display or missing_planning:
            grids = build_debug_grids(
                self.scan_samples,
                self.latest_odom_pose,
                self.config,
                need_display_grid=missing_display,
                need_planning_grid=missing_planning,
            )
            if missing_display:
                self.cached_display_grid = grids.display_grid
            if missing_planning:
                self.cached_planning_grid = grids.planning_grid
        return DebugGridPair(
            self.cached_display_grid if need_display_grid else None,
            self.cached_planning_grid if need_planning_grid else None,
        )

    def ready_for_publish(self):
        if not self.scan_samples or self.latest_odom_pose is None:
            self.log_status_if_due("waiting_for_scan_and_odom")
            return False
        return True

    def map_timer_callback(self):
        if not self.ready_for_publish():
            return
        observed_reason = self.output_skip_reason(
            self.map_pub,
            enabled=self.args.publish_observed_map_viz,
        )
        planning_reason = self.output_skip_reason(
            self.planning_map_pub,
            enabled=self.args.publish_planning_map_viz,
        )
        need_display_grid = observed_reason is None
        need_planning_grid = planning_reason is None
        if not need_display_grid and not need_planning_grid:
            self.log_status_if_due(
                "skipped maps "
                f"samples={len(self.scan_samples)} "
                f"observed={observed_reason} "
                f"planning={planning_reason} "
                f"rejected_scans={self.rejected_scan_count}"
            )
            return
        try:
            grids = self.cached_debug_grids(
                need_display_grid=need_display_grid,
                need_planning_grid=need_planning_grid,
            )
            stamp = self.get_clock().now().to_msg()
            if grids.display_grid is not None and self.map_pub is not None:
                msg = build_occupancy_grid_message(
                    grids.display_grid,
                    self.args.map_frame,
                    stamp,
                )
                self.map_pub.publish(msg)
            if grids.planning_grid is not None and self.planning_map_pub is not None:
                planning_msg = build_occupancy_grid_message(
                    grids.planning_grid,
                    self.args.map_frame,
                    stamp,
                )
                self.planning_map_pub.publish(planning_msg)
        except Exception as exc:
            self.get_logger().warn(f"Could not build temporary map: {exc}")
            return
        status = ["published maps", f"samples={len(self.scan_samples)}"]
        if grids.display_grid is not None:
            display_counts = grid_cell_counts(grids.display_grid)
            status.extend(
                [
                    f"observed_free={display_counts['free']}",
                    f"observed_occupied={display_counts['occupied']}",
                    f"observed_unknown={display_counts['unknown']}",
                ]
            )
        else:
            status.append(f"observed={observed_reason}")
        if grids.planning_grid is not None:
            planning_counts = grid_cell_counts(grids.planning_grid)
            status.extend(
                [
                    f"planning_free={planning_counts['free']}",
                    f"planning_occupied={planning_counts['occupied']}",
                    f"planning_inflated={planning_counts['inflated']}",
                    f"planning_unknown={planning_counts['unknown']}",
                ]
            )
        else:
            status.append(f"planning={planning_reason}")
        status.append(f"rejected_scans={self.rejected_scan_count}")
        self.log_status_if_due(" ".join(status))

    def path_timer_callback(self):
        if not self.ready_for_publish():
            return
        if self.latest_scan is None:
            self.log_status_if_due("waiting_for_latest_scan")
            return
        path_reason = self.path_skip_reason()
        if path_reason is not None:
            self.log_status_if_due(
                "skipped path "
                f"samples={len(self.scan_samples)} "
                f"path={path_reason} "
                f"rejected_scans={self.rejected_scan_count}"
            )
            return
        try:
            grids = self.cached_debug_grids(
                need_display_grid=False,
                need_planning_grid=True,
            )
            plan = build_debug_active_explore_plan(
                self.latest_scan,
                self.latest_odom_pose,
                grids.planning_grid,
                self.config,
            )
            points = ()
            if plan.selected is not None:
                points = executable_path_points(
                    plan.selected,
                    self.latest_odom_pose,
                    self.args.max_single_move_m,
                )
            stamp = self.get_clock().now().to_msg()
            publish_path = self.publisher_should_publish(self.path_pub)
            publish_markers = self.publisher_should_publish(self.marker_pub)
            publish_decisions = self.publisher_should_publish(self.decision_marker_pub)
            path_msg = (
                build_path_message(points, self.args.map_frame, stamp)
                if publish_path
                else None
            )
            marker_msg = (
                build_candidate_marker_array(
                    plan,
                    self.latest_odom_pose,
                    self.args.max_single_move_m,
                    self.args.map_frame,
                    stamp,
                )
                if publish_markers
                else None
            )
            decision_marker_msg = (
                build_candidate_decision_marker_array(
                    plan,
                    self.args.map_frame,
                    stamp,
                )
                if publish_decisions
                else None
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not build active-explore path: {exc}")
            return
        if publish_path:
            self.path_pub.publish(path_msg)
        if publish_markers:
            self.marker_pub.publish(marker_msg)
        if publish_decisions:
            self.decision_marker_pub.publish(decision_marker_msg)
        self.log_status_if_due(
            "published path "
            f"samples={len(self.scan_samples)} "
            f"path={'published' if publish_path else 'skipped'} "
            f"markers={'published' if publish_markers else 'skipped'} "
            f"decisions={'published' if publish_decisions else 'skipped'} "
            f"rejected_scans={self.rejected_scan_count}"
            f" selected={None if plan.selected is None else plan.selected.kind}"
            f" plan={plan.reason}"
        )

    def timer_callback(self):
        self.map_timer_callback()
        self.path_timer_callback()

    def log_status_if_due(self, text):
        now = self.now_sec()
        if now - self.last_status_sec < self.args.status_period_sec:
            return
        self.last_status_sec = now
        self.get_logger().info(text)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Publish read-only RViz OccupancyGrids for the observed and "
            "planning arena-active temporary odom-frame maps."
        ),
    )
    parser.add_argument("--scan-topic", default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--map-topic", default=DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_TOPIC)
    parser.add_argument(
        "--planning-map-topic",
        default=DEFAULT_ARENA_ACTIVE_TEMPORARY_PLANNING_MAP_TOPIC,
    )
    parser.add_argument("--map-frame", default=DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_FRAME)
    parser.add_argument(
        "--no-observed-map-viz",
        dest="publish_observed_map_viz",
        action="store_false",
        help="Do not publish the observed temporary map topic.",
    )
    parser.add_argument(
        "--no-planning-map-viz",
        dest="publish_planning_map_viz",
        action="store_false",
        help="Do not publish the inflated planning map topic.",
    )
    parser.add_argument("--path-topic", default=DEFAULT_ARENA_ACTIVE_EXPLORE_PATH_TOPIC)
    parser.add_argument(
        "--candidate-marker-topic",
        default=DEFAULT_ARENA_ACTIVE_EXPLORE_CANDIDATE_MARKER_TOPIC,
    )
    parser.add_argument(
        "--decision-marker-topic",
        default=DEFAULT_ARENA_ACTIVE_EXPLORE_DECISION_MARKER_TOPIC,
    )
    parser.add_argument(
        "--no-path-viz",
        dest="publish_path_viz",
        action="store_false",
        help=(
            "Publish only the temporary maps; skip the preview path and "
            "candidate markers."
        ),
    )
    parser.add_argument(
        "--no-decision-viz",
        dest="publish_decision_viz",
        action="store_false",
        help="Do not publish the active-explore candidate decision overlay.",
    )
    parser.add_argument(
        "--no-subscriber-gating",
        dest="subscriber_gating",
        action="store_false",
        help="Compute and publish enabled outputs even when RViz has no subscribers.",
    )
    parser.set_defaults(
        publish_observed_map_viz=True,
        publish_planning_map_viz=True,
        publish_path_viz=True,
        publish_decision_viz=True,
        subscriber_gating=True,
    )
    parser.add_argument(
        "--publish-period-sec",
        default=DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_PUBLISH_PERIOD_SEC,
        type=float,
    )
    parser.add_argument("--map-publish-period-sec", type=float)
    parser.add_argument("--path-publish-period-sec", type=float)
    parser.add_argument("--status-period-sec", default=DEFAULT_STATUS_PERIOD_SEC, type=float)
    parser.add_argument(
        "--max-odom-scan-age-sec",
        default=DEFAULT_MAX_ODOM_SCAN_AGE_SEC,
        type=float,
    )
    parser.add_argument("--map-max-samples", default=DEFAULT_MAP_MAX_SAMPLES, type=int)
    parser.add_argument(
        "--max-single-move-m",
        default=arena_active_default("active_explore_max_single_move_m"),
        type=float,
        help="Maximum executable active-explore curve path length shown in RViz.",
    )
    parser.add_argument(
        "--max-total-distance-m",
        default=arena_active_default("active_explore_max_total_distance_m"),
        type=float,
    )
    parser.add_argument("--max-candidate-path-m", type=float)
    parser.add_argument(
        "--unknown-blocked",
        dest="unknown_blocked",
        action="store_true",
        default=arena_active_default("active_explore_unknown_blocked"),
    )
    parser.add_argument(
        "--allow-unknown",
        dest="unknown_blocked",
        action="store_false",
    )
    parser.add_argument(
        "--grid-resolution-m",
        default=arena_active_default("active_explore_grid_resolution_m"),
        type=float,
    )
    parser.add_argument(
        "--grid-size-m",
        default=arena_active_default("active_explore_grid_size_m"),
        type=float,
    )
    parser.add_argument(
        "--inflation-radius-m",
        default=arena_active_default("active_explore_inflation_radius_m"),
        type=float,
    )
    parser.add_argument(
        "--soft-clearance-radius-m",
        default=arena_active_default("active_explore_soft_clearance_radius_m"),
        type=float,
    )
    parser.add_argument(
        "--soft-clearance-weight",
        default=arena_active_default("active_explore_soft_clearance_weight"),
        type=float,
    )
    return parser


def validate_args(parser, args):
    if args.map_publish_period_sec is None:
        args.map_publish_period_sec = args.publish_period_sec
    if args.path_publish_period_sec is None:
        args.path_publish_period_sec = args.publish_period_sec
    for field in [
        "publish_period_sec",
        "map_publish_period_sec",
        "path_publish_period_sec",
        "status_period_sec",
        "max_odom_scan_age_sec",
        "max_single_move_m",
        "max_total_distance_m",
        "grid_resolution_m",
        "grid_size_m",
        "inflation_radius_m",
    ]:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.soft_clearance_radius_m < 0.0:
        parser.error("--soft-clearance-radius-m must be >= 0")
    if args.soft_clearance_weight < 0.0:
        parser.error("--soft-clearance-weight must be >= 0")
    if args.max_candidate_path_m is not None and args.max_candidate_path_m <= 0.0:
        parser.error("--max-candidate-path-m must be greater than zero")
    if args.map_max_samples < 1:
        parser.error("--map-max-samples must be >= 1")
    if not args.map_topic:
        parser.error("--map-topic must not be empty")
    if not args.planning_map_topic:
        parser.error("--planning-map-topic must not be empty")
    if args.planning_map_topic == args.map_topic:
        parser.error("--planning-map-topic must differ from --map-topic")
    if not args.map_frame:
        parser.error("--map-frame must not be empty")
    for field in [
        "scan_topic",
        "odom_topic",
        "map_topic",
        "map_frame",
        "path_topic",
        "candidate_marker_topic",
        "decision_marker_topic",
    ]:
        if not getattr(args, field):
            parser.error(f"--{field.replace('_', '-')} must not be empty")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if rclpy is None:
        raise SystemExit("ROS2 Python packages are required to run this RViz publisher.")

    rclpy.init(args=None)
    node = ArenaActiveTemporaryMapDebugViz(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
