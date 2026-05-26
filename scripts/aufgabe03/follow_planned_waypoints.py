#!/usr/bin/env python3
"""
Follow simplified A* waypoints with a conservative TF-based controller.

This script executes a static waypoint CSV in the map frame. It assumes Nav2
localization/AMCL is already running, but it publishes /cmd_vel itself.
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.time import Time
    from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import Path as NavPath
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
    import tf2_ros
except ImportError:
    rclpy = None
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    qos_profile_sensor_data = None
    Time = None
    Point = None
    PoseStamped = None
    PoseWithCovarianceStamped = object
    NavPath = None
    Twist = None
    LaserScan = object
    Marker = None
    MarkerArray = None
    tf2_ros = None

import lidar_obstacle_map
import replan_runtime


DEFAULT_WAYPOINTS_CSV = Path("results/aufgabe03/aufgabe03_waypoints.csv")
DEFAULT_RESULTS_CSV = Path("results/aufgabe03/aufgabe03_waypoint_follow_runs.csv")
DEFAULT_STATIC_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_REPLAN_OUTPUT_DIR = Path("results/aufgabe03")
DEFAULT_RVIZ_PATH_TOPIC = "/mii_amr/planned_path"
DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC = "/mii_amr/planned_waypoints"
DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC = "/mii_amr/run_local_obstacles"

DEFAULT_LINEAR_SPEED_MPS = 0.03
DEFAULT_MIN_LINEAR_SPEED_MPS = 0.01
DEFAULT_LINEAR_GAIN = 0.25
DEFAULT_MAX_ANGULAR_SPEED_RADPS = 0.12
DEFAULT_YAW_GAIN = 0.5
DEFAULT_WAYPOINT_TOLERANCE_M = 0.12
DEFAULT_GOAL_TOLERANCE_M = 0.12
DEFAULT_ROTATE_START_HEADING_ERROR_DEG = 20.0
DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG = 4.0
DEFAULT_FORWARD_YAW_DEADBAND_DEG = 4.0
DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG = 18.0
DEFAULT_MIN_WAYPOINT_SPACING_M = 0.12
DEFAULT_START_SELECTION = "path-progress"
DEFAULT_START_ON_PATH_TOLERANCE_M = 0.25
DEFAULT_SCAN_HALF_ANGLE_DEG = 35.0
DEFAULT_HARD_STOP_RANGE_M = 0.16
DEFAULT_MIN_SCAN_RANGE_M = 0.40
DEFAULT_ROTATION_STOP_RANGE_M = 0.18
FORWARD_SOFT_STOP_MIN_CLOSE_RANGES = 2
DEFAULT_MAX_POSE_AGE_SEC = 10.0
DEFAULT_MAX_SCAN_AGE_SEC = 8.0
DEFAULT_MAX_AMCL_AGE_SEC = 15.0
DEFAULT_MAX_AMCL_VAR_X = 0.05
DEFAULT_MAX_AMCL_VAR_Y = 0.05
DEFAULT_MAX_AMCL_VAR_YAW = 0.10
DEFAULT_MAX_WAYPOINT_TIME_SEC = 180.0
DEFAULT_MAX_TF_UPDATE_GAP_SEC = 5.0
DEFAULT_TF_RECOVERY_TIME_SEC = 5.0
DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC = 5.0
DEFAULT_CONTROL_RATE_HZ = 10.0
DEFAULT_SETTLE_SEC = 0.5
DEFAULT_REPLAN_TIMEOUT_SEC = 5.0
DEFAULT_MAX_REPLAN_SCAN_AGE_SEC = 1.0
DEFAULT_MAX_REPLAN_TF_AGE_SEC = 1.0
DEFAULT_OBSTACLE_FORWARD_DISTANCE_M = 0.75
DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M = 0.25
DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG = 45.0
DEFAULT_OBSTACLE_MIN_RANGE_M = 0.12
DEFAULT_ROBOT_FOOTPRINT_RADIUS_M = 0.18
DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE = 3
DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M = 0.05
DEFAULT_OBSTACLE_INFLATE_RADIUS_M = 0.22
DEFAULT_MAX_START_SNAP_M = 0.20
DEFAULT_MAX_GOAL_SNAP_M = 0.30
DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO = 3.0
DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE = "forward"
DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT = 5
DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE = "forward"
DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT = 2
DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M = DEFAULT_OBSTACLE_INFLATE_RADIUS_M
DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC = 1.0
DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC = 1.0
DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS = 3
DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO = 0.90
DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M = 0.75
DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M = 0.04
DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES = 3

INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS = {
    lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
    lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS,
}

REPLAN_TRIGGER_SCAN_BLOCKAGE = "scan_blockage"
REPLAN_TRIGGER_KNOWN_CORRIDOR = "known_corridor"


def run_local_map_has_confirmed_obstacles(run_local_map):
    if run_local_map is None:
        return False
    return bool(getattr(run_local_map, "confirmed_raw_cells", None))


def lidar_replan_failure(reason):
    message = str(reason)
    if message.startswith("lidar_replan_failed:"):
        return RuntimeError(message)
    return RuntimeError(f"lidar_replan_failed:{message}")

DEFAULT_STARTUP_TIMEOUT_SEC = 20.0
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0

RVIZ_COLOR_PATH = (0.0, 0.55, 1.0, 0.95)
RVIZ_COLOR_WAYPOINT = (0.0, 0.75, 1.0, 0.85)
RVIZ_COLOR_CURRENT = (1.0, 0.82, 0.16, 0.95)
RVIZ_COLOR_GOAL = (0.95, 0.18, 0.14, 0.95)
RVIZ_COLOR_LABEL = (0.95, 0.95, 0.95, 1.0)
RVIZ_COLOR_CONFIRMED_OBSTACLE = (0.05, 0.95, 0.22, 0.85)
RVIZ_COLOR_INFLATED_OBSTACLE = (0.9, 0.25, 1.0, 0.30)
RVIZ_COLOR_BLOCKED_CORRIDOR = (1.0, 0.45, 0.0, 0.80)

BASE_CSV_HEADER = [
    "timestamp",
    "run_id",
    "waypoint_csv",
    "waypoint_count",
    "reached_count",
    "status",
    "blocked_waypoint_index",
    "blocked_waypoint_x",
    "blocked_waypoint_y",
    "timeout_waypoint_index",
    "base_frame_used",
    "start_x",
    "start_y",
    "start_yaw_deg",
    "final_x",
    "final_y",
    "final_yaw_deg",
    "min_scan_range_m",
    "p05_scan_range_m",
    "amcl_var_x",
    "amcl_var_y",
    "amcl_var_yaw",
    "linear_speed_mps",
    "min_linear_speed_mps",
    "linear_gain",
    "max_angular_speed_radps",
    "yaw_gain",
    "notes",
]

CSV_HEADER = BASE_CSV_HEADER + [
    "selected_start_segment_index",
    "selected_start_waypoint_index",
    "distance_to_path_m",
    "tf_pose_age_sec",
    "max_tf_update_gap_sec",
    "tf_stale_warning_count",
    "localization_warning_count",
    "recovery_pause_count",
    "max_abs_yaw_error_deg",
    "mean_abs_yaw_error_deg",
    "rotate_seconds",
    "forward_seconds",
    "final_status_reason",
    "replan_count",
    "last_replan_reason",
    "updated_map_yaml",
    "updated_waypoints_csv",
    "detected_obstacle_count",
    "candidate_scan_points",
    "filtered_obstacle_points",
    "raw_obstacle_cells",
    "free_obstacle_cells",
    "inflated_cells_total",
    "inflated_cells_newly_occupied",
    "inflated_cells_over_static_occupied",
    "scan_frame",
    "scan_age_sec",
    "tf_age_sec",
    "tf_lookup_mode",
    "start_snap_distance_m",
    "goal_snap_distance_m",
    "old_remaining_waypoint_count",
    "new_waypoint_count",
    "old_path_length_m",
    "new_path_length_m",
    "replan_duration_sec",
    "run_local_map_updates",
    "run_local_replan_count",
    "run_local_last_replan_reason",
    "run_local_no_path_reason",
    "run_local_start_cell_blocked",
    "run_local_goal_cell_blocked",
    "run_local_path_blocked_cell_count",
    "run_local_scan_points_valid",
    "run_local_scan_points_used",
    "run_local_scan_points_rejected_invalid_range",
    "run_local_scan_points_rejected_static",
    "run_local_scan_points_rejected_bounds",
    "run_local_scan_points_rejected_wall_band",
    "run_local_scan_points_rejected_low_confidence",
    "run_local_update_rejected_reason",
    "run_local_initial_scan_count",
    "run_local_corridor_check_distance_m",
    "run_local_inflation_radius_m",
    "run_local_map_yaml",
    "run_local_waypoints_csv",
    "run_local_cell_source_counts",
]


@dataclass(frozen=True)
class Waypoint:
    index: int
    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_deg: float
    stamp_sec: float | None = None
    frame_id: str = ""


@dataclass(frozen=True)
class TargetState:
    distance_m: float
    heading_deg: float
    yaw_error_deg: float


@dataclass(frozen=True)
class ScanSafety:
    safe: bool
    reason: str
    valid_count: int
    min_range_m: float | None
    percentile_5_m: float | None


@dataclass(frozen=True)
class AmclHealth:
    ok: bool
    warnings: list[str]
    cov_x: float | None
    cov_y: float | None
    cov_yaw: float | None
    age_sec: float | None


@dataclass(frozen=True)
class StartSelection:
    waypoints: list[Waypoint]
    selected_segment_index: int | None
    selected_waypoint_index: int | None
    distance_to_path_m: float | None


@dataclass
class RuntimeDiagnostics:
    selected_start_segment_index: int | None = None
    selected_start_waypoint_index: int | None = None
    distance_to_path_m: float | None = None
    tf_pose_age_sec: float | None = None
    max_tf_update_gap_sec: float | None = None
    tf_stale_warning_count: int = 0
    localization_warning_count: int = 0
    recovery_pause_count: int = 0
    max_abs_yaw_error_deg: float = 0.0
    yaw_error_sum_deg: float = 0.0
    yaw_error_count: int = 0
    rotate_seconds: float = 0.0
    forward_seconds: float = 0.0
    final_status_reason: str = ""
    replan_count: int = 0
    last_replan_reason: str = ""
    updated_map_yaml: str = ""
    updated_waypoints_csv: str = ""
    detected_obstacle_count: int = 0
    candidate_scan_points: int = 0
    filtered_obstacle_points: int = 0
    raw_obstacle_cells: int = 0
    free_obstacle_cells: int = 0
    inflated_cells_total: int = 0
    inflated_cells_newly_occupied: int = 0
    inflated_cells_over_static_occupied: int = 0
    scan_frame: str = ""
    scan_age_sec: float | None = None
    tf_age_sec: float | None = None
    tf_lookup_mode: str = ""
    start_snap_distance_m: float | None = None
    goal_snap_distance_m: float | None = None
    old_remaining_waypoint_count: int = 0
    new_waypoint_count: int = 0
    old_path_length_m: float | None = None
    new_path_length_m: float | None = None
    replan_duration_sec: float | None = None
    run_local_map_updates: int = 0
    run_local_replan_count: int = 0
    run_local_last_replan_reason: str = ""
    run_local_no_path_reason: str = ""
    run_local_start_cell_blocked: bool = False
    run_local_goal_cell_blocked: bool = False
    run_local_path_blocked_cell_count: int = 0
    run_local_scan_points_valid: int = 0
    run_local_scan_points_used: int = 0
    run_local_scan_points_rejected_invalid_range: int = 0
    run_local_scan_points_rejected_static: int = 0
    run_local_scan_points_rejected_bounds: int = 0
    run_local_scan_points_rejected_wall_band: int = 0
    run_local_scan_points_rejected_low_confidence: int = 0
    run_local_update_rejected_reason: str = ""
    run_local_initial_scan_count: int = 0
    run_local_corridor_check_distance_m: float | None = None
    run_local_inflation_radius_m: float | None = None
    run_local_map_yaml: str = ""
    run_local_waypoints_csv: str = ""
    run_local_cell_source_counts: dict[str, int] | str = ""

    @property
    def mean_abs_yaw_error_deg(self):
        if self.yaw_error_count == 0:
            return 0.0
        return self.yaw_error_sum_deg / self.yaw_error_count


def rviz_messages_available():
    return all(
        message_type is not None
        for message_type in (NavPath, PoseStamped, Point, Marker, MarkerArray)
    )


def rviz_qos_profile():
    if QoSProfile is None or DurabilityPolicy is None:
        return 1
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def set_header(message, frame_id, stamp):
    message.header.frame_id = frame_id
    message.header.stamp = stamp


def set_pose_xy(pose, x, y, z=0.0):
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0


def point_msg(x, y, z=0.0):
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def set_marker_color(marker, color):
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]


def marker_delete_all(frame_id, stamp):
    marker = Marker()
    set_header(marker, frame_id, stamp)
    marker.action = Marker.DELETEALL
    return marker


def apply_marker_common(marker, frame_id, stamp, namespace, marker_id, marker_type, color):
    set_header(marker, frame_id, stamp)
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    set_marker_color(marker, color)


def build_pose_stamped(frame_id, stamp, x, y):
    pose = PoseStamped()
    set_header(pose, frame_id, stamp)
    set_pose_xy(pose.pose, x, y)
    return pose


def build_rviz_path_message(waypoints, frame_id, stamp, current_pose=None):
    if NavPath is None or PoseStamped is None:
        raise RuntimeError("ROS nav_msgs/geometry_msgs are unavailable.")
    path = NavPath()
    set_header(path, frame_id, stamp)
    if current_pose is not None:
        path.poses.append(build_pose_stamped(frame_id, stamp, current_pose.x, current_pose.y))
    for waypoint in waypoints:
        path.poses.append(build_pose_stamped(frame_id, stamp, waypoint.x, waypoint.y))
    return path


def waypoint_point(waypoint, z=0.04):
    return point_msg(waypoint.x, waypoint.y, z)


def build_point_layer_marker(
    frame_id,
    stamp,
    namespace,
    marker_id,
    marker_type,
    points,
    color,
    scale_m,
):
    if not points:
        return None
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, marker_type, color)
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    marker.points = list(points)
    return marker


def build_single_waypoint_marker(frame_id, stamp, namespace, marker_id, waypoint, color, scale_m, z):
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, Marker.SPHERE, color)
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    set_pose_xy(marker.pose, waypoint.x, waypoint.y, z)
    return marker


def build_waypoint_label_marker(frame_id, stamp, marker_id, waypoint):
    marker = Marker()
    apply_marker_common(
        marker,
        frame_id,
        stamp,
        "planned_waypoint_labels",
        marker_id,
        Marker.TEXT_VIEW_FACING,
        RVIZ_COLOR_LABEL,
    )
    set_pose_xy(marker.pose, waypoint.x, waypoint.y, 0.20)
    marker.scale.z = 0.08
    marker.text = str(waypoint.index)
    return marker


def build_rviz_waypoint_markers(waypoints, frame_id, stamp, current_waypoint_index=0):
    if Marker is None or MarkerArray is None or Point is None:
        raise RuntimeError("ROS visualization messages are unavailable.")
    waypoints = list(waypoints)
    markers = [marker_delete_all(frame_id, stamp)]
    points = [waypoint_point(waypoint) for waypoint in waypoints]
    waypoint_layer = build_point_layer_marker(
        frame_id,
        stamp,
        "planned_waypoints",
        1,
        Marker.SPHERE_LIST,
        points,
        RVIZ_COLOR_WAYPOINT,
        0.07,
    )
    if waypoint_layer is not None:
        markers.append(waypoint_layer)
    if waypoints:
        current_index = max(0, min(int(current_waypoint_index), len(waypoints) - 1))
        markers.append(
            build_single_waypoint_marker(
                frame_id,
                stamp,
                "current_waypoint",
                2,
                waypoints[current_index],
                RVIZ_COLOR_CURRENT,
                0.14,
                0.08,
            )
        )
        markers.append(
            build_single_waypoint_marker(
                frame_id,
                stamp,
                "goal_waypoint",
                3,
                waypoints[-1],
                RVIZ_COLOR_GOAL,
                0.12,
                0.10,
            )
        )
        for label_index, waypoint in enumerate(waypoints):
            markers.append(
                build_waypoint_label_marker(
                    frame_id,
                    stamp,
                    1000 + label_index,
                    waypoint,
                )
            )
    return MarkerArray(markers=markers)


def build_cell_layer_marker(
    run_local_map,
    frame_id,
    stamp,
    namespace,
    marker_id,
    cells,
    color,
    z,
    height_m,
):
    cells = sorted(cells or ())
    if not cells:
        return None
    metadata = run_local_map.static_map.metadata
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, Marker.CUBE_LIST, color)
    marker.scale.x = metadata.resolution
    marker.scale.y = metadata.resolution
    marker.scale.z = height_m
    marker.points = [
        point_msg(
            *lidar_obstacle_map.planner.grid_to_world(cell[0], cell[1], metadata),
            z,
        )
        for cell in cells
    ]
    return marker


def append_marker(markers, marker):
    if marker is not None:
        markers.append(marker)


def build_rviz_obstacle_markers(run_local_map, frame_id, stamp, blocked_cells=None):
    if Marker is None or MarkerArray is None or Point is None:
        raise RuntimeError("ROS visualization messages are unavailable.")
    markers = [marker_delete_all(frame_id, stamp)]
    if run_local_map is None:
        return MarkerArray(markers=markers)
    append_marker(
        markers,
        build_cell_layer_marker(
            run_local_map,
            frame_id,
            stamp,
            "run_local_inflated_obstacle_cells",
            1,
            run_local_map.inflated_obstacle_cells,
            RVIZ_COLOR_INFLATED_OBSTACLE,
            0.005,
            0.02,
        ),
    )
    append_marker(
        markers,
        build_cell_layer_marker(
            run_local_map,
            frame_id,
            stamp,
            "run_local_confirmed_obstacle_cells",
            2,
            run_local_map.confirmed_raw_cells,
            RVIZ_COLOR_CONFIRMED_OBSTACLE,
            0.045,
            0.05,
        ),
    )
    append_marker(
        markers,
        build_cell_layer_marker(
            run_local_map,
            frame_id,
            stamp,
            "run_local_blocked_corridor_cells",
            3,
            blocked_cells or set(),
            RVIZ_COLOR_BLOCKED_CORRIDOR,
            0.075,
            0.06,
        ),
    )
    return MarkerArray(markers=markers)


def publish_rviz_route_if_available(
    node,
    waypoints,
    current_pose=None,
    current_waypoint_index=0,
):
    publish = getattr(node, "publish_rviz_route", None)
    if callable(publish):
        publish(
            waypoints,
            current_pose=current_pose,
            current_waypoint_index=current_waypoint_index,
        )


def publish_rviz_obstacles_if_available(node, blocked_cells=None):
    publish = getattr(node, "publish_rviz_obstacles", None)
    if callable(publish):
        publish(blocked_cells=blocked_cells)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw_deg(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def waypoint_distance(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)


def heading_between(a, b):
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def target_state(current_pose, waypoint):
    dx = waypoint.x - current_pose.x
    dy = waypoint.y - current_pose.y
    heading = math.degrees(math.atan2(dy, dx))
    return TargetState(
        distance_m=math.hypot(dx, dy),
        heading_deg=heading,
        yaw_error_deg=shortest_angle_delta_deg(current_pose.yaw_deg, heading),
    )


def waypoint_reached(distance_m, is_final, waypoint_tolerance_m, goal_tolerance_m):
    tolerance = goal_tolerance_m if is_final else waypoint_tolerance_m
    return distance_m <= tolerance


def should_rotate(current_mode, yaw_error_deg, start_threshold_deg, stop_threshold_deg):
    abs_error = abs(yaw_error_deg)
    if current_mode == "rotate":
        return abs_error > stop_threshold_deg
    return abs_error > start_threshold_deg


def velocity_command(
    distance_m,
    yaw_error_deg,
    rotate_mode,
    linear_speed_mps,
    min_linear_speed_mps,
    linear_gain,
    max_angular_speed_radps,
    yaw_gain,
    forward_yaw_deadband_deg=0.0,
    forward_stop_heading_error_deg=180.0,
):
    angular_z = clamp(
        math.radians(yaw_error_deg) * yaw_gain,
        -max_angular_speed_radps,
        max_angular_speed_radps,
    )
    abs_yaw_error = abs(yaw_error_deg)
    if rotate_mode or abs_yaw_error >= forward_stop_heading_error_deg:
        return 0.0, angular_z

    linear_x = clamp(
        distance_m * linear_gain,
        min_linear_speed_mps,
        linear_speed_mps,
    )
    if abs_yaw_error <= forward_yaw_deadband_deg:
        return linear_x, 0.0

    scale_span = forward_stop_heading_error_deg - forward_yaw_deadband_deg
    heading_scale = 1.0
    if scale_span > 0.0:
        heading_scale = 1.0 - (abs_yaw_error - forward_yaw_deadband_deg) / scale_span
    heading_scale = clamp(heading_scale, 0.0, 1.0)
    linear_x *= heading_scale
    if linear_x > 0.0:
        linear_x = max(min_linear_speed_mps, linear_x)
    return linear_x, angular_z


def load_waypoints(path):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        required = {"index", "world_x_m", "world_y_m"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )

        waypoints = []
        previous_xy = None
        for row in reader:
            waypoint = Waypoint(
                index=int(float(row["index"])),
                x=float(row["world_x_m"]),
                y=float(row["world_y_m"]),
            )
            xy = (waypoint.x, waypoint.y)
            if previous_xy is not None and xy == previous_xy:
                continue
            previous_xy = xy
            waypoints.append(waypoint)

    if not waypoints:
        raise ValueError(f"{path} does not contain any waypoints")
    return waypoints


def is_heading_change(previous_wp, current_wp, next_wp, tolerance_deg=1.0):
    incoming = heading_between(previous_wp, current_wp)
    outgoing = heading_between(current_wp, next_wp)
    return abs(shortest_angle_delta_deg(incoming, outgoing)) > tolerance_deg


def downsample_waypoints(waypoints, min_spacing_m):
    if len(waypoints) <= 2:
        return list(waypoints)

    selected = [waypoints[0]]
    for index in range(1, len(waypoints) - 1):
        current = waypoints[index]
        if is_heading_change(waypoints[index - 1], current, waypoints[index + 1]):
            selected.append(current)
            continue
        if waypoint_distance(selected[-1], current) >= min_spacing_m:
            selected.append(current)

    if selected[-1] != waypoints[-1]:
        selected.append(waypoints[-1])
    return selected


def prepare_executable_waypoints(waypoints, skip_first=True, min_spacing_m=0.0):
    executable = list(waypoints[1:] if skip_first else waypoints)
    if min_spacing_m > 0.0:
        executable = downsample_waypoints(executable, min_spacing_m)
    if len(executable) < 2:
        raise ValueError(
            "Waypoint CSV needs at least two executable waypoints after processing"
        )
    return executable


def distance_point_to_segment_m(point, segment_start, segment_end):
    dx = segment_end.x - segment_start.x
    dy = segment_end.y - segment_start.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(point.x - segment_start.x, point.y - segment_start.y), 0.0
    projection = (
        (point.x - segment_start.x) * dx + (point.y - segment_start.y) * dy
    ) / length_sq
    projection = clamp(projection, 0.0, 1.0)
    closest_x = segment_start.x + projection * dx
    closest_y = segment_start.y + projection * dy
    return math.hypot(point.x - closest_x, point.y - closest_y), projection


def nearest_path_segment(point, waypoints):
    if len(waypoints) < 2:
        raise ValueError("Need at least two waypoints for path-progress selection")

    best = None
    for segment_index in range(len(waypoints) - 1):
        distance_m, projection = distance_point_to_segment_m(
            point,
            waypoints[segment_index],
            waypoints[segment_index + 1],
        )
        candidate = (distance_m, segment_index, projection)
        if best is None or candidate < best:
            best = candidate
    return best


def select_path_progress_waypoints(
    waypoints,
    current_pose,
    start_on_path_tolerance_m,
    waypoint_tolerance_m,
    goal_tolerance_m,
    min_spacing_m=0.0,
):
    distance_to_path_m, segment_index, _projection = nearest_path_segment(
        current_pose,
        waypoints,
    )
    if distance_to_path_m > start_on_path_tolerance_m:
        raise ValueError(
            "Current pose is too far from the planned path: "
            f"distance={distance_to_path_m:.3f} m, "
            f"tolerance={start_on_path_tolerance_m:.3f} m"
        )

    next_index = min(segment_index + 1, len(waypoints) - 1)
    while next_index < len(waypoints) - 1:
        waypoint = waypoints[next_index]
        distance_m = math.hypot(waypoint.x - current_pose.x, waypoint.y - current_pose.y)
        if not waypoint_reached(
            distance_m,
            is_final=False,
            waypoint_tolerance_m=waypoint_tolerance_m,
            goal_tolerance_m=goal_tolerance_m,
        ):
            break
        next_index += 1

    selected = list(waypoints[next_index:])
    if min_spacing_m > 0.0 and len(selected) > 1:
        selected = downsample_waypoints(selected, min_spacing_m)
    if not selected:
        selected = [waypoints[-1]]

    return StartSelection(
        waypoints=selected,
        selected_segment_index=segment_index,
        selected_waypoint_index=selected[0].index,
        distance_to_path_m=distance_to_path_m,
    )


def select_executable_waypoints(
    waypoints,
    current_pose,
    start_selection,
    start_on_path_tolerance_m,
    waypoint_tolerance_m,
    goal_tolerance_m,
    min_spacing_m,
    skip_first=True,
):
    if start_selection == "fixed-skip":
        selected = prepare_executable_waypoints(
            waypoints,
            skip_first=skip_first,
            min_spacing_m=min_spacing_m,
        )
        return StartSelection(
            waypoints=selected,
            selected_segment_index=None,
            selected_waypoint_index=selected[0].index,
            distance_to_path_m=None,
        )
    if start_selection == "path-progress":
        return select_path_progress_waypoints(
            waypoints,
            current_pose,
            start_on_path_tolerance_m,
            waypoint_tolerance_m,
            goal_tolerance_m,
            min_spacing_m=min_spacing_m,
        )
    raise ValueError(f"unsupported start selection mode: {start_selection!r}")


def percentile(values, percent):
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def valid_scan_ranges(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    sector_half_angle_deg=None,
):
    selected = []
    half_angle_rad = (
        math.radians(sector_half_angle_deg)
        if sector_half_angle_deg is not None
        else None
    )
    for index, raw_range in enumerate(ranges):
        if not math.isfinite(raw_range):
            continue
        if raw_range < range_min or raw_range > range_max:
            continue
        if half_angle_rad is not None:
            angle = normalize_angle_rad(angle_min + index * angle_increment)
            if abs(angle) > half_angle_rad:
                continue
        selected.append(float(raw_range))
    return selected


def evaluate_scan_safety(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    mode,
    scan_half_angle_deg,
    hard_stop_range_m,
    min_scan_range_m,
    rotation_stop_range_m,
):
    if mode not in {"forward", "rotate"}:
        raise ValueError(f"unsupported scan mode: {mode!r}")

    sector = scan_half_angle_deg if mode == "forward" else None
    selected = valid_scan_ranges(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        sector_half_angle_deg=sector,
    )
    if not selected:
        return ScanSafety(False, "no_valid_scan_ranges", 0, None, None)

    min_range = min(selected)
    percentile_5 = percentile(selected, 5.0)
    soft_threshold = min_scan_range_m if mode == "forward" else rotation_stop_range_m

    if min_range < hard_stop_range_m:
        return ScanSafety(False, "hard_stop", len(selected), min_range, percentile_5)
    if mode == "forward":
        close_count = sum(1 for value in selected if value < min_scan_range_m)
        if close_count >= FORWARD_SOFT_STOP_MIN_CLOSE_RANGES:
            return ScanSafety(False, "soft_stop", len(selected), min_range, percentile_5)
    if percentile_5 < soft_threshold:
        return ScanSafety(False, "soft_stop", len(selected), min_range, percentile_5)
    return ScanSafety(True, "clear", len(selected), min_range, percentile_5)


def amcl_covariances(covariance):
    return float(covariance[0]), float(covariance[7]), float(covariance[35])


def evaluate_amcl_health(
    covariance,
    age_sec,
    max_age_sec,
    max_var_x,
    max_var_y,
    max_var_yaw,
    fail_on_bad_localization=False,
):
    cov_x, cov_y, cov_yaw = amcl_covariances(covariance)
    warnings = []
    if age_sec is None or age_sec > max_age_sec:
        warnings.append("stale_amcl")
    if cov_x > max_var_x:
        warnings.append("high_cov_x")
    if cov_y > max_var_y:
        warnings.append("high_cov_y")
    if cov_yaw > max_var_yaw:
        warnings.append("high_cov_yaw")
    return AmclHealth(
        ok=not warnings or not fail_on_bad_localization,
        warnings=warnings,
        cov_x=cov_x,
        cov_y=cov_y,
        cov_yaw=cov_yaw,
        age_sec=age_sec,
    )


def age_ok(age_sec, max_age_sec):
    return age_sec is not None and age_sec <= max_age_sec


def ordered_base_frames(base_frame, fallback_base_frame):
    frames = []
    for frame in [base_frame, fallback_base_frame]:
        if frame and frame not in frames:
            frames.append(frame)
    return frames


def build_log_row(
    args,
    waypoint_count,
    reached_count,
    status,
    notes,
    start_pose=None,
    final_pose=None,
    blocked_waypoint=None,
    timeout_waypoint=None,
    base_frame_used="",
    scan_safety=None,
    amcl_health=None,
    diagnostics=None,
):
    diagnostics = diagnostics or RuntimeDiagnostics()
    blocked = blocked_waypoint or Waypoint("", "", "")
    timeout = timeout_waypoint or Waypoint("", "", "")
    return [
        datetime.now().isoformat(timespec="seconds"),
        args.run_id,
        str(args.waypoints),
        waypoint_count,
        reached_count,
        status,
        blocked.index,
        blocked.x,
        blocked.y,
        timeout.index,
        base_frame_used,
        *(pose_fields(start_pose)),
        *(pose_fields(final_pose)),
        "" if scan_safety is None or scan_safety.min_range_m is None else scan_safety.min_range_m,
        "" if scan_safety is None or scan_safety.percentile_5_m is None else scan_safety.percentile_5_m,
        "" if amcl_health is None or amcl_health.cov_x is None else amcl_health.cov_x,
        "" if amcl_health is None or amcl_health.cov_y is None else amcl_health.cov_y,
        "" if amcl_health is None or amcl_health.cov_yaw is None else amcl_health.cov_yaw,
        args.linear_speed,
        args.min_linear_speed,
        args.linear_gain,
        args.max_angular_speed,
        args.yaw_gain,
        notes,
        "" if diagnostics.selected_start_segment_index is None else diagnostics.selected_start_segment_index,
        "" if diagnostics.selected_start_waypoint_index is None else diagnostics.selected_start_waypoint_index,
        "" if diagnostics.distance_to_path_m is None else diagnostics.distance_to_path_m,
        "" if diagnostics.tf_pose_age_sec is None else diagnostics.tf_pose_age_sec,
        "" if diagnostics.max_tf_update_gap_sec is None else diagnostics.max_tf_update_gap_sec,
        diagnostics.tf_stale_warning_count,
        diagnostics.localization_warning_count,
        diagnostics.recovery_pause_count,
        diagnostics.max_abs_yaw_error_deg,
        diagnostics.mean_abs_yaw_error_deg,
        diagnostics.rotate_seconds,
        diagnostics.forward_seconds,
        diagnostics.final_status_reason,
        diagnostics.replan_count,
        diagnostics.last_replan_reason,
        diagnostics.updated_map_yaml,
        diagnostics.updated_waypoints_csv,
        diagnostics.detected_obstacle_count,
        diagnostics.candidate_scan_points,
        diagnostics.filtered_obstacle_points,
        diagnostics.raw_obstacle_cells,
        diagnostics.free_obstacle_cells,
        diagnostics.inflated_cells_total,
        diagnostics.inflated_cells_newly_occupied,
        diagnostics.inflated_cells_over_static_occupied,
        diagnostics.scan_frame,
        "" if diagnostics.scan_age_sec is None else diagnostics.scan_age_sec,
        "" if diagnostics.tf_age_sec is None else diagnostics.tf_age_sec,
        diagnostics.tf_lookup_mode,
        "" if diagnostics.start_snap_distance_m is None else diagnostics.start_snap_distance_m,
        "" if diagnostics.goal_snap_distance_m is None else diagnostics.goal_snap_distance_m,
        diagnostics.old_remaining_waypoint_count,
        diagnostics.new_waypoint_count,
        "" if diagnostics.old_path_length_m is None else diagnostics.old_path_length_m,
        "" if diagnostics.new_path_length_m is None else diagnostics.new_path_length_m,
        "" if diagnostics.replan_duration_sec is None else diagnostics.replan_duration_sec,
        diagnostics.run_local_map_updates,
        diagnostics.run_local_replan_count,
        diagnostics.run_local_last_replan_reason,
        diagnostics.run_local_no_path_reason,
        diagnostics.run_local_start_cell_blocked,
        diagnostics.run_local_goal_cell_blocked,
        diagnostics.run_local_path_blocked_cell_count,
        diagnostics.run_local_scan_points_valid,
        diagnostics.run_local_scan_points_used,
        diagnostics.run_local_scan_points_rejected_invalid_range,
        diagnostics.run_local_scan_points_rejected_static,
        diagnostics.run_local_scan_points_rejected_bounds,
        diagnostics.run_local_scan_points_rejected_wall_band,
        diagnostics.run_local_scan_points_rejected_low_confidence,
        diagnostics.run_local_update_rejected_reason,
        diagnostics.run_local_initial_scan_count,
        "" if diagnostics.run_local_corridor_check_distance_m is None else diagnostics.run_local_corridor_check_distance_m,
        "" if diagnostics.run_local_inflation_radius_m is None else diagnostics.run_local_inflation_radius_m,
        diagnostics.run_local_map_yaml,
        diagnostics.run_local_waypoints_csv,
        diagnostics.run_local_cell_source_counts,
    ]


def pose_fields(pose):
    if pose is None:
        return ["", "", ""]
    return [pose.x, pose.y, pose.yaw_deg]


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), None)
        if existing_header == header:
            pass
        elif existing_header and header[: len(existing_header)] == existing_header:
            migrate_csv_header(path, header)
        else:
            raise RuntimeError(
                f"{path} has an unrecognized schema. Move or migrate it first."
            )
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def migrate_csv_header(path, header):
    path = Path(path)
    with path.open(newline="") as file:
        rows = list(csv.reader(file))

    migrated = [header]
    for row in rows[1:]:
        migrated.append(row + [""] * (len(header) - len(row)))

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(migrated)


class WaypointFollower(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the waypoint follower."
            )
        super().__init__("follow_planned_waypoints")
        self.args = args
        self.last_scan = None
        self.last_scan_received_sec = None
        self.last_amcl = None
        self.last_amcl_received_sec = None
        self.base_frame_used = ""
        self.reached_count = 0
        self.start_pose = None
        self.final_pose = None
        self.last_scan_safety = None
        self.last_amcl_health = None
        self.diagnostics = RuntimeDiagnostics(
            max_tf_update_gap_sec=args.max_tf_update_gap_sec,
        )
        self.last_tf_stamp_sec = None
        self.last_tf_stamp_change_local_sec = None
        self.run_local_map = None
        self.live_replan_attempt_count = 0
        self.rviz_last_blocked_cells = set()

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.rviz_path_pub = None
        self.rviz_waypoint_marker_pub = None
        self.rviz_obstacle_marker_pub = None
        if not args.no_rviz_visualization:
            if not rviz_messages_available():
                raise RuntimeError(
                    "ROS RViz message types are unavailable. Source ROS 2 Humble "
                    "before enabling RViz visualization."
                )
            rviz_qos = rviz_qos_profile()
            self.rviz_path_pub = self.create_publisher(
                NavPath,
                args.rviz_path_topic,
                rviz_qos,
            )
            self.rviz_waypoint_marker_pub = self.create_publisher(
                MarkerArray,
                args.rviz_waypoint_marker_topic,
                rviz_qos,
            )
            self.rviz_obstacle_marker_pub = self.create_publisher(
                MarkerArray,
                args.rviz_obstacle_marker_topic,
                rviz_qos,
            )
            self.get_logger().info(
                "Publishing RViz visualization: "
                f"path={args.rviz_path_topic}, "
                f"waypoints={args.rviz_waypoint_marker_topic}, "
                f"obstacles={args.rviz_obstacle_marker_topic}"
            )
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            10,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        time.sleep(1.0)

    def rviz_visualization_enabled(self):
        return (
            not getattr(self.args, "no_rviz_visualization", False)
            and self.rviz_path_pub is not None
            and self.rviz_waypoint_marker_pub is not None
            and self.rviz_obstacle_marker_pub is not None
        )

    def rviz_stamp(self):
        return self.get_clock().now().to_msg()

    def publish_rviz_route(self, waypoints, current_pose=None, current_waypoint_index=0):
        if not self.rviz_visualization_enabled():
            return
        waypoints = list(waypoints)
        stamp = self.rviz_stamp()
        self.rviz_path_pub.publish(
            build_rviz_path_message(
                waypoints,
                self.args.map_frame,
                stamp,
                current_pose=current_pose,
            )
        )
        self.rviz_waypoint_marker_pub.publish(
            build_rviz_waypoint_markers(
                waypoints,
                self.args.map_frame,
                stamp,
                current_waypoint_index=current_waypoint_index,
            )
        )

    def publish_rviz_obstacles(self, blocked_cells=None):
        if not self.rviz_visualization_enabled():
            return
        if blocked_cells is not None:
            self.rviz_last_blocked_cells = set(blocked_cells)
        self.rviz_obstacle_marker_pub.publish(
            build_rviz_obstacle_markers(
                self.run_local_map,
                self.args.map_frame,
                self.rviz_stamp(),
                blocked_cells=self.rviz_last_blocked_cells,
            )
        )

    def scan_callback(self, msg):
        self.last_scan = msg
        self.last_scan_received_sec = time.time()

    def amcl_callback(self, msg):
        self.last_amcl = msg
        self.last_amcl_received_sec = time.time()

    def publish_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def stop_repeatedly(self):
        msg = Twist()
        sleep_sec = 1.0 / STOP_PUBLISH_HZ
        for _ in range(STOP_PUBLISH_COUNT):
            if rclpy.ok():
                self.pub.publish(msg)
            time.sleep(sleep_sec)

    def spin_once(self, timeout_sec):
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def wait_for_startup_gate(self, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = self.args.startup_timeout_sec
        require_amcl = (
            self.args.require_amcl_startup
            or self.args.fail_on_bad_localization
            or self.args.pause_on_bad_localization
        )
        start = time.time()
        while rclpy.ok():
            have_scan = self.last_scan is not None
            have_amcl = self.last_amcl is not None
            have_tf = False
            try:
                _pose, frame = self.lookup_pose()
                self.base_frame_used = frame
                have_tf = True
            except RuntimeError:
                have_tf = False

            if have_scan and have_tf and (have_amcl or not require_amcl):
                return
            if time.time() - start > timeout_sec:
                missing = []
                if not have_scan:
                    missing.append("/scan")
                if require_amcl and not have_amcl:
                    missing.append("/amcl_pose")
                if not have_tf:
                    missing.append(
                        f"TF {self.args.map_frame}->{self.args.base_frame}/"
                        f"{self.args.fallback_base_frame}"
                    )
                raise RuntimeError(
                    "Timed out waiting for startup data: " + ", ".join(missing)
                )
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("ROS shutdown during startup gate.")

    def lookup_pose(self):
        errors = []
        for frame in ordered_base_frames(self.args.base_frame, self.args.fallback_base_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.map_frame,
                    frame,
                    Time(),
                )
                self.base_frame_used = frame
                return transform_to_pose2d(transform, frame), frame
            except Exception as exc:
                errors.append(f"{frame}: {exc}")
        raise RuntimeError("Could not lookup TF pose: " + "; ".join(errors))

    def update_tf_tracking(self, pose):
        if pose.stamp_sec is None:
            return None
        now = time.time()
        if self.last_tf_stamp_sec is None or pose.stamp_sec != self.last_tf_stamp_sec:
            self.last_tf_stamp_sec = pose.stamp_sec
            self.last_tf_stamp_change_local_sec = now
        if self.last_tf_stamp_change_local_sec is None:
            self.last_tf_stamp_change_local_sec = now
        return now - self.last_tf_stamp_change_local_sec

    def reset_tf_tracking(self):
        self.last_tf_stamp_sec = None
        self.last_tf_stamp_change_local_sec = None

    def refresh_after_operator_wait(self, min_scan_stamp_sec, timeout_sec=None):
        self.reset_tf_tracking()
        timeout_sec = timeout_sec or self.args.startup_timeout_sec
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            scan_stamp_sec = (
                None
                if self.last_scan is None
                else replan_runtime.scan_stamp_sec(self.last_scan)
            )
            if (
                self.last_scan is not None
                and self.last_scan_received_sec is not None
                and self.last_scan_received_sec >= min_scan_stamp_sec
                and (
                    scan_stamp_sec is None
                    or scan_stamp_sec
                    >= min_scan_stamp_sec - replan_runtime.FRESH_SCAN_STAMP_SLACK_SEC
                )
            ):
                return
        raise RuntimeError(
            "Timed out waiting for fresh stamped /scan after handoff pause."
        )

    def record_motion_sample(self, yaw_error_deg, linear_x, angular_z, sample_seconds):
        abs_error = abs(yaw_error_deg)
        self.diagnostics.max_abs_yaw_error_deg = max(
            self.diagnostics.max_abs_yaw_error_deg,
            abs_error,
        )
        self.diagnostics.yaw_error_sum_deg += abs_error
        self.diagnostics.yaw_error_count += 1
        if abs(linear_x) <= 1e-9 and abs(angular_z) > 1e-9:
            self.diagnostics.rotate_seconds += sample_seconds
        else:
            self.diagnostics.forward_seconds += sample_seconds

    def current_amcl_health(self):
        if self.last_amcl is None:
            return AmclHealth(
                ok=not self.args.fail_on_bad_localization,
                warnings=["missing_amcl"],
                cov_x=None,
                cov_y=None,
                cov_yaw=None,
                age_sec=None,
            )
        age_sec = (
            None if self.last_amcl_received_sec is None
            else time.time() - self.last_amcl_received_sec
        )
        covariance = self.last_amcl.pose.covariance
        return evaluate_amcl_health(
            covariance,
            age_sec,
            self.args.max_amcl_age_sec,
            self.args.max_amcl_var_x,
            self.args.max_amcl_var_y,
            self.args.max_amcl_var_yaw,
            fail_on_bad_localization=self.args.fail_on_bad_localization,
        )

    def check_health_or_raise(self):
        pose, frame = self.lookup_pose()
        if pose.stamp_sec is None:
            raise RuntimeError("TF pose has no usable timestamp.")
        pose_age = time.time() - pose.stamp_sec
        self.diagnostics.tf_pose_age_sec = pose_age
        if pose_age > self.args.max_pose_age_sec:
            self.diagnostics.tf_stale_warning_count += 1
            message = f"TF pose is stale: age={pose_age:.3f} sec"
            if self.args.fail_on_stale_tf:
                raise RuntimeError(message)
            self.get_logger().warn(message)

        tf_update_gap_sec = self.update_tf_tracking(pose)
        if (
            tf_update_gap_sec is not None
            and tf_update_gap_sec > self.args.max_tf_update_gap_sec
        ):
            raise RecoverableHealthError(
                "tf_update_gap",
                self.args.tf_recovery_time_sec,
                "TF transform stamp stopped updating: "
                f"gap={tf_update_gap_sec:.3f} sec, "
                f"limit={self.args.max_tf_update_gap_sec:.3f} sec",
            )

        scan_age = (
            None if self.last_scan_received_sec is None
            else time.time() - self.last_scan_received_sec
        )
        if not age_ok(scan_age, self.args.max_scan_age_sec):
            raise RuntimeError(f"/scan is stale: age={scan_age}")

        amcl_health = self.current_amcl_health()
        if amcl_health.warnings:
            self.diagnostics.localization_warning_count += 1
            message = "AMCL localization warning(s): " + ",".join(amcl_health.warnings)
            if not amcl_health.ok:
                raise RuntimeError(message)
            if self.args.pause_on_bad_localization:
                raise RecoverableHealthError(
                    "bad_localization",
                    self.args.localization_recovery_time_sec,
                    message,
                )
            self.get_logger().warn(message)

        return pose, frame, amcl_health

    def check_health_or_recover(self):
        while True:
            try:
                return self.check_health_or_raise()
            except RecoverableHealthError as exc:
                self.diagnostics.recovery_pause_count += 1
                self.stop_repeatedly()
                self.get_logger().warn(
                    f"{exc}; pausing for up to {exc.timeout_sec:.1f} sec"
                )
                deadline = time.time() + exc.timeout_sec
                last_message = str(exc)
                while time.time() < deadline and rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
                    time.sleep(0.1)
                    try:
                        return self.check_health_or_raise()
                    except RecoverableHealthError as retry_exc:
                        last_message = str(retry_exc)
                raise RuntimeError(
                    f"{exc.reason} did not recover within "
                    f"{exc.timeout_sec:.1f} sec: {last_message}"
                )

    def check_scan_or_raise(self, mode):
        if self.last_scan is None:
            raise RuntimeError("No /scan sample is available.")
        safety = evaluate_scan_safety(
            self.last_scan.ranges,
            self.last_scan.angle_min,
            self.last_scan.angle_increment,
            self.last_scan.range_min,
            self.last_scan.range_max,
            mode,
            self.args.scan_half_angle_deg,
            self.args.hard_stop_range_m,
            self.args.min_scan_range_m,
            self.args.rotation_stop_range_m,
        )
        if not safety.safe:
            raise BlockedByScanError(safety)
        return safety

    def update_replan_diagnostics(self, result, count_replan=True):
        diag = result.diagnostics
        if count_replan:
            self.diagnostics.replan_count += 1
        self.diagnostics.last_replan_reason = result.reason
        self.diagnostics.updated_map_yaml = result.updated_map_yaml or ""
        self.diagnostics.updated_waypoints_csv = result.updated_waypoints_csv or ""
        self.diagnostics.detected_obstacle_count = diag.detected_obstacle_count
        self.diagnostics.candidate_scan_points = diag.candidate_scan_points
        self.diagnostics.filtered_obstacle_points = diag.filtered_obstacle_points
        self.diagnostics.raw_obstacle_cells = diag.raw_obstacle_cells
        self.diagnostics.free_obstacle_cells = diag.free_obstacle_cells
        self.diagnostics.inflated_cells_total = diag.inflated_cells_total
        self.diagnostics.inflated_cells_newly_occupied = diag.inflated_cells_newly_occupied
        self.diagnostics.inflated_cells_over_static_occupied = diag.inflated_cells_over_static_occupied
        self.diagnostics.scan_frame = diag.scan_frame
        self.diagnostics.scan_age_sec = diag.scan_age_sec
        self.diagnostics.tf_age_sec = diag.tf_age_sec
        self.diagnostics.tf_lookup_mode = diag.tf_lookup_mode
        self.diagnostics.start_snap_distance_m = diag.start_snap_distance_m
        self.diagnostics.goal_snap_distance_m = diag.goal_snap_distance_m
        self.diagnostics.old_remaining_waypoint_count = diag.old_remaining_waypoint_count
        self.diagnostics.new_waypoint_count = diag.new_waypoint_count
        self.diagnostics.old_path_length_m = diag.old_path_length_m
        self.diagnostics.new_path_length_m = diag.new_path_length_m
        self.diagnostics.replan_duration_sec = diag.replan_duration_sec
        self.diagnostics.run_local_map_updates = diag.run_local_map_updates
        self.diagnostics.run_local_replan_count += diag.run_local_replan_count
        self.diagnostics.run_local_last_replan_reason = diag.run_local_last_replan_reason
        self.diagnostics.run_local_no_path_reason = diag.run_local_no_path_reason
        self.diagnostics.run_local_start_cell_blocked = diag.run_local_start_cell_blocked
        self.diagnostics.run_local_goal_cell_blocked = diag.run_local_goal_cell_blocked
        self.diagnostics.run_local_path_blocked_cell_count = diag.run_local_path_blocked_cell_count
        self.diagnostics.run_local_scan_points_valid = diag.run_local_scan_points_valid
        self.diagnostics.run_local_scan_points_used = diag.run_local_scan_points_used
        self.diagnostics.run_local_scan_points_rejected_invalid_range = (
            diag.run_local_scan_points_rejected_invalid_range
        )
        self.diagnostics.run_local_scan_points_rejected_static = diag.run_local_scan_points_rejected_static
        self.diagnostics.run_local_scan_points_rejected_bounds = diag.run_local_scan_points_rejected_bounds
        self.diagnostics.run_local_scan_points_rejected_wall_band = (
            diag.run_local_scan_points_rejected_wall_band
        )
        self.diagnostics.run_local_scan_points_rejected_low_confidence = (
            diag.run_local_scan_points_rejected_low_confidence
        )
        self.diagnostics.run_local_update_rejected_reason = diag.run_local_update_rejected_reason
        self.diagnostics.run_local_initial_scan_count = max(
            self.diagnostics.run_local_initial_scan_count,
            diag.run_local_initial_scan_count,
        )
        self.diagnostics.run_local_corridor_check_distance_m = (
            diag.run_local_corridor_check_distance_m
        )
        self.diagnostics.run_local_inflation_radius_m = diag.run_local_inflation_radius_m
        self.diagnostics.run_local_map_yaml = diag.run_local_map_yaml
        self.diagnostics.run_local_waypoints_csv = diag.run_local_waypoints_csv
        self.diagnostics.run_local_cell_source_counts = diag.run_local_cell_source_counts
        if result.run_local_map is not None:
            self.run_local_map = result.run_local_map
            publish_rviz_obstacles_if_available(self)

    def replanned_waypoints_from_result(self, result):
        return [
            Waypoint(index, x, y)
            for index, x, y in result.waypoints
        ]

    def first_motion_waypoint(self, replanned, current_pose):
        for waypoint in replanned:
            distance_m = math.hypot(
                waypoint.x - current_pose.x,
                waypoint.y - current_pose.y,
            )
            if distance_m > self.args.waypoint_tolerance_m:
                return waypoint
        return replanned[-1]

    def validate_replan_result(
        self,
        result,
        current_pose,
        old_remaining_waypoints,
        goal_waypoint,
        require_changed=True,
    ):
        if not result.success:
            raise RuntimeError(f"lidar_replan_failed:{result.reason}")
        replanned = self.replanned_waypoints_from_result(result)
        if not replanned:
            raise RuntimeError("lidar_replan_failed:empty_waypoint_list")
        final_error = waypoint_distance(replanned[-1], goal_waypoint)
        if final_error > self.args.goal_tolerance_m:
            raise RuntimeError(
                "lidar_replan_failed:final_goal_mismatch "
                f"error={final_error:.3f}"
            )
        old_pairs = [(round(wp.x, 3), round(wp.y, 3)) for wp in old_remaining_waypoints]
        new_pairs = [(round(wp.x, 3), round(wp.y, 3)) for wp in replanned]
        if require_changed and old_pairs == new_pairs:
            raise RuntimeError("lidar_replan_failed:updated_path_matches_old_path")
        motion_waypoint = self.first_motion_waypoint(replanned, current_pose)
        first_base = lidar_obstacle_map.map_point_to_base(
            motion_waypoint.x,
            motion_waypoint.y,
            lidar_obstacle_map.Pose2D(current_pose.x, current_pose.y, current_pose.yaw_deg),
        )
        if first_base.x < -self.args.robot_footprint_radius_m:
            raise RuntimeError("lidar_replan_failed:first_waypoint_behind_robot")
        first_heading = math.degrees(math.atan2(first_base.y, first_base.x))
        first_heading_error = abs(shortest_angle_delta_deg(0.0, first_heading))
        if not math.isfinite(first_heading_error) or first_heading_error > 180.0:
            raise RuntimeError("lidar_replan_failed:first_segment_heading_unreachable")
        if result.inflated_obstacle_cells and result.path_cells:
            obstacle_path_overlap = set(result.path_cells).intersection(result.inflated_obstacle_cells)
            if obstacle_path_overlap:
                raise RuntimeError("lidar_replan_failed:path_crosses_obstacle_cells")
        return replanned

    def initialize_run_local_route(self, current_pose, waypoints):
        if self.args.run_local_map_initial_scan_mode == "none":
            return list(waypoints)
        self.stop_repeatedly()
        goal_waypoint = waypoints[-1]
        result = replan_runtime.perform_initial_run_local_replan(
            self,
            self.args,
            current_pose,
            goal_waypoint,
            waypoints,
        )
        self.update_replan_diagnostics(result, count_replan=result.success)
        if not result.success and result.reason in INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS:
            self.stop_repeatedly()
            if not run_local_map_has_confirmed_obstacles(self.run_local_map):
                self.run_local_map = None
            self.get_logger().warn(
                "Initial run-local obstacle map did not find a confirmed "
                f"free-space obstacle; continuing with the static route. reason={result.reason}"
            )
            return list(waypoints)
        replanned = self.validate_replan_result(
            result,
            current_pose,
            waypoints,
            goal_waypoint,
            require_changed=False,
        )
        self.stop_repeatedly()
        self.get_logger().info(
            "Initial run-local obstacle map completed: "
            f"waypoints={len(replanned)}, map={result.updated_map_yaml}"
        )
        return replanned

    def corridor_blocked_cells(self, current_pose, remaining_waypoints):
        if self.run_local_map is None or not remaining_waypoints:
            return set()
        check_distance_m = self.args.run_local_map_corridor_check_distance_m
        corridor_radius_m = (
            self.args.run_local_map_corridor_radius_m
            if self.args.run_local_map_corridor_radius_m is not None
            else 0.0
        )
        blocked = lidar_obstacle_map.path_corridor_blocked_cells(
            self.run_local_map.static_map,
            lidar_obstacle_map.Pose2D(
                current_pose.x,
                current_pose.y,
                current_pose.yaw_deg,
            ),
            remaining_waypoints,
            self.run_local_map.inflated_obstacle_cells,
            check_distance_m,
            corridor_radius_m,
        )
        self.diagnostics.run_local_corridor_check_distance_m = check_distance_m
        self.diagnostics.run_local_path_blocked_cell_count = len(blocked)
        return blocked

    def plan_with_existing_run_local_map(
        self,
        current_pose,
        old_remaining_waypoints,
        sequence=None,
        count_replan=True,
    ):
        if self.run_local_map is None:
            raise RuntimeError("lidar_replan_failed:no_run_local_map")
        if not run_local_map_has_confirmed_obstacles(self.run_local_map):
            raise RuntimeError("lidar_replan_failed:no_confirmed_run_local_obstacles")
        sequence = sequence or self.live_replan_attempt_count + 1
        goal_waypoint = old_remaining_waypoints[-1]
        result = replan_runtime.plan_existing_run_local_map(
            self.args,
            self.run_local_map,
            current_pose,
            goal_waypoint,
            old_remaining_waypoints,
            sequence=sequence,
        )
        self.update_replan_diagnostics(result, count_replan=count_replan)
        return self.validate_replan_result(
            result,
            current_pose,
            old_remaining_waypoints,
            goal_waypoint,
            require_changed=True,
        )

    def replan_after_blockage(
        self,
        current_pose,
        old_remaining_waypoints,
        trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
    ):
        if self.live_replan_attempt_count >= self.args.max_replans:
            raise RuntimeError("lidar_replan_failed:max_replans_exceeded")
        sequence = self.live_replan_attempt_count + 1
        goal_waypoint = old_remaining_waypoints[-1]
        if self.args.run_local_map_update_mode == "none":
            replanned = self.plan_with_existing_run_local_map(
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            self.live_replan_attempt_count += 1
            self.get_logger().info("Replanned with existing run-local map.")
            return replanned
        try:
            result = replan_runtime.perform_lidar_replan(
                self,
                self.args,
                current_pose,
                goal_waypoint,
                old_remaining_waypoints,
                sequence=sequence,
            )
        except RuntimeError as exc:
            if trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE:
                raise lidar_replan_failure(exc) from exc
            if self.run_local_map is None:
                raise
            replanned = self.plan_with_existing_run_local_map(
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            self.live_replan_attempt_count += 1
            self.get_logger().warn(
                "LiDAR map update failed; replanned with existing run-local map."
            )
            return replanned
        self.update_replan_diagnostics(result)
        if not result.success and self.run_local_map is not None:
            rejected_reasons = {
                lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_TF,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_MAX_UPDATES_EXCEEDED,
            }
            if result.reason in rejected_reasons:
                if trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE:
                    raise lidar_replan_failure(result.reason)
                self.get_logger().warn(
                    "LiDAR map update rejected; "
                    f"replanning with existing run-local map. reason={result.reason}"
                )
                replanned = self.plan_with_existing_run_local_map(
                    current_pose,
                    old_remaining_waypoints,
                    sequence=sequence,
                )
                self.live_replan_attempt_count += 1
                return replanned
        replanned = self.validate_replan_result(
            result,
            current_pose,
            old_remaining_waypoints,
            goal_waypoint,
        )
        self.live_replan_attempt_count += 1
        self.get_logger().info(
            "LiDAR obstacle replan completed: "
            f"waypoints={len(replanned)}, map={result.updated_map_yaml}"
        )
        return replanned

    def follow_waypoints(self, waypoints):
        reached_count = 0
        start_pose, _frame, amcl_health = self.check_health_or_recover()
        final_pose = start_pose
        last_scan_safety = None
        self.start_pose = start_pose
        self.final_pose = final_pose
        self.last_amcl_health = amcl_health

        waypoints = list(waypoints)
        publish_rviz_route_if_available(self, waypoints, current_pose=start_pose)
        publish_rviz_obstacles_if_available(self)
        if self.args.enable_lidar_map_replan:
            waypoints = self.initialize_run_local_route(start_pose, waypoints)
            publish_rviz_route_if_available(self, waypoints, current_pose=start_pose)
            if self.args.lidar_replan_artifact_only:
                self.stop_repeatedly()
                return {
                    "reached_count": reached_count,
                    "start_pose": start_pose,
                    "final_pose": final_pose,
                    "scan_safety": last_scan_safety,
                    "amcl_health": amcl_health,
                    "base_frame_used": self.base_frame_used,
                    "status": "replan_artifact_only_complete",
                }

        waypoint_index = 0
        while waypoint_index < len(waypoints):
            waypoint = waypoints[waypoint_index]
            publish_rviz_route_if_available(
                self,
                waypoints[waypoint_index:],
                current_pose=final_pose,
                current_waypoint_index=0,
            )
            self.get_logger().info(
                f"[{waypoint_index + 1}/{len(waypoints)}] "
                f"target waypoint {waypoint.index}: "
                f"x={waypoint.x:.3f}, y={waypoint.y:.3f}"
            )
            waypoint_start = time.time()
            mode = "forward"
            reached_current = False
            replanned_current = False

            while rclpy.ok():
                pose, _frame, amcl_health = self.check_health_or_recover()
                final_pose = pose
                self.final_pose = final_pose
                self.last_amcl_health = amcl_health
                state = target_state(pose, waypoint)
                is_final = waypoint_index == len(waypoints) - 1

                if waypoint_reached(
                    state.distance_m,
                    is_final,
                    self.args.waypoint_tolerance_m,
                    self.args.goal_tolerance_m,
                ):
                    reached_count += 1
                    self.reached_count = reached_count
                    self.stop_repeatedly()
                    time.sleep(self.args.settle_sec)
                    reached_current = True
                    break

                if time.time() - waypoint_start > self.args.max_waypoint_time_sec:
                    raise WaypointTimeoutError(waypoint)

                rotate_mode = should_rotate(
                    mode,
                    state.yaw_error_deg,
                    self.args.rotate_start_heading_error_deg,
                    self.args.rotate_stop_heading_error_deg,
                )
                mode = "rotate" if rotate_mode else "forward"
                try:
                    last_scan_safety = self.check_scan_or_raise(mode)
                    self.last_scan_safety = last_scan_safety
                except BlockedByScanError as exc:
                    if self.args.enable_lidar_map_replan:
                        remaining = waypoints[waypoint_index:]
                        replanned = self.replan_after_blockage(
                            pose,
                            remaining,
                            trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
                        )
                        publish_rviz_route_if_available(
                            self,
                            replanned,
                            current_pose=pose,
                            current_waypoint_index=0,
                        )
                        if self.args.lidar_replan_artifact_only:
                            self.stop_repeatedly()
                            return {
                                "reached_count": reached_count,
                                "start_pose": start_pose,
                                "final_pose": final_pose,
                                "scan_safety": exc.scan_safety,
                                "amcl_health": amcl_health,
                                "base_frame_used": self.base_frame_used,
                                "status": "replan_artifact_only_complete",
                            }
                        waypoints = replanned
                        waypoint_index = 0
                        replanned_current = True
                        break
                    raise BlockedByScanError(exc.scan_safety, waypoint) from exc
                if self.args.enable_lidar_map_replan and self.run_local_map is not None:
                    remaining = waypoints[waypoint_index:]
                    blocked_cells = self.corridor_blocked_cells(pose, remaining)
                    if blocked_cells:
                        publish_rviz_obstacles_if_available(self, blocked_cells)
                        self.stop_repeatedly()
                        replanned = self.replan_after_blockage(
                            pose,
                            remaining,
                            trigger=REPLAN_TRIGGER_KNOWN_CORRIDOR,
                        )
                        publish_rviz_route_if_available(
                            self,
                            replanned,
                            current_pose=pose,
                            current_waypoint_index=0,
                        )
                        if self.args.lidar_replan_artifact_only:
                            self.stop_repeatedly()
                            return {
                                "reached_count": reached_count,
                                "start_pose": start_pose,
                                "final_pose": final_pose,
                                "scan_safety": last_scan_safety,
                                "amcl_health": amcl_health,
                                "base_frame_used": self.base_frame_used,
                                "status": "replan_artifact_only_complete",
                            }
                        waypoints = replanned
                        waypoint_index = 0
                        replanned_current = True
                        break
                linear_x, angular_z = velocity_command(
                    state.distance_m,
                    state.yaw_error_deg,
                    rotate_mode,
                    self.args.linear_speed,
                    self.args.min_linear_speed,
                    self.args.linear_gain,
                    self.args.max_angular_speed,
                    self.args.yaw_gain,
                    self.args.forward_yaw_deadband_deg,
                    self.args.forward_stop_heading_error_deg,
                )
                self.record_motion_sample(
                    state.yaw_error_deg,
                    linear_x,
                    angular_z,
                    1.0 / self.args.control_rate_hz,
                )
                self.publish_velocity(linear_x, angular_z)
                rclpy.spin_once(self, timeout_sec=1.0 / self.args.control_rate_hz)
                time.sleep(1.0 / self.args.control_rate_hz)

            if replanned_current:
                continue
            if reached_current:
                waypoint_index += 1
                continue
            raise RuntimeError("ROS shutdown while following waypoints")

        return {
            "reached_count": reached_count,
            "start_pose": start_pose,
            "final_pose": final_pose,
            "scan_safety": last_scan_safety,
            "amcl_health": amcl_health,
            "base_frame_used": self.base_frame_used,
        }


class BlockedByScanError(RuntimeError):
    def __init__(self, scan_safety, waypoint=None):
        super().__init__(
            "Blocked by /scan safety: "
            f"reason={scan_safety.reason}, "
            f"min={scan_safety.min_range_m}, p05={scan_safety.percentile_5_m}"
        )
        self.scan_safety = scan_safety
        self.waypoint = waypoint


class WaypointTimeoutError(RuntimeError):
    def __init__(self, waypoint):
        super().__init__(f"Timed out trying to reach waypoint {waypoint.index}")
        self.waypoint = waypoint


class RecoverableHealthError(RuntimeError):
    def __init__(self, reason, timeout_sec, message):
        super().__init__(message)
        self.reason = reason
        self.timeout_sec = timeout_sec


def transform_to_pose2d(transform, frame_id):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp = transform.header.stamp
    stamp_sec = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    return Pose2D(
        x=float(translation.x),
        y=float(translation.y),
        yaw_deg=quaternion_to_yaw_deg(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def require_motion_confirmation(args, waypoints):
    if args.yes:
        return True

    print("\nThis command will publish /cmd_vel to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - Nav2 localization is running with the saved map")
    print("  - RViz pose estimate is set and /scan aligns with the map")
    print("  - no active Nav2 goal/controller is publishing /cmd_vel")
    print("  - clear the path and keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print(f"Run ID: {args.run_id}")
    print(f"Waypoints: {len(waypoints)} from {args.waypoints}")
    response = input("Type RUN to start waypoint following: ").strip()
    return response == "RUN"


def wait_before_follow_confirmation(args, current_pose, executable_waypoints, input_fn=input):
    if not args.wait_before_follow:
        return True

    print("\nWaypoint follower handoff is ready.")
    print("The robot is stopped after Nav2 staging and before custom waypoint following.")
    print("Place the temporary obstacle on the planned path now.")
    print("Safety requirements:")
    print("  - keep the path area clear except for the test obstacle")
    print("  - keep Ctrl+C and physical stop available")
    print(
        "Current pose: "
        f"x={current_pose.x:.3f}, y={current_pose.y:.3f}, yaw={current_pose.yaw_deg:.1f} deg"
    )
    if executable_waypoints:
        first = executable_waypoints[0]
        print(
            "First follower waypoint: "
            f"index={first.index}, x={first.x:.3f}, y={first.y:.3f}"
        )
    response = input_fn("Type RUN to start custom waypoint following: ").strip()
    return response == "RUN"


def print_dry_run(args, raw_waypoints, executable_waypoints):
    print("Waypoint follower dry run")
    print(f"Waypoint CSV: {args.waypoints}")
    print(f"Raw waypoints: {len(raw_waypoints)}")
    print(f"Executable waypoints: {len(executable_waypoints)}")
    print(f"Map frame: {args.map_frame}")
    print(f"Base frame: {args.base_frame}, fallback: {args.fallback_base_frame}")
    print(f"Linear speed: {args.linear_speed:.3f} m/s")
    print(f"Max angular speed: {args.max_angular_speed:.3f} rad/s")
    print(f"Waypoint tolerance: {args.waypoint_tolerance_m:.3f} m")
    print(f"Goal tolerance: {args.goal_tolerance_m:.3f} m")
    print(f"Start selection: {args.start_selection}")
    print(f"Wait before follow: {'yes' if args.wait_before_follow else 'no'}")
    print(f"RViz visualization: {'disabled' if args.no_rviz_visualization else 'enabled'}")
    if not args.no_rviz_visualization:
        print(f"  path topic: {args.rviz_path_topic}")
        print(f"  waypoint markers: {args.rviz_waypoint_marker_topic}")
        print(f"  obstacle markers: {args.rviz_obstacle_marker_topic}")
    print(f"LiDAR map replan: {'enabled' if args.enable_lidar_map_replan else 'disabled'}")
    if args.enable_lidar_map_replan:
        print(f"  artifact only: {'yes' if args.lidar_replan_artifact_only else 'no'}")
        print(f"  static map: {args.static_map}")
        print(f"  output dir: {args.replan_output_dir}")
        print(
            "  initial scans: "
            f"{args.run_local_map_initial_scan_mode} x "
            f"{args.run_local_map_initial_scan_count}"
        )
        print(f"  update mode: {args.run_local_map_update_mode}")
        print(f"  min hit count: {args.run_local_map_min_hit_count}")
        print(f"  inflation radius: {args.run_local_map_inflation_radius_m:.3f} m")
    if args.start_selection == "path-progress":
        print(
            "Runtime route selection uses live TF after startup; "
            "the route below is a fixed-skip preview."
        )
    print("Executable route:")
    for index, waypoint in enumerate(executable_waypoints, start=1):
        print(f"  {index}. source index {waypoint.index}: x={waypoint.x:.3f}, y={waypoint.y:.3f}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Follow planned A* waypoints using TF pose and /cmd_vel.",
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--run-id", help="Run ID for logging.")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--fallback-base-frame", default="base_link")
    parser.add_argument("--linear-speed", default=DEFAULT_LINEAR_SPEED_MPS, type=float)
    parser.add_argument("--min-linear-speed", default=DEFAULT_MIN_LINEAR_SPEED_MPS, type=float)
    parser.add_argument("--linear-gain", default=DEFAULT_LINEAR_GAIN, type=float)
    parser.add_argument("--max-angular-speed", default=DEFAULT_MAX_ANGULAR_SPEED_RADPS, type=float)
    parser.add_argument("--yaw-gain", default=DEFAULT_YAW_GAIN, type=float)
    parser.add_argument("--forward-yaw-deadband-deg", default=DEFAULT_FORWARD_YAW_DEADBAND_DEG, type=float)
    parser.add_argument("--forward-stop-heading-error-deg", default=DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG, type=float)
    parser.add_argument("--waypoint-tolerance-m", default=DEFAULT_WAYPOINT_TOLERANCE_M, type=float)
    parser.add_argument("--goal-tolerance-m", default=DEFAULT_GOAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--rotate-start-heading-error-deg",
        default=DEFAULT_ROTATE_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--rotate-stop-heading-error-deg",
        default=DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument("--min-waypoint-spacing-m", default=DEFAULT_MIN_WAYPOINT_SPACING_M, type=float)
    parser.add_argument(
        "--start-selection",
        default=DEFAULT_START_SELECTION,
        choices=["path-progress", "fixed-skip"],
    )
    parser.add_argument("--start-on-path-tolerance-m", default=DEFAULT_START_ON_PATH_TOLERANCE_M, type=float)
    parser.add_argument("--scan-half-angle-deg", default=DEFAULT_SCAN_HALF_ANGLE_DEG, type=float)
    parser.add_argument("--hard-stop-range-m", default=DEFAULT_HARD_STOP_RANGE_M, type=float)
    parser.add_argument("--min-scan-range-m", default=DEFAULT_MIN_SCAN_RANGE_M, type=float)
    parser.add_argument("--rotation-stop-range-m", default=DEFAULT_ROTATION_STOP_RANGE_M, type=float)
    parser.add_argument("--max-pose-age-sec", default=DEFAULT_MAX_POSE_AGE_SEC, type=float)
    parser.add_argument("--max-scan-age-sec", default=DEFAULT_MAX_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-age-sec", default=DEFAULT_MAX_AMCL_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-var-x", default=DEFAULT_MAX_AMCL_VAR_X, type=float)
    parser.add_argument("--max-amcl-var-y", default=DEFAULT_MAX_AMCL_VAR_Y, type=float)
    parser.add_argument("--max-amcl-var-yaw", default=DEFAULT_MAX_AMCL_VAR_YAW, type=float)
    parser.add_argument("--max-waypoint-time-sec", default=DEFAULT_MAX_WAYPOINT_TIME_SEC, type=float)
    parser.add_argument("--max-tf-update-gap-sec", default=DEFAULT_MAX_TF_UPDATE_GAP_SEC, type=float)
    parser.add_argument("--tf-recovery-time-sec", default=DEFAULT_TF_RECOVERY_TIME_SEC, type=float)
    parser.add_argument(
        "--localization-recovery-time-sec",
        default=DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC,
        type=float,
    )
    parser.add_argument("--control-rate-hz", default=DEFAULT_CONTROL_RATE_HZ, type=float)
    parser.add_argument("--settle-sec", default=DEFAULT_SETTLE_SEC, type=float)
    parser.add_argument("--startup-timeout-sec", default=DEFAULT_STARTUP_TIMEOUT_SEC, type=float)
    parser.add_argument("--notes", default="follow_planned_waypoints")
    parser.add_argument("--fail-on-bad-localization", action="store_true")
    parser.add_argument("--pause-on-bad-localization", action="store_true")
    parser.add_argument("--require-amcl-startup", action="store_true")
    parser.add_argument("--fail-on-stale-tf", action="store_true")
    parser.add_argument("--no-skip-first-waypoint", action="store_true")
    parser.add_argument("--rviz-path-topic", default=DEFAULT_RVIZ_PATH_TOPIC)
    parser.add_argument(
        "--rviz-waypoint-marker-topic",
        default=DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC,
    )
    parser.add_argument(
        "--rviz-obstacle-marker-topic",
        default=DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC,
    )
    parser.add_argument("--no-rviz-visualization", action="store_true")
    parser.add_argument("--enable-lidar-map-replan", action="store_true")
    parser.add_argument("--lidar-replan-artifact-only", action="store_true")
    parser.add_argument("--static-map", default=DEFAULT_STATIC_MAP, type=Path)
    parser.add_argument("--replan-output-dir", default=DEFAULT_REPLAN_OUTPUT_DIR, type=Path)
    parser.add_argument("--max-replans", default=1, type=int)
    parser.add_argument("--replan-timeout-sec", default=DEFAULT_REPLAN_TIMEOUT_SEC, type=float)
    parser.add_argument("--max-replan-scan-age-sec", default=DEFAULT_MAX_REPLAN_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-replan-tf-age-sec", default=DEFAULT_MAX_REPLAN_TF_AGE_SEC, type=float)
    parser.add_argument("--allow-latest-tf-replan-fallback", action="store_true")
    parser.add_argument("--obstacle-forward-distance-m", default=DEFAULT_OBSTACLE_FORWARD_DISTANCE_M, type=float)
    parser.add_argument("--obstacle-forward-half-width-m", default=DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M, type=float)
    parser.add_argument("--obstacle-angle-window-deg", default=DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG, type=float)
    parser.add_argument("--obstacle-min-range-m", default=DEFAULT_OBSTACLE_MIN_RANGE_M, type=float)
    parser.add_argument("--robot-footprint-radius-m", default=DEFAULT_ROBOT_FOOTPRINT_RADIUS_M, type=float)
    parser.add_argument("--obstacle-min-cluster-size", default=DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE, type=int)
    parser.add_argument("--obstacle-min-cluster-width-m", default=DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M, type=float)
    parser.add_argument("--obstacle-inflate-radius-m", default=DEFAULT_OBSTACLE_INFLATE_RADIUS_M, type=float)
    parser.add_argument("--max-start-snap-m", default=DEFAULT_MAX_START_SNAP_M, type=float)
    parser.add_argument("--max-goal-snap-m", default=DEFAULT_MAX_GOAL_SNAP_M, type=float)
    parser.add_argument(
        "--max-replan-path-length-ratio",
        default=DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-initial-scan-mode",
        default=DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE,
        choices=["none", "forward", "full"],
    )
    parser.add_argument(
        "--run-local-map-initial-scan-count",
        default=DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-update-mode",
        default=DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE,
        choices=["none", "forward", "full"],
    )
    parser.add_argument(
        "--run-local-map-min-hit-count",
        default=DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-inflation-radius-m",
        default=DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-tf-age-sec",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-scan-age-sec",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-min-used-points",
        default=DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-max-rejected-ratio",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-corridor-check-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--run-local-map-corridor-radius-m", type=float)
    parser.add_argument(
        "--run-local-map-clearance-margin-m",
        default=DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-updates",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES,
        type=int,
    )
    parser.add_argument("--run-local-map-artifact-prefix")
    parser.add_argument("--wait-before-follow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("waypoint_follow_%Y%m%d_%H%M%S")
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    positive_fields = [
        "linear_speed",
        "min_linear_speed",
        "linear_gain",
        "max_angular_speed",
        "yaw_gain",
        "waypoint_tolerance_m",
        "goal_tolerance_m",
        "rotate_start_heading_error_deg",
        "rotate_stop_heading_error_deg",
        "scan_half_angle_deg",
        "hard_stop_range_m",
        "min_scan_range_m",
        "rotation_stop_range_m",
        "start_on_path_tolerance_m",
        "max_pose_age_sec",
        "max_scan_age_sec",
        "max_amcl_age_sec",
        "max_amcl_var_x",
        "max_amcl_var_y",
        "max_amcl_var_yaw",
        "max_waypoint_time_sec",
        "max_tf_update_gap_sec",
        "tf_recovery_time_sec",
        "localization_recovery_time_sec",
        "control_rate_hz",
        "startup_timeout_sec",
        "replan_timeout_sec",
        "max_replan_scan_age_sec",
        "max_replan_tf_age_sec",
        "obstacle_forward_distance_m",
        "obstacle_forward_half_width_m",
        "obstacle_angle_window_deg",
        "obstacle_min_range_m",
        "robot_footprint_radius_m",
        "obstacle_min_cluster_width_m",
        "obstacle_inflate_radius_m",
        "max_start_snap_m",
        "max_goal_snap_m",
        "max_replan_path_length_ratio",
        "run_local_map_inflation_radius_m",
        "run_local_map_max_tf_age_sec",
        "run_local_map_max_scan_age_sec",
        "run_local_map_corridor_check_distance_m",
        "run_local_map_clearance_margin_m",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.min_linear_speed > args.linear_speed:
        parser.error("--min-linear-speed must be <= --linear-speed")
    if args.rotate_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error("--rotate-stop-heading-error-deg must be < --rotate-start-heading-error-deg")
    if args.forward_yaw_deadband_deg < 0.0:
        parser.error("--forward-yaw-deadband-deg must be non-negative")
    if args.forward_yaw_deadband_deg >= args.forward_stop_heading_error_deg:
        parser.error("--forward-yaw-deadband-deg must be < --forward-stop-heading-error-deg")
    if args.forward_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error(
            "--forward-stop-heading-error-deg must be < "
            "--rotate-start-heading-error-deg"
        )
    if args.hard_stop_range_m >= args.min_scan_range_m:
        parser.error("--hard-stop-range-m must be < --min-scan-range-m")
    if args.hard_stop_range_m >= args.rotation_stop_range_m:
        parser.error("--hard-stop-range-m must be < --rotation-stop-range-m")
    if not (0.0 < args.scan_half_angle_deg <= 90.0):
        parser.error("--scan-half-angle-deg must be > 0 and <= 90")
    if args.settle_sec < 0.0:
        parser.error("--settle-sec must be non-negative")
    if args.max_replans < 1:
        parser.error("--max-replans must be >= 1")
    if args.run_local_map_initial_scan_count < 1:
        parser.error("--run-local-map-initial-scan-count must be >= 1")
    if args.run_local_map_min_hit_count < 1:
        parser.error("--run-local-map-min-hit-count must be >= 1")
    if args.run_local_map_min_used_points < 1:
        parser.error("--run-local-map-min-used-points must be >= 1")
    if not (0.0 <= args.run_local_map_max_rejected_ratio <= 1.0):
        parser.error("--run-local-map-max-rejected-ratio must be between 0 and 1")
    if (
        args.run_local_map_corridor_radius_m is not None
        and args.run_local_map_corridor_radius_m <= 0.0
    ):
        parser.error("--run-local-map-corridor-radius-m must be greater than zero")
    if args.run_local_map_max_updates < 1:
        parser.error("--run-local-map-max-updates must be >= 1")
    if args.obstacle_min_cluster_size < 1:
        parser.error("--obstacle-min-cluster-size must be >= 1")


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        raw_waypoints = load_waypoints(args.waypoints)
        preview_waypoints = prepare_executable_waypoints(
            raw_waypoints,
            skip_first=not args.no_skip_first_waypoint,
            min_spacing_m=args.min_waypoint_spacing_m,
        )
    except Exception as exc:
        print(f"follow_planned_waypoints.py: error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(args, raw_waypoints, preview_waypoints)
        return 0

    if not require_motion_confirmation(args, preview_waypoints):
        print("Waypoint following cancelled.")
        return 130

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = WaypointFollower(args)
    status = "failed"
    notes = args.notes
    reached_count = 0
    executable_waypoints = preview_waypoints
    start_pose = None
    final_pose = None
    blocked_waypoint = None
    timeout_waypoint = None
    scan_safety = None
    amcl_health = None
    return_code = 1

    try:
        node.wait_for_startup_gate()
        start_pose, _frame, amcl_health = node.check_health_or_recover()
        start_selection = select_executable_waypoints(
            raw_waypoints,
            start_pose,
            args.start_selection,
            args.start_on_path_tolerance_m,
            args.waypoint_tolerance_m,
            args.goal_tolerance_m,
            args.min_waypoint_spacing_m,
            skip_first=not args.no_skip_first_waypoint,
        )
        executable_waypoints = start_selection.waypoints
        node.diagnostics.selected_start_segment_index = (
            start_selection.selected_segment_index
        )
        node.diagnostics.selected_start_waypoint_index = (
            start_selection.selected_waypoint_index
        )
        node.diagnostics.distance_to_path_m = start_selection.distance_to_path_m
        node.get_logger().info(
            "Selected executable route: "
            f"segment={start_selection.selected_segment_index}, "
            f"first_waypoint={start_selection.selected_waypoint_index}, "
            f"distance_to_path={start_selection.distance_to_path_m}"
        )
        node.publish_rviz_route(executable_waypoints, current_pose=start_pose)
        node.publish_rviz_obstacles()
        if not wait_before_follow_confirmation(args, start_pose, executable_waypoints):
            status = "interrupted"
            notes = f"{args.notes};wait_before_follow_cancelled"
            node.diagnostics.final_status_reason = "wait_before_follow_cancelled"
            print("Waypoint following cancelled before custom follower start.")
            return_code = 130
        else:
            if args.wait_before_follow:
                node.refresh_after_operator_wait(time.time())
            result = node.follow_waypoints(executable_waypoints)
            reached_count = result["reached_count"]
            start_pose = result["start_pose"]
            final_pose = result["final_pose"]
            scan_safety = result["scan_safety"]
            amcl_health = result["amcl_health"]
            status = result.get("status", "completed")
            node.diagnostics.final_status_reason = status
            return_code = 0

    except KeyboardInterrupt:
        status = "interrupted"
        notes = f"{args.notes};keyboard_interrupt"
        node.diagnostics.final_status_reason = "keyboard_interrupt"
        print("Interrupted. Sending stop command...")
        return_code = 130

    except BlockedByScanError as exc:
        status = "blocked"
        notes = f"{args.notes};{exc}"
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        amcl_health = node.last_amcl_health
        scan_safety = exc.scan_safety
        blocked_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except WaypointTimeoutError as exc:
        status = "timeout"
        notes = f"{args.notes};{exc}"
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        timeout_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except Exception as exc:
        status = "failed"
        notes = f"{args.notes};{exc}"
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        node.get_logger().error(str(exc))
        return_code = 1

    finally:
        try:
            node.stop_repeatedly()
            if final_pose is None:
                try:
                    final_pose, _frame = node.lookup_pose()
                except Exception:
                    final_pose = None
        finally:
            if not args.no_log:
                try:
                    row = build_log_row(
                        args,
                        len(executable_waypoints),
                        reached_count,
                        status,
                        notes,
                        start_pose=start_pose,
                        final_pose=final_pose,
                        blocked_waypoint=blocked_waypoint,
                        timeout_waypoint=timeout_waypoint,
                        base_frame_used=node.base_frame_used,
                        scan_safety=scan_safety,
                        amcl_health=amcl_health,
                        diagnostics=node.diagnostics,
                    )
                    append_csv_row(args.results_csv, CSV_HEADER, row)
                    node.get_logger().info(f"Saved run log to {args.results_csv}")
                except Exception as log_exc:
                    print(f"Could not write waypoint-follow log: {log_exc}", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
