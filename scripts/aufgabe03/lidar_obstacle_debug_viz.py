#!/usr/bin/env python3
"""
Publish RViz diagnostics for the run-local LiDAR obstacle detector.

The node is read-only: it subscribes to /scan, optionally uses TF and the static
map to run the same temporary-obstacle filtering used by the replan code, and
publishes visualization_msgs/MarkerArray layers for RViz.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

try:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
    import tf2_ros
except ImportError:
    rclpy = None
    Duration = None
    Node = object
    qos_profile_sensor_data = None
    Point = None
    LaserScan = object
    Marker = None
    MarkerArray = None
    tf2_ros = None

import lidar_obstacle_map
import map_path_planner as planner
import replan_runtime


DEFAULT_STATIC_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_SCAN_TOPIC = "/scan"
DEFAULT_MARKER_TOPIC = "/mii_amr/lidar_obstacle_debug/markers"
DEFAULT_MAP_FRAME = "map"
DEFAULT_MARKER_LIFETIME_SEC = 0.75
DEFAULT_WARN_INTERVAL_SEC = 2.0
DEBUG_MAX_UPDATES = 100_000


@dataclass(frozen=True)
class ScanPointLayers:
    raw_points: tuple[lidar_obstacle_map.BaseFramePoint, ...]
    roi_points: tuple[lidar_obstacle_map.BaseFramePoint, ...]

    @property
    def roi_rejected_count(self):
        return len(self.raw_points) - len(self.roi_points)


@dataclass(frozen=True)
class ObservationCellLayers:
    total_observations: int = 0
    accepted_cells: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    rejected_static_cells: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    rejected_wall_band_cells: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    rejected_invalid_range: int = 0
    rejected_bounds: int = 0


@dataclass(frozen=True)
class Rgba:
    r: float
    g: float
    b: float
    a: float = 1.0


COLOR_RAW_SCAN = Rgba(0.65, 0.65, 0.65, 0.32)
COLOR_ROI_SCAN = Rgba(1.0, 0.82, 0.16, 0.95)
COLOR_ROI_BOUNDARY = Rgba(1.0, 0.82, 0.16, 0.65)
COLOR_CURRENT_ACCEPTED = Rgba(0.0, 0.65, 1.0, 0.55)
COLOR_REJECTED_STATIC = Rgba(0.95, 0.22, 0.18, 0.40)
COLOR_REJECTED_WALL_BAND = Rgba(1.0, 0.45, 0.0, 0.40)
COLOR_CONFIRMED_RAW = Rgba(0.05, 0.95, 0.22, 0.85)
COLOR_INFLATED = Rgba(0.9, 0.25, 1.0, 0.30)
COLOR_TEXT = Rgba(0.95, 0.95, 0.95, 1.0)


def scan_point_layers_from_points(points, config):
    raw_points = tuple(lidar_obstacle_map.finite_base_points(points))
    roi_points = tuple(
        point
        for point in raw_points
        if lidar_obstacle_map.base_point_passes_roi(point, config)
    )
    return ScanPointLayers(raw_points=raw_points, roi_points=roi_points)


def scan_point_layers_from_scan(scan, config):
    points = lidar_obstacle_map.scan_ranges_to_base_points(
        scan.ranges,
        scan.angle_min,
        scan.angle_increment,
        scan.range_min,
        scan.range_max,
    )
    return scan_point_layers_from_points(points, config)


def classify_observation_cells(
    occupancy_map,
    observations: Sequence[
        lidar_obstacle_map.MapFrameObservation | lidar_obstacle_map.GridCellObservation
    ],
    wall_band_cells=None,
):
    wall_band_cells = set(wall_band_cells or ())
    accepted_cells = set()
    rejected_static_cells = set()
    rejected_wall_band_cells = set()
    rejected_invalid_range = 0
    rejected_bounds = 0
    total = 0

    for observation in observations:
        total += 1
        cell = lidar_obstacle_map.observation_to_cell(occupancy_map, observation)
        if cell is None:
            rejected_invalid_range += 1
            continue
        if not planner.in_bounds(occupancy_map, cell):
            rejected_bounds += 1
            continue
        state = occupancy_map.cells[cell[1]][cell[0]]
        if state != planner.CELL_FREE:
            rejected_static_cells.add(cell)
            continue
        if cell in wall_band_cells:
            rejected_wall_band_cells.add(cell)
            continue
        accepted_cells.add(cell)

    return ObservationCellLayers(
        total_observations=total,
        accepted_cells=frozenset(accepted_cells),
        rejected_static_cells=frozenset(rejected_static_cells),
        rejected_wall_band_cells=frozenset(rejected_wall_band_cells),
        rejected_invalid_range=rejected_invalid_range,
        rejected_bounds=rejected_bounds,
    )


def obstacle_config_from_args(args):
    return lidar_obstacle_map.ObstacleOverlayConfig(
        forward_distance_m=args.obstacle_forward_distance_m,
        forward_half_width_m=args.obstacle_forward_half_width_m,
        angle_window_deg=args.obstacle_angle_window_deg,
        min_range_m=args.obstacle_min_range_m,
        robot_footprint_radius_m=args.robot_footprint_radius_m,
        min_cluster_size=args.obstacle_min_cluster_size,
        min_cluster_width_m=args.obstacle_min_cluster_width_m,
        inflate_radius_m=args.obstacle_inflate_radius_m,
    )


def run_local_config_from_args(args):
    return lidar_obstacle_map.RunLocalMapConfig(
        min_hit_count=args.run_local_map_min_hit_count,
        inflation_radius_m=args.obstacle_inflate_radius_m,
        robot_footprint_radius_m=args.robot_footprint_radius_m,
        clearance_margin_m=args.run_local_map_clearance_margin_m,
        static_wall_exclusion_radius_m=args.run_local_map_clearance_margin_m,
        min_used_points=args.run_local_map_min_used_points,
        max_rejected_ratio=args.run_local_map_max_rejected_ratio,
        max_updates=args.run_local_map_max_updates,
    )


def point_msg(x, y, z=0.0):
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def apply_marker_common(marker, frame_id, stamp, namespace, marker_id, color, lifetime_sec):
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
    if Duration is not None and lifetime_sec > 0.0:
        marker.lifetime = Duration(seconds=float(lifetime_sec)).to_msg()


def delete_all_marker():
    marker = Marker()
    marker.action = Marker.DELETEALL
    return marker


def point_layer_marker(
    frame_id,
    stamp,
    namespace,
    marker_id,
    points,
    color,
    scale_m,
    lifetime_sec,
    z=0.0,
):
    if not points:
        return None
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, color, lifetime_sec)
    marker.type = Marker.POINTS
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.points = [point_msg(point.x, point.y, z) for point in points]
    return marker


def cell_layer_marker(
    occupancy_map,
    frame_id,
    stamp,
    namespace,
    marker_id,
    cells,
    color,
    lifetime_sec,
    z=0.0,
    height_m=0.02,
):
    if not cells:
        return None
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, color, lifetime_sec)
    marker.type = Marker.CUBE_LIST
    marker.scale.x = occupancy_map.metadata.resolution
    marker.scale.y = occupancy_map.metadata.resolution
    marker.scale.z = height_m
    marker.points = [
        point_msg(*planner.grid_to_world(cell[0], cell[1], occupancy_map.metadata), z)
        for cell in sorted(cells)
    ]
    return marker


def roi_boundary_marker(frame_id, stamp, config, lifetime_sec):
    marker = Marker()
    apply_marker_common(
        marker,
        frame_id,
        stamp,
        "obstacle_roi_boundary",
        20,
        COLOR_ROI_BOUNDARY,
        lifetime_sec,
    )
    marker.type = Marker.LINE_STRIP
    marker.scale.x = 0.01
    half_width = config.forward_half_width_m
    x_min = max(0.0, config.min_range_m, config.robot_footprint_radius_m)
    x_max = config.forward_distance_m
    marker.points = [
        point_msg(x_min, -half_width, 0.02),
        point_msg(x_max, -half_width, 0.02),
        point_msg(x_max, half_width, 0.02),
        point_msg(x_min, half_width, 0.02),
        point_msg(x_min, -half_width, 0.02),
    ]
    return marker


def text_marker(frame_id, stamp, text, lifetime_sec):
    if not text:
        return None
    marker = Marker()
    apply_marker_common(
        marker,
        frame_id,
        stamp,
        "obstacle_debug_text",
        30,
        COLOR_TEXT,
        lifetime_sec,
    )
    marker.type = Marker.TEXT_VIEW_FACING
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.45
    marker.scale.z = 0.08
    marker.text = text
    return marker


def append_marker(markers, marker):
    if marker is not None:
        markers.append(marker)


def marker_stamp_from_scan(node, scan):
    stamp = getattr(getattr(scan, "header", None), "stamp", None)
    if stamp is not None and (getattr(stamp, "sec", 0) or getattr(stamp, "nanosec", 0)):
        return stamp
    return node.get_clock().now().to_msg()


def debug_text(scan_layers, cell_layers, update_diag, lookup_mode, error_text):
    lines = [
        f"scan raw={len(scan_layers.raw_points)} roi={len(scan_layers.roi_points)}",
    ]
    if cell_layers is not None:
        lines.append(
            "current cells "
            f"accepted={len(cell_layers.accepted_cells)} "
            f"static={len(cell_layers.rejected_static_cells)} "
            f"wall={len(cell_layers.rejected_wall_band_cells)} "
            f"bounds={cell_layers.rejected_bounds}"
        )
    if update_diag is not None:
        status = "accepted" if update_diag.update_accepted else update_diag.update_rejected_reason
        lines.append(
            "run-local "
            f"status={status} "
            f"updates={update_diag.update_count} "
            f"confirmed={update_diag.confirmed_raw_cells} "
            f"inflated={update_diag.inflated_cells}"
        )
    if lookup_mode:
        lines.append(f"tf={lookup_mode}")
    if error_text:
        lines.append(f"map layer unavailable: {error_text}")
    return "\n".join(lines)


class LidarObstacleDebugViz(Node):
    def __init__(self, args):
        super().__init__("mii_amr_lidar_obstacle_debug_viz")
        self.args = args
        self.obstacle_config = obstacle_config_from_args(args)
        self.static_map = None
        self.run_local_map = None
        self.last_warn_sec = 0.0

        if not args.no_static_map:
            self.static_map = planner.load_occupancy_map(args.static_map)
            self.run_local_map = lidar_obstacle_map.RunLocalObstacleMap(
                self.static_map,
                run_local_config_from_args(args),
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.get_logger().info(
                f"Loaded static map {args.static_map} for run-local obstacle cells."
            )
        else:
            self.tf_buffer = None
            self.tf_listener = None
            self.get_logger().info("Static map disabled; publishing scan-frame markers only.")

        self.marker_pub = self.create_publisher(MarkerArray, args.marker_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Publishing LiDAR obstacle diagnostics from {args.scan_topic} "
            f"to {args.marker_topic}."
        )

    def warn_throttled(self, message):
        now = time.time()
        if now - self.last_warn_sec >= DEFAULT_WARN_INTERVAL_SEC:
            self.get_logger().warn(message)
            self.last_warn_sec = now

    def scan_callback(self, scan):
        stamp = marker_stamp_from_scan(self, scan)
        scan_frame = getattr(scan.header, "frame_id", "") or self.args.scan_frame_fallback
        scan_layers = scan_point_layers_from_scan(scan, self.obstacle_config)
        markers = [delete_all_marker()]
        cell_layers = None
        update_diag = None
        lookup_mode = ""
        error_text = ""

        if self.args.publish_raw_scan:
            append_marker(
                markers,
                point_layer_marker(
                    scan_frame,
                    stamp,
                    "raw_scan_points",
                    1,
                    scan_layers.raw_points,
                    COLOR_RAW_SCAN,
                    self.args.scan_point_scale_m,
                    self.args.marker_lifetime_sec,
                ),
            )
        append_marker(
            markers,
            point_layer_marker(
                scan_frame,
                stamp,
                "roi_candidate_points",
                2,
                scan_layers.roi_points,
                COLOR_ROI_SCAN,
                self.args.roi_point_scale_m,
                self.args.marker_lifetime_sec,
                z=0.01,
            ),
        )
        append_marker(
            markers,
            roi_boundary_marker(
                scan_frame,
                stamp,
                self.obstacle_config,
                self.args.marker_lifetime_sec,
            ),
        )

        if self.static_map is not None and self.run_local_map is not None:
            try:
                transform, lookup_mode = replan_runtime.lookup_map_from_scan_transform(
                    self,
                    self.args,
                    scan,
                )
                base_filter_config = (
                    self.obstacle_config
                    if self.args.map_update_mode == "forward"
                    else None
                )
                observations = replan_runtime.scan_points_to_map_observations(
                    scan,
                    transform,
                    base_filter_config,
                )
                cell_layers = classify_observation_cells(
                    self.static_map,
                    observations,
                    self.run_local_map._wall_band_cells,
                )
                update_diag = self.run_local_map.add_observations(
                    lidar_obstacle_map.ObservationBatch(
                        observations,
                        source="lidar_obstacle_debug_viz",
                    )
                )
            except Exception as exc:
                error_text = str(exc)
                self.warn_throttled(error_text)
            else:
                append_marker(
                    markers,
                    cell_layer_marker(
                        self.static_map,
                        self.args.map_frame,
                        stamp,
                        "current_accepted_cells",
                        3,
                        cell_layers.accepted_cells,
                        COLOR_CURRENT_ACCEPTED,
                        self.args.marker_lifetime_sec,
                        z=0.015,
                    ),
                )
                append_marker(
                    markers,
                    cell_layer_marker(
                        self.static_map,
                        self.args.map_frame,
                        stamp,
                        "rejected_static_cells",
                        4,
                        cell_layers.rejected_static_cells,
                        COLOR_REJECTED_STATIC,
                        self.args.marker_lifetime_sec,
                        z=0.025,
                    ),
                )
                append_marker(
                    markers,
                    cell_layer_marker(
                        self.static_map,
                        self.args.map_frame,
                        stamp,
                        "rejected_wall_band_cells",
                        5,
                        cell_layers.rejected_wall_band_cells,
                        COLOR_REJECTED_WALL_BAND,
                        self.args.marker_lifetime_sec,
                        z=0.03,
                    ),
                )
                append_marker(
                    markers,
                    cell_layer_marker(
                        self.static_map,
                        self.args.map_frame,
                        stamp,
                        "confirmed_raw_obstacle_cells",
                        6,
                        self.run_local_map.confirmed_raw_cells,
                        COLOR_CONFIRMED_RAW,
                        self.args.marker_lifetime_sec,
                        z=0.045,
                    ),
                )
                append_marker(
                    markers,
                    cell_layer_marker(
                        self.static_map,
                        self.args.map_frame,
                        stamp,
                        "inflated_obstacle_cells",
                        7,
                        self.run_local_map.inflated_obstacle_cells,
                        COLOR_INFLATED,
                        self.args.marker_lifetime_sec,
                        z=0.005,
                    ),
                )

        if self.args.publish_text:
            append_marker(
                markers,
                text_marker(
                    scan_frame,
                    stamp,
                    debug_text(scan_layers, cell_layers, update_diag, lookup_mode, error_text),
                    self.args.marker_lifetime_sec,
                ),
            )

        self.marker_pub.publish(MarkerArray(markers=markers))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Publish RViz MarkerArray diagnostics for /scan obstacle detection."
    )
    parser.add_argument("--scan-topic", default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--marker-topic", default=DEFAULT_MARKER_TOPIC)
    parser.add_argument("--map-frame", default=DEFAULT_MAP_FRAME)
    parser.add_argument("--scan-frame-fallback", default="base_scan")
    parser.add_argument("--static-map", default=DEFAULT_STATIC_MAP, type=Path)
    parser.add_argument(
        "--no-static-map",
        action="store_true",
        help="Publish only scan-frame raw/ROI markers; skip map-cell obstacle layers.",
    )
    parser.add_argument(
        "--allow-latest-tf-fallback",
        dest="allow_latest_tf_replan_fallback",
        action="store_true",
        help="Use latest TF if timestamped map<-scan lookup is unavailable.",
    )
    parser.add_argument(
        "--map-update-mode",
        choices=["forward", "full"],
        default="forward",
        help="Use the forward obstacle ROI or the full scan for run-local map cells.",
    )
    parser.add_argument("--obstacle-forward-distance-m", default=0.55, type=float)
    parser.add_argument("--obstacle-forward-half-width-m", default=0.18, type=float)
    parser.add_argument("--obstacle-angle-window-deg", default=45.0, type=float)
    parser.add_argument("--obstacle-min-range-m", default=0.12, type=float)
    parser.add_argument("--robot-footprint-radius-m", default=0.18, type=float)
    parser.add_argument("--obstacle-min-cluster-size", default=3, type=int)
    parser.add_argument("--obstacle-min-cluster-width-m", default=0.05, type=float)
    parser.add_argument("--obstacle-inflate-radius-m", default=0.22, type=float)
    parser.add_argument("--run-local-map-min-hit-count", default=2, type=int)
    parser.add_argument("--run-local-map-min-used-points", default=3, type=int)
    parser.add_argument("--run-local-map-max-rejected-ratio", default=0.90, type=float)
    parser.add_argument("--run-local-map-clearance-margin-m", default=0.04, type=float)
    parser.add_argument(
        "--run-local-map-max-updates",
        default=DEBUG_MAX_UPDATES,
        type=int,
        help="High default keeps the diagnostic live; set to 3 to mirror follower limits.",
    )
    parser.add_argument("--marker-lifetime-sec", default=DEFAULT_MARKER_LIFETIME_SEC, type=float)
    parser.add_argument("--scan-point-scale-m", default=0.018, type=float)
    parser.add_argument("--roi-point-scale-m", default=0.035, type=float)
    parser.add_argument("--no-raw-scan", dest="publish_raw_scan", action="store_false")
    parser.add_argument("--no-text", dest="publish_text", action="store_false")
    parser.set_defaults(publish_raw_scan=True, publish_text=True)
    return parser


def validate_args(parser, args):
    if args.run_local_map_min_hit_count < 1:
        parser.error("--run-local-map-min-hit-count must be >= 1")
    if args.run_local_map_min_used_points < 1:
        parser.error("--run-local-map-min-used-points must be >= 1")
    if not (0.0 <= args.run_local_map_max_rejected_ratio <= 1.0):
        parser.error("--run-local-map-max-rejected-ratio must be in [0, 1]")
    if args.marker_lifetime_sec < 0.0:
        parser.error("--marker-lifetime-sec must be >= 0")
    if args.obstacle_forward_distance_m <= 0.0:
        parser.error("--obstacle-forward-distance-m must be > 0")
    if args.obstacle_forward_half_width_m <= 0.0:
        parser.error("--obstacle-forward-half-width-m must be > 0")
    if args.obstacle_inflate_radius_m < 0.0:
        parser.error("--obstacle-inflate-radius-m must be >= 0")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if rclpy is None:
        raise SystemExit("ROS2 Python packages are required to run this RViz publisher.")

    rclpy.init(args=None)
    node = LidarObstacleDebugViz(args)
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
