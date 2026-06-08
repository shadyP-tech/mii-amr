#!/usr/bin/env python3
"""
Render presentation PNGs for Aufgabe 03 Sequence B:

    planned route -> raw LiDAR points -> forward ROI -> accepted/rejected cells
    -> run-local obstacle overlay -> old path blocked -> replanned path.

The frames use the real saved arena map, the default Aufgabe 03 A* route, and a
deterministic synthetic LiDAR obstacle on the planned path. The output PNGs have
transparent canvas outside the map so they can be layered in Keynote.

Use the bundled Codex Python runtime because it includes Pillow:

    /Users/stephpark/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
        scripts/aufgabe03/render_sequence_b_lidar_replan_layers.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import lidar_obstacle_map as lidar
import map_path_planner as planner
import render_sequence_a_semantic_layers as seq_a


DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_DIR = Path("docs/presentation/aufgabe03_sequence_b_lidar_replan_layers")
DEFAULT_START = (0.005, -0.015)
DEFAULT_GOAL = (1.005, 0.285)
DEFAULT_ROBOT_POSE = lidar.Pose2D(0.555, -0.015, 45.0)
DEFAULT_OBSTACLE_BASE_X = 0.28
DEFAULT_OBSTACLE_BASE_Y = 0.0
DEFAULT_OBSTACLE_WIDTH_M = 0.12
DEFAULT_OBSTACLE_POINTS = 9
ROBOT_RING_RADIUS_PX = 22.0

COLORS = {
    **seq_a.COLORS,
    "scan": (37, 99, 235),
    "roi": (124, 58, 237, 46),
    "roi_edge": (124, 58, 237, 150),
    "accepted": (22, 163, 74),
    "accepted_soft": (187, 247, 208, 190),
    "rejected": (239, 68, 68),
    "rejected_soft": (254, 202, 202, 190),
    "raw_obstacle": (22, 163, 74),
    "run_local_inflated": (168, 85, 247, 105),
    "old_path": (107, 114, 128),
    "blocked": (239, 68, 68),
    "new_path": (37, 99, 235),
    "robot": (15, 118, 110),
    "robot_dark": (17, 24, 39),
    "robot_halo": (45, 212, 191),
    "lidar_shadow": (245, 158, 11, 58),
    "obstacle_hit": (249, 115, 22),
    "obstacle_hit_alt": (239, 68, 68),
}

SCAN_PALETTE = [
    (37, 99, 235),
    (6, 182, 212),
    (124, 58, 237),
    (236, 72, 153),
    (245, 158, 11),
    (34, 197, 94),
]


@dataclass(frozen=True)
class SequenceBData:
    occupancy_map: planner.OccupancyMap
    raw_image: planner.PgmImage
    static_plan: planner.PlanResult
    current_plan: planner.PlanResult
    replan_result: lidar.ReplanResult
    robot_pose: lidar.Pose2D
    goal_pose: lidar.Pose2D
    obstacle_points: list[lidar.BaseFramePoint]
    wall_scan_points: list[lidar.BaseFramePoint]
    rejected_points: list[lidar.BaseFramePoint]
    roi_points: list[lidar.BaseFramePoint]
    accepted_cells: set[tuple[int, int]]
    rejected_cells: set[tuple[int, int]]
    raw_obstacle_cells: set[tuple[int, int]]
    inflated_obstacle_cells: set[tuple[int, int]]
    blocked_old_path_cells: set[tuple[int, int]]
    obstacle_config: lidar.ObstacleOverlayConfig


def base_to_canvas(
    occupancy_map: planner.OccupancyMap,
    robot_pose: lidar.Pose2D,
    point: lidar.BaseFramePoint,
) -> tuple[float, float]:
    world_x, world_y = lidar.base_point_to_map(point, robot_pose)
    return seq_a.world_to_canvas(occupancy_map, (world_x, world_y))


def pose_to_canvas(
    occupancy_map: planner.OccupancyMap,
    pose: lidar.Pose2D,
) -> tuple[float, float]:
    return seq_a.world_to_canvas(occupancy_map, (pose.x, pose.y))


def grid_cell_from_base_point(
    occupancy_map: planner.OccupancyMap,
    robot_pose: lidar.Pose2D,
    point: lidar.BaseFramePoint,
) -> tuple[int, int] | None:
    map_x, map_y = lidar.base_point_to_map(point, robot_pose)
    cell = planner.world_to_grid(map_x, map_y, occupancy_map.metadata)
    if not planner.in_bounds(occupancy_map, cell):
        return None
    return cell


def cast_wall_scan_points(
    occupancy_map: planner.OccupancyMap,
    robot_pose: lidar.Pose2D,
) -> list[lidar.BaseFramePoint]:
    points: list[lidar.BaseFramePoint] = []
    for angle_deg in range(-135, 136, 10):
        absolute_yaw = math.radians(robot_pose.yaw_deg + angle_deg)
        for step in range(4, 90):
            distance_m = step * occupancy_map.metadata.resolution
            map_x = robot_pose.x + math.cos(absolute_yaw) * distance_m
            map_y = robot_pose.y + math.sin(absolute_yaw) * distance_m
            cell = planner.world_to_grid(map_x, map_y, occupancy_map.metadata)
            if not planner.in_bounds(occupancy_map, cell):
                break
            if occupancy_map.cells[cell[1]][cell[0]] != planner.CELL_FREE:
                points.append(lidar.map_point_to_base(map_x, map_y, robot_pose))
                break
    return points


def explicit_rejected_points() -> list[lidar.BaseFramePoint]:
    return [
        lidar.BaseFramePoint(0.08, -0.03),
        lidar.BaseFramePoint(0.10, 0.04),
        lidar.BaseFramePoint(0.34, 0.31),
        lidar.BaseFramePoint(0.37, -0.30),
        lidar.BaseFramePoint(0.72, 0.02),
    ]


def accepted_and_rejected_cells(
    occupancy_map: planner.OccupancyMap,
    robot_pose: lidar.Pose2D,
    points: list[lidar.BaseFramePoint],
    config: lidar.ObstacleOverlayConfig,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[lidar.BaseFramePoint]]:
    accepted: set[tuple[int, int]] = set()
    rejected: set[tuple[int, int]] = set()
    roi_points: list[lidar.BaseFramePoint] = []
    for point in points:
        cell = grid_cell_from_base_point(occupancy_map, robot_pose, point)
        if cell is None:
            continue
        if not lidar.base_point_passes_roi(point, config):
            rejected.add(cell)
            continue
        roi_points.append(point)
        if occupancy_map.cells[cell[1]][cell[0]] == planner.CELL_FREE:
            accepted.add(cell)
        else:
            rejected.add(cell)
    return accepted, rejected, roi_points


def current_remaining_plan(
    occupancy_map: planner.OccupancyMap,
    robot_pose: lidar.Pose2D,
    goal_pose: lidar.Pose2D,
) -> planner.PlanResult:
    plan, _blocked, _inflation_cells = planner.plan_path(
        occupancy_map,
        (robot_pose.x, robot_pose.y),
        (goal_pose.x, goal_pose.y),
        inflate_radius_m=planner.DEFAULT_INFLATE_RADIUS_M,
        snap_radius_m=planner.DEFAULT_SNAP_RADIUS_M,
    )
    return plan


def build_sequence_data(args: argparse.Namespace) -> SequenceBData:
    occupancy_map = planner.load_occupancy_map(args.map)
    raw_image = planner.read_pgm(occupancy_map.metadata.image_path)
    static_plan, _static_blocked, _inflation_cells = planner.plan_path(
        occupancy_map,
        tuple(args.start),
        tuple(args.goal),
        inflate_radius_m=planner.DEFAULT_INFLATE_RADIUS_M,
        snap_radius_m=planner.DEFAULT_SNAP_RADIUS_M,
    )

    robot_pose = lidar.Pose2D(args.robot_pose[0], args.robot_pose[1], args.robot_pose[2])
    goal_pose = lidar.Pose2D(args.goal[0], args.goal[1], 0.0)
    current_plan = current_remaining_plan(occupancy_map, robot_pose, goal_pose)
    obstacle_points = lidar.synthetic_obstacle_points(
        args.obstacle_base_x,
        args.obstacle_base_y,
        args.obstacle_width_m,
        args.obstacle_points,
    )
    wall_scan_points = cast_wall_scan_points(occupancy_map, robot_pose)
    rejected_points = explicit_rejected_points()
    all_points = wall_scan_points + obstacle_points + rejected_points
    obstacle_config = lidar.ObstacleOverlayConfig(
        forward_distance_m=0.60,
        forward_half_width_m=0.25,
        angle_window_deg=50.0,
        inflate_radius_m=0.15,
        planner_inflate_radius_m=0.0,
        max_replan_path_length_ratio=5.0,
    )
    accepted_cells, rejected_cells, roi_points = accepted_and_rejected_cells(
        occupancy_map,
        robot_pose,
        all_points,
        obstacle_config,
    )
    replan_result = lidar.build_replan_result(
        occupancy_map,
        obstacle_points,
        robot_pose,
        goal_pose,
        "sequence_b_synthetic",
        output_dir=args.output_dir / "artifacts",
        config=obstacle_config,
    )
    if not replan_result.success:
        raise RuntimeError(f"Could not build synthetic replan: {replan_result.reason}")

    raw_obstacle_cells = {
        cell
        for point in obstacle_points
        if (cell := grid_cell_from_base_point(occupancy_map, robot_pose, point)) is not None
        and occupancy_map.cells[cell[1]][cell[0]] == planner.CELL_FREE
    }
    inflated_obstacle_cells = set(replan_result.inflated_obstacle_cells)
    blocked_old_path_cells = set(current_plan.path).intersection(inflated_obstacle_cells)

    return SequenceBData(
        occupancy_map=occupancy_map,
        raw_image=raw_image,
        static_plan=static_plan,
        current_plan=current_plan,
        replan_result=replan_result,
        robot_pose=robot_pose,
        goal_pose=goal_pose,
        obstacle_points=obstacle_points,
        wall_scan_points=wall_scan_points,
        rejected_points=rejected_points,
        roi_points=roi_points,
        accepted_cells=accepted_cells,
        rejected_cells=rejected_cells,
        raw_obstacle_cells=raw_obstacle_cells,
        inflated_obstacle_cells=inflated_obstacle_cells,
        blocked_old_path_cells=blocked_old_path_cells,
        obstacle_config=obstacle_config,
    )


def draw_map(draw: ImageDraw.ImageDraw, data: SequenceBData) -> None:
    seq_a.draw_base_map(
        draw,
        data.occupancy_map,
        data.raw_image,
        "inflated",
        set(),
    )
    seq_a.draw_grid_overlay(draw, data.occupancy_map)


def draw_robot(draw: ImageDraw.ImageDraw, data: SequenceBData) -> None:
    cx, cy = pose_to_canvas(data.occupancy_map, data.robot_pose)
    yaw = math.radians(data.robot_pose.yaw_deg)
    radius = ROBOT_RING_RADIUS_PX
    draw.ellipse(
        (cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3),
        outline=COLORS["surface"],
        width=6,
    )
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=(*COLORS["robot_halo"], 235),
        width=4,
    )

    tip = (cx + math.cos(yaw) * 19.0, cy - math.sin(yaw) * 19.0)
    tail = (cx - math.cos(yaw) * 9.0, cy + math.sin(yaw) * 9.0)
    left = (
        tail[0] + math.cos(yaw + math.pi / 2.0) * 10.0,
        tail[1] - math.sin(yaw + math.pi / 2.0) * 10.0,
    )
    right = (
        tail[0] + math.cos(yaw - math.pi / 2.0) * 10.0,
        tail[1] - math.sin(yaw - math.pi / 2.0) * 10.0,
    )
    draw.polygon((tip, left, right), fill=COLORS["surface"])
    draw.polygon((tip, left, right), fill=COLORS["robot_dark"], outline=COLORS["robot_halo"])


def draw_goal(draw: ImageDraw.ImageDraw, data: SequenceBData) -> None:
    cx, cy = pose_to_canvas(data.occupancy_map, data.goal_pose)
    draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=COLORS["goal"], outline=COLORS["surface"], width=4)


def draw_path(
    draw: ImageDraw.ImageDraw,
    data: SequenceBData,
    cells: list[tuple[int, int]],
    color: tuple[int, int, int],
    *,
    width: int = 9,
    alpha: int = 255,
) -> None:
    if len(cells) < 2:
        return
    points = [seq_a.cell_center(data.occupancy_map, cell) for cell in cells]
    draw.line(points, fill=(*color, alpha), width=width, joint="curve")
    for point in points:
        radius = max(4, width // 2)
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=(*color, alpha),
        )


def draw_scan_points(
    draw: ImageDraw.ImageDraw,
    data: SequenceBData,
    points: list[lidar.BaseFramePoint],
    color: tuple[int, int, int],
    *,
    rays: bool = False,
    radius: int = 4,
    alpha: int = 230,
    colorful: bool = False,
    ray_alpha: int | None = None,
    ray_width: int | None = None,
) -> None:
    robot_canvas = pose_to_canvas(data.occupancy_map, data.robot_pose)
    for index, point in enumerate(points):
        px, py = base_to_canvas(data.occupancy_map, data.robot_pose, point)
        point_color = color
        if colorful:
            point_color = SCAN_PALETTE[index % len(SCAN_PALETTE)]
        if rays:
            dx = px - robot_canvas[0]
            dy = py - robot_canvas[1]
            distance = math.hypot(dx, dy)
            if distance > 1e-9:
                start = (
                    robot_canvas[0] + dx / distance * ROBOT_RING_RADIUS_PX,
                    robot_canvas[1] + dy / distance * ROBOT_RING_RADIUS_PX,
                )
            else:
                start = robot_canvas
            draw.line(
                (start, (px, py)),
                fill=(*point_color, ray_alpha if ray_alpha is not None else (145 if colorful else 70)),
                width=ray_width if ray_width is not None else (3 if colorful else 2),
            )
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*point_color, alpha))


def point_angle_rad(point: lidar.BaseFramePoint) -> float:
    return math.atan2(point.y, point.x)


def point_range_m(point: lidar.BaseFramePoint) -> float:
    return math.hypot(point.x, point.y)


def obstacle_angle_bounds(
    points: list[lidar.BaseFramePoint],
    margin_deg: float = 8.0,
) -> tuple[float, float]:
    angles = [point_angle_rad(point) for point in points]
    margin = math.radians(margin_deg)
    return min(angles) - margin, max(angles) + margin


def obstacle_min_range(points: list[lidar.BaseFramePoint]) -> float:
    return min(point_range_m(point) for point in points)


def point_is_behind_obstacle_shadow(
    point: lidar.BaseFramePoint,
    obstacle_points: list[lidar.BaseFramePoint],
) -> bool:
    min_angle, max_angle = obstacle_angle_bounds(obstacle_points)
    return (
        min_angle <= point_angle_rad(point) <= max_angle
        and point_range_m(point) > obstacle_min_range(obstacle_points) + 0.04
    )


def visible_wall_scan_points(data: SequenceBData) -> list[lidar.BaseFramePoint]:
    return [
        point
        for point in data.wall_scan_points
        if not point_is_behind_obstacle_shadow(point, data.obstacle_points)
    ]


def visible_non_obstacle_scan_points(data: SequenceBData) -> list[lidar.BaseFramePoint]:
    return visible_wall_scan_points(data) + [
        point
        for point in data.rejected_points
        if not point_is_behind_obstacle_shadow(point, data.obstacle_points)
    ]


def draw_lidar_shadow(draw: ImageDraw.ImageDraw, data: SequenceBData, *, alpha: int = 58) -> None:
    min_angle, max_angle = obstacle_angle_bounds(data.obstacle_points, margin_deg=6.0)
    near = obstacle_min_range(data.obstacle_points)
    far = max(0.95, near + 0.60)
    corners = [
        lidar.BaseFramePoint(math.cos(min_angle) * near, math.sin(min_angle) * near),
        lidar.BaseFramePoint(math.cos(max_angle) * near, math.sin(max_angle) * near),
        lidar.BaseFramePoint(math.cos(max_angle) * far, math.sin(max_angle) * far),
        lidar.BaseFramePoint(math.cos(min_angle) * far, math.sin(min_angle) * far),
    ]
    polygon = [base_to_canvas(data.occupancy_map, data.robot_pose, corner) for corner in corners]
    draw.polygon(polygon, fill=(*COLORS["lidar_shadow"][:3], alpha))


def draw_obstacle_hit_rays(draw: ImageDraw.ImageDraw, data: SequenceBData) -> None:
    robot_canvas = pose_to_canvas(data.occupancy_map, data.robot_pose)
    hit_colors = [
        COLORS["obstacle_hit"],
        COLORS["obstacle_hit_alt"],
        (245, 158, 11),
    ]
    for index, point in enumerate(data.obstacle_points):
        px, py = base_to_canvas(data.occupancy_map, data.robot_pose, point)
        dx = px - robot_canvas[0]
        dy = py - robot_canvas[1]
        distance = math.hypot(dx, dy)
        if distance > 1e-9:
            start = (
                robot_canvas[0] + dx / distance * ROBOT_RING_RADIUS_PX,
                robot_canvas[1] + dy / distance * ROBOT_RING_RADIUS_PX,
            )
        else:
            start = robot_canvas
        color = hit_colors[index % len(hit_colors)]
        draw.line((start, (px, py)), fill=(*color, 235), width=5)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(*color, 255), outline=COLORS["surface"], width=2)


def draw_raw_lidar_detection_layer(
    draw: ImageDraw.ImageDraw,
    data: SequenceBData,
    *,
    shadow_alpha: int = 58,
    scan_alpha: int = 230,
    ray_alpha: int = 145,
    ray_width: int = 3,
) -> None:
    draw_lidar_shadow(draw, data, alpha=shadow_alpha)
    draw_scan_points(
        draw,
        data,
        visible_non_obstacle_scan_points(data),
        COLORS["scan"],
        rays=True,
        radius=4,
        alpha=scan_alpha,
        colorful=True,
        ray_alpha=ray_alpha,
        ray_width=ray_width,
    )
    draw_obstacle_hit_rays(draw, data)


def draw_roi_points(draw: ImageDraw.ImageDraw, data: SequenceBData) -> None:
    for point in data.roi_points:
        px, py = base_to_canvas(data.occupancy_map, data.robot_pose, point)
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), outline=COLORS["surface"], width=4)
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=COLORS["roi_edge"], width=3)


def draw_roi(draw: ImageDraw.ImageDraw, data: SequenceBData, *, foreground: bool = False) -> None:
    config = data.obstacle_config
    corners = [
        lidar.BaseFramePoint(config.min_range_m, -config.forward_half_width_m),
        lidar.BaseFramePoint(config.forward_distance_m, -config.forward_half_width_m),
        lidar.BaseFramePoint(config.forward_distance_m, config.forward_half_width_m),
        lidar.BaseFramePoint(config.min_range_m, config.forward_half_width_m),
    ]
    polygon = [base_to_canvas(data.occupancy_map, data.robot_pose, corner) for corner in corners]
    if not foreground:
        draw.polygon(polygon, fill=COLORS["roi"], outline=COLORS["roi_edge"])
        return

    closed_polygon = polygon + [polygon[0]]
    draw.polygon(polygon, fill=(*COLORS["roi"][:3], 70))
    draw.line(closed_polygon, fill=(*COLORS["surface"], 230), width=10, joint="curve")
    draw.line(closed_polygon, fill=(*COLORS["roi_edge"][:3], 245), width=5, joint="curve")


def draw_cells(
    draw: ImageDraw.ImageDraw,
    data: SequenceBData,
    cells: set[tuple[int, int]],
    color: tuple[int, int, int] | tuple[int, int, int, int],
    *,
    pad: int = 1,
) -> None:
    for cell in cells:
        left, top, right, bottom = seq_a.cell_rect(data.occupancy_map, cell)
        draw.rectangle((left + pad, top + pad, right - pad, bottom - pad), fill=color)


def free_space_cells(data: SequenceBData, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        cell
        for cell in cells
        if data.occupancy_map.cells[cell[1]][cell[0]] == planner.CELL_FREE
    }


def draw_blocked_crosses(draw: ImageDraw.ImageDraw, data: SequenceBData) -> None:
    for cell in data.blocked_old_path_cells:
        cx, cy = seq_a.cell_center(data.occupancy_map, cell)
        draw.line((cx - 9, cy - 9, cx + 9, cy + 9), fill=COLORS["blocked"], width=4)
        draw.line((cx - 9, cy + 9, cx + 9, cy - 9), fill=COLORS["blocked"], width=4)


def render_frame(data: SequenceBData, layer: str) -> Image.Image:
    image = Image.new("RGBA", (seq_a.CANVAS_W, seq_a.CANVAS_H), COLORS["transparent"])
    draw = ImageDraw.Draw(image, "RGBA")
    draw_map(draw, data)
    foreground_robot = layer in {
        "planned_route",
        "raw_scan",
        "roi",
        "accepted_rejected",
        "overlay",
        "old_path_blocked",
        "replanned_path",
        "validation",
    }

    if foreground_robot:
        draw_path(draw, data, data.static_plan.path, COLORS["path"], width=9, alpha=235)

    if layer == "raw_scan":
        draw_raw_lidar_detection_layer(draw, data)

    if layer == "roi":
        draw_raw_lidar_detection_layer(draw, data, shadow_alpha=10, scan_alpha=190, ray_alpha=115, ray_width=3)
        draw_roi(draw, data, foreground=True)
        draw_roi_points(draw, data)

    if layer == "accepted_rejected":
        draw_roi(draw, data)
        draw_cells(draw, data, free_space_cells(data, data.rejected_cells), COLORS["rejected_soft"])
        draw_cells(draw, data, data.accepted_cells, COLORS["accepted_soft"])
        draw_scan_points(draw, data, data.rejected_points, COLORS["rejected"], radius=5)
        draw_scan_points(draw, data, data.obstacle_points, COLORS["accepted"], radius=5)

    if layer == "overlay":
        draw_cells(draw, data, data.inflated_obstacle_cells, COLORS["run_local_inflated"])
        draw_cells(draw, data, data.raw_obstacle_cells, COLORS["raw_obstacle"])

    if layer == "old_path_blocked":
        draw_cells(draw, data, data.inflated_obstacle_cells, COLORS["run_local_inflated"])
        draw_cells(draw, data, data.raw_obstacle_cells, COLORS["raw_obstacle"])
        draw_path(draw, data, data.current_plan.path, COLORS["old_path"], width=11, alpha=210)
        draw_blocked_crosses(draw, data)

    if layer == "replanned_path":
        draw_cells(draw, data, data.inflated_obstacle_cells, COLORS["run_local_inflated"])
        draw_cells(draw, data, data.raw_obstacle_cells, COLORS["raw_obstacle"])
        draw_path(draw, data, data.current_plan.path, COLORS["old_path"], width=8, alpha=100)
        draw_path(draw, data, list(data.replan_result.path_cells), COLORS["new_path"], width=10, alpha=255)

    if layer == "validation":
        draw_cells(draw, data, data.inflated_obstacle_cells, COLORS["run_local_inflated"])
        draw_cells(draw, data, data.raw_obstacle_cells, COLORS["raw_obstacle"])
        draw_path(draw, data, list(data.replan_result.path_cells), COLORS["new_path"], width=10, alpha=255)
        for _index, x_m, y_m in data.replan_result.waypoints:
            cx, cy = seq_a.world_to_canvas(data.occupancy_map, (x_m, y_m))
            draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=COLORS["waypoint"], outline=COLORS["surface"], width=3)

    if foreground_robot:
        draw_goal(draw, data)
        draw_robot(draw, data)

    return image


def frame_specs() -> list[tuple[str, str]]:
    return [
        ("001_planned_route_robot.png", "planned_route"),
        ("002_raw_lidar_scan_points.png", "raw_scan"),
        ("003_forward_obstacle_roi.png", "roi"),
        ("004_accepted_rejected_cells.png", "accepted_rejected"),
        ("005_run_local_obstacle_overlay.png", "overlay"),
        ("006_old_path_blocked.png", "old_path_blocked"),
        ("007_replanned_path.png", "replanned_path"),
        ("008_validation_frame.png", "validation"),
    ]


def save_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    thumbs = [
        Image.open(path).convert("RGBA").resize((384, 216), Image.Resampling.LANCZOS)
        for path in frame_paths
    ]
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * 384, rows * 216), COLORS["transparent"])
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * 384
        y = (index // cols) * 216
        sheet.alpha_composite(thumb, (x, y))
    sheet.save(output_path, optimize=True)


def write_manifest(output_dir: Path, frame_paths: list[Path], data: SequenceBData) -> None:
    manifest = {
        "title": "Aufgabe 03 Sequence B LiDAR obstacle and replan layers",
        "map": str(data.occupancy_map.metadata.yaml_path),
        "canvas": {"width": seq_a.CANVAS_W, "height": seq_a.CANVAS_H},
        "robot_pose": {
            "x_m": data.robot_pose.x,
            "y_m": data.robot_pose.y,
            "yaw_deg": data.robot_pose.yaw_deg,
        },
        "goal_pose": {"x_m": data.goal_pose.x, "y_m": data.goal_pose.y},
        "synthetic_obstacle_base_frame": {
            "x_m": DEFAULT_OBSTACLE_BASE_X,
            "y_m": DEFAULT_OBSTACLE_BASE_Y,
            "width_m": DEFAULT_OBSTACLE_WIDTH_M,
            "points": DEFAULT_OBSTACLE_POINTS,
        },
        "accepted_cells": len(data.accepted_cells),
        "raw_obstacle_cells": len(data.raw_obstacle_cells),
        "inflated_obstacle_cells": len(data.inflated_obstacle_cells),
        "blocked_old_path_cells": len(data.blocked_old_path_cells),
        "replanned_path_cells": len(data.replan_result.path_cells),
        "replanned_waypoints": len(data.replan_result.waypoints),
        "frames": [
            {"index": index + 1, "file": str(path.relative_to(output_dir))}
            for index, path in enumerate(frame_paths)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_readme(output_dir: Path) -> None:
    readme = """# Aufgabe 03 Sequence B LiDAR Replan Layers

Transparent, map-only PNG frames for Block B in the Aufgabe 03 deck:

1. Planned route with robot pose
2. Raw LiDAR scan points
3. Forward obstacle ROI
4. Accepted vs rejected cells
5. Run-local obstacle overlay
6. Old path blocked
7. Replanned path
8. Validation frame

The obstacle is deterministic synthetic LiDAR data placed on the existing route
so the sequence can be regenerated without live robot logs.
"""
    (output_dir / "README.md").write_text(readme)


def parse_pose(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("robot pose must be X Y YAW_DEG")
    return float(values[0]), float(values[1]), float(values[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Sequence B LiDAR obstacle/replan PNG layers.")
    parser.add_argument("--map", default=DEFAULT_MAP, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--start", nargs=2, default=DEFAULT_START, type=float, metavar=("X", "Y"))
    parser.add_argument("--goal", nargs=2, default=DEFAULT_GOAL, type=float, metavar=("X", "Y"))
    parser.add_argument(
        "--robot-pose",
        nargs=3,
        default=(DEFAULT_ROBOT_POSE.x, DEFAULT_ROBOT_POSE.y, DEFAULT_ROBOT_POSE.yaw_deg),
        type=float,
        metavar=("X", "Y", "YAW_DEG"),
    )
    parser.add_argument("--obstacle-base-x", default=DEFAULT_OBSTACLE_BASE_X, type=float)
    parser.add_argument("--obstacle-base-y", default=DEFAULT_OBSTACLE_BASE_Y, type=float)
    parser.add_argument("--obstacle-width-m", default=DEFAULT_OBSTACLE_WIDTH_M, type=float)
    parser.add_argument("--obstacle-points", default=DEFAULT_OBSTACLE_POINTS, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    data = build_sequence_data(args)
    frame_paths: list[Path] = []
    for filename, layer in frame_specs():
        image = render_frame(data, layer)
        path = frames_dir / filename
        image.save(path, optimize=True)
        frame_paths.append(path)

    save_contact_sheet(frame_paths, output_dir / "contact_sheet.png")
    write_manifest(output_dir, frame_paths, data)
    write_readme(output_dir)
    print(f"Wrote {len(frame_paths)} PNG frames to {frames_dir}")
    print(f"Wrote contact sheet: {output_dir / 'contact_sheet.png'}")
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
