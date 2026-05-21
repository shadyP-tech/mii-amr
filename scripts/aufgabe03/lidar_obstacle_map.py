#!/usr/bin/env python3
"""
Run-local obstacle overlay and A* replanning helpers.

This module is intentionally stdlib-only and ROS-free. Runtime code may convert
sensor messages into map-frame observations, but this module only sees pure
points/cells, composes a temporary obstacle overlay, writes artifacts, and
replans with the existing A* planner. It does not modify the source map.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import map_path_planner as planner


DEFAULT_STATIC_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_DIR = Path("results/aufgabe03")

CELL_SOURCE_STATIC_OCCUPIED = "static_occupied"
CELL_SOURCE_RUN_LOCAL_RAW = "run_local_raw"
CELL_SOURCE_RUN_LOCAL_INFLATED = "run_local_inflated"
CELL_SOURCE_FREE = "free"
CELL_SOURCE_UNKNOWN = "unknown"

RUN_LOCAL_FAILURE_STALE_TF = "stale_tf"
RUN_LOCAL_FAILURE_STALE_SCAN = "stale_scan"
RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS = "too_few_scan_points"
RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS = "too_many_rejected_points"
RUN_LOCAL_FAILURE_START_IN_COLLISION = "start_in_collision"
RUN_LOCAL_FAILURE_GOAL_BLOCKED = "goal_blocked"
RUN_LOCAL_FAILURE_NO_CONNECTED_PATH = "no_connected_path"
RUN_LOCAL_FAILURE_MAX_UPDATES_EXCEEDED = "max_updates_exceeded"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class BaseFramePoint:
    x: float
    y: float


@dataclass(frozen=True)
class MapFrameObservation:
    x_m: float
    y_m: float
    stamp_sec: float | None = None


@dataclass(frozen=True)
class GridCellObservation:
    grid_x: int
    grid_y: int
    stamp_sec: float | None = None


@dataclass(frozen=True)
class ObservationBatch:
    observations: Sequence[MapFrameObservation | GridCellObservation]
    source: str = ""
    stamp_sec: float | None = None


@dataclass(frozen=True)
class RunLocalMapConfig:
    min_hit_count: int = 2
    inflation_radius_m: float = 0.22
    robot_footprint_radius_m: float = 0.18
    clearance_margin_m: float = 0.04
    static_wall_exclusion_radius_m: float = 0.04
    min_used_points: int = 3
    max_rejected_ratio: float = 0.90
    max_updates: int = 3
    max_replan_path_length_ratio: float = 3.0
    planner_inflate_radius_m: float = 0.0
    planner_snap_radius_m: float = planner.DEFAULT_SNAP_RADIUS_M
    max_start_snap_m: float = 0.20
    max_goal_snap_m: float = 0.30


@dataclass
class RunLocalMapDiagnostics:
    total_observations: int = 0
    used_observations: int = 0
    rejected_invalid_range: int = 0
    rejected_bounds: int = 0
    rejected_static: int = 0
    rejected_wall_band: int = 0
    rejected_low_confidence: int = 0
    confirmed_raw_cells: int = 0
    inflated_cells: int = 0
    inflated_cells_newly_occupied: int = 0
    inflated_cells_over_static_occupied: int = 0
    update_accepted: bool = False
    update_rejected_reason: str = ""
    update_count: int = 0
    cell_source_counts: dict[str, int] = field(default_factory=dict)
    inflation_radius_m: float = 0.0

    @property
    def rejected_observations(self):
        return (
            self.rejected_invalid_range
            + self.rejected_bounds
            + self.rejected_static
            + self.rejected_wall_band
        )

    @property
    def rejected_ratio(self):
        if self.total_observations == 0:
            return 1.0
        return self.rejected_observations / float(self.total_observations)


@dataclass
class RunLocalPlanDiagnostics:
    update_diagnostics: RunLocalMapDiagnostics = field(default_factory=RunLocalMapDiagnostics)
    start_cell_blocked: bool = False
    goal_cell_blocked: bool = False
    path_blocked_cell_count: int = 0
    no_path_reason: str = ""
    start_snap_distance_m: float | None = None
    goal_snap_distance_m: float | None = None
    old_path_length_m: float | None = None
    new_path_length_m: float | None = None
    old_remaining_waypoint_count: int = 0
    new_waypoint_count: int = 0
    replan_duration_sec: float | None = None
    artifact_prefix: str = ""


@dataclass(frozen=True)
class ObstacleOverlayConfig:
    forward_distance_m: float = 0.55
    forward_half_width_m: float = 0.18
    angle_window_deg: float = 45.0
    min_range_m: float = 0.12
    robot_footprint_radius_m: float = 0.18
    min_cluster_size: int = 3
    min_cluster_width_m: float = 0.05
    inflate_radius_m: float = 0.22
    max_start_snap_m: float = 0.20
    max_goal_snap_m: float = 0.30
    max_replan_path_length_ratio: float = 3.0
    planner_inflate_radius_m: float = 0.0
    planner_snap_radius_m: float = planner.DEFAULT_SNAP_RADIUS_M
    run_local_min_hit_count: int = 2
    run_local_clearance_margin_m: float = 0.04
    run_local_min_used_points: int = 3
    run_local_max_rejected_ratio: float = 0.90


@dataclass
class ObstacleOverlayDiagnostics:
    candidate_scan_points: int = 0
    filtered_obstacle_points: int = 0
    detected_obstacle_count: int = 0
    raw_obstacle_cells: int = 0
    free_obstacle_cells: int = 0
    inflated_cells_total: int = 0
    inflated_cells_newly_occupied: int = 0
    inflated_cells_over_static_occupied: int = 0
    obstacle_cluster_width_m: float = 0.0
    start_snap_distance_m: float | None = None
    goal_snap_distance_m: float | None = None
    old_path_length_m: float | None = None
    new_path_length_m: float | None = None
    old_remaining_waypoint_count: int = 0
    new_waypoint_count: int = 0
    replan_duration_sec: float | None = None
    scan_frame: str = ""
    scan_age_sec: float | None = None
    tf_age_sec: float | None = None
    tf_lookup_mode: str = ""
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
    run_local_cell_source_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ReplanResult:
    success: bool
    reason: str
    diagnostics: ObstacleOverlayDiagnostics = field(default_factory=ObstacleOverlayDiagnostics)
    updated_map_yaml: str | None = None
    updated_map_pgm: str | None = None
    updated_path_csv: str | None = None
    updated_waypoints_csv: str | None = None
    updated_path_ppm: str | None = None
    detected_obstacles_csv: str | None = None
    waypoints: list[tuple[int, float, float]] = field(default_factory=list)
    path_cells: list[tuple[int, int]] = field(default_factory=list)
    inflated_obstacle_cells: set[tuple[int, int]] = field(default_factory=set)
    run_local_map: "RunLocalObstacleMap | None" = None


class ObstacleOverlayError(RuntimeError):
    pass


def parse_pose(text):
    values = [part.strip() for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("pose must be x,y,yaw_deg")
    try:
        return Pose2D(float(values[0]), float(values[1]), float(values[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numeric") from exc


def finite_base_points(points: Sequence[BaseFramePoint | tuple[float, float]]):
    result = []
    for point in points:
        x = float(point.x if hasattr(point, "x") else point[0])
        y = float(point.y if hasattr(point, "y") else point[1])
        if math.isfinite(x) and math.isfinite(y):
            result.append(BaseFramePoint(x, y))
    return result


def scan_ranges_to_base_points(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
):
    points = []
    for index, raw_range in enumerate(ranges):
        if raw_range is None or not math.isfinite(raw_range):
            continue
        if raw_range < range_min or raw_range > range_max:
            continue
        angle = angle_min + index * angle_increment
        points.append(BaseFramePoint(
            float(raw_range) * math.cos(angle),
            float(raw_range) * math.sin(angle),
        ))
    return points


def base_point_to_map(point, robot_pose):
    yaw = math.radians(robot_pose.yaw_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        robot_pose.x + cos_yaw * point.x - sin_yaw * point.y,
        robot_pose.y + sin_yaw * point.x + cos_yaw * point.y,
    )


def map_point_to_base(map_x, map_y, robot_pose):
    yaw = math.radians(robot_pose.yaw_deg)
    dx = map_x - robot_pose.x
    dy = map_y - robot_pose.y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return BaseFramePoint(
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
    )


def base_point_passes_roi(point, config):
    distance = math.hypot(point.x, point.y)
    if point.x <= 0.0 or point.x > config.forward_distance_m:
        return False
    if abs(point.y) > config.forward_half_width_m:
        return False
    if abs(math.degrees(math.atan2(point.y, point.x))) > config.angle_window_deg:
        return False
    if distance < config.min_range_m:
        return False
    if distance < config.robot_footprint_radius_m:
        return False
    return True


def cluster_width(points):
    if not points:
        return 0.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def inflate_cells(occupancy_map, source_cells, radius_m):
    inflation_cells = int(math.ceil(radius_m / occupancy_map.metadata.resolution))
    inflated = set(source_cells)
    if inflation_cells <= 0:
        return inflated
    for cell_x, cell_y in source_cells:
        for dy in range(-inflation_cells, inflation_cells + 1):
            for dx in range(-inflation_cells, inflation_cells + 1):
                if dx * dx + dy * dy > inflation_cells * inflation_cells:
                    continue
                candidate = (cell_x + dx, cell_y + dy)
                if planner.in_bounds(occupancy_map, candidate):
                    inflated.add(candidate)
    return inflated


def cells_within_radius(occupancy_map, center_cell, radius_m):
    radius_cells = int(math.ceil(radius_m / occupancy_map.metadata.resolution))
    cells = set()
    if radius_cells < 0:
        return cells
    center_x, center_y = center_cell
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            cell = (center_x + dx, center_y + dy)
            if planner.in_bounds(occupancy_map, cell):
                cells.add(cell)
    return cells


def static_wall_band_cells(occupancy_map, radius_m):
    static_cells = planner.base_blocked_cells(occupancy_map, block_unknown=True)
    if radius_m <= 0.0:
        return set(static_cells)
    return inflate_cells(occupancy_map, static_cells, radius_m)


def observation_to_cell(occupancy_map, observation):
    if isinstance(observation, GridCellObservation):
        if not (math.isfinite(observation.grid_x) and math.isfinite(observation.grid_y)):
            return None
        return int(observation.grid_x), int(observation.grid_y)
    if not (math.isfinite(observation.x_m) and math.isfinite(observation.y_m)):
        return None
    return planner.world_to_grid(observation.x_m, observation.y_m, occupancy_map.metadata)


def map_frame_observations_from_base_points(base_points, robot_pose):
    observations = []
    for point in finite_base_points(base_points):
        map_x, map_y = base_point_to_map(point, robot_pose)
        observations.append(MapFrameObservation(map_x, map_y))
    return observations


class RunLocalObstacleMap:
    def __init__(self, static_map, config=None):
        self.static_map = static_map
        self.config = config or RunLocalMapConfig()
        self.hit_counts: dict[tuple[int, int], int] = {}
        self.confirmed_raw_cells: set[tuple[int, int]] = set()
        self.inflated_obstacle_cells: set[tuple[int, int]] = set()
        self.update_count = 0
        self.last_update_diagnostics = RunLocalMapDiagnostics(
            inflation_radius_m=self.config.inflation_radius_m,
            cell_source_counts=self.cell_source_counts(),
        )
        self._wall_band_cells = static_wall_band_cells(
            static_map,
            self.config.static_wall_exclusion_radius_m,
        )

    def add_observations(self, batch):
        diagnostics = RunLocalMapDiagnostics(
            inflation_radius_m=self.config.inflation_radius_m,
            update_count=self.update_count,
        )
        if self.update_count >= self.config.max_updates:
            diagnostics.update_rejected_reason = RUN_LOCAL_FAILURE_MAX_UPDATES_EXCEEDED
            diagnostics.cell_source_counts = self.cell_source_counts()
            self.last_update_diagnostics = diagnostics
            return diagnostics
        batch = batch if isinstance(batch, ObservationBatch) else ObservationBatch(batch)
        pending_counts: dict[tuple[int, int], int] = {}
        for observation in batch.observations:
            diagnostics.total_observations += 1
            cell = observation_to_cell(self.static_map, observation)
            if cell is None:
                diagnostics.rejected_invalid_range += 1
                continue
            if not planner.in_bounds(self.static_map, cell):
                diagnostics.rejected_bounds += 1
                continue
            state = self.static_map.cells[cell[1]][cell[0]]
            if state != planner.CELL_FREE:
                diagnostics.rejected_static += 1
                continue
            if cell in self._wall_band_cells:
                diagnostics.rejected_wall_band += 1
                continue
            diagnostics.used_observations += 1
            pending_counts[cell] = pending_counts.get(cell, 0) + 1

        if diagnostics.used_observations < self.config.min_used_points:
            diagnostics.update_rejected_reason = RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS
            diagnostics.cell_source_counts = self.cell_source_counts()
            self.last_update_diagnostics = diagnostics
            return diagnostics
        if diagnostics.rejected_ratio > self.config.max_rejected_ratio:
            diagnostics.update_rejected_reason = RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS
            diagnostics.cell_source_counts = self.cell_source_counts()
            self.last_update_diagnostics = diagnostics
            return diagnostics

        for cell, count in pending_counts.items():
            self.hit_counts[cell] = self.hit_counts.get(cell, 0) + count
        self.update_count += 1
        self.rebuild_confirmed_cells()
        diagnostics.update_accepted = True
        diagnostics.update_count = self.update_count
        diagnostics.confirmed_raw_cells = len(self.confirmed_raw_cells)
        diagnostics.inflated_cells = len(self.inflated_obstacle_cells)
        diagnostics.rejected_low_confidence = sum(
            count
            for cell, count in pending_counts.items()
            if self.hit_counts.get(cell, 0) < self.config.min_hit_count
        )
        for cell in self.inflated_obstacle_cells:
            state = self.static_map.cells[cell[1]][cell[0]]
            if state == planner.CELL_OCCUPIED or state == planner.CELL_UNKNOWN:
                diagnostics.inflated_cells_over_static_occupied += 1
            else:
                diagnostics.inflated_cells_newly_occupied += 1
        diagnostics.cell_source_counts = self.cell_source_counts()
        self.last_update_diagnostics = diagnostics
        return diagnostics

    def rebuild_confirmed_cells(self):
        self.confirmed_raw_cells = {
            cell
            for cell, count in self.hit_counts.items()
            if count >= self.config.min_hit_count
        }
        self.inflated_obstacle_cells = inflate_cells(
            self.static_map,
            self.confirmed_raw_cells,
            self.config.inflation_radius_m,
        )

    def raw_cells_in_radius(self, center_cell, radius_m):
        disk = cells_within_radius(self.static_map, center_cell, radius_m)
        return self.confirmed_raw_cells.intersection(disk)

    def overlay_cells_for_planning(self, start_world=None):
        overlay = set(self.inflated_obstacle_cells)
        if start_world is None:
            return overlay
        start_cell = planner.world_to_grid(start_world[0], start_world[1], self.static_map.metadata)
        clearance_radius = (
            self.config.robot_footprint_radius_m
            + self.config.clearance_margin_m
        )
        if self.raw_cells_in_radius(start_cell, clearance_radius):
            raise ObstacleOverlayError(RUN_LOCAL_FAILURE_START_IN_COLLISION)
        return overlay.difference(cells_within_radius(self.static_map, start_cell, clearance_radius))

    def composed_map(self, overlay_cells=None):
        cells = overlay_cells if overlay_cells is not None else self.inflated_obstacle_cells
        return planner.map_with_occupied_cells(self.static_map, cells)

    def cell_source(self, cell):
        x, y = cell
        state = self.static_map.cells[y][x]
        if state == planner.CELL_OCCUPIED:
            return CELL_SOURCE_STATIC_OCCUPIED
        if state == planner.CELL_UNKNOWN:
            return CELL_SOURCE_UNKNOWN
        if cell in self.confirmed_raw_cells:
            return CELL_SOURCE_RUN_LOCAL_RAW
        if cell in self.inflated_obstacle_cells:
            return CELL_SOURCE_RUN_LOCAL_INFLATED
        return CELL_SOURCE_FREE

    def cell_source_counts(self):
        counts = {
            CELL_SOURCE_STATIC_OCCUPIED: 0,
            CELL_SOURCE_RUN_LOCAL_RAW: 0,
            CELL_SOURCE_RUN_LOCAL_INFLATED: 0,
            CELL_SOURCE_FREE: 0,
            CELL_SOURCE_UNKNOWN: 0,
        }
        for y in range(self.static_map.height):
            for x in range(self.static_map.width):
                counts[self.cell_source((x, y))] += 1
        return counts


def run_local_artifact_paths(output_dir, prefix):
    output_dir = Path(output_dir)
    return {
        "map_yaml": output_dir / f"{prefix}_map.yaml",
        "map_pgm": output_dir / f"{prefix}_map.pgm",
        "path_csv": output_dir / f"{prefix}_path.csv",
        "waypoints_csv": output_dir / f"{prefix}_waypoints.csv",
        "path_ppm": output_dir / f"{prefix}_path.ppm",
        "obstacles_csv": output_dir / f"{prefix}_obstacles.csv",
    }


def write_run_local_obstacle_csv(path, run_local_map):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "grid_x",
            "grid_y",
            "world_x_m",
            "world_y_m",
            "hit_count",
            "cell_source",
        ])
        for cell in sorted(run_local_map.confirmed_raw_cells):
            world_x, world_y = planner.grid_to_world(
                cell[0],
                cell[1],
                run_local_map.static_map.metadata,
            )
            writer.writerow([
                cell[0],
                cell[1],
                world_x,
                world_y,
                run_local_map.hit_counts.get(cell, 0),
                run_local_map.cell_source(cell),
            ])


def render_run_local_pixels(run_local_map, planning_overlay, path, waypoints):
    path_cells = set(path)
    waypoint_cells = set(waypoints)
    start = path[0] if path else None
    goal = path[-1] if path else None
    rows = []
    for image_row in range(run_local_map.static_map.height):
        row = []
        for image_col in range(run_local_map.static_map.width):
            grid_x, grid_y = planner.image_to_grid(
                image_col,
                image_row,
                run_local_map.static_map.height,
            )
            cell = (grid_x, grid_y)
            source = run_local_map.cell_source(cell)
            if source == CELL_SOURCE_STATIC_OCCUPIED:
                color = planner.COLOR_OCCUPIED
            elif source == CELL_SOURCE_UNKNOWN:
                color = planner.COLOR_UNKNOWN
            elif source == CELL_SOURCE_RUN_LOCAL_RAW:
                color = (230, 80, 30)
            elif source == CELL_SOURCE_RUN_LOCAL_INFLATED:
                color = planner.COLOR_INFLATED
            else:
                color = planner.COLOR_FREE
            if cell in planning_overlay and source == CELL_SOURCE_FREE:
                color = planner.COLOR_INFLATED
            if cell in path_cells:
                color = planner.COLOR_PATH
            if cell in waypoint_cells:
                color = planner.COLOR_WAYPOINT
            if cell == start:
                color = planner.COLOR_START
            if cell == goal:
                color = planner.COLOR_GOAL
            row.append(color)
        rows.append(row)
    return rows


def populate_overlay_diagnostics_from_run_local(target, update_diag, plan_diag=None):
    target.run_local_map_updates = update_diag.update_count
    target.run_local_scan_points_valid = update_diag.total_observations
    target.run_local_scan_points_used = update_diag.used_observations
    target.run_local_scan_points_rejected_invalid_range = update_diag.rejected_invalid_range
    target.run_local_scan_points_rejected_static = update_diag.rejected_static
    target.run_local_scan_points_rejected_bounds = update_diag.rejected_bounds
    target.run_local_scan_points_rejected_wall_band = update_diag.rejected_wall_band
    target.run_local_scan_points_rejected_low_confidence = update_diag.rejected_low_confidence
    target.run_local_update_rejected_reason = update_diag.update_rejected_reason
    target.run_local_inflation_radius_m = update_diag.inflation_radius_m
    target.run_local_cell_source_counts = dict(update_diag.cell_source_counts)
    target.raw_obstacle_cells = update_diag.confirmed_raw_cells
    target.free_obstacle_cells = update_diag.confirmed_raw_cells
    target.detected_obstacle_count = update_diag.confirmed_raw_cells
    target.inflated_cells_total = update_diag.inflated_cells
    target.inflated_cells_newly_occupied = update_diag.inflated_cells_newly_occupied
    target.inflated_cells_over_static_occupied = update_diag.inflated_cells_over_static_occupied
    if plan_diag is not None:
        target.run_local_no_path_reason = plan_diag.no_path_reason
        target.run_local_start_cell_blocked = plan_diag.start_cell_blocked
        target.run_local_goal_cell_blocked = plan_diag.goal_cell_blocked
        target.run_local_path_blocked_cell_count = plan_diag.path_blocked_cell_count
        target.start_snap_distance_m = plan_diag.start_snap_distance_m
        target.goal_snap_distance_m = plan_diag.goal_snap_distance_m
        target.old_remaining_waypoint_count = plan_diag.old_remaining_waypoint_count
        target.new_waypoint_count = plan_diag.new_waypoint_count
        target.old_path_length_m = plan_diag.old_path_length_m
        target.new_path_length_m = plan_diag.new_path_length_m
        target.replan_duration_sec = plan_diag.replan_duration_sec


def plan_with_run_local_map(
    run_local_map,
    robot_pose,
    goal_pose,
    run_id,
    output_dir=DEFAULT_OUTPUT_DIR,
    artifact_prefix=None,
    old_remaining_waypoints=None,
):
    start_time = time.time()
    artifact_prefix = artifact_prefix or f"{run_id}_run_local"
    overlay_diag = ObstacleOverlayDiagnostics()
    plan_diag = RunLocalPlanDiagnostics(
        update_diagnostics=run_local_map.last_update_diagnostics,
        artifact_prefix=artifact_prefix,
    )
    try:
        start_world = (robot_pose.x, robot_pose.y)
        goal_world = (goal_pose.x, goal_pose.y)
        start_cell = planner.world_to_grid(start_world[0], start_world[1], run_local_map.static_map.metadata)
        goal_cell = planner.world_to_grid(goal_world[0], goal_world[1], run_local_map.static_map.metadata)
        planning_overlay = run_local_map.overlay_cells_for_planning(start_world=start_world)
        if goal_cell in run_local_map.inflated_obstacle_cells:
            plan_diag.goal_cell_blocked = True
            plan_diag.no_path_reason = RUN_LOCAL_FAILURE_GOAL_BLOCKED
            raise ObstacleOverlayError(RUN_LOCAL_FAILURE_GOAL_BLOCKED)

        planning_map = run_local_map.composed_map(planning_overlay)
        plan, inflated_blocked, _inflation_cells = planner.plan_path(
            planning_map,
            start_world,
            goal_world,
            inflate_radius_m=run_local_map.config.planner_inflate_radius_m,
            snap_radius_m=max(
                run_local_map.config.max_start_snap_m,
                run_local_map.config.max_goal_snap_m,
                run_local_map.config.planner_snap_radius_m,
            ),
        )
        plan_diag.start_snap_distance_m = planner.snapped_distance_m(
            start_world,
            plan.start_snapped_world,
        )
        plan_diag.goal_snap_distance_m = planner.snapped_distance_m(
            goal_world,
            plan.goal_snapped_world,
        )
        if plan_diag.start_snap_distance_m > run_local_map.config.max_start_snap_m:
            plan_diag.start_cell_blocked = True
            plan_diag.no_path_reason = RUN_LOCAL_FAILURE_START_IN_COLLISION
            raise ObstacleOverlayError(RUN_LOCAL_FAILURE_START_IN_COLLISION)
        if plan_diag.goal_snap_distance_m > run_local_map.config.max_goal_snap_m:
            plan_diag.goal_cell_blocked = True
            plan_diag.no_path_reason = RUN_LOCAL_FAILURE_GOAL_BLOCKED
            raise ObstacleOverlayError(RUN_LOCAL_FAILURE_GOAL_BLOCKED)

        plan_diag.path_blocked_cell_count = len(set(plan.path).intersection(planning_overlay))
        if plan_diag.path_blocked_cell_count:
            plan_diag.no_path_reason = RUN_LOCAL_FAILURE_NO_CONNECTED_PATH
            raise ObstacleOverlayError(RUN_LOCAL_FAILURE_NO_CONNECTED_PATH)

        plan_diag.new_path_length_m = plan.path_length_m
        plan_diag.new_waypoint_count = len(plan.waypoints)
        if old_remaining_waypoints is not None:
            plan_diag.old_remaining_waypoint_count = len(old_remaining_waypoints)
            plan_diag.old_path_length_m = waypoint_path_length_m(old_remaining_waypoints)
            if (
                plan_diag.old_path_length_m
                and plan.path_length_m > plan_diag.old_path_length_m * run_local_map.config.max_replan_path_length_ratio
            ):
                raise ObstacleOverlayError("replan_path_too_long")

        paths = run_local_artifact_paths(output_dir, artifact_prefix)
        planner.write_occupancy_map_copy(planning_map, paths["map_yaml"], paths["map_pgm"])
        planner.write_path_csv(paths["path_csv"], planner.build_path_rows(plan.path, planning_map.metadata))
        planner.write_path_csv(
            paths["waypoints_csv"],
            planner.build_path_rows(plan.waypoints, planning_map.metadata),
        )
        planner.write_ppm(
            paths["path_ppm"],
            render_run_local_pixels(
                run_local_map,
                planning_overlay,
                plan.path,
                plan.waypoints,
            ),
        )
        write_run_local_obstacle_csv(paths["obstacles_csv"], run_local_map)
        plan_diag.replan_duration_sec = time.time() - start_time
        populate_overlay_diagnostics_from_run_local(
            overlay_diag,
            run_local_map.last_update_diagnostics,
            plan_diag,
        )
        overlay_diag.run_local_replan_count = 1
        overlay_diag.run_local_last_replan_reason = "run_local_replan_completed"
        overlay_diag.updated_map_yaml = str(paths["map_yaml"])
        overlay_diag.updated_waypoints_csv = str(paths["waypoints_csv"])
        overlay_diag.run_local_map_yaml = str(paths["map_yaml"])
        overlay_diag.run_local_waypoints_csv = str(paths["waypoints_csv"])
        waypoint_rows = planner.build_path_rows(plan.waypoints, planning_map.metadata)
        return ReplanResult(
            success=True,
            reason="run_local_replan_completed",
            diagnostics=overlay_diag,
            updated_map_yaml=str(paths["map_yaml"]),
            updated_map_pgm=str(paths["map_pgm"]),
            updated_path_csv=str(paths["path_csv"]),
            updated_waypoints_csv=str(paths["waypoints_csv"]),
            updated_path_ppm=str(paths["path_ppm"]),
            detected_obstacles_csv=str(paths["obstacles_csv"]),
            waypoints=[
                (int(row[0]), float(row[3]), float(row[4]))
                for row in waypoint_rows
            ],
            path_cells=list(plan.path),
            inflated_obstacle_cells=set(planning_overlay),
            run_local_map=run_local_map,
        )
    except Exception as exc:
        plan_diag.replan_duration_sec = time.time() - start_time
        if str(exc) == RUN_LOCAL_FAILURE_START_IN_COLLISION:
            plan_diag.start_cell_blocked = True
        if str(exc) == RUN_LOCAL_FAILURE_GOAL_BLOCKED:
            plan_diag.goal_cell_blocked = True
        if isinstance(exc, ObstacleOverlayError) and not plan_diag.no_path_reason:
            plan_diag.no_path_reason = str(exc)
        elif not isinstance(exc, ObstacleOverlayError):
            plan_diag.no_path_reason = RUN_LOCAL_FAILURE_NO_CONNECTED_PATH
        populate_overlay_diagnostics_from_run_local(
            overlay_diag,
            run_local_map.last_update_diagnostics,
            plan_diag,
        )
        overlay_diag.run_local_replan_count = 1
        overlay_diag.run_local_last_replan_reason = str(exc)
        return ReplanResult(
            success=False,
            reason=str(exc),
            diagnostics=overlay_diag,
            run_local_map=run_local_map,
            inflated_obstacle_cells=set(run_local_map.inflated_obstacle_cells),
        )


def build_run_local_replan_result(
    static_map,
    observations,
    robot_pose,
    goal_pose,
    run_id,
    output_dir=DEFAULT_OUTPUT_DIR,
    artifact_prefix=None,
    config=None,
    old_remaining_waypoints=None,
    run_local_map=None,
):
    config = config or RunLocalMapConfig()
    occupancy_map = (
        static_map
        if isinstance(static_map, planner.OccupancyMap)
        else planner.load_occupancy_map(static_map)
    )
    run_local_map = run_local_map or RunLocalObstacleMap(occupancy_map, config)
    update_diag = run_local_map.add_observations(ObservationBatch(observations))
    if not update_diag.update_accepted:
        diagnostics = ObstacleOverlayDiagnostics()
        populate_overlay_diagnostics_from_run_local(diagnostics, update_diag)
        return ReplanResult(
            success=False,
            reason=update_diag.update_rejected_reason,
            diagnostics=diagnostics,
            run_local_map=run_local_map,
            inflated_obstacle_cells=set(run_local_map.inflated_obstacle_cells),
        )
    return plan_with_run_local_map(
        run_local_map,
        robot_pose,
        goal_pose,
        run_id,
        output_dir=output_dir,
        artifact_prefix=artifact_prefix,
        old_remaining_waypoints=old_remaining_waypoints,
    )


def rasterized_segment_cells(occupancy_map, start, end, radius_m=0.0):
    start_cell = planner.world_to_grid(start.x, start.y, occupancy_map.metadata)
    end_cell = planner.world_to_grid(end.x, end.y, occupancy_map.metadata)
    dx = end_cell[0] - start_cell[0]
    dy = end_cell[1] - start_cell[1]
    steps = max(abs(dx), abs(dy), 1)
    cells = set()
    for index in range(steps + 1):
        t = index / float(steps)
        cell = (
            int(round(start_cell[0] + dx * t)),
            int(round(start_cell[1] + dy * t)),
        )
        if planner.in_bounds(occupancy_map, cell):
            cells.add(cell)
            cells.update(cells_within_radius(occupancy_map, cell, radius_m))
    return cells


def path_corridor_blocked_cells(
    occupancy_map,
    current_pose,
    waypoints,
    blocked_cells,
    check_distance_m,
    corridor_radius_m,
):
    if not waypoints:
        return set()
    checked = set()
    remaining = max(0.0, check_distance_m)
    previous = current_pose
    for waypoint in waypoints:
        segment_length = math.hypot(waypoint.x - previous.x, waypoint.y - previous.y)
        if segment_length <= 0.0:
            previous = waypoint
            continue
        end = waypoint
        if segment_length > remaining > 0.0:
            ratio = remaining / segment_length
            end = Pose2D(
                previous.x + (waypoint.x - previous.x) * ratio,
                previous.y + (waypoint.y - previous.y) * ratio,
                0.0,
            )
        checked.update(rasterized_segment_cells(
            occupancy_map,
            previous,
            end,
            radius_m=corridor_radius_m,
        ))
        remaining -= segment_length
        if remaining <= 0.0:
            break
        previous = waypoint
    return checked.intersection(blocked_cells)


def build_overlay_map(occupancy_map, base_points, robot_pose, config):
    diagnostics = ObstacleOverlayDiagnostics()
    finite_points = finite_base_points(base_points)
    diagnostics.candidate_scan_points = len(finite_points)

    filtered_points = [
        point for point in finite_points
        if base_point_passes_roi(point, config)
    ]
    diagnostics.filtered_obstacle_points = len(filtered_points)
    diagnostics.obstacle_cluster_width_m = cluster_width(filtered_points)
    if len(filtered_points) < config.min_cluster_size:
        raise ObstacleOverlayError("insufficient_obstacle_points")
    if diagnostics.obstacle_cluster_width_m < config.min_cluster_width_m:
        raise ObstacleOverlayError("obstacle_cluster_too_narrow")

    raw_cells = set()
    free_cells = set()
    obstacle_rows = []
    for point in filtered_points:
        map_x, map_y = base_point_to_map(point, robot_pose)
        cell = planner.world_to_grid(map_x, map_y, occupancy_map.metadata)
        if not planner.in_bounds(occupancy_map, cell):
            continue
        raw_cells.add(cell)
        cell_state = occupancy_map.cells[cell[1]][cell[0]]
        if cell_state == planner.CELL_FREE:
            free_cells.add(cell)
            obstacle_rows.append([
                point.x,
                point.y,
                map_x,
                map_y,
                cell[0],
                cell[1],
            ])

    diagnostics.raw_obstacle_cells = len(raw_cells)
    diagnostics.free_obstacle_cells = len(free_cells)
    diagnostics.detected_obstacle_count = len(free_cells)
    if not free_cells:
        raise ObstacleOverlayError("no_obstacle_candidates")

    inflated_cells = inflate_cells(occupancy_map, free_cells, config.inflate_radius_m)
    diagnostics.inflated_cells_total = len(inflated_cells)
    for cell in inflated_cells:
        cell_state = occupancy_map.cells[cell[1]][cell[0]]
        if cell_state == planner.CELL_OCCUPIED:
            diagnostics.inflated_cells_over_static_occupied += 1
        else:
            diagnostics.inflated_cells_newly_occupied += 1

    updated_map = planner.map_with_occupied_cells(occupancy_map, inflated_cells)
    return updated_map, inflated_cells, obstacle_rows, diagnostics


def waypoint_path_length_m(waypoints):
    if not waypoints or len(waypoints) < 2:
        return 0.0
    total = 0.0
    previous = waypoints[0]
    for waypoint in waypoints[1:]:
        px = float(previous.x if hasattr(previous, "x") else previous[1])
        py = float(previous.y if hasattr(previous, "y") else previous[2])
        wx = float(waypoint.x if hasattr(waypoint, "x") else waypoint[1])
        wy = float(waypoint.y if hasattr(waypoint, "y") else waypoint[2])
        total += math.hypot(wx - px, wy - py)
        previous = waypoint
    return total


def artifact_paths(output_dir, run_id, sequence=1):
    output_dir = Path(output_dir)
    stem = f"{run_id}_updated_{sequence:03d}"
    return {
        "map_yaml": output_dir / f"{stem}_map.yaml",
        "map_pgm": output_dir / f"{stem}_map.pgm",
        "path_csv": output_dir / f"{stem}_path.csv",
        "waypoints_csv": output_dir / f"{stem}_waypoints.csv",
        "path_ppm": output_dir / f"{stem}_path.ppm",
        "obstacles_csv": output_dir / f"{stem}_detected_obstacles.csv",
    }


def write_obstacle_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "base_x_m",
            "base_y_m",
            "map_x_m",
            "map_y_m",
            "grid_x",
            "grid_y",
        ])
        writer.writerows(rows)


def build_replan_result(
    static_map,
    base_points,
    robot_pose,
    goal_pose,
    run_id,
    output_dir=DEFAULT_OUTPUT_DIR,
    sequence=1,
    config=None,
    old_remaining_waypoints=None,
):
    start_time = time.time()
    config = config or ObstacleOverlayConfig()
    diagnostics = ObstacleOverlayDiagnostics()
    try:
        occupancy_map = (
            static_map
            if isinstance(static_map, planner.OccupancyMap)
            else planner.load_occupancy_map(static_map)
        )
        updated_map, inflated_cells, obstacle_rows, diagnostics = build_overlay_map(
            occupancy_map,
            base_points,
            robot_pose,
            config,
        )
        plan, inflated_blocked, _inflation_cells = planner.plan_path(
            updated_map,
            (robot_pose.x, robot_pose.y),
            (goal_pose.x, goal_pose.y),
            inflate_radius_m=config.planner_inflate_radius_m,
            snap_radius_m=max(config.max_start_snap_m, config.max_goal_snap_m, config.planner_snap_radius_m),
        )
        diagnostics.start_snap_distance_m = planner.snapped_distance_m(
            (robot_pose.x, robot_pose.y),
            plan.start_snapped_world,
        )
        diagnostics.goal_snap_distance_m = planner.snapped_distance_m(
            (goal_pose.x, goal_pose.y),
            plan.goal_snapped_world,
        )
        if diagnostics.start_snap_distance_m > config.max_start_snap_m:
            raise ObstacleOverlayError("start_snap_distance_exceeded")
        if diagnostics.goal_snap_distance_m > config.max_goal_snap_m:
            raise ObstacleOverlayError("goal_snap_distance_exceeded")

        diagnostics.new_path_length_m = plan.path_length_m
        diagnostics.new_waypoint_count = len(plan.waypoints)
        if old_remaining_waypoints is not None:
            diagnostics.old_remaining_waypoint_count = len(old_remaining_waypoints)
            diagnostics.old_path_length_m = waypoint_path_length_m(old_remaining_waypoints)
            if (
                diagnostics.old_path_length_m
                and plan.path_length_m > diagnostics.old_path_length_m * config.max_replan_path_length_ratio
            ):
                raise ObstacleOverlayError("replan_path_too_long")

        paths = artifact_paths(output_dir, run_id, sequence=sequence)
        planner.write_occupancy_map_copy(updated_map, paths["map_yaml"], paths["map_pgm"])
        planner.write_path_csv(paths["path_csv"], planner.build_path_rows(plan.path, updated_map.metadata))
        planner.write_path_csv(
            paths["waypoints_csv"],
            planner.build_path_rows(plan.waypoints, updated_map.metadata),
        )
        planner.write_ppm(
            paths["path_ppm"],
            planner.render_planner_pixels(
                updated_map,
                inflated_blocked,
                plan.path,
                plan.waypoints,
            ),
        )
        write_obstacle_csv(paths["obstacles_csv"], obstacle_rows)
        diagnostics.replan_duration_sec = time.time() - start_time
        waypoint_rows = planner.build_path_rows(plan.waypoints, updated_map.metadata)
        return ReplanResult(
            success=True,
            reason="replan_completed",
            diagnostics=diagnostics,
            updated_map_yaml=str(paths["map_yaml"]),
            updated_map_pgm=str(paths["map_pgm"]),
            updated_path_csv=str(paths["path_csv"]),
            updated_waypoints_csv=str(paths["waypoints_csv"]),
            updated_path_ppm=str(paths["path_ppm"]),
            detected_obstacles_csv=str(paths["obstacles_csv"]),
            waypoints=[
                (int(row[0]), float(row[3]), float(row[4]))
                for row in waypoint_rows
            ],
            path_cells=list(plan.path),
            inflated_obstacle_cells=inflated_cells,
        )
    except Exception as exc:
        diagnostics.replan_duration_sec = time.time() - start_time
        return ReplanResult(
            success=False,
            reason=str(exc),
            diagnostics=diagnostics,
        )


def synthetic_obstacle_points(base_x, base_y, width_m, point_count):
    if point_count <= 1:
        return [BaseFramePoint(base_x, base_y)]
    start_y = base_y - width_m / 2.0
    step = width_m / float(point_count - 1)
    return [
        BaseFramePoint(base_x, start_y + index * step)
        for index in range(point_count)
    ]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create a one-shot LiDAR obstacle map overlay and replan artifacts.",
    )
    parser.add_argument("--static-map", default=DEFAULT_STATIC_MAP, type=Path)
    parser.add_argument("--replan-output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--run-id", default="synthetic_lidar_overlay_test")
    parser.add_argument("--robot-pose", required=True, type=parse_pose)
    parser.add_argument("--goal-pose", required=True, type=parse_pose)
    parser.add_argument("--synthetic-obstacle-base-x", required=True, type=float)
    parser.add_argument("--synthetic-obstacle-base-y", required=True, type=float)
    parser.add_argument("--synthetic-obstacle-width-m", default=0.08, type=float)
    parser.add_argument("--synthetic-obstacle-points", default=5, type=int)
    parser.add_argument("--obstacle-forward-distance-m", default=ObstacleOverlayConfig.forward_distance_m, type=float)
    parser.add_argument("--obstacle-forward-half-width-m", default=ObstacleOverlayConfig.forward_half_width_m, type=float)
    parser.add_argument("--obstacle-angle-window-deg", default=ObstacleOverlayConfig.angle_window_deg, type=float)
    parser.add_argument("--obstacle-min-range-m", default=ObstacleOverlayConfig.min_range_m, type=float)
    parser.add_argument("--robot-footprint-radius-m", default=ObstacleOverlayConfig.robot_footprint_radius_m, type=float)
    parser.add_argument("--obstacle-min-cluster-size", default=ObstacleOverlayConfig.min_cluster_size, type=int)
    parser.add_argument("--obstacle-min-cluster-width-m", default=ObstacleOverlayConfig.min_cluster_width_m, type=float)
    parser.add_argument("--obstacle-inflate-radius-m", default=ObstacleOverlayConfig.inflate_radius_m, type=float)
    args = parser.parse_args(argv)

    positive_fields = [
        "synthetic_obstacle_width_m",
        "obstacle_forward_distance_m",
        "obstacle_forward_half_width_m",
        "obstacle_angle_window_deg",
        "obstacle_min_range_m",
        "robot_footprint_radius_m",
        "obstacle_min_cluster_width_m",
        "obstacle_inflate_radius_m",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.synthetic_obstacle_points < 1:
        parser.error("--synthetic-obstacle-points must be >= 1")
    if args.obstacle_min_cluster_size < 1:
        parser.error("--obstacle-min-cluster-size must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = ObstacleOverlayConfig(
        forward_distance_m=args.obstacle_forward_distance_m,
        forward_half_width_m=args.obstacle_forward_half_width_m,
        angle_window_deg=args.obstacle_angle_window_deg,
        min_range_m=args.obstacle_min_range_m,
        robot_footprint_radius_m=args.robot_footprint_radius_m,
        min_cluster_size=args.obstacle_min_cluster_size,
        min_cluster_width_m=args.obstacle_min_cluster_width_m,
        inflate_radius_m=args.obstacle_inflate_radius_m,
    )
    points = synthetic_obstacle_points(
        args.synthetic_obstacle_base_x,
        args.synthetic_obstacle_base_y,
        args.synthetic_obstacle_width_m,
        args.synthetic_obstacle_points,
    )
    result = build_replan_result(
        args.static_map,
        points,
        args.robot_pose,
        args.goal_pose,
        args.run_id,
        output_dir=args.replan_output_dir,
        config=config,
    )
    print(f"Status: {'completed' if result.success else 'failed'}")
    print(f"Reason: {result.reason}")
    print(f"Detected obstacle cells: {result.diagnostics.detected_obstacle_count}")
    print(f"Filtered obstacle points: {result.diagnostics.filtered_obstacle_points}")
    if result.success:
        print(f"Updated map YAML: {result.updated_map_yaml}")
        print(f"Updated waypoint CSV: {result.updated_waypoints_csv}")
        print(f"Updated path PPM: {result.updated_path_ppm}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
