#!/usr/bin/env python3
"""
Run-local obstacle overlay and A* replanning helpers.

This module is intentionally stdlib-only and ROS-free. Runtime code may convert
sensor messages into map-frame observations, but this module only sees pure
points/cells, composes a temporary obstacle overlay, and
replans with the existing A* planner. It does not modify the source map.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import map_path_planner as planner
from waypoint_following.models import Waypoint
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
DEFAULT_MIN_REPLAN_PATH_LENGTH_BASELINE_M = 0.75


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
    inflation_radius_m: float = 0.15
    robot_footprint_radius_m: float = 0.10
    clearance_margin_m: float = 0.04
    static_wall_exclusion_radius_m: float = 0.04
    min_used_points: int = 3
    max_rejected_ratio: float = 0.90
    max_updates: int = 3
    max_replan_path_length_ratio: float = 3.0
    min_replan_path_length_baseline_m: float = DEFAULT_MIN_REPLAN_PATH_LENGTH_BASELINE_M
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
@dataclass(frozen=True)
class ObstacleOverlayConfig:
    forward_distance_m: float = 0.55
    forward_half_width_m: float = 0.18
    angle_window_deg: float = 45.0
    min_range_m: float = 0.12
    robot_footprint_radius_m: float = 0.10
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
    run_local_cell_source_counts: dict[str, int] = field(default_factory=dict)
@dataclass
class ReplanResult:
    success: bool
    reason: str
    diagnostics: ObstacleOverlayDiagnostics = field(default_factory=ObstacleOverlayDiagnostics)
    waypoints: list[tuple[int, float, float]] = field(default_factory=list)
    path_points: list[Waypoint] = field(default_factory=list)
    raw_path_points: list[Waypoint] = field(default_factory=list)
    path_cells: list[tuple[int, int]] = field(default_factory=list)
    inflated_obstacle_cells: set[tuple[int, int]] = field(default_factory=set)
    run_local_map: "RunLocalObstacleMap | None" = None


class ObstacleOverlayError(RuntimeError):
    pass


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

    def raw_cells_within_world_radius(self, center_world, radius_m):
        center_x, center_y = center_world
        cells = set()
        for cell in self.confirmed_raw_cells:
            world_x, world_y = planner.grid_to_world(
                cell[0],
                cell[1],
                self.static_map.metadata,
            )
            if math.hypot(world_x - center_x, world_y - center_y) <= radius_m:
                cells.add(cell)
        return cells

    def overlay_cells_for_planning(self, start_world=None):
        overlay = set(self.inflated_obstacle_cells)
        if start_world is None:
            return overlay
        start_cell = planner.world_to_grid(start_world[0], start_world[1], self.static_map.metadata)
        start_collision_radius = self.config.robot_footprint_radius_m
        if self.raw_cells_within_world_radius(start_world, start_collision_radius):
            raise ObstacleOverlayError(RUN_LOCAL_FAILURE_START_IN_COLLISION)
        start_clearance_radius = (
            self.config.robot_footprint_radius_m
            + self.config.clearance_margin_m
        )
        return overlay.difference(cells_within_radius(self.static_map, start_cell, start_clearance_radius))

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
    overlay_diag = ObstacleOverlayDiagnostics()
    plan_diag = RunLocalPlanDiagnostics(
        update_diagnostics=run_local_map.last_update_diagnostics,
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
            if replan_path_too_long(
                plan.path_length_m,
                plan_diag.old_path_length_m,
                run_local_map.config.max_replan_path_length_ratio,
                run_local_map.config.min_replan_path_length_baseline_m,
            ):
                raise ObstacleOverlayError("replan_path_too_long")

        path_rows = planner.build_path_rows(plan.path, planning_map.metadata)
        waypoint_rows = planner.build_path_rows(plan.waypoints, planning_map.metadata)
        plan_diag.replan_duration_sec = time.time() - start_time
        populate_overlay_diagnostics_from_run_local(
            overlay_diag,
            run_local_map.last_update_diagnostics,
            plan_diag,
        )
        overlay_diag.run_local_replan_count = 1
        overlay_diag.run_local_last_replan_reason = "run_local_replan_completed"
        return ReplanResult(
            success=True,
            reason="run_local_replan_completed",
            diagnostics=overlay_diag,
            waypoints=[
                (int(row[0]), float(row[3]), float(row[4]))
                for row in waypoint_rows
            ],
            path_points=[
                Waypoint(int(row[0]), float(row[3]), float(row[4]))
                for row in path_rows
            ],
            raw_path_points=[
                Waypoint(int(row[0]), float(row[3]), float(row[4]))
                for row in path_rows
            ],
            path_cells=list(plan.path),
            inflated_obstacle_cells=set(inflated_blocked),
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
    checked = path_corridor_cells(
        occupancy_map,
        current_pose,
        waypoints,
        check_distance_m,
        corridor_radius_m,
    )
    return checked.intersection(blocked_cells)


def path_corridor_cells(
    occupancy_map,
    current_pose,
    waypoints,
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
    return checked


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


def replan_path_length_limit(old_path_length_m, max_ratio, min_baseline_m):
    if not old_path_length_m:
        return None
    baseline = max(float(old_path_length_m), float(min_baseline_m))
    return baseline * float(max_ratio)


def replan_path_too_long(new_path_length_m, old_path_length_m, max_ratio, min_baseline_m):
    limit = replan_path_length_limit(old_path_length_m, max_ratio, min_baseline_m)
    return limit is not None and float(new_path_length_m) > limit
