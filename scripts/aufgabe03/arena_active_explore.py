#!/usr/bin/env python3
"""
Odom-frame active exploration helpers for arena-prior localization recovery.

This module is ROS-free. It builds a small odom-frame occupancy grid from
LaserScan-like messages, proposes short observation-zone moves, plans with A*,
and returns diagnostics. Runtime motion remains in arena_active_spin.py.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field


CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1
CELL_INFLATED = 2

FAILURE_POSE_NOT_UNIQUE = "pose_not_unique"
FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE = "wall_separation_out_of_tolerance"


@dataclass(frozen=True)
class ActiveExploreConfig:
    max_attempts: int = 2
    max_single_move_m: float = 0.45
    max_total_distance_m: float = 0.90
    max_candidate_path_m: float | None = None
    grid_resolution_m: float = 0.05
    grid_size_m: float = 4.0
    inflation_radius_m: float = 0.15
    soft_clearance_radius_m: float = 0.20
    soft_clearance_weight: float = 3.0
    unknown_blocked: bool = True
    max_path_segments: int = 3
    target_nearest_short_wall_range_m: float = 1.65
    center_min_step_m: float = 0.25
    lateral_offset_threshold_m: float = 0.25
    lateral_target_offset_m: float = 0.10
    heater_approach_target_range_m: float = 1.05
    heater_approach_min_selected_score: float = 0.50
    heater_approach_max_opposite_score: float = 0.30
    heater_approach_min_delta: float = 0.35
    arena_length_m: float = 3.90
    max_short_wall_range_sum_error_m: float = 0.15


@dataclass(frozen=True)
class LocalGrid:
    origin_x: float
    origin_y: float
    resolution_m: float
    width: int
    height: int
    cells: tuple[tuple[int, ...], ...]
    robot_cell: tuple[int, int]

    def to_dict(self):
        counts = grid_cell_counts(self)
        return {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "resolution_m": self.resolution_m,
            "width": self.width,
            "height": self.height,
            "robot_cell": list(self.robot_cell),
            "cell_counts": counts,
        }


@dataclass(frozen=True)
class RawCandidate:
    kind: str
    target_x: float
    target_y: float
    heading_rad: float
    geometry_progress: float = 0.0
    heater_potential: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveExploreCandidate:
    kind: str
    target_x: float
    target_y: float
    heading_rad: float
    accepted: bool
    rejection_reason: str = ""
    score: float | None = None
    score_components: dict = field(default_factory=dict)
    path_cells: tuple[tuple[int, int], ...] = ()
    path_world: tuple[tuple[float, float], ...] = ()
    simplified_path_world: tuple[tuple[float, float], ...] = ()
    path_length_m: float | None = None
    turn_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        data["path_cells"] = [list(cell) for cell in self.path_cells]
        data["path_world"] = [list(point) for point in self.path_world]
        data["simplified_path_world"] = [
            list(point) for point in self.simplified_path_world
        ]
        return data


@dataclass(frozen=True)
class ActiveExplorePlan:
    ok: bool
    reason: str
    selected: ActiveExploreCandidate | None
    candidates: tuple[ActiveExploreCandidate, ...]
    grid: LocalGrid | None

    def to_dict(self):
        return {
            "ok": self.ok,
            "reason": self.reason,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "grid": None if self.grid is None else self.grid.to_dict(),
        }


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, low, high):
    return max(low, min(high, value))


def valid_range(value, range_min, range_max):
    return (
        value is not None
        and math.isfinite(value)
        and value >= range_min
        and value <= range_max
    )


def yaw_rad_from_pose(pose):
    return math.radians(float(getattr(pose, "yaw_deg", 0.0)))


def grid_cell_counts(grid: LocalGrid):
    counts = {
        "unknown": 0,
        "free": 0,
        "occupied": 0,
        "inflated": 0,
    }
    for row in grid.cells:
        for value in row:
            if value == CELL_FREE:
                counts["free"] += 1
            elif value == CELL_OCCUPIED:
                counts["occupied"] += 1
            elif value == CELL_INFLATED:
                counts["inflated"] += 1
            else:
                counts["unknown"] += 1
    return counts


def world_to_cell(grid: LocalGrid, x, y):
    return (
        int(math.floor((x - grid.origin_x) / grid.resolution_m)),
        int(math.floor((y - grid.origin_y) / grid.resolution_m)),
    )


def cell_to_world(grid: LocalGrid, cell):
    return (
        grid.origin_x + (cell[0] + 0.5) * grid.resolution_m,
        grid.origin_y + (cell[1] + 0.5) * grid.resolution_m,
    )


def in_bounds(grid: LocalGrid, cell):
    return 0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height


def bresenham_cells(start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    cells = []
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def set_cell(cells, cell, value):
    x, y = cell
    if 0 <= y < len(cells) and 0 <= x < len(cells[y]):
        cells[y][x] = value


def mark_scan_ray(cells, grid, start_cell, endpoint_cell, occupied, preserve_occupied=False):
    ray = [
        cell
        for cell in bresenham_cells(start_cell, endpoint_cell)
        if in_bounds(grid, cell)
    ]
    if not ray:
        return
    free_cells = ray[:-1] if occupied else ray
    for cell in free_cells:
        if preserve_occupied and grid_cell_value(cells, cell) == CELL_OCCUPIED:
            continue
        set_cell(cells, cell, CELL_FREE)
    if occupied and in_bounds(grid, ray[-1]):
        set_cell(cells, ray[-1], CELL_OCCUPIED)


def grid_cell_value(cells, cell):
    x, y = cell
    if 0 <= y < len(cells) and 0 <= x < len(cells[y]):
        return cells[y][x]
    return CELL_UNKNOWN


def inflated_cells_for(grid, occupied_cells, radius_m):
    radius_cells = int(math.ceil(radius_m / grid.resolution_m))
    inflated = set()
    for cell_x, cell_y in occupied_cells:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                cell = (cell_x + dx, cell_y + dy)
                if in_bounds(grid, cell):
                    inflated.add(cell)
    return inflated


def empty_local_grid(robot_pose, config: ActiveExploreConfig):
    width = int(round(config.grid_size_m / config.grid_resolution_m))
    height = width
    origin_x = float(robot_pose.x) - config.grid_size_m / 2.0
    origin_y = float(robot_pose.y) - config.grid_size_m / 2.0
    mutable = [[CELL_UNKNOWN for _x in range(width)] for _y in range(height)]
    placeholder = LocalGrid(
        origin_x,
        origin_y,
        config.grid_resolution_m,
        width,
        height,
        tuple(tuple(row) for row in mutable),
        (width // 2, height // 2),
    )
    robot_cell = world_to_cell(placeholder, robot_pose.x, robot_pose.y)
    set_cell(mutable, robot_cell, CELL_FREE)
    return placeholder, mutable, robot_cell


def mark_scan_on_grid(mutable, grid, scan, scan_pose, config, preserve_occupied=False):
    start_cell = world_to_cell(grid, scan_pose.x, scan_pose.y)
    if not in_bounds(grid, start_cell):
        return set()
    yaw = yaw_rad_from_pose(scan_pose)
    max_ray_m = config.grid_size_m / 2.0 - config.grid_resolution_m
    occupied_cells = set()

    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        distance = min(float(raw_range), max_ray_m)
        angle = yaw + float(scan.angle_min) + index * float(scan.angle_increment)
        end_x = float(scan_pose.x) + distance * math.cos(angle)
        end_y = float(scan_pose.y) + distance * math.sin(angle)
        end_cell = world_to_cell(grid, end_x, end_y)
        occupied = float(raw_range) < min(float(scan.range_max), max_ray_m)
        mark_scan_ray(
            mutable,
            grid,
            start_cell,
            end_cell,
            occupied,
            preserve_occupied=preserve_occupied,
        )
        if occupied and in_bounds(grid, end_cell):
            occupied_cells.add(end_cell)
    return occupied_cells


def finalize_grid(
    robot_pose,
    config,
    mutable,
    robot_cell,
    occupied_cells,
    inflation_radius_m=None,
):
    placeholder = LocalGrid(
        float(robot_pose.x) - config.grid_size_m / 2.0,
        float(robot_pose.y) - config.grid_size_m / 2.0,
        config.grid_resolution_m,
        len(mutable[0]),
        len(mutable),
        tuple(tuple(row) for row in mutable),
        robot_cell,
    )
    if inflation_radius_m is None:
        inflation_radius_m = config.inflation_radius_m
    if inflation_radius_m > 0.0:
        inflated = inflated_cells_for(
            placeholder,
            occupied_cells,
            inflation_radius_m,
        )
        for cell in inflated:
            x, y = cell
            if mutable[y][x] != CELL_OCCUPIED:
                mutable[y][x] = CELL_INFLATED
    set_cell(mutable, robot_cell, CELL_FREE)
    return LocalGrid(
        placeholder.origin_x,
        placeholder.origin_y,
        config.grid_resolution_m,
        placeholder.width,
        placeholder.height,
        tuple(tuple(row) for row in mutable),
        robot_cell,
    )


def build_local_grid(
    scan,
    robot_pose,
    config: ActiveExploreConfig,
    inflation_radius_m=None,
):
    grid, mutable, robot_cell = empty_local_grid(robot_pose, config)
    occupied_cells = mark_scan_on_grid(
        mutable,
        grid,
        scan,
        robot_pose,
        config,
        preserve_occupied=False,
    )
    return finalize_grid(
        robot_pose,
        config,
        mutable,
        robot_cell,
        occupied_cells,
        inflation_radius_m=inflation_radius_m,
    )


def build_local_grid_from_scan_samples(
    scan_samples,
    robot_pose,
    config: ActiveExploreConfig,
    inflation_radius_m=None,
):
    grid, mutable, robot_cell = empty_local_grid(robot_pose, config)
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
    return finalize_grid(
        robot_pose,
        config,
        mutable,
        robot_cell,
        occupied_cells,
        inflation_radius_m=inflation_radius_m,
    )


def build_observed_local_grid(scan, robot_pose, config: ActiveExploreConfig):
    return build_local_grid(scan, robot_pose, config, inflation_radius_m=0.0)


def build_observed_local_grid_from_scan_samples(
    scan_samples,
    robot_pose,
    config: ActiveExploreConfig,
):
    return build_local_grid_from_scan_samples(
        scan_samples,
        robot_pose,
        config,
        inflation_radius_m=0.0,
    )


def traversable(grid: LocalGrid, cell, unknown_blocked=True):
    if not in_bounds(grid, cell):
        return False
    value = grid.cells[cell[1]][cell[0]]
    if value == CELL_FREE:
        return True
    if value == CELL_UNKNOWN:
        return not unknown_blocked
    return False


def neighbors_8(grid, cell, unknown_blocked=True):
    x, y = cell
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = (x + dx, y + dy)
            if not traversable(grid, neighbor, unknown_blocked):
                continue
            if dx != 0 and dy != 0:
                if (
                    not traversable(grid, (x + dx, y), unknown_blocked)
                    or not traversable(grid, (x, y + dy), unknown_blocked)
                ):
                    continue
            yield neighbor


def movement_cost(a, b, resolution_m):
    return math.hypot(b[0] - a[0], b[1] - a[1]) * resolution_m


def blocked_cells(grid: LocalGrid):
    cells = []
    for y, row in enumerate(grid.cells):
        for x, value in enumerate(row):
            if value in {CELL_OCCUPIED, CELL_INFLATED}:
                cells.append((x, y))
    return tuple(cells)


def blocked_distance_field(grid: LocalGrid):
    blocked = blocked_cells(grid)
    fallback = grid.resolution_m * max(grid.width, grid.height)
    rows = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            cell = (x, y)
            if not blocked:
                row.append(fallback)
                continue
            row.append(
                min(
                    movement_cost(cell, blocked_cell, grid.resolution_m)
                    for blocked_cell in blocked
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def clearance_distance_for_cell(clearance_distance_field, cell):
    x, y = cell
    return clearance_distance_field[y][x]


def soft_clearance_cell_penalty(clearance_m, config: ActiveExploreConfig):
    radius = float(config.soft_clearance_radius_m)
    weight = float(config.soft_clearance_weight)
    if radius <= 0.0 or weight <= 0.0 or clearance_m >= radius:
        return 0.0
    normalized = (radius - max(0.0, clearance_m)) / radius
    return weight * normalized * normalized


def path_soft_clearance_penalty(path, clearance_distance_field, config):
    if not path:
        return 0.0
    penalties = [
        soft_clearance_cell_penalty(
            clearance_distance_for_cell(clearance_distance_field, cell),
            config,
        )
        for cell in path
    ]
    return sum(penalties) / len(penalties)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(
    grid: LocalGrid,
    start,
    goal,
    unknown_blocked=True,
    clearance_distance_field=None,
    config: ActiveExploreConfig | None = None,
):
    if not traversable(grid, start, unknown_blocked):
        raise ValueError("start_not_free")
    if not traversable(grid, goal, unknown_blocked):
        raise ValueError("goal_not_free")
    queue = []
    heapq.heappush(queue, (0.0, 0, start))
    came_from = {}
    g_score = {start: 0.0}
    tie_breaker = 0
    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in neighbors_8(grid, current, unknown_blocked):
            base_cost = movement_cost(
                current,
                neighbor,
                grid.resolution_m,
            )
            clearance_penalty = 0.0
            if clearance_distance_field is not None and config is not None:
                clearance_penalty = (
                    soft_clearance_cell_penalty(
                        clearance_distance_for_cell(
                            clearance_distance_field,
                            neighbor,
                        ),
                        config,
                    )
                    * grid.resolution_m
                )
            tentative = g_score[current] + base_cost + clearance_penalty
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heuristic = movement_cost(neighbor, goal, grid.resolution_m)
            tie_breaker += 1
            heapq.heappush(queue, (tentative + heuristic, tie_breaker, neighbor))
    raise ValueError("no_connected_path")


def path_length_m(path, resolution_m):
    if len(path) < 2:
        return 0.0
    return sum(
        movement_cost(path[index - 1], path[index], resolution_m)
        for index in range(1, len(path))
    )


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    gcd = math.gcd(abs(dx), abs(dy))
    if gcd == 0:
        return 0, 0
    return dx // gcd, dy // gcd


def simplify_path_cells(path):
    if len(path) <= 2:
        return tuple(path)
    simplified = [path[0]]
    previous = direction_between(path[0], path[1])
    for index in range(2, len(path)):
        current = direction_between(path[index - 1], path[index])
        if current != previous:
            simplified.append(path[index - 1])
            previous = current
    simplified.append(path[-1])
    return tuple(simplified)


def turn_count_for_path(path):
    simplified = simplify_path_cells(path)
    return max(0, len(simplified) - 2)


def nearest_blocked_distance_m(grid: LocalGrid, path, clearance_distance_field=None):
    if clearance_distance_field is not None:
        if not path:
            return grid.resolution_m * max(grid.width, grid.height)
        return min(
            clearance_distance_for_cell(clearance_distance_field, path_cell)
            for path_cell in path
        )
    blocked = blocked_cells(grid)
    if not blocked:
        return grid.resolution_m * max(grid.width, grid.height)
    best = math.inf
    for path_cell in path:
        for blocked_cell in blocked:
            best = min(
                best,
                movement_cost(path_cell, blocked_cell, grid.resolution_m),
            )
    return best


def unknown_ratio_near_path(grid: LocalGrid, path):
    checked = set()
    unknown = 0
    for x, y in path:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cell = (x + dx, y + dy)
                if cell in checked or not in_bounds(grid, cell):
                    continue
                checked.add(cell)
                if grid.cells[cell[1]][cell[0]] == CELL_UNKNOWN:
                    unknown += 1
    if not checked:
        return 1.0
    return unknown / float(len(checked))


def point_from_heading(robot_pose, heading_rad, distance_m):
    return (
        float(robot_pose.x) + distance_m * math.cos(heading_rad),
        float(robot_pose.y) + distance_m * math.sin(heading_rad),
    )


def candidate_range(candidate):
    value = getattr(candidate, "short_wall_candidate_range_m", None)
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def candidate_heater_score(candidate):
    value = getattr(candidate, "heater_profile_score", None)
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def candidate_profile_valid(candidate):
    if candidate is None:
        return False
    failed = getattr(candidate, "validity_failed_reason", None)
    features = getattr(candidate, "profile_features", None) or {}
    return failed is None and features.get("validity_failed_reason") is None


def geometry_is_recoverable(result, config: ActiveExploreConfig):
    if result.success:
        return False, "already_localized"
    if result.failure_reason not in {
        FAILURE_POSE_NOT_UNIQUE,
        FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE,
    }:
        return False, "not_recoverable_failure"
    long_fit = result.long_wall_fit
    if long_fit is None:
        return False, "invalid_long_wall_fit"
    if result.failure_reason == FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE:
        return True, "ok"
    if not getattr(long_fit, "ok", False) or long_fit.axis_angle_rad is None:
        return False, "invalid_long_wall_fit"
    candidates = result.short_wall_candidates or {}
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    negative_range = candidate_range(negative)
    positive_range = candidate_range(positive)
    if negative_range is None or positive_range is None:
        return False, "missing_short_wall_ranges"
    range_sum_error = negative_range + positive_range - config.arena_length_m
    if abs(range_sum_error) > config.max_short_wall_range_sum_error_m:
        return False, "range_sum_invalid"
    return True, "ok"


def generate_raw_candidates(
    result,
    scan,
    robot_pose,
    config: ActiveExploreConfig,
    origin_yaw_rad=0.0,
):
    ok, reason = geometry_is_recoverable(result, config)
    if not ok:
        return (), reason

    long_fit = result.long_wall_fit
    candidates = result.short_wall_candidates or {}
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    negative_range = candidate_range(negative)
    positive_range = candidate_range(positive)
    axis_angle = getattr(long_fit, "axis_angle_rad", None)
    axis_heading = (
        yaw_rad_from_pose(robot_pose)
        if axis_angle is None
        else normalize_angle_rad(origin_yaw_rad + axis_angle)
    )
    normal_angle = getattr(long_fit, "normal_angle_rad", None)
    normal_heading = (
        None
        if normal_angle is None
        else normalize_angle_rad(origin_yaw_rad + normal_angle)
    )
    raw = []

    if negative_range is not None and positive_range is not None:
        if negative_range <= positive_range:
            nearest_side = "axis_negative"
            nearest_range = negative_range
            away_heading = axis_heading
        else:
            nearest_side = "axis_positive"
            nearest_range = positive_range
            away_heading = normalize_angle_rad(axis_heading + math.pi)
        center_step = config.target_nearest_short_wall_range_m - nearest_range
        if center_step >= config.center_min_step_m:
            distance = min(center_step, config.max_single_move_m)
            x, y = point_from_heading(robot_pose, away_heading, distance)
            raw.append(
                RawCandidate(
                    "provisional_center",
                    x,
                    y,
                    away_heading,
                    geometry_progress=distance / config.max_single_move_m,
                    metadata={
                        "nearest_side": nearest_side,
                        "nearest_range_m": nearest_range,
                        "requested_step_m": center_step,
                    },
                )
            )

    lateral_offset = getattr(long_fit, "lateral_offset_m", None)
    if (
        lateral_offset is not None
        and math.isfinite(lateral_offset)
        and abs(lateral_offset) > config.lateral_offset_threshold_m
        and normal_heading is not None
    ):
        lateral_step = min(
            max(0.0, abs(lateral_offset) - config.lateral_target_offset_m),
            config.max_single_move_m,
        )
        heading = (
            normal_heading
            if lateral_offset < 0.0
            else normalize_angle_rad(normal_heading + math.pi)
        )
        x, y = point_from_heading(robot_pose, heading, lateral_step)
        raw.append(
            RawCandidate(
                "lateral_recenter",
                x,
                y,
                heading,
                geometry_progress=0.8 * lateral_step / config.max_single_move_m,
                metadata={
                    "lateral_offset_m": lateral_offset,
                    "requested_step_m": lateral_step,
                },
            )
        )

    if (
        negative_range is not None
        and positive_range is not None
        and candidate_profile_valid(negative)
        and candidate_profile_valid(positive)
    ):
        negative_score = candidate_heater_score(negative)
        positive_score = candidate_heater_score(positive)
        if negative_score is not None and positive_score is not None:
            if negative_score >= positive_score:
                selected_side = "axis_negative"
                selected_range = negative_range
                selected_score = negative_score
                opposite_score = positive_score
                heading = normalize_angle_rad(axis_heading + math.pi)
            else:
                selected_side = "axis_positive"
                selected_range = positive_range
                selected_score = positive_score
                opposite_score = negative_score
                heading = axis_heading
            delta = selected_score - opposite_score
            step = selected_range - config.heater_approach_target_range_m
            if (
                step > 0.0
                and selected_score >= config.heater_approach_min_selected_score
                and opposite_score <= config.heater_approach_max_opposite_score
                and delta >= config.heater_approach_min_delta
            ):
                distance = min(step, config.max_single_move_m)
                x, y = point_from_heading(robot_pose, heading, distance)
                raw.append(
                    RawCandidate(
                        "suspected_heater_approach",
                        x,
                        y,
                        heading,
                        geometry_progress=0.7 * distance / config.max_single_move_m,
                        heater_potential=delta,
                        metadata={
                            "selected_side": selected_side,
                            "selected_score": selected_score,
                            "opposite_score": opposite_score,
                            "heater_delta": delta,
                            "requested_step_m": step,
                        },
                    )
                )

    yaw = yaw_rad_from_pose(robot_pose)
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
                },
            )
        )

    return tuple(raw), "ok"


def min_scan_range_in_sector(scan, lower_deg, upper_deg):
    values = []
    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        angle = math.degrees(
            normalize_angle_rad(scan.angle_min + index * scan.angle_increment)
        )
        if lower_deg <= angle <= upper_deg:
            values.append(float(raw_range))
    return min(values) if values else None


def obstacle_shadow_unknown_cells(grid: LocalGrid):
    shadow_cells = set()
    for y, row in enumerate(grid.cells):
        for x, value in enumerate(row):
            if value != CELL_UNKNOWN:
                continue
            cell = (x, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if not in_bounds(grid, neighbor):
                        continue
                    neighbor_value = grid.cells[neighbor[1]][neighbor[0]]
                    if neighbor_value in {CELL_OCCUPIED, CELL_INFLATED}:
                        shadow_cells.add(cell)
                        break
                if cell in shadow_cells:
                    break
    return shadow_cells


def shadow_cell_visible_from(grid: LocalGrid, viewpoint_cell, shadow_cell):
    ray = bresenham_cells(viewpoint_cell, shadow_cell)
    if len(ray) < 2:
        return False
    for cell in ray[1:-1]:
        if not in_bounds(grid, cell):
            return False
        if grid.cells[cell[1]][cell[0]] in {CELL_OCCUPIED, CELL_INFLATED}:
            return False
    return True


def shadow_information_gain_components(grid: LocalGrid, viewpoint_cell):
    shadow_cells = obstacle_shadow_unknown_cells(grid)
    visible_count = sum(
        1
        for shadow_cell in shadow_cells
        if shadow_cell_visible_from(grid, viewpoint_cell, shadow_cell)
    )
    total_count = len(shadow_cells)
    gain = clamp(visible_count / 20.0, 0.0, 1.0)
    return gain, visible_count, total_count


def cluster_shadow_unknown_cells(grid: LocalGrid):
    shadow_cells = obstacle_shadow_unknown_cells(grid)
    remaining = set(shadow_cells)
    clusters = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        cluster = []
        while stack:
            cell = stack.pop()
            cluster.append(cell)
            x, y = cell
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        clusters.append(tuple(sorted(cluster)))
    return tuple(clusters)


def cluster_centroid_world(grid: LocalGrid, cells):
    if not cells:
        return cell_to_world(grid, grid.robot_cell)
    xs = []
    ys = []
    for cell in cells:
        x, y = cell_to_world(grid, cell)
        xs.append(x)
        ys.append(y)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def nearest_cluster_distance_m(grid: LocalGrid, cell, cluster):
    if not cluster:
        return math.inf
    return min(
        movement_cost(cell, shadow_cell, grid.resolution_m)
        for shadow_cell in cluster
    )


def visible_cluster_shadow_cells(grid: LocalGrid, viewpoint_cell, cluster):
    return tuple(
        shadow_cell
        for shadow_cell in cluster
        if shadow_cell_visible_from(grid, viewpoint_cell, shadow_cell)
    )


def generate_obstacle_shadow_frontier_candidates(
    grid: LocalGrid,
    config,
    clearance_distance_field=None,
):
    min_viewpoint_distance_m = 0.15
    max_viewpoint_distance_m = 0.90
    max_clusters = 8
    max_viewpoints_per_cluster = 4
    max_total_candidates = 16
    max_candidate_path_m = (
        config.max_candidate_path_m
        if config.max_candidate_path_m is not None
        else config.max_total_distance_m
    )
    clearance_distance_field = clearance_distance_field or blocked_distance_field(grid)

    clusters = [
        cluster
        for cluster in cluster_shadow_unknown_cells(grid)
        if len(cluster) >= 1
    ]
    clusters.sort(
        key=lambda cluster: (
            -len(cluster),
            nearest_cluster_distance_m(grid, grid.robot_cell, cluster),
        )
    )

    raw = []
    used_target_cells = set()
    max_radius_cells = int(math.ceil(max_viewpoint_distance_m / grid.resolution_m))

    for cluster_index, cluster in enumerate(clusters[:max_clusters]):
        min_x = max(0, min(cell[0] for cell in cluster) - max_radius_cells)
        max_x = min(
            grid.width - 1,
            max(cell[0] for cell in cluster) + max_radius_cells,
        )
        min_y = max(0, min(cell[1] for cell in cluster) - max_radius_cells)
        max_y = min(
            grid.height - 1,
            max(cell[1] for cell in cluster) + max_radius_cells,
        )
        cluster_centroid = cluster_centroid_world(grid, cluster)
        viewpoints = []

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                if cell == grid.robot_cell or cell in used_target_cells:
                    continue
                if grid.cells[y][x] != CELL_FREE:
                    continue
                cluster_distance_m = nearest_cluster_distance_m(grid, cell, cluster)
                if (
                    cluster_distance_m < min_viewpoint_distance_m
                    or cluster_distance_m > max_viewpoint_distance_m
                ):
                    continue
                robot_distance_m = movement_cost(
                    grid.robot_cell,
                    cell,
                    grid.resolution_m,
                )
                if robot_distance_m > max_candidate_path_m + grid.resolution_m:
                    continue
                visible_cells = visible_cluster_shadow_cells(grid, cell, cluster)
                if not visible_cells:
                    continue
                viewpoint_clearance_m = clearance_distance_for_cell(
                    clearance_distance_field,
                    cell,
                )
                visible_centroid = cluster_centroid_world(grid, visible_cells)
                target_x, target_y = cell_to_world(grid, cell)
                heading = math.atan2(
                    visible_centroid[1] - target_y,
                    visible_centroid[0] - target_x,
                )
                raw_candidate = RawCandidate(
                    "obstacle_shadow_frontier",
                    target_x,
                    target_y,
                    heading,
                    geometry_progress=clamp(
                        robot_distance_m
                        / max(config.max_single_move_m, grid.resolution_m),
                        0.0,
                        1.0,
                    ),
                    metadata={
                        "candidate_source": "obstacle_shadow_frontier",
                        "cluster_index": cluster_index,
                        "cluster_size": len(cluster),
                        "cluster_centroid_world": [
                            cluster_centroid[0],
                            cluster_centroid[1],
                        ],
                        "target_cell": [cell[0], cell[1]],
                        "visible_cluster_shadow_count": len(visible_cells),
                        "viewpoint_clearance_m": viewpoint_clearance_m,
                        "nearest_shadow_distance_m": cluster_distance_m,
                        "straight_line_path_estimate_m": robot_distance_m,
                    },
                )
                viewpoints.append(
                    (
                        (
                            -len(visible_cells),
                            -viewpoint_clearance_m,
                            robot_distance_m,
                            cluster_distance_m,
                            y,
                            x,
                        ),
                        cell,
                        raw_candidate,
                    )
                )

        viewpoints.sort(key=lambda item: item[0])
        for _sort_key, cell, raw_candidate in viewpoints[:max_viewpoints_per_cluster]:
            if len(raw) >= max_total_candidates:
                return tuple(raw)
            used_target_cells.add(cell)
            raw.append(raw_candidate)

    return tuple(raw)


def score_candidate(raw, grid, path, config, clearance_distance_field=None):
    length = path_length_m(path, grid.resolution_m)
    turns = turn_count_for_path(path)
    clearance = nearest_blocked_distance_m(grid, path, clearance_distance_field)
    clearance_component = clamp(clearance / 0.50, 0.0, 1.0)
    soft_clearance_penalty = (
        0.0
        if clearance_distance_field is None
        else path_soft_clearance_penalty(path, clearance_distance_field, config)
    )
    path_unknown_ratio = unknown_ratio_near_path(grid, path)
    viewpoint_cell = path[-1]
    (
        shadow_information_gain,
        visible_shadow_unknown_count,
        total_shadow_unknown_count,
    ) = shadow_information_gain_components(grid, viewpoint_cell)
    components = {
        "geometry_progress": raw.geometry_progress,
        "heater_potential": raw.heater_potential,
        "clearance_margin": clearance_component,
        "path_length": length,
        "turn_count": turns,
        "path_unknown_ratio": path_unknown_ratio,
        "path_min_clearance_m": clearance,
        "path_soft_clearance_penalty": soft_clearance_penalty,
        "shadow_information_gain": shadow_information_gain,
        "visible_shadow_unknown_count": visible_shadow_unknown_count,
        "total_shadow_unknown_count": total_shadow_unknown_count,
    }
    score = (
        5.0 * components["shadow_information_gain"]
        + 2.0 * components["heater_potential"]
        + 1.5 * components["clearance_margin"]
        + 0.75 * components["geometry_progress"]
        - 1.0 * components["path_length"]
        - 0.5 * components["turn_count"]
        - 4.0 * components["path_unknown_ratio"]
        - 2.0 * components["path_soft_clearance_penalty"]
    )
    return score, components


def plan_candidate(
    raw,
    grid,
    config: ActiveExploreConfig,
    clearance_distance_field=None,
):
    target_cell = world_to_cell(grid, raw.target_x, raw.target_y)
    if not in_bounds(grid, target_cell):
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason="goal_outside_grid",
            metadata=raw.metadata,
        )
    if not traversable(grid, target_cell, config.unknown_blocked):
        reason = (
            "goal_unknown"
            if grid.cells[target_cell[1]][target_cell[0]] == CELL_UNKNOWN
            else "goal_blocked"
        )
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason=reason,
            metadata=raw.metadata,
        )
    try:
        clearance_distance_field = clearance_distance_field or blocked_distance_field(grid)
        path = astar(
            grid,
            grid.robot_cell,
            target_cell,
            config.unknown_blocked,
            clearance_distance_field=clearance_distance_field,
            config=config,
        )
    except ValueError as exc:
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason=str(exc),
            metadata=raw.metadata,
        )
    length = path_length_m(path, grid.resolution_m)
    max_candidate_path_m = (
        config.max_candidate_path_m
        if config.max_candidate_path_m is not None
        else config.max_total_distance_m
    )
    if length > max_candidate_path_m:
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason="path_too_long",
            path_cells=tuple(path),
            path_world=tuple(cell_to_world(grid, cell) for cell in path),
            path_length_m=length,
            metadata={
                **raw.metadata,
                "path_length_limit_m": max_candidate_path_m,
            },
        )
    score, components = score_candidate(
        raw,
        grid,
        path,
        config,
        clearance_distance_field,
    )
    simplified = simplify_path_cells(path)
    return ActiveExploreCandidate(
        raw.kind,
        raw.target_x,
        raw.target_y,
        raw.heading_rad,
        True,
        score=score,
        score_components=components,
        path_cells=tuple(path),
        path_world=tuple(cell_to_world(grid, cell) for cell in path),
        simplified_path_world=tuple(
            cell_to_world(grid, cell) for cell in simplified
        ),
        path_length_m=length,
        turn_count=components["turn_count"],
        metadata=raw.metadata,
    )


def plan_active_explore_recovery(
    result,
    scan,
    robot_pose,
    config: ActiveExploreConfig,
    origin_yaw_rad=0.0,
    grid=None,
):
    raw_candidates, reason = generate_raw_candidates(
        result,
        scan,
        robot_pose,
        config,
        origin_yaw_rad=origin_yaw_rad,
    )
    if reason != "ok":
        return ActiveExplorePlan(False, reason, None, (), None)
    grid = grid or build_local_grid(scan, robot_pose, config)
    clearance_distance_field = blocked_distance_field(grid)
    frontier_candidates = generate_obstacle_shadow_frontier_candidates(
        grid,
        config,
        clearance_distance_field=clearance_distance_field,
    )
    raw_candidates = tuple(frontier_candidates) + tuple(raw_candidates)
    candidates = tuple(
        plan_candidate(
            raw,
            grid,
            config,
            clearance_distance_field=clearance_distance_field,
        )
        for raw in raw_candidates
    )
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return ActiveExplorePlan(False, "no_reachable_candidate", None, candidates, grid)
    selected = max(
        accepted,
        key=lambda candidate: (
            candidate.score if candidate.score is not None else -math.inf
        ),
    )
    return ActiveExplorePlan(True, "selected", selected, candidates, grid)
