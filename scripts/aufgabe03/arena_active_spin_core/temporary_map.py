from __future__ import annotations

import math

from arena_geometry_localizer import ScanSample
from arena_active_explore import (
    CELL_FREE,
    CELL_INFLATED,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    in_bounds,
    world_to_cell,
)

from .models import (
    LOCALIZER_FILTER_MAX_WALL_THICKNESS_M,
    LOCALIZER_FILTER_MIN_WALL_ASPECT_RATIO,
    LOCALIZER_FILTER_MIN_WALL_LENGTH_M,
    LOCALIZER_FILTER_WALL_EXPAND_CELLS,
    LOCALIZER_FILTER_WALL_MARGIN_CELLS,
)


def temporary_map_cell_to_occupancy(value):
    if value == CELL_UNKNOWN:
        return -1
    if value == CELL_FREE:
        return 0
    if value == CELL_INFLATED:
        return 70
    if value == CELL_OCCUPIED:
        return 100
    return -1


def temporary_map_occupancy_data(grid):
    return [
        temporary_map_cell_to_occupancy(value)
        for row in grid.cells
        for value in row
    ]


def valid_scan_range_count(samples):
    count = 0
    for sample in samples:
        for value in sample.ranges:
            if value is None or not math.isfinite(value):
                continue
            if value < sample.range_min or value > sample.range_max:
                continue
            count += 1
    return count


def scan_endpoint_world(sample, index, raw_range):
    pose = sample.odom_pose
    if pose is None:
        return None
    angle = float(sample.angle_min) + index * float(sample.angle_increment)
    local_x = float(raw_range) * math.cos(angle)
    local_y = float(raw_range) * math.sin(angle)
    yaw = math.radians(float(pose.yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        float(pose.x) + cos_yaw * local_x - sin_yaw * local_y,
        float(pose.y) + sin_yaw * local_x + cos_yaw * local_y,
    )


def neighbors_8_cells(cell):
    x, y = cell
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def cluster_cells_8(cells):
    remaining = set(cells)
    clusters = []
    while remaining:
        start = remaining.pop()
        cluster = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            for neighbor in neighbors_8_cells(cell):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                cluster.add(neighbor)
                stack.append(neighbor)
        clusters.append(frozenset(cluster))
    return tuple(clusters)


def occupied_cells_from_grid(grid):
    cells = []
    for y, row in enumerate(grid.cells):
        for x, value in enumerate(row):
            if value == CELL_OCCUPIED:
                cells.append((x, y))
    return tuple(cells)


def blocked_cells_from_grid(grid):
    cells = []
    for y, row in enumerate(grid.cells):
        for x, value in enumerate(row):
            if value in {CELL_OCCUPIED, CELL_INFLATED}:
                cells.append((x, y))
    return tuple(cells)


def cell_bounds(cells):
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return min(xs), max(xs), min(ys), max(ys)


def cluster_is_wall_like(grid, cluster, occupied_envelope):
    min_x, max_x, min_y, max_y = cell_bounds(cluster)
    env_min_x, env_max_x, env_min_y, env_max_y = occupied_envelope
    margin = LOCALIZER_FILTER_WALL_MARGIN_CELLS
    near_outer_envelope = (
        min_x <= env_min_x + margin
        or max_x >= env_max_x - margin
        or min_y <= env_min_y + margin
        or max_y >= env_max_y - margin
    )
    near_grid_boundary = (
        min_x <= margin
        or min_y <= margin
        or max_x >= grid.width - 1 - margin
        or max_y >= grid.height - 1 - margin
    )
    if not near_outer_envelope and not near_grid_boundary:
        return False

    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    long_cells = max(span_x, span_y)
    short_cells = max(1, min(span_x, span_y))
    long_m = long_cells * grid.resolution_m
    short_m = short_cells * grid.resolution_m
    aspect = long_cells / short_cells
    return (
        long_m >= LOCALIZER_FILTER_MIN_WALL_LENGTH_M
        and short_m <= LOCALIZER_FILTER_MAX_WALL_THICKNESS_M
        and aspect >= LOCALIZER_FILTER_MIN_WALL_ASPECT_RATIO
    )


def expand_cells(grid, cells, radius_cells):
    expanded = set()
    for cell_x, cell_y in cells:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                cell = (cell_x + dx, cell_y + dy)
                if in_bounds(grid, cell):
                    expanded.add(cell)
    return expanded


def temporary_grid_localizer_obstacle_mask(grid):
    occupied = occupied_cells_from_grid(grid)
    if not occupied:
        return set(), set(), {
            "occupied_cluster_count": 0,
            "protected_wall_cluster_count": 0,
        }

    occupied_envelope = cell_bounds(occupied)
    protected_wall_cells = set()
    protected_wall_cluster_count = 0
    clusters = cluster_cells_8(occupied)
    for cluster in clusters:
        if not cluster_is_wall_like(grid, cluster, occupied_envelope):
            continue
        protected_wall_cluster_count += 1
        protected_wall_cells.update(cluster)

    protected_wall_cells = expand_cells(
        grid,
        protected_wall_cells,
        LOCALIZER_FILTER_WALL_EXPAND_CELLS,
    )
    obstacle_mask = set(blocked_cells_from_grid(grid)) - protected_wall_cells
    diagnostics = {
        "occupied_cluster_count": len(clusters),
        "protected_wall_cluster_count": protected_wall_cluster_count,
    }
    return obstacle_mask, protected_wall_cells, diagnostics


def filter_scan_samples_with_temporary_obstacle_map(samples, grid, obstacle_mask):
    filtered = []
    filtered_range_count = 0
    for sample in samples:
        ranges = list(sample.ranges)
        for index, raw_range in enumerate(ranges):
            if raw_range is None or not math.isfinite(raw_range):
                continue
            if raw_range < sample.range_min or raw_range > sample.range_max:
                continue
            endpoint = scan_endpoint_world(sample, index, raw_range)
            if endpoint is None:
                continue
            cell = world_to_cell(grid, endpoint[0], endpoint[1])
            if not in_bounds(grid, cell):
                continue
            if cell not in obstacle_mask:
                continue
            ranges[index] = float("inf")
            filtered_range_count += 1
        filtered.append(
            ScanSample(
                ranges=ranges,
                angle_min=sample.angle_min,
                angle_increment=sample.angle_increment,
                range_min=sample.range_min,
                range_max=sample.range_max,
                odom_pose=sample.odom_pose,
            )
        )
    return filtered, filtered_range_count
