from __future__ import annotations

import math

from .grid import bresenham_cells, cell_to_world, in_bounds
from .math_utils import clamp
from .models import CELL_FREE, CELL_INFLATED, CELL_OCCUPIED, CELL_UNKNOWN, LocalGrid, RawCandidate
from .path_planning import (
    blocked_distance_field,
    clearance_distance_for_cell,
    movement_cost,
)


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

