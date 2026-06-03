from __future__ import annotations

import math

from .grid import inflated_cells_for, in_bounds, mark_scan_ray, set_cell, world_to_cell
from .math_utils import valid_range, yaw_rad_from_pose
from .models import (
    CELL_FREE,
    CELL_INFLATED,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    ActiveExploreConfig,
    LocalGrid,
)


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

