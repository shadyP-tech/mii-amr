from __future__ import annotations

import math

from .models import CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN, LocalGrid


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


def grid_cell_value(cells, cell):
    x, y = cell
    if 0 <= y < len(cells) and 0 <= x < len(cells[y]):
        return cells[y][x]
    return CELL_UNKNOWN


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

