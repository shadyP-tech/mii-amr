from __future__ import annotations

import heapq
import math

from .grid import in_bounds
from .models import (
    CELL_FREE,
    CELL_INFLATED,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    ActiveExploreConfig,
    LocalGrid,
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


def _squared_edt_1d(values):
    sites = [index for index, value in enumerate(values) if math.isfinite(value)]
    if not sites:
        return [math.inf] * len(values)

    v = [0] * len(sites)
    z = [0.0] * (len(sites) + 1)
    k = 0
    v[0] = sites[0]
    z[0] = -math.inf
    z[1] = math.inf

    for q in sites[1:]:
        while True:
            p = v[k]
            s = ((values[q] + q * q) - (values[p] + p * p)) / (2.0 * (q - p))
            if s > z[k]:
                break
            k -= 1
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = math.inf

    distances = [math.inf] * len(values)
    k = 0
    for q in range(len(values)):
        while z[k + 1] < q:
            k += 1
        p = v[k]
        distances[q] = (q - p) * (q - p) + values[p]
    return distances


def blocked_distance_field(grid: LocalGrid):
    fallback = grid.resolution_m * max(grid.width, grid.height)
    blocked = blocked_cells(grid)
    if not blocked:
        return tuple(
            tuple(fallback for _x in range(grid.width))
            for _y in range(grid.height)
        )

    inf = math.inf
    row_pass = []
    blocked_set = set(blocked)
    for y in range(grid.height):
        row_sites = [
            0.0 if (x, y) in blocked_set else inf
            for x in range(grid.width)
        ]
        row_pass.append(_squared_edt_1d(row_sites))

    rows = [[fallback for _x in range(grid.width)] for _y in range(grid.height)]
    for x in range(grid.width):
        column = [row_pass[y][x] for y in range(grid.height)]
        squared_distances = _squared_edt_1d(column)
        for y, squared_distance in enumerate(squared_distances):
            if math.isfinite(squared_distance):
                rows[y][x] = math.sqrt(squared_distance) * grid.resolution_m
    return tuple(tuple(row) for row in rows)


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

