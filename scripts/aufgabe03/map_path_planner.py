#!/usr/bin/env python3
"""
Offline A* path planner for ROS trinary occupancy maps.

The script is intentionally stdlib-only so it can run on the MacBook and on the
lab workstation without ROS, Pillow, PyYAML, or matplotlib.
"""

import argparse
import ast
import csv
import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path


CELL_FREE = 0
CELL_OCCUPIED = 1
CELL_UNKNOWN = 2

DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_CSV = Path("results/aufgabe03/aufgabe03_planned_path.csv")
DEFAULT_OUTPUT_WAYPOINTS_CSV = Path("results/aufgabe03/aufgabe03_waypoints.csv")
DEFAULT_OUTPUT_PPM = Path("results/aufgabe03/aufgabe03_planned_path.ppm")

DEFAULT_INFLATE_RADIUS_M = 0.18
DEFAULT_SNAP_RADIUS_M = 0.30

COLOR_FREE = (255, 255, 255)
COLOR_OCCUPIED = (0, 0, 0)
COLOR_UNKNOWN = (155, 155, 155)
COLOR_INFLATED = (255, 190, 190)
COLOR_PATH = (30, 80, 220)
COLOR_WAYPOINT = (0, 210, 220)
COLOR_START = (0, 180, 70)
COLOR_GOAL = (220, 40, 40)


@dataclass(frozen=True)
class MapMetadata:
    yaml_path: Path
    image_path: Path
    resolution: float
    origin: tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    mode: str


@dataclass(frozen=True)
class PgmImage:
    width: int
    height: int
    maxval: int
    pixels: list[list[int]]


@dataclass(frozen=True)
class OccupancyMap:
    metadata: MapMetadata
    width: int
    height: int
    cells: list[list[int]]


@dataclass(frozen=True)
class PlanResult:
    path: list[tuple[int, int]]
    waypoints: list[tuple[int, int]]
    start_requested_world: tuple[float, float]
    goal_requested_world: tuple[float, float]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    start_snapped_world: tuple[float, float]
    goal_snapped_world: tuple[float, float]
    path_length_m: float


@dataclass(frozen=True)
class TrackingPathSmoothingResult:
    points: list[tuple[float, float]]
    status: str
    raw_point_count: int
    smoothed_point_count: int
    raw_length_m: float
    smoothed_length_m: float


def strip_inline_comment(line):
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def parse_yaml_scalar(text):
    text = text.strip()
    if not text:
        return ""
    if text[0] in {"'", '"'}:
        return ast.literal_eval(text)
    if text.startswith("["):
        return ast.literal_eval(text)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in text for char in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def read_simple_yaml(path):
    path = Path(path)
    data = {}
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            stripped = strip_inline_comment(line).strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise ValueError(f"{path}:{line_number}: expected 'key: value'")
            key, value = stripped.split(":", 1)
            data[key.strip()] = parse_yaml_scalar(value)
    return data


def read_map_metadata(path):
    yaml_path = Path(path)
    data = read_simple_yaml(yaml_path)

    required = [
        "image",
        "resolution",
        "origin",
        "negate",
        "occupied_thresh",
        "free_thresh",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{yaml_path} is missing required field(s): {', '.join(missing)}")

    mode = str(data.get("mode", "trinary")).lower()
    if mode != "trinary":
        raise ValueError("Only trinary maps are supported in v1")

    origin = data["origin"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError(f"{yaml_path} origin must be [x, y, yaw]")

    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
    if abs(origin_tuple[2]) > 1e-12:
        raise ValueError("Only zero-yaw map origins are supported in v1")

    image_path = Path(str(data["image"]))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    return MapMetadata(
        yaml_path=yaml_path,
        image_path=image_path,
        resolution=float(data["resolution"]),
        origin=origin_tuple,
        negate=int(data["negate"]),
        occupied_thresh=float(data["occupied_thresh"]),
        free_thresh=float(data["free_thresh"]),
        mode=mode,
    )


def _next_pgm_token(data, index):
    length = len(data)
    while index < length:
        byte = data[index]
        if byte == ord("#"):
            while index < length and data[index] not in b"\r\n":
                index += 1
            continue
        if chr(byte).isspace():
            index += 1
            continue
        break

    if index >= length:
        raise ValueError("Unexpected end of PGM header")

    start = index
    while index < length:
        byte = data[index]
        if byte == ord("#") or chr(byte).isspace():
            break
        index += 1

    return data[start:index].decode("ascii"), index


def _skip_pgm_whitespace_and_comments(data, index):
    length = len(data)
    while index < length:
        byte = data[index]
        if byte == ord("#"):
            while index < length and data[index] not in b"\r\n":
                index += 1
            continue
        if chr(byte).isspace():
            index += 1
            continue
        break
    return index


def read_pgm(path):
    path = Path(path)
    data = path.read_bytes()

    magic, index = _next_pgm_token(data, 0)
    if magic not in {"P2", "P5"}:
        raise ValueError(f"{path} is not a P2/P5 PGM image")

    width_text, index = _next_pgm_token(data, index)
    height_text, index = _next_pgm_token(data, index)
    maxval_text, index = _next_pgm_token(data, index)

    width = int(width_text)
    height = int(height_text)
    maxval = int(maxval_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} has invalid dimensions")
    if maxval <= 0 or maxval > 255:
        raise ValueError(f"{path} uses unsupported maxval {maxval}")

    if magic == "P2":
        values = []
        while len(values) < width * height:
            token, index = _next_pgm_token(data, index)
            values.append(int(token))
        if any(value < 0 or value > maxval for value in values):
            raise ValueError(f"{path} contains a pixel outside 0..{maxval}")
    else:
        index = _skip_pgm_whitespace_and_comments(data, index)
        expected = width * height
        values = list(data[index:index + expected])
        if len(values) != expected:
            raise ValueError(f"{path} has incomplete binary pixel data")

    rows = [
        values[row_start:row_start + width]
        for row_start in range(0, width * height, width)
    ]
    return PgmImage(width=width, height=height, maxval=maxval, pixels=rows)


def grid_to_image(grid_x, grid_y, height):
    return grid_x, height - 1 - grid_y


def image_to_grid(image_col, image_row, height):
    return image_col, height - 1 - image_row


def grid_to_world(grid_x, grid_y, metadata):
    origin_x, origin_y, _origin_yaw = metadata.origin
    return (
        origin_x + (grid_x + 0.5) * metadata.resolution,
        origin_y + (grid_y + 0.5) * metadata.resolution,
    )


def world_to_grid(world_x, world_y, metadata):
    origin_x, origin_y, _origin_yaw = metadata.origin
    return (
        math.floor((world_x - origin_x) / metadata.resolution),
        math.floor((world_y - origin_y) / metadata.resolution),
    )


def world_bounds(occupancy_map):
    origin_x, origin_y, _origin_yaw = occupancy_map.metadata.origin
    max_x = origin_x + occupancy_map.width * occupancy_map.metadata.resolution
    max_y = origin_y + occupancy_map.height * occupancy_map.metadata.resolution
    return origin_x, max_x, origin_y, max_y


def pixel_to_cell(pixel, metadata, maxval=255):
    if metadata.negate:
        prob = pixel / maxval
    else:
        prob = (maxval - pixel) / maxval

    if prob >= metadata.occupied_thresh:
        return CELL_OCCUPIED
    if prob <= metadata.free_thresh:
        return CELL_FREE
    return CELL_UNKNOWN


def build_occupancy_map(metadata, image):
    cells = [
        [CELL_UNKNOWN for _ in range(image.width)]
        for _ in range(image.height)
    ]

    for image_row in range(image.height):
        grid_y = image.height - 1 - image_row
        for image_col in range(image.width):
            grid_x = image_col
            cells[grid_y][grid_x] = pixel_to_cell(
                image.pixels[image_row][image_col],
                metadata,
                maxval=image.maxval,
            )

    return OccupancyMap(
        metadata=metadata,
        width=image.width,
        height=image.height,
        cells=cells,
    )


def copy_occupancy_map(occupancy_map, cells=None, metadata=None):
    copied_cells = [
        list(row)
        for row in (cells if cells is not None else occupancy_map.cells)
    ]
    return OccupancyMap(
        metadata=metadata if metadata is not None else occupancy_map.metadata,
        width=occupancy_map.width,
        height=occupancy_map.height,
        cells=copied_cells,
    )


def map_with_occupied_cells(occupancy_map, occupied_cells):
    updated = copy_occupancy_map(occupancy_map)
    for grid_x, grid_y in occupied_cells:
        if in_bounds(updated, (grid_x, grid_y)):
            updated.cells[grid_y][grid_x] = CELL_OCCUPIED
    return updated


def load_occupancy_map(path):
    metadata = read_map_metadata(path)
    image = read_pgm(metadata.image_path)
    return build_occupancy_map(metadata, image)


def count_cells(occupancy_map):
    counts = {
        CELL_FREE: 0,
        CELL_OCCUPIED: 0,
        CELL_UNKNOWN: 0,
    }
    for row in occupancy_map.cells:
        for cell in row:
            counts[cell] += 1
    return counts


def in_bounds(occupancy_map, cell):
    grid_x, grid_y = cell
    return 0 <= grid_x < occupancy_map.width and 0 <= grid_y < occupancy_map.height


def base_blocked_cells(occupancy_map, block_unknown=True):
    blocked = set()
    for grid_y, row in enumerate(occupancy_map.cells):
        for grid_x, cell in enumerate(row):
            if cell == CELL_OCCUPIED or (block_unknown and cell == CELL_UNKNOWN):
                blocked.add((grid_x, grid_y))
    return blocked


def inflate_blocked_cells(occupancy_map, inflate_radius_m, block_unknown=True):
    resolution = occupancy_map.metadata.resolution
    inflation_cells = int(math.ceil(inflate_radius_m / resolution))
    source_blocked = base_blocked_cells(occupancy_map, block_unknown=block_unknown)
    inflated = set(source_blocked)

    if inflation_cells <= 0:
        return inflated, inflation_cells

    for blocked_x, blocked_y in source_blocked:
        for dy in range(-inflation_cells, inflation_cells + 1):
            for dx in range(-inflation_cells, inflation_cells + 1):
                if dx * dx + dy * dy > inflation_cells * inflation_cells:
                    continue
                cell = (blocked_x + dx, blocked_y + dy)
                if in_bounds(occupancy_map, cell):
                    inflated.add(cell)

    return inflated, inflation_cells


def is_traversable(occupancy_map, blocked, cell):
    return in_bounds(occupancy_map, cell) and cell not in blocked


def snap_to_traversable(occupancy_map, blocked, requested_cell, snap_radius_m):
    snap_cells = int(math.ceil(snap_radius_m / occupancy_map.metadata.resolution))
    req_x, req_y = requested_cell
    best = None
    best_distance_sq = None

    for dy in range(-snap_cells, snap_cells + 1):
        for dx in range(-snap_cells, snap_cells + 1):
            if dx * dx + dy * dy > snap_cells * snap_cells:
                continue
            cell = (req_x + dx, req_y + dy)
            if not is_traversable(occupancy_map, blocked, cell):
                continue
            distance_sq = dx * dx + dy * dy
            if best is None or distance_sq < best_distance_sq:
                best = cell
                best_distance_sq = distance_sq

    if best is None:
        raise ValueError(
            "Could not snap requested cell "
            f"{requested_cell} to traversable space within {snap_radius_m:.3f} m"
        )

    return best


def movement_cost(a, b, resolution):
    return math.hypot(b[0] - a[0], b[1] - a[1]) * resolution


def neighbors_8_no_corner_cutting(occupancy_map, blocked, cell):
    x, y = cell
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = (x + dx, y + dy)
            if not is_traversable(occupancy_map, blocked, neighbor):
                continue
            if dx != 0 and dy != 0:
                side_a = (x + dx, y)
                side_b = (x, y + dy)
                if (
                    not is_traversable(occupancy_map, blocked, side_a)
                    or not is_traversable(occupancy_map, blocked, side_b)
                ):
                    continue
            yield neighbor


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(occupancy_map, blocked, start, goal):
    if not is_traversable(occupancy_map, blocked, start):
        raise ValueError(f"Start cell is blocked or outside map: {start}")
    if not is_traversable(occupancy_map, blocked, goal):
        raise ValueError(f"Goal cell is blocked or outside map: {goal}")

    resolution = occupancy_map.metadata.resolution
    queue = []
    heapq.heappush(queue, (0.0, 0, start))
    came_from = {}
    g_score = {start: 0.0}
    tie_breaker = 0

    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors_8_no_corner_cutting(occupancy_map, blocked, current):
            tentative_g = g_score[current] + movement_cost(current, neighbor, resolution)
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heuristic = movement_cost(neighbor, goal, resolution)
            tie_breaker += 1
            heapq.heappush(queue, (tentative_g + heuristic, tie_breaker, neighbor))

    raise ValueError(f"No path exists from {start} to {goal}")


def direction_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    gcd = math.gcd(abs(dx), abs(dy))
    if gcd == 0:
        return 0, 0
    return dx // gcd, dy // gcd


def simplify_path(path):
    if len(path) <= 2:
        return list(path)

    waypoints = [path[0]]
    previous_direction = direction_between(path[0], path[1])
    for index in range(2, len(path)):
        current_direction = direction_between(path[index - 1], path[index])
        if current_direction != previous_direction:
            waypoints.append(path[index - 1])
            previous_direction = current_direction
    waypoints.append(path[-1])
    return waypoints


def path_length_m(path, metadata):
    if len(path) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(path)):
        total += movement_cost(path[index - 1], path[index], metadata.resolution)
    return total


def world_path_length_m(points):
    points = list(points)
    if len(points) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(points)):
        total += math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
    return total


def build_path_rows(path, metadata):
    rows = []
    cumulative = 0.0
    previous = None
    for index, cell in enumerate(path):
        if previous is None:
            segment = 0.0
        else:
            segment = movement_cost(previous, cell, metadata.resolution)
            cumulative += segment
        world_x, world_y = grid_to_world(cell[0], cell[1], metadata)
        rows.append([
            index,
            cell[0],
            cell[1],
            world_x,
            world_y,
            segment,
            cumulative,
        ])
        previous = cell
    return rows


def build_world_path_rows(points, metadata):
    rows = []
    cumulative = 0.0
    previous = None
    for index, point in enumerate(points):
        world_x = float(point[0])
        world_y = float(point[1])
        grid_x, grid_y = world_to_grid(world_x, world_y, metadata)
        if previous is None:
            segment = 0.0
        else:
            segment = math.hypot(world_x - previous[0], world_y - previous[1])
            cumulative += segment
        rows.append([
            index,
            grid_x,
            grid_y,
            world_x,
            world_y,
            segment,
            cumulative,
        ])
        previous = (world_x, world_y)
    return rows


def sampled_segment_cells(occupancy_map, start_world, end_world):
    start_x, start_y = start_world
    end_x, end_y = end_world
    distance_m = math.hypot(end_x - start_x, end_y - start_y)
    step_m = max(occupancy_map.metadata.resolution * 0.5, 1e-6)
    steps = max(1, int(math.ceil(distance_m / step_m)))
    cells = []
    previous = None
    for index in range(steps + 1):
        ratio = index / float(steps)
        world_x = start_x + (end_x - start_x) * ratio
        world_y = start_y + (end_y - start_y) * ratio
        cell = world_to_grid(world_x, world_y, occupancy_map.metadata)
        if cell != previous:
            cells.append(cell)
            previous = cell
    return cells


def world_segment_is_clear(occupancy_map, blocked, start_world, end_world):
    for cell in sampled_segment_cells(occupancy_map, start_world, end_world):
        if not is_traversable(occupancy_map, blocked, cell):
            return False
    return True


def cell_segment_is_clear(occupancy_map, blocked, start_cell, end_cell):
    start_world = grid_to_world(start_cell[0], start_cell[1], occupancy_map.metadata)
    end_world = grid_to_world(end_cell[0], end_cell[1], occupancy_map.metadata)
    return world_segment_is_clear(occupancy_map, blocked, start_world, end_world)


def shortcut_cell_path(occupancy_map, blocked, path):
    path = list(path)
    if len(path) < 2:
        raise ValueError("tracking smoothing needs at least two path cells")
    for cell in path:
        if not is_traversable(occupancy_map, blocked, cell):
            raise ValueError(f"tracking smoothing path cell is blocked: {cell}")

    shortcut = [path[0]]
    index = 0
    while index < len(path) - 1:
        next_index = index + 1
        for candidate in range(len(path) - 1, index, -1):
            if cell_segment_is_clear(
                occupancy_map,
                blocked,
                path[index],
                path[candidate],
            ):
                next_index = candidate
                break
        shortcut.append(path[next_index])
        index = next_index
    return shortcut


def shortcut_world_path(occupancy_map, blocked, points):
    points = [(float(point[0]), float(point[1])) for point in points]
    if len(points) < 2:
        raise ValueError("tracking smoothing needs at least two world points")
    for point in points:
        cell = world_to_grid(point[0], point[1], occupancy_map.metadata)
        if not is_traversable(occupancy_map, blocked, cell):
            raise ValueError(
                "tracking smoothing point is blocked or outside map: "
                f"x={point[0]:.3f}, y={point[1]:.3f}, cell={cell}"
            )

    shortcut = [points[0]]
    index = 0
    while index < len(points) - 1:
        next_index = index + 1
        for candidate in range(len(points) - 1, index, -1):
            if world_segment_is_clear(
                occupancy_map,
                blocked,
                points[index],
                points[candidate],
            ):
                next_index = candidate
                break
        shortcut.append(points[next_index])
        index = next_index
    return shortcut


def resample_world_path(points, spacing_m):
    points = [(float(point[0]), float(point[1])) for point in points]
    spacing_m = float(spacing_m)
    if spacing_m <= 0.0:
        raise ValueError("tracking smoothing spacing must be greater than zero")
    if len(points) < 2:
        raise ValueError("tracking smoothing needs at least two world points")

    resampled = [points[0]]
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        distance_m = math.hypot(end[0] - start[0], end[1] - start[1])
        if distance_m <= 1e-12:
            continue
        steps = max(1, int(math.ceil(distance_m / spacing_m)))
        for step in range(1, steps + 1):
            ratio = step / float(steps)
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            duplicate_distance_m = math.hypot(
                point[0] - resampled[-1][0],
                point[1] - resampled[-1][1],
            )
            if duplicate_distance_m > 1e-12:
                resampled.append(point)

    if len(resampled) < 2:
        raise ValueError("tracking smoothing collapsed to fewer than two points")
    endpoint_distance_m = math.hypot(
        resampled[-1][0] - points[-1][0],
        resampled[-1][1] - points[-1][1],
    )
    if endpoint_distance_m > 1e-9:
        resampled.append(points[-1])
    return resampled


def smooth_cell_path_for_tracking(occupancy_map, blocked, path, spacing_m):
    raw_points = [
        grid_to_world(cell[0], cell[1], occupancy_map.metadata)
        for cell in path
    ]
    shortcut_cells = shortcut_cell_path(occupancy_map, blocked, path)
    shortcut_points = [
        grid_to_world(cell[0], cell[1], occupancy_map.metadata)
        for cell in shortcut_cells
    ]
    smoothed = resample_world_path(shortcut_points, spacing_m)
    return TrackingPathSmoothingResult(
        points=smoothed,
        status="smoothed",
        raw_point_count=len(raw_points),
        smoothed_point_count=len(smoothed),
        raw_length_m=world_path_length_m(raw_points),
        smoothed_length_m=world_path_length_m(smoothed),
    )


def smooth_world_path_for_tracking(occupancy_map, blocked, points, spacing_m):
    raw_points = [(float(point[0]), float(point[1])) for point in points]
    shortcut_points = shortcut_world_path(occupancy_map, blocked, raw_points)
    smoothed = resample_world_path(shortcut_points, spacing_m)
    return TrackingPathSmoothingResult(
        points=smoothed,
        status="smoothed",
        raw_point_count=len(raw_points),
        smoothed_point_count=len(smoothed),
        raw_length_m=world_path_length_m(raw_points),
        smoothed_length_m=world_path_length_m(smoothed),
    )


def write_path_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "index",
            "grid_x",
            "grid_y",
            "world_x_m",
            "world_y_m",
            "segment_distance_m",
            "cumulative_distance_m",
        ])
        writer.writerows(rows)


def render_planner_pixels(occupancy_map, inflated_blocked, path, waypoints):
    base_blocked = base_blocked_cells(occupancy_map, block_unknown=True)
    path_cells = set(path)
    waypoint_cells = set(waypoints)
    start = path[0] if path else None
    goal = path[-1] if path else None

    rows = []
    for image_row in range(occupancy_map.height):
        row = []
        for image_col in range(occupancy_map.width):
            grid_x, grid_y = image_to_grid(image_col, image_row, occupancy_map.height)
            cell = (grid_x, grid_y)
            cell_state = occupancy_map.cells[grid_y][grid_x]

            if cell_state == CELL_OCCUPIED:
                color = COLOR_OCCUPIED
            elif cell_state == CELL_UNKNOWN:
                color = COLOR_UNKNOWN
            else:
                color = COLOR_FREE

            if cell in inflated_blocked and cell not in base_blocked:
                color = COLOR_INFLATED
            if cell in path_cells:
                color = COLOR_PATH
            if cell in waypoint_cells:
                color = COLOR_WAYPOINT
            if cell == start:
                color = COLOR_START
            if cell == goal:
                color = COLOR_GOAL

            row.append(color)
        rows.append(row)
    return rows


def write_ppm(path, pixel_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(pixel_rows)
    width = len(pixel_rows[0]) if height else 0
    with path.open("wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for row in pixel_rows:
            for red, green, blue in row:
                file.write(bytes((red, green, blue)))


def cell_to_pgm_pixel(cell, metadata):
    if metadata.negate:
        if cell == CELL_OCCUPIED:
            return 255
        if cell == CELL_FREE:
            return 1
        probability = (metadata.occupied_thresh + metadata.free_thresh) / 2.0
        return int(round(255.0 * probability))
    if cell == CELL_OCCUPIED:
        return 0
    if cell == CELL_FREE:
        return 254
    probability = (metadata.occupied_thresh + metadata.free_thresh) / 2.0
    return int(round(255.0 * (1.0 - probability)))


def occupancy_map_to_pgm_image(occupancy_map):
    pixels = []
    for image_row in range(occupancy_map.height):
        row = []
        for image_col in range(occupancy_map.width):
            grid_x, grid_y = image_to_grid(
                image_col,
                image_row,
                occupancy_map.height,
            )
            row.append(cell_to_pgm_pixel(
                occupancy_map.cells[grid_y][grid_x],
                occupancy_map.metadata,
            ))
        pixels.append(row)
    return PgmImage(
        width=occupancy_map.width,
        height=occupancy_map.height,
        maxval=255,
        pixels=pixels,
    )


def write_pgm(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(f"P5\n{image.width} {image.height}\n{image.maxval}\n".encode("ascii"))
        for row in image.pixels:
            file.write(bytes(row))


def write_map_yaml(path, occupancy_map, image_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    image_value = image_path.name if image_path.parent == path.parent else str(image_path)
    origin = occupancy_map.metadata.origin
    lines = [
        f"image: {image_value}",
        f"mode: {occupancy_map.metadata.mode}",
        f"resolution: {occupancy_map.metadata.resolution}",
        f"origin: [{origin[0]}, {origin[1]}, {origin[2]}]",
        f"negate: {occupancy_map.metadata.negate}",
        f"occupied_thresh: {occupancy_map.metadata.occupied_thresh}",
        f"free_thresh: {occupancy_map.metadata.free_thresh}",
        "",
    ]
    path.write_text("\n".join(lines))


def write_occupancy_map_copy(occupancy_map, yaml_path, pgm_path):
    write_pgm(pgm_path, occupancy_map_to_pgm_image(occupancy_map))
    write_map_yaml(yaml_path, occupancy_map, pgm_path)


def snapped_distance_m(requested_world, snapped_world):
    return math.hypot(
        snapped_world[0] - requested_world[0],
        snapped_world[1] - requested_world[1],
    )


def plan_path(
    occupancy_map,
    start_world,
    goal_world,
    inflate_radius_m=DEFAULT_INFLATE_RADIUS_M,
    snap_radius_m=DEFAULT_SNAP_RADIUS_M,
):
    inflated_blocked, inflation_cells = inflate_blocked_cells(
        occupancy_map,
        inflate_radius_m,
        block_unknown=True,
    )
    start_requested = world_to_grid(start_world[0], start_world[1], occupancy_map.metadata)
    goal_requested = world_to_grid(goal_world[0], goal_world[1], occupancy_map.metadata)
    start_cell = snap_to_traversable(
        occupancy_map,
        inflated_blocked,
        start_requested,
        snap_radius_m,
    )
    goal_cell = snap_to_traversable(
        occupancy_map,
        inflated_blocked,
        goal_requested,
        snap_radius_m,
    )
    path = astar(occupancy_map, inflated_blocked, start_cell, goal_cell)
    waypoints = simplify_path(path)

    return (
        PlanResult(
            path=path,
            waypoints=waypoints,
            start_requested_world=start_world,
            goal_requested_world=goal_world,
            start_cell=start_cell,
            goal_cell=goal_cell,
            start_snapped_world=grid_to_world(start_cell[0], start_cell[1], occupancy_map.metadata),
            goal_snapped_world=grid_to_world(goal_cell[0], goal_cell[1], occupancy_map.metadata),
            path_length_m=path_length_m(path, occupancy_map.metadata),
        ),
        inflated_blocked,
        inflation_cells,
    )


def print_diagnostics(occupancy_map, inflated_blocked, inflation_cells, args, result):
    counts = count_cells(occupancy_map)
    min_x, max_x, min_y, max_y = world_bounds(occupancy_map)
    print("Map diagnostics:")
    print(f"  Map: {occupancy_map.metadata.yaml_path}")
    print(f"  Size: {occupancy_map.width} x {occupancy_map.height} cells")
    print(f"  Resolution: {occupancy_map.metadata.resolution:.3f} m/cell")
    print(f"  World bounds x: [{min_x:.3f}, {max_x:.3f}] m")
    print(f"  World bounds y: [{min_y:.3f}, {max_y:.3f}] m")
    print(f"  Free cells: {counts[CELL_FREE]}")
    print(f"  Occupied cells: {counts[CELL_OCCUPIED]}")
    print(f"  Unknown cells: {counts[CELL_UNKNOWN]}")
    print(f"  Inflated blocked cells: {len(inflated_blocked)}")
    print(
        "  Inflation radius: "
        f"{args.inflate_radius_m:.3f} m = {inflation_cells} cells"
    )
    print(
        "  Start requested: "
        f"({result.start_requested_world[0]:.3f}, {result.start_requested_world[1]:.3f}) m"
    )
    print(
        "  Start snapped: "
        f"cell {result.start_cell}, "
        f"world ({result.start_snapped_world[0]:.3f}, "
        f"{result.start_snapped_world[1]:.3f}) m"
    )
    print(
        "  Goal requested: "
        f"({result.goal_requested_world[0]:.3f}, {result.goal_requested_world[1]:.3f}) m"
    )
    print(
        "  Goal snapped: "
        f"cell {result.goal_cell}, "
        f"world ({result.goal_snapped_world[0]:.3f}, "
        f"{result.goal_snapped_world[1]:.3f}) m"
    )
    print(f"  Dense path cells: {len(result.path)}")
    print(f"  Simplified waypoints: {len(result.waypoints)}")
    print(f"  Path length: {result.path_length_m:.3f} m")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Plan an offline A* path over a ROS trinary map.",
    )
    parser.add_argument("--map", default=DEFAULT_MAP, type=Path, help="ROS map YAML.")
    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        required=True,
        help="Requested start position in map/world meters.",
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        required=True,
        help="Requested goal position in map/world meters.",
    )
    parser.add_argument(
        "--inflate-radius-m",
        default=DEFAULT_INFLATE_RADIUS_M,
        type=float,
        help="Obstacle/unknown inflation radius in meters.",
    )
    parser.add_argument(
        "--snap-radius-m",
        default=DEFAULT_SNAP_RADIUS_M,
        type=float,
        help="Maximum start/goal snapping radius in meters.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        type=Path,
        help="Dense path CSV output.",
    )
    parser.add_argument(
        "--output-waypoints-csv",
        default=DEFAULT_OUTPUT_WAYPOINTS_CSV,
        type=Path,
        help="Simplified waypoint CSV output.",
    )
    parser.add_argument(
        "--output-ppm",
        default=DEFAULT_OUTPUT_PPM,
        type=Path,
        help="PPM visual output.",
    )
    args = parser.parse_args(argv)

    if args.inflate_radius_m < 0.0:
        parser.error("--inflate-radius-m must be non-negative")
    if args.snap_radius_m < 0.0:
        parser.error("--snap-radius-m must be non-negative")

    args.start = (float(args.start[0]), float(args.start[1]))
    args.goal = (float(args.goal[0]), float(args.goal[1]))
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        occupancy_map = load_occupancy_map(args.map)
        result, inflated_blocked, inflation_cells = plan_path(
            occupancy_map,
            args.start,
            args.goal,
            inflate_radius_m=args.inflate_radius_m,
            snap_radius_m=args.snap_radius_m,
        )
        write_path_csv(args.output_csv, build_path_rows(result.path, occupancy_map.metadata))
        write_path_csv(
            args.output_waypoints_csv,
            build_path_rows(result.waypoints, occupancy_map.metadata),
        )
        write_ppm(
            args.output_ppm,
            render_planner_pixels(
                occupancy_map,
                inflated_blocked,
                result.path,
                result.waypoints,
            ),
        )
        print_diagnostics(occupancy_map, inflated_blocked, inflation_cells, args, result)
        print(f"Wrote dense path CSV: {args.output_csv}")
        print(f"Wrote waypoint CSV: {args.output_waypoints_csv}")
        print(f"Wrote visual PPM: {args.output_ppm}")
        return 0
    except Exception as exc:
        print(f"map_path_planner.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
