"""Mapped coverage and physical approach geometry without persisted route protocols."""
from __future__ import annotations
import ast
import hashlib
import heapq
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Mapping, Sequence
from .model import Pose2D, Candidate, StoppedState, PlannedLeg, finite, wrap_angle

CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN = 0, 1, 2
_EPSILON = 1e-10
CELL_SOURCE_STATIC_OCCUPIED='static_occupied'
CELL_SOURCE_UNKNOWN='unknown'
CELL_SOURCE_ARENA_BOUNDARY='arena_boundary'
CELL_SOURCE_INFLATED='inflated'
CELL_SOURCE_RUN_LOCAL='run_local'
@dataclass(frozen=True, order=True)
class GridCell:
    x: int
    y: int


DEFAULT_ARENA_LENGTH_M=3.90
DEFAULT_ARENA_WIDTH_M=1.898
DEFAULT_ARENA_CENTER_X_M=0.0
DEFAULT_ARENA_CENTER_Y_M=0.0
DEFAULT_ARENA_YAW_DEG=0.0
DEFAULT_ARENA_MARGIN_M=0.0

@dataclass(frozen=True)
class ArenaBounds:
    length_m: float = DEFAULT_ARENA_LENGTH_M
    width_m: float = DEFAULT_ARENA_WIDTH_M
    center_x_m: float = DEFAULT_ARENA_CENTER_X_M
    center_y_m: float = DEFAULT_ARENA_CENTER_Y_M
    yaw_deg: float = DEFAULT_ARENA_YAW_DEG
    margin_m: float = DEFAULT_ARENA_MARGIN_M

    def validate(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError("arena length must be positive")
        if self.width_m <= 0.0:
            raise ValueError("arena width must be positive")
        if self.margin_m < 0.0:
            raise ValueError("arena margin must be non-negative")
        if self.margin_m * 2.0 >= self.length_m:
            raise ValueError("arena margin leaves no usable arena length")
        if self.margin_m * 2.0 >= self.width_m:
            raise ValueError("arena margin leaves no usable arena width")

    def contains(self, pose: Pose2D) -> bool:
        return self.boundary_clearance_m(pose) >= self.margin_m

    def boundary_clearance_m(self, pose: Pose2D) -> float:
        """Return signed clearance from a pose to the configured rectangle.

        Positive values are inside the configured rectangle, zero is on its
        boundary, and negative values are outside.  Keeping this independent
        of ``margin_m`` lets perception reject wall returns while route
        planning can continue to apply its own placement margin.
        """

        yaw = math.radians(self.yaw_deg)
        dx = pose.x_m - self.center_x_m
        dy = pose.y_m - self.center_y_m
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        x_clearance = self.length_m / 2.0 - abs(local_x)
        y_clearance = self.width_m / 2.0 - abs(local_y)
        return min(x_clearance, y_clearance)

    def to_metadata(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "yaw_deg": self.yaw_deg,
            "margin_m": self.margin_m,
        }
@dataclass(frozen=True)
class MapMetadata:
    yaml_path: Path
    image_path: Path
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    mode: str


@dataclass(frozen=True)
class PgmImage:
    width: int
    height: int
    maxval: int
    pixels: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class OccupancyGrid:
    metadata: MapMetadata
    width: int
    height: int
    cells: Tuple[Tuple[int, ...], ...]


def strip_inline_comment(line: str) -> str:
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


def parse_yaml_scalar(text: str):
    text = text.strip()
    if not text:
        return ""
    if text[0] in {"'", '"'} or text.startswith("["):
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


def _read_simple_yaml_bytes(path: Path, raw: bytes) -> Dict[str, object]:
    data: Dict[str, object] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 YAML") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = strip_inline_comment(line).strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected 'key: value'")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in data:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        data[key] = parse_yaml_scalar(value)
    return data


def _read_map_metadata_bytes(yaml_path: Path, raw: bytes) -> MapMetadata:
    data = _read_simple_yaml_bytes(yaml_path, raw)
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
        raise ValueError("only trinary maps are supported")

    origin = data["origin"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError(f"{yaml_path} origin must be [x, y, yaw]")
    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
    if abs(origin_tuple[2]) > 1e-12:
        raise ValueError("only zero-yaw map origins are supported")

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


def _next_pgm_token(data: bytes, index: int) -> Tuple[str, int]:
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
        raise ValueError("unexpected end of PGM header")

    start = index
    while index < length:
        byte = data[index]
        if byte == ord("#") or chr(byte).isspace():
            break
        index += 1
    return data[start:index].decode("ascii"), index


def _skip_pgm_whitespace_and_comments(data: bytes, index: int) -> int:
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


def _read_pgm_bytes(path: Path, data: bytes) -> PgmImage:
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
        values: List[int] = []
        while len(values) < width * height:
            token, index = _next_pgm_token(data, index)
            values.append(int(token))
        if any(value < 0 or value > maxval for value in values):
            raise ValueError(f"{path} contains a pixel outside 0..{maxval}")
    else:
        if index >= len(data) or not chr(data[index]).isspace():
            raise ValueError("missing PGM binary raster separator")
        index += 2 if data[index:index+2] == b"\r\n" else 1
        expected = width * height
        values = list(data[index:index + expected])
        if len(values) != expected:
            raise ValueError(f"{path} has incomplete binary pixel data")

    rows = tuple(
        tuple(values[row_start:row_start + width])
        for row_start in range(0, width * height, width)
    )
    return PgmImage(width=width, height=height, maxval=maxval, pixels=rows)


def pixel_to_cell(pixel: int, metadata: MapMetadata, maxval: int = 255) -> int:
    if metadata.negate:
        probability = pixel / maxval
    else:
        probability = (maxval - pixel) / maxval
    if probability >= metadata.occupied_thresh:
        return CELL_OCCUPIED
    if probability <= metadata.free_thresh:
        return CELL_FREE
    return CELL_UNKNOWN


def build_occupancy_grid(metadata: MapMetadata, image: PgmImage) -> OccupancyGrid:
    rows = [[CELL_UNKNOWN for _ in range(image.width)] for _ in range(image.height)]
    for image_row in range(image.height):
        grid_y = image.height - 1 - image_row
        for image_col in range(image.width):
            rows[grid_y][image_col] = pixel_to_cell(
                image.pixels[image_row][image_col],
                metadata,
                maxval=image.maxval,
            )
    return OccupancyGrid(
        metadata=metadata,
        width=image.width,
        height=image.height,
        cells=tuple(tuple(row) for row in rows),
    )
@dataclass(frozen=True)
class Costmap:
    metadata: MapMetadata
    width: int
    height: int
    cells: Tuple[Tuple[int, ...], ...]
    blocked_cells: frozenset[GridCell]
    cell_sources: Mapping[GridCell, str]

    @classmethod
    def from_occupancy_grid(cls, grid: OccupancyGrid, block_unknown: bool = True) -> "Costmap":
        blocked = set()
        sources = {}
        for y, row in enumerate(grid.cells):
            for x, value in enumerate(row):
                cell = GridCell(x, y)
                if value == CELL_OCCUPIED:
                    blocked.add(cell)
                    sources[cell] = CELL_SOURCE_STATIC_OCCUPIED
                elif value == CELL_UNKNOWN and block_unknown:
                    blocked.add(cell)
                    sources[cell] = CELL_SOURCE_UNKNOWN
        return cls(
            metadata=grid.metadata,
            width=grid.width,
            height=grid.height,
            cells=grid.cells,
            blocked_cells=frozenset(blocked),
            cell_sources=sources,
        )

    @property
    def resolution(self) -> float:
        return self.metadata.resolution

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell.x < self.width and 0 <= cell.y < self.height

    def world_to_grid(self, pose_or_x: Pose2D | float, y_m: float | None = None) -> GridCell:
        if isinstance(pose_or_x, Pose2D):
            x_m = pose_or_x.x_m
            y_value = pose_or_x.y_m
        else:
            if y_m is None:
                raise ValueError("y_m is required when x is passed directly")
            x_m = float(pose_or_x)
            y_value = float(y_m)
        origin_x, origin_y, _origin_yaw = self.metadata.origin
        return GridCell(
            math.floor((x_m - origin_x) / self.metadata.resolution),
            math.floor((y_value - origin_y) / self.metadata.resolution),
        )

    def grid_to_world(self, cell: GridCell, yaw_rad: float = 0.0) -> Pose2D:
        origin_x, origin_y, _origin_yaw = self.metadata.origin
        return Pose2D(
            origin_x + (cell.x + 0.5) * self.metadata.resolution,
            origin_y + (cell.y + 0.5) * self.metadata.resolution,
            yaw_rad,
        )


    def is_blocked(self, cell: GridCell) -> bool:
        return not self.in_bounds(cell) or cell in self.blocked_cells

    def is_traversable(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and cell not in self.blocked_cells

    def with_blocked_cells(
        self,
        cells: Iterable[GridCell],
        source: str = CELL_SOURCE_RUN_LOCAL,
    ) -> "Costmap":
        blocked = set(self.blocked_cells)
        sources = dict(self.cell_sources)
        for cell in cells:
            if not self.in_bounds(cell):
                continue
            blocked.add(cell)
            sources[cell] = source
        return replace(self, blocked_cells=frozenset(blocked), cell_sources=sources)

    def with_arena_bounds(self, arena_bounds: ArenaBounds) -> "Costmap":
        """Block map cells whose centres are outside the navigable arena.

        ``ArenaBounds.contains`` includes the configured arena margin.  Only
        previously traversable cells receive the arena-boundary source so an
        occupied or blocked-unknown cell keeps its original provenance.
        Apply this overlay before :meth:`with_inflation` to turn the physical
        arena edge into an interior clearance band.
        """

        arena_bounds.validate()
        if not all(
            math.isfinite(value)
            for value in (
                arena_bounds.length_m,
                arena_bounds.width_m,
                arena_bounds.center_x_m,
                arena_bounds.center_y_m,
                arena_bounds.yaw_deg,
                arena_bounds.margin_m,
            )
        ):
            raise ValueError("arena bounds values must be finite")

        outside_cells = (
            cell
            for y in range(self.height)
            for x in range(self.width)
            if (cell := GridCell(x, y)) not in self.blocked_cells
            and not arena_bounds.contains(self.grid_to_world(cell))
        )
        return self.with_blocked_cells(
            outside_cells,
            source=CELL_SOURCE_ARENA_BOUNDARY,
        )

    def with_inflation(self, radius_m: float) -> "Costmap":
        if not math.isfinite(radius_m) or radius_m < 0.0:
            raise ValueError("inflation radius must be finite and non-negative")
        radius = radius_m
        if radius <= 0.0:
            return self
        # Occupied grid cells represent areas, not point obstacles at their
        # centres.  Block every candidate cell whose axis-aligned square can
        # come within ``radius`` of an occupied/unknown cell square.  A
        # centre-distance disk under-inflates diagonal cell boundaries.
        inflation_cells = int(math.ceil(radius / self.metadata.resolution)) + 1
        inflated = set()
        for blocked in self.blocked_cells:
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    clearance_x_m = (
                        max(abs(dx) - 1, 0) * self.metadata.resolution
                    )
                    clearance_y_m = (
                        max(abs(dy) - 1, 0) * self.metadata.resolution
                    )
                    if math.hypot(clearance_x_m, clearance_y_m) > (
                        radius + 1.0e-12
                    ):
                        continue
                    cell = GridCell(blocked.x + dx, blocked.y + dy)
                    if self.in_bounds(cell) and cell not in self.blocked_cells:
                        inflated.add(cell)
        return self.with_blocked_cells(inflated, source=CELL_SOURCE_INFLATED)
def movement_cost(a: GridCell, b: GridCell, resolution: float) -> float:
    return math.hypot(b.x - a.x, b.y - a.y) * resolution


def _neighbors_8_no_corner_cutting(costmap: Costmap, cell: GridCell) -> Iterable[GridCell]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = GridCell(cell.x + dx, cell.y + dy)
            if not costmap.is_traversable(neighbor):
                continue
            if dx != 0 and dy != 0:
                side_a = GridCell(cell.x + dx, cell.y)
                side_b = GridCell(cell.x, cell.y + dy)
                if not costmap.is_traversable(side_a) or not costmap.is_traversable(side_b):
                    continue
            yield neighbor


def _snap_to_traversable(costmap: Costmap, requested: GridCell, snap_radius_m: float) -> GridCell | None:
    snap_cells = int(math.ceil(max(0.0, snap_radius_m) / costmap.resolution))
    best = None
    best_distance_sq = None
    for dy in range(-snap_cells, snap_cells + 1):
        for dx in range(-snap_cells, snap_cells + 1):
            if dx * dx + dy * dy > snap_cells * snap_cells:
                continue
            cell = GridCell(requested.x + dx, requested.y + dy)
            if not costmap.is_traversable(cell):
                continue
            distance_sq = dx * dx + dy * dy
            if best is None or distance_sq < best_distance_sq or (
                distance_sq == best_distance_sq and cell < best
            ):
                best = cell
                best_distance_sq = distance_sq
    return best


def _reconstruct_path(came_from: Dict[GridCell, GridCell], current: GridCell) -> List[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _astar(costmap: Costmap, start: GridCell, goal: GridCell) -> Tuple[List[GridCell] | None, int]:
    queue: List[Tuple[float, int, GridCell]] = []
    heapq.heappush(queue, (0.0, 0, start))
    came_from: Dict[GridCell, GridCell] = {}
    g_score = {start: 0.0}
    tie_breaker = 0
    expanded = 0

    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        expanded += 1
        if current == goal:
            return _reconstruct_path(came_from, current), expanded
        for neighbor in _neighbors_8_no_corner_cutting(costmap, current):
            tentative_g = g_score[current] + movement_cost(current, neighbor, costmap.resolution)
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heuristic = movement_cost(neighbor, goal, costmap.resolution)
            tie_breaker += 1
            heapq.heappush(queue, (tentative_g + heuristic, tie_breaker, neighbor))
    return None, expanded


def _finite_pose(pose: Pose2D, *, name: str) -> None:
    if not math.isfinite(pose.x_m) or not math.isfinite(pose.y_m):
        raise ValueError(f"{name} position must be finite")


def _distance_point_to_cell_square(costmap: Costmap, pose: Pose2D, cell: GridCell) -> float:
    origin_x, origin_y, _ = costmap.metadata.origin
    x0 = origin_x + cell.x * costmap.resolution
    y0 = origin_y + cell.y * costmap.resolution
    x1 = x0 + costmap.resolution
    y1 = y0 + costmap.resolution
    dx = max(x0 - pose.x_m, 0.0, pose.x_m - x1)
    dy = max(y0 - pose.y_m, 0.0, pose.y_m - y1)
    return math.hypot(dx, dy)


def point_clearance_to_blocked_m(costmap: Costmap, pose: Pose2D) -> float:
    """Return a conservative free-disk radius around an exact world pose."""

    _finite_pose(pose, name="clearance pose")
    containing = costmap.world_to_grid(pose)
    if not costmap.is_traversable(containing):
        return 0.0
    origin_x, origin_y, _ = costmap.metadata.origin
    map_max_x = origin_x + costmap.width * costmap.resolution
    map_max_y = origin_y + costmap.height * costmap.resolution
    clearance = min(
        pose.x_m - origin_x,
        map_max_x - pose.x_m,
        pose.y_m - origin_y,
        map_max_y - pose.y_m,
    )
    for cell in costmap.blocked_cells:
        clearance = min(clearance, _distance_point_to_cell_square(costmap, pose, cell))
        if clearance <= 0.0:
            return 0.0
    # Segment/cell checks use closed squares.  Stay strictly inside the free
    # disk so a join at the advertised limit cannot touch a blocked boundary.
    return max(0.0, clearance - 1.0e-6)


def _finite_route_pose(pose: Pose2D, *, name: str) -> None:
    if not isinstance(pose, Pose2D):
        raise ValueError(f"{name} must be a Pose2D")
    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m)):
        raise ValueError(f"{name} position must be finite")


def _segment_intersects_closed_cell(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
    cell: GridCell,
) -> bool:
    origin_x, origin_y, _ = costmap.metadata.origin
    x_min = origin_x + cell.x * costmap.resolution
    y_min = origin_y + cell.y * costmap.resolution
    x_max = x_min + costmap.resolution
    y_max = y_min + costmap.resolution
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    t_min = 0.0
    t_max = 1.0
    for coordinate, delta, lower, upper in (
        (start.x_m, dx, x_min, x_max),
        (start.y_m, dy, y_min, y_max),
    ):
        if abs(delta) <= _EPSILON:
            if coordinate < lower - _EPSILON or coordinate > upper + _EPSILON:
                return False
            continue
        near = (lower - coordinate) / delta
        far = (upper - coordinate) / delta
        if near > far:
            near, far = far, near
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max + _EPSILON:
            return False
    return True


def supercover_segment_cells(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
) -> tuple[GridCell, ...]:
    """Return every grid square touched by a closed world-space segment.

    Closed-square intersection includes both sides of a grid boundary and all
    four cells at a corner, so shortcuts cannot graze occupied geometry.
    """

    _finite_route_pose(start, name="segment start")
    _finite_route_pose(end, name="segment end")
    origin_x, origin_y, _ = costmap.metadata.origin
    resolution = costmap.resolution
    min_x = math.floor((min(start.x_m, end.x_m) - origin_x) / resolution) - 1
    max_x = math.floor((max(start.x_m, end.x_m) - origin_x) / resolution) + 1
    min_y = math.floor((min(start.y_m, end.y_m) - origin_y) / resolution) - 1
    max_y = math.floor((max(start.y_m, end.y_m) - origin_y) / resolution) + 1
    touched = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = GridCell(x, y)
            if _segment_intersects_closed_cell(costmap, start, end, cell):
                touched.append(cell)
    return tuple(sorted(set(touched)))


def segment_is_collision_free(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
) -> bool:
    """Validate a segment against an already configuration-space costmap."""

    return all(
        costmap.is_traversable(cell)
        for cell in supercover_segment_cells(costmap, start, end)
    )


def greedy_line_of_sight_retained_indices(
    costmap: Costmap,
    poses: Sequence[Pose2D],
    *,
    protected_indices: Iterable[int] = (),
) -> tuple[int, ...]:
    """Return a greedy collision-free subsequence of polyline indices.

    ``protected_indices`` divide the route into independent smoothing spans.
    They are retained even when adjacent protected vertices have the same
    position, which preserves semantic zero-length handoffs and certified
    exact-start connector anchors.
    """

    if not poses:
        return ()
    for index, pose in enumerate(poses):
        _finite_route_pose(pose, name=f"poses[{index}]")
    protected = {0, len(poses) - 1}
    for value in protected_indices:
        if type(value) is not int or not 0 <= value < len(poses):
            raise ValueError("protected route index is outside the polyline")
        protected.add(value)

    retained = [0]
    for boundary in sorted(protected)[1:]:
        anchor = retained[-1]
        while anchor < boundary:
            selected = None
            for candidate in range(boundary, anchor, -1):
                if segment_is_collision_free(
                    costmap,
                    poses[anchor],
                    poses[candidate],
                ):
                    selected = candidate
                    break
            if selected is None:
                raise ValueError("input polyline contains a colliding segment")
            retained.append(selected)
            anchor = selected
    return tuple(retained)
def _requested_boustrophedon_sequences(
    arena: ArenaBounds,
    config: CoverageConfig,
) -> tuple[tuple[Pose2D, ...], ...]:
    edge_clearance = (
        arena.margin_m + config.inflation_radius_m + config.minimum_boundary_clearance_m
    )
    usable_length = arena.length_m - 2.0 * edge_clearance
    usable_width = arena.width_m - 2.0 * edge_clearance
    if usable_length <= 0.0 or usable_width <= 0.0:
        raise ValueError("inflation and margins leave no survey corridor")
    stop_count = max(2, int(math.ceil(usable_length / config.stop_spacing_m)) + 1)
    x_values = tuple(
        -usable_length / 2.0 + usable_length * index / (stop_count - 1)
        for index in range(stop_count)
    )
    lane_values = tuple(
        -usable_width / 2.0
        + usable_width * (index + 0.5) / config.lane_count
        for index in range(config.lane_count)
    )

    def sequence(
        lane_order: tuple[float, ...],
        *,
        reverse_first: bool,
    ) -> tuple[Pose2D, ...]:
        requested = []
        for lane_index, local_y in enumerate(lane_order):
            reverse = reverse_first if lane_index % 2 == 0 else not reverse_first
            lane_x = tuple(reversed(x_values)) if reverse else x_values
            requested.extend(
                _arena_local_pose(arena, local_x, local_y)
                for local_x in lane_x
            )
        return tuple(requested)

    forward_lanes = lane_values
    reverse_lanes = tuple(reversed(lane_values))
    return (
        sequence(forward_lanes, reverse_first=False),
        sequence(forward_lanes, reverse_first=True),
        sequence(reverse_lanes, reverse_first=False),
        sequence(reverse_lanes, reverse_first=True),
    )


def _arena_local_pose(arena: ArenaBounds, local_x: float, local_y: float) -> Pose2D:
    yaw = math.radians(arena.yaw_deg)
    return Pose2D(
        arena.center_x_m + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
        arena.center_y_m + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
        yaw,
    )


def _snap_sequence(
    costmap: Costmap,
    requested: Iterable[Pose2D],
    snap_radius_m: float,
) -> tuple[GridCell, ...]:
    cells = []
    for pose in requested:
        snapped = _nearest_traversable_cell(
            costmap,
            costmap.world_to_grid(pose),
            snap_radius_m,
        )
        if snapped is not None and (not cells or cells[-1] != snapped):
            cells.append(snapped)
    return tuple(cells)


def _nearest_traversable_cell(
    costmap: Costmap,
    requested: GridCell,
    radius_m: float,
) -> GridCell | None:
    radius_cells = int(math.ceil(radius_m / costmap.resolution))
    options = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            cell = GridCell(requested.x + dx, requested.y + dy)
            if costmap.is_traversable(cell):
                options.append((dx * dx + dy * dy, cell))
    return min(options)[1] if options else None


def _cell_distance_m(costmap: Costmap, pose: Pose2D, cell: GridCell) -> float:
    world = costmap.grid_to_world(cell)
    return math.hypot(pose.x_m - world.x_m, pose.y_m - world.y_m)


def _visible_cells(
    costmap: Costmap,
    origin: GridCell,
    surveyable_cells: tuple[GridCell, ...],
    radius_m: float,
) -> tuple[GridCell, ...]:
    radius_cells = radius_m / costmap.resolution
    visible = []
    for target in surveyable_cells:
        if math.hypot(target.x - origin.x, target.y - origin.y) > (
            radius_cells + 1.0e-12
        ):
            continue
        line = _bresenham_cells(origin, target)
        if any(costmap.is_blocked(cell) for cell in line[1:-1]):
            continue
        visible.append(target)
    return tuple(sorted(visible))


def _bresenham_cells(start: GridCell, end: GridCell) -> tuple[GridCell, ...]:
    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append(GridCell(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return tuple(cells)

@dataclass(frozen=True)
class MapGrid:
    metadata: MapMetadata
    width: int
    height: int
    cells: tuple[tuple[int, ...], ...]
    map_id: str


def load_map(path: Path, *, expected_yaml_sha256: str, expected_image_sha256: str,
             expected_map_bundle_sha256: str, semantic_map_id: str,
             planning_frame: str = "map") -> MapGrid:
    """Parse the same frozen bytes that are bound by the physical site config."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("map YAML must be a regular nonsymlink file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_yaml_sha256:
        raise ValueError("map YAML differs from physical site")
    metadata = _read_map_metadata_bytes(path, raw)
    image_path = metadata.image_path
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError("map image must be a regular nonsymlink file")
    image_raw = image_path.read_bytes()
    if hashlib.sha256(image_raw).hexdigest() != expected_image_sha256:
        raise ValueError("map image differs from physical site")
    image = _read_pgm_bytes(image_path, image_raw)
    for number in (*metadata.origin, metadata.resolution, metadata.free_thresh, metadata.occupied_thresh):
        finite(number, "map geometry")
    if (metadata.resolution <= 0 or metadata.negate not in (0, 1)
            or not 0 <= metadata.free_thresh < metadata.occupied_thresh <= 1):
        raise ValueError("invalid map geometry or thresholds")
    bundle = dict(schema_version=1, semantic_map_id=semantic_map_id,
        planning_frame=planning_frame, yaml_sha256=expected_yaml_sha256,
        image_sha256=expected_image_sha256, resolution=metadata.resolution,
        origin=list(metadata.origin), negate=metadata.negate,
        occupied_thresh=metadata.occupied_thresh, free_thresh=metadata.free_thresh,
        mode=metadata.mode, width=image.width, height=image.height, maxval=image.maxval)
    identity = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if identity != expected_map_bundle_sha256:
        raise ValueError("frozen map identity differs from physical site")
    grid = build_occupancy_grid(metadata, image)
    return MapGrid(metadata, grid.width, grid.height, grid.cells, identity)


@dataclass(frozen=True)
class CoverageConfig:
    lane_count: int = 1
    stop_spacing_m: float = .7
    visibility_radius_m: float = 1.35
    inflation_radius_m: float = .25
    snap_radius_m: float = .30
    minimum_boundary_clearance_m: float = .10
    coverage_threshold: float = .95


@dataclass(frozen=True)
class Viewpoint:
    pose: Pose2D
    visible_cells: tuple[GridCell, ...]


@dataclass(frozen=True)
class CoveragePlan:
    viewpoints: tuple[Viewpoint, ...]
    surveyable_cells: tuple[GridCell, ...]
    map_id: str
    threshold: float

    def coverage_ratio(self, visited_indices: Iterable[int]) -> float:
        indices = tuple(visited_indices)
        if any(type(i) is not int or not 0 <= i < len(self.viewpoints) for i in indices):
            raise ValueError("coverage progress index invalid")
        seen = {cell for i in indices for cell in self.viewpoints[i].visible_cells}
        return len(seen) / len(self.surveyable_cells)


def coverage_plan(grid: MapGrid, *, start: Pose2D, stop_spacing_m=.7, lane_count=1,
                  inflation_radius_m=.25, coverage_threshold=.95,
                  visibility_radius_m=1.35, arena: ArenaBounds | None = None) -> CoveragePlan:
    arena = arena or ArenaBounds()
    arena.validate()
    cfg = CoverageConfig(lane_count, stop_spacing_m, visibility_radius_m,
                         inflation_radius_m, .30, .10, coverage_threshold)
    if type(lane_count) is not int or lane_count < 1:
        raise ValueError("coverage lane count must be positive")
    for name in ('stop_spacing_m', 'visibility_radius_m', 'inflation_radius_m', 'coverage_threshold'):
        if finite(getattr(cfg, name), name) <= 0:
            raise ValueError("coverage dimensions must be positive")
    if coverage_threshold > 1:
        raise ValueError("coverage threshold exceeds one")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(arena)
    space = base.with_inflation(inflation_radius_m)
    sequences = tuple(_snap_sequence(space, seq, cfg.snap_radius_m)
                      for seq in _requested_boustrophedon_sequences(arena, cfg))
    sequences = tuple(seq for seq in sequences if seq)
    if not sequences:
        raise ValueError("no traversable coverage viewpoints")
    cells = min(sequences, key=lambda seq: (_cell_distance_m(space, start, seq[0]),
                                            tuple((cell.x, cell.y) for cell in seq)))
    surveyable = tuple(sorted(cell for y in range(base.height) for x in range(base.width)
        if base.is_traversable(cell := GridCell(x, y))
        and arena.boundary_clearance_m(base.grid_to_world(cell)) + 1e-12 >= cfg.minimum_boundary_clearance_m))
    if not surveyable:
        raise ValueError("map has no surveyable cells")
    views = []
    for i, cell in enumerate(cells):
        other = cells[i+1] if i+1 < len(cells) else cells[i-1] if i else cell
        direction = 1 if i+1 < len(cells) else -1
        yaw = math.atan2(direction*(other.y-cell.y), direction*(other.x-cell.x)) if other != cell else start.yaw_rad
        views.append(Viewpoint(space.grid_to_world(cell, yaw),
                               _visible_cells(base, cell, surveyable, visibility_radius_m)))
    result = CoveragePlan(tuple(views), surveyable, grid.map_id, coverage_threshold)
    if result.coverage_ratio(range(len(views))) + 1e-12 < coverage_threshold:
        raise ValueError("mapped survey cannot meet coverage threshold")
    return result


def _distance_to_segment(x, y, a, b):
    dx, dy = b.x_m-a.x_m, b.y_m-a.y_m
    square = dx*dx+dy*dy
    t = 0. if square <= 1e-18 else max(0., min(1., ((x-a.x_m)*dx+(y-a.y_m)*dy)/square))
    return math.hypot(x-a.x_m-t*dx, y-a.y_m-t*dy)


def _keepout_space(space: Costmap, keepouts) -> Costmap:
    blocked = []
    for x, y, radius in keepouts:
        for value in (x, y, radius):
            finite(value, "candidate keepout")
        if radius <= 0:
            raise ValueError("keepout radius must be positive")
        center = space.world_to_grid(x, y)
        count = math.ceil(radius/space.resolution)+1
        for dy in range(-count, count+1):
            for dx in range(-count, count+1):
                cell = GridCell(center.x+dx, center.y+dy)
                if space.in_bounds(cell) and _distance_point_to_cell_square(space, Pose2D(x, y), cell) <= radius:
                    blocked.append(cell)
    return space.with_blocked_cells(blocked)


def route_clearance(base: Costmap, poses: Sequence[Pose2D], *, spacing_m=.01,
                    obstacles=()) -> float:
    """Lipschitz lower bound of raw free-disk radius, including stand bodies."""
    if not poses or finite(spacing_m, "clearance sample spacing") <= 0:
        raise ValueError("invalid route clearance input")
    result = math.inf
    pairs = tuple(zip(poses, poses[1:])) or ((poses[0], poses[0]),)
    for a, b in pairs:
        length = math.hypot(b.x_m-a.x_m, b.y_m-a.y_m)
        count = max(1, math.ceil(length/spacing_m))
        sample_min = math.inf
        for i in range(count+1):
            p = Pose2D(a.x_m+(b.x_m-a.x_m)*i/count, a.y_m+(b.y_m-a.y_m)*i/count)
            clearance = point_clearance_to_blocked_m(base, p)
            for x, y, radius in obstacles:
                clearance = min(clearance, math.hypot(p.x_m-x, p.y_m-y)-radius)
            sample_min = min(sample_min, clearance)
        result = min(result, sample_min-length/(2*count))
    return max(0., result)


def _route(base, space, start, goal, *, inflation_radius_m, keepouts):
    if not base.in_bounds(base.world_to_grid(start)) or not space.is_traversable(space.world_to_grid(goal)):
        raise ValueError("route start or goal outside admitted map")
    first = _snap_to_traversable(space, space.world_to_grid(start), .30)
    if first is None:
        raise ValueError("route start has no traversable grid join")
    cells, _ = _astar(space, first, space.world_to_grid(goal))
    if not cells:
        raise ValueError("no mapped route")
    poses = tuple(space.grid_to_world(c) for c in cells)
    if len(poses) > 1 and first == base.world_to_grid(start) and math.hypot(poses[0].x_m-start.x_m, poses[0].y_m-start.y_m)>1e-10:
        poses = poses[1:]
    if math.hypot(poses[0].x_m-start.x_m, poses[0].y_m-start.y_m)>1e-10:
        if route_clearance(base, (start, poses[0]))+1e-9 < inflation_radius_m:
            raise ValueError("exact start connector lacks physical clearance")
        for x,y,r in keepouts:
            if _distance_to_segment(x,y,start,poses[0]) <= r:
                raise ValueError("exact start connector crosses candidate keepout")
        poses = (start,)+poses
    else:
        poses = (start,)+poses[1:]
    if len(poses) < 2:
        # A stationary final-heading leg still owns a fresh stopped epoch.
        poses = (start, Pose2D(start.x_m, start.y_m, goal.yaw_rad))
    protected = (1,) if not segment_is_collision_free(space,start,start) else ()
    indices = greedy_line_of_sight_retained_indices(space, poses[1:] if protected else poses)
    poses = ((start,)+tuple(poses[1:][i] for i in indices)) if protected else tuple(poses[i] for i in indices)
    return (*poses[:-1], Pose2D(poses[-1].x_m, poses[-1].y_m, goal.yaw_rad))


def plan_to_pose(grid: MapGrid, stopped: StoppedState, goal: Pose2D, *, target_id: str,
                 kind="coverage", inflation_radius_m=.25, keepouts=(),
                 arena: ArenaBounds | None = None, obstacles=()) -> PlannedLeg:
    if stopped.map_id != grid.map_id:
        raise ValueError("stopped state belongs to another map")
    if finite(inflation_radius_m, "inflation radius") <= 0:
        raise ValueError("physical routes require positive inflation")
    finite(goal.yaw_rad, "goal yaw")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(arena or ArenaBounds())
    space = _keepout_space(base.with_inflation(inflation_radius_m), keepouts)
    poses = _route(base, space, stopped.pose_map, goal,
                   inflation_radius_m=inflation_radius_m, keepouts=keepouts)
    clearance = route_clearance(base, poses, obstacles=obstacles)
    if clearance + 1e-9 < inflation_radius_m:
        raise ValueError("route lacks continuous physical clearance")
    return PlannedLeg(kind, target_id, grid.map_id, stopped.epoch, tuple(poses),
                      minimum_clearance_m=clearance)


def camera_facing_yaw(position: Pose2D, target: Candidate, *, offset_xy_m=(0.,0.), heading_offset_rad=0.) -> float:
    """Solve the measured optical ray, retaining the forward-looking solution."""
    if len(offset_xy_m)!=2:
        raise ValueError("camera offset requires two planar coordinates")
    tx,ty=(finite(v,'camera offset') for v in offset_xy_m)
    phi=finite(heading_offset_rad,'camera optical heading')
    dx,dy=target.x_m-position.x_m,target.y_m-position.y_m
    distance=math.hypot(dx,dy)
    cross=tx*math.sin(phi)-ty*math.cos(phi)
    if distance<=math.hypot(tx,ty) or abs(cross)>=distance:
        raise ValueError("target cannot be aligned with measured camera ray")
    yaw=wrap_angle(math.atan2(dy,dx)-phi+math.asin(cross/distance))
    bx,by=math.cos(yaw)*dx+math.sin(yaw)*dy-tx,-math.sin(yaw)*dx+math.cos(yaw)*dy-ty
    if bx*math.cos(phi)+by*math.sin(phi)<=0:
        raise ValueError("target is behind measured camera ray")
    return yaw


def plan_candidate(grid: MapGrid, stopped: StoppedState, candidate: Candidate, *,
                   candidates=(), approach_offset_m=.5, robot_radius_m=.105,
                   collision_margin_m=.01, tracking_margin_m=.03,
                   normal_rad=None, normal_half_width_rad=None,
                   minimum_standoff_m=None, inflation_radius_m=.25,
                   candidate_keepout_radius_m=.34, arena=None,
                   camera_offset_xy_m=(0.,0.), camera_heading_offset_rad=0.) -> PlannedLeg:
    """Rank bounded rasterization-equivalent endpoints, including opposite proof."""
    for name,value in (('approach offset',approach_offset_m),('robot radius',robot_radius_m),
                       ('collision margin',collision_margin_m),('tracking margin',tracking_margin_m)):
        if finite(value,name) < 0 or (name in ('approach offset','robot radius') and value == 0):
            raise ValueError("invalid approach geometry")
    body = candidate.radius_m+candidate.uncertainty_m
    minimum = body+robot_radius_m+collision_margin_m+tracking_margin_m
    if minimum_standoff_m is not None:
        minimum = max(minimum,finite(minimum_standoff_m,'minimum standoff'))
    if approach_offset_m <= max(minimum,candidate_keepout_radius_m)+grid.metadata.resolution/math.sqrt(2):
        raise ValueError("approach offset is inside physical stand keepout")
    if normal_rad is None:
        normal = math.atan2(stopped.pose_map.y_m-candidate.y_m,stopped.pose_map.x_m-candidate.x_m)
    else:
        normal = finite(normal_rad,'opposite normal')
        half = finite(normal_half_width_rad,'orientation half width')
        if not 0 <= half <= math.radians(15):
            raise ValueError("opposite view needs a bounded fifteen-degree interval")
        distance = math.hypot(stopped.pose_map.x_m-candidate.x_m,stopped.pose_map.y_m-candidate.y_m)
        if distance <= candidate.uncertainty_m:
            raise ValueError("opposite observing side is unresolved")
        side = math.atan2(stopped.pose_map.y_m-candidate.y_m,stopped.pose_map.x_m-candidate.x_m)
        if abs(wrap_angle(normal-side-math.pi))+half+math.asin(candidate.uncertainty_m/distance) > math.pi/3+1e-12:
            raise ValueError("orientation does not resolve every plausible opposite face")
    requested = Pose2D(candidate.x_m+math.cos(normal)*approach_offset_m,
                       candidate.y_m+math.sin(normal)*approach_offset_m, wrap_angle(normal+math.pi))
    population = {c.uid:c for c in (*tuple(candidates),candidate)}
    obstacles = tuple((c.x_m,c.y_m,c.radius_m+c.uncertainty_m) for c in population.values())
    keepouts = tuple((x,y,max(candidate_keepout_radius_m,r+robot_radius_m+collision_margin_m+tracking_margin_m)) for x,y,r in obstacles)
    base = Costmap.from_occupancy_grid(grid)
    center = base.world_to_grid(requested)
    options = []
    for y in range(center.y-1,center.y+2):
        for x in range(center.x-1,center.x+2):
            cell=GridCell(x,y)
            if not base.in_bounds(cell):
                continue
            goal=base.grid_to_world(cell)
            error=math.hypot(goal.x_m-requested.x_m,goal.y_m-requested.y_m)
            if error > base.resolution/math.sqrt(2)+1e-9:
                continue
            distance=math.hypot(goal.x_m-candidate.x_m,goal.y_m-candidate.y_m)
            if distance+1e-9 < minimum:
                continue
            yaw=camera_facing_yaw(goal,candidate,offset_xy_m=camera_offset_xy_m,heading_offset_rad=camera_heading_offset_rad)
            if normal_rad is not None:
                reserve=.03+candidate.uncertainty_m
                tx,ty=camera_offset_xy_m
                camera_x=goal.x_m+math.cos(yaw)*tx-math.sin(yaw)*ty
                camera_y=goal.y_m+math.sin(yaw)*tx+math.cos(yaw)*ty
                view_distance=math.hypot(camera_x-candidate.x_m,camera_y-candidate.y_m)
                obliquity=abs(wrap_angle(math.atan2(camera_y-candidate.y_m,camera_x-candidate.x_m)-normal))
                if view_distance<=reserve or obliquity+half+math.asin(reserve/view_distance)>math.radians(20)+1e-12:
                    continue
            goal=Pose2D(goal.x_m,goal.y_m,yaw)
            try:
                leg=plan_to_pose(grid,stopped,goal,target_id=candidate.uid,kind='front' if normal_rad is None else 'opposite',
                    inflation_radius_m=inflation_radius_m,keepouts=keepouts,obstacles=obstacles,arena=arena)
            except ValueError:
                continue
            length=sum(math.hypot(b.x_m-a.x_m,b.y_m-a.y_m) for a,b in zip(leg.poses,leg.poses[1:]))
            options.append((leg.minimum_clearance_m,error,length,x,y,leg))
    if not options:
        raise ValueError("no physically admissible candidate approach")
    best=max(option[0] for option in options)
    winner=min((o for o in options if best-o[0] <= .001+1e-9),key=lambda o:o[1:5])
    return replace(winner[-1],target_center=(candidate.x_m,candidate.y_m))
