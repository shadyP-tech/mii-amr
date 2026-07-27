"""Immutable costmap operations for Aufgabe 04 dry-run navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Tuple

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.map_io import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.stations.models import ApproachTarget, Station


CELL_SOURCE_FREE = "free"
CELL_SOURCE_STATIC_OCCUPIED = "static_occupied"
CELL_SOURCE_UNKNOWN = "unknown"
CELL_SOURCE_INFLATED = "inflated"
CELL_SOURCE_ARENA_BOUNDARY = "arena_boundary"
CELL_SOURCE_STATION_KEEPOUT = "station_keepout"
CELL_SOURCE_RUN_LOCAL = "run_local"


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

    def cell_value(self, cell: GridCell) -> int:
        if not self.in_bounds(cell):
            raise ValueError(f"cell outside map bounds: {cell}")
        return self.cells[cell.y][cell.x]

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

    def with_station_keepouts(
        self,
        stations_or_targets: Iterable[Station | ApproachTarget],
    ) -> "Costmap":
        keepout_cells = []
        for item in stations_or_targets:
            if isinstance(item, Station):
                center_pose = Pose2D(item.pose.x_m, item.pose.y_m, item.pose.yaw_rad)
                radius_m = item.keepout_radius_m
            else:
                center_pose = Pose2D(item.pose.x_m, item.pose.y_m, item.pose.yaw_rad)
                radius_m = item.stop_distance_m
            center = self.world_to_grid(center_pose)
            radius_cells = int(math.ceil(max(0.0, radius_m) / self.metadata.resolution))
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    cell = GridCell(center.x + dx, center.y + dy)
                    if self.in_bounds(cell):
                        keepout_cells.append(cell)
        return self.with_blocked_cells(keepout_cells, source=CELL_SOURCE_STATION_KEEPOUT)


def occupancy_counts(costmap: Costmap) -> dict[int, int]:
    counts = {CELL_FREE: 0, CELL_OCCUPIED: 0, CELL_UNKNOWN: 0}
    for row in costmap.cells:
        for value in row:
            counts[value] += 1
    return counts
