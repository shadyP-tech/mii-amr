from __future__ import annotations

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

