from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import lidar_obstacle_map
import map_path_planner

from .models import LookaheadGuardResult, Waypoint
from .path_curves import polyline_lookahead_target


LOOKAHEAD_GUARD_OFF = "off"
LOOKAHEAD_GUARD_STATIC_MAP = "static-map"
LOOKAHEAD_GUARD_STATIC_AND_RUN_LOCAL = "static-and-run-local"
LOOKAHEAD_GUARD_MODES = (
    LOOKAHEAD_GUARD_OFF,
    LOOKAHEAD_GUARD_STATIC_MAP,
    LOOKAHEAD_GUARD_STATIC_AND_RUN_LOCAL,
)


@dataclass(frozen=True)
class LookaheadCandidate:
    target: tuple[float, float]
    distance_m: float
    index: int


def _rounded_point(point):
    return (round(float(point[0]), 3), round(float(point[1]), 3))


def dense_route_signature(points):
    points = [(float(point[0]), float(point[1])) for point in points]
    if not points:
        return (0, None, None, None, "")
    digest = hashlib.sha1()
    for point in points:
        digest.update(f"{point[0]:.3f},{point[1]:.3f};".encode("ascii"))
    middle = points[len(points) // 2]
    return (
        len(points),
        _rounded_point(points[0]),
        _rounded_point(middle),
        _rounded_point(points[-1]),
        digest.hexdigest()[:16],
    )


def guard_block_signature(pose, tracking_points):
    return (
        round(float(pose.x), 2),
        round(float(pose.y), 2),
        dense_route_signature((point.x, point.y) for point in tracking_points),
    )


def map_geometry_signature(occupancy_map):
    metadata = occupancy_map.metadata
    return (
        occupancy_map.width,
        occupancy_map.height,
        round(float(metadata.resolution), 9),
        tuple(round(float(value), 9) for value in metadata.origin),
    )


def ensure_run_local_grid_compatible(static_map, run_local_map):
    if run_local_map is None:
        return
    run_static_map = getattr(run_local_map, "static_map", None)
    if run_static_map is None:
        raise RuntimeError("lookahead_guard_run_local_map_missing_static_map")
    if map_geometry_signature(static_map) != map_geometry_signature(run_static_map):
        raise RuntimeError("lookahead_guard_run_local_map_grid_mismatch")


def static_inflated_blocked_cells(occupancy_map, inflation_radius_m):
    blocked_cells, _inflation_cells = map_path_planner.inflate_blocked_cells(
        occupancy_map,
        inflation_radius_m,
        block_unknown=True,
    )
    return blocked_cells


def candidate_step_m(min_lookahead_m):
    return min(0.05, max(0.02, float(min_lookahead_m) / 2.0))


def lookahead_candidates(
    path_points,
    current_point,
    nominal_target,
    nominal_lookahead_m,
    min_lookahead_m,
    final_goal=None,
    distance_to_goal_m=None,
):
    candidates = []
    seen = set()

    def add_candidate(target, distance_m):
        rounded = _rounded_point(target)
        if rounded in seen:
            return
        seen.add(rounded)
        candidates.append(
            LookaheadCandidate(
                target=(float(target[0]), float(target[1])),
                distance_m=float(distance_m),
                index=len(candidates),
            )
        )

    add_candidate(nominal_target, nominal_lookahead_m)
    step_m = candidate_step_m(min_lookahead_m)
    distance_m = float(nominal_lookahead_m) - step_m
    while distance_m >= float(min_lookahead_m) - 1e-9:
        add_candidate(
            polyline_lookahead_target(path_points, current_point, distance_m),
            distance_m,
        )
        distance_m -= step_m

    if (
        final_goal is not None
        and distance_to_goal_m is not None
        and distance_to_goal_m <= float(min_lookahead_m)
    ):
        add_candidate((final_goal.x, final_goal.y), distance_to_goal_m)

    return candidates


class LookaheadGuard:
    def __init__(
        self,
        occupancy_map,
        static_blocked_cells,
        mode=LOOKAHEAD_GUARD_STATIC_MAP,
        run_local_map_fn: Callable[[], object | None] | None = None,
    ):
        if mode not in LOOKAHEAD_GUARD_MODES or mode == LOOKAHEAD_GUARD_OFF:
            raise ValueError(f"unsupported lookahead guard mode: {mode!r}")
        self.occupancy_map = occupancy_map
        self.static_blocked_cells = set(static_blocked_cells)
        self.mode = mode
        self.run_local_map_fn = run_local_map_fn

    @classmethod
    def from_static_map(
        cls,
        static_map_path,
        static_inflation_radius_m,
        mode=LOOKAHEAD_GUARD_STATIC_MAP,
        run_local_map_fn=None,
    ):
        occupancy_map = map_path_planner.load_occupancy_map(static_map_path)
        blocked_cells = static_inflated_blocked_cells(
            occupancy_map,
            static_inflation_radius_m,
        )
        return cls(
            occupancy_map,
            blocked_cells,
            mode=mode,
            run_local_map_fn=run_local_map_fn,
        )

    def blocked_cells(self):
        blocked = set(self.static_blocked_cells)
        if self.mode != LOOKAHEAD_GUARD_STATIC_AND_RUN_LOCAL:
            return blocked
        run_local_map = self.run_local_map_fn() if self.run_local_map_fn else None
        if run_local_map is None:
            return blocked
        ensure_run_local_grid_compatible(self.occupancy_map, run_local_map)
        blocked.update(getattr(run_local_map, "inflated_obstacle_cells", set()))
        return blocked

    def candidate_blocked_cells(self, current_pose, target, blocked_cells):
        end = Waypoint(-1, float(target[0]), float(target[1]))
        cells = lidar_obstacle_map.rasterized_segment_cells(
            self.occupancy_map,
            current_pose,
            end,
            radius_m=0.0,
        )
        return cells.intersection(blocked_cells)

    def select_target(
        self,
        current_pose,
        path_points,
        current_point,
        nominal_target,
        nominal_lookahead_m,
        min_lookahead_m,
        final_goal=None,
        distance_to_goal_m=None,
    ):
        candidates = lookahead_candidates(
            path_points,
            current_point,
            nominal_target,
            nominal_lookahead_m,
            min_lookahead_m,
            final_goal=final_goal,
            distance_to_goal_m=distance_to_goal_m,
        )
        blocked_cells = self.blocked_cells()
        last_blocked_count = 0
        for candidate in candidates:
            blocked = self.candidate_blocked_cells(
                current_pose,
                candidate.target,
                blocked_cells,
            )
            last_blocked_count = len(blocked)
            if blocked:
                continue
            status = "clear" if candidate.index == 0 else "shortened"
            return candidate.target, LookaheadGuardResult(
                safe=True,
                status=status,
                reason="ok",
                candidate_count=len(candidates),
                blocked_cell_count=0,
                selected_target_index=candidate.index,
                selected_target_distance_m=candidate.distance_m,
                selected_target_x_m=candidate.target[0],
                selected_target_y_m=candidate.target[1],
            )

        return nominal_target, LookaheadGuardResult(
            safe=False,
            status="blocked",
            reason="pure_pursuit_lookahead_blocked",
            candidate_count=len(candidates),
            blocked_cell_count=last_blocked_count,
            selected_target_index=None,
            selected_target_distance_m=None,
            selected_target_x_m=None,
            selected_target_y_m=None,
        )
