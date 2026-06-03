from __future__ import annotations

import math

from .geometry_candidates import generate_raw_candidates
from .models import ActiveExploreConfig, ActiveExplorePlan
from .path_planning import blocked_distance_field
from .scan_grid import build_local_grid
from .scoring import plan_candidate
from .shadow_frontiers import generate_obstacle_shadow_frontier_candidates


def plan_active_explore_recovery(
    result,
    scan,
    robot_pose,
    config: ActiveExploreConfig,
    origin_yaw_rad=0.0,
    grid=None,
):
    raw_candidates, reason = generate_raw_candidates(
        result,
        scan,
        robot_pose,
        config,
        origin_yaw_rad=origin_yaw_rad,
    )
    if reason != "ok":
        return ActiveExplorePlan(False, reason, None, (), None)
    grid = grid or build_local_grid(scan, robot_pose, config)
    clearance_distance_field = blocked_distance_field(grid)
    frontier_candidates = generate_obstacle_shadow_frontier_candidates(
        grid,
        config,
        clearance_distance_field=clearance_distance_field,
    )
    raw_candidates = tuple(frontier_candidates) + tuple(raw_candidates)
    candidates = tuple(
        plan_candidate(
            raw,
            grid,
            config,
            clearance_distance_field=clearance_distance_field,
        )
        for raw in raw_candidates
    )
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return ActiveExplorePlan(False, "no_reachable_candidate", None, candidates, grid)
    selected = max(
        accepted,
        key=lambda candidate: (
            candidate.score if candidate.score is not None else -math.inf
        ),
    )
    return ActiveExplorePlan(True, "selected", selected, candidates, grid)

