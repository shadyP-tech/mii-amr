from __future__ import annotations

from .grid import cell_to_world, in_bounds, world_to_cell
from .math_utils import clamp
from .models import CELL_UNKNOWN, ActiveExploreCandidate, ActiveExploreConfig
from .path_planning import (
    astar,
    blocked_distance_field,
    nearest_blocked_distance_m,
    path_length_m,
    path_soft_clearance_penalty,
    simplify_path_cells,
    traversable,
    turn_count_for_path,
    unknown_ratio_near_path,
)
from .shadow_frontiers import shadow_information_gain_components


def score_candidate(
    raw,
    grid,
    path,
    config,
    clearance_distance_field=None,
    path_length=None,
    turn_count=None,
):
    length = (
        path_length_m(path, grid.resolution_m)
        if path_length is None
        else path_length
    )
    turns = turn_count_for_path(path) if turn_count is None else turn_count
    clearance = nearest_blocked_distance_m(grid, path, clearance_distance_field)
    clearance_component = clamp(clearance / 0.50, 0.0, 1.0)
    soft_clearance_penalty = (
        0.0
        if clearance_distance_field is None
        else path_soft_clearance_penalty(path, clearance_distance_field, config)
    )
    path_unknown_ratio = unknown_ratio_near_path(grid, path)
    viewpoint_cell = path[-1]
    (
        shadow_information_gain,
        visible_shadow_unknown_count,
        total_shadow_unknown_count,
    ) = shadow_information_gain_components(grid, viewpoint_cell)
    components = {
        "geometry_progress": raw.geometry_progress,
        "heater_potential": raw.heater_potential,
        "clearance_margin": clearance_component,
        "path_length": length,
        "turn_count": turns,
        "path_unknown_ratio": path_unknown_ratio,
        "path_min_clearance_m": clearance,
        "path_soft_clearance_penalty": soft_clearance_penalty,
        "shadow_information_gain": shadow_information_gain,
        "visible_shadow_unknown_count": visible_shadow_unknown_count,
        "total_shadow_unknown_count": total_shadow_unknown_count,
    }
    score = (
        5.0 * components["shadow_information_gain"]
        + 2.0 * components["heater_potential"]
        + 1.5 * components["clearance_margin"]
        + 0.75 * components["geometry_progress"]
        - 1.0 * components["path_length"]
        - 0.5 * components["turn_count"]
        - 4.0 * components["path_unknown_ratio"]
        - 2.0 * components["path_soft_clearance_penalty"]
    )
    return score, components


def plan_candidate(
    raw,
    grid,
    config: ActiveExploreConfig,
    clearance_distance_field=None,
):
    target_cell = world_to_cell(grid, raw.target_x, raw.target_y)
    if not in_bounds(grid, target_cell):
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason="goal_outside_grid",
            metadata=raw.metadata,
        )
    if not traversable(grid, target_cell, config.unknown_blocked):
        reason = (
            "goal_unknown"
            if grid.cells[target_cell[1]][target_cell[0]] == CELL_UNKNOWN
            else "goal_blocked"
        )
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason=reason,
            metadata=raw.metadata,
        )
    try:
        clearance_distance_field = clearance_distance_field or blocked_distance_field(grid)
        path = astar(
            grid,
            grid.robot_cell,
            target_cell,
            config.unknown_blocked,
            clearance_distance_field=clearance_distance_field,
            config=config,
        )
    except ValueError as exc:
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason=str(exc),
            metadata=raw.metadata,
        )
    length = path_length_m(path, grid.resolution_m)
    simplified = simplify_path_cells(path)
    segment_count = max(0, len(simplified) - 1)
    turn_count = max(0, segment_count - 1)
    max_candidate_path_m = (
        config.max_candidate_path_m
        if config.max_candidate_path_m is not None
        else config.max_total_distance_m
    )
    if length > max_candidate_path_m:
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason="path_too_long",
            path_cells=tuple(path),
            path_world=tuple(cell_to_world(grid, cell) for cell in path),
            path_length_m=length,
            metadata={
                **raw.metadata,
                "path_length_limit_m": max_candidate_path_m,
            },
        )
    if segment_count > config.max_path_segments:
        return ActiveExploreCandidate(
            raw.kind,
            raw.target_x,
            raw.target_y,
            raw.heading_rad,
            False,
            rejection_reason="path_too_many_segments",
            path_cells=tuple(path),
            path_world=tuple(cell_to_world(grid, cell) for cell in path),
            simplified_path_world=tuple(
                cell_to_world(grid, cell) for cell in simplified
            ),
            path_length_m=length,
            turn_count=turn_count,
            metadata={
                **raw.metadata,
                "path_segment_count": segment_count,
                "path_segment_limit": config.max_path_segments,
                "turn_count": turn_count,
                "path_length_m": length,
            },
        )
    score, components = score_candidate(
        raw,
        grid,
        path,
        config,
        clearance_distance_field,
        path_length=length,
        turn_count=turn_count,
    )
    return ActiveExploreCandidate(
        raw.kind,
        raw.target_x,
        raw.target_y,
        raw.heading_rad,
        True,
        score=score,
        score_components=components,
        path_cells=tuple(path),
        path_world=tuple(cell_to_world(grid, cell) for cell in path),
        simplified_path_world=tuple(
            cell_to_world(grid, cell) for cell in simplified
        ),
        path_length_m=length,
        turn_count=turn_count,
        metadata=raw.metadata,
    )

