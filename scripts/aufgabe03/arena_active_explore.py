#!/usr/bin/env python3
"""
Compatibility facade for ROS-free active-explore planning helpers.

The implementation lives in arena_active_explore_core. This module preserves
the historical import surface used by tests, runtime modules, and debug tools.
"""

from arena_active_explore_core.geometry_candidates import (
    candidate_heater_score,
    candidate_profile_valid,
    candidate_range,
    generate_raw_candidates,
    geometry_is_recoverable,
    min_scan_range_in_sector,
)
from arena_active_explore_core.grid import (
    bresenham_cells,
    cell_to_world,
    grid_cell_value,
    in_bounds,
    inflated_cells_for,
    mark_scan_ray,
    set_cell,
    world_to_cell,
)
from arena_active_explore_core.math_utils import (
    clamp,
    normalize_angle_rad,
    point_from_heading,
    valid_range,
    yaw_rad_from_pose,
)
from arena_active_explore_core.models import (
    CELL_FREE,
    CELL_INFLATED,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    FAILURE_POSE_NOT_UNIQUE,
    FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE,
    ActiveExploreCandidate,
    ActiveExploreConfig,
    ActiveExplorePlan,
    LocalGrid,
    RawCandidate,
    grid_cell_counts,
)
from arena_active_explore_core.path_planning import (
    astar,
    blocked_cells,
    blocked_distance_field,
    clearance_distance_for_cell,
    direction_between,
    movement_cost,
    nearest_blocked_distance_m,
    neighbors_8,
    path_length_m,
    path_soft_clearance_penalty,
    reconstruct_path,
    simplify_path_cells,
    soft_clearance_cell_penalty,
    traversable,
    turn_count_for_path,
    unknown_ratio_near_path,
)
from arena_active_explore_core.planner import plan_active_explore_recovery
from arena_active_explore_core.scan_grid import (
    build_local_grid,
    build_local_grid_from_scan_samples,
    build_observed_local_grid,
    build_observed_local_grid_from_scan_samples,
    empty_local_grid,
    finalize_grid,
    mark_scan_on_grid,
)
from arena_active_explore_core.scoring import plan_candidate, score_candidate
from arena_active_explore_core.shadow_frontiers import (
    cluster_centroid_world,
    cluster_shadow_unknown_cells,
    generate_obstacle_shadow_frontier_candidates,
    nearest_cluster_distance_m,
    obstacle_shadow_unknown_cells,
    shadow_cell_visible_from,
    shadow_information_gain_components,
    visible_cluster_shadow_cells,
)


__all__ = [name for name in globals() if not name.startswith("_")]

