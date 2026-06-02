from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from arena_active_explore import ActiveExploreConfig

from .models import (
    ACTIVE_EXPLORE_PHASE_SHADOW,
    ArenaActiveSpinConfig,
    SectorClearance,
)


def spin_diagnostics_template():
    return {
        "target_rad": 2.0 * math.pi,
        "accumulated_rad": 0.0,
        "duration_sec": 0.0,
        "timeout": False,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_diagnostics_json(path: Path | str | None, diagnostics):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(json_safe(diagnostics), file, indent=2, sort_keys=True)
        file.write("\n")
    return str(path)


def config_diagnostics(config: ArenaActiveSpinConfig):
    data = asdict(config)
    data["diagnostics_path"] = str(config.diagnostics_path)
    data["arena_config"] = asdict(config.arena_config)
    data["effective_recovery_mode"] = effective_recovery_mode(config)
    return data


def effective_recovery_mode(config: ArenaActiveSpinConfig):
    if config.recovery_mode != "none":
        return config.recovery_mode
    if config.enable_center_reposition:
        return "legacy"
    return "none"


def active_explore_config_from_arena_config(config: ArenaActiveSpinConfig):
    return ActiveExploreConfig(
        max_attempts=config.active_explore_max_attempts,
        max_single_move_m=config.active_explore_max_single_move_m,
        max_total_distance_m=config.active_explore_max_total_distance_m,
        max_candidate_path_m=config.active_explore_max_candidate_path_m,
        grid_resolution_m=config.active_explore_grid_resolution_m,
        grid_size_m=config.active_explore_grid_size_m,
        inflation_radius_m=config.active_explore_inflation_radius_m,
        soft_clearance_radius_m=config.active_explore_soft_clearance_radius_m,
        soft_clearance_weight=config.active_explore_soft_clearance_weight,
        unknown_blocked=config.active_explore_unknown_blocked,
        max_path_segments=config.active_explore_max_path_segments,
        target_nearest_short_wall_range_m=(
            config.center_reposition_target_nearest_short_wall_range_m
        ),
        center_min_step_m=config.center_reposition_min_step_m,
        lateral_offset_threshold_m=config.center_reposition_lateral_offset_threshold_m,
        lateral_target_offset_m=config.center_reposition_lateral_target_offset_m,
        heater_approach_target_range_m=(
            config.center_reposition_heater_approach_target_range_m
        ),
        heater_approach_min_selected_score=(
            config.center_reposition_heater_approach_min_selected_score
        ),
        heater_approach_max_opposite_score=(
            config.center_reposition_heater_approach_max_opposite_score
        ),
        heater_approach_min_delta=config.center_reposition_heater_approach_min_delta,
        arena_length_m=config.arena_config.arena_length_m,
        max_short_wall_range_sum_error_m=(
            config.arena_config.max_short_wall_range_sum_error_m
        ),
    )


def initial_diagnostics(config: ArenaActiveSpinConfig):
    recovery_mode = effective_recovery_mode(config)
    return {
        "mode": "arena-active",
        "success": False,
        "failure_reason": "",
        "fallback_used": False,
        "config": config_diagnostics(config),
        "spin": spin_diagnostics_template(),
        "spin_attempts": [],
        "reposition": {
            "enabled": recovery_mode == "legacy",
            "attempts": [],
        },
        "active_explore": {
            "enabled": recovery_mode == "active_explore",
            "mode": recovery_mode,
            "executor": config.recovery_executor,
            "active_explore_phase": ACTIVE_EXPLORE_PHASE_SHADOW,
            "use_accumulated_map": config.active_explore_use_accumulated_map,
            "map_max_samples": config.active_explore_map_max_samples,
            "temporary_map": {
                "frame": "odom",
                "scan_samples_stored": 0,
                "grid": None,
            },
            "attempts": [],
            "total_distance_m": 0.0,
            "motion_attempts": 0,
            "persistent_frontier_goal": None,
            "shadow_frontier_empty_replans": 0,
            "shadow_explore_complete": False,
            "shadow_frontier_status": None,
            "shadow_map_status": None,
            "mission": {
                "phase": "shadow_mapping",
                "motion_attempts": 0,
                "max_motion_attempts": config.active_explore_max_attempts,
                "shadow_confirmation_count": 0,
                "shadow_completion_confirmations_required": (
                    config.active_explore_shadow_completion_confirmations
                ),
                "shadow_stall_replans": 0,
                "max_shadow_stall_replans": (
                    config.active_explore_max_shadow_stall_replans
                ),
                "localization_pose_attempts": 0,
                "max_localization_pose_attempts": (
                    config.active_explore_max_localization_pose_attempts
                ),
                "last_shadow_unknown_cell_count": None,
                "last_selected_candidate_kind": None,
            },
            "localization_candidate_policy": None,
            "shadow_approach_fallback_policy": None,
            "localizer_filter": {
                "enabled": False,
                "reason": "not_run",
            },
        },
        "samples": {
            "scan_samples_collected": 0,
            "scan_samples_used": 0,
            "rejected_scan_samples": 0,
        },
        "safety": {
            "min_front_range_m": None,
            "min_left_range_m": None,
            "min_right_range_m": None,
            "min_rear_range_m": None,
        },
        "cmd_vel_publishers": {
            "count": None,
            "unexpected_count": None,
            "allowed": config.allow_extra_cmd_vel_publishers,
        },
        "localizer_result": None,
        "exception": None,
        "initialpose": {
            "published": False,
            "reason": "not_reached",
        },
    }


def update_safety_minima(diagnostics, clearance: SectorClearance):
    safety = diagnostics["safety"]
    for key, value in [
        ("min_front_range_m", clearance.front_min_m),
        ("min_left_range_m", clearance.left_min_m),
        ("min_right_range_m", clearance.right_min_m),
        ("min_rear_range_m", clearance.rear_min_m),
    ]:
        if value is None:
            continue
        current = safety.get(key)
        safety[key] = value if current is None else min(current, value)
