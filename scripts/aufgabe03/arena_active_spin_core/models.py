from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from arena_geometry_localizer import ArenaGeometryConfig


DEFAULT_STOP_COUNT = 10
DEFAULT_STOP_HZ = 10.0
ACTIVE_EXPLORE_FRONTIER_CLUSTER_MATCH_M = 0.35
ACTIVE_EXPLORE_FRONTIER_TARGET_MATCH_M = 0.45
ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M = 0.15
ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE = 2
ACTIVE_EXPLORE_PHASE_SHADOW = "shadow_explore"
ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE = "localization_pose"
ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN = "localization_spin"
ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS = (
    "suspected_heater_approach",
    "provisional_center",
    "lateral_recenter",
)
ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS = {
    "no_connected_path",
    "path_too_long",
}
ACTIVE_EXPLORE_SHADOW_APPROACH_MIN_PATH_CLEARANCE_M = 0.10
ACTIVE_EXPLORE_SHADOW_APPROACH_GOAL_DRIFT_TOLERANCE_M = 0.02
LOCALIZER_FILTER_WALL_MARGIN_CELLS = 2
LOCALIZER_FILTER_WALL_EXPAND_CELLS = 1
LOCALIZER_FILTER_MIN_WALL_LENGTH_M = 0.45
LOCALIZER_FILTER_MIN_WALL_ASPECT_RATIO = 3.0
LOCALIZER_FILTER_MAX_WALL_THICKNESS_M = 0.20


class ActiveExploreMotionError(RuntimeError):
    def __init__(self, reason, record):
        super().__init__(reason)
        self.reason = reason
        self.record = record


@dataclass(frozen=True)
class PosePrior:
    x_m: float
    y_m: float
    yaw_rad: float
    covariance: list[float]


@dataclass
class ArenaActiveSpinResult:
    success: bool
    failure_reason: str | None
    pose_prior: PosePrior | None
    diagnostics: dict
    diagnostics_path: str | None = None


@dataclass(frozen=True)
class SectorClearance:
    ok: bool
    reason: str
    front_min_m: float | None
    left_min_m: float | None
    right_min_m: float | None
    rear_min_m: float | None


@dataclass(frozen=True)
class CenterRepositionStep:
    kind: str
    reason: str
    planned_distance_m: float
    local_heading_rad: float | None
    odom_heading_rad: float
    dynamic_heading: bool = False
    dynamic_heading_source: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CenterRepositionAction:
    ok: bool
    reason: str
    nearest_axis_side: str | None = None
    away_axis_side: str | None = None
    suspected_heater_axis_side: str | None = None
    nearest_short_wall_range_m: float | None = None
    far_short_wall_range_m: float | None = None
    suspected_heater_range_m: float | None = None
    target_nearest_short_wall_range_m: float | None = None
    heater_approach_target_range_m: float | None = None
    planned_distance_m: float | None = None
    local_heading_rad: float | None = None
    odom_heading_rad: float | None = None
    range_sum_error_m: float | None = None
    heater_scores: dict[str, float] | None = None
    selected_heater_score: float | None = None
    opposite_heater_score: float | None = None
    heater_profile_delta: float | None = None
    lateral_offset_m: float | None = None
    lateral_target_offset_m: float | None = None
    lateral_planned_distance_m: float | None = None
    lateral_step_skipped: bool = True
    lateral_skip_reason: str | None = None
    steps: tuple[CenterRepositionStep, ...] = ()

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArenaActiveSpinConfig:
    run_id: str
    diagnostics_path: Path
    cmd_vel_topic: str = "/cmd_vel"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    spin_direction: str = "ccw"
    angular_speed_rad_s: float = 0.25
    max_spin_sec: float = 30.0
    spin_complete_tolerance_deg: float = 5.0
    min_angular_progress_rad_s: float = 0.05
    progress_check_sec: float = 2.0
    min_scan_samples: int = 20
    max_odom_scan_age_sec: float = 0.20
    stop_settle_sec: float = 0.5
    min_front_clearance_m: float = 0.35
    min_side_clearance_m: float = 0.20
    min_rear_clearance_m: float = 0.20
    require_operator_confirmation: bool = True
    allow_extra_cmd_vel_publishers: bool = False
    dry_run: bool = False
    range_stride: int = 6
    max_points: int = 3000
    control_rate_hz: float = 10.0
    recovery_mode: str = "none"
    recovery_executor: str = "dry_run"
    enable_center_reposition: bool = False
    center_reposition_max_attempts: int = 1
    center_reposition_target_nearest_short_wall_range_m: float = 1.65
    center_reposition_min_step_m: float = 0.25
    center_reposition_max_step_m: float = 1.10
    center_reposition_linear_speed_mps: float = 0.08
    center_reposition_angular_speed_rad_s: float = 0.25
    center_reposition_heading_tolerance_deg: float = 8.0
    center_reposition_min_front_clearance_m: float = 0.45
    center_reposition_lateral_offset_threshold_m: float = 0.25
    center_reposition_lateral_target_offset_m: float = 0.10
    center_reposition_lateral_min_step_m: float = 0.15
    center_reposition_lateral_max_step_m: float = 0.55
    center_reposition_enable_heater_approach: bool = True
    center_reposition_heater_approach_max_attempts: int = 1
    center_reposition_heater_approach_target_range_m: float = 1.05
    center_reposition_heater_approach_min_selected_score: float = 0.50
    center_reposition_heater_approach_max_opposite_score: float = 0.30
    center_reposition_heater_approach_min_delta: float = 0.35
    center_reposition_heater_approach_min_step_m: float = 0.25
    center_reposition_heater_approach_max_step_m: float = 1.10
    active_explore_max_attempts: int = 6
    active_explore_max_single_move_m: float = 0.45
    active_explore_max_total_distance_m: float = 0.90
    active_explore_max_candidate_path_m: float | None = None
    active_explore_grid_resolution_m: float = 0.05
    active_explore_grid_size_m: float = 4.0
    active_explore_inflation_radius_m: float = 0.15
    active_explore_soft_clearance_radius_m: float = 0.20
    active_explore_soft_clearance_weight: float = 3.0
    active_explore_unknown_blocked: bool = True
    active_explore_max_path_segments: int = 3
    active_explore_use_accumulated_map: bool = True
    active_explore_map_max_samples: int = 240
    active_explore_temporary_map_publish_period_sec: float = 1.0
    active_explore_shadow_completion_confirmations: int = 2
    active_explore_max_shadow_stall_replans: int = 3
    active_explore_max_localization_pose_attempts: int = 2
    active_explore_curve_lookahead_m: float = 0.18
    active_explore_curve_goal_tolerance_m: float = 0.05
    active_explore_curve_linear_speed_mps: float = 0.06
    active_explore_curve_max_angular_rad_s: float = 0.45
    active_explore_min_progress_before_spin_m: float = 0.05
    arena_config: ArenaGeometryConfig = field(default_factory=ArenaGeometryConfig)
