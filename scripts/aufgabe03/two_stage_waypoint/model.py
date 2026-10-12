"""Coordinator defaults and immutable gate results."""
from dataclasses import dataclass, fields
from pathlib import Path

from arena_active_spin_core.models import ArenaActiveSpinConfig
from arena_geometry_localization import ArenaGeometryConfig

DEFAULT_SPIN_MIN_SCAN_RANGE_M = 0.18
DEFAULT_SPIN_MIN_VALID_SCAN_COUNT = 20
MIN_ARENA_ACTIVE_VAR_XY = 0.0025
MIN_ARENA_ACTIVE_VAR_YAW_RAD2 = 0.0076
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0
GOAL_STATUS_NAMES = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING",
                     4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}


@dataclass
class CoordinatorConfig:
    waypoints: Path = Path("results/aufgabe03/aufgabe03_waypoints.csv")
    output_dir: Path = Path("results/aufgabe03")
    run_id: str = ""
    follower_script: Path = Path("scripts/aufgabe03/follow_planned_waypoints.py")
    python_executable: str = "python3"
    static_map: Path = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
    map_frame: str = "map"
    base_frame: str = "base_footprint"
    fallback_base_frame: str = "base_link"
    navigate_action: str = "/navigate_to_pose"
    initial_pose_topic: str = "/initialpose"
    amcl_topic: str = "/amcl_pose"
    cmd_vel_topic: str = "/cmd_vel"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    preflight_timeout_sec: float = 10.0
    nav_to_start_timeout_sec: float = 180.0
    tf_ready_timeout_sec: float = 15.0
    tf_lookup_timeout_sec: float = 10.0
    tf_lookup_retry_period_sec: float = 0.1
    follower_startup_timeout_sec: float = 20.0
    follower_start_on_path_tolerance_m: float = 0.25
    arena_active_validation_timeout_sec: float = 30.0
    arena_active_max_post_amcl_prior_position_error_m: float = 0.25
    arena_active_max_post_amcl_prior_yaw_error_deg: float = 20.0
    max_pose_age_sec: float = 1.0
    max_scan_age_sec: float = 1.0
    max_amcl_age_sec: float = 5.0
    max_amcl_var_x: float = 0.05
    max_amcl_var_y: float = 0.05
    max_amcl_var_yaw_rad2: float = 0.10
    stable_amcl_samples: int = 5
    amcl_settle_min_sec: float = 3.0
    max_stable_pose_jump_m: float = 0.05
    max_stable_yaw_jump_deg: float = 10.0
    spin_min_scan_range_m: float = DEFAULT_SPIN_MIN_SCAN_RANGE_M
    spin_min_valid_scan_count: int = DEFAULT_SPIN_MIN_VALID_SCAN_COUNT
    arrival_tolerance_m: float = 0.15
    arrival_yaw_tolerance_deg: float = 45.0
    waypoint_tolerance_m: float = 0.12
    goal_tolerance_m: float = 0.12
    min_waypoint_spacing_m: float = 0.12
    control_rate_hz: float = 10.0
    max_replans: int = 1
    run_local_map_max_updates: int = 3
    enable_lidar_map_replan: bool = False
    yes: bool = False
    dry_run: bool = False
    verbose: bool = False


# Share config defaults directly, retaining the established geometry CLI names.
GEOMETRY_FIELDS = (
    "arena_length_m", "arena_width_m", "heater_side_width_m", "clean_side_width_m",
    "width_match_min_margin_m", "max_short_wall_range_sum_error_m", "map_center_x",
    "map_center_y", "map_yaw_deg", "heater_wall_side", "min_wall_points",
    "max_wall_separation_error_m", "max_line_rmse_m", "min_parallel_score",
    "min_short_wall_confidence", "min_classification_margin",
    "forced_short_wall_side", "forced_short_wall_type",
)
GEOMETRY_ALIASES = {
    "heater_side_width_m": "arena_heater_wall_width_m",
    "clean_side_width_m": "arena_clean_wall_width_m",
    "forced_short_wall_side": "arena_force_short_wall_side",
    "forced_short_wall_type": "arena_force_short_wall_type",
    "heater_wall_side": "heater_wall_side",
}
SPIN_SHARED_FIELDS = {
    "run_id", "cmd_vel_topic", "scan_topic", "odom_topic", "control_rate_hz", "verbose",
}
SPIN_INTERNAL_FIELDS = {"arena_config", "diagnostics_path", "dry_run",
                        "active_explore_temporary_map_publish_period_sec"}


def geometry_arguments():
    return {name: GEOMETRY_ALIASES.get(name, name if name.startswith("arena_")
                                      else f"arena_{name}") for name in GEOMETRY_FIELDS}


def spin_arguments():
    return {field.name: (field.name if field.name in SPIN_SHARED_FIELDS else
                        f"arena_{field.name}" if field.name.startswith("active_explore_")
                        else f"arena_active_{field.name}")
            for field in fields(ArenaActiveSpinConfig)
            if field.name not in SPIN_INTERNAL_FIELDS}


@dataclass(frozen=True)
class Waypoint:
    index: int
    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_deg: float
    stamp_sec: float | None = None
    frame_id: str = ""


@dataclass(frozen=True)
class StagingGoal:
    waypoint: Waypoint
    yaw_deg: float


@dataclass(frozen=True)
class ScanSafety:
    ok: bool
    reason: str
    valid_count: int
    min_range_m: float | None


@dataclass(frozen=True)
class PreflightRequirements:
    actions: list[str]
    topics: list[str]


@dataclass(frozen=True)
class AmclCovariance:
    x: float
    y: float
    yaw_rad2: float


@dataclass(frozen=True)
class StabilityState:
    stable_count: int = 0
    previous_pose: Pose2D | None = None
    stable_since_sec: float | None = None
    quiet_duration_sec: float = 0.0
    max_pose_jump_m: float = 0.0
    max_yaw_jump_deg: float = 0.0
    cov_x: float | None = None
    cov_y: float | None = None
    cov_yaw_rad2: float | None = None
    samples_seen: int = 0
    reason: str = "waiting_for_amcl"


@dataclass(frozen=True)
class ArrivalCheck:
    pose: Pose2D
    base_frame: str
    position_error_m: float
    yaw_error_deg: float
    distance_to_path_m: float | None = None
    handoff_path_ok: bool = False
    strict_position_ok: bool = True
    strict_yaw_ok: bool = True
