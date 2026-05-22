from dataclasses import dataclass
from pathlib import Path


DEFAULT_WAYPOINTS_CSV = Path("results/aufgabe03/aufgabe03_waypoints.csv")

DEFAULT_RESULTS_CSV = Path("results/aufgabe03/aufgabe03_arena_prior_two_stage_runs.csv")

DEFAULT_FOLLOWER_SCRIPT = Path("scripts/aufgabe03/follow_planned_waypoints.py")

DEFAULT_STATIC_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")

DEFAULT_REPLAN_OUTPUT_DIR = Path("results/aufgabe03")

DEFAULT_PREFLIGHT_TIMEOUT_SEC = 10.0

DEFAULT_NAV_TO_START_TIMEOUT_SEC = 180.0

DEFAULT_TF_READY_TIMEOUT_SEC = 15.0

DEFAULT_TF_LOOKUP_TIMEOUT_SEC = 10.0

DEFAULT_TF_LOOKUP_RETRY_PERIOD_SEC = 0.1

DEFAULT_FOLLOWER_STARTUP_TIMEOUT_SEC = 20.0

DEFAULT_FOLLOWER_START_ON_PATH_TOLERANCE_M = 0.25

DEFAULT_ARENA_ACTIVE_VALIDATION_TIMEOUT_SEC = 30.0

DEFAULT_ARENA_ACTIVE_MAX_POST_AMCL_PRIOR_POSITION_ERROR_M = 0.25

DEFAULT_ARENA_ACTIVE_MAX_POST_AMCL_PRIOR_YAW_ERROR_DEG = 20.0

DEFAULT_MAX_AMCL_AGE_SEC = 15.0

DEFAULT_MAX_AMCL_VAR_X = 0.05

DEFAULT_MAX_AMCL_VAR_Y = 0.05

DEFAULT_MAX_AMCL_VAR_YAW_RAD2 = 0.10

DEFAULT_STABLE_AMCL_SAMPLES = 5

DEFAULT_AMCL_SETTLE_MIN_SEC = 3.0

DEFAULT_MAX_STABLE_POSE_JUMP_M = 0.05

DEFAULT_MAX_STABLE_YAW_JUMP_DEG = 10.0

MIN_ARENA_ACTIVE_VAR_XY = 0.0025

MIN_ARENA_ACTIVE_VAR_YAW_RAD2 = 0.0076

DEFAULT_SPIN_MIN_SCAN_RANGE_M = 0.18

DEFAULT_SPIN_MIN_VALID_SCAN_COUNT = 20

DEFAULT_MAX_SCAN_AGE_SEC = 8.0

DEFAULT_MAX_POSE_AGE_SEC = 10.0

DEFAULT_ARRIVAL_TOLERANCE_M = 0.15

DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG = 45.0

DEFAULT_WAYPOINT_TOLERANCE_M = 0.12

DEFAULT_GOAL_TOLERANCE_M = 0.12

DEFAULT_MIN_WAYPOINT_SPACING_M = 0.12

DEFAULT_CONTROL_RATE_HZ = 10.0

DEFAULT_REPLAN_TIMEOUT_SEC = 5.0

DEFAULT_MAX_REPLAN_SCAN_AGE_SEC = 1.0

DEFAULT_MAX_REPLAN_TF_AGE_SEC = 1.0

DEFAULT_OBSTACLE_FORWARD_DISTANCE_M = 0.55

DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M = 0.18

DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG = 45.0

DEFAULT_OBSTACLE_MIN_RANGE_M = 0.12

DEFAULT_ROBOT_FOOTPRINT_RADIUS_M = 0.18

DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE = 3

DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M = 0.05

DEFAULT_OBSTACLE_INFLATE_RADIUS_M = 0.22

DEFAULT_MAX_START_SNAP_M = 0.20

DEFAULT_MAX_GOAL_SNAP_M = 0.30

DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO = 3.0

DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE = "full"

DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT = 5

DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE = "forward"

DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT = 2

DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M = DEFAULT_OBSTACLE_INFLATE_RADIUS_M

DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC = 1.0

DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC = 1.0

DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS = 3

DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO = 0.90

DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M = 0.75

DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M = 0.04

DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES = 3

STOP_PUBLISH_COUNT = 10

STOP_PUBLISH_HZ = 10.0

CSV_HEADER = [
    "timestamp",
    "start_wall_time",
    "end_wall_time",
    "duration_sec",
    "run_id",
    "waypoint_csv",
    "status",
    "final_status_reason",
    "navigate_action",
    "initial_pose_topic",
    "amcl_topic",
    "cmd_vel_topic",
    "scan_topic",
    "odom_topic",
    "map_frame",
    "selected_base_frame",
    "staging_x",
    "staging_y",
    "staging_yaw_deg",
    "arena_spin_duration_sec",
    "nav2_duration_sec",
    "follower_duration_sec",
    "amcl_var_x",
    "amcl_var_y",
    "amcl_var_yaw_rad2",
    "stable_samples",
    "max_pose_jump_m",
    "max_yaw_jump_deg",
    "nav2_result_status",
    "tf_arrival_x",
    "tf_arrival_y",
    "tf_arrival_yaw_deg",
    "arrival_position_error_m",
    "arrival_yaw_error_deg",
    "follower_command",
    "follower_return_code",
    "final_tf_x",
    "final_tf_y",
    "final_tf_yaw_deg",
    "notes",
]

GOAL_STATUS_NAMES = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}


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


@dataclass
class RunDiagnostics:
    timestamp: str = ""
    start_wall_time: str = ""
    end_wall_time: str = ""
    duration_sec: float | None = None
    status: str = "failed"
    final_status_reason: str = ""
    selected_base_frame: str = ""
    arena_spin_duration_sec: float | None = None
    nav2_duration_sec: float | None = None
    follower_duration_sec: float | None = None
    amcl_var_x: float | None = None
    amcl_var_y: float | None = None
    amcl_var_yaw_rad2: float | None = None
    stable_samples: int = 0
    max_pose_jump_m: float | None = None
    max_yaw_jump_deg: float | None = None
    nav2_result_status: str = ""
    tf_arrival_x: float | None = None
    tf_arrival_y: float | None = None
    tf_arrival_yaw_deg: float | None = None
    arrival_position_error_m: float | None = None
    arrival_yaw_error_deg: float | None = None
    follower_command: str = ""
    follower_return_code: int | None = None
    final_tf_x: float | None = None
    final_tf_y: float | None = None
    final_tf_yaw_deg: float | None = None
    notes: str = ""
