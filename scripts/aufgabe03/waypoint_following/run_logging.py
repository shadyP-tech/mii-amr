from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import Waypoint


BASE_CSV_HEADER = [
    "timestamp",
    "run_id",
    "waypoint_csv",
    "waypoint_count",
    "reached_count",
    "status",
    "blocked_waypoint_index",
    "blocked_waypoint_x",
    "blocked_waypoint_y",
    "timeout_waypoint_index",
    "base_frame_used",
    "start_x",
    "start_y",
    "start_yaw_deg",
    "final_x",
    "final_y",
    "final_yaw_deg",
    "min_scan_range_m",
    "p05_scan_range_m",
    "amcl_var_x",
    "amcl_var_y",
    "amcl_var_yaw",
    "linear_speed_mps",
    "min_linear_speed_mps",
    "linear_gain",
    "max_angular_speed_radps",
    "yaw_gain",
    "notes",
]

CSV_HEADER = BASE_CSV_HEADER + [
    "selected_start_segment_index",
    "selected_start_waypoint_index",
    "distance_to_path_m",
    "tf_pose_age_sec",
    "max_tf_update_gap_sec",
    "tf_stale_warning_count",
    "localization_warning_count",
    "recovery_pause_count",
    "max_abs_yaw_error_deg",
    "mean_abs_yaw_error_deg",
    "rotate_seconds",
    "forward_seconds",
    "final_status_reason",
    "replan_count",
    "last_replan_reason",
    "updated_map_yaml",
    "updated_waypoints_csv",
    "detected_obstacle_count",
    "candidate_scan_points",
    "filtered_obstacle_points",
    "raw_obstacle_cells",
    "free_obstacle_cells",
    "inflated_cells_total",
    "inflated_cells_newly_occupied",
    "inflated_cells_over_static_occupied",
    "scan_frame",
    "scan_age_sec",
    "tf_age_sec",
    "tf_lookup_mode",
    "start_snap_distance_m",
    "goal_snap_distance_m",
    "old_remaining_waypoint_count",
    "new_waypoint_count",
    "old_path_length_m",
    "new_path_length_m",
    "replan_duration_sec",
    "run_local_map_updates",
    "run_local_replan_count",
    "run_local_last_replan_reason",
    "run_local_no_path_reason",
    "run_local_start_cell_blocked",
    "run_local_goal_cell_blocked",
    "run_local_path_blocked_cell_count",
    "run_local_scan_points_valid",
    "run_local_scan_points_used",
    "run_local_scan_points_rejected_invalid_range",
    "run_local_scan_points_rejected_static",
    "run_local_scan_points_rejected_bounds",
    "run_local_scan_points_rejected_wall_band",
    "run_local_scan_points_rejected_low_confidence",
    "run_local_update_rejected_reason",
    "run_local_initial_scan_count",
    "run_local_corridor_check_distance_m",
    "run_local_inflation_radius_m",
    "run_local_map_yaml",
    "run_local_waypoints_csv",
    "run_local_sparse_retry_count",
    "run_local_sparse_retry_mode",
    "run_local_pruned_raw_cells",
    "run_local_pruned_inflated_cells",
    "run_local_cell_source_counts",
]


@dataclass
class RuntimeDiagnostics:
    selected_start_segment_index: int | None = None
    selected_start_waypoint_index: int | None = None
    distance_to_path_m: float | None = None
    tf_pose_age_sec: float | None = None
    max_tf_update_gap_sec: float | None = None
    tf_stale_warning_count: int = 0
    localization_warning_count: int = 0
    recovery_pause_count: int = 0
    max_abs_yaw_error_deg: float = 0.0
    yaw_error_sum_deg: float = 0.0
    yaw_error_count: int = 0
    rotate_seconds: float = 0.0
    forward_seconds: float = 0.0
    final_status_reason: str = ""
    replan_count: int = 0
    last_replan_reason: str = ""
    updated_map_yaml: str = ""
    updated_waypoints_csv: str = ""
    detected_obstacle_count: int = 0
    candidate_scan_points: int = 0
    filtered_obstacle_points: int = 0
    raw_obstacle_cells: int = 0
    free_obstacle_cells: int = 0
    inflated_cells_total: int = 0
    inflated_cells_newly_occupied: int = 0
    inflated_cells_over_static_occupied: int = 0
    scan_frame: str = ""
    scan_age_sec: float | None = None
    tf_age_sec: float | None = None
    tf_lookup_mode: str = ""
    start_snap_distance_m: float | None = None
    goal_snap_distance_m: float | None = None
    old_remaining_waypoint_count: int = 0
    new_waypoint_count: int = 0
    old_path_length_m: float | None = None
    new_path_length_m: float | None = None
    replan_duration_sec: float | None = None
    run_local_map_updates: int = 0
    run_local_replan_count: int = 0
    run_local_last_replan_reason: str = ""
    run_local_no_path_reason: str = ""
    run_local_start_cell_blocked: bool = False
    run_local_goal_cell_blocked: bool = False
    run_local_path_blocked_cell_count: int = 0
    run_local_scan_points_valid: int = 0
    run_local_scan_points_used: int = 0
    run_local_scan_points_rejected_invalid_range: int = 0
    run_local_scan_points_rejected_static: int = 0
    run_local_scan_points_rejected_bounds: int = 0
    run_local_scan_points_rejected_wall_band: int = 0
    run_local_scan_points_rejected_low_confidence: int = 0
    run_local_update_rejected_reason: str = ""
    run_local_initial_scan_count: int = 0
    run_local_corridor_check_distance_m: float | None = None
    run_local_inflation_radius_m: float | None = None
    run_local_map_yaml: str = ""
    run_local_waypoints_csv: str = ""
    run_local_sparse_retry_count: int = 0
    run_local_sparse_retry_mode: str = ""
    run_local_pruned_raw_cells: int = 0
    run_local_pruned_inflated_cells: int = 0
    run_local_cell_source_counts: dict[str, int] | str = ""

    @property
    def mean_abs_yaw_error_deg(self):
        if self.yaw_error_count == 0:
            return 0.0
        return self.yaw_error_sum_deg / self.yaw_error_count


def record_motion_sample(node, yaw_error_deg, linear_x, angular_z, sample_seconds):
    abs_error = abs(yaw_error_deg)
    node.diagnostics.max_abs_yaw_error_deg = max(
        node.diagnostics.max_abs_yaw_error_deg,
        abs_error,
    )
    node.diagnostics.yaw_error_sum_deg += abs_error
    node.diagnostics.yaw_error_count += 1
    if abs(linear_x) <= 1e-9 and abs(angular_z) > 1e-9:
        node.diagnostics.rotate_seconds += sample_seconds
    else:
        node.diagnostics.forward_seconds += sample_seconds


def pose_fields(pose):
    if pose is None:
        return ["", "", ""]
    return [pose.x, pose.y, pose.yaw_deg]


def optional(value):
    return "" if value is None else value


def build_log_row(
    args,
    waypoint_count,
    reached_count,
    status,
    notes,
    start_pose=None,
    final_pose=None,
    blocked_waypoint=None,
    timeout_waypoint=None,
    base_frame_used="",
    scan_safety=None,
    amcl_health=None,
    diagnostics=None,
):
    diagnostics = diagnostics or RuntimeDiagnostics()
    blocked = blocked_waypoint or Waypoint("", "", "")
    timeout = timeout_waypoint or Waypoint("", "", "")
    return [
        datetime.now().isoformat(timespec="seconds"),
        args.run_id,
        str(args.waypoints),
        waypoint_count,
        reached_count,
        status,
        blocked.index,
        blocked.x,
        blocked.y,
        timeout.index,
        base_frame_used,
        *(pose_fields(start_pose)),
        *(pose_fields(final_pose)),
        "" if scan_safety is None or scan_safety.min_range_m is None else scan_safety.min_range_m,
        "" if scan_safety is None or scan_safety.percentile_5_m is None else scan_safety.percentile_5_m,
        "" if amcl_health is None or amcl_health.cov_x is None else amcl_health.cov_x,
        "" if amcl_health is None or amcl_health.cov_y is None else amcl_health.cov_y,
        "" if amcl_health is None or amcl_health.cov_yaw is None else amcl_health.cov_yaw,
        args.linear_speed,
        args.min_linear_speed,
        args.linear_gain,
        args.max_angular_speed,
        args.yaw_gain,
        notes,
        optional(diagnostics.selected_start_segment_index),
        optional(diagnostics.selected_start_waypoint_index),
        optional(diagnostics.distance_to_path_m),
        optional(diagnostics.tf_pose_age_sec),
        optional(diagnostics.max_tf_update_gap_sec),
        diagnostics.tf_stale_warning_count,
        diagnostics.localization_warning_count,
        diagnostics.recovery_pause_count,
        diagnostics.max_abs_yaw_error_deg,
        diagnostics.mean_abs_yaw_error_deg,
        diagnostics.rotate_seconds,
        diagnostics.forward_seconds,
        diagnostics.final_status_reason,
        diagnostics.replan_count,
        diagnostics.last_replan_reason,
        diagnostics.updated_map_yaml,
        diagnostics.updated_waypoints_csv,
        diagnostics.detected_obstacle_count,
        diagnostics.candidate_scan_points,
        diagnostics.filtered_obstacle_points,
        diagnostics.raw_obstacle_cells,
        diagnostics.free_obstacle_cells,
        diagnostics.inflated_cells_total,
        diagnostics.inflated_cells_newly_occupied,
        diagnostics.inflated_cells_over_static_occupied,
        diagnostics.scan_frame,
        optional(diagnostics.scan_age_sec),
        optional(diagnostics.tf_age_sec),
        diagnostics.tf_lookup_mode,
        optional(diagnostics.start_snap_distance_m),
        optional(diagnostics.goal_snap_distance_m),
        diagnostics.old_remaining_waypoint_count,
        diagnostics.new_waypoint_count,
        optional(diagnostics.old_path_length_m),
        optional(diagnostics.new_path_length_m),
        optional(diagnostics.replan_duration_sec),
        diagnostics.run_local_map_updates,
        diagnostics.run_local_replan_count,
        diagnostics.run_local_last_replan_reason,
        diagnostics.run_local_no_path_reason,
        diagnostics.run_local_start_cell_blocked,
        diagnostics.run_local_goal_cell_blocked,
        diagnostics.run_local_path_blocked_cell_count,
        diagnostics.run_local_scan_points_valid,
        diagnostics.run_local_scan_points_used,
        diagnostics.run_local_scan_points_rejected_invalid_range,
        diagnostics.run_local_scan_points_rejected_static,
        diagnostics.run_local_scan_points_rejected_bounds,
        diagnostics.run_local_scan_points_rejected_wall_band,
        diagnostics.run_local_scan_points_rejected_low_confidence,
        diagnostics.run_local_update_rejected_reason,
        diagnostics.run_local_initial_scan_count,
        optional(diagnostics.run_local_corridor_check_distance_m),
        optional(diagnostics.run_local_inflation_radius_m),
        diagnostics.run_local_map_yaml,
        diagnostics.run_local_waypoints_csv,
        diagnostics.run_local_sparse_retry_count,
        diagnostics.run_local_sparse_retry_mode,
        diagnostics.run_local_pruned_raw_cells,
        diagnostics.run_local_pruned_inflated_cells,
        diagnostics.run_local_cell_source_counts,
    ]


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), None)
        if existing_header == header:
            pass
        elif existing_header and header[: len(existing_header)] == existing_header:
            migrate_csv_header(path, header)
        else:
            raise RuntimeError(
                f"{path} has an unrecognized schema. Move or migrate it first."
            )
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def migrate_csv_header(path, header):
    path = Path(path)
    with path.open(newline="") as file:
        rows = list(csv.reader(file))

    migrated = [header]
    for row in rows[1:]:
        migrated.append(row + [""] * (len(header) - len(row)))

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(migrated)
