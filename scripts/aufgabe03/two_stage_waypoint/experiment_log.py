import csv
from pathlib import Path

from .pure import empty_if_none


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    if file_exists:
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), None)
        if existing_header != header:
            raise RuntimeError(f"{path} has an unrecognized schema. Move or migrate it first.")
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def build_log_row(args, staging_goal, diagnostics):
    return [
        diagnostics.timestamp,
        diagnostics.start_wall_time,
        diagnostics.end_wall_time,
        empty_if_none(diagnostics.duration_sec),
        args.run_id,
        str(args.waypoints),
        args.localization_mode,
        diagnostics.status,
        diagnostics.final_status_reason,
        args.global_localization_service,
        args.navigate_action,
        args.initial_pose_topic,
        args.amcl_topic,
        args.cmd_vel_topic,
        args.scan_topic,
        args.map_frame,
        diagnostics.selected_base_frame,
        staging_goal.waypoint.x,
        staging_goal.waypoint.y,
        staging_goal.yaw_deg,
        empty_if_none(diagnostics.localization_duration_sec),
        empty_if_none(diagnostics.nav2_duration_sec),
        empty_if_none(diagnostics.follower_duration_sec),
        empty_if_none(diagnostics.amcl_var_x),
        empty_if_none(diagnostics.amcl_var_y),
        empty_if_none(diagnostics.amcl_var_yaw_rad2),
        diagnostics.stable_samples,
        empty_if_none(diagnostics.max_pose_jump_m),
        empty_if_none(diagnostics.max_yaw_jump_deg),
        diagnostics.nav2_result_status,
        empty_if_none(diagnostics.tf_arrival_x),
        empty_if_none(diagnostics.tf_arrival_y),
        empty_if_none(diagnostics.tf_arrival_yaw_deg),
        empty_if_none(diagnostics.arrival_position_error_m),
        empty_if_none(diagnostics.arrival_yaw_error_deg),
        diagnostics.follower_command,
        empty_if_none(diagnostics.follower_return_code),
        empty_if_none(diagnostics.final_tf_x),
        empty_if_none(diagnostics.final_tf_y),
        empty_if_none(diagnostics.final_tf_yaw_deg),
        diagnostics.notes,
    ]
