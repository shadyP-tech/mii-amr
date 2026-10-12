#!/usr/bin/env python3

import math
import os
import sys
from datetime import datetime

if __package__:
    from .motion import (
        DriveNode as ScriptedDrive, FORWARD_CSV_HEADER, append_csv_row,
        close_motion_node, odom_to_xy_yaw, parse_float_env, pose_columns,
        pose_fields, rclpy, read_tracker_pose, shortest_angle_delta_deg,
    )
else:
    from motion import (
        DriveNode as ScriptedDrive, FORWARD_CSV_HEADER, append_csv_row,
        close_motion_node, odom_to_xy_yaw, parse_float_env, pose_columns,
        pose_fields, rclpy, read_tracker_pose, shortest_angle_delta_deg,
    )



DEFAULT_RUN_MODE = "rotate-in-place"
DEFAULT_LINEAR_SPEED_MPS = 0.10
DEFAULT_FORWARD_DURATION_SEC = 3.0
DEFAULT_ROTATION_ANGLE_DEG = -90.0
DEFAULT_ROTATION_ANGULAR_SPEED_RADPS = 0.30
FORWARD_RESULTS_CSV = "results/aufgabe02/real_scripted_drive_runs.csv"
ROTATION_RESULTS_CSV = "results/aufgabe02/real_rotation_runs.csv"

ROTATION_MOTION_FIELDS = (
    "run_mode", "command_angle_deg", "direction", "linear_x_mps", "angular_z_radps", "duration_sec",
)
DELTA_FIELDS = ("yaw_change_deg", "yaw_error_deg", "dx", "dy", "position_drift_m")
ROTATION_CSV_HEADER = ["timestamp", "run_id", *ROTATION_MOTION_FIELDS]
for source in ("tracker", "odom"):
    ROTATION_CSV_HEADER += pose_columns(f"{source}_start") + pose_columns(f"{source}_final")
    ROTATION_CSV_HEADER += [f"{source}_{field}" for field in DELTA_FIELDS]
ROTATION_CSV_HEADER += ["notes"]


def configured_motion(env=os.environ):
    run_mode = env.get("REAL_RUN_MODE", env.get("RUN_MODE", DEFAULT_RUN_MODE))

    if run_mode == "linear-forward":
        linear_x = parse_float_env(env, "RUN_SPEED", DEFAULT_LINEAR_SPEED_MPS)
        duration = parse_float_env(env, "RUN_DURATION_SEC", DEFAULT_FORWARD_DURATION_SEC)
        if linear_x <= 0.0:
            raise ValueError("RUN_SPEED must be greater than zero")
        if duration <= 0.0:
            raise ValueError("RUN_DURATION_SEC must be greater than zero")

        return {
            "run_mode": "linear-forward",
            "linear_x_mps": linear_x,
            "angular_z_radps": 0.0,
            "duration_sec": duration,
            "results_csv": FORWARD_RESULTS_CSV,
            "notes": "real",
        }

    if run_mode in {"rotate-in-place", "rotation", "rotate"}:
        angle_deg = parse_float_env(env, "RUN_ANGLE_DEG", DEFAULT_ROTATION_ANGLE_DEG)
        angular_speed = abs(
            parse_float_env(
                env,
                "RUN_ANGULAR_SPEED",
                DEFAULT_ROTATION_ANGULAR_SPEED_RADPS,
            )
        )

        if angle_deg == 0.0:
            raise ValueError("RUN_ANGLE_DEG must be non-zero")
        if angular_speed <= 0.0:
            raise ValueError("RUN_ANGULAR_SPEED must be greater than zero")

        angle_rad = math.radians(angle_deg)
        angular_z = math.copysign(angular_speed, angle_rad)
        duration = abs(angle_rad) / angular_speed
        if not math.isfinite(duration):
            raise ValueError("Rotation duration must be finite")

        return {
            "run_mode": "rotate-in-place",
            "command_angle_deg": angle_deg,
            "direction": "clockwise" if angle_deg < 0.0 else "counterclockwise",
            "linear_x_mps": 0.0,
            "angular_z_radps": angular_z,
            "duration_sec": duration,
            "results_csv": ROTATION_RESULTS_CSV,
            "notes": "real_rotation",
        }

    raise ValueError(
        f"unsupported RUN_MODE={run_mode!r}; expected linear-forward or rotate-in-place"
    )


def pose_delta(start_pose, final_pose, command_angle_deg=None):
    if start_pose is None or final_pose is None:
        result = {
            "dx": "",
            "dy": "",
            "position_drift_m": "",
            "yaw_change_deg": "",
        }
    else:
        dx = final_pose["x"] - start_pose["x"]
        dy = final_pose["y"] - start_pose["y"]
        result = {
            "dx": dx,
            "dy": dy,
            "position_drift_m": math.hypot(dx, dy),
            "yaw_change_deg": shortest_angle_delta_deg(
                start_pose["yaw_deg"],
                final_pose["yaw_deg"],
            ),
        }

    if command_angle_deg is not None and result["yaw_change_deg"] != "":
        result["yaw_error_deg"] = shortest_angle_delta_deg(
            command_angle_deg,
            result["yaw_change_deg"],
        )
    elif command_angle_deg is not None:
        result["yaw_error_deg"] = ""

    return result


def save_real_result(run_id, motion, tracker_start, tracker_final, odom_start_msg, odom_final_msg):
    poses = ((tracker_start, tracker_final),
             (odom_to_xy_yaw(odom_start_msg), odom_to_xy_yaw(odom_final_msg)))
    rotation = motion["run_mode"] == "rotate-in-place"
    row = [datetime.now().isoformat(), run_id]
    if rotation:
        row.extend(motion[field] for field in ROTATION_MOTION_FIELDS)
    for start, final in poses:
        row.extend(pose_fields(start) + pose_fields(final))
        if rotation:
            delta = pose_delta(start, final, motion["command_angle_deg"])
            row.extend(delta[field] for field in DELTA_FIELDS)
    row.append(motion["notes"])
    header = ROTATION_CSV_HEADER if rotation else FORWARD_CSV_HEADER
    append_csv_row(motion["results_csv"], header, row)
    print(f"Saved result to {motion['results_csv']}")


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "manual_run"

    try:
        motion = configured_motion()
    except ValueError as exc:
        print(f"Invalid real-run configuration: {exc}", file=sys.stderr)
        return 2

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = ScriptedDrive(run_id)

    try:
        node.get_logger().info(f"{run_id}: {motion['run_mode']}, duration={motion['duration_sec']:.3f}s")
        odom_start = node.wait_for_odom()
        if odom_start is None:
            raise RuntimeError("No initial /odom sample was received.")

        tracker_start = read_tracker_pose()
        if tracker_start is None:
            node.get_logger().warn("No tracker start pose found.")

        node.send_cmd(
            motion["linear_x_mps"],
            motion["angular_z_radps"],
            motion["duration_sec"],
        )

        node.stop()

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

        odom_final = node.last_odom
        tracker_final = read_tracker_pose()
        if tracker_final is None:
            node.get_logger().warn("No tracker final pose found.")

        save_real_result(
            run_id,
            motion,
            tracker_start,
            tracker_final,
            odom_start,
            odom_final,
        )

    except KeyboardInterrupt:
        print("Interrupted. Sending stop command...")
        return 130

    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        return 1

    finally:
        close_motion_node(node, rclpy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
