#!/usr/bin/env python3

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__:
    from .motion import (
        DriveNode, append_csv_row, close_motion_node, odom_to_xy_yaw,
        pose_columns, pose_fields, rclpy, read_tracker_pose, shortest_angle_delta_deg,
    )
else:
    from motion import (
        DriveNode, append_csv_row, close_motion_node, odom_to_xy_yaw,
        pose_columns, pose_fields, rclpy, read_tracker_pose, shortest_angle_delta_deg,
    )



DEFAULT_PREDICTION_FILE = Path("results/aufgabe02/supervisor_route_prediction.json")
DEFAULT_RESULTS_CSV = Path("results/aufgabe02/supervisor_route_validation_runs.csv")
DEFAULT_TRACKER_POSE_FILE = Path("results/aufgabe02/latest_tracker_pose.csv")

DEFAULT_LINEAR_SPEED_MPS = 0.10
DEFAULT_ANGULAR_SPEED_RADPS = 0.30
DEFAULT_SETTLE_SEC = 0.50
DEFAULT_FORWARD_TIMEOUT_MARGIN_SEC = 5.0
DEFAULT_ROTATION_TIMEOUT_MARGIN_SEC = 5.0
DEFAULT_FORWARD_TOLERANCE_M = 0.01
DEFAULT_ROTATION_TOLERANCE_DEG = 2.0
DEFAULT_FINAL_TRACKER_TIMEOUT_SEC = 90.0
COMMAND_PERIOD_SEC = 0.05

ACTION_RE = re.compile(r"^(F|CW|CCW)([0-9]+(?:p[0-9]+)?(?:\.[0-9]+)?)$")
COMPARISON_FRAME = "model_mirror_y"

CSV_HEADER = [
    "timestamp", "run_id", "prediction_file", "actions", "model_actions", "num_actions",
    "nominal_final_x", "nominal_final_y", "predicted_final_x", "predicted_final_y",
    "predicted_final_yaw_deg", "comparison_frame", "tracker_final_timestamp",
    *pose_columns("tracker_final"),
    "tracker_final_x_model", "tracker_final_y_model", "tracker_final_yaw_model_deg",
    "tracker_error_dx", "tracker_error_dy", "tracker_error_m", "tracker_yaw_error_deg",
    *pose_columns("odom_start"), *pose_columns("odom_final"),
    "odom_final_x_model", "odom_final_y_model", "odom_final_yaw_model_deg",
    "odom_dx", "odom_dy", "odom_distance_m", "linear_speed_mps", "angular_speed_radps", "notes",
]


def normalize_angle_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def parse_action_number(text):
    return float(text.replace("p", "."))


def parse_action(action):
    value = action.strip().upper()
    match = ACTION_RE.match(value)
    if match is None:
        raise ValueError(f"invalid supervisor route action: {action!r}")

    kind, amount_text = match.groups()
    amount = parse_action_number(amount_text)
    if not math.isfinite(amount) or amount <= 0.0:
        raise ValueError(f"action amount must be finite and positive: {action!r}")

    if kind == "F":
        return {
            "raw": value,
            "kind": "forward",
            "distance_m": amount / 100.0,
        }

    angle_deg = amount if kind == "CCW" else -amount
    return {
        "raw": value,
        "kind": "rotate",
        "angle_deg": angle_deg,
    }


def parse_actions(actions):
    return [parse_action(action) for action in actions]


def normalized_action_labels(actions):
    return [parse_action(action)["raw"] for action in actions]


def load_prediction(path):
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    actions = data.get("actions")
    if not actions:
        raise ValueError(f"{path} does not contain a non-empty actions list")

    prediction = data.get("prediction") or {}
    endpoint_mu = prediction.get("endpoint_mu")
    if endpoint_mu is None or len(endpoint_mu) != 2:
        raise ValueError(f"{path} is missing prediction.endpoint_mu")

    model_actions = normalized_action_labels(actions)
    execution_actions = data.get("execution_actions") or model_actions
    execution_actions = normalized_action_labels(execution_actions)
    if len(execution_actions) != len(model_actions):
        raise ValueError(
            f"{path} execution_actions length does not match actions length"
        )

    fixed_points = data.get("fixed_points") or []
    nominal_final = fixed_points[-1] if fixed_points else None
    final_yaw_mean_deg = prediction.get("final_yaw_mean_deg")

    return {
        "path": path,
        "actions": execution_actions,
        "model_actions": model_actions,
        "parsed_actions": parse_actions(execution_actions),
        "nominal_final": nominal_final,
        "predicted_final": [float(endpoint_mu[0]), float(endpoint_mu[1])],
        "predicted_final_yaw_deg": (
            float(final_yaw_mean_deg) if final_yaw_mean_deg is not None else None
        ),
    }


def wait_for_tracker_pose(path, timeout_sec, min_mtime):
    deadline = time.time() + timeout_sec

    while time.time() <= deadline:
        pose = read_tracker_pose(path, min_mtime=min_mtime)
        if pose is not None:
            return pose
        time.sleep(0.25)

    return None


def model_frame_pose(pose):
    """Reflect a tracker/physical pose into the mirrored model frame."""
    if pose is None:
        return None
    result = dict(pose)
    result["x"] = pose["x"]
    result["y"] = -pose["y"]
    result["yaw_deg"] = normalize_angle_deg(-pose["yaw_deg"])
    return result


def xy_delta(start_pose, final_pose):
    if start_pose is None or final_pose is None:
        return "", "", ""

    dx = final_pose["x"] - start_pose["x"]
    dy = final_pose["y"] - start_pose["y"]
    return dx, dy, math.hypot(dx, dy)


def tracker_error(prediction, tracker_final):
    if tracker_final is None:
        return "", "", "", ""

    predicted = prediction["predicted_final"]
    dx = tracker_final["x"] - predicted[0]
    dy = tracker_final["y"] - predicted[1]
    error_m = math.hypot(dx, dy)

    predicted_yaw = prediction["predicted_final_yaw_deg"]
    if predicted_yaw is None:
        yaw_error = ""
    else:
        yaw_error = shortest_angle_delta_deg(predicted_yaw, tracker_final["yaw_deg"])

    return dx, dy, error_m, yaw_error


def build_result_row(
    run_id,
    prediction,
    tracker_final,
    odom_start,
    odom_final,
    linear_speed_mps,
    angular_speed_radps,
    notes,
):
    tracker_final_model = model_frame_pose(tracker_final)
    odom_final_model = model_frame_pose(odom_final)
    tracker_dx, tracker_dy, tracker_error_m, tracker_yaw_error = tracker_error(
        prediction,
        tracker_final_model,
    )
    odom_dx, odom_dy, odom_distance = xy_delta(odom_start, odom_final)
    nominal_final = prediction["nominal_final"] or ["", ""]
    model_actions = prediction.get("model_actions", prediction["actions"])

    return [
        datetime.now().isoformat(),
        run_id,
        str(prediction["path"]),
        ",".join(prediction["actions"]),
        ",".join(model_actions),
        len(prediction["actions"]),
        nominal_final[0],
        nominal_final[1],
        prediction["predicted_final"][0],
        prediction["predicted_final"][1],
        (
            prediction["predicted_final_yaw_deg"]
            if prediction["predicted_final_yaw_deg"] is not None
            else ""
        ),
        COMPARISON_FRAME,
        tracker_final["timestamp"] if tracker_final is not None else "",
        *(pose_fields(tracker_final)),
        *(pose_fields(tracker_final_model)),
        tracker_dx,
        tracker_dy,
        tracker_error_m,
        tracker_yaw_error,
        *(pose_fields(odom_start)),
        *(pose_fields(odom_final)),
        *(pose_fields(odom_final_model)),
        odom_dx,
        odom_dy,
        odom_distance,
        linear_speed_mps,
        angular_speed_radps,
        notes,
    ]


class SupervisorRouteValidationNode(DriveNode):
    def __init__(self):
        super().__init__(node_name="supervisor_route_validation")

    def drive_forward(self, distance_m, speed_mps, tolerance_m):
        start_msg = self.wait_for_odom()
        if start_msg is None:
            raise RuntimeError("No /odom sample before forward primitive.")

        start_pose = odom_to_xy_yaw(start_msg)
        start_yaw_rad = math.radians(start_pose["yaw_deg"])
        timeout_sec = distance_m / speed_mps + DEFAULT_FORWARD_TIMEOUT_MARGIN_SEC
        if not math.isfinite(timeout_sec):
            raise RuntimeError("Forward primitive timeout must be finite.")
        start_time = time.time()

        while rclpy.ok():
            current_pose = odom_to_xy_yaw(self.last_odom)
            dx = current_pose["x"] - start_pose["x"]
            dy = current_pose["y"] - start_pose["y"]
            progress_m = dx * math.cos(start_yaw_rad) + dy * math.sin(start_yaw_rad)

            if progress_m >= distance_m - tolerance_m:
                self.stop()
                return

            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                self.stop()
                raise RuntimeError(
                    "Timed out during forward primitive: "
                    f"progress={progress_m:.3f} m, target={distance_m:.3f} m"
                )

            self.publish_velocity(speed_mps, 0.0)
            rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
            time.sleep(COMMAND_PERIOD_SEC)

        self.stop()
        raise RuntimeError("ROS shutdown during forward primitive.")

    def rotate(self, angle_deg, angular_speed_radps, tolerance_deg):
        start_msg = self.wait_for_odom()
        if start_msg is None:
            raise RuntimeError("No /odom sample before rotation primitive.")

        previous_yaw = odom_to_xy_yaw(start_msg)["yaw_deg"]
        accumulated_deg = 0.0
        sign = 1.0 if angle_deg > 0.0 else -1.0
        angular_z = sign * abs(angular_speed_radps)
        target_abs = abs(angle_deg)
        timeout_sec = (
            math.radians(target_abs) / abs(angular_speed_radps)
            + DEFAULT_ROTATION_TIMEOUT_MARGIN_SEC
        )
        if not math.isfinite(timeout_sec):
            raise RuntimeError("Rotation primitive timeout must be finite.")
        start_time = time.time()

        while rclpy.ok():
            current_pose = odom_to_xy_yaw(self.last_odom)
            current_yaw = current_pose["yaw_deg"]
            accumulated_deg += shortest_angle_delta_deg(previous_yaw, current_yaw)
            previous_yaw = current_yaw

            if sign * accumulated_deg >= target_abs - tolerance_deg:
                self.stop()
                return

            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                self.stop()
                raise RuntimeError(
                    "Timed out during rotation primitive: "
                    f"progress={accumulated_deg:.1f} deg, target={angle_deg:.1f} deg"
                )

            self.publish_velocity(0.0, angular_z)
            rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
            time.sleep(COMMAND_PERIOD_SEC)

        self.stop()
        raise RuntimeError("ROS shutdown during rotation primitive.")

    def execute_actions(
        self,
        actions,
        linear_speed_mps,
        angular_speed_radps,
        settle_sec,
        forward_tolerance_m,
        rotation_tolerance_deg,
    ):
        for index, action in enumerate(actions, start=1):
            prefix = f"[{index}/{len(actions)}] {action['raw']}"
            if action["kind"] == "forward":
                self.get_logger().info(
                    f"{prefix}: forward {action['distance_m']:.3f} m"
                )
                self.drive_forward(
                    action["distance_m"],
                    linear_speed_mps,
                    forward_tolerance_m,
                )
            else:
                self.get_logger().info(
                    f"{prefix}: rotate {action['angle_deg']:.1f} deg"
                )
                self.rotate(
                    action["angle_deg"],
                    angular_speed_radps,
                    rotation_tolerance_deg,
                )

            self.stop()
            time.sleep(settle_sec)


def require_motion_confirmation(args, prediction):
    if args.yes:
        return True

    print("\nThis command will publish /cmd_vel to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - clear the test area")
    print("  - keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print(f"Run ID: {args.run_id}")
    print(f"Execution actions: {','.join(prediction['actions'])}")
    model_actions = prediction.get("model_actions", prediction["actions"])
    if model_actions != prediction["actions"]:
        print(f"Model actions: {','.join(model_actions)}")
    response = input("Type RUN to start the validation route: ").strip()
    return response == "RUN"


def print_dry_run(prediction, args):
    print(f"Prediction file: {prediction['path']}")
    print(f"Execution actions: {','.join(prediction['actions'])}")
    model_actions = prediction.get("model_actions", prediction["actions"])
    if model_actions != prediction["actions"]:
        print(f"Model actions: {','.join(model_actions)}")
    print(
        "Predicted final: "
        f"x={prediction['predicted_final'][0]:.3f} m, "
        f"y={prediction['predicted_final'][1]:.3f} m"
    )
    if prediction["predicted_final_yaw_deg"] is not None:
        print(f"Predicted yaw: {prediction['predicted_final_yaw_deg']:.1f} deg")
    print(f"Linear speed: {args.linear_speed:.3f} m/s")
    print(f"Angular speed: {args.angular_speed:.3f} rad/s")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the real supervisor route.")
    parser.add_argument("run_id_arg", nargs="?", help="Optional run ID.")
    parser.add_argument("--run-id", help="Run ID for CSV and bag association.")
    for option, default in (
        ("prediction", DEFAULT_PREDICTION_FILE),
        ("results-csv", DEFAULT_RESULTS_CSV),
        ("tracker-pose-file", DEFAULT_TRACKER_POSE_FILE),
    ):
        parser.add_argument("--" + option, type=Path, default=default)
    numeric_defaults = {
        "linear-speed": DEFAULT_LINEAR_SPEED_MPS,
        "angular-speed": DEFAULT_ANGULAR_SPEED_RADPS,
        "settle-sec": DEFAULT_SETTLE_SEC,
        "forward-tolerance-m": DEFAULT_FORWARD_TOLERANCE_M,
        "rotation-tolerance-deg": DEFAULT_ROTATION_TOLERANCE_DEG,
        "final-tracker-timeout-sec": DEFAULT_FINAL_TRACKER_TIMEOUT_SEC,
    }
    for option, default in numeric_defaults.items():
        parser.add_argument("--" + option, type=float, default=default)
    parser.add_argument("--skip-final-tracker", action="store_true", help="Write blank tracker fields.")
    parser.add_argument("--notes", default="supervisor_route_validation")
    parser.add_argument("--dry-run", action="store_true", help="Print route without ROS or motion.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive RUN confirmation.")
    args = parser.parse_args(argv)
    args.run_id = args.run_id or args.run_id_arg or datetime.now().strftime("supervisor_validation_%Y%m%d_%H%M%S")
    for option in numeric_defaults:
        value = getattr(args, option.replace("-", "_"))
        if not math.isfinite(value):
            parser.error(f"--{option} must be finite")
        if option in {"linear-speed", "angular-speed"}:
            if value <= 0.0:
                parser.error(f"--{option} must be greater than zero")
        elif value < 0.0:
            parser.error(f"--{option} must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        prediction = load_prediction(args.prediction)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not load supervisor prediction: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(prediction, args)
        return 0

    if not require_motion_confirmation(args, prediction):
        print("Validation route cancelled.")
        return 130

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = SupervisorRouteValidationNode()
    odom_start = None
    odom_final = None
    tracker_final = None

    try:
        node.get_logger().info(f"{args.run_id}: {','.join(prediction['actions'])}")
        odom_start_msg = node.wait_for_odom()
        if odom_start_msg is None:
            raise RuntimeError("No initial /odom sample was received.")
        odom_start = odom_to_xy_yaw(odom_start_msg)

        node.execute_actions(
            prediction["parsed_actions"],
            args.linear_speed,
            args.angular_speed,
            args.settle_sec,
            args.forward_tolerance_m,
            args.rotation_tolerance_deg,
        )

        node.stop()
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        odom_final = odom_to_xy_yaw(node.last_odom)

        if args.skip_final_tracker:
            node.get_logger().warn("Skipping final tracker-pose wait.")
        else:
            min_mtime = time.time()
            node.get_logger().info(
                "Waiting for fresh final tracker pose at "
                f"{args.tracker_pose_file}..."
            )
            tracker_final = wait_for_tracker_pose(
                args.tracker_pose_file,
                args.final_tracker_timeout_sec,
                min_mtime=min_mtime,
            )
            if tracker_final is None:
                node.get_logger().error(
                    "No fresh valid final tracker pose was found. "
                    "Check that vision_tracker/main.py is running and that "
                    "results/aufgabe02/latest_tracker_pose.csv is visible on this host."
                )

        row = build_result_row(
            args.run_id,
            prediction,
            tracker_final,
            odom_start,
            odom_final,
            args.linear_speed,
            args.angular_speed,
            args.notes if tracker_final is not None or args.skip_final_tracker
            else f"{args.notes};missing_final_tracker",
        )
        append_csv_row(args.results_csv, CSV_HEADER, row)
        node.get_logger().info(f"Saved result to {args.results_csv}")

        if tracker_final is None and not args.skip_final_tracker:
            return 1

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
