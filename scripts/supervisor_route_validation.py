#!/usr/bin/env python3
"""
Run the real TurtleBot supervisor-route validation experiment.

The route action list is loaded from ``results/supervisor_route_prediction.json``.
This runner is intended for the final-target camera setup: the camera tracker
does not validate the start pose, but it must publish a fresh final pose after
the robot reaches the target area.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
except ImportError:
    rclpy = None
    Node = object
    Twist = None
    Odometry = object


DEFAULT_PREDICTION_FILE = Path("results/supervisor_route_prediction.json")
DEFAULT_RESULTS_CSV = Path("results/supervisor_route_validation_runs.csv")
DEFAULT_TRACKER_POSE_FILE = Path("results/latest_tracker_pose.csv")

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

CSV_HEADER = [
    "timestamp",
    "run_id",
    "prediction_file",
    "actions",
    "num_actions",
    "nominal_final_x",
    "nominal_final_y",
    "predicted_final_x",
    "predicted_final_y",
    "predicted_final_yaw_deg",
    "tracker_final_timestamp",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "tracker_error_dx",
    "tracker_error_dy",
    "tracker_error_m",
    "tracker_yaw_error_deg",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
    "odom_dx",
    "odom_dy",
    "odom_distance_m",
    "linear_speed_mps",
    "angular_speed_radps",
    "notes",
]


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def parse_action_number(text):
    return float(text.replace("p", "."))


def parse_action(action):
    value = action.strip().upper()
    match = ACTION_RE.match(value)
    if match is None:
        raise ValueError(f"invalid supervisor route action: {action!r}")

    kind, amount_text = match.groups()
    amount = parse_action_number(amount_text)
    if amount <= 0.0:
        raise ValueError(f"action amount must be positive: {action!r}")

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

    fixed_points = data.get("fixed_points") or []
    nominal_final = fixed_points[-1] if fixed_points else None
    final_yaw_mean_deg = prediction.get("final_yaw_mean_deg")

    return {
        "path": path,
        "actions": [str(action) for action in actions],
        "parsed_actions": parse_actions(actions),
        "nominal_final": nominal_final,
        "predicted_final": [float(endpoint_mu[0]), float(endpoint_mu[1])],
        "predicted_final_yaw_deg": (
            float(final_yaw_mean_deg) if final_yaw_mean_deg is not None else None
        ),
    }


def odom_to_xy_yaw(msg):
    if msg is None:
        return None

    p = msg.pose.pose.position
    q = msg.pose.pose.orientation

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {
        "x": p.x,
        "y": p.y,
        "yaw_deg": math.degrees(yaw),
    }


def read_tracker_pose(path=DEFAULT_TRACKER_POSE_FILE, min_mtime=None):
    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    if min_mtime is not None and stat.st_mtime < min_mtime:
        return None

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    row = rows[-1]
    valid_pose = row.get("valid_pose")
    if valid_pose is not None and valid_pose.strip().lower() not in {"1", "true"}:
        return None

    try:
        num_detected = int(float(row.get("num_detected", 0) or 0))
        pose = {
            "timestamp": row.get("timestamp", ""),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "yaw_deg": float(row["yaw_deg"]),
            "file_mtime": stat.st_mtime,
            "num_detected": num_detected,
        }
    except (KeyError, TypeError, ValueError):
        return None

    if num_detected < 3:
        return None

    if not all(math.isfinite(pose[key]) for key in ("x", "y", "yaw_deg")):
        return None

    return pose


def wait_for_tracker_pose(path, timeout_sec, min_mtime):
    deadline = time.time() + timeout_sec

    while time.time() <= deadline:
        pose = read_tracker_pose(path, min_mtime=min_mtime)
        if pose is not None:
            return pose
        time.sleep(0.25)

    return None


def pose_fields(pose):
    if pose is None:
        return ["", "", ""]
    return [pose["x"], pose["y"], pose["yaw_deg"]]


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


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        with path.open(newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != header:
            raise RuntimeError(
                f"{path} has an unrecognized schema. Move or migrate it before "
                "appending supervisor validation results."
            )

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


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
    tracker_dx, tracker_dy, tracker_error_m, tracker_yaw_error = tracker_error(
        prediction,
        tracker_final,
    )
    odom_dx, odom_dy, odom_distance = xy_delta(odom_start, odom_final)
    nominal_final = prediction["nominal_final"] or ["", ""]

    return [
        datetime.now().isoformat(),
        run_id,
        str(prediction["path"]),
        ",".join(prediction["actions"]),
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
        tracker_final["timestamp"] if tracker_final is not None else "",
        *(pose_fields(tracker_final)),
        tracker_dx,
        tracker_dy,
        tracker_error_m,
        tracker_yaw_error,
        *(pose_fields(odom_start)),
        *(pose_fields(odom_final)),
        odom_dx,
        odom_dy,
        odom_distance,
        linear_speed_mps,
        angular_speed_radps,
        notes,
    ]


class SupervisorRouteValidationNode(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the validation route."
            )

        super().__init__("supervisor_route_validation")
        self.last_odom = None
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )
        time.sleep(1.0)

    def odom_callback(self, msg):
        self.last_odom = msg

    def wait_for_odom(self, timeout_sec=5.0):
        start = time.time()
        while rclpy.ok() and self.last_odom is None:
            if time.time() - start > timeout_sec:
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_odom

    def publish_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def stop(self):
        msg = Twist()
        for _ in range(10):
            if rclpy.ok():
                self.pub.publish(msg)
            time.sleep(0.05)

    def drive_forward(self, distance_m, speed_mps, tolerance_m):
        start_msg = self.wait_for_odom()
        if start_msg is None:
            raise RuntimeError("No /odom sample before forward primitive.")

        start_pose = odom_to_xy_yaw(start_msg)
        start_yaw_rad = math.radians(start_pose["yaw_deg"])
        timeout_sec = distance_m / speed_mps + DEFAULT_FORWARD_TIMEOUT_MARGIN_SEC
        start_time = time.time()
        last_log_time = start_time

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

            if time.time() - last_log_time >= 2.0:
                self.get_logger().info(
                    f"Forward progress {progress_m:.3f}/{distance_m:.3f} m"
                )
                last_log_time = time.time()

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
        start_time = time.time()
        last_log_time = start_time

        while rclpy.ok():
            current_pose = odom_to_xy_yaw(self.last_odom)
            current_yaw = current_pose["yaw_deg"]
            accumulated_deg += shortest_angle_delta_deg(previous_yaw, current_yaw)
            previous_yaw = current_yaw

            if abs(accumulated_deg) >= target_abs - tolerance_deg:
                self.stop()
                return

            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                self.stop()
                raise RuntimeError(
                    "Timed out during rotation primitive: "
                    f"progress={accumulated_deg:.1f} deg, target={angle_deg:.1f} deg"
                )

            if time.time() - last_log_time >= 2.0:
                self.get_logger().info(
                    f"Rotation progress {accumulated_deg:.1f}/{angle_deg:.1f} deg"
                )
                last_log_time = time.time()

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
    print(f"Actions: {','.join(prediction['actions'])}")
    response = input("Type RUN to start the validation route: ").strip()
    return response == "RUN"


def print_dry_run(prediction, args):
    print(f"Prediction file: {prediction['path']}")
    print(f"Actions: {','.join(prediction['actions'])}")
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
    parser = argparse.ArgumentParser(
        description="Run the real supervisor-route validation experiment.",
    )
    parser.add_argument("run_id_arg", nargs="?", help="Optional run ID.")
    parser.add_argument("--run-id", help="Run ID for CSV and bag association.")
    parser.add_argument(
        "--prediction",
        default=DEFAULT_PREDICTION_FILE,
        type=Path,
        help="Supervisor route prediction JSON.",
    )
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        type=Path,
        help="CSV file for validation rows.",
    )
    parser.add_argument(
        "--tracker-pose-file",
        default=DEFAULT_TRACKER_POSE_FILE,
        type=Path,
        help="latest_tracker_pose.csv path visible on this host.",
    )
    parser.add_argument(
        "--linear-speed",
        default=DEFAULT_LINEAR_SPEED_MPS,
        type=float,
        help="Forward command speed in m/s.",
    )
    parser.add_argument(
        "--angular-speed",
        default=DEFAULT_ANGULAR_SPEED_RADPS,
        type=float,
        help="Absolute angular command speed in rad/s.",
    )
    parser.add_argument(
        "--settle-sec",
        default=DEFAULT_SETTLE_SEC,
        type=float,
        help="Pause after each primitive.",
    )
    parser.add_argument(
        "--forward-tolerance-m",
        default=DEFAULT_FORWARD_TOLERANCE_M,
        type=float,
        help="Odometry tolerance for forward primitives.",
    )
    parser.add_argument(
        "--rotation-tolerance-deg",
        default=DEFAULT_ROTATION_TOLERANCE_DEG,
        type=float,
        help="Odometry tolerance for rotation primitives.",
    )
    parser.add_argument(
        "--final-tracker-timeout-sec",
        default=DEFAULT_FINAL_TRACKER_TIMEOUT_SEC,
        type=float,
        help="Time to wait for a fresh valid final tracker pose.",
    )
    parser.add_argument(
        "--skip-final-tracker",
        action="store_true",
        help="Do not wait for final tracker pose; write blank tracker fields.",
    )
    parser.add_argument(
        "--notes",
        default="supervisor_route_validation",
        help="Notes value written to the validation CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the route and exit without ROS or /cmd_vel.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive safety confirmation.",
    )
    args = parser.parse_args(argv)

    args.run_id = args.run_id or args.run_id_arg
    if not args.run_id:
        args.run_id = datetime.now().strftime("supervisor_validation_%Y%m%d_%H%M%S")

    if args.linear_speed <= 0.0:
        parser.error("--linear-speed must be greater than zero")
    if args.angular_speed <= 0.0:
        parser.error("--angular-speed must be greater than zero")
    if args.settle_sec < 0.0:
        parser.error("--settle-sec must be non-negative")
    if args.final_tracker_timeout_sec < 0.0:
        parser.error("--final-tracker-timeout-sec must be non-negative")

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
        node.get_logger().info(f"Starting supervisor validation run: {args.run_id}")
        node.get_logger().info(
            "Predicted final: "
            f"x={prediction['predicted_final'][0]:.3f} m, "
            f"y={prediction['predicted_final'][1]:.3f} m"
        )
        node.get_logger().info("Waiting for initial odometry...")
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
                    "results/latest_tracker_pose.csv is visible on this host."
                )
            else:
                node.get_logger().info(
                    "Final tracker pose: "
                    f"x={tracker_final['x']:.3f} m, "
                    f"y={tracker_final['y']:.3f} m, "
                    f"yaw={tracker_final['yaw_deg']:.1f} deg"
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
        try:
            node.stop()
        except Exception:
            pass
        return 130

    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        try:
            node.stop()
        except Exception:
            pass
        return 1

    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
