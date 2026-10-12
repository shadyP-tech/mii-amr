"""Shared ROS motion I/O and the pose/CSV format used by Aufgabe02 trials."""
import csv
import math
import time
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
except ImportError:
    rclpy = None
    Node = object
    Twist = Odometry = None


POSE_FIELDS = ("x", "y", "yaw_deg")


def pose_columns(prefix):
    return [f"{prefix}_{field}" for field in POSE_FIELDS]


FORWARD_CSV_HEADER = ["timestamp", "run_id"] + [
    column for prefix in ("tracker_start", "tracker_final", "odom_start", "odom_final")
    for column in pose_columns(prefix)
] + ["notes"]


def pose_fields(pose):
    return [pose[field] for field in POSE_FIELDS] if pose is not None else [""] * 3


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def parse_float_env(env, name, default):
    try:
        value = float(env.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def ensure_csv_schema(path, header):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="") as handle:
        if next(csv.reader(handle), None) != header:
            raise RuntimeError(f"{path} has an unrecognized schema; move or migrate it first.")
    return True


def append_csv_row(path, header, row):
    path = Path(path)
    file_exists = ensure_csv_schema(path, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def odom_to_xy_yaw(msg):
    if msg is None:
        return None
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return {"x": p.x, "y": p.y, "yaw_deg": math.degrees(yaw)}


def read_tracker_pose(path="results/aufgabe02/latest_tracker_pose.csv", min_mtime=None):
    path = Path(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    if min_mtime is not None and mtime < min_mtime:
        return None
    with path.open(newline="") as handle:
        row = next(reversed(list(csv.DictReader(handle))), None)
    if not row:
        return None
    valid = row.get("valid_pose")
    if valid is not None and valid.strip().lower() not in {"1", "true"}:
        return None
    try:
        detected = int(float(row.get("num_detected", 0) or 0))
        pose = {field: float(row[field]) for field in POSE_FIELDS}
        if detected < 3 or not all(math.isfinite(value) for value in pose.values()):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return dict(pose, timestamp=row.get("timestamp", ""), file_mtime=mtime, num_detected=detected)


class DriveNode(Node):
    def __init__(self, run_id="", node_name="scripted_drive"):
        if rclpy is None:
            raise RuntimeError("Source ROS 2 Humble before running motion commands.")
        super().__init__(node_name)
        self.run_id = run_id
        self.last_odom = None
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
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
        msg.linear.x, msg.angular.z = linear_x, angular_z
        self.pub.publish(msg)

    def send_cmd(self, linear_x, angular_z, duration):
        start = time.time()
        while rclpy.ok() and time.time() - start < duration:
            self.publish_velocity(linear_x, angular_z)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)
        self.stop()

    def stop(self):
        msg = Twist()
        for _ in range(10):
            if rclpy.ok():
                self.pub.publish(msg)
            time.sleep(0.05)


def close_motion_node(node, ros):
    try:
        node.stop()
    finally:
        node.destroy_node()
        if ros.ok():
            ros.shutdown()
