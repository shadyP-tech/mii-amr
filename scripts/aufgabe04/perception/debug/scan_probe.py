from __future__ import annotations

import argparse
import math
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print selected LaserScan ranges for debug-only LiDAR checks.")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--bearings-deg",
        default="-30,-20,-10,0,10,20,30",
        help="Comma-separated scan bearings in degrees to print.",
    )
    return parser


def parse_bearings_deg(value: str) -> tuple[float, ...]:
    bearings = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        bearings.append(float(raw))
    if not bearings:
        raise argparse.ArgumentTypeError("--bearings-deg must contain at least one value")
    return tuple(bearings)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bearings_deg = parse_bearings_deg(args.bearings_deg)

    try:
        import rclpy
        from rclpy.qos import QoSProfile, qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
    except ImportError as exc:
        raise SystemExit("scan_probe requires ROS 2 rclpy and sensor_msgs; source the ROS setup first.") from exc

    rclpy.init(args=None)
    node = rclpy.create_node("aufgabe04_scan_probe")
    box = {}

    def callback(msg) -> None:
        if "msg" not in box:
            box["msg"] = msg

    qos_profile = QoSProfile(
        reliability=qos_profile_sensor_data.reliability,
        durability=qos_profile_sensor_data.durability,
        history=qos_profile_sensor_data.history,
        depth=1,
    )
    node.create_subscription(LaserScan, args.scan_topic, callback, qos_profile)

    deadline = node.get_clock().now().nanoseconds / 1_000_000_000.0 + max(0.1, args.timeout_sec)
    while "msg" not in box:
        rclpy.spin_once(node, timeout_sec=0.2)
        now_sec = node.get_clock().now().nanoseconds / 1_000_000_000.0
        if now_sec > deadline:
            node.destroy_node()
            rclpy.shutdown()
            raise SystemExit(f"no LaserScan received on {args.scan_topic!r} within {args.timeout_sec:.1f}s")

    msg = box["msg"]
    print(f"topic {args.scan_topic}")
    print(f"frame {msg.header.frame_id}")
    print(f"angle_min {msg.angle_min:.6f} angle_max {msg.angle_max:.6f} inc {msg.angle_increment:.6f}")
    print(f"range_min {msg.range_min:.3f} range_max {msg.range_max:.3f} count {len(msg.ranges)}")

    for deg in bearings_deg:
        bearing = math.radians(deg)
        index = round((bearing - msg.angle_min) / msg.angle_increment) % len(msg.ranges)
        raw_range = float(msg.ranges[index])
        print(f"{deg:+.1f}deg idx={index} range={raw_range:.3f}")

    valid = [
        (index, float(raw_range), msg.angle_min + index * msg.angle_increment)
        for index, raw_range in enumerate(msg.ranges)
        if math.isfinite(float(raw_range)) and msg.range_min <= float(raw_range) <= msg.range_max
    ]
    if valid:
        nearest_index, nearest_range, nearest_bearing = min(valid, key=lambda item: item[1])
        print(
            "nearest_idx "
            f"{nearest_index} nearest_range {nearest_range:.3f} "
            f"nearest_bearing_rad {nearest_bearing:.3f} "
            f"nearest_bearing_deg {math.degrees(nearest_bearing):.1f}"
        )
    else:
        print("nearest none")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
