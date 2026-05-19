#!/usr/bin/env python3
"""
Analyze offline arena geometry from recorded scan data.

This is a Commit-A diagnostic tool. It reads a rosbag or JSON scan sample file,
writes geometry diagnostics as JSON, and never commands or initializes the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arena_geometry_localizer import (
    ArenaGeometryConfig,
    Pose2D,
    ScanSample,
    analyze_scan_samples,
    load_scan_samples_json,
    write_json,
)


def yaw_from_quaternion(qx, qy, qz, qw):
    import math

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def odom_pose_from_msg(msg):
    pose = msg.pose.pose
    orientation = pose.orientation
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw_deg=yaw_from_quaternion(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ),
    )


def scan_sample_from_msg(msg, odom_pose):
    return ScanSample(
        ranges=list(msg.ranges),
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_increment),
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        odom_pose=odom_pose,
    )


def read_rosbag_samples(
    bag_path,
    scan_topic="/scan",
    odom_topic="/odom",
    storage_id="sqlite3",
    max_scan_samples=None,
):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS bag reading requires a sourced ROS 2 Python environment "
            "with rosbag2_py, rclpy, and rosidl_runtime_py available."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {
        metadata.name: metadata.type
        for metadata in reader.get_all_topics_and_types()
    }
    if scan_topic not in topic_types:
        raise RuntimeError(f"Bag does not contain scan topic {scan_topic!r}")
    if odom_topic not in topic_types:
        raise RuntimeError(f"Bag does not contain odom topic {odom_topic!r}")

    scan_type = get_message(topic_types[scan_topic])
    odom_type = get_message(topic_types[odom_topic])
    latest_odom_pose = None
    samples = []

    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic == odom_topic:
            latest_odom_pose = odom_pose_from_msg(deserialize_message(data, odom_type))
        elif topic == scan_topic:
            scan_msg = deserialize_message(data, scan_type)
            samples.append(scan_sample_from_msg(scan_msg, latest_odom_pose))
            if max_scan_samples is not None and len(samples) >= max_scan_samples:
                break

    if not samples:
        raise RuntimeError(f"No scan samples were read from {scan_topic!r}")
    return samples


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Analyze rectangular arena geometry from an offline bag or JSON scan samples.",
    )
    parser.add_argument("--bag", type=Path, help="ROS 2 bag directory to read.")
    parser.add_argument(
        "--input-json",
        type=Path,
        help="JSON scan sample file for non-ROS tests/debugging.",
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-scan-samples", type=int)

    parser.add_argument("--arena-length-m", default=3.90, type=float)
    parser.add_argument("--arena-width-m", default=1.898, type=float)
    parser.add_argument("--arena-map-center-x", default=0.0, type=float)
    parser.add_argument("--arena-map-center-y", default=0.0, type=float)
    parser.add_argument("--arena-map-yaw-deg", default=0.0, type=float)
    parser.add_argument("--heater-wall-side", default="+x", choices=["+x", "-x"])
    parser.add_argument("--arena-min-wall-points", default=20, type=int)
    parser.add_argument("--arena-max-wall-separation-error-m", default=0.20, type=float)
    parser.add_argument("--arena-max-line-rmse-m", default=0.08, type=float)
    parser.add_argument("--arena-min-parallel-score", default=0.90, type=float)
    parser.add_argument("--arena-min-short-wall-confidence", default=0.75, type=float)
    parser.add_argument("--arena-min-classification-margin", default=0.15, type=float)
    args = parser.parse_args(argv)

    if bool(args.bag) == bool(args.input_json):
        parser.error("Provide exactly one of --bag or --input-json")
    if args.max_scan_samples is not None and args.max_scan_samples < 1:
        parser.error("--max-scan-samples must be >= 1")
    return args


def config_from_args(args):
    return ArenaGeometryConfig(
        arena_length_m=args.arena_length_m,
        arena_width_m=args.arena_width_m,
        map_center_x=args.arena_map_center_x,
        map_center_y=args.arena_map_center_y,
        map_yaw_deg=args.arena_map_yaw_deg,
        heater_wall_side=args.heater_wall_side,
        min_wall_points=args.arena_min_wall_points,
        max_wall_separation_error_m=args.arena_max_wall_separation_error_m,
        max_line_rmse_m=args.arena_max_line_rmse_m,
        min_parallel_score=args.arena_min_parallel_score,
        min_short_wall_confidence=args.arena_min_short_wall_confidence,
        min_classification_margin=args.arena_min_classification_margin,
    )


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.input_json:
            samples = load_scan_samples_json(args.input_json)
            source = str(args.input_json)
            source_type = "json"
        else:
            samples = read_rosbag_samples(
                args.bag,
                scan_topic=args.scan_topic,
                odom_topic=args.odom_topic,
                storage_id=args.storage_id,
                max_scan_samples=args.max_scan_samples,
            )
            source = str(args.bag)
            source_type = "rosbag"
        result = analyze_scan_samples(samples, config_from_args(args))
        output = result.to_dict()
        output["source"] = {
            "type": source_type,
            "path": source,
            "scan_topic": args.scan_topic,
            "odom_topic": args.odom_topic,
            "sample_count": len(samples),
        }
        write_json(args.output, output)
    except Exception as exc:
        print(f"analyze_arena_geometry_from_bag.py: error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote arena geometry diagnostics: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
