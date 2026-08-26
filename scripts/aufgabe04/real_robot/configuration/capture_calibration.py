#!/usr/bin/env python3
"""Capture live CameraInfo and base-to-camera TF into an immutable profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.ros_runtime_config import resolve_topic
from scripts.aufgabe04.real_robot.configuration.profile import (
    CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION,
    CameraCalibrationProfile,
    RigidTransform,
    write_camera_calibration,
)


def _profile_from_messages(args, camera_info, transform) -> CameraCalibrationProfile:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    observed_frame = str(camera_info.header.frame_id).strip("/")
    if observed_frame != args.camera_optical_frame:
        raise ValueError(
            "CameraInfo frame differs from --camera-optical-frame: "
            f"{observed_frame!r}"
        )
    return CameraCalibrationProfile(
        schema_version=CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION,
        calibration_id=args.calibration_id,
        camera_optical_frame=args.camera_optical_frame,
        base_frame=args.base_frame,
        width_px=int(camera_info.width),
        height_px=int(camera_info.height),
        distortion_model=str(camera_info.distortion_model),
        distortion_coefficients=tuple(float(value) for value in camera_info.d),
        camera_matrix=tuple(float(value) for value in camera_info.k),
        rectification_matrix=tuple(float(value) for value in camera_info.r),
        projection_matrix=tuple(float(value) for value in camera_info.p),
        base_to_camera=RigidTransform(
            translation_xyz_m=(
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ),
            rotation_xyzw=(
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
        ),
        measured_unix_sec=time.time(),
        source=args.source,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--camera-info-topic", default="camera/camera_info")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--camera-optical-frame", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument(
        "--source",
        required=True,
        help="Calibration method/report identifier; do not use a vague default.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout_sec <= 0.0:
        parser.error("--timeout-sec must be positive")
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo
        from tf2_ros import Buffer, TransformException, TransformListener
    except ImportError as exc:
        parser.exit(2, f"error: ROS 2 Python packages are required: {exc}\n")

    rclpy.init(args=None)
    node = Node("aufgabe04_capture_real_camera_calibration")
    latest = {"camera_info": None}
    topic = resolve_topic(args.camera_info_topic, args.namespace)
    node.create_subscription(
        CameraInfo,
        topic,
        lambda message: latest.__setitem__("camera_info", message),
        qos_profile_sensor_data,
    )
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    deadline = time.monotonic() + args.timeout_sec
    try:
        if bool(node.get_parameter("use_sim_time").value):
            raise ValueError("real calibration capture requires use_sim_time=false")
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if latest["camera_info"] is None:
                continue
            try:
                transform = tf_buffer.lookup_transform(
                    args.base_frame,
                    args.camera_optical_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except TransformException:
                continue
            profile = _profile_from_messages(
                args,
                latest["camera_info"],
                transform,
            )
            digest = write_camera_calibration(args.output, profile)
            print(
                f"camera_calibration={args.output}\n"
                f"calibration_profile_sha256={digest}\n"
                f"camera_info_topic={topic}"
            )
            return 0
        raise TimeoutError(
            f"timed out waiting for {topic} and "
            f"{args.base_frame}<-{args.camera_optical_frame} TF"
        )
    except (OSError, TimeoutError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        del tf_listener
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
