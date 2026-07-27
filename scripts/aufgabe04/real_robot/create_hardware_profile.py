#!/usr/bin/env python3
"""Create an immutable real-TurtleBot runtime profile."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.real_robot.hardware_profile import (
    REAL_HARDWARE_PROFILE_SCHEMA_VERSION,
    RealRobotProfile,
    camera_calibration_sha256,
    load_camera_calibration,
    write_real_robot_profile,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument(
        "--compressed-image-topic",
        default="camera/image_raw/compressed",
    )
    parser.add_argument("--camera-info-topic", default="camera/camera_info")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--scan-frame", default="base_scan")
    parser.add_argument("--localization-source", choices=("amcl", "tf"), default="amcl")
    parser.add_argument("--physical-site", required=True, type=Path)
    parser.add_argument("--physical-site-id", required=True)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument("--robot-radius-m", required=True, type=float)
    parser.add_argument("--scan-origin-to-base-offset-m", required=True, type=float)
    parser.add_argument("--max-linear-speed-mps", required=True, type=float)
    parser.add_argument("--max-angular-speed-radps", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        calibration = load_camera_calibration(args.camera_calibration)
        if calibration.base_frame != args.base_frame:
            raise ValueError("calibration base frame differs from runtime profile")
        profile = RealRobotProfile(
            schema_version=REAL_HARDWARE_PROFILE_SCHEMA_VERSION,
            profile_id=args.profile_id,
            robot_id=args.robot_id,
            namespace=args.namespace,
            scan_topic=args.scan_topic,
            odom_topic=args.odom_topic,
            cmd_vel_topic=args.cmd_vel_topic,
            amcl_topic=args.amcl_topic,
            compressed_image_topic=args.compressed_image_topic,
            camera_info_topic=args.camera_info_topic,
            map_frame=args.map_frame,
            odom_frame=args.odom_frame,
            base_frame=args.base_frame,
            scan_frame=args.scan_frame,
            camera_optical_frame=calibration.camera_optical_frame,
            localization_source=args.localization_source,
            physical_site_id=args.physical_site_id,
            physical_site_sha256=_file_sha256(args.physical_site),
            calibration_profile_sha256=camera_calibration_sha256(calibration),
            robot_radius_m=args.robot_radius_m,
            scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
            max_linear_speed_mps=args.max_linear_speed_mps,
            max_angular_speed_radps=args.max_angular_speed_radps,
        )
        digest = write_real_robot_profile(args.output, profile)
        print(
            f"real_robot_profile={args.output}\n"
            f"real_robot_profile_sha256={digest}\n"
            f"physical_site_sha256={profile.physical_site_sha256}"
        )
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

