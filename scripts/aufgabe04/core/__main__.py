"""Validate measured inputs, or explicitly start the physical exploration loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config
from .planning import load_map


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-calibration", type=Path, default=root /
                        "configs/aufgabe04/camera_calibration/turtlebot1_camera_base_footprint_20260730.json",
                        help="measured calibration JSON pinned by the robot profile")
    parser.add_argument("--robot-profile", type=Path, default=root /
                        "configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json")
    parser.add_argument("--stand-model", type=Path, default=root /
                        "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    parser.add_argument("--site", type=Path, default=root /
                        "docs/setups/aufgabe04_lab_20260817.json")
    parser.add_argument("--execute", action="store_true",
                        help="start ROS motion after an interactive RUN confirmation")
    args = parser.parse_args(argv)
    try:
        config = load_config(robot_profile=args.robot_profile,
            camera_calibration=args.camera_calibration, stand_model_profile=args.stand_model,
            physical_site=args.site, repository_root=root)
        grid = load_map(config.map_yaml, expected_yaml_sha256=config.expected_yaml_sha256,
            expected_image_sha256=config.expected_image_sha256,
            expected_map_bundle_sha256=config.expected_map_bundle_sha256,
            semantic_map_id=config.semantic_map_id, planning_frame=config.map_frame)
        print(f"Validated {config.robot_id}: {config.semantic_map_id}, "
              f"{grid.width} x {grid.height} cells; maximum "
              f"{config.max_linear_speed_mps:.3f} m/s, {config.max_angular_speed_radps:.3f} rad/s.")
        if not args.execute:
            print("Inputs validated. Add --execute on the robot to start exploration.")
            return 0
        if not sys.stdin.isatty():
            raise ValueError("physical execution requires an interactive terminal")
        if input("Ready with the physical emergency stop. Type RUN to start: ") != "RUN":
            print("Exploration cancelled.")
            return 2
        from .mission import run_mission
        from .ros_runtime import RobotSession
        with RobotSession(config, map_id=grid.map_id) as robot:
            robot.authorize_motion("RUN")
            result = run_mission(robot, config, grid)
        print(f"Exploration complete: {result.visited_viewpoints} viewpoints, "
              f"{result.coverage_ratio:.1%} mapped visibility.")
        for target, status in result.observations:
            print(f"  {target}: {status}")
        return 0
    except KeyboardInterrupt:
        print("Exploration interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Exploration stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
