"""The supported stop-go follower interface; all health gates are mandatory."""

import argparse
import math
from datetime import datetime
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Follow map-frame A* waypoints with strict TF/AMCL/scan gates.")
    parser.add_argument("--waypoints-csv", type=Path, default=Path("results/aufgabe03/aufgabe03_waypoints.csv"))
    parser.add_argument("--static-map", type=Path, default=Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml"))
    parser.add_argument("--run-id", default=datetime.now().strftime("follow_%Y%m%d_%H%M%S"))
    parser.add_argument("--controller", choices=("stop-go",), default="stop-go")
    parser.add_argument("--start-selection", choices=("path-progress",), default="path-progress")
    parser.add_argument("--yes", action="store_true", help="Acknowledge direct /cmd_vel control (required for live runs).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without ROS or robot motion.")
    for name, default in (("cmd-vel-topic", "/cmd_vel"), ("scan-topic", "/scan"),
                          ("amcl-topic", "/amcl_pose"), ("map-frame", "map"),
                          ("base-frame", "base_footprint"), ("fallback-base-frame", "base_link")):
        parser.add_argument("--" + name, default=default)
    settings = {
        "linear-speed-mps": 0.04, "min-linear-speed-mps": 0.012,
        "linear-gain": 0.25, "max-angular-speed-radps": 0.09, "yaw-gain": 0.35,
        "waypoint-tolerance-m": 0.12, "goal-tolerance-m": 0.12,
        "rotate-start-heading-error-deg": 20.0, "rotate-stop-heading-error-deg": 7.0,
        "forward-yaw-deadband-deg": 3.0, "forward-stop-heading-error-deg": 18.0,
        "min-waypoint-spacing-m": 0.12, "start-on-path-tolerance-m": 0.25,
        "scan-half-angle-deg": 35.0, "hard-stop-range-m": 0.16,
        "min-scan-range-m": 0.40, "rotation-stop-range-m": 0.18,
        "max-pose-age-sec": 1.0, "max-scan-age-sec": 1.0, "max-amcl-age-sec": 5.0,
        "max-amcl-var-x": 0.05, "max-amcl-var-y": 0.05, "max-amcl-var-yaw": 0.10,
        "max-waypoint-time-sec": 180.0, "control-rate-hz": 20.0,
        "startup-timeout-sec": 20.0, "health-recovery-time-sec": 5.0,
        "replan-timeout-sec": 5.0, "max-replan-scan-age-sec": 1.0,
        "max-replan-tf-age-sec": 1.0, "obstacle-forward-distance-m": 0.75,
        "obstacle-forward-half-width-m": 0.25, "obstacle-angle-window-deg": 45.0,
        "obstacle-min-range-m": 0.12, "robot-footprint-radius-m": 0.10,
        "max-start-snap-m": 0.20, "max-goal-snap-m": 0.30,
        "max-replan-path-length-ratio": 3.0, "run-local-map-inflation-radius-m": 0.15,
        "run-local-map-max-tf-age-sec": 1.0, "run-local-map-max-scan-age-sec": 1.0,
        "run-local-map-max-rejected-ratio": 0.90, "run-local-map-clearance-margin-m": 0.04,
    }
    for name, default in settings.items():
        parser.add_argument("--" + name, type=float, default=default)
    for name, default in (("max-replans", 3), ("run-local-map-max-updates", 3),
                          ("run-local-map-initial-scan-count", 5),
                          ("run-local-map-min-hit-count", 2),
                          ("run-local-map-min-used-points", 3)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--replan-on-obstacle", action="store_true")
    parser.add_argument("--run-local-obstacle-map", action="store_true",
                        help="Build a stopped fresh-scan overlay and replan before following.")
    parser.add_argument("--run-local-map-initial-scan-mode", choices=("forward", "full"), default="forward")
    parser.add_argument("--run-local-map-update-mode", choices=("forward", "full"), default="forward")
    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    zero_allowed = {"min_waypoint_spacing_m", "forward_yaw_deadband_deg"}
    for name, value in vars(args).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        lower_ok = value >= 0 if name in zero_allowed else value > 0
        if not math.isfinite(value) or not lower_ok:
            parser.error(f"--{name.replace('_', '-')} must be finite and {'nonnegative' if name in zero_allowed else 'positive'}")
    if args.min_linear_speed_mps > args.linear_speed_mps:
        parser.error("minimum linear speed exceeds linear speed")
    if not 0 < args.rotate_stop_heading_error_deg < args.rotate_start_heading_error_deg < 180:
        parser.error("rotate heading thresholds must satisfy 0 < stop < start < 180")
    if not args.forward_yaw_deadband_deg < args.forward_stop_heading_error_deg < 180:
        parser.error("forward heading deadband must be below the stop threshold and 180 degrees")
    if args.hard_stop_range_m < args.robot_footprint_radius_m:
        parser.error("hard stop clearance must cover the robot footprint")
    if min(args.min_scan_range_m, args.rotation_stop_range_m) < args.hard_stop_range_m:
        parser.error("scan stop thresholds must not be below hard stop clearance")
    if not 0 < args.scan_half_angle_deg <= 180 or args.obstacle_angle_window_deg > 180:
        parser.error("scan and obstacle angles must be in (0, 180]")
    if args.run_local_map_max_rejected_ratio > 1 or args.max_replan_path_length_ratio < 1:
        parser.error("rejected ratio must be <= 1 and path length ratio must be >= 1")
    clearance = args.robot_footprint_radius_m + args.run_local_map_clearance_margin_m
    if args.run_local_map_inflation_radius_m < clearance:
        parser.error("run-local inflation must cover footprint plus clearance")
    # Controller uses these physical quantities directly; there are no alternate modes.
    args.linear_speed = args.linear_speed_mps
    args.min_linear_speed = args.min_linear_speed_mps
    args.max_angular_speed = args.max_angular_speed_radps
    return args
