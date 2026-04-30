#!/usr/bin/env python3
"""
Block a real experiment run until the camera start pose is acceptable.

This script never opens the camera.  It only reads the latest pose written by
``vision_tracker/main.py``.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import config
import start_pose


LOG_HEADER = [
    "timestamp",
    "run_id",
    "ref_x",
    "ref_y",
    "ref_yaw_deg",
    "measured_x",
    "measured_y",
    "measured_yaw_deg",
    "dx",
    "dy",
    "position_error_m",
    "yaw_error_deg",
    "position_tolerance_m",
    "yaw_tolerance_deg",
    "pose_age_sec",
    "stable_time_sec",
    "accepted",
]


def main():
    parser = argparse.ArgumentParser(
        description="Wait until the tracked robot start pose is acceptable.",
    )
    parser.add_argument("run_id")
    parser.add_argument(
        "--timeout",
        type=float,
        default=config.START_POSE_GATE_TIMEOUT_SEC,
        help="Maximum seconds to wait for an acceptable stable pose.",
    )
    args = parser.parse_args()

    ok = wait_for_start_pose(args.run_id, timeout_sec=args.timeout)
    return 0 if ok else 1


def wait_for_start_pose(run_id, timeout_sec):
    ref = start_pose.default_reference()
    stable_since = None
    started = time.time()
    last_status = ""

    print("Waiting for camera-confirmed start pose.")
    print("Keep vision_tracker/main.py running in another terminal.")
    print(
        "Reference: "
        f"x={ref.x:.3f}, y={ref.y:.3f}, yaw={ref.yaw_deg:.1f} deg; "
        f"tolerance={config.START_POSE_POSITION_TOLERANCE_M:.3f} m/"
        f"{config.START_POSE_YAW_TOLERANCE_DEG:.1f} deg"
    )
    print(
        f"Stable requirement: {config.START_POSE_STABLE_TIME_SEC:.1f} s; "
        f"timeout: {timeout_sec:.0f} s"
    )
    print(f"Reading latest pose from: {config.LATEST_TRACKER_POSE_FILE}")

    try:
        while True:
            now = time.time()
            elapsed = now - started
            if elapsed > timeout_sec:
                print("\nERROR: Timed out waiting for an acceptable start pose.")
                print("       Is vision_tracker/main.py running and detecting markers?")
                return False

            pose = start_pose.read_latest_pose(config.LATEST_TRACKER_POSE_FILE)
            if pose is None:
                stable_since = None
                status = "missing or unreadable latest pose"
            else:
                check = start_pose.check_start_pose(pose, ref=ref, now=now)
                if check["accepted"]:
                    if stable_since is None:
                        stable_since = now
                    stable_time = now - stable_since
                    status = _format_status(pose, check, stable_time)

                    if stable_time >= config.START_POSE_STABLE_TIME_SEC:
                        _print_status(status, force=True)
                        _log_accepted(run_id, ref, pose, check, stable_time)
                        print(
                            "\nStart pose accepted. "
                            "Press ENTER to start the run, or Ctrl+C to cancel."
                        )
                        input()
                        return True
                else:
                    stable_since = None
                    status = _format_status(pose, check, 0.0)

            if status != last_status:
                _print_status(status)
                last_status = status

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCancelled before run start.")
        return False


def _format_status(pose, check, stable_time):
    if not check["fresh"]:
        state = "STALE"
    elif not check["finite_pose"]:
        state = "INVALID"
    elif not check["valid_pose"]:
        state = "INVALID"
    elif not check["enough_markers"]:
        state = "MARKERS"
    elif check["accepted"]:
        state = "OK"
    else:
        state = "ADJUST"

    return (
        f"{state}: "
        f"pos_err={check['position_error_m']:.3f} m, "
        f"yaw_err={check['yaw_error_deg']:+.1f} deg, "
        f"markers={pose.num_detected}/{config.START_POSE_REQUIRED_MARKERS}, "
        f"age={check['pose_age_sec']:.2f} s, "
        f"stable={stable_time:.1f}/{config.START_POSE_STABLE_TIME_SEC:.1f} s"
    )


def _print_status(status, force=False):
    end = "\n" if force else "\r"
    print(status.ljust(120), end=end, flush=True)


def _log_accepted(run_id, ref, pose, check, stable_time):
    path = config.START_POSE_CHECKS_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_HEADER)

        writer.writerow([
            datetime.now().isoformat(),
            run_id,
            f"{ref.x:.6f}",
            f"{ref.y:.6f}",
            f"{ref.yaw_deg:.6f}",
            f"{pose.x:.6f}",
            f"{pose.y:.6f}",
            f"{pose.yaw_deg:.6f}",
            f"{check['dx']:.6f}",
            f"{check['dy']:.6f}",
            f"{check['position_error_m']:.6f}",
            f"{check['yaw_error_deg']:.6f}",
            f"{check['position_tolerance_m']:.6f}",
            f"{check['yaw_tolerance_deg']:.6f}",
            f"{check['pose_age_sec']:.6f}",
            f"{stable_time:.6f}",
            1,
        ])


if __name__ == "__main__":
    sys.exit(main())
