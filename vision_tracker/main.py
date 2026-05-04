"""
main.py — Full vision tracking pipeline with CSV logging.

Orchestrates:
    camera frame → detect markers → classify → estimate pose → log CSV

Usage:
    python3 main.py

Requires a saved homography (run ``python3 calibration.py`` first).

CSV output schema:
    timestamp, pose_x, pose_y, pose_yaw,
    front_left_x, front_left_y, front_right_x, front_right_y, rear_x, rear_y,
    num_detected, valid_pose

If fewer than 3 markers are detected, a row is written with
valid_pose=0 and pose/marker fields set to NaN.
"""

import cv2
import csv
import os
import sys
import time
import math
import numpy as np
from datetime import datetime

import config
import camera
import tracker
import calibration
import pose_estimator
import start_pose

CSV_HEADER = [
    "timestamp",
    "pose_x",
    "pose_y",
    "pose_yaw",
    "front_left_x",
    "front_left_y",
    "front_right_x",
    "front_right_y",
    "rear_x",
    "rear_y",
    "num_detected",
    "valid_pose",
]

NAN = float("nan")
POSE_ARROW_LENGTH_M = 0.06
POSE_CENTER_COLOR = (255, 0, 255)


def _make_csv_path():
    """Generate a timestamped CSV filename."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(config.DATA_DIR, f"vision_{ts}.csv")


def main():
    # load homography
    H = calibration.load_homography()
    if H is None:
        print("ERROR: No homography found.")
        print("       Run  python3 calibration.py  first.")
        sys.exit(1)

    # open camera
    try:
        cap = camera.open_camera()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    csv_path = _make_csv_path()
    print(f"main.py — logging to {csv_path}")
    print("Press ESC to stop.\n")

    csvfile = open(csv_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(CSV_HEADER)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame")
                break

            frame = cv2.resize(
                frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE
            )

            # detect markers
            centers, mask = tracker.detect_markers(frame)
            tracker.draw_markers(frame, centers)
            num_detected = len(centers)

            ts = time.time()

            if num_detected >= 3:
                # Convert pixel centers to world, keeping radius for
                # size-based classification.
                centers_world = []
                for px, py, r in centers:
                    w = calibration.pixel_to_world((px, py), H)
                    centers_world.append((w, r))

                classified = pose_estimator.classify_markers(centers_world)

                if classified is not None:
                    x, y, yaw = pose_estimator.estimate_pose(classified)
                    _draw_pose_center_overlay(frame, H, x, y, yaw)

                    # Write latest external tracker pose for real experiment scripts
                    pose_estimator.write_latest_pose(
                        x,
                        y,
                        yaw,
                        valid_pose=True,
                        num_detected=num_detected,
                    )

                    row = [
                        f"{ts:.4f}",
                        f"{x:.5f}",
                        f"{y:.5f}",
                        f"{yaw:.5f}",
                        f"{classified['front_left'][0]:.5f}",
                        f"{classified['front_left'][1]:.5f}",
                        f"{classified['front_right'][0]:.5f}",
                        f"{classified['front_right'][1]:.5f}",
                        f"{classified['rear'][0]:.5f}",
                        f"{classified['rear'][1]:.5f}",
                        num_detected,
                        1,
                    ]
                    writer.writerow(row)

                    tracker_pose = start_pose.TrackerPose(
                        timestamp="",
                        x=x,
                        y=y,
                        yaw_rad=yaw,
                        yaw_deg=math.degrees(yaw),
                        valid_pose=True,
                        num_detected=num_detected,
                        timestamp_epoch=ts,
                        file_mtime=ts,
                    )
                    check = start_pose.check_start_pose(tracker_pose, now=ts)
                    _draw_start_overlay(frame, tracker_pose, check)
                else:
                    # Classification failed (shouldn't happen with 3 markers)
                    _write_invalid_row(writer, ts, num_detected)
                    _write_invalid_latest_pose(num_detected)
                    _draw_invalid_start_overlay(frame, num_detected)
            else:
                _write_invalid_row(writer, ts, num_detected)
                _write_invalid_latest_pose(num_detected)
                _draw_invalid_start_overlay(frame, num_detected)

            cv2.imshow("tracking", frame)
            cv2.imshow("mask", mask)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        csvfile.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data saved to {csv_path}")


def _write_invalid_row(writer, ts, num_detected):
    """Write a CSV row for a frame where pose could not be estimated."""
    writer.writerow(
        [
            f"{ts:.4f}",
            NAN,
            NAN,
            NAN,
            NAN,
            NAN,
            NAN,
            NAN,
            NAN,
            NAN,
            num_detected,
            0,
        ]
    )


def _write_invalid_latest_pose(num_detected):
    pose_estimator.write_latest_pose(
        NAN,
        NAN,
        NAN,
        valid_pose=False,
        num_detected=num_detected,
    )


def _draw_start_overlay(frame, tracker_pose, check):
    status = "START OK" if check["accepted"] else "ADJUST START"
    color = (0, 180, 0) if check["accepted"] else (0, 0, 255)

    lines = [
        status,
        (
            f"x={tracker_pose.x:.3f}  y={tracker_pose.y:.3f}  "
            f"yaw={tracker_pose.yaw_deg:.1f} deg"
        ),
        (
            f"dx={check['dx']:+.3f}  dy={check['dy']:+.3f}  "
            f"pos_err={check['position_error_m']:.3f} m"
        ),
        (
            f"yaw_err={check['yaw_error_deg']:+.1f} deg  "
            f"markers={tracker_pose.num_detected}/"
            f"{config.START_POSE_REQUIRED_MARKERS}  valid=1"
        ),
    ]
    _draw_overlay_lines(frame, lines, color)


def _draw_invalid_start_overlay(frame, num_detected):
    lines = [
        "ADJUST START",
        "pose invalid",
        f"markers={num_detected}/{config.START_POSE_REQUIRED_MARKERS}  valid=0",
    ]
    _draw_overlay_lines(frame, lines, (0, 0, 255))


def _pose_center_overlay_points(frame_shape, H, x, y, yaw):
    center_world = np.array([x, y], dtype=float)
    tip_world = np.array(
        [
            x + POSE_ARROW_LENGTH_M * math.cos(yaw),
            y + POSE_ARROW_LENGTH_M * math.sin(yaw),
        ],
        dtype=float,
    )

    try:
        center_pixel = calibration.world_to_pixel(center_world, H)
        tip_pixel = calibration.world_to_pixel(tip_world, H)
    except (cv2.error, np.linalg.LinAlgError, ValueError, TypeError):
        return None

    if not (
        _is_reasonable_pixel(center_pixel, frame_shape)
        and _is_reasonable_pixel(tip_pixel, frame_shape)
    ):
        return None

    center = tuple(int(round(value)) for value in center_pixel)
    tip = tuple(int(round(value)) for value in tip_pixel)
    return center, tip


def _is_reasonable_pixel(pixel, frame_shape):
    if pixel is None or len(pixel) != 2:
        return False

    if not np.all(np.isfinite(pixel)):
        return False

    height, width = frame_shape[:2]
    margin = max(width, height)
    x, y = float(pixel[0]), float(pixel[1])
    return -margin <= x <= width + margin and -margin <= y <= height + margin


def _draw_pose_center_overlay(frame, H, x, y, yaw):
    points = _pose_center_overlay_points(frame.shape, H, x, y, yaw)
    if points is None:
        return

    center, tip = points
    cv2.circle(frame, center, 8, POSE_CENTER_COLOR, 2)
    cv2.drawMarker(
        frame,
        center,
        POSE_CENTER_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    cv2.arrowedLine(frame, center, tip, POSE_CENTER_COLOR, 2, tipLength=0.35)
    cv2.putText(
        frame,
        "center",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        POSE_CENTER_COLOR,
        2,
    )


def _draw_overlay_lines(frame, lines, color):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 25 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


if __name__ == "__main__":
    main()
