"""
main.py — Full vision tracking pipeline with CSV logging.

Orchestrates:
    camera frame → detect markers → classify → estimate pose → log CSV

Usage:
    python3 main.py

Requires a saved homography (run ``python3 calibration.py`` first).

CSV output schema:
    timestamp, pose_x, pose_y, pose_yaw,
    left_x, left_y, right_x, right_y, front_x, front_y,
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
import tracker
import calibration
import pose_estimator

CSV_HEADER = [
    "timestamp",
    "pose_x",
    "pose_y",
    "pose_yaw",
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "front_x",
    "front_y",
    "num_detected",
    "valid_pose",
]

NAN = float("nan")


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
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {config.CAMERA_INDEX}")
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

                    row = [
                        f"{ts:.4f}",
                        f"{x:.5f}",
                        f"{y:.5f}",
                        f"{yaw:.5f}",
                        f"{classified['left'][0]:.5f}",
                        f"{classified['left'][1]:.5f}",
                        f"{classified['right'][0]:.5f}",
                        f"{classified['right'][1]:.5f}",
                        f"{classified['front'][0]:.5f}",
                        f"{classified['front'][1]:.5f}",
                        num_detected,
                        1,
                    ]
                    writer.writerow(row)

                    # On-screen overlay
                    info = f"x={x:.3f}  y={y:.3f}  yaw={math.degrees(yaw):.1f} deg"
                    cv2.putText(
                        frame,
                        info,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # Classification failed (shouldn't happen with 3 markers)
                    _write_invalid_row(writer, ts, num_detected)
            else:
                _write_invalid_row(writer, ts, num_detected)
                cv2.putText(
                    frame,
                    f"MARKERS: {num_detected}/3",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

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


if __name__ == "__main__":
    main()
