"""
config.py — Central configuration for vision-based TurtleBot tracking.

All tunable parameters live here so they can be adjusted in one place
without editing detection, calibration, or pose estimation code.
"""

import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HOMOGRAPHY_FILE = os.path.join(DATA_DIR, "homography.npz")
LATEST_TRACKER_POSE_FILE = os.path.join("results", "latest_tracker_pose.csv")
START_POSE_CHECKS_FILE = os.path.join("results", "real_start_pose_checks.csv")

# Camera
CAMERA_INDEX = 1  # cv2.VideoCapture source (int or device path)
RESIZE_SCALE = 0.7  # Frame downscale factor for processing speed

# Green color detection (HSV)
# Tune these under your actual lab lighting.
# H: 0-179,  S: 0-255,  V: 0-255
HSV_LOWER = np.array([0, 0, 240])
HSV_UPPER = np.array([95, 255, 255])

MIN_CONTOUR_AREA = 100
MIN_RADIUS = 8
MAX_RADIUS = 120
MIN_CIRCULARITY = 0.55
MORPH_KERNEL_SIZE = 9

# Exposure tuning
AUTO_SELECT_EXPOSURE = False

# More negative is usually darker for many webcams.
# Some macOS camera backends may ignore this.
EXPOSURE_CANDIDATES = [-11, -10, -9, -8, -7, -6, -5, -4, -3]

# Used to reject frames with too much clipping
MAX_CLIPPED_FRACTION = 0.005

# Robot marker geometry
# Forward offset (meters) from the rear marker to the actual robot
# center along the heading direction.  Adjust to match your marker placement.
CENTER_FORWARD_OFFSET = 0.067  # 6.7 cm forward of rear marker

# Real-run start pose gate
# Seeded from the current clean real runs; keep fixed during an experiment batch.
START_POSE_REF_X = 0.786072
START_POSE_REF_Y = 0.624328
START_POSE_REF_YAW_DEG = -91.498214
START_POSE_POSITION_TOLERANCE_M = 0.04
START_POSE_YAW_TOLERANCE_DEG = 4.0
START_POSE_STABLE_TIME_SEC = 1.0
START_POSE_MAX_AGE_SEC = 1.0
START_POSE_GATE_TIMEOUT_SEC = 60.0
START_POSE_REQUIRED_MARKERS = 3

# Calibration — real-world reference rectangle
# Four corners of a known rectangle on the floor/table, in meters.
# Order: top-left, top-right, bottom-right, bottom-left  (when viewed from
# the camera's perspective — the click order in calibration.py must match).
WORLD_RECT_METERS = np.array(
    [
        [0.0, 0.0],
        [0.34, 0.0],
        [0.34, 0.2],
        [0.0, 0.2],
    ],
    dtype=np.float32,
)

DEBUG_CONTOURS = False
