"""
config.py — Central configuration for vision-based TurtleBot tracking.

All tunable parameters live here so they can be adjusted in one place
without editing detection, calibration, or pose estimation code.
"""

import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
HOMOGRAPHY_FILE = os.path.join(DATA_DIR, "homography.npz")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LATEST_TRACKER_POSE_FILE = os.path.join(RESULTS_DIR, "latest_tracker_pose.csv")
START_POSE_CHECKS_FILE = os.path.join(RESULTS_DIR, "real_start_pose_checks.csv")

# Camera
CAMERA_INDEX = 0  # D435i grayscale/IR endpoint via OpenCV/AVFoundation
RESIZE_SCALE = 0.7  # Frame downscale factor for processing speed

# Camera stream negotiation.  On macOS, some USB cameras can expose both color
# and monochrome/IR modes; retrying with explicit settings helps avoid opening
# the wrong stream after reconnects.
CAMERA_BACKEND = "opencv"  # "auto", "realsense", or "opencv"
CAMERA_FORCE_AVFOUNDATION = True
CAMERA_FOURCC = "MJPG"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_FPS = 30

# D435i RGB can output 1920x1080@30, while its depth/IR stream is commonly
# 1280x720.  Briefly requesting this mode can nudge AVFoundation toward the RGB
# endpoint, then the camera helper switches back to the configured mode above.
CAMERA_USE_RGB_WAKEUP_MODE = True
CAMERA_RGB_WAKEUP_FRAME_WIDTH = 1920
CAMERA_RGB_WAKEUP_FRAME_HEIGHT = 1080
CAMERA_RGB_WAKEUP_FPS = 30

CAMERA_OPEN_RETRIES = 5
CAMERA_RETRY_DELAY_SEC = 0.5
CAMERA_WARMUP_FRAMES = 15
CAMERA_REQUIRE_COLOR = True
CAMERA_MIN_MEAN_SATURATION = 5.0
CAMERA_MIN_CHANNEL_DIFF = 1.0

# Intel RealSense D435i.  If pyrealsense2 is installed, CAMERA_BACKEND="auto"
# tries the hardware color stream before falling back to OpenCV/AVFoundation.
# If pyrealsense2 segfaults on macOS, keep CAMERA_BACKEND="opencv".
REALSENSE_SERIAL = None
REALSENSE_ENABLE_AUTO_EXPOSURE = True
REALSENSE_EXPOSURE = None
REALSENSE_GAIN = None

# Green color detection (HSV)
# Tune these under your actual lab lighting.
# H: 0-179,  S: 0-255,  V: 0-255
HSV_LOWER = np.array([60, 62, 160])
HSV_UPPER = np.array([91, 255, 255])

MIN_CONTOUR_AREA = 50
MIN_RADIUS = 5
MAX_RADIUS = 120
MIN_CIRCULARITY = 0.35
MIN_FILL_RATIO = 0.35
MORPH_KERNEL_SIZE = 7

# Exposure tuning
AUTO_SELECT_EXPOSURE = False

# More negative is usually darker for many webcams.
# Some macOS camera backends may ignore this.
EXPOSURE_CANDIDATES = [-11, -10, -9, -8, -7, -6, -5, -4, -3]

# Used to reject frames with too much clipping
MAX_CLIPPED_FRACTION = 0.005

# Robot marker geometry
# The three markers form a right-angle layout:
# - rear/small marker to the nearest large front marker: 78 mm
# - distance between the two large front markers: 114 mm
# The two large front markers are approximately on the wheel axis, so the
# tracked robot center is approximated as their midpoint.
MARKER_FORWARD_SPACING_M = 0.078
MARKER_LATERAL_SPACING_M = 0.114
CENTER_FORWARD_OFFSET = MARKER_FORWARD_SPACING_M
CENTER_LATERAL_OFFSET = MARKER_LATERAL_SPACING_M / 2.0

# Real-run start pose gate
# Seeded from the current clean real runs; keep fixed during an experiment batch.
START_POSE_REF_X = 0.0
START_POSE_REF_Y = 0.0
START_POSE_REF_YAW_DEG = 0.0
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
        [4.143, -0.5],
        [3.224, -0.5],
        [3.222, 0.0],
        [4.144, 0.0],
    ],
    dtype=np.float32,
)

DEBUG_CONTOURS = False
