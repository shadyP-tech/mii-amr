import numpy as np
import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
HOMOGRAPHY_FILE = os.path.join(DATA_DIR, "homography.npz")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LATEST_TRACKER_POSE_FILE = os.path.join(RESULTS_DIR, "latest_tracker_pose.csv")
START_POSE_CHECKS_FILE = os.path.join(RESULTS_DIR, "real_start_pose_checks.csv")

# Camera
CAMERA_INDEX = 0  
RESIZE_SCALE = 0.7  

CAMERA_FORCE_AVFOUNDATION = True
CAMERA_FOURCC = "MJPG"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_FPS = 30

CAMERA_WARMUP_FRAMES = 15

# green color detection (HSV)
HSV_LOWER = np.array([60, 62, 160])
HSV_UPPER = np.array([91, 255, 255])

MIN_CONTOUR_AREA = 50
MIN_RADIUS = 5
MAX_RADIUS = 120
MIN_CIRCULARITY = 0.35
MIN_FILL_RATIO = 0.35
MORPH_KERNEL_SIZE = 7

MARKER_FORWARD_SPACING_M = 0.078
MARKER_LATERAL_SPACING_M = 0.114
CENTER_FORWARD_OFFSET = MARKER_FORWARD_SPACING_M
CENTER_LATERAL_OFFSET = MARKER_LATERAL_SPACING_M / 2.0

# real-run start pose gate
START_POSE_REF_X = 0.0
START_POSE_REF_Y = 0.0
START_POSE_REF_YAW_DEG = 0.0
START_POSE_POSITION_TOLERANCE_M = 0.04
START_POSE_YAW_TOLERANCE_DEG = 4.0
START_POSE_STABLE_TIME_SEC = 1.0
START_POSE_MAX_AGE_SEC = 1.0
START_POSE_GATE_TIMEOUT_SEC = 60.0
START_POSE_REQUIRED_MARKERS = 3

# calibration, real-world reference rectangle
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
