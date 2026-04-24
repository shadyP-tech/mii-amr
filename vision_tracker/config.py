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

# Camera
CAMERA_INDEX = 1  # cv2.VideoCapture source (int or device path)
RESIZE_SCALE = 0.7  # Frame downscale factor for processing speed

# Green color detection (HSV)
# Tune these under your actual lab lighting.
# H: 0-179,  S: 0-255,  V: 0-255
HSV_LOWER = np.array([35, 30, 100])
HSV_UPPER = np.array([85, 255, 255])

# Blob filtering
MIN_CONTOUR_AREA = 30  # Minimum contour area in pixels² (ignore noise)
MIN_RADIUS = 3  # Minimum enclosing circle radius in pixels
MAX_RADIUS = 30  # Maximum radius to prevent picking up huge green reflections
MIN_CIRCULARITY = 0.4  # 4πA/P²  —  1.0 is a perfect circle
MORPH_KERNEL_SIZE = 5  # Kernel size for morphological open/close

# Robot marker geometry
# Forward offset (meters) from the rear marker to the actual robot
# center along the heading direction.  Adjust to match your marker placement.
CENTER_FORWARD_OFFSET = 0.067  # 6.7 cm forward of rear marker

# Calibration — real-world reference rectangle
# Four corners of a known rectangle on the floor/table, in meters.
# Order: top-left, top-right, bottom-right, bottom-left  (when viewed from
# the camera's perspective — the click order in calibration.py must match).
WORLD_RECT_METERS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.7],
        [0.0, 0.7],
    ],
    dtype=np.float32,
)
