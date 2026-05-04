import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision_tracker"))

try:
    import numpy as np  # noqa: E402
    import config  # noqa: E402
    import pose_estimator  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    raise unittest.SkipTest("numpy is required for pose estimator tests")


class PoseEstimatorCenterCalibrationTest(unittest.TestCase):
    def test_center_uses_marker_rectangle_midpoint_when_heading_along_x(self):
        rear = np.array([0.0, 0.0])
        straight_front = np.array([config.MARKER_FORWARD_SPACING_M, 0.0])
        diagonal_front = np.array([
            config.MARKER_FORWARD_SPACING_M,
            config.MARKER_LATERAL_SPACING_M,
        ])

        classified = pose_estimator.classify_markers([
            (straight_front, 50.0),
            (diagonal_front, 49.0),
            (rear, 20.0),
        ])
        x, y, yaw = pose_estimator.estimate_pose(classified)

        self.assertAlmostEqual(x, 0.039)
        self.assertAlmostEqual(y, 0.057)
        self.assertAlmostEqual(yaw, 0.0)

    def test_center_uses_marker_rectangle_midpoint_when_heading_along_y(self):
        rear = np.array([1.0, 2.0])
        straight_front = np.array([1.0, 2.0 + config.MARKER_FORWARD_SPACING_M])
        diagonal_front = np.array([
            1.0 - config.MARKER_LATERAL_SPACING_M,
            2.0 + config.MARKER_FORWARD_SPACING_M,
        ])

        classified = pose_estimator.classify_markers([
            (diagonal_front, 50.0),
            (straight_front, 49.0),
            (rear, 20.0),
        ])
        x, y, yaw = pose_estimator.estimate_pose(classified)

        self.assertAlmostEqual(x, 1.0 - 0.057)
        self.assertAlmostEqual(y, 2.0 + 0.039)
        self.assertAlmostEqual(yaw, math.pi / 2)


if __name__ == "__main__":
    unittest.main()
