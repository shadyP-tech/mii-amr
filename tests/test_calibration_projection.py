import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision_tracker"))

try:
    import numpy as np  # noqa: E402
    import calibration  # noqa: E402
    import main as tracker_main  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2", "numpy"}:
        raise
    raise unittest.SkipTest("OpenCV and numpy are required for projection tests")


class CalibrationProjectionTest(unittest.TestCase):
    def test_world_to_pixel_inverts_pixel_to_world(self):
        H = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        pixel = np.array([120.0, 240.0])
        world = calibration.pixel_to_world(pixel, H)
        projected = calibration.world_to_pixel(world, H)

        np.testing.assert_allclose(projected, pixel, atol=1e-5)

    def test_pose_center_overlay_points_use_world_heading(self):
        H = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        points = tracker_main._pose_center_overlay_points(
            frame_shape=(600, 800, 3),
            H=H,
            x=1.0,
            y=2.0,
            yaw=0.0,
        )

        self.assertEqual(points, ((100, 200), (106, 200)))

    def test_pose_center_overlay_points_skip_singular_homography(self):
        H = np.zeros((3, 3), dtype=float)

        points = tracker_main._pose_center_overlay_points(
            frame_shape=(600, 800, 3),
            H=H,
            x=1.0,
            y=2.0,
            yaw=0.0,
        )

        self.assertIsNone(points)


if __name__ == "__main__":
    unittest.main()
