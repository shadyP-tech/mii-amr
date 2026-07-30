import unittest
from types import SimpleNamespace

from scripts.aufgabe04.perception.camera_calibration import (
    camera_calibration_from_info,
)


def camera_info(**overrides):
    values = {
        "width": 800,
        "height": 600,
        "k": [
            640.0,
            0.0,
            406.0,
            0.0,
            641.0,
            301.0,
            0.0,
            0.0,
            1.0,
        ],
        "d": [0.22, -0.58, -0.0014, -0.0008, 0.45],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [
            640.0,
            0.0,
            406.0,
            0.0,
            0.0,
            641.0,
            301.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        "header": SimpleNamespace(frame_id="camera"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CameraCalibrationTest(unittest.TestCase):
    def test_camera_info_is_copied_into_immutable_rectified_geometry(self):
        calibration = camera_calibration_from_info(camera_info())

        self.assertEqual(calibration.width_px, 800)
        self.assertEqual(calibration.height_px, 600)
        self.assertEqual(calibration.frame_id, "camera")
        self.assertEqual(calibration.fx_px, 640.0)
        self.assertEqual(calibration.fy_px, 641.0)
        self.assertEqual(calibration.cx_px, 406.0)
        self.assertEqual(calibration.cy_px, 301.0)
        self.assertIsInstance(calibration.distortion, tuple)

    def test_zero_focal_length_fails_closed(self):
        invalid_projection = list(camera_info().p)
        invalid_projection[0] = 0.0

        with self.assertRaisesRegex(ValueError, "P focal lengths"):
            camera_calibration_from_info(camera_info(p=invalid_projection))

    def test_zero_filled_camera_matrix_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "K focal lengths"):
            camera_calibration_from_info(camera_info(k=[0.0] * 9))

    def test_incomplete_camera_matrix_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "K must contain 9"):
            camera_calibration_from_info(camera_info(k=[1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
