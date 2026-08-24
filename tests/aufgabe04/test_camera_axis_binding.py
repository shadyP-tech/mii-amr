import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.camera_axis_binding import (
    load_opposite_face_normal,
    opposite_face_normal_from_axis_observation,
)


def _observation(*, robot_y_m: float) -> dict[str, object]:
    return {
        "observation_kind": "real_stand_axis_without_qr",
        "stand_axis_rad": 0.0,
        "stand_center": {"x_m": 0.5, "y_m": 0.0},
        "robot_pose": {"x_m": 0.5, "y_m": robot_y_m},
    }


class CameraAxisBindingTest(unittest.TestCase):
    def test_opposite_face_flips_with_observing_side(self):
        above = opposite_face_normal_from_axis_observation(
            _observation(robot_y_m=0.7)
        )
        below = opposite_face_normal_from_axis_observation(
            _observation(robot_y_m=-0.7)
        )

        self.assertAlmostEqual(above, -math.pi / 2.0)
        self.assertAlmostEqual(below, math.pi / 2.0)

    def test_invalid_or_ambiguous_observation_fails_closed(self):
        wrong_kind = _observation(robot_y_m=0.7)
        wrong_kind["observation_kind"] = "other"
        coincident = _observation(robot_y_m=0.0)

        with self.assertRaisesRegex(ValueError, "unexpected.*kind"):
            opposite_face_normal_from_axis_observation(wrong_kind)
        with self.assertRaisesRegex(ValueError, "coincides"):
            opposite_face_normal_from_axis_observation(coincident)

    def test_loader_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text("[]\n")

            with self.assertRaisesRegex(ValueError, "root must be an object"):
                load_opposite_face_normal(path)


if __name__ == "__main__":
    unittest.main()
