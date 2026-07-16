import math
import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.perception.camera_stand_observation import (
    CameraStandObservation,
    load_camera_observation,
    stand_axis_from_camera_yaw,
    validate_camera_observation,
    write_camera_observation,
)


def observation(**changes):
    values = dict(
        schema_version=1, observed_at_sec=100.0, image_topic="/camera/image_raw",
        camera_frame="camera_link", map_frame="odom", robot_x_m=0.0,
        robot_y_m=0.0, stand_x_m=1.0, stand_y_m=0.0, stand_axis_rad=1.57,
        axis_confidence=0.9, side="qr_code_side", side_confidence=0.95,
        qr_texts=("A",),
    )
    values.update(changes)
    return CameraStandObservation(**values)


class CameraStandObservationTest(unittest.TestCase):
    def test_explicit_camera_heading_is_invariant_to_off_center_target(self):
        axis = stand_axis_from_camera_yaw(
            robot_x_m=math.cos(math.radians(60.0)),
            robot_y_m=math.sin(math.radians(60.0)),
            stand_x_m=0.0,
            stand_y_m=0.0,
            camera_yaw_rad=math.radians(-1.586),
            camera_heading_rad=math.radians(-132.0),
        )

        axial_error = 0.5 * abs(
            math.atan2(
                math.sin(2.0 * (axis - math.radians(136.414))),
                math.cos(2.0 * (axis - math.radians(136.414))),
            )
        )
        self.assertAlmostEqual(math.degrees(axial_error), 0.0, places=6)

    def test_camera_face_normal_is_converted_to_map_stand_axis(self):
        axis = stand_axis_from_camera_yaw(
            robot_x_m=0.0,
            robot_y_m=0.0,
            stand_x_m=1.0,
            stand_y_m=0.0,
            camera_yaw_rad=0.0,
        )
        self.assertAlmostEqual(axis, -math.pi / 2.0)

    def test_positive_left_camera_yaw_rotates_map_axis_counterclockwise(self):
        axis = stand_axis_from_camera_yaw(
            robot_x_m=0.0,
            robot_y_m=0.0,
            stand_x_m=1.0,
            stand_y_m=0.0,
            camera_yaw_rad=math.radians(30.0),
        )

        self.assertAlmostEqual(axis, math.radians(-60.0), places=7)

    def test_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observation.json"
            write_camera_observation(path, observation())
            loaded = load_camera_observation(path)
        validate_camera_observation(loaded, required_map_frame="odom")
        self.assertEqual(loaded.qr_texts, ("A",))

    def test_fails_closed_on_confidence_frame_and_age(self):
        with self.assertRaisesRegex(ValueError, "axis confidence"):
            validate_camera_observation(
                observation(axis_confidence=0.2), required_map_frame="odom"
            )
        with self.assertRaisesRegex(ValueError, "map_frame"):
            validate_camera_observation(observation(), required_map_frame="map")
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_camera_observation(
                observation(), required_map_frame="odom", max_age_sec=2.0, now_sec=103.0
            )


if __name__ == "__main__":
    unittest.main()
