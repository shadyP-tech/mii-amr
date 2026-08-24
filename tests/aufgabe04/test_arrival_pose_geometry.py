import math
import unittest
from dataclasses import replace

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.arrival_pose_geometry import (
    ArrivalGeometryConfig,
    angular_distance_rad,
    arrival_face_candidates,
    canonical_axial_angle,
    face_normal_rad,
    observer_facing_arrival_face,
    validate_arrival_face_geometry,
)


class CanonicalAxisTest(unittest.TestCase):
    def test_axis_is_modulo_pi_in_half_open_interval(self):
        reference = canonical_axial_angle(0.23)
        for turns in range(-4, 5):
            self.assertAlmostEqual(
                canonical_axial_angle(0.23 + turns * math.pi), reference
            )
        self.assertGreaterEqual(reference, -math.pi / 2.0)
        self.assertLess(reference, math.pi / 2.0)

    def test_equivalent_boundary_representations_are_identical(self):
        self.assertEqual(
            canonical_axial_angle(math.pi / 2.0),
            canonical_axial_angle(-math.pi / 2.0),
        )

    def test_equivalent_axis_does_not_swap_face_ids(self):
        stand = Pose2D(0.7, -0.2)
        first = arrival_face_candidates(stand, 0.31)
        flipped = arrival_face_candidates(stand, 0.31 + math.pi)
        for expected, actual in zip(first, flipped):
            self.assertEqual(actual.face_id, expected.face_id)
            self.assertAlmostEqual(actual.outward_normal_rad, expected.outward_normal_rad)
            self.assertAlmostEqual(actual.target_pose.x_m, expected.target_pose.x_m)
            self.assertAlmostEqual(actual.target_pose.y_m, expected.target_pose.y_m)


class ArrivalFaceGeometryTest(unittest.TestCase):
    def setUp(self):
        self.stand = Pose2D(1.0, -0.5, 0.0)
        self.config = ArrivalGeometryConfig(
            standoff_distance_m=0.32,
            terminal_corridor_length_m=0.40,
        )

    def test_observer_facing_selection_uses_visible_side(self):
        north = observer_facing_arrival_face(
            self.stand, 0.0, Pose2D(1.0, 2.0), self.config
        )
        south = observer_facing_arrival_face(
            self.stand, 0.0, Pose2D(1.0, -2.0), self.config
        )
        self.assertEqual(north.face_id, 0)
        self.assertEqual(south.face_id, 1)

    def test_target_and_corridor_have_exact_requested_distances(self):
        for face in arrival_face_candidates(self.stand, 0.41, self.config):
            self.assertIs(type(face.target_pose), type(self.stand))
            self.assertIs(type(face.corridor_entry_pose), type(self.stand))
            target_distance = math.hypot(
                face.target_pose.x_m - self.stand.x_m,
                face.target_pose.y_m - self.stand.y_m,
            )
            entry_distance = math.hypot(
                face.corridor_entry_pose.x_m - self.stand.x_m,
                face.corridor_entry_pose.y_m - self.stand.y_m,
            )
            self.assertAlmostEqual(target_distance, 0.32)
            self.assertAlmostEqual(entry_distance, 0.72)
            self.assertAlmostEqual(
                math.hypot(
                    face.corridor_entry_pose.x_m - face.target_pose.x_m,
                    face.corridor_entry_pose.y_m - face.target_pose.y_m,
                ),
                0.40,
            )

    def test_target_yaw_faces_stand_center(self):
        for face in arrival_face_candidates(self.stand, -0.38, self.config):
            bearing_to_center = math.atan2(
                self.stand.y_m - face.target_pose.y_m,
                self.stand.x_m - face.target_pose.x_m,
            )
            self.assertAlmostEqual(
                angular_distance_rad(face.target_pose.yaw_rad, bearing_to_center), 0.0
            )

    def test_arrival_radius_is_perpendicular_to_stand_axis(self):
        axis = 0.57
        axis_unit = (math.cos(axis), math.sin(axis))
        for face in arrival_face_candidates(self.stand, axis, self.config):
            radial = (
                (face.target_pose.x_m - self.stand.x_m) / 0.32,
                (face.target_pose.y_m - self.stand.y_m) / 0.32,
            )
            self.assertAlmostEqual(
                radial[0] * axis_unit[0] + radial[1] * axis_unit[1], 0.0
            )
            self.assertTrue(
                validate_arrival_face_geometry(
                    self.stand, axis, face, self.config
                ).valid
            )

    def test_validation_reports_standoff_and_yaw_corruption(self):
        face = arrival_face_candidates(self.stand, 0.0, self.config)[0]
        corrupted = replace(
            face,
            target_pose=Pose2D(
                face.target_pose.x_m,
                face.target_pose.y_m + 0.08,
                face.target_pose.yaw_rad + 0.2,
            ),
        )
        result = validate_arrival_face_geometry(
            self.stand, 0.0, corrupted, self.config
        )
        self.assertFalse(result.valid)
        self.assertIn("arrival_standoff_mismatch", result.violations)
        self.assertIn("arrival_yaw_not_facing_stand", result.violations)

    def test_validation_rejects_correct_line_on_wrong_face(self):
        first, second = arrival_face_candidates(self.stand, 0.0, self.config)
        mislabeled = replace(
            first,
            target_pose=second.target_pose,
            corridor_entry_pose=second.corridor_entry_pose,
        )
        result = validate_arrival_face_geometry(
            self.stand, 0.0, mislabeled, self.config
        )
        self.assertFalse(result.valid)
        self.assertIn("arrival_on_wrong_face", result.violations)


class InvalidGeometryInputTest(unittest.TestCase):
    def test_invalid_angles_distances_face_ids_and_poses_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            canonical_axial_angle(float("nan"))
        with self.assertRaisesRegex(ValueError, "positive"):
            ArrivalGeometryConfig(standoff_distance_m=0.0)
        with self.assertRaisesRegex(ValueError, "face_id"):
            face_normal_rad(0.0, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            arrival_face_candidates(Pose2D(float("inf"), 0.0), 0.0)
        with self.assertRaisesRegex(ValueError, "coincide"):
            observer_facing_arrival_face(
                Pose2D(0.0, 0.0), 0.0, Pose2D(0.0, 0.0)
            )

    def test_invalid_validation_tolerances_are_rejected(self):
        stand = Pose2D(0.0, 0.0)
        face = arrival_face_candidates(stand, 0.0)[0]
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_arrival_face_geometry(
                stand, 0.0, face, position_tolerance_m=-1.0
            )


if __name__ == "__main__":
    unittest.main()
