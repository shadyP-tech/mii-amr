import math
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    FaceCandidate,
    MaterialTarget,
    SideEvidence,
    StandGeometry,
    SynchronizedViewpointRecommendation,
)
from scripts.aufgabe04.perception.arrival_pose_estimator import (
    arrival_pose_record_from_recommendation,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    validate_arrival_pose_record,
)


def recommendation(axis_state="target_committed", side_valid=False):
    stand = Pose2D(1.0, 2.0)
    selected = Pose2D(1.0, 2.32, -math.pi / 2.0)
    return SynchronizedViewpointRecommendation(
        schema_version=1,
        simulation_only=True,
        stream_id="sim-survey",
        stand_id="stand-a",
        planning_frame="odom",
        source="synchronized_lidar_camera_viewpoint",
        observation_unix_sec=100.0,
        sensor_stamp_sec=42.0,
        stand=StandGeometry(stand, 0.06, 0.02, "lidar_cluster"),
        robot_pose=Pose2D(1.0, 3.0),
        axis_confidence=0.91,
        axis_state=axis_state,
        face_candidates=(
            FaceCandidate("face_a", math.pi / 2.0, selected, True),
            FaceCandidate(
                "face_b",
                -math.pi / 2.0,
                Pose2D(1.0, 1.68, math.pi / 2.0),
                True,
            ),
        ),
        side_evidence=SideEvidence(
            "qr" if side_valid else "none",
            0.98 if side_valid else 0.0,
            side_valid,
            side_valid,
            "face_a" if side_valid else None,
            "sim_qr" if side_valid else "axis_only",
        ),
        material_target=MaterialTarget("face_a", selected, "robot_facing_axis"),
    )


class ArrivalPoseEstimatorTest(unittest.TestCase):
    def test_committed_axis_becomes_valid_explicit_record(self):
        record = arrival_pose_record_from_recommendation(
            recommendation(),
            candidate_uid="candidate-a",
            map_yaml_sha256="a" * 64,
            corridor_length_m=0.40,
            validated_unix_sec=101.0,
        )

        validate_arrival_pose_record(record)
        self.assertAlmostEqual(record.arrival_pose.y_m, 2.32)
        self.assertAlmostEqual(record.corridor_entry_pose.y_m, 2.72)
        self.assertEqual(record.face.evidence_kind, "robot_facing_axis")
        self.assertTrue(record.face.evidence_valid)

    def test_hard_qr_evidence_is_preserved(self):
        record = arrival_pose_record_from_recommendation(
            recommendation(side_valid=True),
            candidate_uid="candidate-a",
            map_yaml_sha256="a" * 64,
            corridor_length_m=0.40,
            validated_unix_sec=101.0,
        )

        self.assertEqual(record.face.evidence_kind, "qr")
        self.assertTrue(record.face.evidence_hard)

    def test_uncommitted_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "committed"):
            arrival_pose_record_from_recommendation(
                recommendation(axis_state="viewpoint_sampling"),
                candidate_uid="candidate-a",
                map_yaml_sha256="a" * 64,
                corridor_length_m=0.40,
                validated_unix_sec=101.0,
            )


if __name__ == "__main__":
    unittest.main()
