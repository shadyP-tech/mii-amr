import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (  # noqa: E402
    FaceCandidate,
    MaterialTarget,
    QrBindingObservation,
    QrFaceLatch,
    SideEvidence,
    StableFaceResolver,
    StandGeometry,
    SynchronizedViewpointRecommendation,
    angular_distance,
    load_recommendation,
    load_viewpoint_recommendation,
    recommendation_to_dict,
    recommendation_to_payload,
    validate_recommendation,
    validate_recommendation_freshness,
)


def recommendation() -> SynchronizedViewpointRecommendation:
    first_pose = Pose2D(0.3, 0.0, math.pi)
    return SynchronizedViewpointRecommendation(
        schema_version=1,
        simulation_only=True,
        stream_id="gazebo_run_001",
        stand_id="stand_01",
        planning_frame="odom",
        source="sim_synchronized_viewpoint",
        observation_unix_sec=100.0,
        sensor_stamp_sec=12.5,
        stand=StandGeometry(Pose2D(0.0, 0.0), 0.06, 0.02, "lidar_cluster"),
        robot_pose=Pose2D(1.0, 0.0, math.pi),
        axis_confidence=0.91,
        axis_state="resolved",
        face_candidates=(
            FaceCandidate("face_a", 0.0, first_pose, True),
            FaceCandidate("face_b", math.pi, Pose2D(-0.3, 0.0, 0.0), True),
        ),
        side_evidence=SideEvidence(
            "qr_registry", 0.98, True, True, "face_a", "sim_qr_consensus"
        ),
        material_target=MaterialTarget("face_a", first_pose, "hard_qr"),
    )


class ViewpointRecommendationModelTest(unittest.TestCase):
    def test_round_trip_path_and_mapping(self):
        payload = recommendation_to_dict(recommendation())
        loaded_mapping = load_viewpoint_recommendation(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recommendation.json"
            import json

            path.write_text(json.dumps(payload))
            loaded_path = load_viewpoint_recommendation(
                path,
                required_planning_frame="odom",
                required_source="sim_synchronized_viewpoint",
            )
        self.assertEqual(loaded_mapping, loaded_path)

    def test_convenience_loader_can_apply_frame_and_freshness(self):
        loaded = load_recommendation(
            recommendation_to_dict(recommendation()),
            expected_frame="odom",
            expected_source="sim_synchronized_viewpoint",
            now_unix_sec=101.0,
            max_age_sec=2.0,
        )
        self.assertEqual(loaded.stand_id, "stand_01")
        with self.assertRaisesRegex(ValueError, "provided together"):
            load_recommendation(
                recommendation_to_dict(recommendation()), max_age_sec=2.0
            )
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            load_recommendation(
                recommendation_to_dict(recommendation()),
                expected_source="unexpected_source",
            )
        with self.assertRaisesRegex(ValueError, "relative ROS frame"):
            load_recommendation(
                recommendation_to_dict(recommendation()),
                expected_frame="/odom",
            )

    def test_frame_finite_and_safe_identifier_validation(self):
        valid = recommendation()
        for changed, message in (
            ({"stream_id": "../../unsafe"}, "safe identifier"),
            ({"planning_frame": "/odom"}, "relative ROS frame"),
            ({"axis_confidence": float("nan")}, "finite"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, message):
                    validate_recommendation(
                        SynchronizedViewpointRecommendation(**{**valid.__dict__, **changed})
                    )

    def test_two_faces_must_be_distinct_and_target_must_reference_one(self):
        payload = recommendation_to_payload(recommendation())
        payload["face_candidates"][1]["face_id"] = "face_a"
        with self.assertRaisesRegex(ValueError, "distinct"):
            load_viewpoint_recommendation(payload)

        payload = recommendation_to_payload(recommendation())
        payload["material_target"]["face_id"] = "face_missing"
        with self.assertRaisesRegex(ValueError, "does not reference"):
            load_viewpoint_recommendation(payload)

    def test_face_geometry_must_be_antipodal_and_match_declared_pose(self):
        payload = recommendation_to_payload(recommendation())
        payload["face_candidates"][1]["outward_normal_rad"] = 0.1
        with self.assertRaisesRegex(ValueError, "antipodal"):
            load_viewpoint_recommendation(payload)

        payload = recommendation_to_payload(recommendation())
        payload["face_candidates"][0]["pose"]["y_m"] = 0.1
        with self.assertRaisesRegex(ValueError, "outward-normal ray"):
            load_viewpoint_recommendation(payload)

        payload = recommendation_to_payload(recommendation())
        payload["face_candidates"][0]["pose"]["yaw_rad"] = 0.0
        payload["material_target"]["pose"]["yaw_rad"] = 0.0
        with self.assertRaisesRegex(ValueError, "yaw must face the stand"):
            load_viewpoint_recommendation(payload)

    def test_malformed_boolean_is_not_coerced(self):
        payload = recommendation_to_payload(recommendation())
        payload["simulation_only"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_viewpoint_recommendation(payload)

    def test_hard_evidence_must_match_a_resolved_material_target_face(self):
        payload = recommendation_to_payload(recommendation())
        payload["material_target"] = {
            "face_id": "face_b",
            "pose": payload["face_candidates"][1]["pose"],
            "evidence_state": "hard_qr",
        }
        with self.assertRaisesRegex(ValueError, "same face"):
            load_viewpoint_recommendation(payload)

        payload = recommendation_to_payload(recommendation())
        payload["face_candidates"][0]["identity_resolved"] = False
        with self.assertRaisesRegex(ValueError, "resolved physical face"):
            load_viewpoint_recommendation(payload)

    def test_freshness_is_separate(self):
        payload = recommendation_to_payload(recommendation())
        loaded = load_viewpoint_recommendation(payload)
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_recommendation_freshness(
                loaded, now_unix_sec=111.0, max_age_sec=10.0
            )


class StableFaceResolverTest(unittest.TestCase):
    def test_provisional_then_resolved_without_unordered_face_swap(self):
        resolver = StableFaceResolver()
        first = resolver.update(stream_id="run_1", outward_normals_rad=(3.13, -0.01))
        self.assertFalse(first.identity_resolved)
        first_by_id = {face.face_id: face.outward_normal_rad for face in first.faces}

        second = resolver.update(
            stream_id="run_1", outward_normals_rad=(0.02, -3.12)
        )
        self.assertTrue(second.identity_resolved)
        second_by_id = {face.face_id: face.outward_normal_rad for face in second.faces}
        for face_id in first_by_id:
            wrapped_delta = math.atan2(
                math.sin(second_by_id[face_id] - first_by_id[face_id]),
                math.cos(second_by_id[face_id] - first_by_id[face_id]),
            )
            self.assertLess(abs(wrapped_delta), 0.05)

    def test_new_stream_returns_to_provisional_identity(self):
        resolver = StableFaceResolver()
        resolver.update(stream_id="run_1", outward_normals_rad=(0.0, math.pi))
        self.assertTrue(
            resolver.update(stream_id="run_1", outward_normals_rad=(0.01, -3.13)).identity_resolved
        )
        reset = resolver.update(stream_id="run_2", outward_normals_rad=(0.0, math.pi))
        self.assertFalse(reset.identity_resolved)

    def test_repeated_axial_outliers_cannot_poison_physical_face_ids(self):
        resolver = StableFaceResolver()
        resolver.update(
            stream_id="run_1",
            outward_normals_rad=(-math.pi / 2.0, math.pi / 2.0),
        )
        trusted = resolver.update(
            stream_id="run_1",
            outward_normals_rad=(math.radians(-89.0), math.radians(91.0)),
        )
        self.assertTrue(trusted.identity_resolved)
        trusted_by_id = {
            face.face_id: face.outward_normal_rad for face in trusted.faces
        }

        for _ in range(2):
            rejected = resolver.update(
                stream_id="run_1",
                outward_normals_rad=(math.pi, 0.0),
            )
            self.assertFalse(rejected.identity_resolved)
            self.assertEqual(
                {face.face_id: face.outward_normal_rad for face in rejected.faces},
                trusted_by_id,
            )

        for _ in range(2):
            recovered = resolver.update(
                stream_id="run_1",
                outward_normals_rad=(math.pi / 2.0, -math.pi / 2.0),
            )
            self.assertTrue(recovered.identity_resolved)
        recovered_by_id = {
            face.face_id: face.outward_normal_rad for face in recovered.faces
        }
        self.assertLess(
            angular_distance(recovered_by_id["face_a"], trusted_by_id["face_a"]),
            math.radians(2.0),
        )


def qr_observation(**overrides) -> QrBindingObservation:
    values = {
        "face_id": "face_a",
        "confidence": 0.96,
        "provenance": "sim_qr_consensus",
        "registry_match": True,
        "inside_target_roi": True,
        "distinct_fresh_frame_consensus": True,
        "visibility_margin_rad": math.radians(15.0),
    }
    values.update(overrides)
    return QrBindingObservation(**values)


class QrFaceLatchTest(unittest.TestCase):
    def test_valid_evidence_hard_latches_and_dropout_persists(self):
        latch = QrFaceLatch(min_visibility_margin_rad=math.radians(8.0))
        accepted = latch.update(
            stream_id="run_1",
            observation=qr_observation(),
            known_face_ids={"face_a", "face_b"},
        )
        self.assertTrue(accepted.accepted)
        self.assertTrue(accepted.evidence.hard)
        dropout = latch.update(stream_id="run_1", observation=None)
        self.assertEqual(dropout.evidence.face_id, "face_a")
        self.assertEqual(dropout.reason, "dropout_latch_retained")

    def test_invalid_registry_roi_duplicate_and_tangent_fail_closed(self):
        cases = (
            ({"registry_match": False}, "registry_mismatch"),
            ({"inside_target_roi": False}, "outside_target_roi"),
            (
                {"distinct_fresh_frame_consensus": False},
                "insufficient_fresh_frame_consensus",
            ),
            ({"visibility_margin_rad": math.radians(7.9)}, "visibility_near_tangent"),
        )
        for changed, reason in cases:
            with self.subTest(reason=reason):
                result = QrFaceLatch().update(
                    stream_id="run_1", observation=qr_observation(**changed)
                )
                self.assertFalse(result.accepted)
                self.assertFalse(result.evidence.valid)
                self.assertEqual(result.reason, reason)

    def test_contradiction_never_replaces_latch(self):
        latch = QrFaceLatch()
        latch.update(stream_id="run_1", observation=qr_observation())
        contradiction = latch.update(
            stream_id="run_1", observation=qr_observation(face_id="face_b")
        )
        self.assertFalse(contradiction.accepted)
        self.assertEqual(contradiction.reason, "contradicts_latch")
        self.assertFalse(contradiction.evidence.valid)
        self.assertIsNone(contradiction.evidence.face_id)
        poisoned = latch.update(stream_id="run_1", observation=None)
        self.assertEqual(poisoned.reason, "contradicts_latch")
        self.assertFalse(poisoned.evidence.valid)

    def test_new_stream_clears_latch(self):
        latch = QrFaceLatch()
        latch.update(stream_id="run_1", observation=qr_observation())
        cleared = latch.update(stream_id="run_2", observation=None)
        self.assertIsNone(cleared.evidence.face_id)
        self.assertFalse(cleared.evidence.valid)

    def test_same_stream_sensor_fault_cannot_clear_sticky_contradiction(self):
        latch = QrFaceLatch()
        latch.update(stream_id="run_1", observation=qr_observation())
        latch.update(
            stream_id="run_1", observation=qr_observation(face_id="face_b")
        )
        latch.invalidate(stream_id="run_1", reason="sensor_frame_mismatch")

        still_poisoned = latch.update(
            stream_id="run_1", observation=qr_observation(face_id="face_a")
        )

        self.assertFalse(still_poisoned.evidence.valid)
        self.assertEqual(still_poisoned.reason, "contradicts_latch")


if __name__ == "__main__":
    unittest.main()
