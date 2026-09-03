import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.camera_axis_binding import (
    BACKSIDE_CLASSIFICATION_BASIS,
    BACKSIDE_CURRENT_FRAME_SOURCE,
    BACKSIDE_MODEL_EVIDENCE_STATE,
    BACKSIDE_VISIBLE_FACE,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
    REAL_STAND_AXIS_OBSERVATION_KIND,
    load_opposite_face_normal,
    opposite_face_normal_from_axis_observation,
    validated_backside_axis_observation,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_materialization import (
    validate_backside_axis_candidate_binding,
)


def _observation(*, robot_y_m: float = 0.7) -> dict[str, object]:
    return {
        "schema_version": 2,
        "observation_kind": REAL_STAND_AXIS_OBSERVATION_KIND,
        "stand_id": "candidate_1",
        "planning_frame": "map",
        "visible_face": BACKSIDE_VISIBLE_FACE,
        "visible_face_source": BACKSIDE_CURRENT_FRAME_SOURCE,
        "visible_face_confidence": 0.82,
        "classification_basis": BACKSIDE_CLASSIFICATION_BASIS,
        "motion_capability": "none",
        "observer_version": PASSIVE_VIEWPOINT_OBSERVER_VERSION,
        "stream_id": "candidate_1_attempt_0",
        "axis_sample_source": BACKSIDE_CURRENT_FRAME_SOURCE,
        "axis_confidence": 0.79,
        "axis_sample_count": 3,
        "qr_absent_sample_count": 3,
        "model_evidence_state": BACKSIDE_MODEL_EVIDENCE_STATE,
        "stand_model_profile_sha256": "a" * 64,
        "stand_model_measurement_status": "measured",
        "robot_profile_sha256": "b" * 64,
        "calibration_profile_sha256": "c" * 64,
        "sensor_stamp_sec": 123.5,
        "head_scale_ratio": 0.98,
        "head_center_error_ratio": 0.07,
        "pose_reprojection_rmse_px": None,
        "pose_ambiguity_gap_px": None,
        "qr_marker_detected": False,
        "qr_texts": [],
        "sample_gate_evidence": {
            "all_samples_stationary": True,
            "all_samples_synchronized": True,
            "all_samples_lidar_associated": True,
            "all_samples_current_frame_model_geometry": True,
            "all_samples_qr_marker_absent": True,
        },
        "stand_axis_rad": 0.0,
        "stand_center": {"x_m": 0.5, "y_m": 0.0},
        "robot_pose": {
            "x_m": 0.5,
            "y_m": robot_y_m,
            "yaw_rad": -math.pi / 2.0,
        },
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

    def test_all_backside_provenance_is_required(self):
        mutations = {
            "schema_version": 1,
            "observation_kind": "real_stand_axis_without_qr",
            "visible_face": "frontside",
            "visible_face_source": "tracker_prediction",
            "axis_sample_source": "tracker_prediction",
            "model_evidence_state": "tracking_supported",
            "classification_basis": "absence_only",
            "motion_capability": "reorient",
            "qr_marker_detected": True,
            "qr_texts": ["A1"],
            "visible_face_confidence": 0.69,
            "axis_confidence": 0.59,
            "axis_sample_count": 1,
            "qr_absent_sample_count": 2,
            "stand_model_profile_sha256": "A" * 64,
        }
        for field, invalid in mutations.items():
            with self.subTest(field=field):
                payload = _observation()
                payload[field] = invalid
                with self.assertRaises(ValueError):
                    opposite_face_normal_from_axis_observation(payload)

    def test_qr_absence_must_cover_every_axis_sample(self):
        payload = _observation()
        payload["axis_sample_count"] = 4

        with self.assertRaisesRegex(ValueError, "must equal"):
            opposite_face_normal_from_axis_observation(payload)

    def test_full_observer_receipt_tampering_fails_closed(self):
        mutations = (
            ("observer_version", "legacy-observer"),
            ("stand_model_measurement_status", "provisional"),
            ("sensor_stamp_sec", -0.001),
            ("sensor_stamp_sec", math.inf),
            ("head_scale_ratio", 0.599),
            ("head_scale_ratio", 1.351),
            ("head_center_error_ratio", -0.001),
            ("head_center_error_ratio", 0.551),
            ("robot_profile_sha256", "not-a-hash"),
            ("calibration_profile_sha256", "D" * 64),
            ("visible_face_confidence", 1.001),
            ("axis_confidence", 1.001),
            ("pose_reprojection_rmse_px", -0.001),
            ("pose_reprojection_rmse_px", math.nan),
            ("pose_ambiguity_gap_px", -0.001),
            ("pose_ambiguity_gap_px", math.inf),
            ("stream_id", ""),
        )
        for field, invalid in mutations:
            with self.subTest(field=field, invalid=invalid):
                payload = _observation()
                payload[field] = invalid
                with self.assertRaises(ValueError):
                    opposite_face_normal_from_axis_observation(payload)

        invalid_robot_yaw = _observation()
        invalid_robot_yaw["robot_pose"]["yaw_rad"] = math.nan
        with self.assertRaisesRegex(ValueError, "robot_pose.yaw_rad"):
            opposite_face_normal_from_axis_observation(invalid_robot_yaw)

    def test_optional_pose_metrics_accept_none_or_nonnegative_finite_values(self):
        payload = _observation()
        payload["pose_reprojection_rmse_px"] = 1.4
        payload["pose_ambiguity_gap_px"] = 0.0

        self.assertTrue(
            math.isfinite(opposite_face_normal_from_axis_observation(payload))
        )

    def test_classification_basis_is_required_and_canonical(self):
        absent = _observation()
        absent.pop("classification_basis")
        unsupported = _observation()
        unsupported["classification_basis"] = "absence_only"

        for payload in (absent, unsupported):
            with self.assertRaisesRegex(ValueError, "classification_basis"):
                opposite_face_normal_from_axis_observation(payload)

    def test_sample_gate_evidence_is_required_exact_and_all_true(self):
        for gate in _observation()["sample_gate_evidence"]:
            with self.subTest(gate=gate):
                payload = _observation()
                payload["sample_gate_evidence"][gate] = False
                with self.assertRaisesRegex(ValueError, gate):
                    opposite_face_normal_from_axis_observation(payload)

        absent = _observation()
        absent.pop("sample_gate_evidence")
        extra = _observation()
        extra["sample_gate_evidence"]["unreviewed_gate"] = True
        for payload in (absent, extra):
            with self.assertRaisesRegex(ValueError, "sample_gate_evidence"):
                opposite_face_normal_from_axis_observation(payload)

    def test_candidate_binding_rejects_cross_target_receipts(self):
        observation = validated_backside_axis_observation(_observation())

        validate_backside_axis_candidate_binding(
            observation,
            candidate_uid="candidate_1",
            planning_frame="map",
            candidate_x_m=0.5,
            candidate_y_m=0.0,
        )
        mismatches = (
            ("candidate_2", "map", 0.5, 0.0, "stand ID"),
            ("candidate_1", "odom", 0.5, 0.0, "planning frame"),
            ("candidate_1", "map", 0.500002, 0.0, "stand center"),
        )
        for uid, frame, x_m, y_m, message in mismatches:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_backside_axis_candidate_binding(
                    observation,
                    candidate_uid=uid,
                    planning_frame=frame,
                    candidate_x_m=x_m,
                    candidate_y_m=y_m,
                )

    def test_loader_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text("[]\n")

            with self.assertRaisesRegex(ValueError, "root must be an object"):
                load_opposite_face_normal(path)


if __name__ == "__main__":
    unittest.main()
