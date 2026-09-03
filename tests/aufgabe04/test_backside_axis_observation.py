import copy
import math
import unittest

from scripts.aufgabe04.real_robot.observer.backside_axis_observation import (
    build_backside_axis_observation,
    validate_backside_axis_observation,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_OBSERVATION_KIND,
    BACKSIDE_AXIS_SAMPLE_SOURCE,
    BACKSIDE_CLASSIFICATION_BASIS,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
)


def valid_inputs() -> dict[str, object]:
    return {
        "stream_id": "survey_candidate_0001_attempt_0",
        "stand_id": "candidate_0001",
        "planning_frame": "map",
        "stand_x_m": 0.4,
        "stand_y_m": -0.2,
        "robot_x_m": 0.8,
        "robot_y_m": -0.2,
        "robot_yaw_rad": math.pi,
        "stand_axis_rad": math.pi / 2.0,
        "axis_confidence": 0.87,
        "axis_sample_count": 7,
        "consensus_source": BACKSIDE_AXIS_SAMPLE_SOURCE,
        "estimate_source": BACKSIDE_AXIS_SAMPLE_SOURCE,
        "estimate_evidence_state": "fresh_backside",
        "estimate_visible_face": "backside_candidate",
        "visible_face_confidence": 0.81,
        "debug_qr_detected": False,
        "qr_texts": (),
        "evidence_qr_sample_count": 0,
        "evidence_tentative_qr_id": None,
        "evidence_latched_qr_id": None,
        "qr_marker_seen_in_stationary_epoch": False,
        "all_samples_stationary": True,
        "all_samples_synchronized": True,
        "all_samples_lidar_associated": True,
        "sensor_stamp_sec": 123.5,
        "stand_model_profile_sha256": "a" * 64,
        "stand_model_measurement_status": "measured",
        "head_scale_ratio": 0.98,
        "head_center_error_ratio": 0.07,
        "pose_reprojection_rmse_px": None,
        "pose_ambiguity_gap_px": None,
        "robot_profile_sha256": "b" * 64,
        "calibration_profile_sha256": "c" * 64,
    }


class BacksideAxisObservationTest(unittest.TestCase):
    def test_builder_emits_strict_motion_neutral_schema_two_contract(self):
        payload = build_backside_axis_observation(**valid_inputs())

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["observation_kind"], BACKSIDE_AXIS_OBSERVATION_KIND)
        self.assertEqual(payload["motion_capability"], "none")
        self.assertEqual(payload["observer_version"], PASSIVE_VIEWPOINT_OBSERVER_VERSION)
        self.assertEqual(payload["visible_face"], "backside_candidate")
        self.assertEqual(
            payload["visible_face_source"],
            BACKSIDE_AXIS_SAMPLE_SOURCE,
        )
        self.assertEqual(payload["axis_sample_source"], BACKSIDE_AXIS_SAMPLE_SOURCE)
        self.assertEqual(payload["model_evidence_state"], "fresh_backside")
        self.assertEqual(payload["classification_basis"], BACKSIDE_CLASSIFICATION_BASIS)
        self.assertIs(payload["qr_marker_detected"], False)
        self.assertEqual(payload["qr_texts"], [])
        self.assertEqual(payload["qr_absent_sample_count"], 7)
        self.assertEqual(
            payload["sample_gate_evidence"],
            {
                "all_samples_stationary": True,
                "all_samples_synchronized": True,
                "all_samples_lidar_associated": True,
                "all_samples_current_frame_model_geometry": True,
                "all_samples_qr_marker_absent": True,
            },
        )
        validate_backside_axis_observation(payload)

    def test_builder_rejects_non_backside_or_predicted_sources(self):
        cases = {
            "consensus_source": "model_current_frame_refined",
            "estimate_source": "model_projection",
            "estimate_evidence_state": "predicted_only",
            "estimate_visible_face": None,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                arguments = valid_inputs()
                arguments[field] = value
                with self.assertRaises(ValueError):
                    build_backside_axis_observation(**arguments)

    def test_builder_rejects_current_or_historical_qr_evidence(self):
        cases = {
            "debug_qr_detected": True,
            "qr_texts": ("STAND_2",),
            "evidence_qr_sample_count": 1,
            "evidence_tentative_qr_id": "STAND_2",
            "evidence_latched_qr_id": "STAND_2",
            "qr_marker_seen_in_stationary_epoch": True,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                arguments = valid_inputs()
                arguments[field] = value
                with self.assertRaises(ValueError):
                    build_backside_axis_observation(**arguments)

        arguments = valid_inputs()
        arguments["debug_qr_detected"] = 0
        with self.assertRaises(ValueError):
            build_backside_axis_observation(**arguments)

    def test_builder_rejects_weak_nonfinite_or_ungated_evidence(self):
        cases = {
            "axis_confidence": 0.59,
            "visible_face_confidence": 0.69,
            "axis_sample_count": 1,
            "all_samples_stationary": False,
            "all_samples_synchronized": False,
            "all_samples_lidar_associated": False,
            "stand_x_m": math.nan,
            "head_scale_ratio": math.inf,
            "head_center_error_ratio": math.nan,
            "stand_model_profile_sha256": "not-a-hash",
            "stand_model_measurement_status": "provisional",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                arguments = valid_inputs()
                arguments[field] = value
                with self.assertRaises(ValueError):
                    build_backside_axis_observation(**arguments)

        for field, value in (
            ("head_scale_ratio", 0.59),
            ("head_scale_ratio", 1.36),
            ("head_center_error_ratio", 0.56),
        ):
            with self.subTest(field=field, value=value):
                arguments = valid_inputs()
                arguments[field] = value
                with self.assertRaises(ValueError):
                    build_backside_axis_observation(**arguments)

        arguments = valid_inputs()
        arguments["all_samples_stationary"] = 1
        with self.assertRaises(ValueError):
            build_backside_axis_observation(**arguments)

    def test_validator_rejects_contract_tampering(self):
        original = build_backside_axis_observation(**valid_inputs())
        cases = {
            "motion_capability": "cmd_vel",
            "visible_face": "backside",
            "axis_sample_source": "model_projection",
            "qr_marker_detected": True,
            "qr_absent_sample_count": 6,
            "observer_version": "legacy-observer",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                payload = copy.deepcopy(original)
                payload[field] = value
                with self.assertRaises(ValueError):
                    validate_backside_axis_observation(payload)

        payload = copy.deepcopy(original)
        payload["sample_gate_evidence"]["all_samples_lidar_associated"] = False
        with self.assertRaises(ValueError):
            validate_backside_axis_observation(payload)


if __name__ == "__main__":
    unittest.main()
