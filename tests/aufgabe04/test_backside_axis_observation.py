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
    MAXIMUM_REGISTRATION_BEARING_DELTA_RAD,
    REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
    TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA,
    TARGET_REGISTRATION_LIDAR_SOURCE_MAP,
    TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR,
    TARGET_REGISTRATION_MODE_MAP_PROJECTION,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
)
from scripts.aufgabe04.artifacts.backside_axis_observation import (
    LEGACY_PASSIVE_VIEWPOINT_OBSERVER_VERSION,
)


def nominal_target_registration() -> dict[str, object]:
    return {
        "mode": TARGET_REGISTRATION_MODE_MAP_PROJECTION,
        "original_head_center_error_ratio": 0.07,
        "center_offset_limit_ratio": 0.55,
        "final_strict_head_center_error_ratio": 0.07,
        "map_bearing_rad": -0.10,
        "lidar_search_bearing_rad": -0.10,
        "camera_map_bearing_delta_rad": 0.02,
        "bearing_delta_limit_rad": math.radians(3.0),
        "lidar_search_bearing_source": TARGET_REGISTRATION_LIDAR_SOURCE_MAP,
        "unique_eligible_lidar_cluster_required": False,
        "eligible_lidar_cluster_count": 2,
    }


def registered_target_registration() -> dict[str, object]:
    map_bearing_rad = math.pi - 0.08
    lidar_search_bearing_rad = -math.pi + 0.08
    wrapped_delta_rad = abs(
        math.atan2(
            math.sin(lidar_search_bearing_rad - map_bearing_rad),
            math.cos(lidar_search_bearing_rad - map_bearing_rad),
        )
    )
    return {
        "mode": TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR,
        "original_head_center_error_ratio": 1.18,
        "center_offset_limit_ratio": 1.50,
        "final_strict_head_center_error_ratio": 0.07,
        "map_bearing_rad": map_bearing_rad,
        "lidar_search_bearing_rad": lidar_search_bearing_rad,
        "camera_map_bearing_delta_rad": wrapped_delta_rad,
        "bearing_delta_limit_rad": MAXIMUM_REGISTRATION_BEARING_DELTA_RAD,
        "lidar_search_bearing_source": TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA,
        "unique_eligible_lidar_cluster_required": True,
        "eligible_lidar_cluster_count": 1,
    }


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
        "target_registration": nominal_target_registration(),
    }


class BacksideAxisObservationTest(unittest.TestCase):
    def test_builder_emits_strict_motion_neutral_schema_three_contract(self):
        payload = build_backside_axis_observation(**valid_inputs())

        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["observation_kind"], BACKSIDE_AXIS_OBSERVATION_KIND)
        self.assertEqual(payload["motion_capability"], "none")
        self.assertEqual(payload["observer_version"], PASSIVE_VIEWPOINT_OBSERVER_VERSION)
        self.assertEqual(payload["visible_face"], "backside_candidate")
        self.assertEqual(
            payload["visible_face_source"],
            BACKSIDE_AXIS_SAMPLE_SOURCE,
        )
        self.assertEqual(payload["axis_sample_source"], BACKSIDE_AXIS_SAMPLE_SOURCE)
        self.assertEqual(
            payload["target_registration"], nominal_target_registration()
        )
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

    def test_builder_emits_registered_receipt_with_distinct_consensus_source(self):
        arguments = valid_inputs()
        arguments["consensus_source"] = REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
        arguments["target_registration"] = registered_target_registration()

        payload = build_backside_axis_observation(**arguments)

        self.assertEqual(
            payload["axis_sample_source"],
            REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
        )
        self.assertEqual(
            payload["visible_face_source"], BACKSIDE_AXIS_SAMPLE_SOURCE
        )
        validate_backside_axis_observation(payload)

    def test_validator_keeps_legacy_schema_two_receipts_readable(self):
        payload = build_backside_axis_observation(**valid_inputs())
        payload["schema_version"] = 2
        payload["observer_version"] = LEGACY_PASSIVE_VIEWPOINT_OBSERVER_VERSION
        payload.pop("target_registration")

        validate_backside_axis_observation(payload)

        mutations = {
            "target_registration": nominal_target_registration(),
            "observer_version": PASSIVE_VIEWPOINT_OBSERVER_VERSION,
            "axis_sample_source": REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                tampered = copy.deepcopy(payload)
                tampered[field] = value
                with self.assertRaises(ValueError):
                    validate_backside_axis_observation(tampered)

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

    def test_builder_rejects_consensus_source_registration_mismatch(self):
        arguments = valid_inputs()
        arguments["target_registration"] = registered_target_registration()
        with self.assertRaisesRegex(ValueError, "consensus source"):
            build_backside_axis_observation(**arguments)

        arguments = valid_inputs()
        arguments["consensus_source"] = REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
        with self.assertRaisesRegex(ValueError, "consensus source"):
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

    def test_validator_rejects_registration_shape_and_nonfinite_values(self):
        original = build_backside_axis_observation(**valid_inputs())
        for field in nominal_target_registration():
            with self.subTest(missing=field):
                payload = copy.deepcopy(original)
                payload["target_registration"].pop(field)
                with self.assertRaisesRegex(ValueError, "unexpected fields"):
                    validate_backside_axis_observation(payload)

        payload = copy.deepcopy(original)
        payload["target_registration"]["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            validate_backside_axis_observation(payload)

        numeric_fields = (
            "original_head_center_error_ratio",
            "center_offset_limit_ratio",
            "final_strict_head_center_error_ratio",
            "map_bearing_rad",
            "lidar_search_bearing_rad",
            "camera_map_bearing_delta_rad",
            "bearing_delta_limit_rad",
        )
        for field in numeric_fields:
            with self.subTest(nonfinite=field):
                payload = copy.deepcopy(original)
                payload["target_registration"][field] = math.nan
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_backside_axis_observation(payload)

    def test_validator_rejects_registration_hard_limit_violations(self):
        original = build_backside_axis_observation(**valid_inputs())
        mutations = (
            ("center_offset_limit_ratio", 1.500001),
            ("original_head_center_error_ratio", 0.550001),
            (
                "bearing_delta_limit_rad",
                MAXIMUM_REGISTRATION_BEARING_DELTA_RAD + 1.0e-9,
            ),
            ("camera_map_bearing_delta_rad", math.radians(3.1)),
            ("final_strict_head_center_error_ratio", 0.56),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                payload = copy.deepcopy(original)
                payload["target_registration"][field] = value
                with self.assertRaises(ValueError):
                    validate_backside_axis_observation(payload)

    def test_registered_mode_fails_closed_on_provenance_tampering(self):
        arguments = valid_inputs()
        arguments["consensus_source"] = REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
        arguments["target_registration"] = registered_target_registration()
        original = build_backside_axis_observation(**arguments)
        mutations = (
            ("lidar_search_bearing_source", TARGET_REGISTRATION_LIDAR_SOURCE_MAP),
            ("unique_eligible_lidar_cluster_required", False),
            ("eligible_lidar_cluster_count", 2),
            ("camera_map_bearing_delta_rad", 0.15),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                payload = copy.deepcopy(original)
                payload["target_registration"][field] = value
                with self.assertRaises(ValueError):
                    validate_backside_axis_observation(payload)

        payload = copy.deepcopy(original)
        payload["axis_sample_source"] = BACKSIDE_AXIS_SAMPLE_SOURCE
        with self.assertRaisesRegex(ValueError, "axis_sample_source"):
            validate_backside_axis_observation(payload)

    def test_nominal_mode_fails_closed_on_registration_tampering(self):
        original = build_backside_axis_observation(**valid_inputs())
        mutations = (
            ("lidar_search_bearing_source", TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA),
            ("unique_eligible_lidar_cluster_required", True),
            ("lidar_search_bearing_rad", -0.09),
            ("original_head_center_error_ratio", 0.08),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                payload = copy.deepcopy(original)
                payload["target_registration"][field] = value
                with self.assertRaises(ValueError):
                    validate_backside_axis_observation(payload)


if __name__ == "__main__":
    unittest.main()
