from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.real_robot import autonomous_observation_tf_contract
from scripts.aufgabe04.real_robot import autonomous_observation_tf_readiness
from scripts.aufgabe04.real_robot.autonomous_observation_tf_readiness import (
    FAILURE_OBSERVATION_EFFECT,
    FAILURE_OBSERVER_CLOCK,
    FAILURE_SCAN_FRAME_EMPTY,
    FAILURE_SCAN_FRAME_MISMATCH,
    FAILURE_SCAN_STAMP_FUTURE,
    FAILURE_SCAN_STAMP_INVALID,
    FAILURE_SCAN_STAMP_STALE,
    FAILURE_SCAN_TIMEOUT,
    FAILURE_TRANSFORM_FRAME_MISMATCH,
    FAILURE_TRANSFORM_NOT_EXACT_TIME,
    FAILURE_TRANSFORM_PAYLOAD_INVALID,
    FAILURE_TRANSFORM_TIMING,
    FAILURE_TRANSFORM_UNAVAILABLE,
    ObservationTfEvidence,
    ObservationTfReadinessConfig,
    ObservationTfReadinessError,
    evaluate_observation_tf_readiness,
    observe_observation_tf_readiness,
)


NOW_NS = 100_000_000_000
SCAN_NS = 99_900_000_000


def config(**changes) -> ObservationTfReadinessConfig:
    values = {
        "scan_topic": "/scan",
        "expected_scan_frame": "base_scan",
        "target_frame": "odom",
        "timeout_sec": 2.0,
        "max_scan_age_sec": 1.0,
        "max_future_timestamp_sec": 0.25,
        "max_tf_age_sec": 1.0,
        "max_tf_scan_skew_sec": 0.02,
        "poll_interval_sec": 0.02,
    }
    values.update(changes)
    return ObservationTfReadinessConfig(**values)


def evidence(**changes) -> ObservationTfEvidence:
    values = {
        "observed_at_ns": NOW_NS,
        "scan_received": True,
        "scan_frame": "base_scan",
        "scan_stamp_ns": SCAN_NS,
        "transform_checked": True,
        "transform_available": True,
        "transform_target_frame": "odom",
        "transform_source_frame": "base_scan",
        "transform_query_stamp_ns": SCAN_NS,
        "transform_stamp_ns": SCAN_NS,
        "transform_x_m": 0.1,
        "transform_y_m": -0.2,
        "transform_z_m": 0.0,
        "transform_yaw_rad": 0.03,
        "transform_quaternion_norm": 1.0,
    }
    values.update(changes)
    return ObservationTfEvidence(**values)


class ObservationTfReadinessTest(unittest.TestCase):
    def test_contract_is_ros_free_acyclic_and_facade_reexports_public_api(self):
        contract_path = Path(autonomous_observation_tf_contract.__file__)
        facade_path = Path(autonomous_observation_tf_readiness.__file__)
        contract_tree = ast.parse(contract_path.read_text(encoding="utf-8"))
        facade_tree = ast.parse(facade_path.read_text(encoding="utf-8"))

        contract_imports = {
            alias.name
            for node in ast.walk(contract_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(contract_tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            {"rclpy", "tf2_ros", "subprocess"}
            & {name.split(".", 1)[0] for name in contract_imports}
        )
        self.assertNotIn(
            "scripts.aufgabe04.real_robot.autonomous_observation_tf_readiness",
            contract_imports,
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"
                for node in ast.walk(contract_tree)
            )
        )

        facade_imports = {
            node.module
            for node in ast.walk(facade_tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertIn(
            "scripts.aufgabe04.real_robot.autonomous_observation_tf_contract",
            facade_imports,
        )
        for public_name in autonomous_observation_tf_contract.__all__:
            with self.subTest(public_name=public_name):
                self.assertIs(
                    getattr(autonomous_observation_tf_readiness, public_name),
                    getattr(autonomous_observation_tf_contract, public_name),
                )

    def test_ready_result_is_json_persistable_and_declares_no_side_effects(self):
        result = evaluate_observation_tf_readiness(config(), evidence())

        self.assertTrue(result.ready)
        self.assertIsNone(result.failure_code)
        self.assertAlmostEqual(result.scan_age_sec, 0.1)
        payload = result.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertFalse(payload["motion_published"])
        self.assertFalse(payload["operator_input_requested"])
        self.assertFalse(payload["subprocess_started"])
        self.assertEqual(payload["evidence"]["scan_stamp_ns"], SCAN_NS)
        json.dumps(payload, sort_keys=True)

    def test_empty_or_wrong_scan_frame_fails_exact_identity_gate(self):
        cases = (
            (None, FAILURE_SCAN_FRAME_EMPTY),
            ("", FAILURE_SCAN_FRAME_EMPTY),
            ("/base_scan", FAILURE_SCAN_FRAME_MISMATCH),
            ("laser", FAILURE_SCAN_FRAME_MISMATCH),
        )
        for scan_frame, expected_code in cases:
            with self.subTest(scan_frame=scan_frame):
                result = evaluate_observation_tf_readiness(
                    config(), evidence(scan_frame=scan_frame)
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_scan_must_arrive_with_positive_fresh_timestamp(self):
        cases = (
            (
                evidence(scan_received=False),
                FAILURE_SCAN_TIMEOUT,
            ),
            (
                evidence(observed_at_ns=0),
                FAILURE_OBSERVER_CLOCK,
            ),
            (
                evidence(scan_stamp_ns=0),
                FAILURE_SCAN_STAMP_INVALID,
            ),
            (
                evidence(scan_stamp_ns=98_000_000_000),
                FAILURE_SCAN_STAMP_STALE,
            ),
            (
                evidence(scan_stamp_ns=100_300_000_000),
                FAILURE_SCAN_STAMP_FUTURE,
            ),
        )
        for observed, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_observation_tf_readiness(config(), observed)
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_transform_must_be_available_at_exact_scan_timestamp(self):
        cases = (
            (
                evidence(transform_checked=False, transform_available=False),
                FAILURE_TRANSFORM_UNAVAILABLE,
            ),
            (
                evidence(
                    transform_available=False,
                    transform_error="disconnected trees",
                ),
                FAILURE_TRANSFORM_UNAVAILABLE,
            ),
            (
                evidence(transform_query_stamp_ns=SCAN_NS + 1),
                FAILURE_TRANSFORM_NOT_EXACT_TIME,
            ),
            (
                evidence(transform_target_frame="map"),
                FAILURE_TRANSFORM_FRAME_MISMATCH,
            ),
            (
                evidence(transform_source_frame="laser"),
                FAILURE_TRANSFORM_FRAME_MISMATCH,
            ),
        )
        for observed, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_observation_tf_readiness(config(), observed)
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_transform_payload_and_timing_fail_closed(self):
        cases = (
            (
                evidence(transform_x_m=math.nan),
                FAILURE_TRANSFORM_PAYLOAD_INVALID,
            ),
            (
                evidence(transform_quaternion_norm=0.5),
                FAILURE_TRANSFORM_PAYLOAD_INVALID,
            ),
            (
                evidence(transform_stamp_ns=SCAN_NS + 20_000_001),
                FAILURE_TRANSFORM_TIMING,
            ),
            (
                evidence(transform_stamp_ns=0),
                FAILURE_TRANSFORM_TIMING,
            ),
        )
        for observed, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_observation_tf_readiness(config(), observed)
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_observation_effect_is_injectable_and_reuses_pure_evaluator(self):
        calls = []

        def effect(selected):
            calls.append(selected)
            return evidence()

        selected = config()
        result = observe_observation_tf_readiness(
            selected,
            observation_effect=effect,
        )

        self.assertTrue(result.ready)
        self.assertEqual(calls, [selected])

    def test_effect_exception_is_a_persistable_fail_closed_result(self):
        def effect(_selected):
            raise RuntimeError("collector broke")

        result = observe_observation_tf_readiness(
            config(),
            observation_effect=effect,
        )

        self.assertFalse(result.ready)
        self.assertEqual(result.failure_code, FAILURE_OBSERVATION_EFFECT)
        self.assertIn("collector broke", result.detail)
        json.dumps(result.to_dict(), sort_keys=True)

    def test_typed_error_preserves_pre_or_post_run_failure_boundary(self):
        rejected = evaluate_observation_tf_readiness(
            config(),
            evidence(transform_available=False),
        )
        error = ObservationTfReadinessError(
            rejected,
            evidence_path="preflight/scan_tf.json",
            evidence_sha256="a" * 64,
            phase="coverage_leg_before_motion",
            typed_run_already_issued=True,
        )

        fields = error.to_failure_fields()
        self.assertEqual(fields["failure_phase"], "coverage_leg_before_motion")
        self.assertTrue(fields["typed_run_already_issued"])
        self.assertTrue(fields["typed_run_requested"])
        self.assertFalse(fields["motion_authorized"])
        self.assertFalse(fields["motion_published"])
        self.assertEqual(fields["observation_tf_readiness_sha256"], "a" * 64)

    def test_config_requires_nonempty_names_and_finite_bounded_timeout(self):
        invalid = (
            config(scan_topic=""),
            config(expected_scan_frame=""),
            config(target_frame=" odom"),
            config(timeout_sec=0.0),
            config(timeout_sec=math.inf),
            config(max_scan_age_sec=-1.0),
            config(max_future_timestamp_sec=-1.0),
            config(max_tf_age_sec=0.0),
            config(max_tf_scan_skew_sec=-1.0),
            config(poll_interval_sec=3.0),
        )
        for selected in invalid:
            with self.subTest(selected=selected):
                with self.assertRaises(ValueError):
                    selected.validated()


if __name__ == "__main__":
    unittest.main()
