from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.real_robot.autonomous_runner import runtime as runner
from scripts.aufgabe04.real_robot.readiness import (
    sensor_timing_contract as autonomous_sensor_timing_contract,
)
from scripts.aufgabe04.real_robot.readiness import (
    sensor_timing_runtime as autonomous_sensor_timing_readiness,
)
from scripts.aufgabe04.real_robot.readiness.sensor_timing_runtime import (
    FAILURE_CAMERA_INFO_FRAME_EMPTY,
    FAILURE_CAMERA_INFO_FRAME_MISMATCH,
    FAILURE_CAMERA_INFO_IMAGE_SKEW,
    FAILURE_CAMERA_INFO_STAMP_FUTURE,
    FAILURE_CAMERA_INFO_STAMP_INVALID,
    FAILURE_CAMERA_INFO_STAMP_STALE,
    FAILURE_CAMERA_INFO_TIMEOUT,
    FAILURE_IMAGE_FRAME_EMPTY,
    FAILURE_IMAGE_FRAME_MISMATCH,
    FAILURE_IMAGE_SCAN_SKEW,
    FAILURE_IMAGE_STAMP_FUTURE,
    FAILURE_IMAGE_STAMP_INVALID,
    FAILURE_IMAGE_STAMP_STALE,
    FAILURE_IMAGE_TIMEOUT,
    FAILURE_OBSERVATION_EFFECT,
    FAILURE_OBSERVER_CLOCK,
    FAILURE_SCAN_FRAME_EMPTY,
    FAILURE_SCAN_FRAME_MISMATCH,
    FAILURE_SCAN_STAMP_FUTURE,
    FAILURE_SCAN_STAMP_INVALID,
    FAILURE_SCAN_STAMP_STALE,
    FAILURE_SCAN_TIMEOUT,
    HeaderSample,
    SensorTimingEvidence,
    SensorTimingReadinessConfig,
    SensorTimingReadinessError,
    evaluate_sensor_timing_readiness,
    observe_sensor_timing_readiness,
)


NOW_NS = 100_000_000_000
IMAGE_NS = 99_900_000_000
CAMERA_INFO_NS = 99_890_000_000
SCAN_NS = 99_910_000_000


def sample(
    stamp_ns: int | None,
    frame_id: str | None,
    *,
    receipt_ns: int = NOW_NS,
) -> HeaderSample:
    return HeaderSample(
        stamp_ns=stamp_ns,
        frame_id=frame_id,
        receipt_ns=receipt_ns,
    )


def config(**changes) -> SensorTimingReadinessConfig:
    values = {
        "image_topic": "/camera/image_raw/compressed",
        "camera_info_topic": "/camera/camera_info",
        "scan_topic": "/scan",
        "expected_image_frame": "camera",
        "expected_camera_info_frame": "camera",
        "expected_scan_frame": "base_scan",
        "timeout_sec": 2.0,
        "max_image_age_sec": 1.0,
        "max_camera_info_age_sec": 1.0,
        "max_scan_age_sec": 1.0,
        "max_future_timestamp_sec": 0.05,
        "max_image_scan_skew_sec": 0.10,
        "max_camera_info_image_skew_sec": 0.10,
        "poll_interval_sec": 0.02,
        "sample_capacity": 16,
    }
    values.update(changes)
    return SensorTimingReadinessConfig(**values)


def evidence(**changes) -> SensorTimingEvidence:
    values = {
        "observed_at_ns": NOW_NS,
        "image_samples": (sample(IMAGE_NS, "camera"),),
        "camera_info_samples": (sample(CAMERA_INFO_NS, "camera"),),
        "scan_samples": (sample(SCAN_NS, "base_scan"),),
    }
    values.update(changes)
    return SensorTimingEvidence(**values)


class AutonomousSensorTimingReadinessTest(unittest.TestCase):
    def test_contract_is_ros_free_and_facade_reexports_public_api(self):
        contract_path = Path(autonomous_sensor_timing_contract.__file__)
        facade_path = Path(autonomous_sensor_timing_readiness.__file__)
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
            {"rclpy", "sensor_msgs", "subprocess"}
            & {name.split(".", 1)[0] for name in contract_imports}
        )
        self.assertNotIn(
            "scripts.aufgabe04.real_robot.readiness.sensor_timing_runtime",
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
            "scripts.aufgabe04.real_robot.readiness.sensor_timing_contract",
            facade_imports,
        )
        for public_name in autonomous_sensor_timing_contract.__all__:
            with self.subTest(public_name=public_name):
                self.assertIs(
                    getattr(autonomous_sensor_timing_readiness, public_name),
                    getattr(autonomous_sensor_timing_contract, public_name),
                )

    def test_aligned_tuple_passes_and_json_declares_no_side_effects(self):
        result = evaluate_sensor_timing_readiness(config(), evidence())

        self.assertTrue(result.ready)
        self.assertIsNone(result.failure_code)
        self.assertAlmostEqual(result.image_age_sec, 0.10)
        self.assertAlmostEqual(result.camera_info_age_sec, 0.11)
        self.assertAlmostEqual(result.scan_age_sec, 0.09)
        self.assertAlmostEqual(result.image_scan_skew_sec, 0.01)
        self.assertAlmostEqual(result.camera_info_image_skew_sec, 0.01)
        payload = result.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["selection_policy"],
            "newest_fresh_complete_header_tuple",
        )
        self.assertFalse(payload["motion_published"])
        self.assertFalse(payload["operator_input_requested"])
        self.assertFalse(payload["subprocess_started"])
        self.assertEqual(payload["selected_image_stamp_ns"], IMAGE_NS)
        json.dumps(payload, sort_keys=True)

    def test_twenty_one_hour_camera_clock_offset_is_rejected_as_stale(self):
        stale_image_ns = 1_000_000_000
        observed_at_ns = stale_image_ns + 21 * 60 * 60 * 1_000_000_000
        fresh_ns = observed_at_ns - 100_000_000

        result = evaluate_sensor_timing_readiness(
            config(),
            evidence(
                observed_at_ns=observed_at_ns,
                image_samples=(sample(stale_image_ns, "camera"),),
                camera_info_samples=(sample(fresh_ns, "camera"),),
                scan_samples=(sample(fresh_ns, "base_scan"),),
            ),
        )

        self.assertFalse(result.ready)
        self.assertEqual(result.failure_code, FAILURE_IMAGE_STAMP_STALE)
        self.assertAlmostEqual(result.image_age_sec, 21 * 60 * 60)

    def test_each_missing_stream_fails_with_its_own_timeout_code(self):
        cases = (
            ({"image_samples": ()}, FAILURE_IMAGE_TIMEOUT),
            ({"camera_info_samples": ()}, FAILURE_CAMERA_INFO_TIMEOUT),
            ({"scan_samples": ()}, FAILURE_SCAN_TIMEOUT),
        )
        for changes, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_sensor_timing_readiness(
                    config(), evidence(**changes)
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_observer_and_each_stream_require_valid_positive_timestamps(self):
        cases = (
            ({"observed_at_ns": 0}, FAILURE_OBSERVER_CLOCK),
            (
                {"image_samples": (sample(0, "camera"),)},
                FAILURE_IMAGE_STAMP_INVALID,
            ),
            (
                {"camera_info_samples": (sample(None, "camera"),)},
                FAILURE_CAMERA_INFO_STAMP_INVALID,
            ),
            (
                {"scan_samples": (sample(-1, "base_scan"),)},
                FAILURE_SCAN_STAMP_INVALID,
            ),
        )
        for changes, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_sensor_timing_readiness(
                    config(), evidence(**changes)
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_each_stream_rejects_stale_and_future_headers(self):
        stale_ns = NOW_NS - 2_000_000_000
        future_ns = NOW_NS + 60_000_000
        cases = (
            (
                {"image_samples": (sample(stale_ns, "camera"),)},
                FAILURE_IMAGE_STAMP_STALE,
            ),
            (
                {"camera_info_samples": (sample(stale_ns, "camera"),)},
                FAILURE_CAMERA_INFO_STAMP_STALE,
            ),
            (
                {"scan_samples": (sample(stale_ns, "base_scan"),)},
                FAILURE_SCAN_STAMP_STALE,
            ),
            (
                {"image_samples": (sample(future_ns, "camera"),)},
                FAILURE_IMAGE_STAMP_FUTURE,
            ),
            (
                {"camera_info_samples": (sample(future_ns, "camera"),)},
                FAILURE_CAMERA_INFO_STAMP_FUTURE,
            ),
            (
                {"scan_samples": (sample(future_ns, "base_scan"),)},
                FAILURE_SCAN_STAMP_FUTURE,
            ),
        )
        for changes, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_sensor_timing_readiness(
                    config(), evidence(**changes)
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_latest_received_stale_image_cannot_hide_behind_earlier_header(self):
        current_image = sample(IMAGE_NS, "camera", receipt_ns=99_901_000_000)
        later_stale_image = sample(
            NOW_NS - 2_000_000_000,
            "camera",
            receipt_ns=99_999_000_000,
        )

        result = evaluate_sensor_timing_readiness(
            config(),
            evidence(image_samples=(current_image, later_stale_image)),
        )

        self.assertFalse(result.ready)
        self.assertEqual(result.failure_code, FAILURE_IMAGE_STAMP_STALE)
        self.assertAlmostEqual(result.image_age_sec, 2.0)

    def test_each_stream_rejects_empty_or_mismatched_frames(self):
        cases = (
            (
                {"image_samples": (sample(IMAGE_NS, ""),)},
                FAILURE_IMAGE_FRAME_EMPTY,
            ),
            (
                {"image_samples": (sample(IMAGE_NS, "camera_optical"),)},
                FAILURE_IMAGE_FRAME_MISMATCH,
            ),
            (
                {"camera_info_samples": (sample(CAMERA_INFO_NS, None),)},
                FAILURE_CAMERA_INFO_FRAME_EMPTY,
            ),
            (
                {
                    "camera_info_samples": (
                        sample(CAMERA_INFO_NS, "camera_info"),
                    )
                },
                FAILURE_CAMERA_INFO_FRAME_MISMATCH,
            ),
            (
                {"scan_samples": (sample(SCAN_NS, ""),)},
                FAILURE_SCAN_FRAME_EMPTY,
            ),
            (
                {"scan_samples": (sample(SCAN_NS, "laser"),)},
                FAILURE_SCAN_FRAME_MISMATCH,
            ),
        )
        for changes, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_sensor_timing_readiness(
                    config(), evidence(**changes)
                )
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_superseded_malformed_sample_does_not_poison_valid_latest_tuple(self):
        result = evaluate_sensor_timing_readiness(
            config(),
            evidence(
                image_samples=(
                    sample(None, "wrong_frame"),
                    sample(IMAGE_NS, "camera"),
                ),
            ),
        )

        self.assertTrue(result.ready)
        self.assertEqual(result.selected_image_stamp_ns, IMAGE_NS)

    def test_image_scan_and_camera_info_image_skew_fail_separately(self):
        cases = (
            (
                evidence(
                    image_samples=(sample(99_900_000_000, "camera"),),
                    camera_info_samples=(sample(99_900_000_000, "camera"),),
                    scan_samples=(sample(99_700_000_000, "base_scan"),),
                ),
                FAILURE_IMAGE_SCAN_SKEW,
            ),
            (
                evidence(
                    image_samples=(sample(99_900_000_000, "camera"),),
                    camera_info_samples=(sample(99_700_000_000, "camera"),),
                    scan_samples=(sample(99_900_000_000, "base_scan"),),
                ),
                FAILURE_CAMERA_INFO_IMAGE_SKEW,
            ),
        )
        for observed, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                result = evaluate_sensor_timing_readiness(config(), observed)
                self.assertFalse(result.ready)
                self.assertEqual(result.failure_code, expected_code)

    def test_newest_incomplete_tuple_falls_back_to_older_complete_tuple(self):
        older_ns = 99_700_000_000
        newest_image_ns = 99_950_000_000
        result = evaluate_sensor_timing_readiness(
            config(),
            evidence(
                image_samples=(
                    sample(older_ns, "camera"),
                    sample(newest_image_ns, "camera"),
                ),
                camera_info_samples=(sample(older_ns, "camera"),),
                scan_samples=(
                    sample(older_ns, "base_scan"),
                    sample(99_800_000_000, "base_scan"),
                ),
            ),
        )

        self.assertTrue(result.ready)
        self.assertEqual(result.selected_image_stamp_ns, older_ns)
        self.assertEqual(result.selected_camera_info_stamp_ns, older_ns)
        self.assertEqual(result.selected_scan_stamp_ns, older_ns)
        self.assertAlmostEqual(result.image_scan_skew_sec, 0.0)
        self.assertAlmostEqual(result.camera_info_image_skew_sec, 0.0)

    def test_tuple_selection_skips_nearer_stale_or_future_scan(self):
        cases = (
            (
                config(max_scan_age_sec=0.05),
                evidence(
                    image_samples=(sample(99_900_000_000, "camera"),),
                    camera_info_samples=(sample(99_900_000_000, "camera"),),
                    scan_samples=(
                        # Exact timestamp match, but stale for the scan policy.
                        sample(99_900_000_000, "base_scan"),
                        # Latest received and fresh; still within skew tolerance.
                        sample(99_960_000_000, "base_scan"),
                    ),
                ),
                99_960_000_000,
            ),
            (
                config(),
                evidence(
                    image_samples=(sample(100_040_000_000, "camera"),),
                    camera_info_samples=(sample(100_040_000_000, "camera"),),
                    scan_samples=(
                        # Numerically nearest, but beyond the future limit.
                        sample(100_060_000_000, "base_scan"),
                        # Latest received and current; within skew tolerance.
                        sample(99_980_000_000, "base_scan"),
                    ),
                ),
                99_980_000_000,
            ),
        )

        for selected_config, observed, expected_scan_ns in cases:
            with self.subTest(expected_scan_ns=expected_scan_ns):
                result = evaluate_sensor_timing_readiness(
                    selected_config,
                    observed,
                )
                self.assertTrue(result.ready, result.detail)
                self.assertEqual(result.selected_scan_stamp_ns, expected_scan_ns)
                self.assertAlmostEqual(result.image_scan_skew_sec, 0.06)

    def test_collection_effect_is_injectable_and_reuses_pure_evaluator(self):
        calls = []

        def effect(selected):
            calls.append(selected)
            return evidence()

        selected = config()
        result = observe_sensor_timing_readiness(
            selected,
            sensor_timing_effect=effect,
        )

        self.assertTrue(result.ready)
        self.assertEqual(calls, [selected])

    def test_effect_exception_is_a_persistable_fail_closed_result(self):
        def effect(_selected):
            raise RuntimeError("collector broke")

        result = observe_sensor_timing_readiness(
            config(),
            sensor_timing_effect=effect,
        )

        self.assertFalse(result.ready)
        self.assertEqual(result.failure_code, FAILURE_OBSERVATION_EFFECT)
        self.assertIn("collector broke", result.detail)
        json.dumps(result.to_dict(), sort_keys=True)

    def test_typed_error_preserves_pre_or_post_run_failure_boundary(self):
        rejected = evaluate_sensor_timing_readiness(
            config(), evidence(image_samples=())
        )
        error = SensorTimingReadinessError(
            rejected,
            evidence_path="preflight/sensor_timing.json",
            evidence_sha256="b" * 64,
            phase="candidate_leg_before_motion",
            typed_run_already_issued=True,
        )

        fields = error.to_failure_fields()
        self.assertEqual(fields["failure_phase"], "candidate_leg_before_motion")
        self.assertTrue(fields["typed_run_already_issued"])
        self.assertTrue(fields["typed_run_requested"])
        self.assertFalse(fields["motion_authorized"])
        self.assertFalse(fields["motion_published"])
        self.assertEqual(fields["sensor_timing_readiness_sha256"], "b" * 64)

    def test_runner_adapter_persists_rejection_before_raising_typed_error(self):
        rejected = evaluate_sensor_timing_readiness(
            config(), evidence(image_samples=())
        )
        runtime = SimpleNamespace(scan_topic="/scan")
        profile = SimpleNamespace(
            resolved_compressed_image_topic="/camera/image_raw/compressed",
            resolved_camera_info_topic="/camera/camera_info",
            camera_optical_frame="camera",
            scan_frame="base_scan",
            resolved_runtime=lambda: runtime,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight" / "camera_lidar_timing.json"
            with patch.object(
                runner,
                "observe_sensor_timing_readiness",
                return_value=rejected,
            ):
                with self.assertRaises(SensorTimingReadinessError) as captured:
                    runner._admit_sensor_timing_readiness(
                        profile,
                        path,
                        phase="preauthorization_sensor_timing_readiness",
                    )

            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["ready"])
            self.assertEqual(
                payload["phase"],
                "preauthorization_sensor_timing_readiness",
            )
            self.assertFalse(payload["typed_run_already_issued"])
            self.assertFalse(payload["motion_published"])
            self.assertEqual(
                payload["sensor_timing_readiness_sha256"],
                captured.exception.evidence_sha256,
            )
            self.assertEqual(captured.exception.evidence_path, str(path))
            loaded = load_content_hashed_json(
                path,
                hash_field="sensor_timing_readiness_sha256",
            )
            self.assertFalse(loaded["ready"])

    def test_config_rejects_invalid_names_limits_and_capacity(self):
        invalid = (
            config(image_topic=""),
            config(camera_info_topic=" /camera/camera_info"),
            config(scan_topic=""),
            config(expected_image_frame=""),
            config(expected_camera_info_frame="camera "),
            config(expected_scan_frame=""),
            config(timeout_sec=0.0),
            config(timeout_sec=math.inf),
            config(max_image_age_sec=-1.0),
            config(max_camera_info_age_sec=0.0),
            config(max_scan_age_sec=math.nan),
            config(max_future_timestamp_sec=-1.0),
            config(max_image_scan_skew_sec=-1.0),
            config(max_camera_info_image_skew_sec=math.inf),
            config(poll_interval_sec=3.0),
            config(sample_capacity=0),
            config(sample_capacity=True),
        )
        for selected in invalid:
            with self.subTest(selected=selected):
                with self.assertRaises(ValueError):
                    selected.validated()


if __name__ == "__main__":
    unittest.main()
