from dataclasses import replace
from pathlib import Path
import unittest

from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    _metric_model_only_mode,
    _metric_model_status_payload,
    _resolved_fallback_face_to_qr_ratio,
    _select_axis_pipeline_result,
    _unavailable_target_estimate,
    _validate_runtime_args,
    build_parser,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_stand_model
from scripts.aufgabe04.perception.stand_axis.models import (
    StandAxisEdgeDebugArtifacts,
)


class StandAxisViewerHandoffTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_stand_model(
            Path("configs/aufgabe04/stand_models/physical_stand_assumptions_v1.json")
        )

    def test_parser_exposes_safe_calibrated_handoff_defaults(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--scan-topic",
                "/scan",
                "--calibrated-handoff",
            ]
        )

        _validate_runtime_args(args)
        self.assertEqual(args.camera_info_topic, "/camera/camera_info")
        self.assertEqual(args.camera_optical_frame, "camera")
        self.assertEqual(args.scan_frame, "base_scan")
        self.assertEqual(args.handoff_lidar_window_scans, 20)
        self.assertEqual(args.handoff_max_axis_difference_deg, 15.0)
        self.assertEqual(args.handoff_max_center_difference_m, 0.10)

    def test_calibrated_handoff_rejects_nonpositive_center_gate(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--scan-topic",
                "/scan",
                "--calibrated-handoff",
                "--handoff-max-center-difference-m",
                "0",
            ]
        )

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _validate_runtime_args(args)

    def test_calibrated_handoff_requires_scan_topic(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--calibrated-handoff",
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires --scan-topic"):
            _validate_runtime_args(args)

    def test_calibrated_handoff_rejects_simulation_camera(self):
        args = build_parser().parse_args(
            [
                "--sim-raw-image-topic",
                "/camera/image_raw",
                "--scan-topic",
                "/scan",
                "--calibrated-handoff",
            ]
        )

        with self.assertRaisesRegex(ValueError, "real-camera-only"):
            _validate_runtime_args(args)

    def test_model_profile_overrides_inconsistent_fallback_ratio(self):
        ratio = _resolved_fallback_face_to_qr_ratio(1.0, self.profile)

        self.assertAlmostEqual(ratio, 1.30)

    def test_real_camera_model_profile_enables_model_only_mode_by_default(self):
        args = build_parser().parse_args(
            ["--compressed-image-topic", "/camera/image_raw/compressed"]
        )

        self.assertTrue(_metric_model_only_mode(args, self.profile))

    def test_legacy_edge_fallback_is_an_explicit_diagnostic_opt_in(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--legacy-edge-fallback",
            ]
        )

        self.assertFalse(_metric_model_only_mode(args, self.profile))

    def test_metric_model_rejects_legacy_color_mask_axis_source(self):
        args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--axis-source",
                "color-mask",
                "--stand-model-profile",
                "configs/aufgabe04/stand_models/physical_stand_assumptions_v1.json",
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires --axis-source edges"):
            _validate_runtime_args(args)

    def test_model_only_selection_never_falls_through_to_legacy_axis(self):
        metric = replace(
            _unavailable_target_estimate("model_pose_seed_unavailable"),
            evidence_state="unobservable",
        )
        fallback = replace(
            _unavailable_target_estimate("legacy_candidate"),
            usable=True,
            evidence_state="fresh_refined",
        )
        metric_artifacts = StandAxisEdgeDebugArtifacts(edges=None)
        fallback_artifacts = StandAxisEdgeDebugArtifacts(edges=None)

        selected, selected_artifacts = _select_axis_pipeline_result(
            model_only=True,
            metric_estimate=metric,
            metric_artifacts=metric_artifacts,
            fallback_estimate=fallback,
            fallback_artifacts=fallback_artifacts,
        )

        self.assertIs(selected, metric)
        self.assertIs(selected_artifacts, metric_artifacts)

    def test_model_only_selection_fails_closed_without_metric_inputs(self):
        fallback = replace(
            _unavailable_target_estimate("legacy_candidate"),
            usable=True,
            evidence_state="fresh_refined",
        )

        selected, _ = _select_axis_pipeline_result(
            model_only=True,
            metric_estimate=None,
            metric_artifacts=None,
            fallback_estimate=fallback,
            fallback_artifacts=StandAxisEdgeDebugArtifacts(edges=None),
        )

        self.assertFalse(selected.usable)
        self.assertEqual(selected.reason, "metric_model_inputs_unavailable")

    def test_model_status_preserves_seed_failure(self):
        estimate = replace(
            _unavailable_target_estimate("model_pose_seed_unavailable"),
            evidence_state="unobservable",
            model_profile_sha256=self.profile.sha256,
            model_measurement_status=self.profile.measurement_status,
        )
        artifacts = StandAxisEdgeDebugArtifacts(
            edges=None,
            evidence_state="unobservable",
            model_profile_sha256=self.profile.sha256,
            pose_seed_source="none",
            model_reason=estimate.reason,
            model_measurement_status=self.profile.measurement_status,
        )

        payload = _metric_model_status_payload(
            profile=self.profile,
            inputs_ready=True,
            estimate=estimate,
            artifacts=artifacts,
        )

        self.assertTrue(payload["enabled"])
        self.assertFalse(payload["usable"])
        self.assertEqual(payload["reason"], "model_pose_seed_unavailable")
        self.assertEqual(payload["pose_seed_source"], "none")
        self.assertFalse(payload["committable"])


if __name__ == "__main__":
    unittest.main()
