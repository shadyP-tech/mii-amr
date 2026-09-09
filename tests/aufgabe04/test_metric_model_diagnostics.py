from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_stand_model
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.model_refinement import RefinedHeadMeasurement
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import (
    metric_corner_arm_support,
    observed_metric_corner_arms,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics import (
    CORNER_NAMES,
    collect_metric_model_diagnostics,
    rectified_head_qr_ratios,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
    QrQuadDetection,
    RectifiedCameraMatrix,
    estimate_planar_pose_ippe,
)

PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline."
DIAGNOSTICS = "scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics."


def pose_at(yaw: float, *, x_m: float = 0.0) -> PlanarPoseHypothesis:
    angle = math.radians(yaw)
    return PlanarPoseHypothesis(
        rotation_vector=(0.0, angle, 0.0),
        translation_xyz_m=(x_m, 0.0, 0.40),
        face_normal_xyz=(math.sin(angle), 0.0, math.cos(angle)),
        yaw_deg=-yaw,
        reprojection_rmse_px=0.0,
        positive_depth=True,
    )


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV and numpy are required")
class MetricModelDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_stand_model(
            Path(__file__).resolve().parents[2]
            / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
        )
        self.camera = RectifiedCameraMatrix(800.0, 800.0, 400.0, 300.0)
        self.pose = pose_at(25.0)
        self.projected = project_stand_model(cv2, self.profile, self.pose, self.camera)
        self.qr_corners = tuple(
            self.projected.landmarks[f"qr_{suffix}"] for suffix in CORNER_NAMES
        )
        self.frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        self.options = dict(
            model_profile=self.profile,
            camera_fx_px=self.camera.fx_px,
            camera_fy_px=self.camera.fy_px,
            camera_cx_px=self.camera.cx_px,
            camera_cy_px=self.camera.cy_px,
        )

    def _refinement(self, corners):
        return RefinedHeadMeasurement(
            True, "model_current_frame_border_refined", corners,
            numpy.zeros(self.frame.shape[:2], dtype=numpy.uint8), None,
        )

    def test_overlay_uses_refined_pose_and_keeps_seed_corners_separate(self):
        # Explicit stem geometry makes this cover every 3D overlay component.
        profile = replace(self.profile, stem_width_m=0.010, stem_visible_height_m=0.080)
        seed = pose_at(21.0, x_m=0.001)
        seed_projection = project_stand_model(cv2, profile, seed, self.camera)
        measured = project_stand_model(cv2, profile, self.pose, self.camera)
        with (
            patch(PIPELINE + "detect_qr_quad", return_value=None),
            patch(PIPELINE + "refine_projected_head_border",
                  return_value=self._refinement(measured.head_corners)),
            patch(DIAGNOSTICS + "estimate_planar_pose_ippe",
                  side_effect=AssertionError("accepted head pose must be reused")),
        ):
            estimate, artifacts = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **{**self.options, "model_profile": profile}, pose_hint=seed
            )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(artifacts.predicted_corners, seed_projection.head_corners)
        self.assertEqual(artifacts.refined_corners, measured.head_corners)
        actual = project_stand_model(cv2, profile, artifacts.model_pose, self.camera)
        self.assertEqual(artifacts.projected_landmarks, dict(actual.landmarks))
        self.assertNotEqual(artifacts.projected_landmarks, dict(seed_projection.landmarks))
        self.assertIn("stem_bottom_left", artifacts.projected_landmarks)
        self.assertLess(artifacts.model_diagnostics.group_reprojection_rmse_px["head"], 1e-5)
        self.assertIsNotNone(artifacts.model_diagnostics.head_only)
        self.assertIsNone(artifacts.model_diagnostics.qr_only)

    def test_rejected_paper_border_keeps_group_residuals_without_head_only_fallback(self):
        paper = replace(self.profile, head_width_m=0.071, head_height_m=0.071)
        paper_corners = project_stand_model(cv2, paper, self.pose, self.camera).head_corners
        with (
            patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr_corners, 1.0)),
            patch(PIPELINE + "refine_projected_head_border",
                  return_value=self._refinement(paper_corners)),
            patch(DIAGNOSTICS + "estimate_planar_pose_ippe", wraps=estimate_planar_pose_ippe) as head_fit,
        ):
            estimate, artifacts = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **self.options
            )
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "reprojection_error_too_high")
        self.assertEqual(artifacts.evidence_state, "ambiguous")
        self.assertEqual(artifacts.model_pose_fit_source, "joint_qr_head")
        self.assertEqual(artifacts.refined_corners, paper_corners)
        self.assertEqual(head_fit.call_count, 1)
        diagnostics = artifacts.model_diagnostics
        self.assertEqual(set(diagnostics.group_reprojection_rmse_px), {"head", "qr"})
        self.assertEqual(len(diagnostics.corner_reprojection_error_px), 8)
        self.assertGreater(diagnostics.max_corner_error_px, 2.0)
        self.assertTrue(diagnostics.head_only.accepted)
        self.assertLess(diagnostics.head_only.reprojection_rmse_px, 1e-5)
        self.assertTrue(diagnostics.qr_only.accepted)
        self.assertAlmostEqual(diagnostics.observed_head_qr_width_ratio, 71 / 62, places=5)
        self.assertAlmostEqual(diagnostics.observed_head_qr_height_ratio, 71 / 62, places=5)
        # A good independent head pose remains diagnostic-only.
        self.assertAlmostEqual(artifacts.model_pose.translation_xyz_m[2], 0.4, places=5)

    def test_correct_joint_fit_reuses_qr_solve_and_does_not_add_head_solve(self):
        with (
            patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr_corners, 1.0)),
            patch(PIPELINE + "refine_projected_head_border",
                  return_value=self._refinement(self.projected.head_corners)),
            patch(DIAGNOSTICS + "estimate_planar_pose_ippe",
                  side_effect=AssertionError("accepted joint fit needs no extra solve")),
        ):
            estimate, artifacts = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **self.options
            )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertLess(artifacts.model_diagnostics.max_corner_error_px, 1e-5)
        self.assertIsNone(artifacts.model_diagnostics.head_only)
        self.assertIsNotNone(artifacts.model_diagnostics.qr_only)
        stages = artifacts.stage_timings_ms
        for name in ("qr_detection", "qr_seed_pose", "border_refinement", "pose_fit", "diagnostics"):
            self.assertIn(name, stages)
            self.assertGreaterEqual(stages[name], 0.0)
        self.assertGreaterEqual(stages["total"], sum(value for name, value in stages.items() if name != "total"))

    def test_semantic_corner_errors_identify_the_disagreeing_corner(self):
        perturbed = list(self.projected.head_corners)
        perturbed[1] = ImagePoint(perturbed[1].u_px + 3.0, perturbed[1].v_px + 4.0)
        diagnostics = collect_metric_model_diagnostics(
            cv2, profile=self.profile, camera=self.camera,
            head_corners=tuple(perturbed), qr_corners=self.qr_corners,
            diagnostic_pose=self.pose, qr_pose=None,
        )
        self.assertAlmostEqual(diagnostics.corner_reprojection_error_px["head_top_right"], 5.0)
        self.assertAlmostEqual(diagnostics.group_reprojection_rmse_px["head"], 2.5)
        self.assertAlmostEqual(diagnostics.group_reprojection_rmse_px["qr"], 0.0)
        self.assertAlmostEqual(diagnostics.max_corner_error_px, 5.0)

    def test_rectified_ratios_distinguish_head_and_paper_under_perspective(self):
        for angle in (0.0, 25.0, 55.0):
            for border_m in (0.078, 0.071):
                with self.subTest(angle=angle, border_m=border_m):
                    projected = project_stand_model(cv2, self.profile, pose_at(angle), self.camera)
                    qr = tuple(projected.landmarks[f"qr_{name}"] for name in CORNER_NAMES)
                    border = project_stand_model(
                        cv2, replace(self.profile, head_width_m=border_m, head_height_m=border_m),
                        pose_at(angle), self.camera,
                    )
                    width, height = rectified_head_qr_ratios(cv2, border.head_corners, qr, self.profile)
                    self.assertAlmostEqual(width, border_m / 0.062, places=5)
                    self.assertAlmostEqual(height, border_m / 0.062, places=5)

    def test_degenerate_ratios_are_unavailable_instead_of_infinite(self):
        invalid_quads = (
            None,
            (ImagePoint(10.0, 10.0),) * 4,
            (ImagePoint(float("nan"), 10.0),) * 4,
            tuple(self.qr_corners[index] for index in (0, 2, 1, 3)),
        )
        for qr in invalid_quads:
            with self.subTest(qr=qr):
                self.assertEqual(
                    rectified_head_qr_ratios(cv2, self.projected.head_corners, qr, self.profile),
                    (None, None),
                )

    def test_failed_refinement_has_qr_diagnostics_without_inventing_head_measurements(self):
        with patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr_corners, 1.0)):
            estimate, artifacts = estimate_stand_axis_from_metric_model(cv2, self.frame, **self.options)
        self.assertFalse(estimate.usable)
        self.assertEqual(artifacts.evidence_state, "predicted_only")
        self.assertIsNone(artifacts.refined_corners)
        self.assertEqual(set(artifacts.model_diagnostics.group_reprojection_rmse_px), {"qr"})
        self.assertIsNone(artifacts.model_diagnostics.observed_head_qr_width_ratio)
        self.assertNotIn("pose_fit", artifacts.stage_timings_ms)

    def test_seed_unavailable_still_reports_completed_detection_stages(self):
        with patch(PIPELINE + "detect_qr_quad", return_value=None):
            estimate, artifacts = estimate_stand_axis_from_metric_model(cv2, self.frame, **self.options)
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "model_pose_seed_unavailable")
        self.assertIn("qr_detection", artifacts.stage_timings_ms)
        self.assertNotIn("pose_fit", artifacts.stage_timings_ms)

    def test_corner_gate_identifies_the_single_missing_incident_arm(self):
        corners = tuple(ImagePoint(x, y) for x, y in ((80., 60.), (200., 60.), (200., 180.), (80., 180.)))
        edges = numpy.zeros((260, 300), dtype=numpy.uint8)
        # The top-left's top arm stops 15 px short; all seven other arms
        # remain observed, as do the middle sections used for side fitting.
        for start, end in (((95, 60), (200, 60)), ((200, 60), (200, 180)),
                           ((200, 180), (80, 180)), ((80, 180), (80, 60))):
            cv2.line(edges, start, end, 255, 1)
        original = edges.copy()
        with patch(
            "scripts.aufgabe04.perception.stand_axis.model_refinement.metric_corner_arm_support",
            wraps=metric_corner_arm_support,
        ) as corner_check:
            result = refine_projected_head_border(cv2, edges, corners, corridor_half_width_px=8.0)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "model_corner_evidence_insufficient")
        self.assertIsNone(result.corners)
        for candidate, expected in zip(result.candidate_corners, corners):
            self.assertAlmostEqual(candidate.u_px, expected.u_px, places=4)
            self.assertAlmostEqual(candidate.v_px, expected.v_px, places=4)
        self.assertEqual(corner_check.call_count, 1)
        self.assertFalse(result.corner_arm_support.accepted)
        self.assertEqual(result.corner_arm_support.minimum_bins, 2)
        self.assertEqual(result.corner_arm_support.radius_px, 6.0)
        for corner_name, arms in result.corner_arm_support.bins_by_corner.items():
            for arm_name, count in arms.items():
                if (corner_name, arm_name) == ("head_top_left", "top"):
                    self.assertEqual(count, 0)
                else:
                    self.assertGreaterEqual(count, 2)
        numpy.testing.assert_array_equal(edges, original)

    def test_corner_counts_keep_the_original_two_bin_threshold(self):
        corners = tuple(ImagePoint(x, y) for x, y in ((80., 60.), (200., 60.), (200., 180.), (80., 180.)))
        full = numpy.zeros((260, 300), dtype=numpy.uint8)
        cv2.rectangle(full, (80, 60), (200, 180), 255, 1)
        full[52:69, 72:89] = 0
        full[62:64, 80] = 255  # two distinct longitudinal bins on the left arm
        for top_bins in (1, 2):
            with self.subTest(top_bins=top_bins):
                edges = full.copy()
                edges[60, 82:82 + top_bins] = 255
                support = metric_corner_arm_support(cv2, edges, corners)
                self.assertEqual(support.bins_by_corner["head_top_left"]["top"], top_bins)
                self.assertEqual(support.bins_by_corner["head_top_left"]["left"], 2)
                self.assertEqual(support.accepted, top_bins == 2)
                self.assertEqual(observed_metric_corner_arms(cv2, edges, corners), support.accepted)

    def test_rejected_corner_candidate_and_support_reach_pipeline_artifacts(self):
        edges = numpy.zeros(self.frame.shape[:2], dtype=numpy.uint8)
        corners = self.projected.head_corners
        for first, second in zip(corners, corners[1:] + corners[:1]):
            start = numpy.array((first.u_px, first.v_px))
            finish = numpy.array((second.u_px, second.v_px))
            direction = (finish - start) / numpy.linalg.norm(finish - start)
            cv2.line(edges, tuple(numpy.rint(start + 15 * direction).astype(int)),
                     tuple(numpy.rint(finish - 15 * direction).astype(int)), 255, 1)
        with (
            patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr_corners, 1.0)),
            patch(PIPELINE + "_canny_edges_from_frame", return_value=edges),
        ):
            estimate, artifacts = estimate_stand_axis_from_metric_model(cv2, self.frame, **self.options)
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "model_corner_evidence_insufficient")
        self.assertIsNone(artifacts.refined_corners)
        self.assertIsNotNone(artifacts.candidate_corners)
        self.assertEqual(len(artifacts.candidate_corners), 4)
        self.assertFalse(artifacts.corner_arm_support.accepted)
        self.assertEqual(len(artifacts.corner_arm_support.bins_by_corner), 4)
        self.assertNotIn("pose_fit", artifacts.stage_timings_ms)


if __name__ == "__main__":
    unittest.main()
