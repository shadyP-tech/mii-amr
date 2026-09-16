"""Independent angle authority survives QR changes but never lost raw proof."""

from dataclasses import replace
import unittest
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = numpy = None

from scripts.aufgabe04.perception.stand_axis.head_model_quality import validated_head_model_quality
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import estimate_planar_pose_ippe
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from tests.aufgabe04 import test_geometry_contract as geometry_fixture

PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline."
HEAD_FIT = "scripts.aufgabe04.perception.stand_axis.head_model_fit."


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV/numpy required")
class IndependentHeadFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geometry_fixture.GeometryContractTest.setUpClass()
        cls.fixture = geometry_fixture.GeometryContractTest

    def run_fit(self, **overrides):
        options = {**self.fixture.options, **overrides}
        return estimate_stand_axis_from_metric_model(cv2, self.fixture.crop, **options)

    def assert_same_head(self, estimate, debug):
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.source, "model_current_measured_head")
        self.assertTrue(validated_head_model_quality(debug.head_model_quality))
        self.assertEqual(estimate.corners, self.fixture.estimate.corners)
        self.assertAlmostEqual(estimate.yaw_deg, self.fixture.estimate.yaw_deg, places=9)
        self.assertEqual(estimate.camera_face_normal_xyz, self.fixture.estimate.camera_face_normal_xyz)
        self.assertEqual(debug.model_pose, self.fixture.debug.model_pose)
        self.assertIsNone(estimate.visible_face)

    def test_qr_payload_geometry_and_model_dimensions_cannot_change_head_angle(self):
        cases = (
            {"qr_observations": ()},
            {"qr_observations": (DecodedQrObservation("different_identity", None, "test"),)},
            {"qr_observations": (DecodedQrObservation("QR_003", None, "test"),
                                 DecodedQrObservation("conflict", None, "test"))},
            {"model_profile": replace(self.fixture.profile, qr_symbol_width_m=0.071,
                                      qr_symbol_height_m=0.071, qr_center_x_m=0.005)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), patch(
                PIPELINE + "select_temporally_consistent_pose",
                side_effect=AssertionError("head cannot borrow QR ambiguity resolution"),
            ):
                self.assert_same_head(*self.run_fit(**overrides))
        self.assertEqual(self.fixture.profile.qr_symbol_width_m, 0.062)

    def test_tracked_pose_cannot_donate_angle_or_resolve_current_head(self):
        wrong_hint = replace(self.fixture.debug.model_pose, yaw_deg=-70.0,
                             rotation_vector=(0.0, 1.2, 0.0), translation_xyz_m=(0.3, 0.2, 0.9))
        with patch(PIPELINE + "select_temporally_consistent_pose", side_effect=AssertionError("no history tie-break")):
            self.assert_same_head(*self.run_fit(pose_hint=wrong_hint))

    def test_only_current_head_points_are_solved_without_qr_or_joint_diagnostics(self):
        calls = []

        def solve(cv, points, model, camera, **kwargs):
            calls.append(len(points))
            self.assertEqual(tuple(model), self.fixture.profile.head_corners)
            return estimate_planar_pose_ippe(cv, points, model, camera, **kwargs)

        with (
            patch(HEAD_FIT + "estimate_planar_pose_ippe", side_effect=solve),
            patch(PIPELINE + "estimate_planar_pose_ippe", side_effect=AssertionError("QR/joint pose forbidden")),
            patch(PIPELINE + "collect_metric_model_diagnostics", side_effect=AssertionError("QR geometry forbidden")),
            patch(PIPELINE + "classify_joint_geometry_contract", side_effect=AssertionError("joint agreement forbidden")),
        ):
            estimate, debug = self.run_fit()
            self.assert_same_head(estimate, debug)
            self.assertIsNone(debug.model_diagnostics)
            self.assertIsNone(debug.head_marker_boundary)
        self.assertEqual(calls, [4])

    def test_current_crop_preserves_unresolved_physical_borders_before_any_qr_or_pose(self):
        with patch(PIPELINE + "detect_qr_quad", return_value=None):
            estimate, debug = self.run_fit(current_head_proposal_corners=None,
                                           qr_observations=(), pose_hint=None)
        # This crop contains complete current alternatives at left x≈6 and
        # x≈13. The former size-based grouping silently merged them. Without a
        # resolved physical boundary, QR absence cannot select one as backside.
        acquisition = debug.head_acquisition_diagnostics["acquisition"]
        self.assertEqual(acquisition["reason"], "head_cold_acquisition_verification_budget_exceeded")
        diagnostics = acquisition["joint_border_diagnostics"]
        self.assertGreater(diagnostics["unverified_independent_hypotheses"], 0)
        # Canonical refinement may move an inset hint onto enclosing rails.
        # The invariant is unresolved current families, not their old spacing.
        from scripts.aufgabe04.perception.stand_axis.head_border_families import CurrentBorderFamilies
        frames = [tuple(ImagePoint(*p) for p in item["corners"])
                  for item in diagnostics["strict_verifications"] if item["accepted"]]
        families = CurrentBorderFamilies(debug.raw_edges, self.fixture.crop)
        self.assertTrue(any(not families.same(a, b)
                            for i, a in enumerate(frames) for b in frames[i+1:]))
        self.assertIsNone(debug.refined_corners)
        self.assertFalse(estimate.usable)
        self.assertIsNone(debug.head_neck_junction)
        self.assertIsNone(debug.model_pose)
        self.assertIsNone(estimate.visible_face)

    def test_neck_pixels_cannot_determine_current_head_fit(self):
        pixels = self.fixture.crop.copy()
        bottom = int(max(p.v_px for p in self.fixture.debug.refined_corners)) + 2
        # Continue the actual background; do not draw a new artificial bottom
        # edge through the physical head while removing its neck.
        pixels[bottom:, :] = pixels[bottom:, :1]
        estimate, debug = estimate_stand_axis_from_metric_model(
            cv2, pixels, **self.fixture.options, pose_hint=self.fixture.debug.model_pose,
        )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertAlmostEqual(estimate.yaw_deg, self.fixture.estimate.yaw_deg, places=5)
        self.assertIsNone(debug.head_neck_junction)
        self.assertTrue(debug.head_model_quality.outer_border_verified)


if __name__ == "__main__":
    unittest.main()
