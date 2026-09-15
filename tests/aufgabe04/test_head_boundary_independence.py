"""Raw outer-frame proof remains independent of QR size and identity."""

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_outer_border import (
    check_current_head_marker_boundary, current_head_boundary_eligible,
    select_current_outer_head_border,
    validated_current_head_boundary,
)
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class HeadBoundaryIndependenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")

    def projection(self, scale=1., *, angle_deg=45., distance_m=.35):
        camera = RectifiedCameraMatrix(640., 640., 400., 300.)
        matrix = np.array(((640., 0., 400.), (0., 640., 300.), (0., 0., 1.)))
        points = np.array([(p.x_m, p.y_m, p.z_m) for p in self.profile.head_corners]) * scale
        pixels = cv2.projectPoints(points, np.array((0., math.radians(angle_deg), 0.)),
            np.array((0., 0., distance_m)), matrix, np.zeros(4))[0].reshape(-1, 2)
        return tuple(ImagePoint(float(x), float(y)) for x, y in pixels), camera

    def raw(self, *quads):
        raw = np.zeros((600, 800), np.uint8)
        for corners in quads:
            cv2.polylines(raw, [np.rint([(p.u_px, p.v_px) for p in corners]).astype(np.int32)],
                          True, 255, 1)
        return raw

    def collapsed(self, complete=True):
        outer = tuple(ImagePoint(x, y) for x, y in ((50, 30), (170, 30), (170, 140), (50, 140)))
        inner = tuple(ImagePoint(x, y) for x, y in ((59, 30), (170, 30), (170, 131), (59, 131)))
        raw = self.raw(inner, *((outer,) if complete else ()))
        initial = refine_projected_head_border(cv2, raw, inner, corridor_half_width_px=2.)
        self.assertTrue(initial.accepted, initial.reason)
        recovered, evidence = select_current_outer_head_border(
            cv2, raw, model_profile=self.profile, refinement=initial,
            corridor_half_width_px=6., neutral_proposal_corners=outer)
        return outer, inner, recovered, evidence

    def test_original_current_proposal_recovers_nine_pixel_inset_switch(self):
        outer, _inner, recovered, evidence = self.collapsed()
        self.assertGreater(evidence.proposal_inward_shift_px, 8.)
        self.assertTrue(evidence.accepted, evidence.reason)
        self.assertTrue(evidence.independently_resolved)
        self.assertLessEqual(evidence.attempted_raw_refinements, 3)
        for expected, actual in zip(outer, recovered.corners):
            self.assertLess(math.hypot(expected.u_px-actual.u_px, expected.v_px-actual.v_px), 1.)

    def test_missing_original_outer_pixels_do_not_authorize_inner_rectangle(self):
        _outer, _inner, recovered, evidence = self.collapsed(complete=False)
        self.assertTrue(recovered.accepted)  # A raw paper rectangle really exists.
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.reason, "current_physical_head_boundary_unresolved")
        self.assertFalse(validated_current_head_boundary(evidence, corners=recovered.corners,
                                                        profile_sha256=self.profile.sha256))

    def test_recovered_outer_angle_is_identical_under_qr_ratio_disagreement(self):
        outer, camera = self.projection()
        inner, _ = self.projection(.071/.078)
        raw = self.raw(outer, inner)
        image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        results = []
        for qr_scale in (None, .062/.078, .068/.078):
            observations = ()
            if qr_scale is not None:
                qr, _ = self.projection(qr_scale)
                observations = (DecodedQrObservation(
                    "QR_003", tuple((p.u_px, p.v_px) for p in qr), "fixture"),)
            with patch("scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame",
                       return_value=raw):
                result = estimate_stand_axis_from_metric_model(cv2, image,
                    model_profile=self.profile, camera_fx_px=camera.fx_px,
                    camera_fy_px=camera.fy_px, camera_cx_px=camera.cx_px,
                    camera_cy_px=camera.cy_px, current_head_proposal_corners=inner,
                    qr_observations=observations)
            self.assertTrue(result[0].usable, result[0].reason)
            self.assertTrue(current_head_boundary_eligible(*result))
            self.assertIsNone(result[0].visible_face)  # This test supplies no candidate side proof.
            self.assertEqual(result[1].qr_marker_verified, qr_scale is not None)
            self.assertIsNone(result[1].head_marker_boundary)  # No QR geometry runs on this path.
            results.append(result)
        for result in results[1:]:
            self.assertEqual(results[0][0].corners, result[0].corners)
            self.assertEqual(results[0][0].yaw_deg, result[0].yaw_deg)

    def test_qr_profile_dimensions_cannot_change_current_border_search_or_angle(self):
        outer, camera = self.projection()
        inner, _ = self.projection(.071/.078)
        raw = self.raw(outer, inner)
        baseline, baseline_debug, _ = fit_current_measured_head(
            cv2, raw, model_profile=self.profile, camera=camera, proposal_corners=inner)
        self.assertTrue(baseline.usable, baseline.reason)
        for fields in (
            dict(qr_symbol_width_m=.02, qr_symbol_height_m=.03,
                 qr_panel_width_m=.025, qr_panel_height_m=.04),
            dict(qr_symbol_width_m=.075, qr_symbol_height_m=.074,
                 qr_panel_width_m=None, qr_panel_height_m=None),
            dict(qr_center_x_m=.03, qr_center_y_m=-.03),
        ):
            with self.subTest(fields=fields):
                profile = replace(self.profile, **fields)
                current, debug, _ = fit_current_measured_head(
                    cv2, raw, model_profile=profile, camera=camera, proposal_corners=inner)
                self.assertTrue(current.usable, current.reason)
                self.assertEqual(current.corners, baseline.corners)
                self.assertEqual(current.yaw_deg, baseline.yaw_deg)
                self.assertEqual(debug.head_outer_recovery.attempted_growth_factors,
                                 baseline_debug.head_outer_recovery.attempted_growth_factors)
                self.assertEqual(debug.head_outer_recovery.current_raw_alternatives,
                                 baseline_debug.head_outer_recovery.current_raw_alternatives)

    def test_lone_quad_scale_remains_conditional_on_candidate_not_qr(self):
        # A single raw rectangle has no independent physical-scale reference.
        # The pipeline must leave candidate/model association to the observer;
        # a decoded marker's measured size can no longer decide this angle.
        paper, camera = self.projection(.071/.078)
        qr, _ = self.projection(.062/.078)
        estimate, debug, _ = fit_current_measured_head(cv2, self.raw(paper),
            model_profile=self.profile, camera=camera, proposal_corners=paper)
        self.assertTrue(estimate.usable)
        boundary = check_current_head_marker_boundary(cv2, head_corners=estimate.corners,
            qr_corners=qr, marker_verified=True, model_profile=self.profile)
        self.assertFalse(boundary.accepted)
        self.assertTrue(boundary.diagnostic_only)
        self.assertFalse(boundary.requests_reconsideration)
        self.assertTrue(current_head_boundary_eligible(estimate,
            replace(debug, head_marker_boundary=boundary)))

        raw = self.raw(paper)
        observation = DecodedQrObservation("QR_003", tuple((p.u_px, p.v_px) for p in qr), "fixture")
        with patch("scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame",
                   return_value=raw):
            result, artifacts = estimate_stand_axis_from_metric_model(
                cv2, cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), model_profile=self.profile,
                camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
                camera_cx_px=camera.cx_px, camera_cy_px=camera.cy_px,
                current_head_proposal_corners=paper, qr_observations=(observation,))
        self.assertTrue(result.usable, result.reason)
        self.assertEqual(result.corners, estimate.corners)
        self.assertEqual(result.yaw_deg, estimate.yaw_deg)
        self.assertTrue(artifacts.head_model_quality.outer_border_verified)
        self.assertTrue(current_head_boundary_eligible(result, artifacts))
        self.assertTrue(artifacts.qr_marker_verified)  # Identity/front evidence is retained.

    def test_independent_flag_cannot_replace_bound_raw_recovery(self):
        _outer, _inner, recovered, evidence = self.collapsed()
        check = lambda proof: validated_current_head_boundary(proof,
            corners=recovered.corners, profile_sha256=self.profile.sha256, require_independent=True)
        self.assertTrue(check(evidence))
        for corrupted in (replace(evidence, original_corners=evidence.recovered_corners),
                          replace(evidence, recovered=False),
                          replace(evidence, profile_sha256="other"),
                          replace(evidence, current_raw_alternatives=()),
                          replace(evidence, neutral_proposal_corners=(ImagePoint(float("nan"), 0.),)*4),
                          replace(evidence, recovered_corners=evidence.original_corners)):
            self.assertFalse(check(corrupted))

    def test_marker_diagnostic_cannot_modify_angle_gate_across_stroke_widths(self):
        for angle in (20., 35., 45., 60.):
            for distance in (.25, .35, .5, .7):
                paper, camera = self.projection(.071/.078, angle_deg=angle, distance_m=distance)
                qr, _ = self.projection(.062/.078, angle_deg=angle, distance_m=distance)
                for thickness in (1, 2, 3, 4, 5, 6):
                    with self.subTest(angle=angle, distance=distance, thickness=thickness):
                        raw = np.zeros((600, 800), np.uint8)
                        cv2.polylines(raw, [np.rint([(p.u_px, p.v_px) for p in paper]).astype(np.int32)],
                                      True, 255, thickness)
                        estimate, debug, _ = fit_current_measured_head(cv2, raw,
                            model_profile=self.profile, camera=camera, proposal_corners=paper)
                        boundary = check_current_head_marker_boundary(cv2,
                            head_corners=estimate.corners, qr_corners=qr, marker_verified=True,
                            model_profile=self.profile)
                        self.assertFalse(boundary.requests_reconsideration)
                        self.assertEqual(current_head_boundary_eligible(estimate, debug),
                            current_head_boundary_eligible(estimate,
                                replace(debug, head_marker_boundary=boundary)))


if __name__ == "__main__":
    unittest.main()
