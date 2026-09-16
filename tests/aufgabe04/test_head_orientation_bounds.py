"""Ambiguous current pixels remain a bounded detection, never a precise axis."""

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = numpy = None

from scripts.aufgabe04.perception.stand_axis.head_model_quality import evaluate_head_model_quality
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import (
    HeadOrientationHypothesis, enclosing_axial_interval,
    evaluate_current_head_orientation_bounds, validated_current_head_orientation_bounds,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix, estimate_planar_pose_ippe

MODULE = "scripts.aufgabe04.perception.stand_axis.head_orientation_bounds."
REPO = Path(__file__).resolve().parents[2]


def hypothesis(yaw, std):
    return HeadOrientationHypothesis(math.radians(yaw), math.radians(std), 0.1,
                                     .75, 10., True, (0., 0., 0.), (0., 0., .5), (0., 0., 1.))


class AxialIntervalTest(unittest.TestCase):
    def test_noise_expansion_encloses_each_alternative(self):
        center, half = enclosing_axial_interval((hypothesis(5., 1.), hypothesis(10., 2.)))
        self.assertAlmostEqual(math.degrees(center - half), 2.)
        self.assertAlmostEqual(math.degrees(center + half), 16.)

    def test_wrap_is_short_connected_axial_hull(self):
        center, half = enclosing_axial_interval((hypothesis(88., 1.), hypothesis(-88., 1.)))
        self.assertAlmostEqual(abs(math.degrees(center)), 90.)
        self.assertAlmostEqual(math.degrees(half), 5.)

    def test_expanded_alternatives_covering_circle_are_unbounded(self):
        self.assertIsNone(enclosing_axial_interval((hypothesis(0., 20.), hypothesis(90., 20.))))
        self.assertIsNone(enclosing_axial_interval((hypothesis(0., 30.),)))
        self.assertIsNone(enclosing_axial_interval((hypothesis(0., math.nan),)))


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV/numpy required")
class CurrentHeadOrientationBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(
            REPO / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
        cls.camera = RectifiedCameraMatrix(600., 600., 400., 300.)
        pixels = cv2.projectPoints(
            numpy.asarray([(p.x_m, p.y_m, p.z_m) for p in cls.profile.head_corners]),
            numpy.asarray((0., math.radians(5.), 0.)), numpy.asarray((0., 0., .5)),
            numpy.asarray(((600., 0., 400.), (0., 600., 300.), (0., 0., 1.))), None,
        )[0].reshape(-1, 2)
        cls.corners = tuple(ImagePoint(float(u), float(v)) for u, v in pixels)
        cls.pose = estimate_planar_pose_ippe(cv2, cls.corners, cls.profile.head_corners, cls.camera)

    def bounds(self, **overrides):
        args = dict(profile=self.profile, camera=self.camera, corners=self.corners,
                    pose_result=self.pose, raw_border_support_mean=.9,
                    raw_corner_support_accepted=True, outer_border_verified=True, frame_shape=(600, 800))
        return evaluate_current_head_orientation_bounds(cv2, **{**args, **overrides})

    def test_same_pixels_retain_both_poses_without_relaxing_strict_angle(self):
        quality = evaluate_head_model_quality(
            cv2, profile=self.profile, camera=self.camera, corners=self.corners,
            pose_result=self.pose, raw_border_support_mean=.9, raw_corner_support_accepted=True,
            outer_border_verified=True, centered_neck_supported=False,
        )
        self.assertFalse(quality.accepted)
        self.assertEqual(quality.reason, "head_model_planar_axis_ambiguous")
        bounds = self.bounds()
        self.assertTrue(validated_current_head_orientation_bounds(bounds), bounds.reason)
        self.assertEqual(len(bounds.hypotheses), 2)
        self.assertGreater(math.degrees(bounds.half_width_rad), 15.)
        for pose in bounds.hypotheses:
            delta = abs(math.remainder(pose.yaw_rad - bounds.center_rad, math.pi))
            self.assertLessEqual(delta + 3 * pose.yaw_std_rad, bounds.half_width_rad + 1e-12)

    def test_uncertain_alternative_cannot_be_dropped(self):
        with patch(MODULE + "_local_yaw_uncertainty", side_effect=[(1., 10., True), (None, 1e9, True)]):
            bounds = self.bounds()
        self.assertFalse(bounds.accepted)
        self.assertEqual(bounds.reason, "head_orientation_alternative_unbounded")

    def test_raw_fit_exposes_bound_without_promoting_uncertain_angle(self):
        from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
        raw_edges = numpy.zeros((600, 800), numpy.uint8)
        polygon = numpy.asarray([(round(p.u_px), round(p.v_px)) for p in self.corners], numpy.int32)
        cv2.polylines(raw_edges, [polygon], True, 255, 1)
        estimate, debug, _ = fit_current_measured_head(
            cv2, raw_edges, model_profile=self.profile, camera=self.camera, proposal_corners=self.corners)
        self.assertFalse(estimate.usable)
        self.assertFalse(debug.head_model_quality.accepted)
        self.assertIsNone(estimate.yaw_deg)
        self.assertIsNone(debug.model_pose)
        self.assertTrue(validated_current_head_orientation_bounds(
            debug.head_orientation_bounds, estimate=estimate, debug=debug))

    def test_behind_camera_corner_invalidates_plausible_alternative(self):
        with patch(MODULE + "_local_yaw_uncertainty", side_effect=[(1., 10., True), (1., 10., False)]):
            self.assertFalse(self.bounds().accepted)

    def test_raw_structure_profile_clipping_and_span_remain_required(self):
        cases = (
            dict(profile=replace(self.profile, environment="simulation")),
            dict(profile=replace(self.profile, measurement_status="provisional")),
            dict(raw_corner_support_accepted=False), dict(outer_border_verified=False),
            dict(raw_border_support_mean=.59), dict(raw_border_support_mean=math.nan),
            dict(corners=tuple(ImagePoint(p.u_px - 400., p.v_px) for p in self.corners)),
            dict(corners=(ImagePoint(0., 10.), ImagePoint(100., 10.), ImagePoint(100., 110.), ImagePoint(0., 110.))),
            dict(corners=(ImagePoint(10., 10.), ImagePoint(20., 10.), ImagePoint(20., 20.), ImagePoint(10., 20.))),
            dict(corners=(ImagePoint(math.nan, 10.),) * 4), dict(pose_result=None),
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(self.bounds(**case).accepted)

    def test_altered_interval_or_hypotheses_fail_bound_proof(self):
        bounds = self.bounds()
        self.assertFalse(validated_current_head_orientation_bounds(replace(bounds, half_width_rad=.001)))
        self.assertFalse(validated_current_head_orientation_bounds(replace(bounds, hypotheses=bounds.hypotheses[:1])))
        self.assertFalse(validated_current_head_orientation_bounds(bounds, profile_sha256="0" * 64))

    def test_real_fit_preserves_strict_result_and_binds_current_boundary(self):
        from tests.aufgabe04.test_geometry_contract import GeometryContractTest
        from scripts.aufgabe04.perception.stand_axis.model_diagnostics import metric_fit_diagnostics_payload
        GeometryContractTest.setUpClass()
        estimate, debug = GeometryContractTest.estimate, GeometryContractTest.debug
        bounds = debug.head_orientation_bounds
        self.assertTrue(estimate.usable)
        self.assertTrue(validated_current_head_orientation_bounds(bounds, estimate=estimate, debug=debug))
        self.assertTrue(metric_fit_diagnostics_payload(debug)["head_orientation_bounds"]["accepted"])
        for altered in (replace(estimate, source="model_projection"),
                        replace(estimate, evidence_state="predicted_only"),
                        replace(estimate, corners=self.corners)):
            self.assertFalse(validated_current_head_orientation_bounds(bounds, estimate=altered, debug=debug))
        self.assertFalse(validated_current_head_orientation_bounds(
            bounds, estimate=estimate, debug=replace(debug, head_outer_recovery=None)))
        self.assertFalse(validated_current_head_orientation_bounds(
            bounds, estimate=estimate, debug=replace(debug, model_pose_fit_source="model_projection")))


if __name__ == "__main__":
    unittest.main()
