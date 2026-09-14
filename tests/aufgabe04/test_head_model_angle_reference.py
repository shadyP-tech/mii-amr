"""Synthetic angle references and an original-pixel oblique recording replay.

Synthetic pinhole projections supply known angles, not hardware calibration.
The saved viewer angle is a previous estimate and is never labelled ground truth.
"""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.camera_calibration import CameraCalibration, rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.head_outer_border import check_current_head_marker_boundary
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    evaluate_head_model_quality, validated_head_model_quality,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    RectifiedCameraMatrix, estimate_planar_pose_ippe,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/aufgabe04/fixtures/head_model_reference_20260911"
MODEL = ROOT / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
REFERENCE_ANGLES_DEG = (0., 20., 35., 45., 60., 75., 89.)


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class HeadModelAngleReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(MODEL)

    def projection(self, angle_deg, *, distance_m=.35, resolution_scale=1., physical_size_scale=1.):
        camera = RectifiedCameraMatrix(
            640. * resolution_scale, 640. * resolution_scale,
            400. * resolution_scale, 300. * resolution_scale,
        )
        matrix = np.array(((camera.fx_px, 0., camera.cx_px),
                           (0., camera.fy_px, camera.cy_px), (0., 0., 1.)))
        points = np.array([(p.x_m, p.y_m, p.z_m) for p in self.profile.head_corners]) * physical_size_scale
        pixels = cv2.projectPoints(points, np.array((0., math.radians(angle_deg), 0.)),
            np.array((0., 0., distance_m)), matrix, np.zeros(4))[0].reshape(-1, 2)
        return tuple(ImagePoint(float(u), float(v)) for u, v in pixels), camera

    def quality(self, corners, camera, **evidence):
        pose = estimate_planar_pose_ippe(cv2, corners, self.profile.head_corners, camera)
        settings = dict(raw_border_support_mean=.99, raw_corner_support_accepted=True,
                        centered_neck_supported=True, neck_junction_verified=True,
                        outer_border_verified=True)
        settings.update(evidence)
        quality = evaluate_head_model_quality(cv2, profile=self.profile, camera=camera,
            corners=corners, pose_result=pose, **settings)
        return pose, quality

    def raster(self, corners, *, resolution_scale=1., include_neck=True):
        """Known projected one-pixel borders with paired centered neck rails."""
        raw = np.zeros((round(600 * resolution_scale), round(800 * resolution_scale)), np.uint8)
        pixels = np.array([(p.u_px, p.v_px) for p in corners])
        cv2.polylines(raw, [np.rint(pixels).astype(np.int32)], True, 255, 1)
        if include_neck:
            bottom_center = (pixels[2] + pixels[3]) / 2.
            width = np.linalg.norm(pixels[2] - pixels[3])
            height = (np.linalg.norm(pixels[0] - pixels[3]) + np.linalg.norm(pixels[1] - pixels[2])) / 2.
            for side in (-1, 1):
                x = round(bottom_center[0] + side * .07 * width)
                y = round(bottom_center[1])
                cv2.line(raw, (x, y), (x, round(y + .6 * height)), 255, 1)
        return raw

    def test_known_pinhole_angles_recover_without_a_35_degree_cutoff(self):
        for distance in (.35, .5, .7, 1.):
            for angle in REFERENCE_ANGLES_DEG:
                with self.subTest(distance=distance, angle=angle):
                    corners, camera = self.projection(angle, distance_m=distance)
                    pose, _quality = self.quality(corners, camera)
                    self.assertTrue(pose.accepted)
                    self.assertAlmostEqual(pose.best.yaw_deg, -angle, delta=1.e-3)
                    self.assertLess(pose.best.reprojection_rmse_px, 1.e-5)

    def test_current_rasterized_head_borders_support_angles_through_75_degrees(self):
        for angle in REFERENCE_ANGLES_DEG[:-1]:
            with self.subTest(angle=angle):
                corners, camera = self.projection(angle)
                estimate, debug, _pose = fit_current_measured_head(cv2, self.raster(corners),
                    model_profile=self.profile, camera=camera, proposal_corners=corners)
                self.assertTrue(estimate.usable, estimate.reason)
                self.assertAlmostEqual(estimate.yaw_deg, -angle, delta=2.)
                self.assertTrue(validated_head_model_quality(debug.head_model_quality))
                self.assertEqual(estimate.evidence_state, "fresh_refined")
                self.assertIsNone(estimate.visible_face)

    def test_opposite_signed_views_preserve_axial_angle_sign_without_face_labels(self):
        for angle in (-75., -45., 45., 75.):
            with self.subTest(angle=angle):
                corners, camera = self.projection(angle)
                estimate, debug, _pose = fit_current_measured_head(cv2, self.raster(corners),
                    model_profile=self.profile, camera=camera, proposal_corners=corners)
                self.assertTrue(estimate.usable, estimate.reason)
                self.assertAlmostEqual(estimate.yaw_deg, -angle, delta=2.)
                self.assertTrue(validated_head_model_quality(debug.head_model_quality))
                self.assertIsNone(estimate.visible_face)

    def test_high_angle_admission_depends_on_pixel_span_and_resolution(self):
        for distance, scale, accepted in ((.35, 1., True), (.5, 1., True),
                                          (.7, 1., False), (.35, .5, False)):
            with self.subTest(distance=distance, scale=scale):
                corners, camera = self.projection(75., distance_m=distance, resolution_scale=scale)
                pose, quality = self.quality(corners, camera)
                self.assertAlmostEqual(pose.best.yaw_deg, -75., delta=1.e-3)
                self.assertEqual(validated_head_model_quality(quality), accepted)
                if not accepted:
                    self.assertEqual(quality.reason, "head_model_pixel_span_insufficient")

    def test_near_frontal_far_view_can_be_less_observable_than_an_oblique_view(self):
        corners, camera = self.projection(0., distance_m=.5)
        _pose, frontal = self.quality(corners, camera)
        corners, camera = self.projection(60., distance_m=.5)
        _pose, oblique = self.quality(corners, camera)
        self.assertFalse(validated_head_model_quality(frontal))
        self.assertEqual(frontal.reason, "head_model_yaw_uncertainty_too_high")
        self.assertTrue(validated_head_model_quality(oblique))

    def test_planar_ambiguity_stays_rejected_despite_a_small_best_fit_error(self):
        corners, camera = self.projection(20., distance_m=1.)
        pose, quality = self.quality(corners, camera)
        self.assertLess(pose.best.reprojection_rmse_px, 1.e-5)
        self.assertEqual(quality.reason, "head_model_planar_axis_ambiguous")
        self.assertFalse(validated_head_model_quality(quality))

    def test_near_edge_on_measurement_is_not_admitted(self):
        for distance in (.35, .5, 1.):
            with self.subTest(distance=distance):
                corners, camera = self.projection(89., distance_m=distance)
                _pose, quality = self.quality(corners, camera)
                self.assertFalse(validated_head_model_quality(quality))
                estimate, _debug, _pose = fit_current_measured_head(cv2, self.raster(corners),
                    model_profile=self.profile, camera=camera, proposal_corners=corners)
                self.assertFalse(estimate.usable)
                self.assertIsNone(estimate.yaw_deg)

    def test_crop_intrinsics_preserve_angle_and_change_no_metric_dimensions(self):
        corners, camera = self.projection(60., distance_m=.5)
        full_pose, full_quality = self.quality(corners, camera)
        crop_corners = tuple(ImagePoint(p.u_px - 250., p.v_px - 150.) for p in corners)
        crop_camera = replace(camera, cx_px=camera.cx_px - 250., cy_px=camera.cy_px - 150.)
        crop_pose, crop_quality = self.quality(crop_corners, crop_camera)
        self.assertAlmostEqual(full_pose.best.yaw_deg, crop_pose.best.yaw_deg, delta=1.e-8)
        self.assertAlmostEqual(full_quality.yaw_std_deg, crop_quality.yaw_std_deg, delta=1.e-8)
        self.assertEqual(crop_quality.head_size_m, (.078, .078))
        self.assertEqual(crop_quality.profile_sha256, self.profile.sha256)

    def test_pixel_noise_does_not_create_false_precision_from_low_residual(self):
        random = np.random.default_rng(119)
        errors = []
        for angle in (35., 45., 60., 75.):
            for _ in range(10):
                corners, camera = self.projection(angle, distance_m=.5)
                noise = random.normal(0., .4, (4, 2))
                noisy = tuple(ImagePoint(p.u_px + float(n[0]), p.v_px + float(n[1]))
                              for p, n in zip(corners, noise))
                pose, quality = self.quality(noisy, camera)
                self.assertTrue(pose.accepted)
                errors.append(abs(pose.best.yaw_deg + angle))
                self.assertGreaterEqual(quality.corner_sigma_px, .75)
        self.assertLess(max(errors), 3.)

    def test_current_structure_is_required_even_for_perfect_projected_corners(self):
        corners, camera = self.projection(45.)
        for raw in (np.zeros((600, 800), np.uint8),):
            estimate, _debug, _pose = fit_current_measured_head(cv2, raw, model_profile=self.profile,
                camera=camera, proposal_corners=corners)
            self.assertFalse(estimate.usable)
            self.assertIsNone(estimate.yaw_deg)
        for evidence in ({"raw_border_support_mean": .2}, {"raw_corner_support_accepted": False},
                         {"outer_border_verified": False}):
            _pose, quality = self.quality(corners, camera, **evidence)
            self.assertFalse(validated_head_model_quality(quality))

    def test_complete_current_head_does_not_require_any_neck_pixels(self):
        for angle in (20., 45., 60., 75.):
            corners, camera = self.projection(angle)
            fit = [fit_current_measured_head(cv2, self.raster(corners, include_neck=neck),
                model_profile=self.profile, camera=camera, proposal_corners=corners)
                for neck in (False, True)]
            self.assertTrue(fit[0][0].usable, fit[0][0].reason)
            self.assertEqual(fit[0][0].corners, fit[1][0].corners)
            self.assertEqual(fit[0][0].yaw_deg, fit[1][0].yaw_deg)
            self.assertIsNone(fit[0][1].head_neck_junction)

    def test_enclosing_raw_border_and_verified_symbol_disambiguate_inner_paper(self):
        outer, camera = self.projection(45., distance_m=.35)
        inner, _camera = self.projection(45., distance_m=.35,
                                       physical_size_scale=.071 / .078)
        raw = self.raster(outer)
        cv2.polylines(raw, [np.rint([(p.u_px, p.v_px) for p in inner]).astype(np.int32)],
                      True, 255, 1)
        physical, _debug, _pose = fit_current_measured_head(cv2, raw,
            model_profile=self.profile, camera=camera, proposal_corners=outer)
        inner_seed, recovered_debug, recovered_pose = fit_current_measured_head(cv2, raw,
            model_profile=self.profile, camera=camera, proposal_corners=inner)
        self.assertTrue(physical.usable, physical.reason)
        self.assertTrue(inner_seed.usable, inner_seed.reason)
        self.assertTrue(recovered_debug.head_outer_recovery.accepted)
        self.assertTrue(recovered_debug.head_outer_recovery.recovered)
        for expected, actual in zip(outer, inner_seed.corners):
            self.assertLess(math.hypot(expected.u_px - actual.u_px,
                                       expected.v_px - actual.v_px), 1.)
        self.assertAlmostEqual(recovered_pose.best.translation_xyz_m[2], .35, delta=.003)
        # A lone quad has an undirected angle but cannot establish scale. A
        # current verified symbol supplies independent paper/head evidence.
        missing_outer = raw.copy()
        cv2.polylines(missing_outer, [np.rint([(p.u_px, p.v_px) for p in outer]).astype(np.int32)],
                      True, 0, 3)
        paper_only, _debug, _pose = fit_current_measured_head(cv2, missing_outer,
            model_profile=self.profile, camera=camera, proposal_corners=inner)
        self.assertTrue(paper_only.usable, paper_only.reason)
        self.assertFalse(_debug.head_outer_recovery.recovered)
        qr, _camera = self.projection(45., distance_m=.35,
                                      physical_size_scale=.062/.078)
        rejected = check_current_head_marker_boundary(cv2,
            head_corners=paper_only.corners, qr_corners=qr, marker_verified=True,
            model_profile=self.profile)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "current_border_matches_verified_qr_panel")
        self.assertTrue(check_current_head_marker_boundary(cv2,
            head_corners=physical.corners, qr_corners=qr, marker_verified=True,
            model_profile=self.profile).accepted)

    def recorded_inputs(self):
        fixture = json.loads((FIXTURE / "inputs.json").read_text())
        raw = (FIXTURE / fixture["source_frame"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), fixture["source_sha256"])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        rectified = rectify_bgr_frame(image, CameraCalibration(**fixture["camera_calibration"]), cv2, np)
        options = fixture["options"]
        edges = _canny_edges_from_frame(cv2, rectified,
            edge_preprocess=options["edge_preprocess"].replace("-", "_"),
            blur_kernel=options["edge_blur_kernel"], canny_low=options["canny_low"],
            canny_high=options["canny_high"])
        # These are saved current-head points used only to locate a new raw
        # search. No historical normal, rvec, QR corners or pose is supplied.
        corners = tuple(ImagePoint(**p) for p in fixture["recorded_model"]["candidate_corners"])
        return fixture, edges, corners, RectifiedCameraMatrix(**fixture["processing_intrinsics"])

    def test_recorded_oblique_frame_refines_current_pixels_without_qr_or_pose_hint(self):
        fixture, edges, corners, camera = self.recorded_inputs()
        estimate, debug, _pose = fit_current_measured_head(cv2, edges, model_profile=self.profile,
            camera=camera, proposal_corners=corners)
        self.assertTrue(fixture["result_freshness"]["accepted"])
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertLess(estimate.yaw_deg, -35.)
        # Compare with the saved estimator output, not an asserted physical angle.
        self.assertAlmostEqual(estimate.yaw_deg,
            fixture["recorded_display_estimate"]["yaw_deg"], delta=.5)
        self.assertTrue(validated_head_model_quality(debug.head_model_quality))
        self.assertLess(debug.head_model_quality.yaw_std_deg, 1.5)
        self.assertIsNone(estimate.visible_face)

    def test_recorded_proposal_without_current_pixels_grants_no_angle(self):
        _fixture, edges, corners, camera = self.recorded_inputs()
        estimate, _debug, _pose = fit_current_measured_head(cv2, np.zeros_like(edges),
            model_profile=self.profile, camera=camera, proposal_corners=corners)
        self.assertFalse(estimate.usable)
        self.assertIsNone(estimate.camera_face_normal_xyz)
        self.assertIsNone(estimate.yaw_deg)


if __name__ == "__main__":
    unittest.main()
