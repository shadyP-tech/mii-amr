"""Canonical current borders survive fitting, but never a new or altered image."""

from dataclasses import replace
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.current_head_refinement import refine_current_physical_head
from scripts.aufgabe04.perception.stand_axis.current_head_refinement_proof import capture_current_head_refinement
from scripts.aufgabe04.perception.stand_axis.head_model_fit import fit_current_measured_head
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix
from tests.aufgabe04 import test_head_boundary_independence as fixtures


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class CurrentHeadRefinementProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HeadBoundaryIndependenceTest.setUpClass()
        cls.fixture = fixtures.HeadBoundaryIndependenceTest()
        cls.profile = cls.fixture.profile

    def data(self):
        outer, camera = self.fixture.projection()
        panel, _ = self.fixture.projection(.071/.078)
        raw = self.fixture.raw(outer, panel)
        frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        refinement, boundary, seed = refine_current_physical_head(
            cv2, raw, model_profile=self.profile, proposal_corners=panel)
        proof = capture_current_head_refinement(frame, raw, model_profile=self.profile,
            refinement=refinement, outer_recovery=boundary, seed=seed)
        return frame, raw, camera, panel, refinement, proof

    def fit(self, frame, raw, camera, corners, proof, profile=None):
        return fit_current_measured_head(cv2, raw, frame_bgr=frame,
            model_profile=self.profile if profile is None else profile, camera=camera,
            proposal_corners=corners, current_head_refinement=proof)

    def test_selected_canonical_border_is_solved_without_second_outward_search(self):
        frame, raw, camera, panel, refinement, proof = self.data()
        baseline, _, _ = fit_current_measured_head(cv2, raw, model_profile=self.profile,
            camera=camera, proposal_corners=panel)
        with patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                   side_effect=AssertionError("same image must not select another boundary")):
            estimate, debug, _ = self.fit(frame, raw, camera, refinement.corners, proof)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.corners, baseline.corners)
        self.assertEqual(estimate.yaw_deg, baseline.yaw_deg)
        self.assertEqual(debug.refined_corners, refinement.corners)

    def test_pipeline_passes_the_current_proof_to_the_same_3d_solver(self):
        frame, raw, camera, _panel, refinement, proof = self.data()
        with patch("scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame",
                   return_value=raw), patch(
                   "scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                   side_effect=AssertionError("canonical boundary already measured")):
            estimate, debug = estimate_stand_axis_from_metric_model(cv2, frame,
                model_profile=self.profile, camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
                camera_cx_px=camera.cx_px, camera_cy_px=camera.cy_px,
                current_head_proposal_corners=refinement.corners,
                current_head_proposal_verified=True, current_head_refinement=proof,
                qr_marker_policy="disabled")
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertTrue(debug.head_acquisition_diagnostics["current_boundary_reused"])
        self.assertTrue(debug.head_acquisition_diagnostics["selected_border_binding"]["accepted"])

    def test_cold_pipeline_solves_exactly_its_selected_corners_without_refining_again(self):
        outer, camera = self.fixture.projection()
        raw = self.fixture.raw(outer)
        frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        with patch("scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame",
                   return_value=raw), patch(
                   "scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                   side_effect=AssertionError("cold selector has already measured these pixels")):
            estimate, debug = estimate_stand_axis_from_metric_model(cv2, frame,
                model_profile=self.profile, camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
                camera_cx_px=camera.cx_px, camera_cy_px=camera.cy_px, qr_marker_policy="disabled")
        self.assertTrue(estimate.usable, estimate.reason)
        acquisition = debug.head_acquisition_diagnostics["acquisition"]
        selected = tuple(ImagePoint(**point) for point in acquisition["proposal"]["corners"])
        self.assertEqual(estimate.corners, selected)
        self.assertTrue(debug.head_acquisition_diagnostics["current_boundary_reused"])

    def test_invalid_proof_never_falls_back_to_another_search(self):
        for mutation in ("new_image", "mutated_image", "changed_raw", "changed_model",
                         "different_corners", "boolean_proof", "missing_frame"):
            with self.subTest(mutation=mutation):
                frame, raw, camera, _panel, refinement, proof = self.data()
                corners, profile = refinement.corners, self.profile
                if mutation == "new_image":
                    frame = frame.copy()
                elif mutation == "mutated_image":
                    frame[0, 0] = (1, 2, 3)
                elif mutation == "changed_raw":
                    p = corners[0]
                    raw[round(p.v_px), round(p.u_px)] ^= 255
                elif mutation == "changed_model":
                    profile = replace(profile, head_width_m=profile.head_width_m + .001)
                elif mutation == "different_corners":
                    corners = tuple(ImagePoint(p.u_px + 1., p.v_px) for p in corners)
                elif mutation == "boolean_proof":
                    proof = True
                elif mutation == "missing_frame":
                    frame = None
                with patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                           side_effect=AssertionError("invalid proof must not trigger reacquisition")), \
                     patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.estimate_planar_pose_ippe",
                           side_effect=AssertionError("invalid proof cannot supply fit corners")):
                    estimate, debug, pose = self.fit(frame, raw, camera, corners, proof, profile)
                self.assertFalse(estimate.usable)
                self.assertEqual(estimate.reason, "current_head_refinement_invalid")
                self.assertIsNone(debug.head_outer_recovery)
                self.assertIsNone(pose)

    def test_recentered_crop_translates_current_evidence_and_preserves_angle(self):
        frame, raw, camera, _panel, refinement, proof = self.data()
        first, _, _ = self.fit(frame, raw, camera, refinement.corners, proof)
        dx, dy = 250, 150
        cropped = frame[dy:450, dx:550]
        cropped_raw = raw[dy:450, dx:550].copy()
        moved = proof.rebase(frame, cropped, dx, dy)
        corners = tuple(ImagePoint(p.u_px - dx, p.v_px - dy) for p in refinement.corners)
        camera = RectifiedCameraMatrix(camera.fx_px, camera.fy_px, camera.cx_px-dx, camera.cy_px-dy)
        # Recomputed Canny may differ at the new canvas perimeter, away from all
        # selected head support. It cannot change the canonical head's evidence.
        cropped_raw[0, :] ^= 255
        estimate, debug, _ = self.fit(cropped, cropped_raw, camera, corners, moved)
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertAlmostEqual(first.yaw_deg, estimate.yaw_deg, places=9)
        self.assertEqual(debug.refined_corners, corners)
        self.assertEqual(debug.face_mask.shape, cropped.shape[:2])
        for invalid in (cropped.copy(), frame[dy:450, dx+1:551]):
            with self.assertRaises(ValueError):
                proof.rebase(frame, invalid, dx, dy)
        with self.assertRaises(ValueError):
            proof.rebase(frame, frame[200:250, 390:440], 390, 200)

    def test_pipeline_reuses_measured_edges_when_crop_loses_hysteresis_connection(self):
        outer, camera = self.fixture.projection()
        frame = np.zeros((600, 800, 3), np.uint8)
        polygon = np.rint([(p.u_px, p.v_px) for p in outer]).astype(np.int32)
        cv2.fillConvexPoly(frame, polygon, (12, 12, 12))
        cv2.rectangle(frame, (395, 350), (400, 590), (12, 12, 12), -1)
        # The weak head outline connects through a stem to strong contrast
        # below the crop. Cropping removes that Canny hysteresis seed, while
        # every pixel of the complete head and its twelve-pixel margin persists.
        for row in range(460, 600):
            frame[row][np.any(frame[row] > 0, axis=1)] = min(255, 12 + 2 * (row - 460))

        def edges(image):
            return _canny_edges_from_frame(cv2, image, edge_preprocess="channel_union",
                                          blur_kernel=5, canny_low=20, canny_high=60)

        raw = edges(frame)
        refinement, boundary, seed = refine_current_physical_head(
            cv2, raw, model_profile=self.profile, proposal_corners=outer)
        proof = capture_current_head_refinement(frame, raw, model_profile=self.profile,
            refinement=refinement, outer_recovery=boundary, seed=seed)
        baseline, _, _ = self.fit(frame, raw, camera, refinement.corners, proof)
        self.assertTrue(baseline.usable, baseline.reason)
        dx, dy = 250, 150
        crop = frame[dy:450, dx:550]
        cropped_edges = edges(crop)
        moved = proof.rebase(frame, crop, dx, dy)
        corners = tuple(ImagePoint(p.u_px - dx, p.v_px - dy) for p in refinement.corners)
        self.assertGreater(np.count_nonzero(raw[200:400, 300:500] !=
                                            cropped_edges[50:250, 50:250]), 100)
        # Direct resolve remains strict: independently changed support cannot
        # pass just because the underlying BGR image is the same view.
        with self.assertRaisesRegex(ValueError, "raw supporting pixels changed"):
            moved.resolve(crop, cropped_edges, model_profile=self.profile,
                          proposal_corners=corners)
        with patch("scripts.aufgabe04.perception.stand_axis.model_pipeline._canny_edges_from_frame",
                   side_effect=AssertionError("original edges already measured")), patch(
                   "scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                   side_effect=AssertionError("original boundary already selected")):
            estimate, debug = estimate_stand_axis_from_metric_model(cv2, crop,
                model_profile=self.profile, camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
                camera_cx_px=camera.cx_px-dx, camera_cy_px=camera.cy_px-dy,
                current_head_proposal_corners=corners, current_head_proposal_verified=True,
                current_head_refinement=moved, qr_marker_policy="disabled")
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertAlmostEqual(estimate.yaw_deg, baseline.yaw_deg, places=9)
        self.assertEqual(estimate.corners, corners)
        np.testing.assert_array_equal(debug.raw_edges, raw[dy:450, dx:550])
        self.assertTrue(debug.head_acquisition_diagnostics["current_boundary_reused"])
        self.assertTrue(debug.head_acquisition_diagnostics["selected_border_binding"]["accepted"])

    def test_raw_edge_reuse_rechecks_frame_model_and_selected_corners(self):
        for mutation in ("new_image", "mutated_image", "changed_model", "different_corners"):
            with self.subTest(mutation=mutation):
                frame, _raw, camera, _panel, refinement, proof = self.data()
                corners, profile = refinement.corners, self.profile
                if mutation == "new_image":
                    frame = frame.copy()
                elif mutation == "mutated_image":
                    frame[0, 0] = (1, 2, 3)
                elif mutation == "changed_model":
                    profile = replace(profile, head_width_m=profile.head_width_m + .001)
                elif mutation == "different_corners":
                    corners = tuple(ImagePoint(p.u_px + 1., p.v_px) for p in corners)
                with self.assertRaises(ValueError):
                    proof.raw_edges_for(frame, model_profile=profile, proposal_corners=corners)
                with patch("scripts.aufgabe04.perception.stand_axis.head_model_fit.refine_current_physical_head",
                           side_effect=AssertionError("invalid proof must not reacquire")), patch(
                           "scripts.aufgabe04.perception.stand_axis.head_model_fit.estimate_planar_pose_ippe",
                           side_effect=AssertionError("invalid proof must not reach the solver")):
                    estimate, _debug = estimate_stand_axis_from_metric_model(cv2, frame,
                        model_profile=profile, camera_fx_px=camera.fx_px, camera_fy_px=camera.fy_px,
                        camera_cx_px=camera.cx_px, camera_cy_px=camera.cy_px,
                        current_head_proposal_corners=corners, current_head_proposal_verified=True,
                        current_head_refinement=proof, qr_marker_policy="disabled")
                self.assertFalse(estimate.usable)
                self.assertEqual(estimate.reason, "current_head_refinement_invalid")

    def test_returned_raw_edges_cannot_mutate_captured_support(self):
        frame, raw, _camera, _panel, refinement, proof = self.data()
        copied = proof.raw_edges_for(frame, model_profile=self.profile,
                                    proposal_corners=refinement.corners)
        copied[:] = 0
        np.testing.assert_array_equal(proof.raw_edges_for(
            frame, model_profile=self.profile, proposal_corners=refinement.corners), raw)
        proof.resolve(frame, raw, model_profile=self.profile, proposal_corners=refinement.corners)

    def test_debug_mask_mutation_cannot_modify_retained_current_evidence(self):
        frame, raw, _camera, _panel, refinement, proof = self.data()
        first, _, _ = proof.resolve(frame, raw, model_profile=self.profile,
            proposal_corners=refinement.corners)
        original = first.evidence_mask.copy()
        first.evidence_mask[:] = 0
        second, _, _ = proof.resolve(frame, raw, model_profile=self.profile,
            proposal_corners=refinement.corners)
        np.testing.assert_array_equal(second.evidence_mask, original)


if __name__ == "__main__":
    unittest.main()
