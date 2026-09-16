"""Independent 2D proposals never bypass current metric evidence gates."""

import math
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis.head_border_seed import (
    select_head_border_seed,
    validate_current_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis.head_model_quality import validated_head_model_quality
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support
from scripts.aufgabe04.perception.stand_axis.model_profile import stand_model_from_payload
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    QrQuadDetection,
    RectifiedCameraMatrix,
    estimate_planar_pose_ippe,
)
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from tests.aufgabe04.test_stand_metric_model import (
    oblique_pose,
    profile_payload,
)


PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline."
HEAD_FIT = "scripts.aufgabe04.perception.stand_axis.head_model_fit."


class HeadBorderSeedTest(unittest.TestCase):
    def test_invalid_proposals_fail_closed(self):
        valid = (ImagePoint(10, 10), ImagePoint(90, 10), ImagePoint(90, 90), ImagePoint(10, 90))
        for corners in (
            valid[:3], valid[:3] + (ImagePoint(math.nan, 50),),
            valid[:3] + (ImagePoint(-1, 90),),
            valid[:3] + (ImagePoint(10, 100),),
            (valid[0],) * 4, ((10, 10),) * 4,
        ):
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                validate_current_head_proposal(corners, frame_shape=(100, 100, 3))
        self.assertEqual(validate_current_head_proposal(valid, frame_shape=(100, 100)), valid)
        self.assertIsNone(validate_current_head_proposal(None, frame_shape=(100, 100)))

    def test_proposal_only_positions_the_same_bounded_corridor(self):
        profile = stand_model_from_payload(profile_payload())
        corners = (ImagePoint(10, 10), ImagePoint(90, 10), ImagePoint(90, 90), ImagePoint(10, 90))
        seed = select_head_border_seed(
            model_profile=profile, projected_corners=None,
            pose_reprojection_rmse_px=None, current_head_proposal_corners=corners,
        )
        self.assertEqual(seed.source, "current_head_proposal")
        self.assertEqual(seed.corners, corners)
        self.assertGreaterEqual(seed.corridor_half_width_px, 4.)
        self.assertLessEqual(seed.corridor_half_width_px, 8.)
        self.assertFalse(hasattr(seed, "model_pose"))


@unittest.skipUnless(cv2 is not None and np is not None, "requires OpenCV/numpy")
class CurrentHeadProposalPipelineTest(unittest.TestCase):
    def setUp(self):
        self.profile = stand_model_from_payload(profile_payload())
        self.camera = RectifiedCameraMatrix(800., 800., 320., 240.)
        self.projected = project_stand_model(cv2, self.profile, oblique_pose(), self.camera)
        self.head = self.projected.head_corners
        self.qr = tuple(self.projected.landmarks["qr_" + name] for name in (
            "top_left", "top_right", "bottom_right", "bottom_left",
        ))
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.polylines(self.frame, np.asarray([[
            (round(p.u_px), round(p.v_px)) for p in self.head
        ]], dtype=np.int32), True, (255, 255, 255), 2)
        bottom = (self.head[2].v_px + self.head[3].v_px) / 2.0
        center = (self.head[2].u_px + self.head[3].u_px) / 2.0
        for offset in (-6, 6):
            cv2.line(self.frame, (round(center + offset), round(bottom)),
                     (round(center + offset), round(bottom + 35)), (255, 255, 255), 2)

    def options(self, **overrides):
        result = dict(
            model_profile=self.profile, camera_fx_px=self.camera.fx_px,
            camera_fy_px=self.camera.fy_px, camera_cx_px=self.camera.cx_px,
            camera_cy_px=self.camera.cy_px, blur_kernel=1,
            current_head_proposal_corners=self.head,
        )
        result.update(overrides)
        return result

    def evaluate(self, frame=None, qr=None, **options):
        with patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(
            self.qr if qr is None else qr, 1., text="QR_001",
        )):
            return estimate_stand_axis_from_metric_model(
                cv2, self.frame if frame is None else frame, **self.options(**options),
            )

    def test_current_head_fits_once_without_any_qr_or_joint_pose(self):
        calls = []

        def solve(cv, image_points, model_points, camera, **kwargs):
            calls.append(len(image_points))
            self.assertEqual(tuple(model_points), self.profile.head_corners)
            return estimate_planar_pose_ippe(cv, image_points, model_points, camera, **kwargs)

        with (
            patch(HEAD_FIT + "estimate_planar_pose_ippe", side_effect=solve),
            patch(PIPELINE + "estimate_planar_pose_ippe", side_effect=AssertionError("QR/joint pose forbidden")),
            patch(PIPELINE + "collect_metric_model_diagnostics", side_effect=AssertionError("QR geometry forbidden")),
        ):
            estimate, debug = self.evaluate()
        self.assertEqual(calls, [4])
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(debug.model_pose_fit_source, "model_current_measured_head")
        self.assertEqual(debug.pose_seed_source, "current_head_proposal")
        self.assertTrue(debug.qr_marker_verified)
        self.assertAlmostEqual(estimate.yaw_deg, -25., delta=3.)

    def test_cold_and_proposed_heads_ignore_misaligned_qr_pixels(self):
        wrong_qr = tuple(ImagePoint(p.u_px + 18., p.v_px) for p in self.qr)
        for proposal in (None, self.head):
            with (
                self.subTest(proposal=proposal),
                patch(PIPELINE + "estimate_planar_pose_ippe", side_effect=AssertionError("QR seed forbidden")),
            ):
                baseline, baseline_debug = self.evaluate(current_head_proposal_corners=proposal)
                current, debug = self.evaluate(qr=wrong_qr, current_head_proposal_corners=proposal)
                self.assertTrue(baseline.usable, baseline.reason)
                self.assertTrue(current.usable, current.reason)
                self.assertTrue(validated_head_model_quality(baseline_debug.head_model_quality))
                self.assertTrue(validated_head_model_quality(debug.head_model_quality))
                self.assertAlmostEqual(baseline.yaw_deg, -25., delta=3.)
                if proposal is None:
                    # The shared physical-border refinement now resolves the
                    # thick rendered rail before comparison. All measured top
                    # rails agree; the 3D fit must preserve the selected current
                    # boundary rather than choosing again after selection.
                    acquisition = baseline_debug.head_acquisition_diagnostics["acquisition"]
                    diagnostics = acquisition["joint_border_diagnostics"]
                    records = diagnostics["strict_verifications"]
                    top_slopes = [(record["corners"][1][1] - record["corners"][0][1])
                                  for record in records if record["accepted"]]
                    self.assertLess(max(top_slopes) - min(top_slopes), 2.)
                    self.assertEqual(diagnostics["unverified_independent_hypotheses"], 0)
                    selected = tuple(ImagePoint(**point) for point in acquisition["proposal"]["corners"])
                    self.assertEqual(baseline.corners, selected)
                    self.assertTrue(baseline_debug.head_acquisition_diagnostics["current_boundary_reused"])
                    self.assertTrue(metric_corner_arm_support(
                        cv2, baseline_debug.raw_edges, selected).accepted)
                self.assertEqual(current.corners, baseline.corners)
                self.assertEqual(current.yaw_deg, baseline.yaw_deg)
                self.assertEqual(debug.model_pose, baseline_debug.model_pose)
                self.assertEqual(debug.head_acquisition_diagnostics["source"],
                                 "cold_current_head_search" if proposal is None else "current_candidate_proposal")

    def test_qr_size_and_translation_do_not_trigger_joint_geometric_fitting(self):
        baseline, _ = self.evaluate()
        center = tuple(sum(getattr(p, field) for p in self.qr) / 4.
                       for field in ("u_px", "v_px"))
        for scale, du, dv in ((1., 18., 0.), (1.2, 0., 0.), (.8, -12., 9.)):
            wrong_qr = tuple(ImagePoint(center[0] + (p.u_px-center[0])*scale + du,
                                       center[1] + (p.v_px-center[1])*scale + dv) for p in self.qr)
            with (
                self.subTest(scale=scale, du=du, dv=dv),
                patch(PIPELINE + "estimate_planar_pose_ippe", side_effect=AssertionError("QR/joint pose forbidden")),
                patch(PIPELINE + "collect_metric_model_diagnostics", side_effect=AssertionError("QR geometry forbidden")),
                patch(PIPELINE + "classify_joint_geometry_contract", side_effect=AssertionError("joint agreement forbidden")),
            ):
                estimate, debug = self.evaluate(qr=wrong_qr)
                self.assertTrue(estimate.usable, estimate.reason)
                self.assertEqual(estimate.corners, baseline.corners)
                self.assertEqual(estimate.yaw_deg, baseline.yaw_deg)
                self.assertTrue(validated_head_model_quality(debug.head_model_quality))
                self.assertTrue(debug.head_model_quality.raw_corner_support_accepted)
                self.assertTrue(debug.head_model_quality.outer_border_verified)
                self.assertLess(estimate.pose_reprojection_rmse_px, 2.)
                self.assertIsNone(debug.model_diagnostics)
                self.assertIsNone(debug.head_marker_boundary)
                self.assertEqual(estimate.evidence_state, "fresh_refined")

    def test_proposal_without_current_raw_borders_is_never_a_measurement(self):
        estimate, debug = self.evaluate(frame=np.zeros_like(self.frame))
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.evidence_state, "unobservable")
        self.assertIsNone(debug.refined_corners)

    def test_text_without_geometry_cannot_veto_head_or_certify_backside(self):
        with patch("scripts.aufgabe04.perception.stand_axis.model_backside_acquisition.estimate_stand_axis_from_model_backside",
                   side_effect=AssertionError("front text is not backside evidence")):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **self.options(),
                qr_observations=(DecodedQrObservation("QR_001", None, "wechat"),),
            )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(estimate.reason, "axis_estimated_current_measured_head")
        self.assertIsNotNone(debug.model_pose)
        self.assertIsNone(estimate.visible_face)
        self.assertTrue(debug.qr_marker_verified)

    def test_no_qr_current_head_has_side_evidence_independent_of_neck(self):
        head = (ImagePoint(120., 50.), ImagePoint(200., 50.), ImagePoint(200., 130.), ImagePoint(120., 130.))
        for neck in (True, False):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(frame, (120, 50), (200, 130), (255, 255, 255), 2)
            if neck:
                for u in (153, 167):
                    cv2.line(frame, (u, 131), (u, 205), (255, 255, 255), 2)
            with self.subTest(neck=neck), patch(PIPELINE + "detect_qr_quad", return_value=None):
                estimate, debug = estimate_stand_axis_from_metric_model(
                    cv2, frame, **self.options(
                        current_head_proposal_corners=head,
                        expected_head_center_u_px=160., expected_head_center_v_px=90.,
                        expected_head_height_px=80.,
                    ),
                )
                self.assertTrue(estimate.usable, estimate.reason)
                self.assertEqual(estimate.evidence_state, "fresh_backside")
                self.assertIsNone(estimate.camera_face_normal_xyz)
                self.assertIsNotNone(debug.model_pose)
                self.assertEqual(estimate.visible_face, "backside_candidate")
                self.assertTrue(debug.head_backside_classification.accepted)
                self.assertFalse(debug.qr_marker_verified)

    def test_crop_adjusted_intrinsics_preserve_pose(self):
        full, _ = self.evaluate()
        x0, y0, x1, y1 = 220, 140, 420, 380
        local = lambda points: tuple(ImagePoint(p.u_px - x0, p.v_px - y0) for p in points)
        cropped, _ = self.evaluate(
            frame=self.frame[y0:y1, x0:x1], qr=local(self.qr),
            current_head_proposal_corners=local(self.head),
            camera_cx_px=self.camera.cx_px - x0,
            camera_cy_px=self.camera.cy_px - y0,
        )
        self.assertTrue(full.usable, full.reason)
        self.assertTrue(cropped.usable, cropped.reason)
        self.assertAlmostEqual(full.yaw_deg, cropped.yaw_deg, delta=1.0e-4)
        for a, b in zip(full.camera_face_center_xyz_m, cropped.camera_face_center_xyz_m):
            self.assertAlmostEqual(a, b, places=6)

    def test_tentative_marker_remains_current_frame_veto_but_is_unverified(self):
        with patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr, 1.)):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **self.options(),
            )
        self.assertTrue(debug.qr_detected)
        self.assertFalse(debug.qr_marker_verified)
        self.assertNotEqual(estimate.evidence_state, "fresh_backside")

    def test_unverified_quad_does_not_veto_independent_current_head(self):
        with (
            patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr, 1.)),
            patch("scripts.aufgabe04.perception.stand_axis.model_backside_acquisition.estimate_stand_axis_from_model_backside",
                  side_effect=AssertionError("tentative QR still vetoes backside now")),
        ):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **self.options(),
            )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertIsNotNone(debug.model_pose)
        self.assertIsNone(estimate.visible_face)
        self.assertTrue(debug.qr_detected)
        self.assertFalse(debug.qr_marker_verified)

    def test_qr_pose_cannot_replace_an_unrelated_current_proposal(self):
        unrelated_proposal = tuple(ImagePoint(p.u_px + 20., p.v_px) for p in self.head)
        with patch(PIPELINE + "detect_qr_quad", return_value=QrQuadDetection(self.qr, 1.)):
            estimate, debug = estimate_stand_axis_from_metric_model(
                cv2, self.frame, **self.options(current_head_proposal_corners=unrelated_proposal),
            )
        self.assertFalse(estimate.usable)
        self.assertFalse(debug.qr_marker_verified)
        self.assertEqual(debug.pose_seed_source, "current_head_proposal")
        self.assertEqual(debug.predicted_corners, unrelated_proposal)
        self.assertIsNone(debug.model_pose)

    def test_multiple_decoded_markers_remain_ambiguous_and_verified(self):
        observations = (DecodedQrObservation("QR_001", None, "wechat"),
                        DecodedQrObservation("QR_002", None, "wechat"))
        estimate, debug = estimate_stand_axis_from_metric_model(
            cv2, self.frame, **self.options(), qr_observations=observations,
        )
        self.assertTrue(estimate.usable, estimate.reason)
        self.assertEqual(debug.qr_marker_reason, "multiple_decoded_qr_identities")
        self.assertTrue(debug.qr_marker_verified)
        self.assertIsNone(estimate.visible_face)


if __name__ == "__main__":
    unittest.main()
