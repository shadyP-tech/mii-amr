"""Proposal pixels select a crop only after the real LiDAR binding gate."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint, StandAxisEdgeDebugArtifacts, StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform, rectified_pixel_bearing_in_scan
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    HeadRoiEvaluation, select_camera_target_measurement,
)
from scripts.aufgabe04.real_robot.observer.head_proposal_registration import (
    acquire_registered_head_measurement, recenter_head_proposal, unresolved_front_framing_hint,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt, REGISTERED_BACKSIDE_REACQUISITION_SOURCE,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE, TARGET_CENTERED_REACQUISITION_SOURCE,
)


CAMERA_TO_SCAN = RigidTransform("base_scan", "camera", (0., 0., 0.), (.5, -.5, .5, -.5))


class _Frame:
    """The image locator is injected; no OpenCV runtime is needed here."""
    def __getitem__(self, slices):
        return self


class HeadProposalRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(640, 480, 520., 515., 320., 240.)
        self.search = HeadRoiAttempt(
            ImageRoi(180, 60, 620, 460, 100.), TARGET_CENTERED_REACQUISITION_SOURCE,
            4.5, 320., 240., 100., 2.25,
        )
        self.proposal = HeadProposal(
            corners=tuple(ImagePoint(u, v) for u, v in ((120., 130.), (220., 130.),
                                                      (220., 230.), (120., 230.))),
            bounds_xyxy=(108, 118, 233, 276), head_bounds_xyxy=(120., 130., 220., 230.),
            center_u_px=170., center_v_px=180., observed_height_px=100.,
            expected_height_ratio=1., center_offset_head_heights=.3,
            raw_edge_support=.99, geometry_quality=.99,
        )
        self.bearing = rectified_pixel_bearing_in_scan(
            u_px=350., v_px=240., fx_px=520., fy_px=515., cx_px=320., cy_px=240.,
            scan_from_camera=CAMERA_TO_SCAN,
        )
        self.scan = PlainLaserScan(
            (float("inf"), float("inf"), .60, .601, .60, float("inf"), float("inf")),
            self.bearing - math.radians(3), math.radians(1), .10, 3.5,
            "base_scan", scan_stamp_sec=10., receipt_sec=10.01,
        )
        self.estimate = StandAxisImageEstimate(
            usable=False, reason="model_joint_qr_head_fit_rejected", mode="model",
            corners=None, axis_line=None, left_height_px=0., right_height_px=0.,
            height_ratio=None, yaw_proxy=None, yaw_deg=None, closer_side=None,
            contour_area_px=0., source="model_refined_head",
        )
        self.debug = StandAxisEdgeDebugArtifacts(
            edges=None, qr_detected=True, qr_marker_verified=True,
            model_pose_fit_source="joint_qr_head", model_pose=None,
        )
        self.diagnostics, self.calls = {}, []

    def acquire(self, *, scan=None, proposal=None, debug=None, estimate=None, **overrides):
        def evaluate(attempt, corners):
            self.assertTrue(self.diagnostics["candidate_associated"], "strict fit preceded LiDAR binding")
            self.calls.append((attempt, corners))
            return HeadRoiEvaluation(attempt, _Frame(), estimate or self.estimate, debug or self.debug)
        kwargs = dict(
            intrinsics=self.intrinsics, scan_from_camera=CAMERA_TO_SCAN,
            scan=self.scan if scan is None else scan, map_bearing_rad=0.,
            cone_half_angle_rad=math.radians(3), accepted_range_m=(.49, .71),
            now_sec=10.1, max_scan_age_sec=.5, min_cluster_sample_count=2,
            max_camera_map_bearing_delta_rad=math.radians(12), max_center_offset_ratio=1.5,
            edge_preprocess="channel_union", canny_low=20, canny_high=60,
            evaluate=evaluate, diagnostics=self.diagnostics,
        )
        with patch(
            "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
            return_value=HeadProposalResult(self.proposal if proposal is None else proposal,
                                           "current_head_proposal", 2, 1),
        ) as locate:
            result = acquire_registered_head_measurement(object(), _Frame(), self.search,
                                                        **{**kwargs, **overrides})
            self.assertEqual(locate.call_count, 1)
            self.assertEqual(locate.call_args.kwargs["expected_head_center_u_px"], 140.)
        return result

    def test_recenter_keeps_complete_head_neck_bounds_and_offsets_exactly_once(self):
        result = recenter_head_proposal(self.proposal, self.search, max_center_offset_ratio=1.5)
        self.assertEqual(result.attempt.roi, ImageRoi(288, 178, 413, 336, 100.))
        self.assertEqual(result.metadata["head_bounds_full_image"], [300., 190., 400., 290.])
        self.assertEqual(result.attempt.expected_center_u_px, 350.)
        self.assertEqual(result.attempt.expected_center_v_px, 240.)
        for original, cropped in zip(self.proposal.corners, result.corners):
            full_u, full_v = original.u_px + self.search.roi.x0, original.v_px + self.search.roi.y0
            self.assertEqual(cropped.u_px + result.attempt.roi.x0, full_u)
            self.assertEqual(cropped.v_px + result.attempt.roi.y0, full_v)
            # The callback's ordinary crop-adjusted principal point must
            # represent precisely the original rectified ray, with no offset twice.
            crop_cx, crop_cy = self.intrinsics.cx_px - result.attempt.roi.x0, self.intrinsics.cy_px - result.attempt.roi.y0
            self.assertAlmostEqual((cropped.u_px - crop_cx) / self.intrinsics.fx_px,
                                   (full_u - self.intrinsics.cx_px) / self.intrinsics.fx_px)
            self.assertAlmostEqual((cropped.v_px - crop_cy) / self.intrinsics.fy_px,
                                   (full_v - self.intrinsics.cy_px) / self.intrinsics.fy_px)
        self.assertFalse(result.metadata["candidate_associated"])
        self.assertFalse(result.metadata["measurement_reused"])

    def test_recenter_rejects_clipped_border_out_of_search_bounds_and_large_offset(self):
        for proposal in (
            replace(self.proposal, bounds_xyxy=(120, 118, 233, 276)),
            replace(self.proposal, bounds_xyxy=(108, 118, 450, 276)),
            replace(self.proposal, bounds_xyxy=(-1, 118, 233, 276)),
        ):
            with self.subTest(bounds=proposal.bounds_xyxy):
                self.assertIsNone(recenter_head_proposal(proposal, self.search, max_center_offset_ratio=1.5))
        self.assertIsNone(recenter_head_proposal(self.proposal, self.search, max_center_offset_ratio=.2))

    def test_neutral_proposal_and_unique_lidar_allow_strict_fit_without_any_pose(self):
        selection = self.acquire()
        self.assertIsNotNone(selection)
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(selection.selected.debug.model_pose)
        self.assertFalse(selection.selected.estimate.usable)
        self.assertTrue(selection.registered)
        self.assertEqual(selection.selected.attempt.source, REGISTERED_QR_MODEL_REACQUISITION_SOURCE)
        self.assertIsNone(selection.proposal, "2D proposal must not masquerade as a measured ROI")
        association = selection.head_acquisition["lidar_association"]
        self.assertTrue(association["associated"])
        self.assertEqual(association["search_association"]["eligible_cluster_count"], 1)

    def test_no_qr_strict_result_keeps_backside_source_without_promoting_axis(self):
        selection = self.acquire(debug=replace(self.debug, qr_detected=False, qr_marker_verified=False))
        self.assertEqual(selection.selected.attempt.source, REGISTERED_BACKSIDE_REACQUISITION_SOURCE)
        self.assertFalse(selection.selected.estimate.usable)

    def test_wrong_bearing_range_ambiguous_and_stale_scan_reject_before_strict_fit(self):
        cases = (
            {"map_bearing_rad": math.radians(20)},
            {"accepted_range_m": (.8, 1.)},
            {"scan": replace(self.scan, ranges=(float("inf"), .60, .601, float("inf"), .61, .611, float("inf")))},
            {"scan": replace(self.scan, scan_stamp_sec=9., receipt_sec=9.01)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.diagnostics.clear()
                self.calls.clear()
                self.assertIsNone(self.acquire(**overrides))
                self.assertEqual(self.calls, [])
                self.assertFalse(self.diagnostics["candidate_associated"])
                self.assertEqual(self.diagnostics["reason"], "head_proposal_candidate_association_rejected")

    def test_unavailable_proposal_and_invalid_crop_never_call_strict_fit(self):
        with patch("scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
                   return_value=HeadProposalResult(None, "head_proposal_unavailable")):
            # Call directly so the acquisition wrapper does not replace this result.
            selection = acquire_registered_head_measurement(
                object(), _Frame(), self.search, intrinsics=self.intrinsics,
                scan_from_camera=CAMERA_TO_SCAN, scan=self.scan, map_bearing_rad=0.,
                cone_half_angle_rad=.05, accepted_range_m=(.49, .71), now_sec=10.1,
                max_scan_age_sec=.5, min_cluster_sample_count=2,
                max_camera_map_bearing_delta_rad=.2, max_center_offset_ratio=1.5,
                edge_preprocess="channel_union", canny_low=20, canny_high=60,
                evaluate=lambda *args: self.fail("fit with no proposal"), diagnostics=self.diagnostics,
            )
            self.assertIsNone(selection)
        self.assertIsNone(self.acquire(proposal=replace(self.proposal, bounds_xyxy=(120, 118, 233, 276))))
        self.assertEqual(self.calls, [])

    def hint(self, selection, **overrides):
        kwargs = dict(target_key="candidate", source_image_stamp_sec=10., source_fresh=True,
                      range_m=.413, optical_depth_m=.372, intrinsics=self.intrinsics)
        return unresolved_front_framing_hint(selection, **{**kwargs, **overrides})

    def test_framing_hint_requires_associated_fresh_verified_front_failed_joint_fit(self):
        selection = self.acquire()
        hint = self.hint(selection)
        self.assertEqual(hint["target_key"], "candidate")
        self.assertTrue(hint["candidate_associated"])
        self.assertFalse(hint["motion_authorized"])
        self.assertIsNone(self.hint(selection, source_fresh=False))
        self.assertIsNone(self.hint(replace(selection, strict_retry=None)))
        self.assertIsNone(self.hint(replace(selection, head_acquisition=None)))
        self.assertIsNone(self.hint(replace(selection, head_acquisition={**selection.head_acquisition,
                                                                       "candidate_associated": False})))
        for changes in ({"qr_marker_verified": False}, {"qr_marker_verified": None},
                        {"model_pose_fit_source": "head_only"}):
            with self.subTest(changes=changes):
                current = replace(selection.selected, debug=replace(selection.selected.debug, **changes))
                self.assertIsNone(self.hint(replace(selection, selected=current, strict_retry=current)))
        current = replace(selection.selected, estimate=replace(selection.selected.estimate, usable=True))
        self.assertIsNone(self.hint(replace(selection, selected=current, strict_retry=current)))

    def select_with_locator_result(self, result, *, accepted_range_m=(.49, .71)):
        metric_calls = []
        nominal = replace(self.search, source="nominal_projection", padding_scale=1.8,
                          backside_target_crop_half_width_ratio=1.25)
        def evaluate(attempt, pose_or_corners):
            metric_calls.append(attempt)
            return HeadRoiEvaluation(attempt, _Frame(), self.estimate, self.debug)
        def acquire(search, primary):
            return acquire_registered_head_measurement(
                object(), _Frame(), search, intrinsics=self.intrinsics,
                scan_from_camera=CAMERA_TO_SCAN, scan=self.scan, map_bearing_rad=0.,
                cone_half_angle_rad=math.radians(3), accepted_range_m=accepted_range_m,
                now_sec=10.1, max_scan_age_sec=.5, min_cluster_sample_count=2,
                max_camera_map_bearing_delta_rad=math.radians(12), max_center_offset_ratio=1.5,
                edge_preprocess="channel_union", canny_low=20, canny_high=60,
                evaluate=evaluate, diagnostics=self.diagnostics, primary=primary,
            )
        with patch("scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
                   return_value=result):
            selection = select_camera_target_measurement(
                (nominal, self.search), tracked_pose=None, evaluate=evaluate,
                enable_reacquisition=True, max_center_offset_ratio=1.5,
                acquire_registered=acquire,
            )
        return selection, metric_calls

    def test_positive_head_with_failed_real_lidar_binding_skips_redundant_wide_fit(self):
        selection, calls = self.select_with_locator_result(
            HeadProposalResult(self.proposal, "current_head_proposal", 2, 1),
            accepted_range_m=(.8, 1.),
        )
        self.assertEqual([attempt.source for attempt in calls], ["nominal_projection"])
        self.assertEqual(len(selection.evaluations), 1)
        self.assertIs(selection.selected, selection.evaluations[0])
        self.assertFalse(selection.registered)
        self.assertFalse(selection.selected.estimate.usable)
        self.assertIsNone(selection.selected.debug.model_pose)
        self.assertIsNone(selection.proposal)
        self.assertEqual(selection.head_acquisition["reason"], "head_proposal_candidate_association_rejected")
        self.assertFalse(selection.head_acquisition["lidar_association"]["associated"])
        self.assertFalse(selection.metadata(enabled=True)["measurement_accepted"])
        self.assertIsNone(self.hint(selection))

    def test_ambiguous_head_search_ends_reacquisition_without_another_metric_fit(self):
        selection, calls = self.select_with_locator_result(
            HeadProposalResult(None, "head_proposal_ambiguous", 3, 2),
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(selection.registered)
        self.assertFalse(selection.selected.estimate.usable)
        self.assertEqual(selection.head_acquisition["reason"], "head_proposal_ambiguous")
        self.assertIsNone(self.hint(selection))

    def test_no_head_proposal_preserves_existing_bounded_wide_fallback(self):
        selection, calls = self.select_with_locator_result(
            HeadProposalResult(None, "head_proposal_unavailable", 2, 1),
        )
        self.assertEqual([attempt.source for attempt in calls],
                         ["nominal_projection", TARGET_CENTERED_REACQUISITION_SOURCE])
        self.assertEqual(len(selection.evaluations), 2)
        self.assertIsNotNone(selection.proposal)
        self.assertFalse(selection.registered)
        self.assertFalse(selection.selected.estimate.usable)
        self.assertIsNone(self.hint(selection))


if __name__ == "__main__":
    unittest.main()
