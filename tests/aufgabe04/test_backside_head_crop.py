"""A clipped nominal front cannot supply complete-head marker absence."""

from dataclasses import replace
import unittest

from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint, StandAxisImageEstimate, StandAxisEdgeDebugArtifacts,
)
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.backside_head_crop import (
    gate_backside_head_crop, review_backside_head_crop,
)
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    CameraTargetRegistrationSelection, HeadRoiEvaluation, select_camera_target_measurement,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt


class BacksideHeadCropTest(unittest.TestCase):
    def setUp(self):
        self.attempt = HeadRoiAttempt(ImageRoi(100, 100, 240, 260, 100.),
                                      "camera_registered_backside_reacquisition", 1.8,
                                      170., 170., 100.)
        estimate = StandAxisImageEstimate(
            True, "axis_estimated_current_measured_head_backside", "model",
            tuple(ImagePoint(x, y) for x, y in ((20., 20.), (120., 20.), (120., 120.), (20., 120.))),
            None, 100., 100., 1., 0., 15., None, 10000.,
            source="model_backside_current_frame", evidence_state="fresh_backside")
        self.current = HeadRoiEvaluation(self.attempt, object(), estimate,
                                        StandAxisEdgeDebugArtifacts(None, qr_detected=False,
                                                                    qr_marker_verified=False), ())
        self.acquisition = dict(reason="current_head_proposal_strict_retry", candidate_associated=True,
                                head_bounds_full_image=[120., 120., 220., 220.],
                                lidar_association=dict(associated=True, search_association=dict(eligible_cluster_count=1)))
        self.selection = CameraTargetRegistrationSelection(
            self.current, (self.current,), None, None, self.current,
            "measured_head", head_acquisition=self.acquisition)

    def test_current_complete_head_and_unique_scan_do_not_need_neck(self):
        self.assertIsNone(self.current.debug.head_neck_junction)
        result, review = gate_backside_head_crop(self.selection)
        self.assertTrue(review.accepted)
        self.assertIs(result, self.selection)
        self.assertFalse(review.metadata()["neck_required"])

    def test_nominal_missing_association_clipped_and_different_rectangles_are_withheld(self):
        cases = (
            replace(self.selection, strict_retry=None),
            replace(self.selection, head_acquisition={}),
            replace(self.selection, head_acquisition={**self.acquisition, "candidate_associated": False}),
            replace(self.selection, head_acquisition={**self.acquisition, "head_bounds_full_image": [120., 120., 245., 220.]}),
            replace(self.selection, head_acquisition={**self.acquisition, "head_bounds_full_image": [102., 120., 150., 200.]}),
            replace(self.selection, head_acquisition={**self.acquisition, "lidar_association": dict(
                associated=True, search_association=dict(eligible_cluster_count=2))}),
        )
        for selection in cases:
            with self.subTest(selection=selection):
                result, review = gate_backside_head_crop(selection)
                self.assertFalse(review.accepted)
                self.assertFalse(result.selected.estimate.usable)
                self.assertIsNone(result.selected.estimate.yaw_deg)
                self.assertEqual(result.selected.estimate.reason, "backside_complete_head_crop_unverified")

    def test_no_marker_check_or_positive_current_marker_cannot_supply_absence(self):
        for current in (replace(self.current, qr_observations=None),
                        replace(self.current, debug=replace(self.current.debug, qr_detected=True)),
                        replace(self.current, debug=replace(self.current.debug, qr_marker_verified=True))):
            selection = replace(self.selection, selected=current, strict_retry=current, evaluations=(current,))
            result, review = gate_backside_head_crop(selection)
            self.assertFalse(review.accepted)
            self.assertEqual(result.selected.debug.qr_detected, current.debug.qr_detected)
            self.assertEqual(result.selected.debug.qr_marker_verified, current.debug.qr_marker_verified)

    def test_usable_nominal_backside_still_invokes_current_wide_registration(self):
        wide = replace(self.attempt, roi=ImageRoi(0, 0, 340, 340, 100.))
        calls = []
        def acquire(attempt, primary):
            calls.append((attempt, primary))
            return self.selection
        selected = select_camera_target_measurement(
            (self.attempt, wide), tracked_pose=None, evaluate=lambda *_: self.current,
            enable_reacquisition=True, max_center_offset_ratio=1.5, acquire_registered=acquire)
        self.assertEqual(calls, [(wide, self.current)])
        self.assertTrue(review_backside_head_crop(selected).accepted)

    def test_unavailable_wide_proposal_cannot_promote_nominal_backside(self):
        wide = replace(self.attempt, roi=ImageRoi(0, 0, 340, 340, 100.))
        calls = []
        def evaluate(attempt, _hint):
            calls.append(attempt)
            return self.current
        selected = select_camera_target_measurement(
            (self.attempt, wide), tracked_pose=None, evaluate=evaluate,
            enable_reacquisition=True, max_center_offset_ratio=1.5,
            acquire_registered=lambda *_: None)
        result, review = gate_backside_head_crop(selected)
        self.assertEqual(calls, [self.attempt])
        self.assertFalse(review.accepted)
        self.assertFalse(result.selected.estimate.usable)


if __name__ == "__main__":
    unittest.main()
