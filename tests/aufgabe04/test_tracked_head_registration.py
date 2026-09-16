"""Tracked current fits need complete pixels, fresh geometry and association."""

from dataclasses import replace
import json
import math
import unittest

from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi, OpticalProjection
from scripts.aufgabe04.real_robot.observer.backside_head_crop import (
    gate_backside_head_crop, review_backside_head_crop, review_current_head_crop,
)
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import (
    register_current_tracked_head, tracked_head_selection,
)
from tests.aufgabe04.test_head_backside_classification import classified_head
from tests.aufgabe04.test_head_model_admission import quality


class TrackedHeadRegistrationTests(unittest.TestCase):
    def current(self, *, backside=False):
        estimate, debug, options = classified_head(qr_detected=not backside,
                                                   qr_marker_verified=not backside)
        debug = replace(debug, head_neck_junction=None,
                        head_model_quality=quality(centered_neck_supported=False,
                                                   neck_junction_verified=False))
        if backside:
            estimate, debug = classify_current_head_backside(estimate, debug, **options)
        attempt = HeadRoiAttempt(ImageRoi(280, 220, 440, 380, 90.),
                                 "candidate_tracked_head_search", 1.8, 400., 300., 90.)
        return HeadRoiEvaluation(attempt, object(), estimate, debug, ())

    def associate(self, current, **changes):
        options = dict(
            estimate=current.estimate, debug=current.debug, attempt=current.attempt,
            profile_sha256="a" * 64, projection=OpticalProjection(400., 300., .5, 90., True),
            expected_head_height_px=90., intrinsics=CameraIntrinsics(800, 600, 640., 640., 400., 300.),
            scan_from_camera=RigidTransform("base_scan", "camera", (0., 0., 0.), (-.5, .5, -.5, .5)),
            scan=PlainLaserScan((.55,) * 9, math.radians(2), math.radians(.5), .1, 3., "base_scan",
                               scan_stamp_sec=10., receipt_sec=10.),
            map_bearing_rad=0., cone_half_angle_rad=math.radians(3), accepted_range_m=(.45, .65),
            now_sec=10.1, max_scan_age_sec=.5, min_cluster_sample_count=1,
            max_center_offset_ratio=1.5, max_camera_map_bearing_delta_rad=math.radians(12),
        )
        return associate_current_measured_head(**{**options, **changes})

    def registered(self, current=None, **changes):
        current = self.current() if current is None else current
        options = dict(association=self.associate(current), observed_at_sec=10., now_sec=10.4,
                       max_age_sec=.5, expected_model_sha256="a" * 64)
        return register_current_tracked_head(tracked_head_selection(current), **{**options, **changes})

    def test_current_front_and_back_have_explicit_registration_without_proposal_retry_or_neck(self):
        for backside in (False, True):
            current = self.current(backside=backside)
            selection = self.registered(current)
            self.assertTrue(selection.current_measured_head_registration.accepted)
            self.assertFalse(selection.registered)
            self.assertIsNone(selection.proposal)
            self.assertIsNone(selection.strict_retry)
            self.assertTrue(review_current_head_crop(selection).accepted)
            if backside:
                gated, review = gate_backside_head_crop(selection)
                self.assertTrue(review.accepted)
                self.assertIs(gated, selection)
            else:
                self.assertFalse(review_backside_head_crop(selection).accepted)
            payload = selection.metadata(enabled=True)
            json.dumps(payload, allow_nan=False)
            self.assertFalse(payload["strict_retry_applied"])
            self.assertTrue(payload["measurement_accepted"])
            self.assertEqual(review_current_head_crop(selection).metadata()["basis"],
                             "current_measured_candidate_head")

    def test_hint_selection_alone_cannot_admit_backside(self):
        selected, review = gate_backside_head_crop(tracked_head_selection(self.current(backside=True)))
        self.assertFalse(review.accepted)
        self.assertFalse(selected.selected.estimate.usable)

    def test_age_bound_is_current_observer_budget_and_cannot_be_disabled(self):
        for changes in ({"now_sec": 10.51}, {"now_sec": 9.99}, {"max_age_sec": 0.},
                        {"observed_at_sec": math.nan}):
            selection = self.registered(**changes)
            self.assertFalse(review_current_head_crop(selection).accepted)
        self.assertTrue(review_current_head_crop(self.registered(now_sec=10.4)).accepted)

    def test_original_projection_range_and_unique_cluster_rejections_remain(self):
        current = self.current()
        for changes in (
            {"projection": OpticalProjection(550., 300., .5, 90., True)},
            {"accepted_range_m": (.8, 1.)},
            {"scan": PlainLaserScan((.55, .55, math.nan, .55, .55, .55, .55, .55, .55),
                                    math.radians(2), math.radians(.5), .1, 3., "base_scan",
                                    scan_stamp_sec=10., receipt_sec=10.)},
            {"expected_head_height_px": 200.},
        ):
            association = self.associate(current, **changes)
            self.assertFalse(association.accepted)
            self.assertFalse(review_current_head_crop(self.registered(current, association=association)).accepted)

    def test_accepted_flag_cannot_replace_unique_current_association(self):
        current = self.current()
        association = self.associate(current)
        for changed in (
            replace(association, full_image_center_px=(380., 300.)),
            replace(association, scale_gate={"accepted": False}),
            replace(association, roi_source="nominal_projection"),
            replace(association, lidar_association=replace(association.lidar_association,
                search_association=replace(association.lidar_association.search_association,
                                           eligible_cluster_count=2))),
        ):
            self.assertFalse(review_current_head_crop(self.registered(current, association=changed)).accepted)

    def test_earlier_association_cannot_supply_a_stale_scan(self):
        current = self.current()
        association = self.associate(current)
        scan = replace(association.lidar_association.search_association, scan_stamp_sec=9.8)
        old = replace(association, lidar_association=replace(association.lidar_association,
                                                             search_association=scan))
        selection = self.registered(current, association=old)
        self.assertEqual(review_current_head_crop(selection).reason,
                         "complete_head_current_scan_freshness_required")

    def test_bad_quality_profile_and_noncurrent_geometry_cannot_receive_crop_proof(self):
        current = self.current()
        for changed in (
            replace(current, estimate=replace(current.estimate, evidence_state="predicted_only")),
            replace(current, debug=replace(current.debug, head_model_quality=quality(axis_ambiguous=True))),
            replace(current, debug=replace(current.debug, head_model_quality=quality(outer_border_verified=False))),
            replace(current, attempt=replace(current.attempt, source="nominal_projection")),
        ):
            self.assertFalse(review_current_head_crop(self.registered(changed)).accepted)
        self.assertFalse(review_current_head_crop(self.registered(expected_model_sha256="b" * 64)).accepted)

    def test_complete_current_crop_requires_margin_even_when_candidate_association_passes(self):
        current = self.current()
        # Same current corners, but only one pixel survives to the crop's right.
        current = replace(current, attempt=replace(current.attempt,
                          roi=ImageRoi(280, 220, 406, 380, 90.)))
        self.assertTrue(self.associate(current).accepted)
        selection = self.registered(current)
        self.assertEqual(selection.current_measured_head_registration.reason, "complete_head_crop_clipped")
        self.assertFalse(review_current_head_crop(selection).accepted)

    def test_previous_proof_cannot_attach_to_another_frame_or_selected_fit(self):
        selection = self.registered()
        other = replace(selection.selected, frame=object())
        copied = replace(selection, selected=other, evaluations=(other,))
        self.assertEqual(review_current_head_crop(copied).reason,
                         "current_head_crop_proof_evaluation_mismatch")

    def test_complete_front_or_unknown_marker_check_cannot_claim_backside_absence(self):
        for current in (self.current(), replace(self.current(backside=True), qr_observations=None)):
            selection = self.registered(current)
            self.assertTrue(review_current_head_crop(selection).accepted)
            self.assertFalse(review_backside_head_crop(selection).accepted)


if __name__ == "__main__":
    unittest.main()
