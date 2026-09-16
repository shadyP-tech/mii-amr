"""A bounded current head can associate and track without a unique angle."""

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import cv2

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import (
    evaluate_current_head_orientation_bounds, validated_current_head_orientation_bounds,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix, estimate_planar_pose_ippe
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.backside_head_crop import review_current_head_crop
from scripts.aufgabe04.real_robot.observer.candidate_head_tracking import (
    CandidateHeadContext, CandidateHeadTracking,
)
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.registration_evidence import build_backside_target_registration_evidence
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import (
    register_current_tracked_head, tracked_head_selection,
)
from tests.aufgabe04 import test_current_head_association as association_fixtures
from tests.aufgabe04 import test_tracked_head_registration as tracking_fixtures
from tests.aufgabe04.test_head_model_admission import outer_boundary


def bounded_detection():
    """Real IPPE/covariance on synthetic pixels; no recording accuracy claim."""
    profile = load_measured_physical_stand_model(Path(__file__).resolve().parents[2] /
        "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
    current = tracking_fixtures.TrackedHeadRegistrationTests().current()
    corners = current.estimate.corners
    camera = RectifiedCameraMatrix(640., 640., 120., 80.)
    pose = estimate_planar_pose_ippe(cv2, corners, profile.head_corners, camera)
    bounds = evaluate_current_head_orientation_bounds(
        cv2, profile=profile, camera=camera, corners=corners, pose_result=pose,
        raw_border_support_mean=.98, raw_corner_support_accepted=True,
        outer_border_verified=True, frame_shape=(160, 160))
    assert bounds.accepted, bounds.reason
    estimate = replace(current.estimate, usable=False, yaw_deg=None,
        evidence_state="unobservable", reason="head_model_planar_axis_ambiguous",
        model_profile_sha256=profile.sha256, left_height_px=0., right_height_px=0.)
    debug = replace(current.debug, model_pose=None, evidence_state="unobservable",
        model_profile_sha256=profile.sha256, refined_corners=corners,
        head_orientation_bounds=bounds,
        head_outer_recovery=outer_boundary(corners, profile.sha256),
        head_model_quality=replace(current.debug.head_model_quality, accepted=False,
            reason="head_model_planar_axis_ambiguous", axis_ambiguous=True,
            profile_sha256=profile.sha256))
    return replace(current, estimate=estimate, debug=debug), bounds


class BoundedHeadDetectionTests(unittest.TestCase):
    def options(self, current=None):
        if current is None:
            current, _ = bounded_detection()
        return {**association_fixtures.CurrentHeadAssociationTests().options(),
                "estimate": current.estimate, "debug": current.debug,
                "attempt": current.attempt, "profile_sha256": current.estimate.model_profile_sha256}

    def test_current_bound_associates_without_promoting_strict_angle(self):
        current, bounds = bounded_detection()
        self.assertTrue(validated_current_head_orientation_bounds(
            bounds, estimate=current.estimate, debug=current.debug))
        result = associate_current_measured_head(**self.options(current))
        self.assertTrue(result.accepted, result.reason)
        self.assertFalse(result.head_admission.accepted)
        self.assertIs(result.head_orientation_bounds, bounds)
        self.assertAlmostEqual(result.scale_gate["measured_height_px"], 90.)
        self.assertFalse(current.estimate.usable)
        self.assertIsNone(current.estimate.yaw_deg)
        self.assertFalse(admit_measured_head_model(
            estimate=current.estimate, debug=current.debug, yaw_rad=bounds.center_rad).accepted)
        metadata = result.metadata()
        self.assertTrue(metadata["bounded_head_detection"])
        self.assertFalse(metadata["single_angle_admitted"])
        self.assertFalse(metadata["motion_authorized"])
        json.dumps(metadata, allow_nan=False)

    def test_proof_cannot_borrow_other_pixels_provenance_or_raw_boundary(self):
        current, bounds = bounded_detection()
        cases = (
            replace(current, debug=replace(current.debug, head_orientation_bounds=None)),
            replace(current, debug=replace(current.debug,
                head_orientation_bounds=replace(bounds, half_width_rad=bounds.half_width_rad / 2))),
            replace(current, estimate=replace(current.estimate, evidence_state="predicted_only")),
            replace(current, estimate=replace(current.estimate, source="model_projection")),
            replace(current, debug=replace(current.debug, model_pose_fit_source="joint_qr_head")),
            replace(current, debug=replace(current.debug,
                head_outer_recovery=replace(current.debug.head_outer_recovery, accepted=False))),
            replace(current, estimate=replace(current.estimate,
                corners=tuple(ImagePoint(p.u_px + 1, p.v_px) for p in current.estimate.corners))),
        )
        for changed in cases:
            with self.subTest(changed=changed.estimate.reason):
                self.assertFalse(associate_current_measured_head(**self.options(changed)).accepted)

    def test_bound_keeps_original_projection_scan_freshness_and_uniqueness_gates(self):
        options = self.options()
        for changed in (
            {"now_sec": 10.6}, {"accepted_range_m": (.8, 1.)},
            {"projection": replace(options["projection"], u_px=550.)},
            {"intrinsics": replace(options["intrinsics"], cx_px=401.)},
            {"scan": replace(options["scan"],
                             ranges=(.55, .55, math.inf, .55, .55, .55, .55, .55, .55))},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(associate_current_measured_head(**{**options, **changed}).accepted)

    def register(self, current, association, **changes):
        return register_current_tracked_head(tracked_head_selection(current), **{
            "association": association, "observed_at_sec": 10., "now_sec": 10.4,
            "max_age_sec": .5, "expected_model_sha256": current.estimate.model_profile_sha256,
            **changes})

    def test_tracked_crop_requires_the_same_current_bounded_detection_and_scan(self):
        current, bounds = bounded_detection()
        association = associate_current_measured_head(**self.options(current))
        selection = self.register(current, association)
        self.assertTrue(review_current_head_crop(selection).accepted)
        self.assertFalse(selection.selected.estimate.usable)
        for changed in (replace(association, head_orientation_bounds=None),
                        replace(association, full_image_center_px=(380., 300.))):
            self.assertFalse(review_current_head_crop(self.register(current, changed)).accepted)
        self.assertFalse(review_current_head_crop(
            self.register(current, association, now_sec=10.51)).accepted)
        clipped = replace(current, attempt=replace(current.attempt,
                            roi=ImageRoi(280, 220, 406, 380, 90.)))
        clipped_association = associate_current_measured_head(**self.options(clipped))
        self.assertFalse(review_current_head_crop(self.register(clipped, clipped_association)).accepted)

    def test_bounded_pose_only_seeds_a_fresh_stopped_search_locator(self):
        current, bounds = bounded_detection()
        context = CandidateHeadContext("candidate", bounds.profile_sha256,
                                      (640., 640., 400., 300.), (600, 800, 3), 0)
        tracker = CandidateHeadTracking()
        self.assertTrue(tracker.remember(current, context=context,
            observed_at_sec=10., now_sec=10.4, max_age_sec=.5,
            robot_pose=Pose2D(0., 0.), candidate_associated=True))
        wide = replace(current.attempt, roi=ImageRoi(100, 80, 650, 550, 90.))
        hint = tracker.hint((current.attempt, wide), context=context,
                           observed_at_sec=10.6, robot_pose=Pose2D(0., 0.))
        self.assertIsNotNone(hint)
        self.assertEqual(hint.pose_hint.rotation_vector, bounds.hypotheses[0].rotation_vector)
        self.assertFalse(tracker.last_metadata["motion_authorized"])
        self.assertFalse(current.estimate.usable)
        self.assertIsNone(tracker.hint((current.attempt, wide), context=context,
                                      observed_at_sec=10.7, robot_pose=Pose2D(.02, 0.)))

    def test_bounded_registration_receipt_requires_explicit_opt_in_and_exact_proof(self):
        current, bounds = bounded_detection()
        association = associate_current_measured_head(**self.options(current))
        wrapper = association.lidar_association
        options = dict(final_head_center_error_ratio=association.center_offset_ratio,
                       current_head_association=association, registered_lidar_association=wrapper,
                       candidate_lidar_association=wrapper.search_association)
        with self.assertRaises(ValueError):
            build_backside_target_registration_evidence(**options)
        evidence = build_backside_target_registration_evidence(
            **options, allow_bounded_orientation=True, head_orientation_bounds=bounds)
        self.assertTrue(evidence["unique_eligible_lidar_cluster_required"])
        for changes in ({"head_orientation_bounds": None},
                        {"allow_bounded_orientation": False},
                        {"head_orientation_bounds": replace(bounds, accepted=False)},
                        {"candidate_lidar_association": replace(wrapper.search_association,
                                                                 scan_stamp_sec=9.)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_backside_target_registration_evidence(**{
                    **options, "allow_bounded_orientation": True,
                    "head_orientation_bounds": bounds, **changes})


if __name__ == "__main__":
    unittest.main()
