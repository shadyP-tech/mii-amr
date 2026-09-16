"""Backside search hints must never carry measurement or motion authority."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint, StandAxisEdgeDebugArtifacts, StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.backside_head_crop import gate_backside_head_crop
from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import (
    BacksideProposalContext, BacksideProposalReuse,
)
from scripts.aufgabe04.real_robot.observer.camera_publication import camera_source_freshness
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_SAMPLE_SOURCE, REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
)
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose, PassiveObserverEvidence
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt, REGISTERED_BACKSIDE_REACQUISITION_SOURCE,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE, TARGET_CENTERED_REACQUISITION_SOURCE,
)
from scripts.aufgabe04.real_robot.observer.head_proposal_registration import acquire_registered_head_measurement


class _Frame:
    """Current pixel-stage fixtures do not need an OpenCV image runtime."""
    def __getitem__(self, _slices):
        return self


class BacksideProposalReuseTest(unittest.TestCase):
    def setUp(self):
        self.reuse = BacksideProposalReuse()
        self.context = BacksideProposalContext(
            target_key="survey_candidate_0001", model_sha256="a" * 64,
            camera_signature=(640, 480, 600.0, 600.0, 320.0, 240.0),
            image_shape=(480, 640, 3),
        )
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.nominal = HeadRoiAttempt(
            roi=ImageRoi(320, 250, 480, 390, 80.0), source="nominal_projection",
            padding_scale=1.8, expected_center_u_px=400.0,
            expected_center_v_px=310.0, expected_head_height_px=80.0,
        )
        self.wide = HeadRoiAttempt(
            roi=ImageRoi(220, 130, 580, 480, 80.0),
            source=TARGET_CENTERED_REACQUISITION_SOURCE, padding_scale=4.5,
            expected_center_u_px=400.0, expected_center_v_px=310.0,
            expected_head_height_px=80.0, backside_target_crop_half_width_ratio=2.25,
        )

    def observe(self, stamp, *, context=None, pose=None, attempts=None,
                center=(310.0, 312.0), yaw_deg=-2.0, usable=True,
                marker_seen=False, tracked_pose=None, enabled=True,
                qr_detected=False, qr_observations=(), estimate_changes=None,
                acquire_current_head=False, proposal_available=True,
                scan_ranges=(.6,) * 5):
        """A deterministic current-image fit; deliberately no elapsed-time claim."""
        calls = []
        frame = _Frame()
        attempts = (self.nominal, self.wide) if attempts is None else attempts
        context = self.context if context is None else context

        def evaluate(attempt, pose_hint):
            calls.append((attempt, pose_hint))
            nominal = attempt.source == "nominal_projection"
            local_u, local_v = center[0] - attempt.roi.x0, center[1] - attempt.roi.y0
            corners = tuple(ImagePoint(local_u + du, local_v + dv)
                            for du, dv in ((-40, -40), (40, -40), (40, 40), (-40, 40)))
            estimate = StandAxisImageEstimate(
                usable=usable and not nominal,
                reason=("model_backside_head_and_neck_unavailable" if nominal or not usable
                        else "axis_estimated_model_backside_current_frame"),
                mode="metric_model_only", corners=None if nominal or not usable else corners,
                axis_line=None, left_height_px=80.0, right_height_px=80.0,
                height_ratio=1.0, yaw_proxy=None, yaw_deg=yaw_deg, closer_side=None,
                contour_area_px=6400.0, source=BACKSIDE_AXIS_SAMPLE_SOURCE,
                evidence_state="fresh_backside", model_profile_sha256=context.model_sha256,
                model_measurement_status="measured", visible_face="backside_candidate",
            )
            if estimate_changes:
                estimate = replace(estimate, **estimate_changes)
            return HeadRoiEvaluation(
                attempt=attempt, frame=frame, estimate=estimate,
                debug=StandAxisEdgeDebugArtifacts(edges=None, qr_detected=qr_detected,
                                                 qr_marker_verified=qr_detected),
                qr_observations=qr_observations,
            )

        def acquire_registered(search, primary):
            u, v = center[0] - search.roi.x0, center[1] - search.roi.y0
            proposal = HeadProposal(
                tuple(ImagePoint(u + du, v + dv) for du, dv in
                      ((-40, -40), (40, -40), (40, 40), (-40, 40))),
                (int(u - 50), int(v - 50), int(u + 50), int(v + 70)),
                (u - 40, v - 40, u + 40, v + 40), u, v, 80., 1.,
                abs(center[0] - search.expected_center_u_px) / 80., .99, .99,
            ) if proposal_available else None
            with patch(
                "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_viewer_candidate_head",
                return_value=HeadProposalResult(proposal, "current_head_proposal" if proposal
                                               else "head_proposal_unavailable", 1, 1),
            ):
                # Only the 2D pixel proposal is supplied by the fixture. Run
                # real bearing, unique scan binding and crop registration.
                return acquire_registered_head_measurement(
                    object(), frame, search,
                    intrinsics=CameraIntrinsics(640, 480, 600., 600., 320., 240.),
                    scan_from_camera=RigidTransform("scan", "camera", (0., 0., 0.),
                                                    (.5, -.5, .5, -.5)),
                    scan=PlainLaserScan(scan_ranges, -.02, .01, .01, 10., "scan",
                                       scan_stamp_sec=stamp, receipt_sec=stamp + .01),
                    map_bearing_rad=-math.atan2(search.expected_center_u_px - 320., 600.),
                    cone_half_angle_rad=math.radians(3.), accepted_range_m=(.49, .71),
                    now_sec=stamp + .1, max_scan_age_sec=.5, min_cluster_sample_count=2,
                    max_camera_map_bearing_delta_rad=math.radians(12.),
                    max_center_offset_ratio=1.5, edge_preprocess="channel_union",
                    canny_low=20, canny_high=60,
                    evaluate=lambda attempt, _corners: evaluate(attempt, None),
                    diagnostics={}, primary=primary,
                )

        selection = self.reuse.select(
            attempts, context=context, observed_at_sec=stamp,
            robot_pose=self.pose if pose is None else pose,
            marker_seen_in_stationary_epoch=marker_seen, tracked_pose=tracked_pose,
            evaluate=evaluate, enable_reacquisition=enabled, max_center_offset_ratio=1.5,
            acquire_registered=acquire_registered if acquire_current_head else None,
        )
        return selection, calls

    def seed(self):
        selection, calls = self.observe(10.0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(selection.registered)
        self.assertTrue(self.reuse.last_metadata["hint_retained"])
        return selection

    def test_warm_hint_relocates_complete_current_head_before_accepting_current_angle(self):
        bootstrap, cold_calls = self.observe(10., acquire_current_head=True)
        self.assertEqual(len(cold_calls), 2)
        self.assertTrue(gate_backside_head_crop(bootstrap)[1].accepted)
        self.assertTrue(self.reuse.last_metadata["hint_retained"])
        current, calls = self.observe(10.3, center=(312., 311.), yaw_deg=-5.,
                                      acquire_current_head=True)
        result, review = gate_backside_head_crop(current)
        self.assertEqual(len(calls), 2)  # Hinted fit, then current proposal's strict fit.
        self.assertTrue(current.search_hint_used)
        self.assertTrue(review.accepted)
        self.assertTrue(result.selected.estimate.usable)
        self.assertEqual(result.selected.estimate.yaw_deg, -5.)
        self.assertEqual(current.head_acquisition["head_bounds_full_image"], [272., 271., 352., 351.])
        self.assertEqual(current.head_acquisition["lidar_association"]["search_association"]
                         ["eligible_cluster_count"], 1)
        self.assertNotEqual(current.selected.attempt.roi, bootstrap.selected.attempt.roi)
        self.assertIsNot(current.selected.frame, bootstrap.selected.frame)
        self.assertTrue(all(pose_hint is None for _, pose_hint in calls))
        self.assertFalse(self.reuse.last_metadata["measurement_reused"])

    def test_usable_warm_fit_cannot_replace_missing_current_head_or_unique_scan(self):
        for failure in ({"proposal_available": False},
                        {"scan_ranges": (.6, .6, math.inf, .6, .6)}):
            with self.subTest(failure=failure):
                self.reuse.reset()
                self.observe(10., acquire_current_head=True)
                current, calls = self.observe(10.3, acquire_current_head=True, **failure)
                self.assertEqual(len(calls), 1)
                self.assertTrue(current.selected.estimate.usable, "the hinted fit alone looks usable")
                result, review = gate_backside_head_crop(current)
                self.assertFalse(review.accepted)
                self.assertFalse(result.selected.estimate.usable)
                self.assertIsNone(result.selected.estimate.yaw_deg)
                self.assertFalse(self.reuse.last_metadata["hint_retained"])

    def test_stale_bootstrap_only_locates_one_strict_fit_of_the_next_image(self):
        bootstrap = self.seed()
        self.assertFalse(camera_source_freshness(
            image_stamp_sec=10.0, scan_stamp_sec=10.0, now_sec=10.824,
            max_age_sec=0.5, max_future_sec=0.05,
        ).accepted)

        current, calls = self.observe(10.9, center=(312.0, 311.0), yaw_deg=-3.25)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].source, REGISTERED_BACKSIDE_REACQUISITION_SOURCE)
        self.assertIsNone(calls[0][1], "an old pose must not seed the new fit")
        self.assertEqual(calls[0][0].roi, self.wide.roi, "full wide QR pixels must remain")
        self.assertEqual(calls[0][0].backside_target_crop_half_width_ratio, 1.25)
        self.assertTrue(current.registered)
        self.assertIsNone(current.proposal)
        self.assertIsNot(current.selected.estimate, bootstrap.selected.estimate)
        self.assertIsNot(current.selected.frame, bootstrap.selected.frame)
        self.assertNotEqual(current.selected.estimate.corners, bootstrap.selected.estimate.corners)
        self.assertEqual(current.selected.estimate.yaw_deg, -3.25)
        self.assertEqual(current.decision.detected_center_u_px, 312.0)
        self.assertEqual(current.decision.projected_center_u_px, 400.0)
        self.assertFalse(self.reuse.last_metadata["measurement_reused"])

    def test_current_fit_is_bounded_against_original_current_projection(self):
        self.seed()
        attempts = tuple(replace(attempt, expected_center_u_px=390.0)
                         for attempt in (self.nominal, self.wide))
        current, calls = self.observe(10.3, center=(260.0, 310.0), attempts=attempts)

        # Current displacement is 130 / 80 = 1.625, even though it moved only
        # 50 pixels from the old hint; chained recentering must not admit it.
        self.assertEqual(len(calls), 1)
        self.assertFalse(current.selected.estimate.usable)
        self.assertFalse(current.registered)
        self.assertEqual(current.selected.estimate.reason, "model_backside_target_center_mismatch")
        self.assertEqual(current.selected.attempt.source, TARGET_CENTERED_REACQUISITION_SOURCE)
        self.assertEqual(self.reuse.last_metadata["current_registration"]["projected_center_u_px"], 390.0)
        self.assertAlmostEqual(self.reuse.last_metadata["current_registration"]["center_offset_ratio"], 1.625)
        self.assertFalse(self.reuse.last_metadata["hint_retained"])

    def test_warm_qr_evidence_remains_visible_and_clears_backside_hint(self):
        decoded = (DecodedQrObservation("QR_002", None, "test-current-image"),)
        for detected, observations in ((True, ()), (False, decoded), (True, decoded)):
            with self.subTest(marker=detected, decoded=bool(observations)):
                self.reuse.reset()
                self.seed()
                selection, calls = self.observe(
                    10.3, qr_detected=detected, qr_observations=observations,
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(selection.selected.debug.qr_detected, detected)
                self.assertEqual(selection.selected.qr_observations, observations)
                if detected:
                    self.assertEqual(selection.selected.attempt.source, REGISTERED_QR_MODEL_REACQUISITION_SOURCE)
                self.assertFalse(self.reuse.last_metadata["hint_retained"])
                self.assertEqual(len(self.observe(10.6)[1]), 3)

    def test_failed_current_fit_does_not_retry_expensive_acquisition_on_same_image(self):
        self.seed()
        failed, calls = self.observe(10.3, usable=False)
        self.assertEqual(len(calls), 1)
        self.assertFalse(failed.selected.estimate.usable)
        self.assertFalse(self.reuse.last_metadata["hint_retained"])
        self.assertEqual(len(self.observe(10.6)[1]), 3)

    def test_expired_duplicate_and_rollback_images_reacquire_without_old_hint(self):
        for stamp in (12.001, 10.0, 9.9):
            with self.subTest(stamp=stamp):
                self.reuse.reset()
                self.seed()
                selection, calls = self.observe(stamp)
                self.assertEqual(len(calls), 3)
                self.assertEqual(calls[0][0], self.nominal)
                self.assertEqual(self.reuse.last_metadata["hint_reason"], "backside_hint_expired_or_nonadvancing_image")
                self.assertIsNotNone(selection.proposal)

    def test_context_changes_invalidate_old_pixels(self):
        for changed in (
            replace(self.context, target_key="survey_candidate_0002"),
            replace(self.context, model_sha256="b" * 64),
            replace(self.context, camera_signature=(640, 480, 590.0, 600.0, 320.0, 240.0)),
            replace(self.context, image_shape=(600, 800, 3)),
        ):
            with self.subTest(context=changed):
                self.reuse.reset()
                self.seed()
                self.assertEqual(len(self.observe(10.3, context=changed)[1]), 3)
                self.assertEqual(self.reuse.last_metadata["hint_reason"], "backside_hint_context_changed")

    def test_motion_is_measured_from_original_anchor_not_previous_hint(self):
        for first, second in (
            (Pose2D(0.006, 0.0), Pose2D(0.012, 0.0)),
            (Pose2D(0.0, 0.0, math.radians(1.2)), Pose2D(0.0, 0.0, math.radians(2.4))),
        ):
            with self.subTest(first=first, second=second):
                self.reuse.reset()
                self.seed()
                self.assertEqual(len(self.observe(10.3, pose=first)[1]), 1)
                self.assertEqual(len(self.observe(10.6, pose=second)[1]), 3)
                self.assertEqual(self.reuse.last_metadata["hint_reason"], "backside_hint_anchor_moved")

    def test_motion_angle_wrap_does_not_invalidate_stationary_hint(self):
        self.pose = Pose2D(0.0, 0.0, math.pi - 0.005)
        self.seed()
        self.assertEqual(len(self.observe(10.3, pose=Pose2D(0.0, 0.0, -math.pi + 0.005))[1]), 1)

    def test_wide_roi_must_contain_nominal_pixels_before_skipping_nominal(self):
        self.seed()
        overlapping = replace(self.wide, roi=ImageRoi(350, 130, 580, 480, 80.0))
        selection, calls = self.observe(10.3, attempts=(self.nominal, overlapping), center=(410.0, 312.0))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], self.nominal)
        self.assertIsNotNone(selection.proposal)
        self.assertEqual(self.reuse.last_metadata["hint_reason"], "backside_hint_would_skip_nominal_pixels")

    def test_stationary_qr_latch_forbids_reusing_or_retaining_backside_hint(self):
        self.seed()
        for stamp in (10.3, 10.6):
            self.assertEqual(len(self.observe(stamp, marker_seen=True)[1]), 3)
            self.assertEqual(self.reuse.last_metadata["hint_reason"], "backside_hint_ineligible")
            self.assertFalse(self.reuse.last_metadata["hint_retained"])

    def test_disabled_registration_and_pose_tracker_bypass_hint(self):
        for kwargs in ({"enabled": False}, {"tracked_pose": object()}):
            with self.subTest(kwargs=kwargs):
                self.reuse.reset()
                self.seed()
                selection, calls = self.observe(10.3, **kwargs)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], self.nominal)
                self.assertFalse(selection.registered)
                self.assertFalse(self.reuse.last_metadata["hint_retained"])

    def test_missing_wide_roi_cannot_skip_nominal(self):
        self.seed()
        self.assertEqual(len(self.observe(10.3, attempts=(self.nominal,))[1]), 1)
        self.assertEqual(self.reuse.last_metadata["hint_reason"], "backside_hint_no_wide_roi")
        self.assertFalse(self.reuse.last_metadata["hint_retained"])

    def test_unverified_model_or_unknown_qr_absence_cannot_seed_hint(self):
        for kwargs in (
            {"estimate_changes": {"model_profile_sha256": "b" * 64}},
            {"estimate_changes": {"evidence_state": "predicted_only"}},
            {"qr_observations": None},
        ):
            with self.subTest(kwargs=kwargs):
                self.reuse.reset()
                self.observe(10.0, **kwargs)
                self.assertFalse(self.reuse.last_metadata["hint_retained"])
                self.assertEqual(len(self.observe(10.3)[1]), 3)

    def test_only_seven_distinct_fresh_current_fits_reach_evidence_consensus(self):
        # Exercise policy composition, not hardware timing or physical angles.
        anchor = EvidencePose(0.0, 0.0, 0.0)
        evidence = PassiveObserverEvidence(
            target_key=self.context.target_key, anchor_pose=anchor,
            required_axis_samples=7, max_axis_deviation_rad=math.radians(8.0),
        )
        self.seed()  # A slow bootstrap can locate the next search, but adds no evidence.
        self.assertFalse(camera_source_freshness(
            image_stamp_sec=10.0, scan_stamp_sec=10.0, now_sec=10.824,
            max_age_sec=0.5, max_future_sec=0.05,
        ).accepted)
        for index in range(7):
            stamp = 11.0 + index * 0.4
            selection, calls = self.observe(stamp, center=(310.0 + index * 0.2, 312.0),
                                            yaw_deg=-2.0 + index * 0.1)
            self.assertEqual(len(calls), 1)
            self.assertTrue(camera_source_freshness(
                image_stamp_sec=stamp, scan_stamp_sec=stamp, now_sec=stamp + 0.3,
                max_age_sec=0.5, max_future_sec=0.05,
            ).accepted)
            update = evidence.record_frame(
                target_key=self.context.target_key, pose=anchor,
                frame_stamp_sec=stamp, lidar_stamp_sec=stamp, observed_at_sec=stamp + 0.3,
                lidar_associated=True, axis_yaw_rad=math.radians(selection.selected.estimate.yaw_deg),
                axis_source=REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE, qr_texts=(),
            )
            self.assertTrue(update.axis_sample_accepted)
            self.assertEqual(update.snapshot.current_axis_sample_count, index + 1)
            if index < 6:
                self.assertIsNone(update.axis_consensus)
        self.assertEqual(update.axis_consensus.sample_count, 7)
        self.assertEqual(update.axis_consensus.source, REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE)
        duplicate = evidence.record_frame(
            target_key=self.context.target_key, pose=anchor,
            frame_stamp_sec=stamp, lidar_stamp_sec=stamp, observed_at_sec=stamp + 0.3,
            lidar_associated=True, axis_yaw_rad=0.9,
            axis_source=REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE, qr_texts=(),
        )
        self.assertFalse(duplicate.axis_sample_accepted)
        self.assertEqual(duplicate.reason, "duplicate_frame_stamp")
        self.assertEqual(duplicate.snapshot.current_axis_sample_count, 7)


if __name__ == "__main__":
    unittest.main()
