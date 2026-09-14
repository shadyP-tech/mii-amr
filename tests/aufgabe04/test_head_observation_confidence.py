"""Backside appearance can be stable while its orientation remains unresolved."""

from dataclasses import replace
import math
from contextlib import redirect_stderr
from io import StringIO
import unittest

from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose, PassiveObserverEvidence
from scripts.aufgabe04.real_robot.observer.head_observation_confidence import (
    HeadConfidenceInput, HeadObservationConfidence,
)
from scripts.aufgabe04.real_robot.observer.inspection_progress import classify_inspection_progress
from tests.aufgabe04.test_head_backside_classification import classified_head


def appearance(*, ambiguous=False, marker=False):
    estimate, debug, options = classified_head(qr_detected=marker, qr_marker_verified=marker)
    debug = replace(debug, head_neck_junction=None, head_model_quality=replace(
        debug.head_model_quality, centered_neck_supported=False, neck_junction_verified=False,
        accepted=not ambiguous, axis_ambiguous=ambiguous))
    if ambiguous:
        estimate = replace(estimate, usable=False, yaw_deg=None, evidence_state="unobservable")
    estimate, debug = classify_current_head_backside(estimate, debug, **options)
    return estimate, debug.head_backside_appearance


class HeadObservationConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.pose = EvidencePose(0., 0., 0.)
        self.evidence = PassiveObserverEvidence(
            target_key="candidate", anchor_pose=self.pose, required_axis_samples=7,
            max_axis_deviation_rad=math.radians(5.), axis_ttl_sec=5.)
        self.confidence = HeadObservationConfidence(required_samples=7, ttl_sec=5.)

    def record(self, stamp, *, complete=True, associated=True, marker=False,
               ambiguous=True, age=.05, pose=None, angle=False):
        estimate, proof = appearance(ambiguous=ambiguous, marker=marker)
        update = self.evidence.record_frame(
            target_key="candidate", pose=pose or self.pose, frame_stamp_sec=stamp,
            lidar_stamp_sec=stamp, observed_at_sec=stamp+age,
            lidar_associated=associated,
            axis_yaw_rad=.05 if angle else None,
            axis_source="model_current_measured_head" if angle else None)
        return self.confidence.observe(
            HeadConfidenceInput(stamp, proof, complete, marker, marker,
                                estimate.usable, estimate.reason),
            update=update, observed_at_sec=stamp+age,
            angle_temporally_consistent=angle)

    def test_ambiguous_angle_does_not_erase_valid_neck_independent_appearance(self):
        estimate, proof = appearance(ambiguous=True)
        self.assertFalse(estimate.usable)
        self.assertIsNone(estimate.yaw_deg)
        self.assertTrue(proof.accepted)
        self.assertFalse(proof.supplies_angle)
        for n in range(7):
            result = self.record(100+n*.1)
        self.assertEqual(result["backside"]["state"], "backside_supported")
        self.assertFalse(result["angle"]["observable_now"])
        self.assertFalse(result["angle"]["consensus_ready"])
        self.assertFalse(result["motion_authorized"])
        progress = classify_inspection_progress("metric_model_measurement_unavailable", {
            "stand_axis_debug": {"advisory_camera_relative_yaw_rad": .2,
                                 "metric_model": {"observation_confidence": result}}})
        self.assertEqual(progress.classification, "backside_unresolved")
        self.assertIsNone(progress.camera_relative_yaw_rad)

    def test_qr_absence_without_complete_associated_raw_head_is_insufficient(self):
        for change in ({"complete": False}, {"associated": False}, {"age": .6}):
            self.setUp()
            for n in range(8):
                result = self.record(100+n*.1, **change)
            self.assertEqual(result["backside"]["sample_count"], 0)
        estimate, debug, options = classified_head()
        for changed in (replace(debug, head_model_quality=None),
                        replace(debug, head_model_quality=replace(debug.head_model_quality,
                                                                 outer_border_verified=False)),
                        replace(debug, qr_marker_verified=None)):
            _, proof = classify_current_head_backside(estimate, changed, **options)
            self.assertFalse(proof.head_backside_appearance.accepted)

    def test_verified_front_marker_vetoes_complete_and_clipped_views_and_epoch_history(self):
        for complete in (True, False):
            self.setUp()
            for n in range(7):
                self.record(100+n*.1, ambiguous=False, angle=True)
            veto = self.record(101., complete=complete, marker=True, ambiguous=False, angle=True)
            self.assertEqual(veto["backside"]["state"], "front_marker_veto")
            self.assertEqual(veto["backside"]["sample_count"], 0)
            for n in range(7):
                veto = self.record(102+n*.1)
            self.assertEqual(veto["backside"]["state"], "front_marker_veto")

    def test_duplicate_stale_and_new_epoch_cannot_reuse_seven_old_samples(self):
        for _ in range(7):
            result = self.record(100.)
        self.assertEqual(result["backside"]["sample_count"], 1)
        result = self.record(106.)
        self.assertEqual(result["backside"]["sample_count"], 1)
        for n in range(6):
            result = self.record(106.1+n*.1)
        self.assertEqual(result["backside"]["state"], "backside_supported")
        result = self.record(107., pose=EvidencePose(.1, 0., 0.))
        self.assertEqual(result["backside"]["sample_count"], 1)
        self.assertEqual(result["motion_epoch"], 1)

    def test_temporal_configuration_is_rejected_before_starting_ros(self):
        from scripts.aufgabe04.real_robot.observer.node import build_parser, _validate_args
        required = ["--robot-profile", "robot.json", "--camera-calibration", "camera.json",
                    "--stand-model-profile", "stand.json", "--status-json", "status.json",
                    "--recommended-pose-json", "recommendation.json", "--stream-id", "test",
                    "--stand-id", "candidate", "--expected-qr-id", "auto",
                    "--stand-x", ".6", "--stand-y", "0"]
        for options in (("--consensus-frames", "33"),
                        ("--consensus-axis-ttl-sec", "61"),
                        ("--consensus-max-deviation-deg", "91")):
            parser = build_parser()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _validate_args(parser, parser.parse_args([*required, *options]))


if __name__ == "__main__":
    unittest.main()
