"""Stopped tour scan evidence is source-timed, stationary, and immutable."""

import copy
from collections import Counter, deque
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.real_robot.readiness import tour_scan_capture as capture
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import (
    HASH_FIELD, capture_payload, scan_evidence,
)


def stamp(value):
    sec = int(value)
    return NS(sec=sec, nanosec=round((value-sec)*1e9))


def transform(value, source="laser", x=.04, yaw=0.):
    return NS(header=NS(frame_id="odom", stamp=stamp(value)), child_frame_id=source,
        transform=NS(translation=NS(x=x, y=0., z=0.),
                     rotation=NS(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))))


def sample(value=100.):
    message = NS(header=NS(frame_id="laser", stamp=stamp(value)),
        angle_min=-1., angle_increment=.1, range_min=.05, range_max=8.,
        ranges=[.5, float("inf"), float("nan"), .01, 10., .75])
    return scan_evidence(message, received_at_unix_sec=value+.01,
        scan_transform=transform(value), base_transform=transform(value, "base", 0.),
        odom_frame="odom", base_frame="base", scan_frame="laser")


def payload(scans=None, now=100.21):
    return capture_payload(scans or [sample(100.), sample(100.1), sample(100.2)],
        tour_id="tour", odom_frame="odom", base_frame="base", scan_frame="laser",
        captured_at_unix_sec=now)


class TourScanEvidenceTest(unittest.TestCase):
    def test_invalid_rays_remain_unknown_and_scanner_offset_is_preserved(self):
        result = payload()
        self.assertEqual(result["scans"][0]["ranges"], [.5, None, None, None, None, .75])
        self.assertEqual(result["scans"][0]["scan_pose_odom"]["x_m"], .04)
        self.assertEqual(result["scans"][0]["base_pose_odom"]["x_m"], 0.)
        json.dumps(result, allow_nan=False)

    def test_exact_transform_source_time_and_identity_are_mandatory(self):
        msg = NS(header=NS(frame_id="laser", stamp=stamp(100.)), angle_min=0.,
            angle_increment=.1, range_min=.05, range_max=8., ranges=[.5, .6])
        for transform_change in (transform(99.99), transform(100., "wrong")):
            with self.subTest(change=transform_change), self.assertRaises(ValueError):
                scan_evidence(msg, received_at_unix_sec=100.01,
                    scan_transform=transform_change, base_transform=transform(100., "base"),
                    odom_frame="odom", base_frame="base", scan_frame="laser")

    def test_distinct_stamp_spacing_window_freshness_and_stationarity(self):
        cases = []
        cases.append([sample(100.), sample(100.), sample(100.2)])
        cases.append([sample(100.), sample(100.05), sample(100.2)])
        cases.append([sample(99.6), sample(100.1), sample(100.2)])
        moved = [sample(100.), sample(100.1), sample(100.2)]
        moved[-1]["base_pose_odom"]["x_m"] = .016
        cases.append(moved)
        turned = copy.deepcopy(moved)
        turned[-1]["base_pose_odom"] = {"x_m": 0., "y_m": 0., "yaw_rad": math.radians(2.1)}
        cases.append(turned)
        stale_receipt = [sample(100.), sample(100.1), sample(100.2)]
        stale_receipt[0]["received_at_unix_sec"] = 100.26
        cases.append(stale_receipt)
        wrong_tf = [sample(100.), sample(100.1), sample(100.2)]
        wrong_tf[0]["scan_pose_stamp_sec"] = 99.
        cases.append(wrong_tf)
        for scans in cases:
            with self.subTest(scans=scans), self.assertRaises(ValueError):
                payload(scans)
        with self.assertRaisesRegex(ValueError, "latest capture scan"):
            payload(now=100.46)

    def test_cohort_can_span_point_four_seconds_with_each_scan_fresh_at_receipt(self):
        self.assertEqual(len(payload([sample(99.8), sample(100.), sample(100.2)])["scans"]), 3)

    def test_capture_rejects_changed_scanner_mount_and_invalid_persisted_rays(self):
        scans = [sample(100.), sample(100.1), sample(100.2)]
        scans[-1]["scan_pose_odom"]["x_m"] += .006
        with self.assertRaisesRegex(ValueError, "mounting"):
            payload(scans)
        scans = [sample(100.), sample(100.1), sample(100.2)]
        scans[-1]["ranges"][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            payload(scans)

    def test_collector_publishes_hashed_artifact_and_cleans_up_node(self):
        node = NS(payload=payload(), poll=Mock(), destroy_node=Mock())
        ros = NS(ok=Mock(return_value=True), spin_once=Mock(), init=Mock(), shutdown=Mock())
        with tempfile.TemporaryDirectory() as directory, patch.object(capture, "rclpy", ros), \
             patch.object(capture, "_TourScanNode", return_value=node):
            output = Path(directory)/"scan.json"
            self.assertEqual(capture.capture_tour_scan(NS(), tour_id="tour", output_path=output), output)
            self.assertEqual(load_content_hashed_json(output, hash_field=HASH_FIELD), payload())
            with self.assertRaisesRegex(ValueError, "fresh"):
                capture.capture_tour_scan(NS(), tour_id="tour", output_path=output)
        node.destroy_node.assert_called_once()
        ros.init.assert_not_called()
        ros.shutdown.assert_not_called()

    def test_timeout_leaves_no_accepted_artifact(self):
        diagnostics = {"rejection_attempt_counts": {"exact_time_transform_unavailable": 12},
                       "accepted_scan_count": 2, "last_error": "missing exact-time TF"}
        node = NS(payload=None, poll=Mock(), destroy_node=Mock(), last_error="missing exact-time TF",
                  capture_diagnostics=lambda: diagnostics)
        ros = NS(ok=Mock(return_value=True), spin_once=Mock(), init=Mock(), shutdown=Mock())
        with tempfile.TemporaryDirectory() as directory, patch.object(capture, "rclpy", ros), \
             patch.object(capture, "_TourScanNode", return_value=node), \
             patch.object(capture.time, "monotonic", side_effect=[0., 4.]):
            output = Path(directory)/"scan.json"
            with self.assertRaisesRegex(capture.TourScanCaptureError, "missing exact-time TF") as caught:
                capture.capture_tour_scan(NS(), tour_id="tour", output_path=output)
            self.assertFalse(output.exists())
            self.assertTrue(caught.exception.retryable)
            self.assertEqual(caught.exception.reason_code, "stationary_scan_capture_timeout")
            self.assertEqual(caught.exception.diagnostics, diagnostics)
        node.destroy_node.assert_called_once()

    def test_missing_ros_dependencies_are_terminal_not_a_passive_timeout(self):
        with patch.object(capture, "rclpy", None), self.assertRaises(capture.TourScanCaptureError) as caught:
            capture.capture_tour_scan(NS(), tour_id="tour", output_path=Path("unused.json"))
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.reason_code, "scan_capture_dependencies_unavailable")

    def test_poll_diagnostics_retain_tf_timestamp_and_stationarity_rejections(self):
        class MissingTransform(Exception):
            pass
        node = capture._TourScanNode.__new__(capture._TourScanNode)
        node.profile = NS(odom_frame="odom", base_frame="base", scan_frame="laser")
        node.tour_id = "tour"
        node.observation_not_before_sec = 100.
        node.rejection_counts = Counter()
        node.accepted_scan_count = 0
        node.last_error = "no fresh scans received"
        node.pending = deque((NS(header=NS(stamp=stamp(value))), value)
                             for value in (99.9, 100., 100.1, 100.2, 100.3))
        node.samples = deque(maxlen=3)
        node.payload = None
        node.buffer = NS(lookup_transform=Mock(side_effect=MissingTransform("not ready")))
        node.payload_builder = Mock(side_effect=ValueError("robot moved during stopped scan capture"))
        with patch.object(capture, "TransformException", MissingTransform, create=True), \
             patch.object(capture, "Time", NS(from_msg=lambda value: value), create=True), \
             patch.object(capture, "Duration", lambda **kwargs: kwargs, create=True), \
             patch.object(capture.time, "time", return_value=100.3), \
             patch.object(capture, "scan_evidence", side_effect=lambda message, **_: sample(
                 capture.stamp_seconds(message.header.stamp))):
            node.poll()
            node.buffer.lookup_transform.side_effect = None
            node.poll()
        self.assertIsNone(node.payload)
        self.assertEqual(node.capture_diagnostics(), {
            "rejection_attempt_counts": {"scan_before_observation_floor": 1,
                "source scan expired before exact-time TF": 1,
                "exact_time_transform_unavailable": 1,
                "robot moved during stopped scan capture": 1},
            "accepted_scan_count": 3, "pending_scan_count": 0, "cohort_sample_count": 3,
            "last_error": "robot moved during stopped scan capture"})


if __name__ == "__main__":
    unittest.main()
