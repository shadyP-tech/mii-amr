"""Synthetic live epochs exercise the real observer and opposite-view branch.

Only pixel fitting/ROS transport are injected. Classification, LiDAR binding,
freshness, seven-frame consensus, receipt validation and branch selection run.
Recorded-pixel tests separately verify fitting; this is not hardware evidence.
"""

from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE, load_backside_axis_observation,
)
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, execute_candidate_inspection,
)
from tests.aufgabe04 import test_camera_observer_processing as processing_fixture
from tests.aufgabe04.test_head_backside_classification import classified_head


class MeasuredBacksideObserverHandoffTests(unittest.TestCase):
    def observe(self, root, scenario="backside"):
        fixture = processing_fixture.CameraObserverProcessingTest()
        adapter = fixture.make_adapter()
        adapter.args.axis_observation_json = root / "backside.json"
        adapter.args.recommended_pose_json = root / "recommendation.json"
        adapter.node.get_logger = lambda: SimpleNamespace(info=lambda _: None)
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        index = [0]

        def metric(_cv2, _crop, **options):
            estimate, debug, classification_options = classified_head(
                yaw_deg=(-11.6 if index[0] == 6 else -2.8) if scenario == "angle_jump" else 37.815,
                u=options["expected_head_center_u_px"], v=options["expected_head_center_v_px"],
                height=52., profile_sha256=adapter.stand_model_profile.sha256,
            )
            debug = replace(debug, head_neck_junction=None,
                            head_model_quality=replace(debug.head_model_quality,
                                                       centered_neck_supported=False,
                                                       neck_junction_verified=False))
            # Preserve a plain measured plane for one marker frame. Earlier
            # classified samples must never complete after this contradiction.
            if scenario in ("verified_marker", "tentative_marker") and index[0] == 6:
                debug = replace(debug, qr_detected=True,
                                qr_marker_verified=scenario == "verified_marker")
            if scenario == "missing_classification":
                return estimate, debug
            if scenario == "ambiguous_front":
                debug = replace(debug, qr_detected=True, qr_marker_verified=True)
            if scenario == "border_jump" and index[0] == 6:
                estimate = replace(estimate, corners=tuple(
                    ImagePoint(p.u_px + (4 if n in (1, 2) else 0), p.v_px)
                    for n, p in enumerate(estimate.corners)))
            if scenario in ("ambiguous_pose", "ambiguous_front") and index[0] == 6:
                estimate = replace(estimate, usable=False, yaw_deg=None, evidence_state="unobservable")
                debug = replace(debug,
                    head_model_quality=replace(debug.head_model_quality, accepted=False, axis_ambiguous=True),
                    head_pose_hypotheses=tuple(PlanarPoseHypothesis(
                        (0., 0., 0.), (0., 0., .5), (0., 0., 1.), yaw, residual, True)
                        for yaw, residual in ((-7., .2), (7., .25))))
            return classify_current_head_backside(estimate, debug, **classification_options)

        def delayed_debug(*_args, **_kwargs):
            if scenario == "late_publication" and index[0] == 6:
                fixture.clock_sec += .5

        adapter._write_debug.side_effect = delayed_debug
        sensor = adapter._next_sensor_tuple.return_value
        if scenario == "ambiguous_lidar":
            sensor.scan.value.ranges = (.6, .6, float("inf"), .6, .6)
        module = "scripts.aufgabe04.real_robot.observer.node."
        def head_proposal(_cv2, _crop, **options):
            # Only current pixel localization is injected, like the metric
            # fit above. Real recentering, scan binding and crop gates run.
            u, v = options["expected_head_center_u_px"], options["expected_head_center_v_px"]
            corners = tuple(ImagePoint(x, y) for x, y in (
                (u - 26, v - 26), (u + 26, v - 26),
                (u + 26, v + 26), (u - 26, v + 26)))
            proposal = HeadProposal(corners, (int(u - 33), int(v - 33), int(u + 34), int(v + 50)),
                                    (u - 26, v - 26, u + 26, v + 26), u, v, 52.,
                                    52. / options["expected_head_height_px"], 0., .99, .99)
            return HeadProposalResult(proposal, "current_head_proposal", 1, 1)
        with ExitStack() as stack:
            stack.enter_context(patch(
                "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
                side_effect=head_proposal))
            for name, opts in {
                "camera_info_mismatches": dict(return_value=()),
                "transform_mismatches": dict(return_value=()),
                "real_robot_profile_sha256": dict(return_value="a" * 64),
                "camera_calibration_sha256": dict(return_value="b" * 64),
                "compressed_msg_to_bgr_frame": dict(return_value=frame),
                "_rectify_bgr_frame": dict(side_effect=lambda value, *_: value),
                "detect_qr_observations_bgr": dict(return_value=()),
                "detect_native_qr_observations_bgr": dict(return_value=()),
                "estimate_stand_axis_from_metric_model": dict(side_effect=metric),
            }.items():
                stack.enter_context(patch(module + name, **opts))
            for count in range(9 if "marker" in scenario else 7):
                index[0] = count
                if scenario == "fragmented_scan" and count == 6:
                    sensor.scan.value.ranges = (.6, .6, math.nan, .6, .6)
                stamp = 100. + count * .2
                fixture.clock_sec = stamp + .1
                for sample in (sensor.image, sensor.scan, sensor.camera_info):
                    sample.stamp_sec = stamp
                    sample.received_ros_sec = fixture.clock_sec
                ns = round(stamp * 1e9)
                sensor.image.value.header.stamp.sec = ns // 1_000_000_000
                sensor.image.value.header.stamp.nanosec = ns % 1_000_000_000
                if count:
                    adapter.tf_retry_scheduler.offer(sensor, stamp_sec=stamp)
                adapter._process_latest()
                if count < 6:
                    self.assertFalse(adapter.args.axis_observation_json.exists())
        return adapter

    def test_first_view_commits_seven_samples_and_selects_opposite_before_generic_views(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = self.observe(root)
            self.assertTrue(adapter.completed, adapter._write_status.call_args)
            self.assertFalse(adapter.args.recommended_pose_json.exists())
            receipt = load_backside_axis_observation(adapter.args.axis_observation_json)
            payload = json.loads(adapter.args.axis_observation_json.read_text())
            self.assertEqual(payload["axis_sample_source"], REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE)
            self.assertEqual(payload["axis_sample_count"], 7)
            self.assertEqual(payload["motion_capability"], "none")
            self.assertEqual(payload["target_registration"]["eligible_lidar_cluster_count"], 1)
            # The opposing normal must lie on the other side of the stand.
            self.assertLess(math.cos(receipt.opposite_face_normal_rad - math.pi), -.5)
            backside = SimpleNamespace(recommendation_path=None, inspection_observation_path=None,
                                       axis_observation_path=adapter.args.axis_observation_json)
            front = SimpleNamespace(recommendation_path=root / "front.json", qr_id="QR_002")
            moves = []

            def opposite(frame, observation, _output, attempt):
                self.assertIs(observation, backside)
                self.assertEqual(attempt, 1)
                moves.append(load_backside_axis_observation(observation.axis_observation_path).opposite_face_normal_rad)
                return "opposite"

            result, frame = execute_candidate_inspection(
                candidate_uid="candidate", candidate_root=root, initial_frame="first", max_views=8,
                effects=CandidateInspectionEffects(
                    capture=lambda frame, *_: backside if frame == "first" else front,
                    canonical_normal=lambda _: math.pi, move_opposite=opposite,
                    move_view=lambda *_: self.fail("certified backside chose a generic local view"),
                    progress_evidence=lambda *_: {},
                ),
            )
            self.assertIs(result, front)
            self.assertEqual(frame, "opposite")
            self.assertEqual(len(moves), 1)

    def test_absence_alone_ambiguity_marker_epoch_and_late_publication_cannot_commit(self):
        for scenario in ("missing_classification", "ambiguous_lidar", "verified_marker", "late_publication"):
            with self.subTest(scenario=scenario), TemporaryDirectory() as directory:
                adapter = self.observe(Path(directory), scenario)
                self.assertFalse(adapter.completed)
                self.assertFalse(adapter.args.axis_observation_json.exists())
                if scenario == "verified_marker":
                    self.assertTrue(adapter._qr_marker_seen_in_stationary_epoch)
                    self.assertEqual(adapter._last_observation_update.snapshot.current_axis_sample_count_by_source.get(
                        REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE, 0), 0)
                if scenario == "late_publication":
                    self.assertEqual(adapter._write_status.call_args.args, ("obsolete_publication_evidence",))

    def test_tentative_marker_vetoes_its_frame_without_poisoning_later_clear_frames(self):
        with TemporaryDirectory() as directory:
            adapter = self.observe(Path(directory), "tentative_marker")
            self.assertTrue(adapter.completed)
            self.assertFalse(adapter._qr_marker_seen_in_stationary_epoch)
            self.assertTrue(adapter.args.axis_observation_json.exists())

    def test_six_old_angles_plus_an_unstable_current_choice_cannot_commit(self):
        for scenario in ("angle_jump", "border_jump", "ambiguous_pose", "ambiguous_front"):
            with self.subTest(scenario=scenario), TemporaryDirectory() as directory:
                adapter = self.observe(Path(directory), scenario)
                self.assertFalse(adapter.completed)
                self.assertFalse(adapter.args.axis_observation_json.exists())
                update = adapter._last_observation_update
                self.assertTrue(update.frame_accepted)
                self.assertFalse(update.axis_sample_accepted)
                self.assertEqual(update.snapshot.current_axis_sample_count, 0)
                temporal = adapter._head_window_decision
                self.assertFalse(temporal.current_sample_accepted)
                self.assertTrue(temporal.reset_axis_evidence)
                self.assertEqual(temporal.sample_count, 7)
                self.assertEqual(adapter._head_confidence_metadata["backside"]["state"],
                                 "front_marker_veto" if scenario == "ambiguous_front" else "backside_supported")

    def test_real_observer_receipt_preserves_witnessed_current_scan_fragments(self):
        with TemporaryDirectory() as directory:
            adapter = self.observe(Path(directory), "fragmented_scan")
            self.assertTrue(adapter.completed)
            receipt = json.loads(adapter.args.axis_observation_json.read_text())
            self.assertEqual(receipt["schema_version"], 4)
            parsed = load_backside_axis_observation(adapter.args.axis_observation_json)
            self.assertEqual(parsed.axis_sample_count, 7)
            registration = receipt["target_registration"]
            self.assertEqual(registration["eligible_lidar_cluster_count"], 2)
            self.assertFalse(registration["unique_eligible_lidar_cluster_required"])
            proof = registration["witnessed_fragmentation"]
            self.assertEqual(len(proof["witnesses"]), 3)
            self.assertEqual(proof["persistent_target_count"], 1)


if __name__ == "__main__":
    unittest.main()
