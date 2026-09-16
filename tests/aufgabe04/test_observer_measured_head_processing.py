"""Adapter integration with synthetic independent head/QR contracts.

The mocked quality-approved 37.815-degree measurement checks angle admission,
not the semantic admissibility or ground-truth angle of a recorded image.
"""

from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy

from scripts.aufgabe04.perception.stand_axis.models import ImagePoint, StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MEASURED_HEAD_AXIS_SOURCE
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_head_model_admission as head_fixtures


class MeasuredHeadObserverProcessingTests(unittest.TestCase):
    def run_view(self, scenario):
        shifted = scenario.startswith("shifted_")
        scenario = scenario.removeprefix("shifted_")
        registered = scenario.startswith("registered_")
        scenario = scenario.removeprefix("registered_")
        fixture = processing_fixtures.CameraObserverProcessingTest()
        adapter = fixture.make_adapter()
        if shifted:
            # Complete head remains inside this synthetic nominal crop while
            # its ray lies outside the original map-centered three-degree cone.
            adapter.args.head_roi_padding_scale = 2.2
        adapter.node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        current_index = [0]
        decode_modes = []
        pose_hints = []

        def decode(crop, _cv2, *, diagnostics=None, max_elapsed_sec=None):
            index = current_index[0]
            if scenario == "head_only" or scenario == "historical_qr" and index >= 2:
                return ()
            # The same physical symbol stays at a full-image position as the
            # candidate search recenters its crop and adjusts intrinsics.
            pixel_offset = (crop.ctypes.data - frame.ctypes.data) // 3
            origin_y, origin_x = divmod(pixel_offset, frame.shape[1])
            u = 400. - origin_x - (28. if shifted else 0.) + (10. if registered else 0.)
            v = 300. - origin_y
            corners = tuple((u + x, v + y) for x, y in
                            ((-15, -15), (15, -15), (15, 15), (-15, 15)))
            observations = (DecodedQrObservation("QR_003", None if scenario == "unbound_qr" else corners,
                                                "test_decoder", 1.),)
            if scenario == "conflicting_qr" and index == 5:
                return (*observations, DecodedQrObservation("QR_004", corners, "test_decoder", 1.))
            return observations

        def full_decode(*args, **kwargs):
            decode_modes.append("full")
            return decode(*args, **kwargs)

        def native_decode(*args, **kwargs):
            decode_modes.append("native")
            if scenario == "native_miss" and current_index[0] == 2:
                return ()
            return decode(*args, **kwargs)

        def metric(_cv2, _crop, **options):
            pose_hints.append(options["pose_hint"])
            u, v = options["expected_head_center_u_px"], options["expected_head_center_v_px"]
            if shifted:
                u -= 28.
            if registered and options["pose_hint"] is not None:
                u += 10.
            if scenario == "wrong_head_bearing":
                u += 70.
            corners = tuple(ImagePoint(u + x, v + y) for x, y in
                            ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            quality = head_fixtures.quality(profile_sha256=adapter.stand_model_profile.sha256)
            if scenario == "uncertain_head":
                quality = replace(quality, yaw_std_deg=3.1)
            estimate = head_fixtures.head_estimate(
                corners=corners, left_height_px=52., right_height_px=52.,
                model_profile_sha256=adapter.stand_model_profile.sha256,
            )
            if (registered and options["current_head_proposal_corners"] is None
                    and options["pose_hint"] is None):
                estimate = replace(estimate, usable=False, yaw_deg=None,
                                   reason="head_proposal_unavailable")
            debug = StandAxisEdgeDebugArtifacts(
                edges=None, model_pose_fit_source=MEASURED_HEAD_AXIS_SOURCE,
                evidence_state="fresh_refined", model_measurement_status="measured",
                model_profile_sha256=adapter.stand_model_profile.sha256,
                head_model_quality=quality, qr_detected=bool(options["qr_observations"]),
                head_outer_recovery=head_fixtures.outer_boundary(
                    corners, adapter.stand_model_profile.sha256),
                qr_marker_verified=bool(options["qr_observations"]),
                model_pose=PlanarPoseHypothesis((0., -.66, 0.), (0., 0., .6),
                                               (-.61, 0., .79), 37.815, .457, True),
            )
            if scenario == "panel_then_recovery" and current_index[0] == 6:
                estimate = replace(estimate, usable=False, yaw_deg=None,
                    evidence_state="unobservable", reason="current_physical_head_boundary_unresolved")
                debug = replace(debug, model_pose=None,
                    head_outer_recovery=replace(debug.head_outer_recovery, accepted=False,
                        reason="current_physical_head_boundary_unresolved"),
                    head_model_quality=replace(quality, accepted=False, outer_border_verified=False,
                        reason="current_physical_head_boundary_unresolved"),
                    head_pose_hypotheses=(PlanarPoseHypothesis(
                        (0., 0., 0.), (0., 0., .5), (0., 0., 1.), -16.121, .2, True),))
            return estimate, debug

        def locate(_cv2, _crop, **options):
            u, v = options["expected_head_center_u_px"] + 10., options["expected_head_center_v_px"]
            corners = tuple(ImagePoint(u + x, v + y) for x, y in
                            ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            return HeadProposalResult(HeadProposal(
                corners, (int(u - 34), int(v - 34), int(u + 34), int(v + 55)),
                (u - 26, v - 26, u + 26, v + 26), u, v, 52., 1., 10 / 52., .98, .98,
            ), "current_head_proposal", 1, 1, "test_locator")

        def delayed_debug(*_args, **_kwargs):
            if scenario == "late_publication" and current_index[0] == 6:
                fixture.clock_sec += .5

        adapter._write_debug.side_effect = delayed_debug
        module = "scripts.aufgabe04.real_robot.observer.node."
        with TemporaryDirectory() as directory, ExitStack() as stack:
            output = adapter.args.recommended_pose_json = Path(directory) / "recommendation.json"
            adapter.args.axis_observation_json = Path(directory) / "backside.json"
            patches = {
                "camera_info_mismatches": {"return_value": ()},
                "transform_mismatches": {"return_value": ()},
                "real_robot_profile_sha256": {"return_value": "a" * 64},
                "camera_calibration_sha256": {"return_value": "b" * 64},
                "compressed_msg_to_bgr_frame": {"return_value": frame},
                "_rectify_bgr_frame": {"side_effect": lambda value, *_: value},
                "detect_qr_observations_bgr": {"side_effect": full_decode},
                "detect_native_qr_observations_bgr": {"side_effect": native_decode},
                "estimate_stand_axis_from_metric_model": {"side_effect": metric},
            }
            for name, options in patches.items():
                stack.enter_context(patch(module + name, **options))
            if registered:
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
                    side_effect=locate,
                ))
            if scenario == "panel_then_recovery":
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
                    return_value=HeadProposalResult(None, "head_proposal_unavailable")))
            backside = stack.enter_context(patch(module + "build_backside_axis_observation"))
            sensor_tuple = adapter._next_sensor_tuple.return_value
            if shifted:
                sensor_tuple.scan.value.angle_min = math.radians(3.5)
                sensor_tuple.scan.value.angle_increment = math.radians(.2)
            if scenario == "wrong_lidar":
                sensor_tuple.scan.value.ranges = (3.,) * 5
            if scenario == "ambiguous_lidar":
                sensor_tuple.scan.value.ranges = (.6, .6, float("inf"), .6, .6)
            for index in range(8 if scenario == "panel_then_recovery" else 7):
                current_index[0] = index
                stamp = 100. + index * (.6 if scenario == "slow_bootstrap" else .2)
                fixture.clock_sec = stamp + (.4 if scenario == "slow_bootstrap" else .1)
                for sample in (sensor_tuple.image, sensor_tuple.scan, sensor_tuple.camera_info):
                    sample.stamp_sec = stamp
                    sample.received_ros_sec = fixture.clock_sec
                stamp_ns = round(stamp * 1e9)
                sensor_tuple.image.value.header.stamp.sec = stamp_ns // 1_000_000_000
                sensor_tuple.image.value.header.stamp.nanosec = stamp_ns % 1_000_000_000
                if index:
                    adapter.tf_retry_scheduler.offer(sensor_tuple, stamp_sec=stamp)
                adapter._process_latest()
                if scenario == "panel_then_recovery" and index == 6:
                    self.assertFalse(adapter.completed)
                    self.assertFalse(output.exists())
                    self.assertEqual(adapter._last_observation_update.snapshot.current_axis_sample_count, 6)
                    self.assertIsNone(adapter._head_window_decision)
            backside.assert_not_called()
            self.assertFalse(adapter.args.axis_observation_json.exists())
            payload = json.loads(output.read_text()) if output.exists() else None
            adapter._test_decode_modes = decode_modes
            adapter._test_pose_hints = pose_hints
            return adapter, payload

    def test_seven_quality_head_frames_and_independent_bound_qr_commit_above_35_degrees(self):
        adapter, payload = self.run_view("bound_qr")
        self.assertTrue(adapter.completed)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["axis"]["sample_count"], 7)
        measurement = payload["axis_measurement"]
        self.assertEqual(measurement["source"], MEASURED_HEAD_AXIS_SOURCE)
        self.assertEqual(measurement["head_model_quality"]["pose_model"], "measured_head_only")
        self.assertFalse(measurement["sample_admission"]["qr_bound_model_fallback"])
        self.assertAlmostEqual(abs(measurement["sample_admission"]["yaw_rad"]), math.radians(37.815))
        self.assertEqual(adapter._last_observation_update.resolved_qr_id, "QR_003")

    def test_observer_fresh_slow_seed_tracks_seven_front_frames_despite_legacy_ttl(self):
        adapter, payload = self.run_view("slow_bootstrap")
        self.assertTrue(adapter.completed)
        self.assertEqual(payload["axis"]["sample_count"], 7)
        self.assertIsNone(adapter._test_pose_hints[0])
        self.assertTrue(all(hint is not None for hint in adapter._test_pose_hints[1:]))
        self.assertEqual(adapter._candidate_search().last_metadata["reason"],
                         "current_associated_head_retained")

    def test_qr_front_head_survives_rejected_panel_and_commits_on_fresh_seventh_fit(self):
        adapter, payload = self.run_view("panel_then_recovery")
        self.assertTrue(adapter.completed)
        self.assertEqual(payload["axis"]["sample_count"], 7)
        self.assertEqual(payload["axis_measurement"]["source"], MEASURED_HEAD_AXIS_SOURCE)
        self.assertEqual(adapter._head_window_decision.sample_count, 7)
        self.assertNotIn(101.2, adapter._head_window_decision.window_stamps_sec)
        self.assertEqual(adapter._last_observation_update.resolved_qr_id, "QR_003")

    def test_head_axis_cannot_certify_a_face_from_missing_unbound_or_historical_qr(self):
        for scenario in ("head_only", "unbound_qr", "historical_qr"):
            with self.subTest(scenario=scenario):
                adapter, payload = self.run_view(scenario)
                self.assertIsNone(payload)
                self.assertFalse(adapter.completed)
                self.assertEqual(adapter._last_observation_update.axis_consensus.sample_count, 7)
                self.assertEqual(adapter._write_status.call_args.args, ("axis_observation_not_committable",))
                self.assertEqual(adapter._write_status.call_args.kwargs["reason"],
                                 "measured_head_front_identity_unresolved")

    def test_geometry_quality_lidar_conflict_and_publication_freshness_remain_required(self):
        for scenario in ("uncertain_head", "wrong_lidar", "wrong_head_bearing", "ambiguous_lidar",
                         "conflicting_qr", "late_publication"):
            with self.subTest(scenario=scenario):
                adapter, payload = self.run_view(scenario)
                self.assertIsNone(payload)
                self.assertFalse(adapter.completed)
                if scenario == "late_publication":
                    self.assertEqual(adapter._write_status.call_args.args, ("obsolete_publication_evidence",))
                elif scenario == "conflicting_qr":
                    self.assertTrue(adapter._last_observation_update.snapshot.poisoned)
                else:
                    self.assertFalse(adapter._last_observation_update.axis_sample_accepted)

    def test_native_head_track_keeps_periodic_empty_search_without_repeated_pyramids(self):
        head_only, _ = self.run_view("head_only")
        self.assertEqual(head_only._test_decode_modes.count("native"), 7)
        # Empty-symbol checks remain periodic after the crop changes.
        self.assertLessEqual(head_only._test_decode_modes.count("full"), 2)
        bound, _ = self.run_view("bound_qr")
        self.assertEqual(bound._test_decode_modes, ["native"] * 7)
        recovered, payload = self.run_view("native_miss")
        self.assertIsNotNone(payload)
        self.assertEqual(recovered._test_decode_modes.count("native"), 7)
        self.assertEqual(recovered._last_observation_update.resolved_qr_id, "QR_003")

    def test_recentered_head_uses_current_fitted_ray_and_unique_lidar_before_consensus(self):
        accepted, payload = self.run_view("registered_bound_qr")
        self.assertIsNotNone(payload)
        registration = accepted._write_status.call_args.kwargs["stand_axis_debug"]["metric_model"]["camera_target_registration"]
        self.assertFalse(registration["strict_retry_applied"])
        self.assertTrue(registration["current_measured_head_registration"]["accepted"])
        first = accepted._write_status.call_args_list[0].kwargs["stand_axis_debug"]["metric_model"]
        self.assertTrue(first["camera_target_registration"]["strict_retry_applied"])
        for scenario in ("registered_wrong_head_bearing", "registered_ambiguous_lidar", "registered_wrong_lidar"):
            with self.subTest(scenario=scenario):
                rejected, payload = self.run_view(scenario)
                self.assertIsNone(payload)
                self.assertFalse(rejected._last_observation_update.axis_sample_accepted)

    def test_shifted_nominal_head_commits_in_first_view_without_strict_retry(self):
        adapter, payload = self.run_view("shifted_bound_qr")
        self.assertTrue(adapter.completed)
        self.assertEqual(payload["axis"]["sample_count"], 7)
        metadata = adapter._write_status.call_args.kwargs["stand_axis_debug"]
        head = metadata["current_head_candidate_association"]
        self.assertTrue(head["accepted"])
        self.assertEqual(head["roi_source"], "candidate_tracked_head_search")
        self.assertGreater(head["lidar_association"]["camera_map_bearing_delta_rad"], math.radians(3))
        self.assertFalse(metadata["metric_model"]["camera_target_registration"]["strict_retry_applied"])
        self.assertEqual(len(metadata["metric_model"]["head_roi_attempts"]), 1)
        self.assertTrue(metadata["decoded_qr_target_binding"]["current_head_binding"]["accepted"])
        self.assertEqual(adapter._last_observation_update.resolved_qr_id, "QR_003")

    def test_shifted_nominal_head_preserves_qr_conflicts_and_publication_freshness(self):
        for scenario in ("conflicting_qr", "unbound_qr", "late_publication", "uncertain_head"):
            with self.subTest(scenario=scenario):
                adapter, payload = self.run_view("shifted_" + scenario)
                self.assertIsNone(payload)
                self.assertFalse(adapter.completed)
                if scenario == "conflicting_qr":
                    self.assertTrue(adapter._last_observation_update.snapshot.poisoned)


if __name__ == "__main__":
    unittest.main()
