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
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import create_head_geometry_tracker
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.perception.stand_axis.qr_marker_validation import QrMarkerEvidence
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.viewer_head_acquisition import evaluate_viewer_head
from scripts.aufgabe04.real_robot.observer.current_scan_head_proposal_filter import CurrentScanHeadProposalFilter
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_head_model_admission as head_fixtures


class MeasuredHeadObserverProcessingTests(unittest.TestCase):
    def run_view(self, scenario, *, publish_immediate=False):
        physical = scenario.startswith("physical_")
        scenario = scenario.removeprefix("physical_")
        shifted = scenario.startswith("shifted_")
        scenario = scenario.removeprefix("shifted_")
        registered = scenario.startswith("registered_")
        scenario = scenario.removeprefix("registered_")
        registered = registered or physical
        fixture = processing_fixtures.CameraObserverProcessingTest()
        adapter = fixture.make_adapter()
        if physical:
            # Full metric context for candidate-volume and source-pixel gates.
            # This synthetic stand's head is at the fixture camera's height.
            import cv2
            adapter.cv2 = cv2
            adapter.stand_model_profile.environment = "physical"
            adapter.stand_model_profile.committable = True
            adapter.stand_model_profile.head_top_height_m = adapter.stand_head_center_height_m + .039
            adapter.stand_model_profile.head_depth_m = .006
            adapter.stand_model_profile.tolerance_m = .002
            info = adapter._next_sensor_tuple.return_value.camera_info.value
            info.k = (400., 0., 400., 0., 400., 300., 0., 0., 1.)
            info.r = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
            info.d = (0., 0., 0., 0., 0.)
            if scenario in {"offset_scan", "offset_scan_ambiguous"}:
                scan = adapter._next_sensor_tuple.return_value.scan.value
                scan.angle_min = .18
                if scenario == "offset_scan_ambiguous":
                    scan.ranges = (.6, .6, math.inf, .6, .6)
            adapter.model_pose_tracker = create_head_geometry_tracker()
            # This nominal crop would clip the off-center head. Physical
            # acquisition uses the whole image before candidate association.
            adapter.args.head_roi_padding_scale = 1.0
            if scenario == "projection_behind_camera":
                adapter.args.stand_x = -.6
        if shifted:
            # Complete head remains inside this synthetic nominal crop while
            # its ray lies outside the original map-centered three-degree cone.
            adapter.args.head_roi_padding_scale = 2.2
        adapter.node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        current_index = [0]
        decode_modes = []
        pose_hints = []
        head_calls = []
        qr_geometry_priorities = []
        metric_options = []
        metric_frame_indices = []
        scan_filters = []

        def scan_filter(**options):
            value = CurrentScanHeadProposalFilter(**options)
            scan_filters.append(value)
            return value

        def decode(crop, _cv2, *, diagnostics=None, max_elapsed_sec=None,
                   prefer_native_geometry=False):
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
            qr_geometry_priorities.append(kwargs.get("prefer_native_geometry", False))
            return decode(*args, **kwargs)

        def native_decode(*args, **kwargs):
            decode_modes.append("native")
            if scenario == "late_identity":
                fixture.clock_sec += .5
            if scenario == "native_miss" and current_index[0] == 2:
                return ()
            decoded = decode(*args, **kwargs)
            if scenario == "recover_qr":
                return tuple(replace(item, corners=None) for item in decoded)
            return decoded

        def metric_geometry(_cv2, _crop, **options):
            head_calls.append("fit")
            metric_options.append(options)
            metric_frame_indices.append(current_index[0])
            pose_hints.append(options["pose_hint"])
            if physical:
                self.assertIs(_crop, frame)
                u, v = 410., 300.
                self.assertNotIn("expected_head_center_u_px", options)
                self.assertNotIn("expected_head_center_v_px", options)
                self.assertNotIn("expected_head_height_px", options)
                self.assertNotIn("current_head_proposal_corners", options)
            else:
                u, v = options["expected_head_center_u_px"], options["expected_head_center_v_px"]
            if shifted:
                u -= 28.
            if registered and not physical and options["pose_hint"] is not None:
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
            if (registered and not physical and options["current_head_proposal_corners"] is None
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
            if scenario == "late_fit":
                fixture.clock_sec += .5
            return estimate, debug

        def metric(_cv2, _crop, **options):
            if not physical:
                return metric_geometry(_cv2, _crop, **options)
            # The full-image fit runs once before independent marker work.
            estimate, debug = metric_geometry(_cv2, _crop, **options)
            signal = (None if options["qr_marker_policy"] == "disabled" else
                      bool(options["qr_observations"]))
            return estimate, replace(debug, qr_detected=signal, qr_marker_verified=signal)

        def locate(_cv2, _crop, **options):
            head_calls.append("locate")
            u, v = options["expected_center"]
            u += 10.
            corners = tuple(ImagePoint(u + x, v + y) for x, y in
                            ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            proposal = HeadProposal(
                corners, (int(u - 34), int(v - 34), int(u + 34), int(v + 55)),
                (u - 26, v - 26, u + 26, v + 26), u, v, 52., 1., 10 / 52., .98, .98,
            )
            if physical:
                self.assertNotIn(current_index[0], metric_frame_indices,
                                 "Cold proposal must precede this frame's head fit")
                eligible = options["proposal_filter"](proposal)
                if not eligible:
                    return HeadProposalResult(None, "head_proposal_unavailable", 1, 1,
                                              "test_locator")
            return HeadProposalResult(proposal, "current_head_proposal", 1, 1, "test_locator")

        def delayed_debug(*_args, **_kwargs):
            if scenario == "late_publication" and current_index[0] == 6:
                fixture.clock_sec += .5

        adapter._write_debug.side_effect = delayed_debug
        module = "scripts.aufgabe04.real_robot.observer.node."
        with TemporaryDirectory() as directory, ExitStack() as stack:
            output = adapter.args.recommended_pose_json = Path(directory) / "recommendation.json"
            adapter.args.axis_observation_json = Path(directory) / "backside.json"
            adapter.args.status_json = Path(directory) / "status.json"
            if publish_immediate:
                adapter._write_status = Mock(side_effect=lambda *args, **kwargs:
                    PassiveRealViewpointNode._write_status(adapter, *args, **kwargs))
            patches = {
                "camera_info_mismatches": {"return_value": ()},
                "transform_mismatches": {"return_value": ()},
                "real_robot_profile_sha256": {"return_value": "a" * 64},
                "camera_calibration_sha256": {"return_value": "b" * 64},
                "compressed_msg_to_bgr_frame": {"return_value": frame},
                "_rectify_bgr_frame": {"side_effect": lambda value, *_, **_kwargs: value},
                "detect_qr_observations_bgr": {"side_effect": full_decode},
                "detect_native_qr_observations_bgr": {"side_effect": native_decode},
                "estimate_stand_axis_from_metric_model": {"side_effect": metric},
                "CurrentScanHeadProposalFilter": {"side_effect": scan_filter},
            }
            for name, options in patches.items():
                stack.enter_context(patch(module + name, **options))
            # Synthetic OpenCV fixture injects the shared current-head locator;
            # its pixel acquisition and border comparison have dedicated tests.
            if registered and not physical:
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_viewer_candidate_head",
                    side_effect=locate,
                ))
            if physical:
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_viewer_candidate_head",
                    side_effect=AssertionError("Physical acquisition must use the full-image viewer fitter"),
                ))
                # Geometry is synthetic; empty identity must not manufacture
                # a completed pixel-level backside marker-absence proof.
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.viewer_head_acquisition.detect_qr_quad",
                    return_value=None,
                ))
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.viewer_head_acquisition.validate_qr_marker",
                    return_value=QrMarkerEvidence(None, "synthetic_marker_check_unavailable"),
                ))
            if physical and scenario in {"late_identity", "late_fit"}:
                # Couple simulated ROS/monotonic clocks so overruns are
                # attributed to geometry or identity at their actual stage.
                stack.enter_context(patch(module + "time.monotonic",
                    side_effect=lambda: fixture.clock_sec - 90.))
                stack.enter_context(patch(module + "evaluate_viewer_head",
                    side_effect=lambda *args, **kwargs: evaluate_viewer_head(
                        *args, **kwargs, now=lambda: fixture.clock_sec - 90.)))
            if scenario == "panel_then_recovery":
                stack.enter_context(patch(
                    "scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_viewer_candidate_head",
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
                stamp = 100. + index * (.6 if scenario in {
                    "slow_bootstrap", "late_identity", "late_fit"} else .2)
                fixture.clock_sec = stamp + (.4 if scenario == "slow_bootstrap" else .1)
                for sample in (sensor_tuple.image, sensor_tuple.scan, sensor_tuple.camera_info):
                    sample.stamp_sec = stamp
                    sample.received_ros_sec = fixture.clock_sec
                if scenario == "association_then_recovery":
                    sensor_tuple.scan.value.ranges = ((3.,) * 5 if index == 0 else (.6,) * 5)
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
            adapter._test_head_calls = head_calls
            adapter._test_metric_options = metric_options
            adapter._test_scan_filters = scan_filters
            adapter._test_qr_geometry_priorities = qr_geometry_priorities
            return adapter, payload

    def test_first_physical_head_fit_and_bound_qr_use_real_immediate_status_path(self):
        adapter, payload = self.run_view("physical_bound_qr", publish_immediate=True)
        self.assertTrue(adapter.completed)
        self.assertIsNotNone(payload)
        recommendation = load_recommendation(payload)
        self.assertEqual(recommendation.schema_version, 3)
        self.assertEqual(recommendation.axis_sample_count, 1)
        self.assertEqual(recommendation.sensor_stamp_sec, 100.)
        self.assertEqual(recommendation.axis_measurement["qr_id"], "QR_003")
        self.assertEqual(recommendation.axis_measurement["camera_signature"][0], "camera")
        self.assertIsNone(adapter._last_observation_update.axis_consensus)
        self.assertIsNone(adapter._last_observation_update.resolved_qr_id)
        self.assertEqual(adapter._test_head_calls, ["fit"])

    def test_real_immediate_status_path_preserves_quality_association_and_no_qr_gates(self):
        for scenario in ("head_only", "unbound_qr", "uncertain_head", "wrong_lidar", "ambiguous_lidar", "late_fit"):
            with self.subTest(scenario=scenario):
                adapter, payload = self.run_view("physical_" + scenario, publish_immediate=True)
                self.assertIsNone(payload)
                self.assertFalse(adapter.completed)

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

    def test_physical_full_image_acquisition_fits_once_then_tracks_seven_samples(self):
        adapter, payload = self.run_view("physical_bound_qr")
        self.assertTrue(adapter.completed)
        self.assertEqual(payload["axis"]["sample_count"], 7)
        self.assertEqual(adapter._test_head_calls, ["fit"] * 7)
        first, *tracked = adapter._test_metric_options
        self.assertIsNone(first["pose_hint"])
        self.assertEqual(first["camera_cx_px"], 400.)
        self.assertEqual(first["camera_cy_px"], 300.)
        self.assertEqual(first["min_edge_height_px"], 8.)
        self.assertTrue(all(item["qr_marker_policy"] == "disabled"
                            for item in adapter._test_metric_options))
        self.assertTrue(all(item["pose_hint"] is not None for item in tracked))
        self.assertTrue(all("current_head_proposal_corners" not in item for item in tracked))
        filters = adapter._test_scan_filters
        self.assertEqual(len({id(value) for value in filters}), 7)
        self.assertEqual([value.scan.scan_stamp_sec for value in filters],
                         [100. + index * .2 for index in range(7)])
        self.assertTrue(all(callable(value.preview_lidar_association) for value in filters))
        self.assertTrue(all(callable(value.current_ros_sec) for value in filters))
        self.assertTrue(all(value.metadata()["persistence_read_only"] for value in filters))
        self.assertTrue(all(item["candidate_search"].edge_region is not None
                            for item in adapter._test_metric_options))
        first_metadata = adapter._write_status.call_args_list[0].kwargs["stand_axis_debug"]
        self.assertTrue(first_metadata["current_head_candidate_association"]["accepted"])
        first_model = first_metadata["metric_model"]
        self.assertEqual(len(first_model["head_roi_attempts"]), 1)
        self.assertFalse(first_model["camera_target_registration"]["strict_retry_applied"])
        self.assertEqual(adapter._last_observation_update.resolved_qr_id, "QR_003")

    def test_physical_region_uses_unique_registered_scan_envelope_before_geometry(self):
        adapter, payload = self.run_view("physical_offset_scan")
        self.assertTrue(all(item["candidate_search"].edge_region is not None
                            for item in adapter._test_metric_options))
        # The broad pre-search hint cannot authorize the deliberately mismatched
        # synthetic visual head; normal current-head association still governs.
        self.assertIsNone(payload)
        self.assertFalse(adapter.completed)

    def test_physical_region_does_not_choose_between_registered_scan_clusters(self):
        adapter, payload = self.run_view("physical_offset_scan_ambiguous")
        self.assertTrue(all(item["candidate_search"].edge_region is None
                            for item in adapter._test_metric_options))
        self.assertIsNone(payload)
        self.assertFalse(adapter.completed)

    def test_physical_bound_head_marker_crop_prioritizes_native_qr_geometry(self):
        adapter, payload = self.run_view("physical_recover_qr")
        self.assertIsNotNone(payload)
        self.assertTrue(adapter._test_qr_geometry_priorities)
        self.assertTrue(all(adapter._test_qr_geometry_priorities))

    def test_physical_cold_acquisition_preserves_identity_and_publication_gates(self):
        for scenario in ("unbound_qr", "conflicting_qr", "late_publication"):
            with self.subTest(scenario=scenario):
                adapter, payload = self.run_view("physical_" + scenario)
                self.assertIsNone(payload)
                self.assertFalse(adapter.completed)
                self.assertEqual(adapter._test_head_calls.count("fit"), 7)
                if scenario == "conflicting_qr":
                    self.assertTrue(adapter._last_observation_update.snapshot.poisoned)
                    self.assertEqual(adapter._test_head_calls.count("locate"), 0)
                elif scenario == "late_publication":
                    self.assertEqual(adapter._write_status.call_args.args,
                                     ("obsolete_publication_evidence",))
                else:
                    self.assertEqual(adapter._write_status.call_args.kwargs["reason"],
                                     "measured_head_front_identity_unresolved")

    def test_physical_atomic_fit_overrun_never_seeds_tracking_or_axis_evidence(self):
        adapter, payload = self.run_view("physical_late_fit")
        self.assertIsNone(payload)
        self.assertFalse(adapter.completed)
        self.assertEqual(adapter._test_head_calls, ["fit"] * 7)
        self.assertTrue(all(hint is None for hint in adapter._test_pose_hints))
        self.assertTrue(all(call.args == ("obsolete_detector_result",)
                            for call in adapter._write_status.call_args_list))

    def test_physical_late_identity_cannot_erase_timely_geometry_or_admit_stale_result(self):
        adapter, payload = self.run_view("physical_late_identity", publish_immediate=True)
        self.assertIsNone(payload)
        self.assertFalse(adapter.completed)
        self.assertEqual(adapter._test_head_calls, ["fit"] * 7)
        self.assertIsNone(adapter._test_pose_hints[0])
        self.assertTrue(all(hint is not None for hint in adapter._test_pose_hints[1:]))
        self.assertTrue(all(call.args == ("obsolete_detector_result",)
                            for call in adapter._write_status.call_args_list))

    def test_physical_geometry_tracks_without_admitting_wrong_or_ambiguous_lidar(self):
        for scenario in ("wrong_lidar", "ambiguous_lidar"):
            with self.subTest(scenario=scenario):
                adapter, payload = self.run_view("physical_" + scenario)
                self.assertIsNone(payload)
                self.assertFalse(adapter.completed)
                self.assertEqual(adapter._test_head_calls, ["fit"] * 7)
                self.assertIsNone(adapter._test_pose_hints[0])
                self.assertTrue(all(hint is not None for hint in adapter._test_pose_hints[1:]))
                self.assertTrue(adapter._test_decode_modes)

    def test_physical_lidar_recovery_uses_previous_geometry_hint_and_admits_first_valid_frame(self):
        adapter, payload = self.run_view("physical_association_then_recovery", publish_immediate=True)
        self.assertTrue(adapter.completed)
        recommendation = load_recommendation(payload)
        self.assertEqual(recommendation.axis_sample_count, 1)
        self.assertEqual(recommendation.sensor_stamp_sec, 100.2)
        self.assertEqual(recommendation.axis_measurement["qr_id"], "QR_003")
        self.assertEqual(adapter._test_head_calls, ["fit", "fit"])
        self.assertIsNone(adapter._test_pose_hints[0])
        self.assertIsNotNone(adapter._test_pose_hints[1])

    def test_physical_projection_behind_camera_still_fits_but_cannot_admit_visible_head(self):
        adapter, payload = self.run_view("physical_projection_behind_camera", publish_immediate=True)
        self.assertIsNone(payload)
        self.assertFalse(adapter.completed)
        self.assertEqual(adapter._test_head_calls, ["fit"] * 7)
        self.assertIsNone(adapter._test_pose_hints[0])
        self.assertTrue(all(hint is not None for hint in adapter._test_pose_hints[1:]))
        metadata = adapter._write_status.call_args.kwargs["stand_axis_debug"]
        self.assertLess(metadata["metric_model"]["target_projection"]["depth_m"], 0.)
        self.assertFalse(metadata["current_head_candidate_association"]["accepted"])
        self.assertFalse(adapter._last_observation_update.axis_sample_accepted)


if __name__ == "__main__":
    unittest.main()
