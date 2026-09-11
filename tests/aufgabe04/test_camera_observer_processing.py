"""Offline adapter coverage of real ROI registration and per-image decoding."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint, StandAxisEdgeDebugArtifacts, StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis.pose_tracking import MetricPoseTracker
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.node import (
    PassiveRealViewpointNode, _stand_axis_profile_from_args, build_parser,
)
from scripts.aufgabe04.real_robot.observer.tf_retry import PassiveObserverTfRetryScheduler
from scripts.aufgabe04.real_robot.observer.backside_proposal_reuse import BacksideProposalReuse
from scripts.aufgabe04.real_robot.observer.contract import BACKSIDE_AXIS_SAMPLE_SOURCE


def transform(translation=(0., 0., 0.), rotation=(0., 0., 0., 1.)):
    return SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(**dict(zip(("x", "y", "z"), translation))),
        rotation=SimpleNamespace(**dict(zip(("x", "y", "z", "w"), rotation))),
    ))


class CameraObserverProcessingTest(unittest.TestCase):
    def make_adapter(self):
        adapter = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        adapter.args = build_parser().parse_args([
            "--robot-profile", "unused_robot.json",
            "--camera-calibration", "unused_camera.json",
            "--stand-model-profile", "unused_stand.json",
            "--status-json", "unused_status.json",
            "--recommended-pose-json", "unused_recommendation.json",
            "--stream-id", "test", "--stand-id", "candidate", "--expected-qr-id", "auto",
            "--stand-x", "0.6", "--stand-y", "0.0",
        ])
        adapter.profile = SimpleNamespace(
            map_frame="map", base_frame="base", camera_optical_frame="camera", scan_frame="scan",
        )
        adapter.calibration = SimpleNamespace(base_to_camera=object())
        adapter.stand_model_profile = SimpleNamespace(
            head_width_m=.078, head_height_m=.078, sha256="c" * 64,
            profile_id="test", environment="real", measurement_status="measured",
        )
        adapter.stand_head_center_height_m = .45
        adapter.stand_axis_profile = _stand_axis_profile_from_args(adapter.args)
        adapter.model_pose_tracker = MetricPoseTracker()
        adapter.backside_proposal_reuse = BacksideProposalReuse()
        adapter.last_pose = Pose2D(0., 0., 0.)
        adapter.observation_evidence = None
        adapter.completed = False
        adapter.cv2 = object()
        adapter.numpy = numpy
        adapter._reset_qr_marker_epoch()
        self.clock_sec = 100.1
        adapter.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=round(self.clock_sec * 1e9)),
        ))
        stamp = SimpleNamespace(sec=100, nanosec=0)
        image = SimpleNamespace(header=SimpleNamespace(frame_id="camera", stamp=stamp))
        scan = SimpleNamespace(
            header=SimpleNamespace(frame_id="scan", stamp=stamp),
            ranges=(.6,) * 5, angle_min=-.02, angle_increment=.01,
            range_min=.01, range_max=10.,
        )
        info = SimpleNamespace(width=800, height=600,
                               p=(400., 0., 400., 0., 0., 400., 300., 0., 0., 0., 1., 0.))
        def sample(value):
            return SimpleNamespace(value=value, stamp_sec=100.,
                                   received_ros_sec=100.1, received_monotonic_sec=10.)
        sensor_tuple = SimpleNamespace(image=sample(image), scan=sample(scan), camera_info=sample(info))
        adapter._next_sensor_tuple = Mock(return_value=sensor_tuple)
        adapter.tf_retry_scheduler = PassiveObserverTfRetryScheduler()
        adapter.tf_retry_scheduler.offer(sensor_tuple, stamp_sec=100.)
        # Optical +z points along base +x; the head and camera have equal height.
        optical_rotation = (.5, -.5, .5, -.5)
        transforms = {
            ("map", "base"): transform(),
            ("map", "camera"): transform((0., 0., .45), optical_rotation),
            ("camera", "map"): transform((0., .45, 0.), (-.5, .5, -.5, -.5)),
            ("scan", "map"): transform(),
        }
        adapter._lookup = lambda target, source, _stamp: transforms[(target, source)]
        adapter._lookup_static_transform = lambda _target, _source: transform(
            (0., 0., .45), optical_rotation,
        )
        adapter.TransformException = RuntimeError
        adapter._write_debug = Mock()
        adapter._write_status = Mock()
        return adapter

    def test_actual_three_roi_evaluations_reuse_decode_but_repeat_strict_geometry(self):
        adapter = self.make_adapter()
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        decoded_by_shape = {}
        metric_calls = []

        def decode(crop, _cv2, *, diagnostics=None):
            observations = (DecodedQrObservation("QR_1", None, "test_decoder", 1.),)
            decoded_by_shape[crop.shape] = observations
            diagnostics["test_crop_shape"] = list(crop.shape)
            return observations

        def metric(_cv2, crop, **options):
            metric_calls.append((crop.shape, options))
            self.assertIs(options["qr_observations"], decoded_by_shape[crop.shape])
            self.assertIsNone(options["pose_hint"])
            # The wider result proposes a bounded horizontal displacement.
            u = options["expected_head_center_u_px"] + (10. if len(metric_calls) == 2 else 0.)
            v = options["expected_head_center_v_px"]
            corners = tuple(ImagePoint(u + x, v + y)
                            for x, y in ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            if len(metric_calls) == 3:
                # Finish after the source-age limit; real result freshness must
                # reject this frame without bypassing any registration work.
                self.clock_sec = 100.6
            estimate = StandAxisImageEstimate(
                usable=len(metric_calls) == 2,
                reason="model_pose_seed_unavailable" if len(metric_calls) == 1 else "planar_pose_axis_ambiguous",
                mode="face_visible", corners=corners, axis_line=None,
                left_height_px=52., right_height_px=52., height_ratio=1.,
                yaw_proxy=0., yaw_deg=0., closer_side="equal", contour_area_px=2704.,
                source="model_projection" if len(metric_calls) == 1 else "model_current_frame_refined",
                model_profile_sha256=adapter.stand_model_profile.sha256,
            )
            debug = StandAxisEdgeDebugArtifacts(
                edges=None, qr_detected=True,
                model_pose=PlanarPoseHypothesis((0., 0., 0.), (0., 0., .6),
                                               (0., 0., 1.), 0., .2, True),
            )
            return estimate, debug

        module = "scripts.aufgabe04.real_robot.observer.node."
        with patch(module + "camera_info_mismatches", return_value=()), \
             patch(module + "transform_mismatches", return_value=()), \
             patch(module + "compressed_msg_to_bgr_frame", return_value=frame), \
             patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_: value), \
             patch(module + "detect_qr_observations_bgr", side_effect=decode) as decoder, \
             patch(module + "detect_native_qr_observations_bgr") as native_decoder, \
             patch(module + "acquire_registered_head_measurement", return_value=None), \
             patch(module + "estimate_stand_axis_from_metric_model", side_effect=metric):
            adapter._process_latest()

        self.assertEqual(decoder.call_count, 2)
        native_decoder.assert_not_called()
        self.assertEqual(len(metric_calls), 3)
        self.assertNotEqual(metric_calls[0][0], metric_calls[1][0])
        self.assertEqual(metric_calls[1][0], metric_calls[2][0])
        self.assertIs(metric_calls[1][1]["qr_observations"], metric_calls[2][1]["qr_observations"])
        self.assertAlmostEqual(metric_calls[2][1]["expected_head_center_u_px"]
                               - metric_calls[1][1]["expected_head_center_u_px"], 10.)
        adapter._write_status.assert_called_once()
        self.assertEqual(adapter._write_status.call_args.args, ("obsolete_detector_result",))
        metadata = adapter._write_status.call_args.kwargs["stand_axis_debug"]["metric_model"]
        self.assertTrue(metadata["camera_target_registration"]["strict_retry_applied"])
        self.assertFalse(metadata["camera_target_registration"]["measurement_accepted"])
        self.assertEqual([attempt["qr_decode"]["cache_hit"]
                          for attempt in metadata["processing_timing"]["attempts"]],
                         [False, False, True])
        provenance = [attempt["qr_decode"]["decoder_provenance"]
                      for attempt in metadata["processing_timing"]["attempts"]]
        self.assertEqual(provenance[1], provenance[2])
        self.assertNotEqual(provenance[0], provenance[1])
        self.assertFalse(metadata["result_freshness"]["accepted"])
        self.assertFalse(adapter.completed)

    def test_real_seven_frame_qr_consensus_cannot_publish_after_delayed_debug(self):
        adapter = self.make_adapter()
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        debug_calls = []

        def decode(crop, _cv2, *, diagnostics=None):
            u, v = crop.shape[1] / 2., crop.shape[0] / 2.
            corners = tuple((u + x, v + y) for x, y in
                            ((-20, -20), (20, -20), (20, 20), (-20, 20)))
            return (DecodedQrObservation("QR_1", corners, "test_decoder", 1.),)

        def metric(_cv2, _crop, **options):
            u = options["expected_head_center_u_px"]
            v = options["expected_head_center_v_px"]
            corners = tuple(ImagePoint(u + x, v + y) for x, y in
                            ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            estimate = StandAxisImageEstimate(
                usable=True, reason="axis_estimated_model_current_frame_refined",
                mode="face_visible", corners=corners, axis_line=None,
                left_height_px=52., right_height_px=52., height_ratio=1.,
                yaw_proxy=0., yaw_deg=0., closer_side="equal", contour_area_px=2704.,
                source="model_current_frame_refined", evidence_state="fresh_refined",
                model_profile_sha256=adapter.stand_model_profile.sha256,
            )
            debug = StandAxisEdgeDebugArtifacts(
                edges=None, qr_detected=True, evidence_state="fresh_refined",
                model_profile_sha256=adapter.stand_model_profile.sha256,
                model_pose=PlanarPoseHypothesis((0., 0., 0.), (0., 0., .6),
                                               (0., 0., 1.), 0., .2, True),
            )
            return estimate, debug

        def slow_last_debug(*_args, **_kwargs):
            debug_calls.append(self.clock_sec)
            if len(debug_calls) == 7:
                self.clock_sec += .5

        adapter._write_debug.side_effect = slow_last_debug
        module = "scripts.aufgabe04.real_robot.observer.node."
        with TemporaryDirectory() as directory, \
             patch(module + "camera_info_mismatches", return_value=()), \
             patch(module + "transform_mismatches", return_value=()), \
             patch(module + "real_robot_profile_sha256", return_value="a" * 64), \
             patch(module + "camera_calibration_sha256", return_value="b" * 64), \
             patch(module + "compressed_msg_to_bgr_frame", return_value=frame), \
             patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_: value), \
             patch(module + "detect_qr_observations_bgr", side_effect=decode), \
             patch(module + "detect_native_qr_observations_bgr", side_effect=decode), \
             patch(module + "estimate_stand_axis_from_metric_model", side_effect=metric) as estimator:
            output = adapter.args.recommended_pose_json = Path(directory) / "recommendation.json"
            sensor_tuple = adapter._next_sensor_tuple.return_value
            for index in range(7):
                stamp = 100. + index * .2
                self.clock_sec = stamp + .1
                for sample in (sensor_tuple.image, sensor_tuple.scan, sensor_tuple.camera_info):
                    sample.stamp_sec = stamp
                    sample.received_ros_sec = self.clock_sec
                stamp_ns = round(stamp * 1e9)
                sensor_tuple.image.value.header.stamp.sec = stamp_ns // 1_000_000_000
                sensor_tuple.image.value.header.stamp.nanosec = stamp_ns % 1_000_000_000
                if index:
                    adapter.tf_retry_scheduler.offer(sensor_tuple, stamp_sec=stamp)
                adapter._process_latest()
            self.assertEqual(estimator.call_count, 7)
            self.assertEqual(len(debug_calls), 7)
            update = adapter._last_observation_update
            self.assertTrue(update.frame_accepted)
            self.assertTrue(update.axis_sample_accepted)
            self.assertEqual(update.axis_consensus.sample_count, 7)
            self.assertEqual(update.resolved_qr_id, "QR_1")
            self.assertTrue(adapter._last_evidence_source_freshness["accepted"])
            self.assertEqual(adapter._write_status.call_args.args, ("obsolete_publication_evidence",))
            self.assertEqual(adapter._last_camera_publication_freshness["artifact_kind"], "recommendation")
            self.assertFalse(adapter._last_camera_publication_freshness["accepted"])
            self.assertFalse(output.exists())
            self.assertFalse(adapter.completed)

    def test_stale_backside_only_seeds_search_then_current_frame_enters_evidence(self):
        adapter = self.make_adapter()
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        metric_calls = []

        def metric(_cv2, crop, **options):
            metric_calls.append(options)
            # The nominal crop cannot acquire this displaced head. Wide
            # acquisition locates it; later strict fits use CURRENT pixels.
            first = len(metric_calls) == 1
            u = options["expected_head_center_u_px"]
            if len(metric_calls) == 2:
                u -= 10.
            v = options["expected_head_center_v_px"]
            corners = None if first else tuple(ImagePoint(u + x, v + y) for x, y in
                ((-26., -26.), (26., -26.), (26., 26.), (-26., 26.)))
            estimate = StandAxisImageEstimate(
                usable=not first,
                reason=("model_backside_head_and_neck_unavailable" if first
                        else "axis_estimated_model_backside_current_frame"),
                mode="face_visible", corners=corners, axis_line=None,
                left_height_px=52., right_height_px=52., height_ratio=1.,
                yaw_proxy=0., yaw_deg=2. if len(metric_calls) < 4 else 4.,
                closer_side="equal", contour_area_px=2704.,
                source=BACKSIDE_AXIS_SAMPLE_SOURCE, evidence_state="fresh_backside",
                model_profile_sha256=adapter.stand_model_profile.sha256,
                model_measurement_status="measured", visible_face="backside_candidate",
                visible_face_confidence=.99,
            )
            debug = StandAxisEdgeDebugArtifacts(
                edges=None, qr_detected=False, evidence_state="fresh_backside",
                model_profile_sha256=adapter.stand_model_profile.sha256,
                head_scale_ratio=1., head_center_error_ratio=0.,
            )
            if len(metric_calls) == 3:
                self.clock_sec = 100.824  # Recorded failure, still no authority.
            return estimate, debug

        module = "scripts.aufgabe04.real_robot.observer.node."
        with patch(module + "camera_info_mismatches", return_value=()), \
             patch(module + "transform_mismatches", return_value=()), \
             patch(module + "compressed_msg_to_bgr_frame", return_value=frame), \
             patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_: value), \
             patch(module + "detect_qr_observations_bgr", return_value=()) as decoder, \
             patch(module + "acquire_registered_head_measurement", return_value=None), \
             patch(module + "estimate_stand_axis_from_metric_model", side_effect=metric):
            adapter._process_latest()
            self.assertEqual(len(metric_calls), 3)
            self.assertEqual(adapter._write_status.call_args.args, ("obsolete_detector_result",))
            self.assertIsNone(adapter.observation_evidence)
            self.assertTrue(adapter.backside_proposal_reuse.last_metadata["hint_retained"])
            sensor_tuple = adapter._next_sensor_tuple.return_value
            self.clock_sec = 101.1
            for sample in (sensor_tuple.image, sensor_tuple.scan, sensor_tuple.camera_info):
                sample.stamp_sec = 101.
                sample.received_ros_sec = self.clock_sec
            sensor_tuple.image.value.header.stamp.sec = 101
            adapter.tf_retry_scheduler.offer(sensor_tuple, stamp_sec=101.)
            adapter._process_latest()
            self.assertEqual(len(metric_calls), 4)
            self.assertEqual(decoder.call_count, 3)  # Two cold crops + new wide crop.
            update = adapter._last_observation_update
            self.assertTrue(update.axis_sample_accepted)
            self.assertEqual(update.snapshot.accepted_frame_count, 1)
            self.assertEqual(update.snapshot.current_axis_sample_count, 1)
            self.assertIsNone(update.axis_consensus)
            axis = adapter._write_status.call_args.kwargs["stand_axis_debug"]
            self.assertAlmostEqual(axis["advisory_camera_relative_yaw_rad"], numpy.deg2rad(4.))
            registration = axis["metric_model"]["camera_target_registration"]
            self.assertTrue(registration["attempted"])
            self.assertTrue(registration["search_hint_used"])
            self.assertFalse(adapter.completed)

    def test_pose_free_recentered_current_fit_and_fresh_framing_recovery(self):
        from dataclasses import replace
        from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
        from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult

        for stale, conflict, nominal_miss in (
            (False, False, False), (True, False, False), (False, True, False),
            (False, False, True),
        ):
            with self.subTest(stale=stale, conflict=conflict, nominal_miss=nominal_miss):
                adapter = self.make_adapter()
                frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
                calls = []

                def locate(_cv2, _crop, **options):
                    u = options["expected_head_center_u_px"] + 10.
                    v = options["expected_head_center_v_px"]
                    corners = tuple(ImagePoint(u + x, v + y) for x, y in
                                    ((-26, -26), (26, -26), (26, 26), (-26, 26)))
                    return HeadProposalResult(HeadProposal(
                        corners, (int(u - 34), int(v - 34), int(u + 34), int(v + 55)),
                        (u - 26, v - 26, u + 26, v + 26), u, v, 52., 1., 10 / 52., .98, .98,
                    ), "current_head_proposal", 1, 1, "test_locator")

                def metric(_cv2, crop, **options):
                    calls.append((crop.shape, options))
                    proposal = options["current_head_proposal_corners"]
                    strict = proposal is not None
                    self.assertIsNone(options["pose_hint"])
                    if strict:
                        # Same full-image geometry after the crop origin and
                        # principal point are each adjusted exactly once.
                        self.assertAlmostEqual(sum(p.u_px for p in proposal) / 4
                                               - options["camera_cx_px"], 10.)
                        self.assertEqual(crop.shape[:2], (89, 68))
                    if strict and stale:
                        self.clock_sec = 100.6
                    return StandAxisImageEstimate(
                        usable=False, reason=("planar_pose_reprojection_error" if strict
                                              else "model_qr_text_without_geometry"),
                        mode="face_visible", corners=proposal, axis_line=None,
                        left_height_px=52., right_height_px=52., height_ratio=1.,
                        yaw_proxy=0., yaw_deg=None, closer_side="equal", contour_area_px=2704.,
                        source="model_refined_head" if strict else "model_seed",
                        model_profile_sha256=adapter.stand_model_profile.sha256,
                    ), StandAxisEdgeDebugArtifacts(
                        edges=None, qr_detected=True, qr_marker_verified=True,
                        model_pose_fit_source="joint_qr_head" if strict else None,
                    )

                module = "scripts.aufgabe04.real_robot.observer.node."
                def decode(crop, _cv2, *, diagnostics=None):
                    text = "QR_2" if conflict and crop.shape[1] == 68 else "QR_1"
                    return (DecodedQrObservation(text, None, "test", 1.),)

                def preliminary(*args, **kwargs):
                    result = associate_candidate_lidar_target(*args, **kwargs)
                    return replace(result, associated=False, rejection_reason="nominal_cone_miss") if nominal_miss else result

                with patch(module + "camera_info_mismatches", return_value=()), \
                     patch(module + "transform_mismatches", return_value=()), \
                     patch(module + "compressed_msg_to_bgr_frame", return_value=frame), \
                     patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_: value), \
                     patch(module + "detect_qr_observations_bgr", side_effect=decode), \
                     patch(module + "associate_candidate_lidar_target", side_effect=preliminary), \
                     patch("scripts.aufgabe04.real_robot.observer.head_proposal_registration.acquire_head_proposal",
                           side_effect=locate), \
                     patch(module + "estimate_stand_axis_from_metric_model", side_effect=metric):
                    adapter._process_latest()
                self.assertEqual(len(calls), 2)  # No wider-ROI metric/PnP prerequisite.
                self.assertFalse(adapter.completed)
                if conflict:
                    self.assertEqual(adapter._write_status.call_args.args, ("evidence_not_committable",))
                    self.assertTrue(adapter._last_observation_update.snapshot.poisoned)
                    self.assertIsNone(adapter._camera_framing)
                elif stale:
                    self.assertEqual(adapter._write_status.call_args.args, ("obsolete_detector_result",))
                    self.assertIsNone(adapter._camera_framing)
                    self.assertFalse(adapter._qr_marker_seen_in_stationary_epoch)
                else:
                    self.assertEqual(adapter._write_status.call_args.args,
                                     ("metric_model_measurement_unavailable",))
                    self.assertEqual(adapter._camera_framing["reason"], "head_qr_geometry_mismatch")
                    self.assertTrue(adapter._camera_framing["candidate_associated"])
                    self.assertFalse(adapter._camera_framing["motion_authorized"])
                    self.assertTrue(adapter._last_observation_update.frame_accepted)
                    self.assertEqual(adapter._last_observation_update.snapshot.current_axis_sample_count, 0)
                    adapter._reset_qr_marker_epoch()
                    self.assertIsNone(adapter._camera_framing)


if __name__ == "__main__":
    unittest.main()
