"""QR discovery fallback never invents a stand angle or bypasses source gates."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
    HASH_FIELD, load_qr_verified_observation_pose, validate_qr_verified_observation_pose,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import (
    prepare_qr_observation_pose, qr_observation_grace_pending,
)
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_current_head_association as association_fixtures


class QrObservationPoseTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = processing_fixtures.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        self.adapter.stand_model_profile.sha256 = "a" * 64
        self.adapter.profile.scan_frame = "base_scan"
        self.adapter.args.qr_observation_pose_json = self.root / "qr_pose.json"
        self.adapter.args.qr_pose_fallback_delay_sec = 2.
        self.adapter.args.recommended_pose_json = self.root / "recommendation.json"
        self.adapter.args.status_json = self.root / "status.json"
        for name in ("real_robot_profile_sha256", "camera_calibration_sha256"):
            mocked = patch("scripts.aufgabe04.real_robot.observer.qr_observation_pose." + name,
                           return_value="b" * 64)
            mocked.start()
            self.addCleanup(mocked.stop)

    def frame(self, stamp=100., *, decode=True, associated=True, age=.1,
              scan_offset=0., qr_id="QR_003", symbols=None, pose=None,
              camera_signature=(640., 640., 400., 300.), publish=True):
        self.fixture.clock_sec = stamp + age
        options = association_fixtures.CurrentHeadAssociationTests().options()
        options["scan"] = replace(options["scan"], scan_stamp_sec=stamp + scan_offset,
                                   receipt_sec=stamp + scan_offset)
        options["now_sec"] = stamp + .1
        _, binding, observations = association_fixtures.CurrentHeadAssociationTests().qr_binding(options)
        observations = tuple(replace(o, text=qr_id) for o in observations)
        binding = replace(binding, qr_texts_for_evidence=(qr_id,))
        if not associated:
            binding = replace(binding, accepted=False, qr_texts_for_evidence=())
        if not decode:
            observations, binding = (), QrTargetBinding(False, "no_decoded_qr_geometry")
        robot_pose = pose or Pose2D(0., 0., 0.)
        metadata = {}
        self.adapter._pending_qr_observation_pose = prepare_qr_observation_pose(
            qr_binding=binding, qr_observations=observations,
            observed_qr_texts=() if not decode else (qr_id,), image_stamp_sec=stamp,
            scan_stamp_sec=stamp + scan_offset, robot_pose=robot_pose,
            target_key=self.adapter._target_evidence_key(), camera_signature=camera_signature,
            image_shape=(600, 800, 3), roi=options["attempt"].roi,
            model_profile_sha256=self.adapter.stand_model_profile.sha256, metadata=metadata)
        with patch("scripts.aufgabe04.real_robot.observer.qr_observation_pose.time.monotonic", return_value=stamp):
            update = self.adapter._record_observation_frame(
                robot_pose=robot_pose, image_stamp_sec=stamp, scan_stamp_sec=stamp + scan_offset,
                observed_at_sec=stamp + age, lidar_associated=associated,
                axis_yaw_rad=None, axis_source=None,
                qr_texts=binding.qr_texts_for_evidence,
                qr_symbol_count=binding.symbol_count if symbols is None else symbols)
            if publish:
                PassiveRealViewpointNode._write_status(self.adapter, "metric_model_measurement_unavailable")
        return update, metadata

    def result(self):
        path = self.adapter.args.qr_observation_pose_json
        return load_qr_verified_observation_pose(path) if path.exists() else None

    def test_default_delay_admits_one_current_decode_without_a_head_angle(self):
        from scripts.aufgabe04.real_robot.observer.node import build_parser
        self.adapter.args.qr_pose_fallback_delay_sec = build_parser().get_default("qr_pose_fallback_delay_sec")
        self.assertEqual(self.adapter.args.qr_pose_fallback_delay_sec, 0.)
        update, metadata = self.frame(100.)
        self.assertIsNotNone(self.result(), metadata)
        self.assertEqual(self.result()["sensor_stamp_sec"], 100.)
        self.assertIsNone(update.resolved_qr_id)  # No second decoder success required.

    def test_fresh_bound_decode_after_grace_commits_without_head_or_angle(self):
        self.frame(100.)
        self.assertIsNone(self.result())
        self.frame(101.9)
        self.assertIsNone(self.result())
        update, metadata = self.frame(102.)
        result = self.result()
        self.assertIsNotNone(result, metadata)
        self.assertEqual(result["qr_id"], "QR_003")
        self.assertIsNone(result["stand_axis_rad"])
        self.assertFalse(result["facing_ready"])
        self.assertFalse(result["motion_authorized"])
        self.assertEqual(result["completion_scope"], "discovery_only")
        self.assertEqual(result["sensor_stamp_sec"], 102.)
        self.assertNotEqual(result["robot_pose"]["x_m"], result["stand_center"]["x_m"])
        self.assertTrue(self.adapter.completed)
        self.assertFalse(self.adapter.args.recommended_pose_json.exists())
        status = json.loads(self.adapter.args.status_json.read_text())
        self.assertEqual(status["state"], "qr_observation_pose_committed")
        self.assertIsNone(update.axis_consensus)

    def test_missing_decode_after_grace_cannot_reuse_latched_identity(self):
        self.frame(100.)
        self.frame(100.5)
        self.frame(102., decode=False)
        self.assertIsNone(self.result())

    def test_duplicate_frame_cannot_complete_grace(self):
        self.frame(100.)
        self.frame(100., age=2.1)
        self.assertIsNone(self.result())

    def test_source_age_scan_skew_and_unassociated_decodes_do_not_complete(self):
        for kwargs in ({"age": .7}, {"scan_offset": -.2}, {"associated": False}):
            with self.subTest(kwargs=kwargs):
                self.adapter._reset_observation_evidence()
                self.frame(100.)
                self.frame(102., **kwargs)
                self.assertIsNone(self.result())

    def test_motion_or_calibration_context_change_restarts_grace(self):
        for kwargs in ({"pose": Pose2D(.1, 0., 0.)},
                       {"camera_signature": (640., 641., 400., 300.)}):
            with self.subTest(kwargs=kwargs):
                self.adapter._reset_observation_evidence()
                self.frame(100.)
                self.frame(102., **kwargs)
                self.assertIsNone(self.result())

    def test_identity_conflict_and_multiple_symbols_poison_epoch(self):
        for kwargs in ({"qr_id": "OTHER"}, {"symbols": 2}):
            with self.subTest(kwargs=kwargs):
                self.adapter._reset_observation_evidence()
                self.frame(100.)
                self.frame(101., **kwargs)
                self.frame(102.)
                self.assertIsNone(self.result())

    def test_inspection_advisory_is_deferred_during_grace(self):
        with patch.object(self.adapter, "_maybe_commit_inspection_progress") as progress:
            self.frame(100.)
            progress.assert_not_called()
        with patch("scripts.aufgabe04.real_robot.observer.qr_observation_pose.time.monotonic", return_value=100.5):
            self.assertTrue(qr_observation_grace_pending(self.adapter))

    def test_publication_rechecks_sources_after_processing(self):
        self.adapter.args.qr_pose_fallback_delay_sec = 0.
        self.frame(100., publish=False)
        self.fixture.clock_sec = 100.7
        PassiveRealViewpointNode._write_status(self.adapter, "metric_model_measurement_unavailable")
        self.assertIsNone(self.result())
        self.assertFalse(self.adapter.completed)
        self.assertFalse(self.adapter._last_camera_publication_freshness["accepted"])

    def test_geometry_recommendation_takes_priority_over_ready_fallback(self):
        self.frame(100.)
        self.frame(102., publish=False)
        def geometry():
            self.adapter.completed = True
            return "recommendation_committed", {"recommendation": "geometry"}
        with patch("scripts.aufgabe04.real_robot.observer.node.commit_immediate_front", side_effect=lambda adapter: geometry()):
            PassiveRealViewpointNode._write_status(self.adapter, "collecting_consensus")
        self.assertIsNone(self.result())

    def test_processing_head_acquisition_failure_still_commits_current_qr_pose(self):
        adapter = self.adapter
        # ROS Humble supplies ndarray fields. Retain those live scalar types
        # through the real producer path; a JSON fixture hides the type bug.
        info = adapter._next_sensor_tuple.return_value.camera_info.value
        info.k = numpy.array([400., 0., 400., 0., 400., 300., 0., 0., 1.])
        info.d = numpy.zeros(5)
        info.r = numpy.eye(3).ravel()
        info.p = numpy.asarray(info.p)
        info.distortion_model = "plumb_bob"
        adapter.profile.scan_frame = "scan"
        adapter.stand_model_profile.environment = "physical"
        adapter.stand_model_profile.committable = True
        adapter._write_status = PassiveRealViewpointNode._write_status.__get__(adapter)
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        module = "scripts.aufgabe04.real_robot.observer.node."

        def failed_head(*_args, diagnostics, **_kwargs):
            diagnostics["reason"] = "head_proposal_ambiguous"
            return None

        def decode(crop, *_args, **_kwargs):
            u, v = crop.shape[1] / 2, crop.shape[0] / 2
            return (DecodedQrObservation("QR_003", tuple((u + x, v + y)
                for x, y in ((-20, -20), (20, -20), (20, 20), (-20, 20))), "test"),)

        with ExitStack() as stack:
            for name in ("camera_info_mismatches", "transform_mismatches"):
                stack.enter_context(patch(module + name, return_value=()))
            stack.enter_context(patch(module + "compressed_msg_to_bgr_frame", return_value=frame))
            stack.enter_context(patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_, **_kw: value))
            stack.enter_context(patch(module + "acquire_registered_head_measurement", side_effect=failed_head))
            stack.enter_context(patch(module + "detect_qr_observations_bgr", side_effect=decode))
            estimator = stack.enter_context(patch(module + "estimate_stand_axis_from_metric_model"))
            clock = stack.enter_context(patch("scripts.aufgabe04.real_robot.observer.qr_observation_pose.time.monotonic"))
            sensor_tuple = adapter._next_sensor_tuple.return_value
            for stamp in (100., 102.):
                self.fixture.clock_sec = stamp + .1
                clock.return_value = stamp
                for sample in (sensor_tuple.image, sensor_tuple.scan, sensor_tuple.camera_info):
                    sample.stamp_sec = stamp
                    sample.received_ros_sec = self.fixture.clock_sec
                    if hasattr(sample.value, "header"):
                        sample.value.header.stamp.sec = int(stamp)
                if stamp > 100.:
                    adapter.tf_retry_scheduler.offer(sensor_tuple, stamp_sec=stamp)
                adapter._process_latest()
            estimator.assert_not_called()
        self.assertIsNotNone(self.result(), json.loads(adapter.args.status_json.read_text()))
        self.assertEqual(self.result()["sensor_stamp_sec"], 102.)

    def test_usable_but_wrongly_associated_head_cannot_block_independent_qr(self):
        from scripts.aufgabe04.perception.stand_axis.models import ImagePoint, StandAxisEdgeDebugArtifacts
        from tests.aufgabe04.test_head_model_admission import head_estimate, head_debug, outer_boundary
        adapter = self.adapter
        adapter.profile.scan_frame = "scan"
        adapter.args.qr_pose_fallback_delay_sec = 0.
        adapter._write_status = PassiveRealViewpointNode._write_status.__get__(adapter)
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        module = "scripts.aufgabe04.real_robot.observer.node."
        options = association_fixtures.CurrentHeadAssociationTests().options()
        head, _, _ = association_fixtures.CurrentHeadAssociationTests().qr_binding(options)
        rejected = replace(head, accepted=False, reason="wrong_head_lidar_cluster")

        def decode(crop, *_args, **_kwargs):
            u, v = crop.shape[1] / 2, crop.shape[0] / 2
            return (DecodedQrObservation("QR_003", tuple((u + x, v + y)
                for x, y in ((-20, -20), (20, -20), (20, 20), (-20, 20))), "test"),)

        def metric(_cv2, _crop, **options):
            u, v = options["expected_head_center_u_px"], options["expected_head_center_v_px"]
            corners = tuple(ImagePoint(u + x, v + y)
                for x, y in ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            return (head_estimate(yaw_deg=10., corners=corners, left_height_px=52., right_height_px=52.),
                    StandAxisEdgeDebugArtifacts(**vars(head_debug(head_outer_recovery=outer_boundary(corners)))))

        with ExitStack() as stack:
            for name in ("camera_info_mismatches", "transform_mismatches"):
                stack.enter_context(patch(module + name, return_value=()))
            stack.enter_context(patch(module + "compressed_msg_to_bgr_frame", return_value=frame))
            stack.enter_context(patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_, **_kw: value))
            stack.enter_context(patch(module + "detect_qr_observations_bgr", side_effect=decode))
            stack.enter_context(patch(module + "detect_native_qr_observations_bgr", side_effect=decode))
            stack.enter_context(patch(module + "estimate_stand_axis_from_metric_model", side_effect=metric))
            association = stack.enter_context(patch(module + "associate_current_measured_head", return_value=rejected))
            update = stack.enter_context(patch.object(adapter, "_record_observation_frame", wraps=adapter._record_observation_frame))
            adapter._process_latest()
            association.assert_called_once()
            self.assertTrue(update.call_args.kwargs["lidar_associated"])
            self.assertIsNone(update.call_args.kwargs["axis_yaw_rad"])
        self.assertIsNotNone(self.result(), json.loads(adapter.args.status_json.read_text()))
        self.assertIsNone(self.result()["stand_axis_rad"])

    def test_artifact_rechecks_corners_cluster_provenance_and_discovery_scope(self):
        self.frame(100.)
        self.frame(102.)
        original = self.result()
        self.assertIsNotNone(original)
        mutations = [lambda p: p.update(stand_axis_rad=0.),
                     lambda p: p.update(facing_ready=True),
                     lambda p: p.update(qr_corners_px=((0, 0),) * 4),
                     lambda p: p.update(checked_at_sec=104.),
                     lambda p: p["localization_provenance"].update(exact_image_transform_stamp_sec=101.),
                     lambda p: (p["qr_binding"]["association"].get("search_association") or p["qr_binding"]["association"]).update(eligible_cluster_count=2)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                invalid = deepcopy(original)
                invalid.pop(HASH_FIELD)
                mutate(invalid)
                invalid = content_hashed_payload(invalid, hash_field=HASH_FIELD)
                with self.assertRaises(ValueError):
                    validate_qr_verified_observation_pose(invalid)
