"""Current stopped head location can yield advice before angle consensus."""

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.real_robot.observer.backside_head_crop import BacksideHeadCropReview
from scripts.aufgabe04.real_robot.observer.candidate_centering_receipt import (
    prepare_candidate_centering,
)
from scripts.aufgabe04.real_robot.observer.current_head_association import (
    associate_current_measured_head,
)
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode, build_parser, _validate_args
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.tf_retry import PassiveObserverTfRetryScheduler
from tests.aufgabe04 import test_camera_observer_processing as processing
from tests.aufgabe04 import test_current_head_association as association_fixtures
from tests.aufgabe04 import test_bounded_head_detection as bounded_fixtures
from tests.aufgabe04 import test_observer_measured_head_processing as measured_fixtures


class ObserverCandidateCenteringTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.fixture = processing.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        self.adapter.args.candidate_centering_json = self.root / "centering.json"
        self.adapter.args.status_json = self.root / "status.json"
        self.adapter.args.recommended_pose_json = self.root / "recommendation.json"
        self.adapter.stand_model_profile.sha256 = "a" * 64
        self.adapter.profile.odom_frame = "odom"
        self.adapter.profile.scan_frame = "base_scan"
        self.adapter.args.stand_model_profile = Path(__file__).resolve().parents[2] / (
            "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
        for module in ("candidate_centering_receipt", "qr_observation_pose"):
            for name in ("real_robot_profile_sha256", "camera_calibration_sha256"):
                mocked = patch("scripts.aufgabe04.real_robot.observer." + module + "." + name,
                               return_value="b" * 64)
                mocked.start()
                self.addCleanup(mocked.stop)

    def frame(self, stamp=100., *, bounded=False, complete=True, associated=True,
              ambiguous=False, age=.1, skew=0., pose=None, odom=True,
              symbols=0, qr=False, publish=True, axis=False, scan_topology=None):
        adapter = self.adapter
        self.fixture.clock_sec = stamp + age
        options = (bounded_fixtures.BoundedHeadDetectionTests().options() if bounded
                   else association_fixtures.CurrentHeadAssociationTests().options())
        adapter.stand_model_profile.sha256 = options["profile_sha256"]
        options["scan"] = replace(options["scan"], scan_stamp_sec=stamp + skew,
                                  receipt_sec=stamp + skew)
        if ambiguous:
            options["scan"] = replace(options["scan"],
                ranges=(.55, .55, math.inf, .55, .55, .55, .55, .55, .55))
        options["now_sec"] = stamp + .1
        association = associate_current_measured_head(**options)
        if scan_topology is not None:
            association = replace(association, lidar_association=replace(association.lidar_association,
                search_association=replace(association.lidar_association.search_association,
                    scan_topology=scan_topology)))
        if not associated:
            association = replace(association, accepted=False)
        pose = pose or Pose2D(0., 0., 0.)
        metadata = {}
        adapter._pending_candidate_centering = prepare_candidate_centering(
            crop=BacksideHeadCropReview(complete, "test_current_crop"),
            association=association, image_stamp_sec=stamp, scan_stamp_sec=stamp + skew,
            target_key=adapter._target_evidence_key(), robot_pose=pose,
            odom_pose=pose if odom else None, intrinsics=options["intrinsics"],
            scan_from_camera=options["scan_from_camera"],
            base_from_camera=RigidTransform("base", "camera", (0., 0., 0.),
                                           options["scan_from_camera"].rotation_xyzw),
            metadata=metadata)
        texts = ()
        if qr:
            _, binding, observations = association_fixtures.CurrentHeadAssociationTests().qr_binding(options)
            texts = binding.qr_texts_for_evidence
            adapter._pending_qr_observation_pose = prepare_qr_observation_pose(
                qr_binding=binding, qr_observations=observations, observed_qr_texts=texts,
                image_stamp_sec=stamp, scan_stamp_sec=stamp + skew, robot_pose=pose,
                target_key=adapter._target_evidence_key(), camera_signature=(640., 640., 400., 300.),
                image_shape=(600, 800, 3), roi=options["attempt"].roi,
                model_profile_sha256=adapter.stand_model_profile.sha256, metadata=metadata)
            symbols = binding.symbol_count
        update = adapter._record_observation_frame(
            robot_pose=pose, image_stamp_sec=stamp, scan_stamp_sec=stamp + skew,
            observed_at_sec=stamp + age, lidar_associated=association.accepted,
            axis_yaw_rad=.1 if axis else None, axis_source="test_geometry" if axis else None,
            qr_texts=texts, qr_symbol_count=symbols)
        if publish:
            PassiveRealViewpointNode._write_status(adapter, "collecting_consensus")
        return update, metadata

    def result(self):
        output = self.adapter.args.candidate_centering_json
        return json.loads(output.read_text()) if output is not None and output.exists() else None

    def test_first_stopped_current_head_does_not_wait_for_angle_samples(self):
        update, metadata = self.frame()
        payload = self.result()
        self.assertIsNotNone(payload, metadata)
        self.assertEqual(update.snapshot.current_axis_sample_count, 0)
        self.assertIsNone(update.axis_consensus)
        self.assertFalse(payload["motion_authorized"])
        self.assertTrue(self.adapter.completed)
        self.assertEqual(json.loads(self.adapter.args.status_json.read_text())["state"],
                         "candidate_centering_committed")
        self.assertFalse(self.adapter.args.recommended_pose_json.exists())

    def test_validated_orientation_bounds_locate_without_admitting_single_angle(self):
        update, metadata = self.frame(bounded=True)
        self.assertIsNotNone(self.result(), metadata)
        self.assertFalse(update.axis_sample_accepted)

    def run_processing(self, scenario, *, odom_available=True):
        original = processing.CameraObserverProcessingTest.make_adapter
        output = self.root / "processed_centering.json"
        def make_adapter(fixture):
            adapter = original(fixture)
            adapter.args.candidate_centering_json = output
            adapter.profile.odom_frame = "odom"
            lookup = adapter._lookup
            def lookup_with_odom(target, source, stamp):
                if target == "odom":
                    if not odom_available:
                        raise RuntimeError("exact odometry transform unavailable")
                    return processing.transform()
                return lookup(target, source, stamp)
            adapter._lookup = lookup_with_odom
            return adapter
        with patch.object(processing.CameraObserverProcessingTest, "make_adapter", make_adapter):
            adapter, recommendation = measured_fixtures.MeasuredHeadObserverProcessingTests().run_view(
                "physical_shifted_registered_" + scenario, publish_immediate=True)
        return adapter, recommendation, output

    def test_real_processing_preserves_productive_geometry_before_centering(self):
        adapter, recommendation, output = self.run_processing("head_only")
        self.assertIsNone(recommendation)
        self.assertFalse(output.exists())
        self.assertEqual(adapter._camera_pipeline_counters["processed_images"], 7)
        self.assertEqual(adapter.observation_evidence.snapshot().current_axis_sample_count, 7)
        self.assertFalse(adapter.completed)

    def test_productive_opportunity_is_fixed_and_does_not_renew_on_miss(self):
        module = "scripts.aufgabe04.real_robot.observer.candidate_centering_receipt.time.monotonic"
        for stamp, axis in ((100., True), (101., False), (104.9, True)):
            with patch(module, return_value=stamp):
                update, metadata = self.frame(stamp, axis=axis)
            self.assertIsNone(self.result())
            self.assertEqual(update.axis_sample_accepted, axis)
            self.assertEqual(metadata["candidate_centering"]["reason"], "preserve_productive_geometry_view")
        with patch(module, return_value=105.):
            self.frame(105.)
        self.assertIsNotNone(self.result())  # Linear scanner fixture: no boundary veto.

    def test_boundary_veto_does_not_publish_or_complete(self):
        _, metadata = self.frame(scan_topology=dict(profile="full_rotation", sample_count=360,
            angle_min_rad=0., angle_increment_rad=math.tau/360))
        self.assertIsNone(self.result())
        self.assertFalse(self.adapter.completed)
        self.assertEqual(metadata["candidate_centering"]["reason"],
                         "preserve_view_centering_would_cross_scan_boundary")


    def test_missing_optional_odom_does_not_suppress_current_qr_completion(self):
        adapter, recommendation, output = self.run_processing("bound_qr", odom_available=False)
        self.assertFalse(output.exists())
        self.assertIsNotNone(recommendation)
        self.assertTrue(adapter.completed)

    def test_missing_current_geometry_scan_uniqueness_or_odom_cannot_advise(self):
        for kwargs in ({"complete": False}, {"associated": False}, {"ambiguous": True},
                       {"odom": False}, {"age": .6}, {"skew": -.2}):
            with self.subTest(kwargs=kwargs):
                self.adapter._reset_observation_evidence()
                self.frame(**kwargs)
                self.assertIsNone(self.result())

    def test_poisoned_or_changed_stationary_epoch_cannot_advise(self):
        self.frame(publish=False)
        self.frame(100.2, symbols=2)
        self.assertIsNone(self.result())
        self.frame(100.4)
        self.assertIsNone(self.result())
        self.adapter._reset_observation_evidence()
        self.frame(publish=False)
        update, _ = self.frame(100.2, pose=Pose2D(.03, 0., 0.))
        self.assertTrue(update.motion_epoch_reset)
        self.assertIsNone(self.result())

    def test_current_qr_completion_has_priority_over_centering(self):
        self.adapter.args.qr_observation_pose_json = self.root / "qr_pose.json"
        self.frame(qr=True)
        self.assertTrue(self.adapter.args.qr_observation_pose_json.exists())
        self.assertIsNone(self.result())
        self.assertEqual(json.loads(self.adapter.args.status_json.read_text())["state"],
                         "qr_observation_pose_committed")

    def test_publication_rechecks_age_and_does_not_reuse_declined_receipt(self):
        self.frame(publish=False)
        self.fixture.clock_sec = 100.6
        PassiveRealViewpointNode._write_status(self.adapter, "collecting_consensus")
        self.assertIsNone(self.result())
        self.fixture.clock_sec = 100.1
        PassiveRealViewpointNode._write_status(self.adapter, "waiting_for_tf")
        self.assertIsNone(self.result())

    def test_disabled_by_default_and_outputs_are_distinct(self):
        self.assertIsNone(build_parser().get_default("candidate_centering_json"))
        self.adapter.args.candidate_centering_json = None
        self.frame()
        self.assertFalse(self.adapter.completed)
        self.adapter.args.candidate_centering_json = self.adapter.args.status_json
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            _validate_args(build_parser(), self.adapter.args)
        self.assertIn("must be distinct", stderr.getvalue())

    def test_both_sensor_stamps_must_advance_after_any_turn(self):
        for image_stamp, scan_stamp in ((100., 100.1), (100.1, 100.), (99.9, 100.1)):
            with self.subTest(image_stamp=image_stamp, scan_stamp=scan_stamp):
                adapter = self.fixture.make_adapter()
                adapter.args.observation_not_before_sec = 100.
                sensor = adapter._next_sensor_tuple.return_value
                # The delivery scheduler binds the offered image timestamp.
                adapter.tf_retry_scheduler = PassiveObserverTfRetryScheduler()
                sensor.image.stamp_sec, sensor.scan.stamp_sec = image_stamp, scan_stamp
                adapter.tf_retry_scheduler.offer(sensor, stamp_sec=image_stamp)
                adapter._process_latest()
                self.assertEqual(adapter._write_status.call_args.args,
                                 ("awaiting_post_turn_sensor_tuple",))
                self.assertIsNone(adapter.observation_evidence)
        for value in (-1., math.inf, math.nan):
            adapter.args.observation_not_before_sec = value
            with self.subTest(cutoff=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                _validate_args(build_parser(), adapter.args)


if __name__ == "__main__":
    unittest.main()
