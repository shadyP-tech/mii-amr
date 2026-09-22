"""One real head-quality decision and independently bound QR end observation.

Synthetic geometry exercises admission/receipt transport, not camera accuracy.
"""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.backside_head_crop import BacksideHeadCropReview
from scripts.aufgabe04.real_robot.observer.current_head_qr_binding import bind_qr_to_current_head
from scripts.aufgabe04.real_robot.observer.immediate_front_observation import (
    prepare_immediate_front, commit_immediate_front,
)
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_current_head_association as association_fixtures
from tests.aufgabe04.test_head_model_admission import head_debug, outer_boundary


class ImmediateFrontObservationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = processing_fixtures.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        self.adapter.stand_model_profile.sha256 = "a" * 64
        self.adapter.args.recommended_pose_json = self.root / "recommendation.json"
        self.adapter.args.axis_observation_json = self.root / "backside.json"
        self.adapter.args.status_json = self.root / "status.json"

    def frame(self, stamp=100., *, decode=True, usable=True, associated=True,
              complete=True, age=.1, scan_offset=0., proof_offset=0.,
              qr_id="QR_003", symbols=None, pose=None, camera_signature=(640., 640., 400., 300.),
              head_shift=0., head_width=90., predicted=False, publish=True):
        self.fixture.clock_sec = stamp + age
        adapter = self.adapter
        options = association_fixtures.CurrentHeadAssociationTests().options()
        options["scan"] = replace(options["scan"], scan_stamp_sec=stamp + proof_offset,
                                   receipt_sec=stamp + proof_offset)
        options["now_sec"] = stamp + .1
        if head_shift or head_width != 90.:
            corners = tuple(ImagePoint(80. + head_shift + x, 80. + y)
                for x, y in ((-head_width / 2, -45.), (head_width / 2, -45.),
                             (head_width / 2, 45.), (-head_width / 2, 45.)))
            options["estimate"] = replace(options["estimate"], corners=corners)
            options["debug"] = head_debug(head_outer_recovery=outer_boundary(corners))
        head, binding, observations = association_fixtures.CurrentHeadAssociationTests().qr_binding(options)
        observations = tuple(replace(o, text=qr_id) for o in observations)
        binding = replace(binding, qr_texts_for_evidence=(qr_id,))
        binding = bind_qr_to_current_head(binding, observations,
            head_corners=options["estimate"].corners, head_association=head)
        if not associated:
            head = replace(head, accepted=False)
            binding = replace(binding, accepted=False, qr_texts_for_evidence=())
        if not decode:
            observations, binding = (), QrTargetBinding(False, "no_decoded_qr_geometry")
        estimate = options["estimate"]
        if not usable:
            estimate = replace(estimate, usable=False, yaw_deg=None)
        if predicted:
            estimate = replace(estimate, evidence_state="predicted_only")
        pose = pose or Pose2D(0., 0., 0.)
        metadata = {}
        adapter._pending_immediate_front = prepare_immediate_front(
            estimate=estimate, debug=options["debug"], association=head,
            crop=BacksideHeadCropReview(complete, "test_current_crop"),
            qr_binding=binding, qr_observations=observations,
            observed_qr_texts=() if not decode else (qr_id,),
            image_stamp_sec=stamp, scan_stamp_sec=stamp + scan_offset, robot_pose=pose,
            camera_heading_rad=0., target_key=adapter._target_evidence_key(),
            camera_signature=camera_signature, image_shape=(600, 800, 3),
            roi=options["attempt"].roi, metadata=metadata)
        if getattr(adapter.args, "qr_observation_pose_json", None) is not None:
            adapter._pending_qr_observation_pose = prepare_qr_observation_pose(
                qr_binding=binding, qr_observations=observations,
                observed_qr_texts=() if not decode else (qr_id,), image_stamp_sec=stamp,
                scan_stamp_sec=stamp+scan_offset, robot_pose=pose,
                target_key=adapter._target_evidence_key(), camera_signature=camera_signature,
                image_shape=(600, 800, 3), roi=options["attempt"].roi,
                model_profile_sha256=adapter.stand_model_profile.sha256, metadata=metadata)
        update = adapter._record_observation_frame(
            robot_pose=pose, image_stamp_sec=stamp, scan_stamp_sec=stamp + scan_offset,
            observed_at_sec=stamp + age, lidar_associated=associated,
            axis_yaw_rad=None, axis_source=None, qr_texts=binding.qr_texts_for_evidence,
            qr_symbol_count=binding.symbol_count if symbols is None else symbols)
        if publish:
            PassiveRealViewpointNode._write_status(adapter, "collecting_consensus")
        return update, metadata

    def recommendation(self):
        path = self.adapter.args.recommended_pose_json
        return load_recommendation(json.loads(path.read_text())) if path.exists() else None

    def test_first_bound_qr_and_current_head_commits_without_either_consensus(self):
        update, metadata = self.frame()
        recommendation = self.recommendation()
        self.assertIsNotNone(recommendation, metadata)
        self.assertIsNone(update.axis_consensus)
        self.assertIsNone(update.resolved_qr_id)  # Legacy two-QR latch is unchanged.
        self.assertEqual(update.snapshot.tentative_qr_id, "QR_003")
        self.assertEqual(recommendation.axis_sample_count, 1)
        self.assertEqual(recommendation.axis_confidence, 0.)
        self.assertEqual(recommendation.side_evidence.kind, "qr_observation")
        self.assertEqual(recommendation.axis_measurement["policy"], "current_head_and_bound_qr")
        self.assertTrue(self.adapter.completed)
        self.assertIsNone(commit_immediate_front(self.adapter))

    def test_qr_can_precede_geometry_without_redecoding(self):
        self.frame(100., usable=False)
        self.assertIsNone(self.recommendation())
        update, metadata = self.frame(100.4, decode=False)
        recommendation = self.recommendation()
        self.assertIsNotNone(recommendation, metadata)
        self.assertFalse(update.qr_sample_accepted)
        self.assertEqual(recommendation.axis_measurement["qr_sensor_stamp_sec"], 100.)
        self.assertEqual(recommendation.sensor_stamp_sec, 100.4)

    def test_mission_grace_leaves_observer_alive_for_following_current_geometry(self):
        from scripts.aufgabe04.real_robot.autonomous_runner.cli import DEFAULT_QR_POSE_FALLBACK_DELAY_SEC
        self.adapter.args.qr_observation_pose_json = self.root / "qr_pose.json"
        self.adapter.args.qr_pose_fallback_delay_sec = DEFAULT_QR_POSE_FALLBACK_DELAY_SEC
        with patch("scripts.aufgabe04.real_robot.observer.qr_observation_pose.time.monotonic", return_value=100.):
            _, metadata = self.frame(100., usable=False)
        self.assertEqual(metadata["qr_observation_pose_fallback"]["reason"], "same_pose_geometry_grace_pending")
        self.assertFalse(self.adapter.completed)
        with patch("scripts.aufgabe04.real_robot.observer.qr_observation_pose.time.monotonic", return_value=100.4):
            self.frame(100.4, decode=False)
        self.assertIsNotNone(self.recommendation())
        self.assertTrue(self.adapter.completed)
        self.assertFalse(self.adapter.args.qr_observation_pose_json.exists())

    def test_geometry_alone_cannot_admit_or_authorize_backside(self):
        for index in range(8):
            self.frame(100. + index * .1, decode=False)
        self.assertIsNone(self.recommendation())
        self.assertFalse(self.adapter.args.axis_observation_json.exists())

    def test_expired_latch_and_context_change_cannot_supply_identity(self):
        for changes in ({"stamp": 101.01},
                        {"pose": Pose2D(.03, 0., 0.)},
                        {"camera_signature": (641., 640., 400., 300.)}):
            with self.subTest(changes=changes):
                self.setUp()
                self.frame(100., usable=False)
                self.frame(**{"stamp": 100.4, "decode": False, **changes})
                self.assertIsNone(self.recommendation())

    def test_cached_qr_must_remain_inside_current_complete_head(self):
        self.frame(100., usable=False)
        _, metadata = self.frame(100.4, decode=False, head_shift=24., head_width=50.)
        self.assertIsNone(self.recommendation())
        self.assertEqual(metadata["immediate_front_admission"]["reason"],
                         "bound_qr_outside_current_complete_head")

    def test_motion_reset_frame_cannot_commit_even_with_new_qr_and_geometry(self):
        self.frame(100., usable=False)
        pose = Pose2D(.03, 0., 0.)
        update, metadata = self.frame(100.2, pose=pose)
        self.assertTrue(update.motion_epoch_reset)
        self.assertIsNone(self.recommendation())
        self.assertEqual(metadata["immediate_front_admission"]["reason"],
                         "stationary_epoch_changed_awaiting_stopped_frame")
        self.frame(100.4, pose=pose)
        self.assertIsNotNone(self.recommendation())

    def test_source_currentness_crop_and_own_scan_remain_required(self):
        for changes in ({"usable": False}, {"associated": False}, {"complete": False},
                        {"age": .501}, {"scan_offset": -.11}, {"proof_offset": -.05},
                        {"predicted": True}):
            with self.subTest(changes=changes):
                self.setUp()
                self.frame(**changes)
                self.assertIsNone(self.recommendation())

    def test_multiple_or_conflicting_identities_cannot_use_previous_qr(self):
        for changes in ({"symbols": 2}, {"qr_id": "QR_004"}):
            with self.subTest(changes=changes):
                self.setUp()
                self.frame(100., usable=False)
                update, _ = self.frame(100.2, **changes)
                self.assertTrue(update.snapshot.poisoned)
                self.frame(100.4, decode=False)
                self.assertIsNone(self.recommendation())

    def test_repeated_source_cannot_complete_after_previous_geometry_miss(self):
        self.frame(100., usable=False)
        update, _ = self.frame(100.)
        self.assertFalse(update.frame_accepted)
        self.assertIsNone(self.recommendation())

    def test_fresh_fit_cannot_reuse_qr_that_expires_during_publication(self):
        self.frame(100., usable=False)
        self.frame(100.8, decode=False, publish=False)
        self.fixture.clock_sec = 101.01
        self.assertIsNone(commit_immediate_front(self.adapter))
        self.assertIsNone(self.recommendation())

    def test_current_fit_is_rechecked_after_serialization(self):
        self.frame(publish=False)
        module = "scripts.aufgabe04.real_robot.observer.node"
        from scripts.aufgabe04.real_robot.observer.node import _atomic_json

        def delayed_write(path, payload, *, before_commit):
            self.fixture.clock_sec = 100.51
            return _atomic_json(path, payload, before_commit=before_commit)

        with patch(module + "._atomic_json", side_effect=delayed_write):
            self.assertIsNone(commit_immediate_front(self.adapter))
        self.assertIsNone(self.recommendation())

    def test_latched_qr_is_rechecked_after_serialization_while_head_remains_fresh(self):
        self.frame(100., usable=False)
        self.frame(100.8, decode=False, publish=False)
        from scripts.aufgabe04.real_robot.observer.node import _atomic_json

        def delayed_write(path, payload, *, before_commit):
            self.fixture.clock_sec = 101.01  # Head age .21 s; QR age exceeds 1 s.
            return _atomic_json(path, payload, before_commit=before_commit)

        with patch("scripts.aufgabe04.real_robot.observer.node._atomic_json", side_effect=delayed_write):
            self.assertIsNone(commit_immediate_front(self.adapter))
        self.assertIsNone(self.recommendation())

    def test_immediate_admission_takes_priority_over_pending_bounded_window(self):
        self.frame(publish=False)
        with patch("scripts.aufgabe04.real_robot.observer.node.commit_bounded_head",
                   side_effect=AssertionError("strict front was already admitted")):
            PassiveRealViewpointNode._write_status(self.adapter, "collecting_consensus")
        self.assertIsNotNone(self.recommendation())


if __name__ == "__main__":
    unittest.main()
