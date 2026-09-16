"""Opposite-side inspection consumes a new current-head front receipt.

The image/model fixtures are synthetic: these tests exercise real association,
admission, serialization and inspection transitions, not camera accuracy or
robot motion. Recorded-pixel acquisition is covered by the perception tests.
"""

from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.aufgabe04.artifacts.backside_axis_observation import load_backside_axis_observation
from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_preapproach_materialization import (
    validate_backside_axis_candidate_binding,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateObservation, FacingValidationRequest, validate_facing_pose,
)
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.observer.backside_head_crop import BacksideHeadCropReview
from scripts.aufgabe04.real_robot.observer.camera_context import camera_context_signature
from scripts.aufgabe04.real_robot.observer.current_head_qr_binding import bind_qr_to_current_head
from scripts.aufgabe04.real_robot.observer.immediate_front_observation import prepare_immediate_front
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_current_head_association as association_fixtures
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload


class OppositeSideFrontAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate_uid = "candidate"
        self.back_pose = Pose2D(1.2, 0., math.pi)
        self.front_stamp = 124.
        self.captures = []
        self.moves = []

    def front_frame(self, output, pose, *, age=.1, predicted=False,
                    decoded=True, neighboring_qr=False):
        """Run current scan association and production publication gates."""
        self.fixture = processing_fixtures.CameraObserverProcessingTest()
        self.adapter = adapter = self.fixture.make_adapter()
        self.fixture.clock_sec = self.front_stamp + age
        adapter.stand_model_profile.sha256 = "a" * 64
        output.mkdir(parents=True, exist_ok=True)
        adapter.args.recommended_pose_json = output / "recommendation.json"
        adapter.args.axis_observation_json = output / "axis_observation.json"
        adapter.args.status_json = output / "status.json"
        fixture = association_fixtures.CurrentHeadAssociationTests()
        options = fixture.options()
        scan = options["scan"]
        if neighboring_qr:
            # Two separate real scan clusters: a decoded neighboring symbol
            # can be independently associated but cannot identify this head.
            scan = replace(scan, ranges=(.55,) * 7 + (math.inf,) * 6 + (.55,) * 7)
        options.update(
            scan=replace(scan, scan_stamp_sec=self.front_stamp, receipt_sec=self.front_stamp),
            now_sec=self.front_stamp + .1,
        )
        head, binding, observations = fixture.qr_binding(
            options, center=(20., 80.) if neighboring_qr else (80., 80.))
        self.assertTrue(head.accepted, head.reason)
        self.assertTrue(binding.accepted, binding.reason)
        binding = bind_qr_to_current_head(
            binding, observations, head_corners=options["estimate"].corners,
            head_association=head)
        if neighboring_qr:
            self.assertEqual(binding.reason, "qr_outside_current_head")
        if not decoded:
            observations = ()
            binding = QrTargetBinding(False, "no_decoded_qr_geometry")
        estimate = options["estimate"]
        if predicted:
            estimate = replace(estimate, evidence_state="predicted_only")
        camera_info = SimpleNamespace(
            k=np.array([640., 0., 400., 0., 640., 300., 0., 0., 1.]),
            d=np.zeros(5), r=np.eye(3).ravel(),
            p=np.array([640., 0., 400., 0., 0., 640., 300., 0., 0., 0., 1., 0.]),
            distortion_model="plumb_bob",
        )
        signature = camera_context_signature(
            camera_frame="camera", intrinsics=np.array([640., 640., 400., 300.]),
            camera_info=camera_info, scan_translation=np.zeros(3),
            scan_rotation=np.array([-.5, .5, -.5, .5]))
        self.metadata = {}
        adapter._pending_immediate_front = prepare_immediate_front(
            estimate=estimate, debug=options["debug"], association=head,
            crop=BacksideHeadCropReview(True, "current_complete_crop"),
            qr_binding=binding, qr_observations=observations,
            observed_qr_texts=tuple(item.text for item in observations),
            image_stamp_sec=self.front_stamp, scan_stamp_sec=self.front_stamp,
            robot_pose=pose, camera_heading_rad=pose.yaw_rad,
            target_key=adapter._target_evidence_key(), camera_signature=signature,
            image_shape=(600, 800, 3), roi=options["attempt"].roi, metadata=self.metadata)
        self.update = adapter._record_observation_frame(
            robot_pose=pose, image_stamp_sec=self.front_stamp, scan_stamp_sec=self.front_stamp,
            observed_at_sec=self.front_stamp + age, lidar_associated=head.accepted,
            axis_yaw_rad=None, axis_source=None, qr_texts=binding.qr_texts_for_evidence,
            qr_symbol_count=binding.symbol_count)
        PassiveRealViewpointNode._write_status(adapter, "collecting_consensus")
        path = adapter.args.recommended_pose_json
        if not path.exists():
            raise CandidateObservationUnavailableError(
                candidate_uid=self.candidate_uid, observation_attempt_index=1,
                reason="front_not_admitted", process_evidence={},
                status_evidence=self.metadata)
        # Reload the actual artifact before the mission can consume it.
        self.recommendation = load_recommendation(path)
        return CandidateObservation(path, self.recommendation.axis_measurement["qr_id"], None)

    def inspect(self, **front_options):
        def capture(pose, output, index):
            self.captures.append((pose, index))
            if index == 0:
                output.mkdir(parents=True)
                self.backside_path = output / "axis_observation.json"
                payload = backside_axis_payload(
                    stand_id=self.candidate_uid, stand_x_m=.6,
                    robot_x_m=pose.x_m, robot_y_m=pose.y_m,
                    robot_yaw_rad=pose.yaw_rad, stand_axis_rad=math.pi / 2.)
                self.backside_path.write_text(json.dumps(payload))
                self.backside_bytes = self.backside_path.read_bytes()
                return CandidateObservation(None, None, self.backside_path)
            return self.front_frame(output, pose, **front_options)

        def opposite(pose, observation, output, index):
            # Motion is injected, while its source receipt and candidate binding
            # use the same validators as the opposite-face route materializer.
            axis = load_backside_axis_observation(observation.axis_observation_path)
            validate_backside_axis_candidate_binding(
                axis, candidate_uid=self.candidate_uid, planning_frame="map",
                candidate_x_m=.6, candidate_y_m=0.)
            normal = axis.opposite_face_normal_rad
            self.moves.append((index, observation.axis_observation_path, normal))
            return Pose2D(.6 + .6 * math.cos(normal), .6 * math.sin(normal),
                          math.remainder(normal + math.pi, 2. * math.pi))

        return execute_candidate_inspection(
            candidate_uid=self.candidate_uid, candidate_root=self.root,
            initial_frame=self.back_pose, max_views=2,
            effects=CandidateInspectionEffects(
                capture=capture,
                canonical_normal=lambda pose: math.atan2(pose.y_m, pose.x_m - .6),
                move_view=lambda *args: self.fail("unexpected additional inspection move"),
                move_opposite=opposite,
                progress_evidence=lambda *args: self.fail("front receipt became advisory")))

    def test_first_fresh_front_fit_after_opposite_arrival_completes_same_candidate(self):
        observation, front_pose = self.inspect()
        recommendation = self.recommendation
        self.assertEqual([index for _, index in self.captures], [0, 1])
        self.assertEqual(len(self.moves), 1)
        self.assertAlmostEqual(front_pose.x_m, 0.)
        self.assertEqual(observation.qr_id, "QR_003")
        self.assertEqual(recommendation.stand_id, self.candidate_uid)
        self.assertEqual(recommendation.sensor_stamp_sec, self.front_stamp)
        self.assertEqual(recommendation.axis_sample_count, 1)
        self.assertIsNone(self.update.axis_consensus)
        self.assertIsNone(self.update.resolved_qr_id)
        measurement = recommendation.axis_measurement
        self.assertEqual(measurement["policy"], "current_head_and_bound_qr")
        self.assertEqual(measurement["target_key"], self.adapter._target_evidence_key())
        self.assertEqual(measurement["qr_sensor_stamp_sec"], self.front_stamp)
        self.assertEqual(measurement["head_lidar_association"]["search_association"]["scan_stamp_sec"],
                         self.front_stamp)
        self.assertEqual(measurement["camera_signature"][0], "camera")
        self.assertTrue(self.adapter.completed)
        self.assertFalse(self.adapter.args.axis_observation_json.exists())
        self.assertEqual(self.backside_path.read_bytes(), self.backside_bytes)
        progress = json.loads((self.root / "inspection_progress.json").read_text())
        receipt = load_content_hashed_json(
            Path(progress["latest_revision_path"]),
            hash_field="candidate_inspection_progress_sha256")
        self.assertEqual(receipt["view_history"], progress["view_history"])
        self.assertEqual(receipt["candidate_uid"], self.candidate_uid)
        self.assertEqual(receipt["termination_reason"], "joint_observation_ready")
        self.assertEqual(receipt["view_history"][0]["observation"]["classification"], "certified_backside")
        self.assertEqual(receipt["view_history"][1]["observation"]["recommendation_path"],
                         str(observation.recommendation_path))

    def test_backside_route_cannot_replace_fresh_geometry_or_same_head_identity(self):
        for options in ({"age": .501}, {"predicted": True},
                        {"decoded": False}, {"neighboring_qr": True}):
            with self.subTest(options=options):
                self.setUp()
                with self.assertRaises(CandidateObservationUnavailableError):
                    self.inspect(**options)
                self.assertEqual(len(self.moves), 1)
                self.assertFalse(self.adapter.completed)
                self.assertFalse(self.adapter.args.recommended_pose_json.exists())
                progress = json.loads((self.root / "inspection_progress.json").read_text())
                self.assertFalse(progress["joint_observation_ready"])
                self.assertEqual(progress["view_history"][-1]["outcome"], "observation_unavailable")

    def test_published_front_receipt_cannot_be_assigned_to_another_candidate(self):
        observation, front_pose = self.inspect()
        with self.assertRaisesRegex(ValueError, "stand_id mismatch"):
            validate_facing_pose(FacingValidationRequest(
                config=SimpleNamespace(planning_frame="map"),
                candidate=SimpleNamespace(candidate_uid="neighboring_candidate"),
                recommendation_path=observation.recommendation_path,
                current_pose=front_pose, output_dir=self.root / "wrong_candidate"))


if __name__ == "__main__":
    unittest.main()
