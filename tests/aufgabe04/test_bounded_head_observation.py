"""Bounded head receipts retain real sensor, identity and artifact gates.

Synthetic current pixels exercise real IPPE/covariance, association, crop,
ordinary observer evidence and receipt validators. They do not claim accuracy
for a recorded stand or authorize motion.
"""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy

from scripts.aufgabe04.artifacts.backside_axis_observation import validated_backside_axis_observation
from scripts.aufgabe04.artifacts.candidate_inspection_observation import load_candidate_inspection_observation
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.head_backside_appearance import assess_current_head_backside_appearance
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import evaluate_current_head_orientation_bounds
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix, estimate_planar_pose_ippe
from scripts.aufgabe04.real_robot.observer.backside_head_crop import review_current_head_crop, review_backside_head_crop
from scripts.aufgabe04.real_robot.observer.bounded_head_observation import prepare_bounded_head, commit_bounded_head
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.current_head_qr_binding import bind_qr_to_current_head
from scripts.aufgabe04.real_robot.observer.head_observation_confidence import HeadConfidenceInput
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import register_current_tracked_head, tracked_head_selection
from scripts.aufgabe04.real_robot.candidate.inspection_policy import candidate_view_options
from tests.aufgabe04.test_bounded_head_detection import bounded_detection
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_current_head_association as association_fixtures
from tests.aufgabe04.test_head_model_admission import outer_boundary
from tests.aufgabe04.test_candidate_snapshot import _candidate, _snapshot
from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation


class BoundedHeadObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_measured_physical_stand_model(Path(__file__).resolve().parents[2] /
            "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
        cls.camera = RectifiedCameraMatrix(640., 640., 120., 80.)
        model = numpy.asarray([(p.x_m, p.y_m, p.z_m) for p in cls.profile.head_corners])
        points, _ = cv2.projectPoints(model, numpy.asarray((0., .1, 0.)),
            numpy.asarray((-.03, 0., .55)), numpy.asarray(((640., 0., 120.),
            (0., 640., 80.), (0., 0., 1.))), None)
        corners = tuple(ImagePoint(float(u), float(v)) for u, v in points.reshape(-1, 2))
        poses = estimate_planar_pose_ippe(cv2, corners, cls.profile.head_corners, cls.camera)
        cls.proof = evaluate_current_head_orientation_bounds(
            cv2, profile=cls.profile, camera=cls.camera, corners=corners, pose_result=poses,
            raw_border_support_mean=.98, raw_corner_support_accepted=True,
            outer_border_verified=True, frame_shape=(160, 160))
        # This current fit exceeds the strict 3-degree single-angle error gate,
        # but its explicit noise-expanded range remains within 15 degrees.
        assert cls.proof.accepted and math.degrees(cls.proof.hypotheses[0].yaw_std_rad) > 3.
        assert cls.proof.half_width_rad < math.radians(15.)
        current, _ = bounded_detection()
        cls.current = replace(current,
            estimate=replace(current.estimate, corners=corners,
                reason="head_model_yaw_uncertainty_too_high"),
            debug=replace(current.debug, refined_corners=corners,
                head_orientation_bounds=cls.proof,
                head_outer_recovery=outer_boundary(corners, cls.profile.sha256),
                pose_reprojection_rmse_px=poses.hypotheses[0].reprojection_rmse_px,
                pose_ambiguity_gap_px=0., qr_detected=False, qr_marker_verified=False))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = processing_fixtures.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        self.adapter.stand_model_profile = self.profile
        self.adapter.args.recommended_pose_json = self.root / "recommendation.json"
        self.adapter.args.axis_observation_json = self.root / "backside.json"
        self.adapter.args.status_json = self.root / "status.json"
        for name, digest in (("real_robot_profile_sha256", "a" * 64),
                             ("camera_calibration_sha256", "b" * 64)):
            for module in ("bounded_head_observation", "node"):
                mocked = patch("scripts.aufgabe04.real_robot.observer." + module + "." + name,
                               return_value=digest)
                mocked.start()
                self.addCleanup(mocked.stop)

    def frame(self, stamp, *, face="front", associated=True, complete=True,
              marker=None, marker_seen=None, age=.1, scan_stamp=None,
              scan_for_proof=None, qr_id="QR_003", qr_conflict=False,
              pose=None, publish=True, no_appearance=False, reconcile=False):
        marker = face == "front" if marker is None else marker
        marker_seen = marker if marker_seen is None else marker_seen
        scan_stamp = stamp if scan_stamp is None else scan_stamp
        scan_for_proof = scan_stamp if scan_for_proof is None else scan_for_proof
        self.fixture.clock_sec = stamp + age
        pose = Pose2D(0., 0., 0.) if pose is None else pose
        current = replace(self.current, debug=replace(self.current.debug,
            qr_detected=marker, qr_marker_verified=marker), qr_observations=())
        appearance = assess_current_head_backside_appearance(
            current.estimate, current.debug, model_profile=self.profile, camera=self.camera,
            expected_center_u_px=sum(p.u_px for p in current.estimate.corners) / 4,
            expected_center_v_px=80., expected_height_px=90.)
        current = replace(current, debug=replace(current.debug,
            head_backside_appearance=None if no_appearance else appearance))
        options = association_fixtures.CurrentHeadAssociationTests().options()
        options.update(estimate=current.estimate, debug=current.debug, attempt=current.attempt,
                       profile_sha256=self.profile.sha256, now_sec=stamp+.1,
                       scan=replace(options["scan"], scan_stamp_sec=scan_for_proof, receipt_sec=scan_for_proof))
        if reconcile:
            if not hasattr(self, "reconciliation"):
                self.reconciliation = StoppedTargetReconciliation()
                candidate = _candidate(uid=self.adapter.args.stand_id, x_m=.6)
                candidate = replace(candidate, geometry=replace(candidate.geometry,
                    y_m=0., radius_m=.04, uncertainty_m=.02))
                self.snapshot_path = self.root / "snapshot.json"
                write_candidate_snapshot(self.snapshot_path, _snapshot(candidate))
            options["target_reconciliation"] = self.reconciliation.observe(
                snapshot_path=self.snapshot_path, candidate_uid=self.adapter.args.stand_id,
                planning_frame="map", stand_center=(.6, 0.), target_key="test", epoch=0,
                scan=options["scan"], scan_from_map=RigidTransform("base_scan", "map",
                    (0., 0., 0.), (0., 0., 0., 1.)), robot_pose=(pose.x_m, pose.y_m, pose.yaw_rad),
                image_stamp_sec=stamp, now_sec=stamp+.1,
                options={key: options[key] for key in ("map_bearing_rad", "cone_half_angle_rad",
                    "max_camera_map_bearing_delta_rad", "accepted_range_m")})
        association = associate_current_measured_head(**options)
        if not associated:
            association = replace(association, accepted=False)
        selection = register_current_tracked_head(tracked_head_selection(current),
            association=association, observed_at_sec=stamp, now_sec=stamp+.1,
            max_age_sec=.5, expected_model_sha256=self.profile.sha256)
        crop = review_current_head_crop(selection)
        appearance_crop = review_backside_head_crop(selection)
        if not complete:
            crop = replace(crop, accepted=False, reason="fitted_head_crop_clipped")
            appearance_crop = replace(appearance_crop, accepted=False, reason="fitted_head_crop_clipped")
        binding = QrTargetBinding(False, "no_decoded_qr_geometry")
        if face == "front":
            _, binding, observations = association_fixtures.CurrentHeadAssociationTests().qr_binding(
                options, center=(sum(p.u_px for p in current.estimate.corners) / 4, 80.))
            observations = tuple(replace(item, text=qr_id) for item in observations)
            binding = replace(binding, qr_texts_for_evidence=(qr_id,))
            binding = bind_qr_to_current_head(binding, observations,
                head_corners=current.estimate.corners, head_association=association)
        metadata = {}
        self.adapter._pending_bounded_head = prepare_bounded_head(
            estimate=current.estimate, debug=current.debug, association=association,
            crop=crop, appearance_crop=appearance_crop, qr_binding=binding,
            marker_verified=marker, marker_seen_in_epoch=marker_seen,
            image_stamp_sec=stamp, scan_stamp_sec=scan_stamp, robot_pose=pose,
            camera_heading_rad=0., stand_x_m=.6, stand_y_m=0.,
            camera_signature=(640., 640., 400., 300.), roi=current.attempt.roi,
            metadata=metadata, projected_center_px=(400., 300.), expected_head_height_px=90.,
            head_position_evidence=(None if association.target_reconciliation is None else dict(
                head_bounds=asdict(self.proof), scan_from_camera=asdict(options["scan_from_camera"]),
                model_path=str(Path(__file__).resolve().parents[2] /
                    "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"))))
        self.adapter._qr_marker_seen_in_stationary_epoch = marker_seen
        self.adapter._pending_head_confidence = (HeadConfidenceInput(
            stamp, None if no_appearance else appearance, appearance_crop.accepted,
            marker, marker_seen, False, current.estimate.reason), metadata)
        texts = binding.qr_texts_for_evidence
        if qr_conflict:
            texts = ("QR_003", "QR_004")
        update = self.adapter._record_observation_frame(
            robot_pose=pose, image_stamp_sec=stamp, scan_stamp_sec=scan_stamp,
            observed_at_sec=stamp+age, lidar_associated=associated,
            axis_yaw_rad=None, axis_source=None, qr_texts=texts,
            qr_symbol_count=2 if qr_conflict else binding.symbol_count)
        if publish:
            # Exercise the real node ordering: bounded receipt gets its chance
            # before advisory inspection completion, with real durable writes.
            PassiveRealViewpointNode._write_status(self.adapter, "metric_model_measurement_unavailable")
        return update, metadata

    def payload(self, face):
        path = self.adapter.args.recommended_pose_json if face == "front" else self.adapter.args.axis_observation_json
        return json.loads(path.read_text()) if path.exists() else None

    def test_front_seven_real_fresh_frames_commit_interval_without_single_angle(self):
        for index in range(7):
            update, metadata = self.frame(100.+index*.2)
            if index < 6:
                self.assertIsNone(self.payload("front"))
        payload = self.payload("front")
        self.assertIsNotNone(payload, metadata)
        recommendation = load_recommendation(payload)
        self.assertEqual(recommendation.axis_sample_count, 7)
        self.assertEqual(recommendation.axis_confidence, 0.)
        self.assertAlmostEqual(recommendation.bounded_orientation["half_width_rad"], self.proof.half_width_rad)
        self.assertFalse(payload["axis_measurement"]["single_angle_confidence_claimed"])
        self.assertEqual(update.resolved_qr_id, "QR_003")
        self.assertFalse(update.axis_sample_accepted)
        self.assertIsNone(update.axis_consensus)
        self.assertTrue(self.adapter.completed)
        self.assertEqual(json.loads(self.adapter.args.status_json.read_text())["state"], "recommendation_committed")
        self.assertIsNone(commit_bounded_head(self.adapter))

    def test_backside_seven_current_appearance_samples_commit_opposite_receipt(self):
        for index in range(7):
            update, metadata = self.frame(100.+index*.2, face="backside")
            if index < 6:
                self.assertIsNone(self.payload("backside"))
        payload = self.payload("backside")
        self.assertIsNotNone(payload, metadata)
        observation = validated_backside_axis_observation(payload)
        self.assertEqual(observation.axis_sample_count, 7)
        self.assertEqual(observation.axis_confidence, 0.)
        self.assertIsNotNone(observation.opposite_face_normal_rad)
        self.assertTrue(self.adapter.axis_observation_committed)
        self.assertFalse(self.adapter._head_confidence_metadata["backside"]["neck_required"])
        self.assertIsNone(update.axis_consensus)
        self.assertIsNone(update.resolved_qr_id)
        self.assertIsNone(self.payload("front"))

    def test_reconciled_backside_retains_metric_center_without_diagnostic_lookup(self):
        for index in range(7):
            _, metadata = self.frame(100.+index*.2, face="backside", reconcile=True)
        # Production passes the nested metric diagnostics to the window. The
        # current association/position proofs must not be looked up there.
        self.assertNotIn("current_head_candidate_association", metadata)
        self.assertNotIn("head_position_evidence", metadata)
        payload = self.payload("backside")
        self.assertIsNotNone(payload, metadata)
        observation = validated_backside_axis_observation(payload)
        self.assertEqual(observation.target_reconciliation["entries"][-1]["image_stamp_sec"], 101.2)
        self.assertEqual(observation.head_position_evidence["head_bounds"],
                         json.loads(json.dumps(asdict(self.proof))))
        center = observation.validated_target_center
        self.assertEqual(center["policy"], "reconciled_metric_head_position_engineering_bound")
        self.assertGreater(math.dist((center["x_m"], center["y_m"]), (.6, 0.)), .03)
        self.assertAlmostEqual(observation.bounded_orientation["half_width_rad"], self.proof.half_width_rad)
        self.assertEqual(payload["stand_center"], {"x_m": .6, "y_m": 0.})

    def test_node_passes_current_position_to_bounded_preparation(self):
        # Inject an already tested current detector/association result, then
        # execute the node's actual metadata assembly and preparation call.
        for stamp in (99.6, 99.8, 100.):
            self.frame(stamp, face="backside", reconcile=True, publish=False)
        pending = self.adapter._pending_bounded_head
        current = replace(self.current, debug=pending.debug, qr_observations=(),
                          frame=numpy.zeros((160, 160, 3), dtype=numpy.uint8))
        options = association_fixtures.CurrentHeadAssociationTests().options()
        options.update(estimate=current.estimate, debug=current.debug, attempt=current.attempt,
            profile_sha256=self.profile.sha256, now_sec=100.1,
            scan=replace(options["scan"], scan_stamp_sec=100., receipt_sec=100.),
            target_reconciliation=pending.target_reconciliation)
        association = associate_current_measured_head(**options)
        selection = register_current_tracked_head(tracked_head_selection(current),
            association=association, observed_at_sec=100., now_sec=100.1,
            max_age_sec=.5, expected_model_sha256=self.profile.sha256)
        adapter = self.fixture.make_adapter()
        adapter.stand_model_profile.sha256 = self.profile.sha256
        adapter.args.stand_model_profile = Path(pending.head_position_evidence["model_path"])
        adapter.args.candidate_crop_snapshot = self.snapshot_path
        adapter._target_reconciliation = Mock(metadata={"ready": True})
        adapter._target_reconciliation.observe.return_value = pending.target_reconciliation
        adapter.backside_proposal_reuse.select = Mock(return_value=selection)
        module = "scripts.aufgabe04.real_robot.observer.node."
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        # Stop at this boundary: downstream receipt/route consumers are covered
        # separately with their real validators and recorded sensor evidence.
        with patch(module+"camera_info_mismatches", return_value=()), \
             patch(module+"transform_mismatches", return_value=()), \
             patch(module+"compressed_msg_to_bgr_frame", return_value=frame), \
             patch(module+"_rectify_bgr_frame", side_effect=lambda value, *a, **k: value), \
             patch(module+"probe_identity_after_head_miss", side_effect=lambda value, **k: value), \
             patch(module+"associate_current_measured_head", return_value=association), \
             patch(module+"prepare_bounded_head", side_effect=StopIteration) as prepare:
            with self.assertRaises(StopIteration):
                adapter._process_latest()
        fields = prepare.call_args.kwargs
        self.assertNotIn("head_position_evidence", fields["metadata"])
        self.assertNotIn("current_head_candidate_association", fields["metadata"])
        self.assertEqual(fields["head_position_evidence"]["head_bounds"], asdict(self.proof))
        prepared = prepare_bounded_head(**fields)
        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.target_reconciliation, pending.target_reconciliation)
        self.assertEqual(prepared.head_position_evidence, fields["head_position_evidence"])

    def test_bounded_receipt_does_not_reuse_previous_frames_center(self):
        for index in range(6):
            self.frame(100.+index*.2, face="backside", reconcile=True)
        self.frame(101.2, face="backside", reconcile=False)
        payload = self.payload("backside")
        self.assertIsNotNone(payload)
        self.assertNotIn("target_reconciliation", payload)
        self.assertNotIn("head_position_evidence", payload)
        self.assertIsNone(validated_backside_axis_observation(payload).validated_target_center)

    def test_bounded_receipt_rejects_reconciliation_for_another_candidate(self):
        for index in range(7):
            self.frame(100.+index*.2, face="backside", reconcile=True, publish=False)
        current, bounds, update = self.adapter._bounded_head_ready
        forged = {**current.target_reconciliation, "candidate_uid": "another_candidate"}
        self.adapter._bounded_head_ready = (replace(current, target_reconciliation=forged), bounds, update)
        self.assertIsNone(commit_bounded_head(self.adapter))
        self.assertIsNone(self.payload("backside"))
        self.assertIn("bounded_orientation_rejection", current.metadata)

    def test_no_complete_associated_current_head_or_fresh_sources_cannot_commit(self):
        for face in ("front", "backside"):
            for changes in ({"associated": False}, {"complete": False}, {"age": .501},
                            {"scan_stamp": 99.}):
                with self.subTest(face=face, changes=changes):
                    self.setUp()
                    for index in range(7):
                        self.frame(100.+index*.2, face=face, **changes)
                    self.assertFalse(self.adapter.completed)
                    self.assertIsNone(self.payload(face))

    def test_backside_current_or_historical_marker_and_absent_appearance_veto_commit(self):
        for changes in ({"marker": True}, {"marker_seen": True}, {"no_appearance": True}):
            with self.subTest(changes=changes):
                self.setUp()
                for index in range(7):
                    self.frame(100.+index*.2, face="backside", **changes)
                self.assertFalse(self.adapter.completed)
                self.assertIsNone(self.payload("backside"))

    def test_front_current_marker_and_bound_qr_are_required_even_after_latch(self):
        for index in range(6):
            self.frame(100.+index*.2)
        update, _ = self.frame(101.2, marker=False)
        self.assertEqual(update.resolved_qr_id, "QR_003")
        self.assertFalse(self.adapter.completed)
        self.assertIsNone(self.payload("front"))
        self.frame(101.4)
        self.assertTrue(self.adapter.completed)

    def test_conflicting_current_qr_poisons_and_cannot_reuse_prior_six_bounds(self):
        for index in range(6):
            self.frame(100.+index*.2)
        update, _ = self.frame(101.2, qr_conflict=True)
        self.assertTrue(update.snapshot.poisoned)
        for index in range(7):
            self.frame(101.4+index*.2)
        self.assertFalse(self.adapter.completed)
        self.assertIsNone(self.payload("front"))

    def test_scan_proof_cannot_be_borrowed_for_another_current_scan(self):
        for index in range(7):
            _, metadata = self.frame(100.+index*.2, scan_for_proof=100.+index*.2-.05)
            self.assertEqual(metadata.get("bounded_orientation_rejection"),
                             "current_head_scan_or_bounds_binding_mismatch")
        self.assertFalse(self.adapter.completed)
        self.assertIsNone(self.payload("front"))

    def test_duplicate_frames_and_motion_cannot_complete_using_six_old_samples(self):
        for index in range(6):
            self.frame(100.+index*.2)
        self.frame(101.)
        self.assertFalse(self.adapter.completed)
        update, _ = self.frame(101.2, pose=Pose2D(.03, 0., 0.))
        self.assertTrue(update.motion_epoch_reset)
        self.assertFalse(self.adapter.completed)
        self.assertEqual(self.adapter._bounded_head_window.metadata["sample_count"], 1)

    def test_seventh_frame_expiring_before_publication_cannot_publish_receipt(self):
        for index in range(6):
            self.frame(100.+index*.2)
        self.frame(101.2, publish=False)
        self.assertIsNotNone(self.adapter._bounded_head_ready)
        self.fixture.clock_sec = 101.701
        self.assertIsNone(commit_bounded_head(self.adapter))
        self.assertFalse(self.adapter.completed)
        self.assertIsNone(self.payload("front"))
        self.assertFalse(self.adapter._last_camera_publication_freshness["accepted"])

    def test_unusable_bounded_endpoint_exits_through_bounded_recovery_and_small_view_hint(self):
        self.adapter.args.target_distance_m = .04
        self.adapter.args.inspection_observation_json = self.root / "inspection.json"
        details = {
            "reason": "collecting_bounded_orientation", "qr_texts": ["QR_003"],
            "stand_axis_debug": {
                "estimator_source": "model_current_measured_head", "estimator_usable": False,
                "decoded_qr_target_binding": {"accepted": True,
                    "reason": "decoded_qr_target_associated"},
                "metric_model": {"qr_marker_verified": True},
                "bounded_head_view_hint": {
                    "camera_relative_yaw_rad": self.proof.center_rad,
                    "orientation_half_width_rad": self.proof.half_width_rad,
                    "purpose": "orientation_disambiguation", "candidate_associated": True,
                    "source_fresh": True,
                },
            },
        }
        for index in range(61):
            stamp = 100. + index * .5
            _, metadata = self.frame(stamp, publish=False)
            with patch("scripts.aufgabe04.real_robot.observer.node.time.monotonic", return_value=stamp):
                PassiveRealViewpointNode._write_status(self.adapter, "collecting_consensus", **details)
            status = json.loads(self.adapter.args.status_json.read_text())
            if index >= 6:
                self.assertIn("insufficient stand range", metadata["bounded_orientation_rejection"])
                self.assertNotEqual(status["state"], "collecting_consensus")
            self.assertIsNone(self.payload("front"))
            self.assertIsNone(self.payload("backside"))
            if index < 60:
                self.assertFalse(self.adapter.completed)
                self.assertFalse(self.adapter.args.inspection_observation_json.exists())
        observation = load_candidate_inspection_observation(self.adapter.args.inspection_observation_json)
        self.assertEqual(status["state"], "inspection_progress_committed")
        self.assertEqual(status["preceding_state"], "evidence_not_committable")
        self.assertEqual(observation["classification"], "front_readable")
        self.assertIn("bounded_head_orientation_disambiguation", observation["reasons"])
        self.assertFalse(observation["completion_authorized"])
        self.assertFalse(observation["motion_authorized"])
        self.assertTrue(observation["front_view_recovery"]["budget_exhausted"])
        self.assertEqual(observation["front_view_recovery"]["deadline_monotonic_sec"], 130.)
        options = candidate_view_options(0., classification=observation["classification"],
            achieved_normals=[0.], attempted_normals=[],
            advisory_yaw_rad=observation["camera_relative_yaw_rad"])
        self.assertAlmostEqual(options[0], math.radians(20.))
        self.assertAlmostEqual(options[1], math.radians(-20.))

    def test_completed_stronger_path_cannot_be_overwritten_by_pending_ready_bounds(self):
        for index in range(7):
            self.frame(100.+index*.2, publish=False)
        self.assertIsNotNone(self.adapter._bounded_head_ready)
        self.adapter.completed = True
        with patch.object(self.adapter, "_commit_sensor_artifact") as write:
            self.assertIsNone(commit_bounded_head(self.adapter))
        write.assert_not_called()
        self.assertIsNone(self.adapter._bounded_head_ready)
        self.assertIsNone(self.payload("front"))


if __name__ == "__main__":
    unittest.main()
