"""Opposite-side text identity retains certified angle without a front fit."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import validate_qr_verified_observation_pose, HASH_FIELD
from scripts.aufgabe04.artifacts.retained_backside_orientation import (
    load_opposite_identity_context, orientation_record, opposite_view_matches,
)
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import write_backside_axis_frame_projection
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, bind_crop_text, POLICY
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload, write_candidate_frame_projection_fixture
from tests.aufgabe04 import test_qr_observation_pose as qr_fixture
from tests.aufgabe04.test_bounded_orientation_planning import bounds


def retained_fixture(root, *, yaw=0.):
    source = root/'source.json'
    digest, x, y = write_candidate_frame_projection_fixture(source, candidate_uid='candidate',
        canonical_x_m=.6, canonical_y_m=0., transform_x_m=0., transform_y_m=0., transform_yaw_rad=yaw)
    axis = backside_axis_payload(stand_id='candidate', stand_x_m=x, stand_y_m=y,
        robot_x_m=x+.7*math.cos(yaw), robot_y_m=y+.7*math.sin(yaw), stand_axis_rad=math.pi/2+yaw)
    axis['bounded_orientation'] = bounds(math.pi/2+yaw)
    axis.update(stand_model_profile_sha256='a'*64, robot_profile_sha256='b'*64, calibration_profile_sha256='b'*64)
    raw = root/'axis.json';raw.write_text(json.dumps(axis))
    path = root/'orientation.json'
    write_backside_axis_frame_projection(path, axis_evidence_path=raw,
        source_candidate_projection_path=source, source_candidate_projection_sha256=digest,
        target_candidate_projection_path=source, target_candidate_projection_sha256=digest,
        target_candidate_x_m=x, target_candidate_y_m=y)
    snapshot = Path(json.loads(source.read_text())['projected_candidate_snapshot_path'])
    return path, snapshot


def crop_evidence(scan_frame='base_scan'):
    cluster = dict(associated=True, eligible_cluster_count=1, scan_stamp_sec=100.,
        scan_frame_id=scan_frame, selected_cluster_source_indices=[0, 1, 2])
    return dict(policy=POLICY, accepted=True, candidate_uid='candidate', image_stamp_sec=100.,
        scan_stamp_sec=100., bounds_xyxy=[300, 200, 500, 400], target_center_px=[400., 300.],
        competitors=[], search=dict(accepted=True, envelope=cluster))


class OppositeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.base = qr_fixture.QrObservationPoseTests();self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.adapter = self.base.adapter
        self.path, self.snapshot = retained_fixture(self.base.root)
        self.context = load_opposite_identity_context(self.path, self.snapshot, candidate_uid='candidate',
            planning_frame='map', stand_center=dict(x_m=.6, y_m=0.), model_sha256='a'*64)
        self.adapter._opposite_identity_context = self.context

    def run_image(self, observations=None, *, late=False):
        adapter = self.adapter
        adapter.profile.scan_frame = 'scan'
        crop = crop_evidence('scan')
        attempt = HeadRoiAttempt(ImageRoi(300, 200, 500, 400, 100.), POLICY, 1.6, 400., 300., 100.)
        observations = (DecodedQrObservation('Start', None, 'payload_only', 1.),) if observations is None else observations
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        adapter._write_status = lambda state, **details: PassiveRealViewpointNode._write_status(adapter, state, **details)
        def decode(image, *_args, **kwargs):
            self.assertEqual(image.shape, (200, 200, 3))
            if late:
                self.base.fixture.clock_sec = 100.6
            return observations
        with ExitStack() as stack:
            module = 'scripts.aufgabe04.real_robot.observer.node.'
            for name in ('camera_info_mismatches', 'transform_mismatches'):
                stack.enter_context(patch(module+name, return_value=()))
            stack.enter_context(patch(module+'compressed_msg_to_bgr_frame', return_value=frame))
            stack.enter_context(patch(module+'_rectify_bgr_frame', return_value=frame))
            fit = stack.enter_context(patch(module+'estimate_stand_axis_from_metric_model', side_effect=AssertionError('front fit requested')))
            viewer = stack.enter_context(patch(module+'evaluate_viewer_head', side_effect=AssertionError('front search requested')))
            stack.enter_context(patch('scripts.aufgabe04.real_robot.observer.opposite_identity.current_scan_qr_search', return_value=(attempt, crop['search'])))
            stack.enter_context(patch('scripts.aufgabe04.real_robot.observer.opposite_identity.detect_opposite_target_support', return_value=None))
            stack.enter_context(patch('scripts.aufgabe04.real_robot.observer.opposite_identity.exclusive_identity_crop', return_value=(attempt, crop)))
            stack.enter_context(patch('scripts.aufgabe04.real_robot.observer.opposite_identity.detect_qr_observations_bgr', side_effect=decode))
            adapter._process_latest()
            fit.assert_not_called();viewer.assert_not_called()
        return self.base.result()

    def test_first_cornerless_decode_commits_and_retains_angle(self):
        payload = self.run_image()
        self.assertIsNotNone(payload)
        self.assertEqual(payload['qr_id'], 'Start')
        self.assertIsNone(payload['qr_corners_px'])
        self.assertAlmostEqual(payload['stand_axis_rad'], self.context.orientation['stand_axis_rad'])
        self.assertEqual(payload['retained_backside_orientation']['axis_sample_count'], 7)
        self.assertFalse(payload['facing_ready'])
        self.assertTrue(self.adapter.completed)
        self.assertEqual(self.adapter.observation_evidence.snapshot().current_axis_sample_count, 0)
        self.assertEqual(self.adapter._camera_pipeline_counters['processed_images'], 1)
        self.assertEqual(json.loads(self.adapter.args.status_json.read_text())['state'], 'qr_observation_pose_committed')

    def test_discovery_catalog_preserves_retained_axis(self):
        from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
        from scripts.aufgabe04.real_robot.candidate.qr_pose_discovery import bind_qr_pose_discovery
        self.adapter.args.stream_id = 'session_candidate'
        payload = self.run_image()
        frame = SimpleNamespace(candidate=self.context.snapshot.candidate_for('candidate'),
            decision_binding=None, planning_frame=None, config=SimpleNamespace(planning_frame='map'))
        config = SimpleNamespace(session_id='session', snapshot=self.context.snapshot,
            robot_profile_sha256='b'*64, calibration_profile_sha256='b'*64)
        record = bind_qr_pose_discovery(observation=CandidateObservation(None, 'Start', None,
            qr_observation_pose_path=self.adapter.args.qr_observation_pose_json),
            observation_frame=frame, source_config=config, source_registry=None,
            source_registry_sha256=None)
        self.assertEqual(record['retained_backside_orientation'], payload['retained_backside_orientation'])
        self.assertEqual(record['stand_axis_rad'], payload['stand_axis_rad'])
        self.assertFalse(record['facing_ready'])
        self.assertTrue(record['candidate_geometry_unchanged'])

    def test_late_payload_does_not_commit(self):
        self.assertIsNone(self.run_image(late=True))
        self.assertFalse(self.adapter.completed)

    def test_conflicting_current_payloads_poison_epoch(self):
        values = tuple(DecodedQrObservation(t, None, 'test', 1.) for t in ('Start', 'Other'))
        self.assertIsNone(self.run_image(values))
        self.assertTrue(self.adapter.observation_evidence.snapshot().poisoned)

    def test_receipt_rejects_retained_angle_tamper_even_with_new_outer_hash(self):
        payload = self.run_image()
        payload['retained_backside_orientation']['stand_axis_rad'] += .1
        payload['stand_axis_rad'] += .1
        payload.pop(HASH_FIELD)
        with self.assertRaisesRegex(ValueError, 'certified source'):
            validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))

    def test_receipt_rejects_neighbor_overlap(self):
        payload = self.run_image()
        payload['qr_binding']['current_head_binding']['competitors'] = [dict(candidate_uid='other', bounds_xyxy=[350, 250, 450, 350])]
        payload.pop(HASH_FIELD)
        with self.assertRaisesRegex(ValueError, 'neighboring'):
            validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))

    def test_reprojected_orientation_keeps_original_uncertainty(self):
        target = self.base.root / 'rotated_target.json'
        digest, x, y = write_candidate_frame_projection_fixture(target, candidate_uid='candidate',
            canonical_x_m=.6, canonical_y_m=0., transform_x_m=.1, transform_y_m=.2, transform_yaw_rad=.4)
        from scripts.aufgabe04.real_robot.candidate.approach import _CandidateObservationFrame
        from scripts.aufgabe04.real_robot.candidate.retained_orientation import retain_orientation_after_arrival
        source = _CandidateObservationFrame(None,None,None,None,retained_backside_axis_path=self.path)
        arrival = _CandidateObservationFrame(None,SimpleNamespace(geometry=SimpleNamespace(x_m=x,y_m=y)),
            None,SimpleNamespace(projection_path=target,projection_sha256=digest))
        arrived = retain_orientation_after_arrival(source,arrival,self.base.root/'new_arrival')
        record = orientation_record(arrived.retained_backside_axis_path)
        self.assertAlmostEqual(math.remainder(record['stand_axis_rad'] - math.pi/2 - .4, math.pi), 0.)
        self.assertEqual(record['bounded_orientation']['half_width_rad'], bounds()['half_width_rad'])
        self.assertEqual(record['axis_sample_count'], 7)

    def test_receipt_rejects_robot_on_original_backside(self):
        payload = self.run_image()
        payload['robot_pose']['x_m'] = 1.3
        payload.pop(HASH_FIELD)
        with self.assertRaisesRegex(ValueError, 'opposite side'):
            validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))

    def test_context_rejects_wrong_candidate_profile_and_snapshot(self):
        options=dict(candidate_uid='candidate', planning_frame='map', stand_center=dict(x_m=.6,y_m=0.), model_sha256='a'*64)
        for changes in (dict(candidate_uid='other'), dict(model_sha256='c'*64), dict(stand_center=dict(x_m=.61,y_m=0.))):
            with self.assertRaises(ValueError):
                load_opposite_identity_context(self.path, self.snapshot, **{**options, **changes})
        with self.assertRaises(ValueError):
            load_opposite_identity_context(self.path, self.base.root/'other.json', **options)
        self.assertTrue(opposite_view_matches(self.context.orientation, Pose2D(0., 0., 0.)))
        self.assertFalse(opposite_view_matches(self.context.orientation, Pose2D(1.3, 0., 0.)))


class ExclusiveCropTests(unittest.TestCase):
    def test_neighbor_volume_is_excluded_and_overlapping_center_rejected(self):
        intr = CameraIntrinsics(800, 600, 400., 400., 400., 300.)
        model = SimpleNamespace(head_width_m=.1, head_height_m=.1, head_center_height_m=.2)
        attempt = HeadRoiAttempt(ImageRoi(280,180,520,420,100.), 'search', 2.4,400.,300.,100.)
        search = dict(accepted=True, envelope=crop_evidence()['search']['envelope'])
        neighbor = SimpleNamespace(candidate_uid='other', geometry=SimpleNamespace(x_m=.2,y_m=0.,uncertainty_m=0.))
        # Test projection is injected independently of search; no image fitting.
        opts=dict(candidate_uid='candidate', snapshot=SimpleNamespace(snapshot_id='test',candidates=(neighbor,)),
            camera_from_map=RigidTransform('camera','map',(0.,0.,1.),(0.,0.,0.,1.)),
            intrinsics=intr,model_profile=model,image_stamp_sec=100.,scan=SimpleNamespace(scan_stamp_sec=100.))
        with patch('scripts.aufgabe04.real_robot.observer.opposite_identity_crop.current_scan_qr_search', return_value=(attempt,search)):
            result, meta = exclusive_identity_crop(**opts)
            self.assertIsNone(result)
            self.assertEqual(meta['reason'], 'target_crop_overlap_unresolved')
            neighbor.geometry.x_m=0.
            result, meta = exclusive_identity_crop(**opts)
            self.assertIsNone(result)
            self.assertEqual(meta['reason'],'target_crop_overlap_unresolved')

    def test_crop_search_failure_never_binds_text(self):
        with patch('scripts.aufgabe04.real_robot.observer.opposite_identity_crop.current_scan_qr_search',
                   return_value=(None,dict(accepted=False,reason='stale_or_unsynchronized_search'))):
            result, meta=exclusive_identity_crop(candidate_uid='x',snapshot=None,camera_from_map=None,
                intrinsics=None,model_profile=None,image_stamp_sec=100.)
        self.assertIsNone(result)
        self.assertFalse(bind_crop_text((DecodedQrObservation('Start',None,'test',1.),),meta).accepted)
