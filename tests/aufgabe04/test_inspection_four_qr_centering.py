"""Recorded final inspection: complete QR frames a stand with a clipped head."""
from dataclasses import fields, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.observer.candidate_centering import (
    build_camera_centering_advisory, validate_camera_centering_advisory,
)
from scripts.aufgabe04.real_robot.observer.candidate_centering_receipt import prepare_candidate_centering
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
from scripts.aufgabe04.real_robot.observer.inspection_framing import review_centering_destination
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import QrObservationFrame, QrObservationPoseFallback
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from scripts.aufgabe04.real_robot.observer.qr_target_support import prepare_qr_target_support, POLICY
from tests.aufgabe04 import test_camera_observer_processing as processing

ROOT = Path(__file__).parent / 'fixtures/inspection_four_qr_20260930'


def recorded_current():
    payload = json.loads((ROOT / 'qr_observation_pose.json').read_text())
    context = json.loads((ROOT / 'capture_context.json').read_text())
    for name, digest in context['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    proof = payload['qr_binding']['target_reconciliation']
    proof['snapshot_path'] = str(ROOT / 'candidate_snapshot.json')
    for entry in proof['entries']:
        entry['position_epoch']['path'] = str(ROOT / 'candidate_frame_projection.json')
    binding = {f.name: payload['qr_binding'][f.name] for f in fields(QrTargetBinding)}
    binding['qr_texts_for_evidence'] = tuple(binding['qr_texts_for_evidence'])
    current = QrObservationFrame(payload['sensor_stamp_sec'], payload['scan_stamp_sec'],
        Pose2D(**payload['robot_pose']), payload['target_key'], tuple(payload['camera_signature']),
        tuple(payload['image_shape']), QrTargetBinding(**binding),
        tuple(tuple(p) for p in payload['qr_corners_px']), (payload['qr_id'],),
        payload['stand_model_profile_sha256'], {})
    def transform(parent, child):
        raw = next(t for t in context['tf_samples']
                   if t['target_frame'] == parent and t['source_frame'] == child)
        return RigidTransform(parent, child, tuple(raw['translation_xyz_m']), tuple(raw['rotation_xyzw']))
    odom = transform('odom', 'base_footprint')
    qx, qy, qz, qw = odom.rotation_xyzw
    odom_pose = Pose2D(*odom.translation_xyz_m[:2], math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    geometry = dict(intrinsics=CameraIntrinsics(**binding['finite_bearing']['intrinsics']),
        scan_from_camera=transform('base_scan', 'camera'),
        base_from_camera=transform('base_footprint', 'camera'), odom_pose=odom_pose)
    return current, payload, geometry


def advisory(current, payload, geometry):
    support = prepare_qr_target_support(current)
    assert support is not None
    return build_camera_centering_advisory(association=support,
        **{k: v for k, v in geometry.items() if k != 'odom_pose'},
        candidate_uid=payload['candidate_uid'], target_key=current.target_key,
        stream_id=payload['stream_id'], planning_frame='map', motion_epoch=0,
        anchor_pose=EvidencePose(**payload['robot_pose']),
        anchor_odom_pose=EvidencePose(geometry['odom_pose'].x_m, geometry['odom_pose'].y_m,
                                     geometry['odom_pose'].yaw_rad),
        odom_stamp_sec=current.stamp_sec, image_stamp_sec=current.stamp_sec,
        now_sec=payload['checked_at_sec'],
        **{k: payload[k] for k in ('robot_profile_sha256', 'calibration_profile_sha256',
                                  'stand_model_profile_sha256')})


def test_recorded_qr_yields_bounded_right_turn_without_head_or_angle():
    current, payload, geometry = recorded_current()
    advice = advisory(current, payload, geometry)
    assert advice is not None
    assert advice.measured_center_px == pytest.approx((726.1566391, 269.7903900))
    assert math.degrees(advice.required_yaw_rad) == pytest.approx(-24.78287644)
    assert math.degrees(advice.requested_yaw_rad) == pytest.approx(-17.78287644)
    assert advice.arrival_recovery
    search = prepare_qr_target_support(current).lidar_association.search_association
    assert review_centering_destination(advice, search_association=search).allowed
    assert not review_centering_destination(replace(advice, requested_yaw_rad=advice.required_yaw_rad),
                                           search_association=search).allowed
    metadata = json.loads(json.dumps(advice.metadata()))
    assert validate_camera_centering_advisory(metadata).arrival_recovery
    assert metadata['association_source'] == POLICY
    assert metadata['target_support']['supplies_angle'] is False
    assert metadata['target_support']['supplies_identity'] is False
    assert metadata['motion_authorized'] is False
    assert metadata['completion_authorized'] is False
    with pytest.raises(ValueError, match='did not advance'):
        validate_camera_centering_advisory(metadata, min_image_stamp_sec=current.stamp_sec)
    with pytest.raises(ValueError, match='did not advance'):
        validate_camera_centering_advisory(metadata, min_scan_stamp_sec=current.scan_stamp_sec)


def test_recorded_centering_precedes_qr_only_completion(tmp_path, monkeypatch):
    current, payload, geometry = recorded_current()
    fixture = processing.CameraObserverProcessingTest()
    adapter = fixture.make_adapter()
    fixture.clock_sec = payload['checked_at_sec']
    adapter.args.stand_id = payload['candidate_uid']
    adapter.args.stream_id = payload['stream_id']
    adapter.args.stand_x, adapter.args.stand_y = payload['stand_center'].values()
    assert adapter._target_evidence_key() == current.target_key
    for key in ('candidate_centering', 'qr_observation_pose', 'status', 'recommended_pose'):
        setattr(adapter.args, key + '_json', tmp_path / (key + '.json'))
    adapter.profile.base_frame = 'base_footprint'
    adapter.profile.scan_frame = 'base_scan'
    adapter.profile.odom_frame = 'odom'
    adapter.stand_model_profile.sha256 = payload['stand_model_profile_sha256']
    adapter.last_pose = current.robot_pose
    for module in ('candidate_centering_receipt', 'qr_observation_pose'):
        prefix = 'scripts.aufgabe04.real_robot.observer.' + module + '.'
        monkeypatch.setattr(prefix + 'real_robot_profile_sha256', lambda _: payload['robot_profile_sha256'])
        monkeypatch.setattr(prefix + 'camera_calibration_sha256', lambda _: payload['calibration_profile_sha256'])
    adapter._pending_qr_observation_pose = current
    args = dict(crop=SimpleNamespace(accepted=False), association=SimpleNamespace(accepted=False),
        image_stamp_sec=current.stamp_sec, scan_stamp_sec=current.scan_stamp_sec,
        target_key=current.target_key, robot_pose=current.robot_pose, metadata=current.metadata,
        qr_observation=current, **geometry)
    assert prepare_candidate_centering(**{**args, 'odom_pose': None}) is None
    assert prepare_candidate_centering(**{**args, 'target_key': 'other-candidate'}) is None
    adapter._pending_candidate_centering = prepare_candidate_centering(**args)
    assert adapter._pending_candidate_centering is not None
    update = adapter._record_observation_frame(robot_pose=current.robot_pose,
        image_stamp_sec=current.stamp_sec, scan_stamp_sec=current.scan_stamp_sec,
        observed_at_sec=payload['checked_at_sec'], lidar_associated=True,
        axis_yaw_rad=None, axis_source=None, qr_texts=(payload['qr_id'],), qr_symbol_count=1)
    assert update.frame_accepted and update.qr_sample_accepted
    assert not update.axis_sample_accepted
    assert update.snapshot.current_axis_sample_count == 0
    assert adapter._candidate_centering_ready is not None
    # The QR would otherwise be immediately eligible for discovery. Centering
    # deliberately suppresses that pre-turn receipt while preserving identity.
    assert QrObservationPoseFallback().observe(current, update=update,
        observed_at_sec=payload['checked_at_sec'], now_monotonic_sec=1.) is not None
    assert adapter._qr_observation_pose_ready is None
    PassiveRealViewpointNode._write_status(adapter, 'metric_model_measurement_unavailable')
    committed = json.loads(adapter.args.candidate_centering_json.read_text())
    assert validate_camera_centering_advisory(committed).arrival_recovery
    assert not adapter.args.qr_observation_pose_json.exists()
    assert not adapter.args.recommended_pose_json.exists()
    status = json.loads(adapter.args.status_json.read_text())
    assert status['state'] == 'candidate_centering_committed'
    assert status['camera_centering']['reason'] == 'fresh_reconciled_qr_off_center'


@pytest.mark.parametrize('field', ['intrinsics', 'scan_from_camera', 'target_reconciliation',
                                 'candidate_uid', 'target_key', 'motion_epoch', 'anchor_pose'])
def test_rehashed_qr_advisory_cannot_change_bound_geometry_or_candidate(field):
    current, payload, geometry = recorded_current()
    advice = advisory(current, payload, geometry)
    if field == 'intrinsics':
        value = replace(advice.intrinsics, fx_px=advice.intrinsics.fx_px + 1.)
    elif field == 'scan_from_camera':
        value = replace(advice.scan_from_camera, translation_xyz_m=(0., 0., 0.))
    elif field == 'target_reconciliation':
        value = None
    elif field in ('candidate_uid', 'target_key'):
        value = 'other-candidate'
    elif field == 'motion_epoch':
        value = 1
    else:
        value = replace(advice.anchor_pose, x_m=advice.anchor_pose.x_m + .01)
    with pytest.raises(ValueError):
        validate_camera_centering_advisory(replace(advice, **{field: value}).metadata())
