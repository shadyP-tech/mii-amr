"""ID-only ordinary acquisition preserves current head, scan and crop gates.

Recorded pixels provide real border and LiDAR support. Payload-only decoder
results are controlled here; this exercises association, not decoder accuracy.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import HASH_FIELD, validate_qr_verified_observation_pose
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.current_head_identity import (
    acquire_current_head_identity, validate_current_head_identity_binding, validate_identity_startup,
)
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.opposite_head_support import detect_opposite_head_region, support_opposite_head_region
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose, commit_qr_observation_pose
from tests.aufgabe04 import test_camera_observer_processing as observer_fixtures
from tests.aufgabe04.opposite_head_identity_fixture import recorded_head_identity, recorded_head_image


def test_measured_physical_profile_requires_explicit_snapshot_before_ros_startup(monkeypatch):
    model = SimpleNamespace(committable=True, environment="physical")
    with pytest.raises(ValueError, match="--candidate-crop-snapshot is required"):
        validate_identity_startup(model, candidate_crop_snapshot=None)
    monkeypatch.setattr('scripts.aufgabe04.real_robot.observer.node.load_measured_physical_stand_model', lambda _: model)
    # __init__ must reject before importing/creating any ROS node.
    with pytest.raises(ValueError, match="--candidate-crop-snapshot is required"):
        PassiveRealViewpointNode(SimpleNamespace(stand_model_profile='unused', candidate_crop_snapshot=None))


@pytest.mark.parametrize('environment,committable,snapshot', (
    ('physical', True, 'admitted_candidates.json'), ('simulation', True, None),
    ('real', True, None), ('physical', False, None),
))
def test_snapshot_startup_configuration_preserves_nonphysical_legacy(environment, committable, snapshot):
    validate_identity_startup(SimpleNamespace(environment=environment, committable=committable),
        candidate_crop_snapshot=snapshot)


def test_candidate_snapshot_cli_help_documents_measured_physical_requirement():
    from scripts.aufgabe04.real_robot.observer.node import build_parser
    help_text = build_parser().format_help()
    assert 'Required for measured physical stand profiles' in help_text


def make_ordinary(tmp_path, monkeypatch):
    case = recorded_head_identity(tmp_path, 31)
    frame = recorded_head_image(case)
    row = case['row']
    search_options = dict(scan=case['scan'], scan_from_map=case['scan_from_map'],
        camera_from_map=case['camera_from_map'], intrinsics=case['intrinsics'], model_profile=case['model'],
        image_stamp_sec=row['image_stamp_sec'], now_sec=case['now_sec'], max_scan_age_sec=.5,
        sync_tolerance_sec=.1, target_reconciliation=case['reconciliation'], **row['options'])
    attempt, _ = current_scan_qr_search(**search_options)
    region = detect_opposite_head_region(frame, cv2, attempt=attempt, model_profile=case['model'],
        image_stamp_sec=row['image_stamp_sec'], now_sec=case['now_sec'], max_scan_age_sec=.5, max_elapsed_sec=.3)
    support_options = {key: value for key, value in search_options.items()
        if key not in ('scan_from_map', 'camera_from_map', 'sync_tolerance_sec')}
    support = support_opposite_head_region(region, attempt=attempt, image_shape=frame.shape[:2],
        scan_from_camera=case['scan_from_camera'], **support_options)
    assert support is not None
    fixture = observer_fixtures.CameraObserverProcessingTest()
    adapter = fixture.make_adapter()
    fixture.clock_sec = case['now_sec']
    adapter.cv2 = cv2
    adapter.calibration = case['calibration_profile']
    adapter.stand_model_profile = case['model']
    adapter.args.stand_id = row['candidate_uid']
    adapter.args.stream_id = row['target_key'].split(':'+row['candidate_uid']+':', 1)[0]
    adapter.args.stand_x, adapter.args.stand_y = row['stand_center']
    adapter.args.candidate_crop_snapshot = case['snapshot_path']
    adapter.args.stand_model_profile = case['model_path']
    adapter.args.qr_observation_pose_json = tmp_path/'qr.json'
    adapter.args.qr_pose_fallback_delay_sec = 0.
    adapter.args.recommended_pose_json = tmp_path/'recommendation.json'
    adapter.args.status_json = tmp_path/'status.json'
    adapter.args.candidate_centering_json = None
    adapter.profile.map_frame, adapter.profile.base_frame = 'map', 'base_footprint'
    adapter.profile.scan_frame, adapter.profile.camera_optical_frame = 'base_scan', 'camera'
    digest = case['orientation']['robot_profile_sha256']
    monkeypatch.setattr('scripts.aufgabe04.real_robot.configuration.profile.real_robot_profile_sha256', lambda _: digest)
    monkeypatch.setattr('scripts.aufgabe04.real_robot.observer.qr_observation_pose.real_robot_profile_sha256', lambda _: digest)
    decoder = Mock()

    def decoded(*args, **kwargs):
        kwargs['diagnostics'].update(events=[{'stage': 'wechat', 'symbol_count': 1}])
        return (DecodedQrObservation('Start', None, 'payload_test', 1.),)

    decoder.side_effect = decoded
    monkeypatch.setattr('scripts.aufgabe04.qr_scanning.opencv_qr_detector.detect_qr_observations_bgr', decoder)
    kwargs = dict(frame=frame, estimate=SimpleNamespace(corners=tuple(ImagePoint(*p) for p in region.corners_px)),
        association=SimpleNamespace(accepted=True, lidar_association=support.lidar_association),
        crop_review=SimpleNamespace(accepted=True), selected_roi=ImageRoi(0, 0, 800, 600, 100.),
        intrinsics=case['intrinsics'], scan=case['scan'], scan_from_camera=case['scan_from_camera'],
        scan_from_map=case['scan_from_map'], camera_from_map=case['camera_from_map'],
        map_bearing_rad=row['options']['map_bearing_rad'], accepted_range_m=row['options']['accepted_range_m'],
        image_stamp_sec=row['image_stamp_sec'], robot_pose=case['robot_pose'],
        camera_signature=(case['intrinsics'].fx_px, case['intrinsics'].fy_px,
            case['intrinsics'].cx_px, case['intrinsics'].cy_px), target_reconciliation=case['reconciliation'], metadata={})
    return SimpleNamespace(case=case, adapter=adapter, clock=fixture, decoder=decoder, kwargs=kwargs, support=support)


@pytest.fixture
def ordinary(tmp_path, monkeypatch):
    return make_ordinary(tmp_path, monkeypatch)


def acquire(fixture):
    return acquire_current_head_identity(fixture.adapter, **fixture.kwargs)


def publish(fixture, observations, binding):
    adapter, options = fixture.adapter, fixture.kwargs
    adapter._pending_qr_observation_pose = prepare_qr_observation_pose(qr_binding=binding,
        qr_observations=observations, observed_qr_texts=tuple(o.text for o in observations),
        image_stamp_sec=options['image_stamp_sec'], scan_stamp_sec=options['scan'].scan_stamp_sec,
        robot_pose=options['robot_pose'], target_key=adapter._target_evidence_key(),
        camera_signature=options['camera_signature'], image_shape=options['frame'].shape,
        roi=options['selected_roi'], model_profile_sha256=adapter.stand_model_profile.sha256,
        metadata=options['metadata'])
    update = adapter._record_observation_frame(robot_pose=options['robot_pose'],
        image_stamp_sec=options['image_stamp_sec'], scan_stamp_sec=options['scan'].scan_stamp_sec,
        observed_at_sec=fixture.clock.clock_sec, lidar_associated=binding.accepted,
        axis_yaw_rad=None, axis_source=None, qr_texts=binding.qr_texts_for_evidence,
        qr_symbol_count=binding.symbol_count)
    PassiveRealViewpointNode._write_status(adapter, 'collecting_consensus')
    return update


def test_current_head_payload_without_corners_crosses_real_node_and_persisted_receipt(ordinary):
    observations, binding = acquire(ordinary)
    assert binding.accepted, ordinary.kwargs['metadata']
    assert observations[0].corners is None
    assert ordinary.decoder.call_args.kwargs['identity_only'] is True
    crop = binding.current_head_binding
    assert crop['sampling'] == 'masked_current_head_region'
    assert any(c['occluded_by_target_head'] for c in crop['competitors'])
    update = publish(ordinary, observations, binding)
    assert update.qr_sample_accepted and not update.axis_consensus
    payload = validate_qr_verified_observation_pose(json.loads(ordinary.adapter.args.qr_observation_pose_json.read_text()))
    assert payload['schema_version'] == 3 and payload['qr_id'] == 'Start'
    assert payload['qr_corners_px'] is None and payload['stand_axis_rad'] is None
    assert not payload['facing_ready'] and not payload['motion_authorized']
    assert not ordinary.adapter.args.recommended_pose_json.exists()


def prepare_strict_front(ordinary, observations, binding):
    # The strict angle-quality fixture is synthetic; real recorded borders,
    # support and payload crop still exercise their production contracts.
    from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model
    from scripts.aufgabe04.real_robot.observer.immediate_front_observation import prepare_immediate_front
    from tests.aufgabe04.test_head_model_admission import head_estimate, head_debug, quality, outer_boundary
    corners = ordinary.kwargs['estimate'].corners
    sha = ordinary.case['model'].sha256
    estimate = head_estimate(yaw_deg=10., corners=corners, model_profile_sha256=sha)
    debug = head_debug(head_model_quality=quality(profile_sha256=sha),
        head_outer_recovery=outer_boundary(corners, profile_sha256=sha),
        model_profile_sha256=sha)
    admission = admit_measured_head_model(estimate=estimate, debug=debug, yaw_rad=10.*3.141592653589793/180.)
    assert admission.accepted
    associated = SimpleNamespace(accepted=True, head_admission=admission,
        lidar_association=ordinary.support.lidar_association)
    options = ordinary.kwargs
    ordinary.adapter._pending_immediate_front = prepare_immediate_front(
        estimate=estimate, debug=debug, association=associated,
        crop=SimpleNamespace(accepted=True), qr_binding=binding,
        qr_observations=observations, observed_qr_texts=('Start',),
        image_stamp_sec=options['image_stamp_sec'], scan_stamp_sec=options['scan'].scan_stamp_sec,
        robot_pose=options['robot_pose'], camera_heading_rad=0.,
        target_key=ordinary.adapter._target_evidence_key(), camera_signature=options['camera_signature'],
        image_shape=options['frame'].shape, roi=options['selected_roi'], metadata=options['metadata'])


def test_same_frame_strict_head_geometry_commits_cornerless_front_identity(ordinary):
    from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
    observations, binding = acquire(ordinary)
    prepare_strict_front(ordinary, observations, binding)
    publish(ordinary, observations, binding)
    recommendation = load_recommendation(ordinary.adapter.args.recommended_pose_json)
    assert recommendation.axis_sample_count == 1
    assert recommendation.axis_measurement['qr_corners_px'] is None
    assert recommendation.axis_measurement['qr_id'] == 'Start'
    assert not ordinary.adapter.args.qr_observation_pose_json.exists()


def test_decoder_systemic_error_does_not_claim_attempted_empty_identity(ordinary):
    def failed(*args, **kwargs):
        kwargs['diagnostics'].update(events=[{'stage': 'wechat', 'reason': 'decoder_error'}])
        return ()
    ordinary.decoder.side_effect = failed
    _, binding = acquire(ordinary)
    details = ordinary.kwargs['metadata']['current_head_identity']
    assert not binding.accepted and details['accepted_crop']
    assert details['attempted'] is False and details['decoded_texts'] == []


@pytest.mark.parametrize('missing', ('association', 'crop', 'snapshot'))
def test_no_current_head_or_exclusive_context_never_decodes(ordinary, missing):
    if missing == 'association':
        ordinary.kwargs['association'] = None
    elif missing == 'crop':
        ordinary.kwargs['crop_review'] = SimpleNamespace(accepted=False)
    else:
        ordinary.adapter.args.candidate_crop_snapshot = None
    _, binding = acquire(ordinary)
    assert not binding.accepted
    ordinary.decoder.assert_not_called()


@pytest.mark.parametrize('texts', (('Start', 'Start'), ('Start', 'QR_001')))
def test_multiple_payloads_even_same_id_remain_ambiguous(ordinary, texts):
    ordinary.decoder.side_effect = None
    ordinary.decoder.return_value = tuple(DecodedQrObservation(text, None, 'test', 1.) for text in texts)
    _, binding = acquire(ordinary)
    assert not binding.accepted and binding.symbol_count == 2


def test_decoder_expiry_cannot_admit_a_payload_or_negative_sample(ordinary):
    def delayed(*args, **kwargs):
        kwargs['diagnostics'].update(events=[{'stage': 'wechat'}])
        ordinary.clock.clock_sec = ordinary.kwargs['image_stamp_sec'] + .6
        return (DecodedQrObservation('Start', None, 'test', 1.),)
    ordinary.decoder.side_effect = delayed
    _, binding = acquire(ordinary)
    assert not binding.accepted
    assert ordinary.kwargs['metadata']['current_head_identity']['decoded_texts'] == ['Start']


@pytest.mark.parametrize('field', ('sensor_stamp_sec', 'scan_stamp_sec', 'image_shape', 'target_key', 'camera_signature'))
def test_head_identity_context_cannot_be_reused(ordinary, field):
    _, binding = acquire(ordinary)
    context = binding.current_head_binding['ordinary_context']
    value = context[field]
    change = {field: value+.001 if isinstance(value, float) else 'other' if isinstance(value, str) else [1, 2]}
    name = {'sensor_stamp_sec': 'image_stamp_sec'}.get(field, field)
    with pytest.raises(ValueError):
        validate_current_head_identity_binding(binding.metadata(), **{name: change[field]})


@pytest.mark.parametrize('tamper', ('head_border', 'scan_stamp', 'missing_support', 'neighbor_depth', 'context_target'))
def test_rehashed_persisted_receipt_rejects_head_crop_tampering(ordinary, tamper):
    observations, binding = acquire(ordinary)
    publish(ordinary, observations, binding)
    payload = json.loads(ordinary.adapter.args.qr_observation_pose_json.read_text())
    crop = payload['qr_binding']['current_head_binding']
    if tamper == 'head_border':
        crop['target_support']['corners_px'][0][0] += 20
    elif tamper == 'scan_stamp':
        crop['scan_stamp_sec'] += .001
    elif tamper == 'missing_support':
        crop['target_support'] = None
    elif tamper == 'neighbor_depth':
        next(c for c in crop['competitors'] if c['occluded_by_target_head'])['depth_interval_m'] = [.1, .2]
    else:
        crop['ordinary_context']['target_key'] += ':different'
    payload.pop(HASH_FIELD)
    with pytest.raises(ValueError):
        validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))
