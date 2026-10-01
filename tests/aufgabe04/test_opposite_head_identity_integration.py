"""Latest failed opposite pixels cross head/scan and persisted ID-only gates."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.artifacts.bounded_orientation import BoundedOrientationViewUnavailableError
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import HASH_FIELD, validate_qr_verified_observation_pose
from scripts.aufgabe04.artifacts.retained_facing import build_retained_facing
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation, recommendation_to_dict
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer import opposite_identity
from scripts.aufgabe04.real_robot.observer.opposite_head_support import HEAD_POLICY
from scripts.aufgabe04.real_robot.observer.qr_observation_binding import load_bound_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import commit_qr_observation_pose
from tests.aufgabe04 import test_camera_observer_processing as observer_fixtures
from tests.aufgabe04.opposite_head_identity_fixture import recorded_head_identity, recorded_head_image


@pytest.fixture
def latest_opposite(tmp_path, monkeypatch, request):
    index = getattr(request, 'param', 31)
    case = recorded_head_identity(tmp_path, index)
    clock = observer_fixtures.CameraObserverProcessingTest()
    adapter = clock.make_adapter()
    row, orientation = case['row'], case['orientation']
    clock.clock_sec = case['now_sec']
    adapter.cv2 = cv2
    adapter.calibration = case['calibration_profile']
    adapter.stand_model_profile = case['model']
    adapter.args.stand_id = row['candidate_uid']
    adapter.args.stream_id = row['target_key'].split(':' + row['candidate_uid'] + ':', 1)[0]
    adapter.args.stand_x, adapter.args.stand_y = row['stand_center']
    adapter.args.candidate_crop_snapshot = case['snapshot_path']
    adapter.args.stand_model_profile = case['model_path']
    adapter.args.qr_observation_pose_json = tmp_path / 'qr_observation_pose.json'
    adapter.args.recommended_pose_json = tmp_path / 'recommendation.json'
    adapter.args.status_json = tmp_path / 'observer_status.json'
    adapter.args.candidate_centering_json = None
    adapter.profile.map_frame = 'map'
    adapter.profile.base_frame = 'base_footprint'
    adapter.profile.scan_frame = 'base_scan'
    adapter.profile.camera_optical_frame = 'camera'
    assert adapter._target_evidence_key() == row['target_key']
    adapter._capture_pending = {}
    adapter._target_reconciliation = case['tracker']
    intrinsics = case['intrinsics']
    kwargs = dict(context=SimpleNamespace(orientation=orientation, snapshot=case['snapshot']),
        frame=recorded_head_image(case), intrinsics=intrinsics, robot_pose=case['robot_pose'],
        image_stamp_sec=row['image_stamp_sec'], scan=case['scan'],
        camera_signature=(intrinsics.fx_px, intrinsics.fy_px, intrinsics.cx_px, intrinsics.cy_px),
        scan_from_map=case['scan_from_map'], camera_from_map=case['camera_from_map'],
        map_bearing_rad=row['options']['map_bearing_rad'], accepted_range_m=row['options']['accepted_range_m'],
        scan_from_camera=case['scan_from_camera'], base_from_camera=case['base_from_camera'], image_stamp=None,
        target_reconciliation=case['reconciliation'], require_target_reconciliation=True)
    # The platform-neutral adapter uses the recorded robot digest. Calibration,
    # head pixels, scan association, crop, receipt and retained-facing loaders
    # remain production code. Only payload decoding is stubbed: desktop OpenCV
    # lacks the workstation WeChat backend that read Start in these images.
    monkeypatch.setattr('scripts.aufgabe04.real_robot.observer.qr_observation_pose.real_robot_profile_sha256',
        lambda _: orientation['robot_profile_sha256'])
    # This is a structural replay, not a latency benchmark. Keep the border
    # calculation real but avoid host scheduling making its .06s comparison
    # incomplete. Dedicated deadline tests cover cooperative cancellation, and
    # the source-clock tests below still reject stale decode/publication.
    replay_time = SimpleNamespace(monotonic=lambda: 1.)
    monkeypatch.setattr('scripts.aufgabe04.real_robot.observer.opposite_head_support.time', replay_time)
    monkeypatch.setattr('scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time', replay_time)
    decoder = Mock(return_value=(DecodedQrObservation('Start', None, 'recorded_wechat_payload', 4.),))
    monkeypatch.setattr(opposite_identity, 'detect_qr_observations_bgr', decoder)
    return case, adapter, clock, kwargs, decoder


def _bound_load(case, adapter):
    return load_bound_qr_observation_pose(adapter.args.qr_observation_pose_json,
        candidate_uid=adapter.args.stand_id, stream_id=adapter.args.stream_id,
        planning_frame='map', stand_x_m=adapter.args.stand_x, stand_y_m=adapter.args.stand_y,
        stand_model_profile_sha256=case['model'].sha256,
        robot_profile_sha256=case['orientation']['robot_profile_sha256'],
        calibration_profile_sha256=case['orientation']['calibration_profile_sha256'],
        base_frame='base_footprint', scan_frame='base_scan', camera_frame='camera')


@pytest.mark.parametrize('latest_opposite', (5, 16, 31), indirect=True)
def test_recorded_head_and_unique_scan_admit_cornerless_id_and_retained_facing(latest_opposite):
    case, adapter, _, kwargs, decoder = latest_opposite
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    decoder.assert_called_once()
    assert decoder.call_args.kwargs['identity_only'] is True
    current = adapter._pending_qr_observation_pose
    assert current.qr_binding.accepted and current.qr_corners is None
    assert current.qr_binding.qr_texts_for_evidence == ('Start',)
    metadata = adapter._capture_pending['detector_metadata']
    crop = metadata['identity_crop']
    assert crop['sampling'] == 'masked_current_head_region'
    assert crop['target_support']['policy'] == HEAD_POLICY
    assert crop['target_support']['supplies_angle'] is False
    assert any(c['candidate_uid'] == 'survey_candidate_0003' and c['occluded_by_target_head']
        for c in crop['competitors'])
    assert metadata['endpoint_confirmation'] == {}
    assert metadata['current_angle_refit'] is False
    state = adapter.observation_evidence.snapshot()
    assert state.current_axis_sample_count == 0 and state.current_qr_sample_count == 1
    assert commit_qr_observation_pose(adapter) is not None
    qr = _bound_load(case, adapter)
    assert qr['qr_id'] == 'Start' and qr['qr_corners_px'] is None
    assert qr['motion_authorized'] is False and qr['facing_ready'] is False
    assert qr['retained_backside_orientation'] == case['orientation']
    assert 'validated_target_center' not in qr['retained_backside_orientation']
    assert qr['arrival_target_reconciliation'] == json.loads(json.dumps(case['reconciliation']))
    rec = build_retained_facing(adapter.args.qr_observation_pose_json,
        stand_radius_m=case['snapshot'].candidate_for(adapter.args.stand_id).geometry.radius_m,
        target_distance_m=.4)
    assert rec.axis_sample_count == case['orientation']['axis_sample_count'] == 7
    assert rec.bounded_orientation == case['orientation']['bounded_orientation']
    adapter.args.recommended_pose_json.write_text(json.dumps(recommendation_to_dict(rec)))
    assert load_recommendation(adapter.args.recommended_pose_json) == rec


def test_missing_current_head_does_not_decode_background_overlap(latest_opposite):
    _, adapter, _, kwargs, decoder = latest_opposite
    kwargs['frame'] = np.full_like(kwargs['frame'], 255)
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    decoder.assert_not_called()
    assert not adapter._capture_pending['detector_metadata']['identity_crop']['accepted']
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


def test_decoded_id_preserves_discovery_but_cannot_exceed_thirty_degree_facing_bound(latest_opposite):
    case, adapter, _, kwargs, _ = latest_opposite
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert commit_qr_observation_pose(adapter) is not None
    # Original 3.883-degree angle plus the .14m terminal/center reserve at
    # .30m exceeds 30 degrees. Payload success cannot waive that endpoint bound.
    with pytest.raises(BoundedOrientationViewUnavailableError):
        build_retained_facing(adapter.args.qr_observation_pose_json,
            stand_radius_m=case['snapshot'].candidate_for(adapter.args.stand_id).geometry.radius_m,
            target_distance_m=.3)
    assert _bound_load(case, adapter)['qr_id'] == 'Start'


def test_head_pixels_without_current_reconciliation_cannot_admit_identity(latest_opposite):
    _, adapter, _, kwargs, decoder = latest_opposite
    kwargs['target_reconciliation'] = None
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    decoder.assert_not_called()
    assert not adapter._capture_pending['detector_metadata']['identity_crop']['accepted']
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


@pytest.mark.parametrize('texts', (('Start', 'QR_001'), ('Start', 'Start')))
def test_multiple_decoded_symbols_cannot_admit_candidate(latest_opposite, texts):
    _, adapter, _, kwargs, decoder = latest_opposite
    decoder.return_value = tuple(DecodedQrObservation(t, None, 'test_payload', 1.) for t in texts)
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    decoder.assert_called_once()
    assert not adapter._pending_qr_observation_pose.qr_binding.accepted
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


def test_source_expiry_after_decode_prevents_admission(latest_opposite):
    case, adapter, clock, kwargs, decoder = latest_opposite

    def slow_decode(*args, **options):
        clock.clock_sec = case['row']['image_stamp_sec'] + .6
        return (DecodedQrObservation('Start', None, 'test_payload', 1.),)

    decoder.side_effect = slow_decode
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    decoder.assert_called_once()
    assert not adapter._last_observation_update.frame_accepted
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


def test_source_expiry_after_ready_prevents_persisted_receipt(latest_opposite):
    case, adapter, clock, kwargs, _ = latest_opposite
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert adapter._qr_observation_pose_ready is not None
    clock.clock_sec = case['row']['image_stamp_sec'] + .6
    assert commit_qr_observation_pose(adapter) is None
    assert not adapter.args.qr_observation_pose_json.exists()


@pytest.mark.parametrize('tamper', ('missing_support', 'source_stamp', 'head_authority'))
def test_rehashed_receipt_cannot_drop_or_rebind_current_head_support(latest_opposite, tamper):
    case, adapter, _, kwargs, _ = latest_opposite
    opposite_identity.process_opposite_identity(adapter, **kwargs)
    assert commit_qr_observation_pose(adapter) is not None
    payload = deepcopy(_bound_load(case, adapter))
    crop = payload['qr_binding']['current_head_binding']
    if tamper == 'missing_support':
        crop['target_support'] = None
    elif tamper == 'source_stamp':
        crop['target_support']['image_stamp_sec'] += .001
    else:
        crop['target_support']['supplies_angle'] = True
    payload.pop(HASH_FIELD)
    with pytest.raises(ValueError):
        validate_qr_verified_observation_pose(content_hashed_payload(payload, hash_field=HASH_FIELD))
