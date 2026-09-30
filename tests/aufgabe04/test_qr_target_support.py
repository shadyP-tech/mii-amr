"""Decoded QR framing remains bound to its current candidate and raw scan."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import prepare_qr_observation_pose
from scripts.aufgabe04.real_robot.observer.qr_target_support import (
    prepare_qr_target_support, validate_qr_target_support,
)
from tests.aufgabe04 import test_target_reconciliation as reconciliation_fixture


def current_frame(*, processing_delay=0., minimum_samples=1, cone_deg=3.):
    case = reconciliation_fixture.TargetReconciliationTest()
    case.setUp()
    for item in case.rows:
        item['options']['cone_half_angle_rad'] = math.radians(cone_deg)
    row = case.rows[-1]
    binding = case.bind(case.proof(), now_sec=row['now_sec']+processing_delay,
        min_cluster_sample_count=minimum_samples)
    assert binding.accepted, binding.reason
    current = prepare_qr_observation_pose(qr_binding=binding,
        qr_observations=(DecodedQrObservation('QR_004',
            tuple(map(tuple, case.data['qr_corners_px'])), 'recorded'),),
        observed_qr_texts=('QR_004',), image_stamp_sec=row['image_stamp_sec'],
        scan_stamp_sec=row['scan'].scan_stamp_sec, robot_pose=EvidencePose(*row['robot_pose']),
        target_key=row['target_key'], camera_signature=(640., 640., 400., 300.),
        image_shape=(600, 800), roi=ImageRoi(0, 0, 800, 600, 100),
        model_profile_sha256='c'*64, metadata={})
    return current


@pytest.mark.parametrize('processing_delay', (0., .15))
def test_reconciled_decoded_quad_supplies_only_current_framing(processing_delay):
    current = current_frame(processing_delay=processing_delay)
    support = prepare_qr_target_support(current)
    assert support is not None
    assert support.accepted
    assert support.full_image_center_px == tuple(sum(p[k] for p in current.qr_corners)/4 for k in (0, 1))
    payload = json.loads(json.dumps(support.metadata()))
    assert validate_qr_target_support(payload) == payload
    assert payload['supplies_angle'] is False
    assert payload['supplies_identity'] is False
    assert payload['qr_binding']['motion_authorized'] is False
    assert payload['target_reconciliation']['candidate_geometry_updated'] is False
    assert payload['lidar_association'] == payload['qr_binding']['association']


@pytest.mark.parametrize('minimum_samples', (2, 3))
def test_stronger_configured_cluster_minimum_is_preserved(minimum_samples):
    support = prepare_qr_target_support(current_frame(minimum_samples=minimum_samples))
    assert support is not None
    assert support.lidar_association.search_association.min_cluster_sample_count == minimum_samples
    validate_qr_target_support(support.metadata())


def test_narrower_reconciled_cone_is_preserved():
    support = prepare_qr_target_support(current_frame(cone_deg=2.5))
    assert support is not None
    assert support.lidar_association.search_association.cone_half_angle_rad == math.radians(2.5)
    validate_qr_target_support(support.metadata())


@pytest.mark.parametrize('changes', (
    {'qr_corners': None}, {'observed_qr_texts': ('QR_004', 'Start')},
    {'observed_qr_texts': ()}, {'target_key': 'another-candidate'},
    {'retained_backside_orientation': {}}, {'arrival_target_reconciliation': {}},
    {'robot_pose': EvidencePose(1., 2., 3.)}, {'stamp_sec': 0.},
    {'scan_stamp_sec': 0.}, {'image_shape': (600, 640)},
))
def test_unbound_or_retained_frame_cannot_supply_ordinary_framing(changes):
    assert prepare_qr_target_support(replace(current_frame(), **changes)) is None


@pytest.mark.parametrize('changes', (
    {'accepted': False}, {'symbol_count': 2}, {'target_reconciliation': None},
    {'finite_bearing': None}, {'current_head_binding': {'accepted': True}},
    {'range_resolution': {'accepted': True}},
    {'reason': 'decoded_qr_exclusive_opposite_crop'}, {'qr_texts_for_evidence': ('Other',)},
))
def test_missing_independent_qr_proof_does_not_enable_framing(changes):
    current = current_frame()
    assert prepare_qr_target_support(replace(current,
        qr_binding=replace(current.qr_binding, **changes))) is None


def _alter_proof(payload, field, value):
    # Keep redundant transport fields consistent, so rejection must come from
    # recomputing the current sensor geometry rather than a duplicate mismatch.
    payload['target_reconciliation'][field] = value
    payload['qr_binding']['target_reconciliation'][field] = value


def _alter_lidar(payload, field, value, *, cluster=False):
    for record in (payload['lidar_association'], payload['qr_binding']['association']):
        (record['search_association'] if cluster else record)[field] = value


def _alter_finite(payload, field, value):
    payload['finite_bearing'][field] = value
    payload['qr_binding']['finite_bearing'][field] = value


@pytest.mark.parametrize('mutate', (
    lambda p: p.__setitem__('supplies_angle', True),
    lambda p: p.__setitem__('supplies_identity', True),
    lambda p: p.__setitem__('image_stamp_sec', p['image_stamp_sec']+.01),
    lambda p: p['corners_px'][0].__setitem__(0, 400.),
    lambda p: p.__setitem__('corners_px', [[-1., 1.], [5., 1.], [5., 5.], [-1., 5.]]),
    lambda p: _alter_finite(p, 'range_m', 1.),
    lambda p: _alter_finite(p, 'uncertainty_rad', math.nan),
    lambda p: _alter_finite(p, 'optical_depth_m', 1.),
    lambda p: _alter_lidar(p, 'distance_m', 1.),
    lambda p: _alter_lidar(p, 'max_camera_map_bearing_delta_rad', math.radians(12)),
    lambda p: _alter_lidar(p, 'eligible_cluster_count', 2, cluster=True),
    lambda p: _alter_lidar(p, 'selected_cluster_source_indices', [0, 1, 2], cluster=True),
    lambda p: _alter_lidar(p, 'scan_stamp_sec', 0., cluster=True),
    lambda p: _alter_lidar(p, 'scan_age_sec', .6, cluster=True),
    lambda p: _alter_lidar(p, 'scan_frame_id', 'another_scan', cluster=True),
    lambda p: p['qr_binding']['independent_registration']['envelope'].__setitem__('eligible_cluster_count', 2),
    lambda p: p['qr_binding']['independent_registration'].__setitem__('motion_authorized', True),
    lambda p: _alter_proof(p, 'snapshot_sha256', '0'*64),
    lambda p: _alter_proof(p, 'entries', []),
))
def test_current_support_recomputes_tampered_transport_geometry(mutate):
    payload = json.loads(json.dumps(prepare_qr_target_support(current_frame()).metadata()))
    mutate(payload)
    with pytest.raises(ValueError):
        validate_qr_target_support(payload)


def test_quad_shift_with_matching_center_still_has_to_match_current_scan_ray():
    payload = json.loads(json.dumps(prepare_qr_target_support(current_frame()).metadata()))
    for point in payload['corners_px']:
        point[0] += 100.
    payload['center_px'][0] += 100.
    with pytest.raises(ValueError, match='ray'):
        validate_qr_target_support(payload)


def test_changed_source_scan_is_not_hidden_by_matching_association_fields():
    payload = json.loads(json.dumps(prepare_qr_target_support(current_frame()).metadata()))
    proof = deepcopy(payload['target_reconciliation'])
    indices = payload['lidar_association']['search_association']['selected_cluster_source_indices']
    for index in indices:
        proof['entries'][-1]['scan']['ranges'][index] = None
    payload['target_reconciliation'] = proof
    payload['qr_binding']['target_reconciliation'] = proof
    with pytest.raises(ValueError):
        validate_qr_target_support(payload)
