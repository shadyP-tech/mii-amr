"""The Sep29 range displacement reaches proof consumers without widening gates."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation
from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import validated_reconciliation_envelope
from tests.aufgabe04.test_candidate_position_epoch import recorded_rows, camera_inputs

ROOT = Path(__file__).parent/'fixtures/candidate_position_epoch_radial'


def radial_proof():
    tracker = StoppedTargetReconciliation()
    rows = list(recorded_rows(ROOT))
    assert tracker.observe(**rows[0]) is None
    assert tracker.observe(**rows[1]) is None
    proof = tracker.observe(**rows[2])
    assert proof is not None, tracker.metadata
    return proof, rows[-1]


def qr_binding(proof, row, **changes):
    from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
    from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
    from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
    intrinsics, tf, _ = camera_inputs(ROOT)
    recorded = json.loads((ROOT/'observations.json').read_text())['qr_observation']
    quad = tuple(map(tuple,recorded['corners']))
    args = dict(roi=ImageRoi(0,0,800,600,100),intrinsics=intrinsics,
        scan_from_camera=tf('base_scan','camera'),scan=row['scan'],now_sec=row['now_sec'],
        max_scan_age_sec=.5,min_cluster_sample_count=1,camera_registration_accepted=False,
        allow_independent_registration=True,target_reconciliation=proof,**row['options'])
    args.update(changes)
    result = bind_qr_observations_to_target(
        (DecodedQrObservation(recorded['text'],quad,recorded['detector'],recorded['scale']),),**args)
    return result, quad


def test_radial_recovery_retains_source_and_requires_three_stopped_scans():
    proof, row = radial_proof()
    _, envelope, _, reference = validate_reconciliation(proof)
    options = row['options']
    assert options['accepted_range_m'][1] < .58
    assert envelope.accepted_range_m[0] > .52
    assert envelope.accepted_range_m[1] < .75
    assert .65 < envelope.distance_m < .68
    assert envelope.selected_cluster_sample_count >= 3
    assert proof['entries'][-1]['options'] == options
    assert envelope.map_bearing_rad == options['map_bearing_rad']
    assert abs(reference-options['map_bearing_rad']) > math.radians(10)
    assert proof['stand_center'] == list(row['stand_center'])
    assert proof['candidate_geometry_updated'] is False
    assert proof['motion_authorized'] is False
    ordinary = associate_candidate_lidar_target(row['scan'],map_bearing_rad=options['map_bearing_rad'],
        cone_half_angle_rad=math.radians(35),accepted_range_m=options['accepted_range_m'],
        now_sec=row['now_sec'],max_scan_age_sec=.5)
    assert ordinary.eligible_cluster_count == 0
    assert not qr_binding(None,row)[0].accepted


@pytest.mark.parametrize('bearing_deg, distance', [(-25,.50), (30,.66), (30,.57)])
def test_recovery_counts_single_beam_competitors_across_both_ranges(bearing_deg,distance):
    # The first point is visible only to the current range; the second is only
    # in the frozen range. Even one competing beam must prevent recovery.
    tracker = StoppedTargetReconciliation()
    for row in recorded_rows(ROOT):
        scan = row['scan']
        index = round(((math.radians(bearing_deg)-scan.angle_min) % math.tau)/scan.angle_increment)
        ranges = list(scan.ranges)
        ranges[index] = distance
        assert tracker.observe(**{**row,'scan':replace(scan,ranges=tuple(ranges))}) is None
    assert 'competing clusters' in tracker.metadata['reason']


@pytest.mark.parametrize('beam_count, reason', [(2,'three-beam'), (12,'not compact')])
def test_recovery_rejects_insufficient_or_extended_support(beam_count,reason):
    tracker = StoppedTargetReconciliation()
    for row in recorded_rows(ROOT):
        scan = row['scan']
        ranges = [math.inf]*len(scan.ranges)
        ranges[5:5+beam_count] = [.665]*beam_count
        assert tracker.observe(**{**row,'scan':replace(scan,ranges=tuple(ranges))}) is None
    assert reason in tracker.metadata['reason']


@pytest.mark.parametrize('defect', ['range', 'bearing', 'scan', 'epoch', 'stale', 'moved', 'support'])
def test_radial_proof_rejects_tampered_and_stale_sources(defect):
    proof, row = radial_proof()
    entry = proof['entries'][-1]
    if defect == 'range':
        entry['options']['accepted_range_m'] = (.3,.8)
    elif defect == 'bearing':
        entry['options']['map_bearing_rad'] += .1
    elif defect == 'epoch':
        for e in proof['entries']:
            e['position_epoch']['sha256'] = '0'*64
    elif defect == 'stale':
        # Sensor tuples remain fresh, but the authenticated epoch expires.
        for e in proof['entries']:
            for key in ('checked_at_sec','image_stamp_sec'):
                e[key] += 31
            for key in ('receipt_sec','scan_stamp_sec'):
                e['scan'][key] += 31
    elif defect == 'moved':
        entry['robot_pose'][0] += .04
    elif defect == 'support':
        proof['entries'].pop()
    elif defect == 'scan':
        ranges = list(row['scan'].ranges)
        ranges[6] += .01
        row = {**row,'scan':replace(row['scan'],ranges=tuple(ranges))}
    with pytest.raises(ValueError):
        validated_reconciliation_envelope(proof,scan=row['scan'],
            map_bearing_rad=row['options']['map_bearing_rad'],accepted_range_m=row['options']['accepted_range_m'])


def test_qr_consumer_and_receipt_use_authenticated_range_and_retain_source_binding():
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
        build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES, HASH_FIELD)
    from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
    proof, row = radial_proof()
    binding, quad = qr_binding(proof,row)
    assert binding.accepted, binding.reason
    assert binding.qr_texts_for_evidence == ('QR_001',)
    assert binding.finite_bearing['range_interval_m'][1] > row['options']['accepted_range_m'][1]
    assert not qr_binding(proof,row,accepted_range_m=(.3,.8))[0].accepted
    assert not qr_binding(proof,row,map_bearing_rad=.2)[0].accepted
    intrinsics, _, _ = camera_inputs(ROOT)
    receipt = build_qr_verified_observation_pose(candidate_uid=row['candidate_uid'],stream_id='fixture',
        planning_frame='map',qr_id='QR_001',stand_center=dict(zip(('x_m','y_m'),row['stand_center'])),
        robot_pose=dict(zip(('x_m','y_m','yaw_rad'),row['robot_pose'])),sensor_stamp_sec=row['image_stamp_sec'],
        scan_stamp_sec=row['scan'].scan_stamp_sec,checked_at_sec=row['now_sec'],target_key=row['target_key'],motion_epoch=0,
        robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='c'*64,
        camera_signature=(intrinsics.fx_px,intrinsics.fy_px,intrinsics.cx_px,intrinsics.cy_px),
        qr_corners_px=quad,image_shape=(600,800),qr_binding=binding.metadata(),
        source_gates={k:True for k in SOURCE_GATES},localization_provenance=dict(map_frame='map',base_frame='base_footprint',
            scan_frame='base_scan',camera_frame='camera',exact_image_transform_stamp_sec=row['image_stamp_sec'],
            exact_scan_transform_stamp_sec=row['scan'].scan_stamp_sec))
    assert validate_qr_verified_observation_pose(json.loads(json.dumps(receipt)))['qr_id'] == 'QR_001'
    altered = copy.deepcopy(receipt)
    altered['qr_binding']['finite_bearing']['range_interval_m'] = (.3,.8)
    altered.pop(HASH_FIELD)
    with pytest.raises(ValueError):
        validate_qr_verified_observation_pose(content_hashed_payload(altered,hash_field=HASH_FIELD))


def test_qr_search_and_head_consumer_use_the_same_recovered_cluster():
    from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    from scripts.aufgabe04.real_robot.configuration.geometry import OpticalProjection, ImageRoi
    from scripts.aufgabe04.real_robot.observer.qr_candidate_search import current_scan_qr_search
    from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
    from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
    from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import recovered_search
    from tests.aufgabe04.test_head_model_admission import head_estimate, head_debug, outer_boundary
    proof, row = radial_proof()
    intrinsics, tf, _ = camera_inputs(ROOT)
    profile = load_measured_physical_stand_model(Path('configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json'))
    args = dict(scan=row['scan'],scan_from_map=row['scan_from_map'],camera_from_map=tf('camera','map'),
        intrinsics=intrinsics,model_profile=profile,image_stamp_sec=row['image_stamp_sec'],sync_tolerance_sec=.1,
        now_sec=row['now_sec'],max_scan_age_sec=.5,**row['options'])
    assert current_scan_qr_search(**args)[0] is None
    attempt, metadata = current_scan_qr_search(**args,target_reconciliation=proof)
    assert attempt is not None, metadata
    assert metadata['envelope']['accepted_range_m'][1] > .7
    data = json.loads((ROOT/'observations.json').read_text())
    projection = OpticalProjection(**data['projection'])
    hint, _ = recovered_search(proof,scan=row['scan'],original_projection=projection,
        camera_from_map=tf('camera','map'),intrinsics=intrinsics,model_profile=profile)
    corners = tuple(ImagePoint(**p) for p in data['viewer_corners'])
    # Synthetic admitted quality contract at recorded corners: this isolates
    # range propagation and does not claim the saved axis fit was admitted.
    estimate = head_estimate(corners=corners,left_height_px=85.34,right_height_px=83.87)
    options = dict(estimate=estimate,debug=head_debug(head_outer_recovery=outer_boundary(corners)),
        attempt=HeadRoiAttempt(ImageRoi(0,0,800,600,100),'fixture_full_image',1.8,400.,300.,109.),
        projection=projection,expected_head_height_px=projection.expected_size_px,profile_sha256='a'*64,
        intrinsics=intrinsics,scan_from_camera=tf('base_scan','camera'),scan=row['scan'],now_sec=row['now_sec'],
        max_scan_age_sec=.5,min_cluster_sample_count=1,max_center_offset_ratio=1.5,
        search_reconciliation=hint,**row['options'])
    assert not associate_current_measured_head(**options).accepted
    head = associate_current_measured_head(**options,target_reconciliation=proof)
    assert head.accepted, head.reason
    assert head.lidar_association.search_association.accepted_range_m == tuple(metadata['envelope']['accepted_range_m'])
    from scripts.aufgabe04.real_robot.observer.candidate_centering import (
        build_camera_centering_advisory, validate_camera_centering_advisory)
    from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
    odom = tf('odom','base_footprint')
    q = odom.rotation_xyzw
    advice = build_camera_centering_advisory(association=head,intrinsics=intrinsics,
        scan_from_camera=tf('base_scan','camera'),base_from_camera=tf('base_footprint','camera'),
        candidate_uid=row['candidate_uid'],target_key=row['target_key'],stream_id='fixture',planning_frame='map',motion_epoch=0,
        anchor_pose=EvidencePose(*row['robot_pose']),anchor_odom_pose=EvidencePose(*odom.translation_xyz_m[:2],2*math.atan2(q[2],q[3])),
        odom_stamp_sec=row['image_stamp_sec'],image_stamp_sec=row['image_stamp_sec'],now_sec=row['now_sec'],
        robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='a'*64)
    assert advice is not None
    assert advice.associated_range_m > .65
    validate_camera_centering_advisory(json.loads(json.dumps(advice.metadata())))
