"""Retained-center admission and bounded failure for the 20260928 real run."""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tests.aufgabe04.opposite_center_fixture import recorded_center, ROOT
from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation, RETAINED_POLICY
from scripts.aufgabe04.real_robot.observer.opposite_identity import process_opposite_identity
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import ScanPersistenceContext, _target_for_context, scan_pose_in_map, scan_pose_from_camera_extrinsics
from scripts.aufgabe04.real_robot.observer.opposite_target_geometry import retained_target_center
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


@pytest.fixture
def recorded(tmp_path):
    return recorded_center(tmp_path)


def test_recorded_center_fixes_bearing_and_range_without_joining_scan_seam(recorded):
    data,rows,_,_,snapshot,orientation,_ = recorded
    tracker = StoppedTargetReconciliation()
    proof = tracker.observe(**rows[1])
    assert proof is not None, tracker.metadata
    assert proof['policy'] == RETAINED_POLICY
    assert len(proof['entries']) == 1  # Confirmation of an already certified metric center.
    _,envelope,_,_ = validate_reconciliation(proof)
    assert envelope.eligible_cluster_count == 1
    assert envelope.selected_cluster_sample_count >= 3
    assert rows[1]['options']['accepted_range_m'][1] > .52
    assert data['frames'][1]['search']['envelope']['accepted_range_m'][1] < .481
    assert math.degrees(rows[1]['options']['map_bearing_rad']) < 1
    assert math.degrees(data['frames'][1]['search']['envelope']['map_bearing_rad']) > 12
    assert proof['retained_orientation']['bounded_orientation'] == orientation['bounded_orientation']
    for row in (rows[0],rows[2],rows[3]):
        assert tracker.observe(**row) is None
        assert 'compact three-beam cluster' in tracker.metadata['reason']


def test_retained_confirmation_rejects_missing_changed_stale_or_competing_sources(recorded,tmp_path):
    from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot
    _,rows,_,_,snapshot,orientation,_ = recorded
    r = rows[1];proof=StoppedTargetReconciliation().observe(**r)
    for mutate in (
        lambda p:p.pop('retained_orientation'),
        lambda p:p['retained_orientation']['validated_target_center'].__setitem__('y_m',0.),
        lambda p:p['entries'][0].__setitem__('checked_at_sec',r['now_sec']+1),
        lambda p:p['entries'][0]['options'].__setitem__('map_bearing_rad',.3),
        lambda p:p['entries'][0]['options'].__setitem__('accepted_range_m',(.1,1.)),
        lambda p:p['entries'][0].pop('target_geometry_basis'),
        lambda p:p['entries'].append(deepcopy(p['entries'][0])),
    ):
        bad=deepcopy(proof)
        mutate(bad)
        with pytest.raises((ValueError,KeyError)):validate_reconciliation(bad)
    _,_,world,_=validate_reconciliation(proof)
    other=snapshot.candidate_for('survey_candidate_0003')
    other=replace(other,geometry=replace(other.geometry,x_m=world[0],y_m=world[1]))
    snapshot=replace(snapshot,candidates=tuple(other if c.candidate_uid==other.candidate_uid else c for c in snapshot.candidates))
    path=tmp_path/'competitor.json';write_candidate_snapshot(path,snapshot)
    tracker=StoppedTargetReconciliation()
    assert tracker.observe(**{**r,'snapshot_path':path}) is None
    assert 'another candidate' in tracker.metadata['reason']
    with pytest.raises(ValueError):retained_target_center(orientation,stand_center=dict(x_m=0.,y_m=0.))


def frame_context(recorded):
    data,rows,intrinsics,camera,snapshot,orientation,model_path=recorded
    r=rows[1];f=data['frames'][1]
    def tf(parent,child):
        t=next(t for t in f['tf_samples'] if t['target_frame']==parent and t['source_frame']==child)
        return RigidTransform(parent,child,tuple(t['translation_xyz_m']),tuple(t['rotation_xyzw']))
    ci=dict(f['sensors']['camera_info']);ci['header']=SimpleNamespace(**ci['header'])
    image=rectify_bgr_frame(cv2.imread(str(ROOT/'frame_000024.jpg')),SimpleNamespace(**ci),cv2,np)
    return r,tf,image,load_measured_physical_stand_model(model_path)


def test_camera_and_independent_scan_witness_use_identical_certified_geometry(recorded):
    from dataclasses import asdict
    _,_,_,_,snapshot,orientation,_=recorded
    r,tf,_,_=frame_context(recorded);g=snapshot.candidate_for(r['candidate_uid']).geometry
    base=tf('base_footprint','camera');laser=tf('base_scan','camera')
    context=ScanPersistenceContext(target_key=r['target_key'],epoch_key='0',
        robot_pose=Pose2D(*r['robot_pose']),scan_pose_map=scan_pose_in_map(r['scan_from_map'].translation_xyz_m,r['scan_from_map'].rotation_xyzw),
        image_stamp_sec=r['image_stamp_sec'],candidate_x_m=g.x_m,candidate_y_m=g.y_m,
        stand_radius_m=g.radius_m,stand_uncertainty_m=g.uncertainty_m,lidar_range_tolerance_m=.04,
        scan_pose_robot=scan_pose_from_camera_extrinsics(base.translation_xyz_m,base.rotation_xyzw,laser.translation_xyz_m,laser.rotation_xyzw),
        retained_orientation=orientation)
    actual=_target_for_context(asdict(context))
    assert actual.accepted_range_m == pytest.approx(r['options']['accepted_range_m'])
    assert actual.bearing_rad == pytest.approx(r['options']['map_bearing_rad'])


@pytest.mark.parametrize("confirmed", [True, False])
def test_recorded_producer_immediate_qr_receipt_keeps_angle_and_candidate_identity(recorded, confirmed):
    from scripts.aufgabe04.real_robot.observer.qr_observation_pose import QrObservationPoseFallback
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES
    _,_,intrinsics,_,snapshot,orientation,_=recorded
    r,tf,image,model=frame_context(recorded);proof=StoppedTargetReconciliation().observe(**r)
    frames=[];states=[]
    snapshot_update=SimpleNamespace(target_key=r['target_key'],motion_epoch=0,poisoned=False,
        tentative_qr_id='Start',latched_qr_id=None,as_dict=lambda:{})
    update=SimpleNamespace(snapshot=snapshot_update,motion_epoch_reset=False,frame_accepted=True,qr_sample_accepted=True)
    adapter=SimpleNamespace(args=SimpleNamespace(stand_id=r['candidate_uid'],sync_tolerance_sec=.1,
        lidar_cone_half_angle_deg=3.,backside_registration_max_bearing_delta_deg=12.,max_sensor_age_sec=.5,
        candidate_centering_json=None),cv2=cv2,stand_model_profile=model,_capture_pending={},
        node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(r['now_sec']*1e9)))),
        _target_evidence_key=lambda:r['target_key'],_write_status=lambda state,**kw:states.append((state,kw)),
        _record_observation_frame=lambda **kw:frames.append(kw) or update)
    # Payload backend is separately replayed on deployed OpenCV; association,
    # complete outline, crop, and downstream receipt validation are real here.
    with patch('scripts.aufgabe04.real_robot.observer.opposite_identity.detect_qr_observations_bgr',
               return_value=(DecodedQrObservation('Start',None,'recorded'),)) as decode:
        process_opposite_identity(adapter,context=SimpleNamespace(orientation=orientation,snapshot=snapshot),
            frame=image,intrinsics=intrinsics,robot_pose=Pose2D(*r['robot_pose']),camera_signature=(1,2,3,4),
            image_stamp_sec=r['image_stamp_sec'],scan=r['scan'],scan_from_map=r['scan_from_map'],camera_from_map=tf('camera','map'),
            map_bearing_rad=r['options']['map_bearing_rad'],accepted_range_m=r['options']['accepted_range_m'],
            scan_from_camera=tf('base_scan','camera'),base_from_camera=tf('base_footprint','camera'),image_stamp=None,
            target_reconciliation=proof if confirmed else None,require_target_reconciliation=True)
        if not confirmed:
            decode.assert_not_called()
            assert not frames[-1]['qr_texts'] and not frames[-1]['lidar_associated']
            assert states[-1][1]['stand_axis_debug']['identity_crop']['reason']=='retained_target_reconciliation_pending'
            assert getattr(adapter,'_pending_qr_observation_pose',None) is None
            return
        decode.assert_called_once()
    assert frames[-1]['qr_texts']==('Start',) and frames[-1]['axis_yaw_rad'] is None
    current=adapter._pending_qr_observation_pose
    assert QrObservationPoseFallback(delay_sec=1.5).observe(current,update=update,
        observed_at_sec=r['now_sec'],now_monotonic_sec=1.) is not None
    receipt=build_qr_verified_observation_pose(candidate_uid=r['candidate_uid'],stream_id='recorded',planning_frame='map',
        qr_id='Start',stand_center=dict(zip(('x_m','y_m'),r['stand_center'])),robot_pose=dict(zip(('x_m','y_m','yaw_rad'),r['robot_pose'])),
        sensor_stamp_sec=r['image_stamp_sec'],scan_stamp_sec=r['scan'].scan_stamp_sec,checked_at_sec=r['now_sec'],
        robot_profile_sha256=orientation['robot_profile_sha256'],calibration_profile_sha256=orientation['calibration_profile_sha256'],
        stand_model_profile_sha256=model.sha256,target_key=r['target_key'],motion_epoch=0,
        camera_signature=current.camera_signature,qr_corners_px=None,image_shape=current.image_shape,
        qr_binding=current.qr_binding.metadata(),source_gates={k:True for k in SOURCE_GATES},
        retained_backside_orientation=orientation,arrival_target_reconciliation=proof,
        localization_provenance=dict(map_frame='map',base_frame='base_footprint',scan_frame='base_scan',camera_frame='camera',
            exact_image_transform_stamp_sec=r['image_stamp_sec'],exact_scan_transform_stamp_sec=r['scan'].scan_stamp_sec))
    assert validate_qr_verified_observation_pose(receipt)['qr_id']=='Start'
    assert receipt['stand_axis_rad']==orientation['stand_axis_rad']
    assert adapter._opposite_identity_failure is None
