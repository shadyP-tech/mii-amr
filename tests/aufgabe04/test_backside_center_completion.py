"""The seven-angle/one-center transition and no-refit arrival recovery."""
from dataclasses import asdict
from copy import deepcopy
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.aufgabe04.real_robot.observer.backside_center_opportunity import backside_center_pending


def test_backside_waits_for_center_without_losing_angle_window():
    from tests.aufgabe04.test_bounded_head_observation import BoundedHeadObservationTests as _BoundedHeadObservationTests
    f=_BoundedHeadObservationTests();f.setUpClass();f.setUp()
    try:
        f.adapter.args.candidate_crop_snapshot=f.root/'snapshot.json'
        for i in range(9):
            with patch('scripts.aufgabe04.real_robot.observer.backside_center_opportunity.time.monotonic',return_value=100+i*.2):
                _,metadata=f.frame(100+i*.2,face='backside',reconcile=i>=6)
            if i in (6,7):
                assert f.payload('backside') is None
                assert metadata['backside_center_opportunity']['pending']
        payload=f.payload('backside')
        assert payload is not None
        assert len(payload['target_reconciliation']['entries'])==3
        assert payload['bounded_orientation']['half_width_rad']==pytest.approx(f.proof.half_width_rad)
        assert payload['axis_sample_count']==7
    finally:
        f.doCleanups()


def test_center_wait_cannot_be_renewed_by_soft_misses():
    snapshot=SimpleNamespace(target_key='a',motion_epoch=0)
    adapter=SimpleNamespace(args=SimpleNamespace(candidate_crop_snapshot='fixture'),
                            observation_evidence=SimpleNamespace(snapshot=lambda:snapshot))
    assert backside_center_pending(adapter,center_ready=False,metadata={},now=10.)
    assert backside_center_pending(adapter,center_ready=False,metadata={},now=11.49)
    diagnostic={}
    assert not backside_center_pending(adapter,center_ready=False,metadata=diagnostic,now=11.5)
    assert diagnostic['backside_center_opportunity']['reason']=='center_opportunity_exhausted'
    assert not backside_center_pending(adapter,center_ready=False,metadata={},now=15.)
    snapshot.motion_epoch=1
    assert backside_center_pending(adapter,center_ready=False,metadata={},now=16.)


@pytest.fixture
def legacy_arrival(tmp_path):
    from tests.aufgabe04.backside_axis_fixture import backside_axis_payload, write_candidate_frame_projection_fixture
    from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import write_backside_axis_frame_projection
    from scripts.aufgabe04.artifacts.retained_backside_orientation import orientation_record
    from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
    from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
    from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation
    from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
    from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import bind_crop_text, POLICY
    from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, SOURCE_GATES
    axis=tmp_path/'axis.json';source=tmp_path/'source.json'
    raw=backside_axis_payload(stand_x_m=1.,robot_x_m=1.,robot_y_m=.7)
    raw['bounded_orientation']=dict(policy='current_head_noise_expanded_interval',center_rad=0.,half_width_rad=.1,sample_count=7)
    axis.write_text(json.dumps(raw))
    sha,_,_=write_candidate_frame_projection_fixture(source,candidate_uid='candidate_1',canonical_x_m=1.,
        canonical_y_m=0.,transform_x_m=0.,transform_y_m=0.,transform_yaw_rad=0.)
    projection=tmp_path/'orientation.json'
    write_backside_axis_frame_projection(projection,axis_evidence_path=axis,source_candidate_projection_path=source,
        source_candidate_projection_sha256=sha,target_candidate_projection_path=source,
        target_candidate_projection_sha256=sha,target_candidate_x_m=1.,target_candidate_y_m=0.)
    orientation=orientation_record(projection)
    assert 'validated_target_center' not in orientation
    snapshot=json.loads(source.read_text())['projected_candidate_snapshot_path']
    pose=dict(x_m=1.,y_m=-.5,yaw_rad=math.pi/2)
    options=dict(map_bearing_rad=0.,cone_half_angle_rad=math.radians(3),
        max_camera_map_bearing_delta_rad=math.radians(12),accepted_range_m=(.32,.54))
    tracker=StoppedTargetReconciliation()
    for stamp in (200.,200.1,200.2):
        scan=PlainLaserScan(ranges=tuple(.45 if 88<=i<=92 else math.nan for i in range(181)),
            angle_min=-.9,angle_increment=.01,range_min=.1,range_max=3.,scan_frame_id='base_scan',
            scan_stamp_sec=stamp,receipt_sec=stamp,angle_max=.9,scan_topology_profile='linear')
        proof=tracker.observe(snapshot_path=snapshot,candidate_uid='candidate_1',planning_frame='map',
            stand_center=(1.,0.),target_key='fixture',epoch=0,scan=scan,
            scan_from_map=RigidTransform('base_scan','map',(.5,1.,0.),(0.,0.,-math.sqrt(.5),math.sqrt(.5))),
            robot_pose=tuple(pose.values()),image_stamp_sec=stamp,now_sec=stamp+.1,options=options,
            retained_orientation=orientation)
    assert proof is not None,tracker.metadata
    envelope=qr_registration_envelope(scan,**options,now_sec=stamp+.1,max_scan_age_sec=.5)
    crop=dict(policy=POLICY,accepted=True,candidate_uid='candidate_1',image_stamp_sec=stamp,scan_stamp_sec=stamp,
        bounds_xyxy=[300,200,500,400],target_center_px=[400.,300.],competitors=[],
        search=dict(accepted=True,envelope=asdict(envelope)))
    binding=bind_crop_text((DecodedQrObservation('Start',None,'fixture'),),crop)
    fields=dict(candidate_uid='candidate_1',stream_id='fixture',qr_id='Start',planning_frame='map',
        stand_center=orientation['stand_center'],robot_pose=pose,sensor_stamp_sec=stamp,scan_stamp_sec=stamp,checked_at_sec=stamp+.1,
        robot_profile_sha256=orientation['robot_profile_sha256'],calibration_profile_sha256=orientation['calibration_profile_sha256'],
        stand_model_profile_sha256=orientation['stand_model_profile_sha256'],target_key='fixture',motion_epoch=0,
        camera_signature=(640.,640.,400.,300.),qr_corners_px=None,image_shape=(600,800),qr_binding=binding.metadata(),
        retained_backside_orientation=orientation,arrival_target_reconciliation=proof,
        source_gates={k:True for k in SOURCE_GATES},localization_provenance=dict(map_frame='map',base_frame='base_footprint',
            scan_frame='base_scan',camera_frame='camera',exact_image_transform_stamp_sec=stamp,exact_scan_transform_stamp_sec=stamp))
    qr=build_qr_verified_observation_pose(**fields)
    path=tmp_path/'qr.json';path.write_text(json.dumps(qr))
    return path,qr,fields


def test_arrival_center_promotes_legacy_angle_without_front_corners_or_refit(legacy_arrival):
    from scripts.aufgabe04.artifacts.retained_facing import build_retained_facing
    path,qr,_=legacy_arrival
    rec=build_retained_facing(path,stand_radius_m=.06,target_distance_m=.6)
    with pytest.raises(ValueError,match='obliquity'):
        build_retained_facing(path,stand_radius_m=.06,target_distance_m=.35)
    assert qr['qr_corners_px'] is None
    assert rec.bounded_orientation==qr['retained_backside_orientation']['bounded_orientation']
    assert rec.axis_sample_count==7 and rec.axis_measurement['current_angle_refit'] is False
    face=next(f for f in rec.face_candidates if f.face_id==rec.material_target.face_id)
    assert abs(math.remainder(face.outward_normal_rad+math.pi/2,math.tau))<1e-9
    assert rec.stand.uncertainty_m==pytest.approx(.11)  # Surface uncertainty is not silently reduced.


@pytest.mark.parametrize('mutation',[
    lambda f:f['arrival_target_reconciliation'].update(epoch=2),
    lambda f:f['arrival_target_reconciliation']['entries'].__setitem__(0,f['arrival_target_reconciliation']['entries'][1]),
    lambda f:f['arrival_target_reconciliation']['entries'][-1]['robot_pose'].__setitem__(0,2.),
    lambda f:f['retained_backside_orientation']['bounded_orientation'].update(half_width_rad=0.),
])
def test_arrival_center_cannot_rebind_historical_angle(legacy_arrival,mutation):
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose
    fields=deepcopy(legacy_arrival[2]);mutation(fields)
    with pytest.raises(ValueError):build_qr_verified_observation_pose(**fields)


def test_runtime_reads_arrival_proof_from_receipt_not_discovery_summary(legacy_arrival,tmp_path):
    from scripts.aufgabe04.real_robot.candidate.retained_facing import try_retained_facing
    from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
    from scripts.aufgabe04.navigation.foundation.models import Pose2D
    path,qr,_=legacy_arrival
    observation=CandidateObservation(None,'Start',None,qr_observation_pose_path=path)
    frame=SimpleNamespace(candidate=SimpleNamespace(geometry=SimpleNamespace(radius_m=.06),candidate_uid='candidate_1'),
                          config=SimpleNamespace(final_facing_offset_m=.6))
    summary={'retained_backside_orientation':qr['retained_backside_orientation']}
    with patch('scripts.aufgabe04.real_robot.candidate.approach._read_finite_pose2d',return_value=Pose2D(**qr['robot_pose'])):
        result,facing=try_retained_facing(observation=observation,discovery=summary,frame=frame,
            effects=SimpleNamespace(validate_facing=lambda r:{}),output_dir=tmp_path)
    assert result.recommendation_path.is_file() and facing['facing_ready']
    assert facing['validated_target_center']['uncertainty_m']==pytest.approx(.11)
