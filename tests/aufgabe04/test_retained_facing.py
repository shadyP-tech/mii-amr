"""Retained orientation is promoted only with its immutable current QR chain."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest

from tests.aufgabe04.opposite_reconciliation_fixture import recorded_opposite
from tests.aufgabe04.test_opposite_reconciliation import support_for
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import exclusive_identity_crop, bind_crop_text
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, SOURCE_GATES
from scripts.aufgabe04.artifacts.retained_facing import build_retained_facing
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import validate_recommendation, recommendation_to_dict, load_recommendation
from scripts.aufgabe04.navigation.foundation.models import Pose2D


@pytest.fixture
def retained_qr(tmp_path):
    recorded = recorded_opposite(tmp_path)
    image, options, snapshot, tf, proof, orientation, row, axis = recorded
    support, search = support_for(recorded)
    _, crop = exclusive_identity_crop(candidate_uid=row['candidate_uid'], snapshot=snapshot,
        support=support, search_result=search, **options)
    binding = bind_crop_text((DecodedQrObservation('Start',None,'fixture'),),crop)
    stamp, scan = row['image_stamp_sec'], row['scan'].scan_stamp_sec
    payload = build_qr_verified_observation_pose(candidate_uid=row['candidate_uid'],
        stream_id='fixture', qr_id='Start', planning_frame='map',
        stand_center=orientation['stand_center'], robot_pose=asdict(Pose2D(*row['robot_pose'])),
        sensor_stamp_sec=stamp, scan_stamp_sec=scan, checked_at_sec=row['now_sec'],
        robot_profile_sha256=orientation['robot_profile_sha256'],
        calibration_profile_sha256=orientation['calibration_profile_sha256'],
        stand_model_profile_sha256=orientation['stand_model_profile_sha256'],
        target_key=row['target_key'], motion_epoch=0, camera_signature=(640.,640.,400.,300.),
        qr_corners_px=None, image_shape=image.shape[:2], qr_binding=binding.metadata(),
        retained_backside_orientation=orientation, source_gates={k:True for k in SOURCE_GATES},
        localization_provenance=dict(map_frame='map', base_frame='base_footprint',scan_frame='base_scan',
            camera_frame='camera',exact_image_transform_stamp_sec=stamp,exact_scan_transform_stamp_sec=scan))
    path=tmp_path/'qr.json';path.write_text(json.dumps(payload))
    return path,payload,snapshot


def test_retains_real_angle_interval_and_measured_center_without_refit(retained_qr,tmp_path):
    path,qr,snapshot=retained_qr
    rec=build_retained_facing(path,stand_radius_m=.07,target_distance_m=.4)
    source=qr['retained_backside_orientation'];center=source['validated_target_center']
    assert rec.bounded_orientation==source['bounded_orientation']
    assert rec.axis_sample_count==source['axis_sample_count']
    assert rec.stand.center==Pose2D(center['x_m'],center['y_m'])
    assert rec.stand.uncertainty_m==center['uncertainty_m']
    assert rec.sensor_stamp_sec==qr['sensor_stamp_sec']
    assert rec.axis_measurement['current_angle_refit'] is False
    assert qr['facing_ready'] is False
    out=tmp_path/'rec.json';out.write_text(json.dumps(recommendation_to_dict(rec)))
    assert load_recommendation(out)==rec


@pytest.mark.parametrize('mutation',[
    lambda r:replace(r,sensor_stamp_sec=r.sensor_stamp_sec+1),
    lambda r:replace(r,stand=replace(r.stand,center=Pose2D(0,0))),
    lambda r:replace(r,bounded_orientation={**r.bounded_orientation,'half_width_rad':0.}),
    lambda r:replace(r,axis_sample_count=8),
    lambda r:replace(r,schema_version=2),
    lambda r:replace(r,face_candidates=()),
    lambda r:replace(r,axis_measurement={'policy':'retained_backside_current_qr_facing'}),
])
def test_rejects_retained_proof_rebinding(retained_qr,mutation):
    path,_,_=retained_qr
    rec=build_retained_facing(path,stand_radius_m=.07,target_distance_m=.4)
    with pytest.raises(ValueError):validate_recommendation(mutation(rec))


def test_source_mutation_invalidates_even_an_existing_recommendation(retained_qr):
    path,_,_=retained_qr;rec=build_retained_facing(path,stand_radius_m=.07,target_distance_m=.4)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='source changed'):validate_recommendation(rec)


def test_failed_facing_validation_preserves_qr_discovery(retained_qr,tmp_path):
    from scripts.aufgabe04.real_robot.candidate.retained_facing import try_retained_facing
    from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
    path,qr,snapshot=retained_qr;candidate=snapshot.candidate_for(qr['candidate_uid'])
    observation=CandidateObservation(None,'Start',None,qr_observation_pose_path=path)
    frame=SimpleNamespace(candidate=candidate,config=SimpleNamespace(final_facing_offset_m=.4))
    def fail(_request):raise ValueError('blocked route')
    effects=SimpleNamespace(validate_facing=fail)
    with patch('scripts.aufgabe04.real_robot.candidate.approach._read_finite_pose2d',return_value=Pose2D(**qr['robot_pose'])):
        assert try_retained_facing(observation=observation,discovery=qr,frame=frame,effects=effects,output_dir=tmp_path) is None
    assert observation.qr_observation_pose_path==path
    assert json.loads((tmp_path/'retained_facing_status.json').read_text())['reason']=='blocked route'


def test_successful_optional_facing_keeps_qr_source_and_has_no_motion_effect(retained_qr,tmp_path):
    from scripts.aufgabe04.real_robot.candidate.retained_facing import try_retained_facing
    from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
    path,qr,snapshot=retained_qr;candidate=snapshot.candidate_for(qr['candidate_uid'])
    observation=CandidateObservation(None,'Start',None,qr_observation_pose_path=path)
    frame=SimpleNamespace(candidate=candidate,config=SimpleNamespace(final_facing_offset_m=.4))
    requests=[]
    def validate(request):
        requests.append(request)
        rec=load_recommendation(request.recommendation_path)
        assert rec.stand.center!=Pose2D(candidate.geometry.x_m,candidate.geometry.y_m)
        return {'candidate_uid':candidate.candidate_uid,'motion_to_facing_pose_authorized':False}
    with patch('scripts.aufgabe04.real_robot.candidate.approach._read_finite_pose2d',return_value=Pose2D(**qr['robot_pose'])):
        result,facing=try_retained_facing(observation=observation,discovery=qr,frame=frame,
            effects=SimpleNamespace(validate_facing=validate),output_dir=tmp_path)
    assert result.recommendation_path.is_file() and result.qr_observation_pose_path is None
    assert path.is_file() and len(requests)==1
    assert facing['facing_ready'] and facing['qr_id']=='Start'
    assert not facing['motion_to_facing_pose_authorized']


def test_retained_center_binding_authenticates_original_candidate(retained_qr,tmp_path):
    import hashlib
    from scripts.aufgabe04.navigation.approach.camera_decision_geometry_binding import require_camera_recommendation_binding
    path,qr,snapshot=retained_qr;candidate=snapshot.candidate_for(qr['candidate_uid'])
    rec=build_retained_facing(path,stand_radius_m=candidate.geometry.radius_m,target_distance_m=.4)
    output=tmp_path/'rec.json';output.write_text(json.dumps(recommendation_to_dict(rec)))
    receipt=dict(camera_evidence_path=str(output),camera_recommendation_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    require_camera_recommendation_binding(receipt,candidate=candidate,planning_frame='map')
    moved=replace(candidate,geometry=replace(candidate.geometry,x_m=candidate.geometry.x_m+.01))
    with pytest.raises(ValueError,match='immutable candidate'):
        require_camera_recommendation_binding(receipt,candidate=moved,planning_frame='map')


def test_retained_facing_runs_real_route_and_clearance_validation(retained_qr,tmp_path):
    from tests.aufgabe04.test_autonomous_candidate_approach import AutonomousCandidateApproachTest
    from tests.aufgabe04.test_detected_station_exploration import write_free_map
    from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
    from scripts.aufgabe04.real_robot.candidate.approach import validate_facing_pose, FacingValidationRequest, load_occupancy_grid_with_bundle
    path,qr,snapshot=retained_qr;candidate=snapshot.candidate_for(qr['candidate_uid'])
    rec=build_retained_facing(path,stand_radius_m=candidate.geometry.radius_m,target_distance_m=.4)
    output=tmp_path/'rec.json';output.write_text(json.dumps(recommendation_to_dict(rec)))
    map_path=write_free_map(tmp_path,width=60,height=60)
    map_path.write_text(map_path.read_text().replace('[-1.0, -1.0, 0.0]','[-3.0, -3.0, 0.0]'))
    _,bundle=load_occupancy_grid_with_bundle(map_path,semantic_map_id='arena',planning_frame='map')
    route_candidate=replace(candidate,source=replace(candidate.source,perception_advisories=()))
    config=AutonomousCandidateApproachTest()._config(tmp_path,(route_candidate,))
    config=replace(config,map_yaml=map_path,plan=replace(config.plan,
        map_bundle_sha256=bundle.bundle_sha256,arena_bounds=ArenaBounds(length_m=6.,width_m=6.,center_x_m=0.,center_y_m=0.)))
    request=FacingValidationRequest(config,candidate,output,Pose2D(**qr['robot_pose']),tmp_path/'facing')
    result=validate_facing_pose(request)
    assert result['bounded_orientation']==qr['retained_backside_orientation']['bounded_orientation']
    assert result['active_stand_clearance']['continuous_centerline_validated']
    assert Path(result['validation_route_csv']).is_file()
    assert not result['motion_to_facing_pose_authorized']
    # The current head point must not replace/remove the original keepout.
    blocked=replace(config,physical_clearance={**config.physical_clearance,
        'minimum_active_standoff_m':.6,'minimum_collision_standoff_m':.5})
    with pytest.raises(ValueError,match='standoff'):
        validate_facing_pose(replace(request,config=blocked))
