"""Replay the displaced 20260928 candidate; no ROS or motion."""
import json
import math
from pathlib import Path
from dataclasses import asdict

import pytest

from scripts.aufgabe04.real_robot.observer.target_reconciliation import StoppedTargetReconciliation, validate_reconciliation
from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import is_epoch_recovery
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import transform_point
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot

ROOT=Path(__file__).parent/'fixtures/candidate_position_epoch'


def recorded_rows(root=ROOT):
    snapshot=load_candidate_snapshot(root/'candidate_snapshot.json')
    g=snapshot.candidate_for('survey_candidate_0005').geometry
    data=json.loads((root/'observations.json').read_text())
    for row in data['rows']:
        s=row['sensors']['scan'];h=s['header']
        scan=PlainLaserScan(ranges=tuple(float(v) for v in s['ranges']),
            **{k:s[k] for k in ('angle_min','angle_max','angle_increment','range_min','range_max')},
            scan_frame_id=h['frame_id'],scan_stamp_sec=h['stamp_sec']+h['stamp_nanosec']/1e9,
            receipt_sec=row['scan_received_ros_sec'],scan_topology_profile='full_rotation')
        t=next(t for t in row['tf_samples'] if t['target_frame']=='base_scan' and t['source_frame']=='map')
        tf=RigidTransform('base_scan','map',tuple(t['translation_xyz_m']),tuple(t['rotation_xyzw']))
        point=transform_point((g.x_m,g.y_m,0.),tf);dist=math.hypot(*point[:2])
        pose=next(t for t in row['tf_samples'] if t['target_frame']=='map' and t['source_frame']=='base_footprint')
        q=pose['rotation_xyzw']
        yield dict(snapshot_path=root/'candidate_snapshot.json',candidate_uid='survey_candidate_0005',
            planning_frame='map',stand_center=(g.x_m,g.y_m),target_key='recorded-target',epoch=0,
            scan=scan,scan_from_map=tf,robot_pose=(*pose['translation_xyz_m'][:2],2*math.atan2(q[2],q[3])),
            image_stamp_sec=row['image_stamp_sec'],now_sec=row['image_received_ros_sec']+row['selected_monotonic_sec']-row['image_received_monotonic_sec'],
            options=dict(map_bearing_rad=math.atan2(point[1],point[0]),cone_half_angle_rad=math.radians(3),
                         max_camera_map_bearing_delta_rad=math.radians(12),accepted_range_m=(dist-2*g.radius_m-g.uncertainty_m-.04,dist+.04)),
            position_epoch_path=root/'candidate_frame_projection.json')


def recorded_proof():
    tracker=StoppedTargetReconciliation()
    results=[tracker.observe(**row) for row in recorded_rows()]
    assert results[:2]==[None,None]
    assert results[-1] is not None, tracker.metadata
    return results[-1]


def test_recorded_epoch_recovers_unique_current_cluster_without_changing_landmark():
    proof=recorded_proof()
    assert is_epoch_recovery(proof)
    _,envelope,point,bearing=validate_reconciliation(proof)
    assert envelope.selected_cluster_sample_count>=3
    assert 18 < math.degrees(bearing) < 24
    assert .20 < math.dist(point,proof['stand_center']) < .26
    assert proof['candidate_geometry_updated'] is False
    assert proof['motion_authorized'] is False


def test_normal_gate_stays_closed_without_epoch_context():
    tracker=StoppedTargetReconciliation()
    for row in recorded_rows():
        row.pop('position_epoch_path')
        assert tracker.observe(**row) is None


@pytest.mark.parametrize('defect',['stale','moved','duplicate','epoch_hash','ray','missing_sample'])
def test_recovery_rejects_invalid_evidence(defect):
    proof=recorded_proof();entry=proof['entries'][-1]
    if defect=='stale':entry['checked_at_sec']+=1
    if defect=='moved':entry['robot_pose'][0]+=.1
    if defect=='duplicate':entry['image_stamp_sec']=proof['entries'][0]['image_stamp_sec']
    if defect=='epoch_hash':entry['position_epoch']['sha256']='0'*64
    if defect=='ray':entry['options']['map_bearing_rad']+=.1
    if defect=='missing_sample':proof['entries'].pop()
    with pytest.raises(ValueError):validate_reconciliation(proof)


def camera_inputs(root=ROOT):
    from types import SimpleNamespace
    from scripts.aufgabe04.real_robot.configuration.geometry import intrinsics_from_camera_info
    data=json.loads((root/'observations.json').read_text())
    row=data['rows'][-1]
    ci=dict(row['sensors']['camera_info']);ci['header']=SimpleNamespace(**ci['header'])
    def tf(parent,child):
        t=next(t for t in row['tf_samples'] if t['target_frame']==parent and t['source_frame']==child)
        return RigidTransform(parent,child,tuple(t['translation_xyz_m']),tuple(t['rotation_xyzw']))
    corners=data['viewer_corners']
    center=tuple(sum(p[k] for p in corners)/4 for k in ('u_px','v_px'))
    return intrinsics_from_camera_info(SimpleNamespace(**ci)),tf,center


def recovery_advisory():
    """Recorded run scan/calibration plus the later same-framing viewer head center.

    This tests centering geometry, not a QR decode or a simultaneous head fit.
    """
    from types import SimpleNamespace
    from scripts.aufgabe04.real_robot.observer.candidate_centering import build_camera_centering_advisory
    from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
    from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
    from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
    proof=recorded_proof();r=list(recorded_rows())[-1];intrinsics,tf,center=camera_inputs()
    _,env,_,reference=validate_reconciliation(proof)
    bearing,_,_=finite_target_bearing(center_px=center,intrinsics=intrinsics,scan_from_camera=tf('base_scan','camera'),
        distance_m=env.distance_m,range_interval_m=env.accepted_range_m)
    lidar=associate_camera_registered_candidate_lidar_target(r['scan'],map_bearing_rad=reference,
        observed_camera_bearing_rad=bearing,cone_half_angle_rad=math.radians(3),accepted_range_m=env.accepted_range_m,
        now_sec=r['now_sec'],max_scan_age_sec=.5,max_camera_map_bearing_delta_rad=math.radians(3))
    assert lidar.associated
    association=SimpleNamespace(accepted=True,lidar_association=lidar,head_admission=SimpleNamespace(accepted=True),
        head_orientation_bounds=None,full_image_center_px=center,target_reconciliation=proof)
    odom=tf('odom','base_footprint');q=odom.rotation_xyzw
    advice=build_camera_centering_advisory(association=association,intrinsics=intrinsics,scan_from_camera=tf('base_scan','camera'),
        base_from_camera=tf('base_footprint','camera'),candidate_uid=r['candidate_uid'],target_key=r['target_key'],stream_id='fixture',planning_frame='map',motion_epoch=0,
        anchor_pose=EvidencePose(*r['robot_pose']),anchor_odom_pose=EvidencePose(*odom.translation_xyz_m[:2],2*math.atan2(q[2],q[3])),odom_stamp_sec=r['image_stamp_sec'],
        image_stamp_sec=r['image_stamp_sec'],now_sec=r['now_sec'],robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='c'*64)
    assert advice is not None
    return advice, lidar.search_association


def test_recorded_arrival_produces_separate_scan_safe_coarse_advice():
    from dataclasses import replace
    from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory
    from scripts.aufgabe04.real_robot.observer.inspection_framing import review_centering_destination
    advice,search=recovery_advisory()
    assert 21 < math.degrees(advice.required_yaw_rad) < 24
    assert 6 < math.degrees(advice.requested_yaw_rad) < math.degrees(advice.required_yaw_rad)
    assert review_centering_destination(advice,search_association=search).allowed
    assert not review_centering_destination(replace(advice,requested_yaw_rad=advice.required_yaw_rad),search_association=search).allowed
    validate_camera_centering_advisory(json.loads(json.dumps(advice.metadata())))
    # Rehashing an ordinary receipt cannot grant the extended budget.
    with pytest.raises(ValueError):
        validate_camera_centering_advisory(replace(advice,target_reconciliation=None).metadata())


def test_epoch_qr_binds_current_ray_and_rejects_a_changed_scan():
    from dataclasses import replace
    from scripts.aufgabe04.real_robot.observer.qr_target_binding import bind_qr_observations_to_target
    from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
    from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
    proof=recorded_proof();r=list(recorded_rows())[-1];intrinsics,tf,center=camera_inputs()
    # Synthetic complete symbol at the measured head center: association test,
    # not a claim that the recorded run decoded this identity.
    u,v=center;quad=((u-12,v-12),(u+12,v-12),(u+12,v+12),(u-12,v+12))
    obs=(DecodedQrObservation('QR_004',quad,'synthetic_binding_test'),)
    options=dict(roi=ImageRoi(0,0,800,600,100),intrinsics=intrinsics,scan_from_camera=tf('base_scan','camera'),
        scan=r['scan'],now_sec=r['now_sec'],max_scan_age_sec=.5,min_cluster_sample_count=1,
        camera_registration_accepted=False,allow_independent_registration=True,**r['options'])
    assert not bind_qr_observations_to_target(obs,**options).accepted
    result=bind_qr_observations_to_target(obs,**options,target_reconciliation=proof)
    assert result.accepted,result.reason
    assert result.qr_texts_for_evidence==('QR_004',)
    from scripts.aufgabe04.artifacts.qr_verified_observation_pose import build_qr_verified_observation_pose, validate_qr_verified_observation_pose, SOURCE_GATES
    receipt=build_qr_verified_observation_pose(candidate_uid=r['candidate_uid'],stream_id='fixture',planning_frame='map',qr_id='QR_004',
        stand_center=dict(zip(('x_m','y_m'),r['stand_center'])),robot_pose=dict(zip(('x_m','y_m','yaw_rad'),r['robot_pose'])),
        sensor_stamp_sec=r['image_stamp_sec'],scan_stamp_sec=r['scan'].scan_stamp_sec,checked_at_sec=r['now_sec'],
        robot_profile_sha256='a'*64,calibration_profile_sha256='b'*64,stand_model_profile_sha256='c'*64,
        target_key=r['target_key'],motion_epoch=0,camera_signature=(intrinsics.fx_px,intrinsics.fy_px,intrinsics.cx_px,intrinsics.cy_px),
        qr_corners_px=quad,image_shape=(600,800),qr_binding=result.metadata(),source_gates={k:True for k in SOURCE_GATES},
        localization_provenance=dict(map_frame='map',base_frame='base_footprint',scan_frame='base_scan',camera_frame='camera',
            exact_image_transform_stamp_sec=r['image_stamp_sec'],exact_scan_transform_stamp_sec=r['scan'].scan_stamp_sec))
    assert validate_qr_verified_observation_pose(json.loads(json.dumps(receipt)))['qr_id']=='QR_004'

    altered=list(r['scan'].ranges);altered[13]+=.02
    options['scan']=replace(r['scan'],ranges=tuple(altered))
    rejected=bind_qr_observations_to_target(obs,**options,target_reconciliation=proof)
    assert not rejected.accepted
    assert 'differs from current' in rejected.reason


def test_saved_run_image_acquires_visible_blue_head_in_recovered_search():
    """Late run image at the same stopped pose; not a synchronized tuple replay."""
    cv2=pytest.importorskip('cv2')
    from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
    from scripts.aufgabe04.real_robot.configuration.geometry import project_optical_point
    from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import recovered_search
    from scripts.aufgabe04.real_robot.observer.viewer_head_acquisition import evaluate_viewer_head
    from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionPolicy
    from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache
    from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
    r=list(recorded_rows())[-1];proof=recorded_proof();intrinsics,tf,_=camera_inputs()
    profile=load_measured_physical_stand_model(Path('configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json'))
    height=profile.head_top_height_m-profile.head_height_m/2
    point=transform_point((*r['stand_center'],height),tf('camera','map'))
    projection=project_optical_point(point,intrinsics,physical_size_m=profile.head_height_m)
    hint,_=recovered_search(proof,scan=r['scan'],original_projection=projection,camera_from_map=tf('camera','map'),intrinsics=intrinsics,model_profile=profile)
    assert projection.u_px > 400
    assert 70 < hint.projection.u_px < 130
    image=cv2.imread(str(ROOT/'late_run_frame.png'))
    assert image is not None
    budget=QrAcquisitionPolicy().begin_frame(target_key='replay',image_stamp_sec=r['image_stamp_sec'],
        started_ros_sec=r['now_sec'],started_monotonic_sec=10.,max_sensor_age_sec=.5)
    result=evaluate_viewer_head(cv2,image,model_profile=profile,intrinsics=intrinsics,pose_hint=None,
        projection=hint.projection,expected_head_height_px=projection.expected_size_px,
        fallback_attempt=None,cache=RoiQrDecodeCache(),budget=budget,
        native_decoder=lambda crop:(),full_decoder=lambda *args:(),deadline_monotonic_sec=None,now=lambda:10.,
        depth_uncertainty_m=.08,position_uncertainty_m=.08,
        nearest_context=dict(scan=r['scan'],image_stamp_sec=r['image_stamp_sec'],now_sec=r['now_sec'],max_scan_age_sec=.5,
            scan_from_camera=tf('base_scan','camera'),base_from_camera=tf('base_footprint','camera'),
            accepted_range_m=r['options']['accepted_range_m'],sync_tolerance_sec=.1))
    assert result.estimate.corners is not None,result.estimate.reason
    center=tuple(sum(getattr(p,k) for p in result.estimate.corners)/4 for k in ('u_px','v_px'))
    assert 80 < center[0] < 130  # blue stand, not the radiator at x~524
    assoc=associate_current_measured_head(estimate=result.estimate,debug=result.debug,attempt=result.attempt,
        projection=projection,expected_head_height_px=projection.expected_size_px,profile_sha256=profile.sha256,
        intrinsics=intrinsics,scan_from_camera=tf('base_scan','camera'),scan=r['scan'],now_sec=r['now_sec'],
        max_scan_age_sec=.5,min_cluster_sample_count=1,max_center_offset_ratio=1.5,
        search_reconciliation=hint,target_reconciliation=proof,**r['options'])
    assert assoc.accepted,assoc.reason


def test_competing_candidate_blocks_epoch_recovery(tmp_path):
    import copy
    from dataclasses import replace
    from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot, candidate_snapshot_sha256
    from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
    snapshot=load_candidate_snapshot(ROOT/'candidate_snapshot.json')
    target=snapshot.candidate_for('survey_candidate_0005')
    neighbor=snapshot.candidate_for('survey_candidate_0004')
    changed=replace(neighbor,geometry=replace(neighbor.geometry,x_m=target.geometry.x_m,y_m=target.geometry.y_m))
    snapshot=replace(snapshot,candidates=tuple(changed if c.candidate_uid==neighbor.candidate_uid else c for c in snapshot.candidates))
    path=tmp_path/'snapshot.json';write_candidate_snapshot(path,snapshot)
    epoch=json.loads((ROOT/'candidate_frame_projection.json').read_text());epoch.pop('candidate_frame_projection_sha256')
    epoch['candidate_reprojections'][neighbor.candidate_uid]=copy.deepcopy(epoch['candidate_reprojections'][target.candidate_uid])
    epoch['projected_candidate_snapshot_sha256']=candidate_snapshot_sha256(snapshot)
    projection=tmp_path/'projection.json';projection.write_text(json.dumps(content_hashed_payload(epoch,hash_field='candidate_frame_projection_sha256')))
    tracker=StoppedTargetReconciliation()
    for row in recorded_rows():
        assert tracker.observe(**{**row,'snapshot_path':path,'position_epoch_path':projection}) is None
    assert 'another candidate' in tracker.metadata['reason']


def shifted_proof(seconds):
    proof=recorded_proof()
    for e in proof['entries']:
        for field in ('image_stamp_sec','checked_at_sec'):e[field]+=seconds
        for field in ('scan_stamp_sec','receipt_sec'):e['scan'][field]+=seconds
    return proof


def test_position_opportunity_is_bounded_and_never_authorizes_motion():
    from scripts.aufgabe04.real_robot.observer.position_epoch_opportunity import PositionEpochOpportunity
    tracker=PositionEpochOpportunity()
    for seconds in range(6):
        proof=shifted_proof(seconds);e=proof['entries'][-1]
        frame=dict(frame_stamp_sec=e['image_stamp_sec'],scan_stamp_sec=e['scan']['scan_stamp_sec'],
            frame_accepted=False,poisoned=False,motion_epoch_reset=False)
        result=tracker.observe(proof,frame,now_sec=e['checked_at_sec'])
        if seconds<5:assert result is None
    assert result['elapsed_sec']==5.
    assert result['motion_authorized'] is False
    assert result['recovery']=='bounded_inspection_view'
    assert tracker.observe(proof,{**frame,'frame_accepted':True},now_sec=e['checked_at_sec']) is None
    assert tracker.start is None
    assert tracker.observe(proof,frame,now_sec=e['checked_at_sec']+1) is None
    assert tracker.start is None


def test_live_turn_preflight_uses_calibrated_current_cluster():
    from types import SimpleNamespace
    from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.candidate_centering import fresh_centering_target
    advice,_=recovery_advisory();row=list(recorded_rows())[-1];s=row['scan']
    stamp=s.scan_stamp_sec+.1
    scan=SimpleNamespace(**{k:getattr(s,k) for k in ('ranges','angle_min','angle_max','angle_increment','range_min','range_max')},
        header=SimpleNamespace(frame_id=s.scan_frame_id,stamp=SimpleNamespace(sec=int(stamp),nanosec=int((stamp-int(stamp))*1e9))))
    result=fresh_centering_target(scan,advice.metadata(),row['now_sec']+.1)
    assert result is not None
    assert set(result['selected_cluster_source_indices']).issubset({11,12,13,14,15})
