"""Original beam evidence, fixed head capture and bounded sampling recovery."""
from dataclasses import replace
import copy
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json, load_content_hashed_json
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import _scan_surface_analysis, fit_current_lidar_view
from scripts.aufgabe04.perception.lidar_scan_metadata import LidarScanMetadata, metadata_from_message
from scripts.aufgabe04.perception.lidar_visibility_evidence import _receipt_from_hashed_payload
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import head_capture_payload, HASH_FIELD
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import CandidateLidarCaptureRequest, capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_sampling import (
    build_sampling_advisory, validate_sampling_advisory, select_sampling_yaw,
    fresh_sampling_target, predicted_head_support, MAX_TOTAL_TRAVEL_RAD,
)
from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot, candidate_snapshot_sha256
from tests.aufgabe04.test_tour_scan_capture import sample, stamp
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture, line_receipts
from tests.aufgabe04.test_coverage_visibility_reporting import _plan
from tests.aufgabe04.test_candidate_centering_child import turn_request, runner, motion_result
from scripts.aufgabe04.real_robot.execution import candidate_centering as child


def head_scans(*, circular_gap=2, spacing=.1):
    count=220
    step=math.tau/(count-1+circular_gap)
    ranges=[]
    for i in range(count):
        angle=i*step
        distance=.4/math.cos(angle) if math.cos(angle)>0 else -1.
        ranges.append(distance if distance>0 and abs(distance*math.sin(angle))<=.039 else None)
    scans=[]
    for i in range(8):
        record=sample(100.+i*spacing)
        record.update(angle_min=0., angle_increment=step, range_min=.05, range_max=8., ranges=list(ranges),
            base_pose_odom=dict(x_m=-.44,y_m=0.,yaw_rad=0.),
            scan_pose_odom=dict(x_m=-.4,y_m=0.,yaw_rad=0.))
        record['scan_metadata']=LidarScanMetadata((count-1)*step, 0., .1, 'full_rotation', (),
            tuple('nan' if r is None else None for r in ranges)).to_mapping()
        scans.append(record)
    return scans


def capture_payload(scans):
    return head_capture_payload(scans,tour_id='local_head',odom_frame='odom',base_frame='base_footprint',
        scan_frame='base_scan',captured_at_unix_sec=scans[-1]['stamp_sec']+.01)


def source_view(tmp_path, *, circular_gap=2):
    snapshot,registry,frame=hint_fixture()
    snapshot_path=tmp_path/'snapshot.json'
    write_candidate_snapshot(snapshot_path,snapshot)
    plan=replace(_plan(),survey_id=registry.survey_id)
    raw=capture_payload(head_scans(circular_gap=circular_gap))
    capture_path=tmp_path/'capture.json'
    write_content_hashed_json(capture_path,raw,hash_field=HASH_FIELD)
    request=CandidateLidarCaptureRequest(plan,candidate_snapshot_sha256(snapshot),'candidate_1',
        'local_head',tmp_path,100.,frame,'base_footprint','base_scan','/scan','full_rotation')
    clock=iter((100.,100.72))
    view=capture_candidate_lidar_view(request,capture_cohort=lambda _:capture_path,clock=lambda:next(clock))
    return view,snapshot_path,raw


def advice(tmp_path, profile):
    view,snapshot_path,_=source_view(tmp_path)
    from scripts.aufgabe04.real_robot.configuration.profile import real_robot_profile_sha256
    payload=build_sampling_advisory(source_view_path=view.evidence_path,snapshot_path=snapshot_path,
        session_id='sampling_session',robot_profile_sha256=real_robot_profile_sha256(profile),
        calibration_profile_sha256=profile.calibration_profile_sha256,base_frame='base_footprint',
        scan_frame='base_scan',now_sec=100.72)
    return payload,view


def test_original_invalid_ranges_timing_and_intensities_are_preserved():
    message=NS(ranges=[.5,math.nan,math.inf,-math.inf,0.,.01,9.],range_min=.05,range_max=8.,
        intensities=[1.,math.nan,3.,4.,5.,6.,7.],angle_max=6.25,time_increment=.0004,scan_time=.1)
    metadata=metadata_from_message(message,topology_profile='full_rotation')
    metadata.validate([.5,None,None,None,None,None,None])
    assert metadata.invalid_range_reasons==(None,'nan','positive_infinity','negative_infinity','zero','below_range_min','above_range_max')
    assert metadata.intensities==(1.,None,3.,4.,5.,6.,7.)
    assert LidarScanMetadata.from_mapping(metadata.to_mapping())==metadata
    assert metadata.angle_max_rad==message.angle_max and metadata.time_increment_sec==.0004
    with pytest.raises(ValueError): metadata.validate([.5]*7)


def test_eight_scans_accept_one_point_four_second_window_without_relaxing_motion_gates():
    scans=head_scans(spacing=.2)
    assert capture_payload(scans)['scan_count']==8
    mutations=[lambda s:s.pop(),lambda s:s[-1]['base_pose_odom'].update(x_m=-.42),
        lambda s:s[0].update(scan_pose_stamp_sec=99.),lambda s:s[-1]['scan_metadata']['invalid_range_reasons'].pop(),
        lambda s:s[0].update(received_at_unix_sec=100.3)]
    for mutate in mutations:
        broken=copy.deepcopy(scans);mutate(broken)
        with pytest.raises(ValueError): capture_payload(broken)
    with pytest.raises(ValueError,match='window'): capture_payload(head_scans(spacing=.22))


def test_production_capture_propagates_schema_three_metadata_and_hash_binding(tmp_path):
    view,_,raw=source_view(tmp_path)
    assert len(view.receipts)==8
    for receipt,scan in zip(view.receipts,raw['scans']):
        assert receipt.schema_version==3
        assert receipt.scan_metadata.to_mapping()==scan['scan_metadata']
        assert _receipt_from_hashed_payload(receipt.to_evidence_dict())==receipt
    bad=copy.deepcopy(view.receipts[0].to_evidence_dict())
    bad['scan_metadata']['angle_max_rad']-=.01
    with pytest.raises(ValueError): _receipt_from_hashed_payload(bad)


@pytest.mark.parametrize('gap,joined',[(1,True),(2,False),(2.69,False)])
def test_only_original_one_step_seam_can_be_joined(tmp_path,gap,joined):
    view,_,_=source_view(tmp_path,circular_gap=gap)
    surface,diag=_scan_surface_analysis(view.receipts[0],Pose2D(0.,0.),.08,[])
    assert diag.get('seam_joined',False)==joined
    assert (surface is not None)==joined
    legacy=replace(view.receipts[0],schema_version=2,scan_metadata=None)
    assert _scan_surface_analysis(legacy,Pose2D(0.,0.),.08,[])[0] is None
    linear=replace(view.receipts[0],scan_metadata=replace(view.receipts[0].scan_metadata,scan_topology_profile='linear'))
    assert _scan_surface_analysis(linear,Pose2D(0.,0.),.08,[])[0] is None


def test_internal_missing_bin_is_not_filled_by_seam_join(tmp_path):
    view,_,_=source_view(tmp_path,circular_gap=1)
    receipt=view.receipts[0]
    ranges=list(receipt.ranges_m);ranges[-2]=None
    reasons=list(receipt.scan_metadata.invalid_range_reasons);reasons[-2]='nan'
    receipt=replace(receipt,ranges_m=tuple(ranges),scan_metadata=replace(receipt.scan_metadata,invalid_range_reasons=tuple(reasons)))
    fit,diag=_scan_surface_analysis(receipt,Pose2D(0.,0.),.08,[])
    assert fit is None and diag['reason']=='fragmented_candidate_returns'
    assert 218 not in diag['candidate_indices'] and 217 in diag['candidate_indices'] and 219 in diag['candidate_indices']


def test_eight_scan_fraction_keeps_failures_in_denominator():
    snapshot,registry,frame=hint_fixture()
    scans=tuple(replace(line_receipts()[i%4],scan_stamp_sec=100.+i*.1,pose_stamp_sec=100.+i*.1,
        observer_clock_sec=100.+i*.1+.01) for i in range(8))
    def with_failures(count):
        return tuple(replace(r,ranges_m=(None,)*len(r.ranges_m)) if i<count else r for i,r in enumerate(scans))
    arguments=dict(snapshot=snapshot,registry=registry,planning_frame=frame,candidate_uid='candidate_1')
    fit=fit_current_lidar_view(**arguments,receipts=with_failures(2))
    assert fit is not None and fit.evidence['scan_count']==6
    assert len(fit.evidence['examined_receipt_sha256s'])==8
    assert fit_current_lidar_view(**arguments,receipts=with_failures(3)) is None


def test_sampling_turn_is_derived_from_original_geometry_and_not_head_normal(tmp_path,turn_request):
    payload,view=advice(tmp_path,turn_request.profile)
    validated=validate_sampling_advisory(payload,now_sec=100.8)
    assert math.radians(15)<=abs(validated.requested_yaw_rad)<=math.radians(25)
    assert not payload['head_alignment_verified'] and not payload['camera_centered']
    assert not payload['stand_axis_authorized'] and not payload['motion_authorized']
    for key,value in [('requested_yaw_rad',.001),('target_center_odom',dict(x_m=.1,y_m=0.)),('source_view_sha256','f'*64)]:
        with pytest.raises(ValueError): validate_sampling_advisory({**payload,key:value})
    with pytest.raises(ValueError,match='expired'): validate_sampling_advisory(payload,now_sec=106.)
    center=Pose2D(0.,0.);base=Pose2D(-.44,0.,validated.requested_yaw_rad)
    assert select_sampling_yaw(base=base,scan_pose_robot=Pose2D(.04,0.),center=center,radius=.08,receipts=view.receipts) is None
    assert select_sampling_yaw(base=Pose2D(-.44,0.),scan_pose_robot=Pose2D(.04,0.),center=center,radius=.18,receipts=view.receipts) is None


def test_fresh_sampling_target_rejects_sparse_competing_or_stale_support(tmp_path,turn_request):
    payload,view=advice(tmp_path,turn_request.profile)
    r=view.receipts[-1]
    scan=NS(header=NS(frame_id='base_scan',stamp=stamp(100.8)),ranges=list(r.ranges_m),
        range_min=r.range_min_m,range_max=r.range_max_m,angle_min=r.angle_min_rad,angle_increment=r.angle_increment_rad)
    scan.ranges=[math.nan if x is None else x for x in scan.ranges]
    assert fresh_sampling_target(scan,payload,100.82)['real_return_count']==6
    assert fresh_sampling_target(scan,payload,101.5) is None
    assert fresh_sampling_target(scan,{**payload,'competing_candidate_envelopes':[[0.,0.,.08]]},100.82) is None
    scan.ranges=[.4]+[math.nan]*219
    assert fresh_sampling_target(scan,payload,100.82) is None


def test_support_prediction_prefers_broad_face_and_close_range():
    def predict(distance,angle):return predicted_head_support(distance_m=distance,incidence_rad=angle,angular_step_rad=math.radians(1.67478))
    assert predict(.55,0.)['minimum_phase_return_count']==4
    assert predict(.55,math.radians(60))['minimum_phase_return_count']==2
    assert predict(.65,0.)['expected_return_count']<predict(.55,0.)['expected_return_count']
    assert predict(.55,0.)['physical_success_probability'] is None


def sampling_request(tmp_path,turn_request,monkeypatch):
    from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import load_mission_leg_motion_authorization,write_mission_leg_motion_authorization
    payload,_=advice(tmp_path,turn_request.profile)
    master=load_mission_leg_motion_authorization(turn_request.master_authorization_path)
    # The fixture's authorization already contains the new scope.
    new_master=tmp_path/'sampling_master.json'
    write_mission_leg_motion_authorization(new_master,replace(master,session_id='sampling_session'))
    monkeypatch.setattr(child.time,'time',lambda:100.8)
    return replace(turn_request,session_id='sampling_session',master_authorization_path=new_master,
        candidate_id='candidate_1',view_id='sampling_session:candidate_1:lidar_sampling',turn_index=0,
        advisory=payload,signed_turn_rad=payload['requested_yaw_rad'],remaining_travel_rad=MAX_TOTAL_TRAVEL_RAD,
        purpose='candidate_lidar_sampling')


def test_sampling_child_binds_purpose_and_prevents_new_view_budget_reset(tmp_path,turn_request,monkeypatch):
    request=sampling_request(tmp_path,turn_request,monkeypatch)
    motion=Mock(side_effect=motion_result)
    outcome=child.run_candidate_centering_child(request,run_process=runner(monkeypatch,motion))
    assert outcome.result['purpose']=='candidate_lidar_sampling' and outcome.result['status']=='completed'
    with pytest.raises(ValueError,match='one turn per candidate'):
        child.run_candidate_centering_child(replace(request,view_id='different',output_dir=tmp_path/'second'),run_process=runner(monkeypatch,motion))
    with pytest.raises(RuntimeError,match="child failed"):
        child.run_candidate_centering_child(replace(request,output_dir=tmp_path/'second'),run_process=runner(monkeypatch,motion))
    assert motion.call_count==1


def test_old_run_scope_does_not_authorize_sampling(tmp_path,turn_request,monkeypatch):
    from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import LEGACY_OPPOSITE_CHECKPOINT_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,load_mission_leg_motion_authorization,write_mission_leg_motion_authorization
    request=sampling_request(tmp_path,turn_request,monkeypatch)
    master=load_mission_leg_motion_authorization(request.master_authorization_path)
    legacy=tmp_path/'legacy.json'
    write_mission_leg_motion_authorization(legacy,replace(master,scope_text=LEGACY_OPPOSITE_CHECKPOINT_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE))
    process=Mock()
    with pytest.raises(ValueError,match='authorize LiDAR sampling'):
        child.run_candidate_centering_child(replace(request,master_authorization_path=legacy),run_process=process)
    process.assert_not_called()


def test_sampling_recovery_yields_after_one_move_and_does_not_reset_budget():
    from scripts.aufgabe04.real_robot.candidate.lidar_acquisition import create_bounded_lidar_recovery
    from tests.aufgabe04.test_candidate_lidar_acquisition import frame_fixture
    frame,_=frame_fixture()
    events=[]
    def observe(frame,index,checkpoint):
        events.append('observe')
        return frame,None,None,dict(boundary_fragmentation_detected=True,head_alignment_verified=False,camera_centered_verified=False)
    def move(kind,frame):
        events.append(kind);return frame
    recovery=create_bounded_lidar_recovery(observe=observe,persist=lambda _:None,
        move_probe=lambda frame,*_:move('probe',frame),move_aligned=lambda frame,*_:move('align',frame),
        move_sampling=lambda frame,*_:move('sample',frame),monotonic=lambda:0.)
    frame,report,_=recovery(frame)
    assert events==['observe','sample','observe'] and report['motion_completed']
    assert report['sampling_moves_attempted']==1 and report['probe_moves_attempted']==0
    frame,report,_=recovery(frame)
    assert events==['observe','sample','observe','observe','probe','observe']
    assert report['sampling_moves_attempted']==1 and not report['head_alignment_verified']


def test_sampling_runtime_uses_same_stopped_translation_and_safety_limits(tmp_path,turn_request,monkeypatch):
    from tests.aufgabe04.test_candidate_centering_runtime import harness
    from scripts.aufgabe04.real_robot.candidate import lidar_sampling
    request=sampling_request(tmp_path,turn_request,monkeypatch)
    permit=child.build_candidate_centering_permit(request)
    monkeypatch.setattr(lidar_sampling,'fresh_sampling_target',lambda *args,**kwargs:{'head_alignment_verified':False})
    node,commands,events,state=harness(permit,monkeypatch)
    result=node.run_candidate_centering(permit)
    assert result['status']=='completed' and abs(result['actual_angular_travel_rad'])<=MAX_TOTAL_TRAVEL_RAD
    assert commands and all(c.linear_x_mps==0. and abs(c.angular_z_radps)<=.06 for c in commands)
    assert result['zero_command_count']>=20 and result['stationary_odom']['accepted']
    node,commands,events,state=harness(permit,monkeypatch)
    service=node._service_or_wait_for_callbacks
    def drift(dt):
        service(dt)
        if commands:state['pose']=replace(state['pose'],x_m=state['pose'].x_m+.02)
    node._service_or_wait_for_callbacks=drift
    result=node.run_candidate_centering(permit)
    assert result['status']=='stopped' and 'translation' in result['stop_reason']
    assert len(commands)==1 and events[-1]=='zero'


def test_runtime_capture_selects_eight_scan_adapter_and_declared_topology(tmp_path,monkeypatch):
    from scripts.aufgabe04.real_robot.autonomous_runner import runtime
    from scripts.aufgabe04.real_robot.candidate import lidar_head_capture
    from scripts.aufgabe04.real_robot.candidate import lidar_acquisition_capture
    request=CandidateLidarCaptureRequest(replace(_plan(),survey_id='survey'),'f'*64,'candidate_1','local_head',
        tmp_path,100.,hint_fixture()[2],'base_footprint','base_scan','/scan')
    profile=object()
    capture=Mock(return_value=tmp_path/'capture.json')
    monkeypatch.setattr(lidar_head_capture,'capture_head_scan',capture)
    def adapter(request,*,capture_cohort):
        assert request.scan_topology_profile=='full_rotation'
        return capture_cohort(request)
    monkeypatch.setattr(lidar_acquisition_capture,'capture_candidate_lidar_view',adapter)
    assert runtime._capture_candidate_lidar_view(profile=profile,request=request,topology_profile='full_rotation')==tmp_path/'capture.json'
    assert capture.call_args.kwargs['topology_profile']=='full_rotation'
    assert capture.call_args.kwargs['observation_not_before_sec']==100.


def test_rehashed_source_changes_cannot_become_sampling_advice(tmp_path,turn_request):
    from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import HASH_FIELD as VIEW_HASH
    payload,view=advice(tmp_path,turn_request.profile)
    raw=load_content_hashed_json(view.capture_path,hash_field=HASH_FIELD)
    raw['scans'][0]['scan_metadata']['angle_max_rad']-=.03
    view.capture_path.unlink()
    write_content_hashed_json(view.capture_path,raw,hash_field=HASH_FIELD)
    with pytest.raises(ValueError,match='capture binding'):validate_sampling_advisory(payload)
    # Rehashing the wrapper as well cannot change the already-bound advisory.
    wrapper=load_content_hashed_json(view.evidence_path,hash_field=VIEW_HASH)
    wrapper['candidate_uid']='another_candidate'
    view.evidence_path.unlink()
    write_content_hashed_json(view.evidence_path,wrapper,hash_field=VIEW_HASH)
    with pytest.raises(ValueError):validate_sampling_advisory(payload)
