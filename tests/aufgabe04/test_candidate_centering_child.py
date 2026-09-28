"""Recorded centering receipt crosses the real parent/child contract without ROS."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.real_robot.execution import candidate_centering as child
from scripts.aufgabe04.navigation.execution.candidate_centering_permit import (
    load_candidate_centering_permit, load_candidate_centering_result, write_candidate_centering_permit,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    LEGACY_CENTERING_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    RECOVERABLE_MISSION_LEG_KINDS, load_mission_leg_motion_authorization,
    MissionLegMotionAuthorization, ROUTINE_MISSION_LEG_KINDS,
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, write_mission_leg_motion_authorization,
)
from scripts.aufgabe04.real_robot.configuration.profile import load_real_robot_profile
from scripts.aufgabe04.real_robot.candidate.centering_execution import capture_with_centering

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def turn_request(tmp_path, monkeypatch):
    advisory = json.loads((Path(__file__).parent/'fixtures/candidate_centering/recorded_advisory.json').read_text())
    profile = load_real_robot_profile(ROOT/'configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json')
    monkeypatch.setattr(child.time, 'time', lambda: advisory['created_at_sec']+.1)
    session = advisory['stream_id'].removesuffix('_'+advisory['candidate_uid'])
    master = tmp_path/'master.json'
    write_mission_leg_motion_authorization(master, MissionLegMotionAuthorization(
        session_id=session, robot_id=profile.robot_id, namespace='', cmd_vel_topic='/cmd_vel',
        semantic_map_id='arena', localization_branch_proof_id='known-start',
        allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS, scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
        operator_confirmation='RUN'))
    return child.CandidateCenteringChildRequest(session, tmp_path/'turn', profile, master,
        advisory['candidate_uid'], 'inspection:0', 0, advisory, advisory['requested_yaw_rad'],
        math.radians(12), .20)


def motion_result(permit, **changes):
    Path(permit['controller_trace_path']).write_text('{"linear_x_mps":0.0}\n')
    return dict(status='completed', stop_reason='', translation_commanded=False, motion_published=True,
        actual_angular_travel_rad=abs(permit['signed_turn_rad']),
        total_angular_travel_rad=permit['previous_angular_travel_rad']+abs(permit['signed_turn_rad']),
        maximum_translation_m=0., stopped_at_sec=permit['advisory']['created_at_sec']+1.,
        final_yaw_error_rad=0., zero_command_count=20, stationary_odom={'accepted':True}, **changes)


def runner(monkeypatch, motion):
    monkeypatch.setattr('scripts.aufgabe04.navigation.waypoint_follower.runtime.run_candidate_centering_motion', motion)
    def run(command, check):
        assert Path(command[1]).name == 'run_candidate_centering.py'
        assert Path(command[1]).is_file()
        return SimpleNamespace(returncode=child.main(command[2:]))
    return run


def test_recorded_five_degree_request_completes_and_binds_exact_result(turn_request, monkeypatch):
    motion=Mock(side_effect=motion_result)
    outcome=child.run_candidate_centering_child(turn_request, run_process=runner(monkeypatch,motion))
    assert math.degrees(outcome.result['signed_turn_rad']) == pytest.approx(5.3823594107)
    assert outcome.result == load_candidate_centering_result(outcome.result_path, permit_path=outcome.permit_path)
    assert outcome.result['stationary_odom']['accepted']
    assert motion.call_count == 1
    with pytest.raises(RuntimeError, match='refusing to reuse'):
        child.run_candidate_centering_child(turn_request, run_process=runner(monkeypatch,motion))
    assert motion.call_count == 1


@pytest.mark.parametrize("legacy_scope", (
    LEGACY_CENTERING_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
))
def test_prior_centering_scope_remains_valid_without_return_authority(turn_request, monkeypatch, legacy_scope):
    master = load_mission_leg_motion_authorization(turn_request.master_authorization_path)
    legacy_path = turn_request.output_dir.parent / 'legacy-centering-master.json'
    write_mission_leg_motion_authorization(legacy_path, replace(
        master,
        allowed_leg_kinds=RECOVERABLE_MISSION_LEG_KINDS,
        scope_text=legacy_scope,
    ))
    request = replace(turn_request, master_authorization_path=legacy_path)
    motion = Mock(side_effect=motion_result)
    outcome = child.run_candidate_centering_child(request, run_process=runner(monkeypatch, motion))
    assert outcome.result['status'] == 'completed'
    assert motion.call_count == 1


def test_parent_runtime_adapter_reaches_real_child_boundary(turn_request, monkeypatch):
    from scripts.aufgabe04.real_robot.autonomous_runner import runtime
    advice=turn_request.output_dir.parent/'advice.json';advice.write_text(json.dumps(turn_request.advisory))
    motion=Mock(side_effect=motion_result)
    monkeypatch.setattr(child.subprocess, 'run', runner(monkeypatch,motion))
    x,y=map(float,turn_request.advisory['target_key'].split(':')[-2:])
    candidate=SimpleNamespace(candidate_uid=turn_request.candidate_id,geometry=SimpleNamespace(x_m=x,y_m=y))
    args=SimpleNamespace(session_id=turn_request.session_id,
        stand_model_profile=ROOT/'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json')
    outcome=runtime._run_camera_centering_turn(profile=turn_request.profile,args=args,
        master_authorization_path=turn_request.master_authorization_path,minimum_clearance_m=.2,
        candidate=candidate,advisory_path=advice,output_dir=turn_request.output_dir,
        view_id=turn_request.view_id,turn_index=0,remaining_travel_rad=turn_request.remaining_travel_rad,
        previous_result_path=None)
    assert outcome.result['status']=='completed' and motion.call_count==1


def test_copied_permit_and_new_outputs_cannot_replay_spent_turn(turn_request, monkeypatch, tmp_path):
    motion=Mock(side_effect=motion_result)
    outcome=child.run_candidate_centering_child(turn_request,run_process=runner(monkeypatch,motion))
    permit=load_candidate_centering_permit(outcome.permit_path)
    permit.update(result_path=str(tmp_path/'copy/result.json'),controller_trace_path=str(tmp_path/'copy/trace.jsonl'))
    copied=tmp_path/'copy/permit.json';write_candidate_centering_permit(copied,permit)
    with pytest.raises(FileExistsError):
        child.execute_candidate_centering_permit(copied,motion=motion)
    assert motion.call_count==1


@pytest.mark.parametrize('changes', [dict(session_id='another-session'),dict(signed_turn_rad=.2),
    dict(signed_turn_rad=-.05),dict(remaining_travel_rad=.02),dict(turn_index=2)])
def test_invalid_authority_or_turn_cannot_start_child(turn_request,changes):
    process=Mock()
    with pytest.raises(ValueError):
        child.run_candidate_centering_child(replace(turn_request,**changes),run_process=process)
    process.assert_not_called()


def test_stale_advisory_cannot_start_child(turn_request,monkeypatch):
    monkeypatch.setattr(child.time,'time',lambda:turn_request.advisory['created_at_sec']+6.)
    process=Mock()
    with pytest.raises(ValueError,match='stale'):
        child.run_candidate_centering_child(turn_request,run_process=process)
    process.assert_not_called()


@pytest.mark.parametrize('changes', [dict(status='stopped',stop_reason='obstacle'),
    dict(stationary_odom={'accepted':False}),dict(zero_command_count=0),
    dict(maximum_translation_m=.02),dict(final_yaw_error_rad=.02),
    dict(stopped_at_sec=1.),dict(motion_published=False)])
def test_rejected_or_unstopped_result_cannot_resume_observation(turn_request,monkeypatch,changes):
    def motion(permit): return {**motion_result(permit),**changes}
    with pytest.raises(RuntimeError):
        child.run_candidate_centering_child(turn_request,run_process=runner(monkeypatch,motion))


def test_child_exception_records_failure_and_consumes_attempt(turn_request,monkeypatch):
    motion=Mock(side_effect=RuntimeError('controller unavailable'))
    with pytest.raises(RuntimeError,match='child failed'):
        child.run_candidate_centering_child(turn_request,run_process=runner(monkeypatch,motion))
    failure=json.loads((turn_request.output_dir/'candidate_centering_failure.json').read_text())
    assert not failure['motion_continues_authorized']
    assert 'controller unavailable' in failure['reason']
    assert list(turn_request.master_authorization_path.parent.glob('candidate_centering_claims/*.json'))


def test_successful_turn_persists_revisions_and_requires_new_sensor_epoch(tmp_path):
    initial=SimpleNamespace(recommendation_path=None,centering_advisory_path=tmp_path/'advice.json')
    complete=SimpleNamespace(recommendation_path=tmp_path/'recommendation.json')
    calls=[]
    def capture(frame,root,index,enabled,remaining,not_before):
        calls.append((frame,not_before));return initial if len(calls)==1 else complete
    outcome=SimpleNamespace(result={'actual_angular_travel_rad':.09,'stopped_at_sec':12.},result_path=tmp_path/'result.json')
    args=dict(candidate_uid='candidate',frame='before',output_dir=tmp_path/'view',view_index=0,
        timeout_sec=90.,capture=capture,turn=lambda *args:('after',outcome),monotonic=lambda:10.)
    result,frame=capture_with_centering(**args)
    assert result is complete and frame=='after' and calls==[('before',None),('after',12.)]
    progress=json.loads((tmp_path/'view/centering_progress.json').read_text())
    assert progress['phase']=='observation_returned'
    assert len(list((tmp_path/'view/centering_history').glob('*.json')))==3
    with pytest.raises(RuntimeError,match='refusing to reset'):
        capture_with_centering(**args)


def test_failed_turn_records_progress_without_reobservation(tmp_path):
    capture=Mock(return_value=SimpleNamespace(recommendation_path=None,centering_advisory_path='advice'))
    with pytest.raises(RuntimeError,match='lost sensor'):
        capture_with_centering(candidate_uid='candidate',frame='frame',output_dir=tmp_path,
            view_index=0,timeout_sec=90.,capture=capture,
            turn=Mock(side_effect=RuntimeError('lost sensor')),monotonic=lambda:10.)
    assert capture.call_count==1
    assert json.loads((tmp_path/'centering_progress.json').read_text())['phase']=='turn_failed'


def test_second_turn_requires_fresh_observation_and_preserves_spent_travel(turn_request,monkeypatch):
    from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory
    run=runner(monkeypatch,motion_result)
    first=child.run_candidate_centering_child(turn_request,run_process=run)
    original=validate_camera_centering_advisory(turn_request.advisory)
    fresh=replace(original,created_at_sec=original.created_at_sec+3,
        image_stamp_sec=original.image_stamp_sec+3,scan_stamp_sec=original.scan_stamp_sec+3,
        odom_stamp_sec=original.odom_stamp_sec+3)
    monkeypatch.setattr(child.time,'time',lambda:fresh.created_at_sec+.1)
    second_request=replace(turn_request,turn_index=1,output_dir=turn_request.output_dir.parent/'second',
        previous_result_path=first.result_path,advisory=fresh.metadata(),
        remaining_travel_rad=turn_request.remaining_travel_rad-first.result['total_angular_travel_rad'])
    second=child.run_candidate_centering_child(second_request,run_process=run)
    assert second.result['total_angular_travel_rad']==pytest.approx(2*abs(turn_request.signed_turn_rad))
    assert second.result['total_angular_travel_rad']<math.radians(12)
    process=Mock()
    # Receipt is recently issued, but its image/scan belong to the pre-turn stop.
    stale_request=replace(second_request,output_dir=turn_request.output_dir.parent/'stale',
                          advisory=turn_request.advisory)
    with pytest.raises(ValueError,match='predates'):
        child.run_candidate_centering_child(stale_request,run_process=process)
    process.assert_not_called()
    with pytest.raises(ValueError,match='cumulative'):
        child.run_candidate_centering_child(replace(second_request,
            output_dir=turn_request.output_dir.parent/'extra',remaining_travel_rad=math.radians(12)),
            run_process=process)
    process.assert_not_called()


def test_two_turn_budget_disables_further_centering_but_allows_fresh_capture(tmp_path):
    calls=[]
    def capture(frame,root,index,enabled,remaining,not_before):
        calls.append((enabled,not_before))
        return SimpleNamespace(recommendation_path='complete' if len(calls)==3 else None,
                               centering_advisory_path='advice')
    def turn(frame,advice,root,index,remaining,previous):
        assert remaining==pytest.approx(math.radians(12)-index*.08)
        return frame,SimpleNamespace(result={'actual_angular_travel_rad':.08,'stopped_at_sec':12.+index},
                                     result_path=root/'result.json')
    capture_with_centering(candidate_uid='candidate',frame='frame',output_dir=tmp_path,
        view_index=0,timeout_sec=90.,capture=capture,turn=turn,monotonic=lambda:10.)
    assert calls==[(True,None),(True,12.),(False,13.)]
    assert len(list((tmp_path/'centering_history').glob('*.json')))==5


def epoch_turn_request(turn_request, monkeypatch):
    from tests.aufgabe04.test_candidate_position_epoch import recovery_advisory
    advice,_=recovery_advisory()
    original=turn_request.advisory
    advice=replace(advice,stream_id=f'{turn_request.session_id}_{advice.candidate_uid}',
        robot_profile_sha256=original['robot_profile_sha256'],
        calibration_profile_sha256=original['calibration_profile_sha256'],
        stand_model_profile_sha256=original['stand_model_profile_sha256'])
    monkeypatch.setattr(child.time,'time',lambda:advice.created_at_sec+.1)
    return replace(turn_request,candidate_id=advice.candidate_uid,advisory=advice.metadata(),
        signed_turn_rad=advice.requested_yaw_rad,remaining_travel_rad=math.radians(42))


def test_proven_epoch_arrival_crosses_real_child_boundary(turn_request,monkeypatch):
    request=epoch_turn_request(turn_request,monkeypatch)
    motion=Mock(side_effect=motion_result)
    outcome=child.run_candidate_centering_child(request,run_process=runner(monkeypatch,motion))
    assert math.degrees(outcome.result['signed_turn_rad']) > 6
    assert outcome.result['arrival_recovery'] is True
    assert outcome.result==load_candidate_centering_result(outcome.result_path)
    assert motion.call_count==1
    with pytest.raises(RuntimeError,match='refusing to reuse'):
        child.run_candidate_centering_child(request,run_process=runner(monkeypatch,motion))


def test_epoch_turn_requires_new_master_scope(turn_request,monkeypatch):
    from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import LEGACY_BOUNDED_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE
    request=epoch_turn_request(turn_request,monkeypatch)
    master=load_mission_leg_motion_authorization(request.master_authorization_path)
    legacy=request.master_authorization_path.with_name("legacy.json")
    write_mission_leg_motion_authorization(legacy,
        replace(master,scope_text=LEGACY_BOUNDED_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE))
    request=replace(request,master_authorization_path=legacy)
    motion=Mock(side_effect=motion_result)
    with pytest.raises(ValueError,match='does not authorize extended'):
        child.run_candidate_centering_child(request,run_process=runner(monkeypatch,motion))
    motion.assert_not_called()


def test_forged_epoch_allowance_cannot_cross_child_boundary(turn_request,monkeypatch):
    request=epoch_turn_request(turn_request,monkeypatch)
    request.advisory['target_reconciliation']['entries'][0]['position_epoch']['sha256']='0'*64
    from scripts.aufgabe04.real_robot.observer.candidate_centering import _digest
    request.advisory.pop('camera_centering_advisory_sha256')
    request.advisory['camera_centering_advisory_sha256']=_digest(request.advisory)
    motion=Mock(side_effect=motion_result)
    with pytest.raises(ValueError,match='position epochs changed'):
        child.run_candidate_centering_child(request,run_process=runner(monkeypatch,motion))
    motion.assert_not_called()


def test_coarse_arrival_allows_two_fresh_fine_turns_without_resetting_budget(turn_request,monkeypatch):
    from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory
    request=epoch_turn_request(turn_request,monkeypatch)
    motion=Mock(side_effect=motion_result)
    outcome=child.run_candidate_centering_child(request,run_process=runner(monkeypatch,motion))
    total=outcome.result['total_angular_travel_rad']
    ordinary=validate_camera_centering_advisory(turn_request.advisory)
    # Contract replay: each following receipt is a new ordinary small-angle
    # observation. Live odometry/clearance remain tested in the runtime harness.
    for index in (1,2):
        delta=request.advisory['created_at_sec']+2*index-ordinary.created_at_sec
        fresh=replace(ordinary,candidate_uid=request.candidate_id,stream_id=request.advisory['stream_id'],
            target_key=request.advisory['target_key'],image_stamp_sec=ordinary.image_stamp_sec+delta,
            scan_stamp_sec=ordinary.scan_stamp_sec+delta,odom_stamp_sec=ordinary.odom_stamp_sec+delta,
            created_at_sec=ordinary.created_at_sec+delta)
        monkeypatch.setattr(child.time,'time',lambda:fresh.created_at_sec+.1)
        fine=replace(request,output_dir=request.output_dir.parent/f'fine_{index}',turn_index=index,
            advisory=fresh.metadata(),signed_turn_rad=fresh.requested_yaw_rad,
            remaining_travel_rad=math.radians(42)-total,previous_result_path=outcome.result_path)
        outcome=child.run_candidate_centering_child(fine,run_process=runner(monkeypatch,motion))
        assert outcome.result['arrival_recovery'] is True
        assert load_candidate_centering_result(outcome.result_path)==outcome.result
        assert outcome.result['total_angular_travel_rad']==pytest.approx(total+abs(fine.signed_turn_rad))
        total=outcome.result['total_angular_travel_rad']
    assert motion.call_count==3
    assert total>math.radians(12)
    with pytest.raises(ValueError,match='turn index'):
        child.run_candidate_centering_child(replace(fine,output_dir=request.output_dir.parent/'extra',turn_index=3,
            previous_result_path=outcome.result_path),run_process=runner(monkeypatch,motion))
    assert motion.call_count==3
