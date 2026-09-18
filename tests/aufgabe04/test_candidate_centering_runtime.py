"""Exercise the existing centering controller with simulated odometry only."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower import runtime
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components import candidate_centering as controller
from scripts.aufgabe04.real_robot.execution.candidate_centering import build_candidate_centering_permit
from tests.aufgabe04.test_candidate_centering_child import turn_request


def harness(permit, monkeypatch, *, safety=lambda: '', target=True, stationary=True):
    node=object.__new__(runtime.SimpleWaypointFollowerNode)
    node.follower_config=FollowerConfig(controller=ControllerConfig(max_angular_radps=.06),max_scan_age_sec=.5,max_odom_age_sec=.5)
    node.runtime_config=SimpleNamespace(odom_frame='odom',base_frame='base_footprint')
    anchor=permit['advisory']['anchor_odom_pose']
    state=dict(pose=Pose2D(**anchor), speed=0., elapsed=0.)
    commands=[]; events=[]
    node.motion_published=False;node.zero_command_publish_count=0
    node._latest_odom_pose=lambda:state['pose']
    node._ros_now_sec=lambda:permit['advisory']['created_at_sec']+.2+state['elapsed']
    node._safety_failure=safety
    def service(dt):
        state['elapsed']+=dt
        state['pose']=replace(state['pose'],yaw_rad=state['pose'].yaw_rad+state['speed']*dt)
    node._service_or_wait_for_callbacks=service
    def zero(count):
        events.append('zero');state['speed']=0.;node.zero_command_publish_count+=count
    node.publish_repeated_zero=zero
    node._wait_for_stationary_odom_pair=lambda **kwargs:(state['pose'] if stationary else None,{'accepted':stationary})
    def publish(command):
        commands.append(command);state['speed']=command.angular_z_radps;node.motion_published=True
    node._publish_velocity_command=publish
    node._append_controller_trace=lambda **kwargs:events.append('trace') or ''
    stamp=permit['advisory']['created_at_sec']+.1
    header=lambda frame:SimpleNamespace(frame_id=frame,stamp=SimpleNamespace(sec=int(stamp),nanosec=int((stamp%1)*1e9)))
    node.latest_scan=SimpleNamespace(header=header('base_scan'))
    node.latest_odom=SimpleNamespace(header=header('odom'),child_frame_id='base_footprint')
    monkeypatch.setattr(controller,'rclpy',SimpleNamespace(ok=lambda:True))
    monkeypatch.setattr(controller.time,'monotonic',lambda:state['elapsed'])
    monkeypatch.setattr(controller,'fresh_centering_target',lambda *args:{} if target else None)
    return node,commands,events,state


def test_recorded_turn_converges_under_profile_speed_with_zero_translation(turn_request,monkeypatch):
    permit=build_candidate_centering_permit(turn_request)
    node,commands,events,_=harness(permit,monkeypatch)
    result=node.run_candidate_centering(permit)
    assert result['status']=='completed',result
    assert commands and all(c.linear_x_mps==0 and 0<c.angular_z_radps<=.06 for c in commands)
    assert abs(result['final_yaw_error_rad'])<=controller.STOP_TOLERANCE_RAD
    assert result['zero_command_count']>=20 and result['stationary_odom']['accepted']
    assert events[-1]=='zero'


@pytest.mark.parametrize('failure', ('unsafe','target_changed','not_stationary','wrong_odom_frame','stale_scan'))
def test_invalid_live_context_never_publishes_rotation(turn_request,monkeypatch,failure):
    permit=build_candidate_centering_permit(turn_request)
    node,commands,events,_=harness(permit,monkeypatch,
        safety=lambda:'obstacle' if failure=='unsafe' else '',
        target=failure!='target_changed',stationary=failure!='not_stationary')
    if failure=='wrong_odom_frame': node.latest_odom.header.frame_id='wrong'
    if failure=='stale_scan': node.latest_scan.header.stamp.sec=0
    result=node.run_candidate_centering(permit)
    assert result['status']=='stopped' and not commands
    assert result['zero_command_count']>=20


def test_translation_during_turn_stops_before_further_commands(turn_request,monkeypatch):
    permit=build_candidate_centering_permit(turn_request)
    node,commands,events,state=harness(permit,monkeypatch)
    service=node._service_or_wait_for_callbacks
    def drift(dt):
        service(dt)
        if commands: state['pose']=replace(state['pose'],x_m=state['pose'].x_m+.02)
    node._service_or_wait_for_callbacks=drift
    result=node.run_candidate_centering(permit)
    assert result['status']=='stopped' and 'translation' in result['stop_reason']
    assert len(commands)==1 and events[-1]=='zero'


def test_ros_wrapper_stops_on_exception_before_tearing_down_publisher(turn_request,monkeypatch):
    permit=build_candidate_centering_permit(turn_request)
    events=[]
    node=SimpleNamespace(enable_background_callback_service=lambda:None,
        disable_background_callback_service=lambda:None,destroy_node=lambda:events.append('destroy'),
        publish_repeated_zero=lambda count:events.append(('zero',count)),
        run_candidate_centering=Mock(side_effect=RuntimeError('lost callback')))
    make_node=Mock(return_value=node)
    monkeypatch.setattr(runtime,'_require_ros',lambda:None)
    monkeypatch.setattr(runtime,'rclpy',SimpleNamespace(init=lambda **kwargs:None,ok=lambda:True,shutdown=lambda:events.append('shutdown')))
    monkeypatch.setattr(runtime,'SimpleWaypointFollowerNode',make_node)
    monkeypatch.setattr(runtime,'MultiThreadedExecutor',lambda **kwargs:SimpleNamespace(
        add_node=lambda node:True,spin=lambda:None,shutdown=lambda:None,remove_node=lambda node:None))
    monkeypatch.setattr(runtime.threading,'Thread',lambda **kwargs:SimpleNamespace(start=lambda:None,ident=1,join=lambda:None))
    with pytest.raises(RuntimeError,match='lost callback'):
        runtime.run_candidate_centering_motion(permit)
    assert events==[('zero',10),'destroy','shutdown']
    config=make_node.call_args.args[2]
    assert config.min_obstacle_distance_m>=.2 and config.controller.max_angular_radps<=.12
