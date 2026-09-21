"""Startup uses the local TF buffer without blocking or selecting old tuples."""

from types import SimpleNamespace
from unittest.mock import Mock

from scripts.aufgabe04.real_robot.observer.tf_startup import ObserverTfStartup
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode


def test_missing_tf_is_rate_limited_and_readiness_does_not_extend_time():
    now = [10.]
    gate = ObserverTfStartup(clock=lambda: now[0])
    ready, report = gate.poll([("map<-base", lambda: False)])
    assert not ready and report["missing"] == ["map<-base"]
    now[0] += .2
    assert gate.poll([("map<-base", lambda: False)]) == (False, None)
    now[0] = 15.
    ready, report = gate.poll([("map<-base", lambda: True)])
    assert ready and report["waited_sec"] == 5.
    assert report["exact_sensor_time_still_required"]
    probe = Mock()
    assert gate.poll([("map<-base", probe)]) == (True, None)
    probe.assert_not_called()


def test_node_does_not_pin_a_sensor_tuple_while_its_buffer_is_unready():
    node = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
    node.completed = False
    node._tf_startup = ObserverTfStartup()
    node.profile = SimpleNamespace(map_frame="map", base_frame="base", scan_frame="scan",
                                   camera_optical_frame="camera")
    node.tf_buffer = SimpleNamespace(can_transform=Mock(return_value=False))
    node.Time, node.Duration = Mock(), Mock()
    node._write_status = Mock()
    node._next_sensor_tuple = Mock(return_value=None)
    node._process_latest()
    node._next_sensor_tuple.assert_not_called()
    assert node._write_status.call_args.args == ("waiting_for_observer_tf",)
    node.tf_buffer.can_transform.return_value = True
    node._process_latest()
    node._next_sensor_tuple.assert_called_once()
