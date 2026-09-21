"""The final buffer sample must catch up with direct stationary /tf capture."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.aufgabe04.navigation.localization import ros_preflight as module
from tests.aufgabe04.test_ros_preflight_map_odom_samples import _transform


class FakeTime:
    def __init__(self, nanoseconds=0):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_msg(cls, stamp):
        return cls(stamp.sec * 1_000_000_000 + stamp.nanosec)

    def __sub__(self, other):
        return FakeTime(self.nanoseconds - other.nanoseconds)


def observe(sequence, minimum=10.27):
    node = object.__new__(module.RosPreflightNode)
    node.tf_buffer = SimpleNamespace(lookup_transform=Mock(side_effect=sequence))
    node.get_clock = lambda: SimpleNamespace(now=lambda: FakeTime(10_400_000_000))
    node.max_future_timestamp_sec = .25
    observations, failures = [], []
    with patch.object(module, 'Time', FakeTime), patch.object(module, 'Duration', Mock()), \
            patch.object(module, 'rclpy', SimpleNamespace(spin_once=Mock())) as ros:
        ok, data = node._observe_tf(observations, failures, 'map', 'odom', 1.,
                                   minimum_stamp_sec=minimum)
        return ok, data, failures, ros.spin_once.call_count


def test_recorded_callback_lag_recovers_without_relabeling_the_old_sample():
    # Recorded failure: cache lagged the direct sample by 101 ms, same pose.
    ok, data, failures, spins = observe([
        _transform(stamp_sec=10.169), _transform(stamp_sec=10.169),
        _transform(stamp_sec=10.270)])
    assert ok and not failures and spins == 2
    assert data['stamp_sec'] == 10.270
    assert data['stationary_window_reacquisition_count'] == 2


def test_buffer_that_never_catches_up_fails_after_five_retries():
    ok, data, failures, spins = observe([_transform(stamp_sec=10.169)] * 6)
    assert not ok and spins == 5
    assert data['stamp_sec'] == 10.169
    assert 'predates stationary sample window' in failures[0]


@pytest.mark.parametrize('new', [
    _transform(stamp_sec=10.270, map_frame='wrong'),
    _transform(stamp_sec=10.270, quaternion=(0., 0., 0., 0.)),
    _transform(stamp_sec=10.270, x_m=float('nan')),
    _transform(stamp_sec=11.),
])
def test_newer_sample_still_must_pass_existing_transform_gates(new):
    ok, _, failures, spins = observe([_transform(stamp_sec=10.169), new])
    assert not ok and failures and spins == 1


def test_already_ordered_sample_needs_no_retry():
    ok, _, failures, spins = observe([_transform(stamp_sec=10.270)])
    assert ok and not failures and spins == 0
