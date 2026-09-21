"""The final buffer sample must catch up with direct stationary /tf capture."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from functools import partial

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


def observe(sequence, minimum=10.27, callback_elapsed=None):
    node = object.__new__(module.RosPreflightNode)
    node.tf_buffer = SimpleNamespace(lookup_transform=Mock(
        side_effect=sequence if callable(sequence) else iter(sequence)))
    node.get_clock = lambda: SimpleNamespace(now=lambda: FakeTime(10_400_000_000))
    node.max_future_timestamp_sec = .25
    observations, failures = [], []
    elapsed = [0.0]

    def spin(_node, timeout_sec):
        elapsed[0] += timeout_sec if callback_elapsed is None else callback_elapsed

    with patch.object(module, 'Time', FakeTime), patch.object(module, 'Duration', Mock()), \
            patch.object(module, 'TransformException', LookupError), \
            patch.object(module, 'acquire_tf_snapshot', partial(
                module.acquire_tf_snapshot, monotonic=lambda: elapsed[0])), \
            patch.object(module, 'rclpy', SimpleNamespace(spin_once=Mock(side_effect=spin))) as ros:
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


def test_buffer_that_never_catches_up_fails_at_deadline():
    ok, data, failures, spins = observe(lambda *a, **kw: _transform(stamp_sec=10.169))
    assert not ok and spins > 5
    assert data['stamp_sec'] == 10.169
    assert data['stationary_window_wait_sec'] == pytest.approx(0.5)
    assert data['stationary_window_wait_timed_out']
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


def test_more_than_five_ready_callbacks_can_be_drained_before_success():
    ok, data, failures, spins = observe(
        [_transform(stamp_sec=10.169)] * 9 + [_transform(stamp_sec=10.270)])
    assert ok and not failures and spins == 9
    assert not data['stationary_window_wait_timed_out']


def test_missing_transform_can_arrive_inside_deadline():
    ok, data, failures, spins = observe([LookupError('missing'), _transform(stamp_sec=10.270)])
    assert ok and not failures and spins == 1


def test_missing_transform_times_out_with_diagnostics():
    def missing(*args, **kwargs):
        raise LookupError('missing map')
    ok, data, failures, _ = observe(missing)
    assert not ok and failures
    assert data['stationary_window_wait_timed_out']
    assert data['error'] == 'missing map'


def test_new_transform_after_slow_callback_still_fails_deadline_gate():
    ok, data, failures, _ = observe(
        [_transform(stamp_sec=10.169), _transform(stamp_sec=10.270)],
        callback_elapsed=1.0,
    )
    assert not ok and data['stationary_window_wait_timed_out']
    assert 'deadline expired' in failures[0]
