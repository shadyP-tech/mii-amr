import pytest

from scripts.aufgabe04.navigation.localization.tf_snapshot_barrier import acquire_tf_snapshot


def run(sequence, timeout=0.05):
    clock = [0.0]
    waits = []
    values = iter(sequence)

    def lookup():
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    def service(duration):
        waits.append(duration)
        clock[0] += duration

    result = acquire_tf_snapshot(
        lookup=lookup, stamp_sec=lambda value: value, service_callbacks=service,
        minimum_stamp_sec=10.27, lookup_errors=(LookupError,),
        timeout_sec=timeout, monotonic=lambda: clock[0],
    )
    return result, waits


def test_fast_path_does_not_service_callbacks():
    result, waits = run([10.27])
    assert result.transform == 10.27 and not waits and result.elapsed_sec == 0


def test_deadline_caps_last_callback_wait():
    result, waits = run([10.16] * 5, timeout=0.025)
    assert result.timed_out and result.transform == 10.16
    assert sum(waits) == pytest.approx(0.025)
    assert waits[-1] == pytest.approx(0.005)


def test_disappearing_transform_never_returns_previous_snapshot():
    result, _ = run([10.16] + [LookupError('reset')] * 6)
    assert result.timed_out and result.transform is None
    assert result.last_error == 'reset'


def test_lookup_programming_errors_are_not_retried():
    with pytest.raises(TypeError, match='bug'):
        run([TypeError('bug')])


def test_slow_callback_cannot_admit_a_snapshot_after_deadline():
    clock = [0.0]
    values = iter([10.16, 10.27])
    result = acquire_tf_snapshot(
        lookup=lambda: next(values), stamp_sec=lambda value: value,
        service_callbacks=lambda wait: clock.__setitem__(0, 1.0),
        minimum_stamp_sec=10.27, lookup_errors=(LookupError,),
        timeout_sec=0.5, monotonic=lambda: clock[0],
    )
    assert result.timed_out and result.transform == 10.27


@pytest.mark.parametrize('timeout', [-1, float('nan'), float('inf')])
def test_invalid_deadline_rejected(timeout):
    with pytest.raises(ValueError):
        run([10.27], timeout)
