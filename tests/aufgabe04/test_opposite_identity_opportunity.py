"""Bounded association recovery cannot become admission or motion authority."""
from dataclasses import replace
import pytest

from scripts.aufgabe04.real_robot.observer.opposite_identity_opportunity import OppositeIdentityOpportunity, STATE
from scripts.aufgabe04.real_robot.observer.timeout_policy import is_candidate_local_observer_timeout


def row(stamp=10., **overrides):
    return dict(target_key='target',epoch=0,pose=(0.,0.,0.),image_stamp_sec=stamp,
        scan_stamp_sec=stamp,now_sec=stamp+.1,poisoned=False,motion_epoch_reset=False,
        conflict=True,**overrides)


def test_fresh_unassociated_frames_finish_after_five_seconds():
    tracker=OppositeIdentityOpportunity()
    for t in range(10,16):
        result=tracker.observe(**row(float(t)))
        if t<15:assert result is None
    assert result['elapsed_sec']==5.
    assert result['motion_authorized'] is False
    assert result['candidate_geometry_updated'] is False
    assert result['recovery']=='bounded_inspection_view'


@pytest.mark.parametrize('changes',[
    {'conflict':False},{'poisoned':True},{'motion_epoch_reset':True},
    {'now_sec':30.},{'pose':(.1,0.,0.)},{'epoch':1},{'target_key':'another'},
    {'image_stamp_sec':13.},{'scan_stamp_sec':12.},
])
def test_progress_stale_motion_poison_rebinding_and_repeated_samples_cannot_expire(changes):
    tracker=OppositeIdentityOpportunity()
    for t in range(10,14):assert tracker.observe(**row(float(t))) is None
    assert tracker.observe(**{**row(14.),**changes}) is None
    assert tracker.observe(**row(15.)) is None


def test_clean_exit_enters_bounded_recovery_but_crash_or_poison_does_not():
    from tests.aufgabe04.test_passive_observer_diagnostics import PassiveObserverDiagnosticsTests
    case=PassiveObserverDiagnosticsTests()
    status=case._load_payload(dict(state=STATE,
        observation_evidence=dict(accepted_frame_count=0,lidar_rejection_count=30,poisoned=False)))
    process=replace(case._process('child_exit'),returncode=0,signals_sent=(),cleanup_actions=('exit_observed',))
    assert is_candidate_local_observer_timeout(process=process,status=status)
    for changes in (dict(returncode=1),dict(signals_sent=('SIGINT',)),dict(completion_kind='deadline',deadline_expired=True)):
        assert not is_candidate_local_observer_timeout(process=replace(process,**changes),status=status)
    for changes in (dict(observation_evidence_poisoned=True),dict(observation_evidence_poisoned=None)):
        assert not is_candidate_local_observer_timeout(process=process,status=replace(status,**changes))
