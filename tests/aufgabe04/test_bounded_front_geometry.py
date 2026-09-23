"""Immediate QR completion retains inspection bounds without claiming consensus."""
from dataclasses import replace
from unittest.mock import patch
import pytest
from tests.aufgabe04 import test_bounded_head_observation as bounded_fixture
from tests.aufgabe04 import test_qr_observation_pose as qr_fixture
from scripts.aufgabe04.artifacts.bounded_front_geometry import (
    save_bounded_front_geometry, load_bounded_front_geometry, validate_bounded_front_geometry)
from scripts.aufgabe04.real_robot.observer.qr_observation_pose import commit_qr_observation_pose


@pytest.fixture
def stopped():
    bounded_fixture.BoundedHeadObservationTests.setUpClass()
    bounded=bounded_fixture.BoundedHeadObservationTests();bounded.setUp()
    qr=qr_fixture.QrObservationPoseTests();qr.setUp()
    try:
        bounded.frame(100.,publish=False)
        current=bounded.adapter._pending_bounded_head
        assert current is not None
        qr.adapter.stand_model_profile.sha256=current.proof.profile_sha256
        qr.adapter.args.qr_pose_fallback_delay_sec=0.
        qr.frame(100.,publish=False)
        qr.adapter._pending_bounded_head=current
        yield qr,current
    finally:
        qr.doCleanups();bounded.doCleanups()


def test_one_frame_qr_commits_and_retains_bounds_without_seven_sample_facing(stopped):
    qr,current=stopped
    result=commit_qr_observation_pose(qr.adapter)
    assert qr.adapter.completed and qr.result()['qr_id']=='QR_003'
    assert result[1]['facing_ready'] is False
    path=result[1]['bounded_front_geometry_json']
    assert path, current.metadata
    payload=load_bounded_front_geometry(path)
    assert payload['axis_sample_count']==1
    assert not payload['facing_ready'] and not payload['motion_authorized']
    assert payload['head_orientation_bounds']['half_width_rad']==current.proof.half_width_rad
    assert not qr.adapter.args.recommended_pose_json.exists()
    for change in ({'facing_ready':True},{'axis_sample_count':7},{'sensor_stamp_sec':101.}):
        with pytest.raises(ValueError):validate_bounded_front_geometry({**payload,**change})


def test_optional_retention_failure_does_not_revoke_committed_qr(stopped):
    qr,_=stopped
    with patch('scripts.aufgabe04.artifacts.bounded_front_geometry.save_bounded_front_geometry',side_effect=OSError('full')):
        result=commit_qr_observation_pose(qr.adapter)
    assert qr.adapter.completed and qr.result()
    assert result[1]['bounded_front_geometry_json'] is None


def test_stale_or_different_front_cannot_attach_to_current_qr(stopped):
    qr,current=stopped
    qr.adapter._pending_bounded_head=None
    commit_qr_observation_pose(qr.adapter)
    payload=qr.result();path=qr.adapter.args.qr_observation_pose_json
    for sample in (replace(current.sample,stamp_sec=99.),replace(current.sample,qr_id='other'),replace(current.sample,face='backside'),replace(current.sample,camera_signature=(1.,2.,3.,4.))):
        assert save_bounded_front_geometry(replace(current,sample=sample),path,payload) is None
    assert save_bounded_front_geometry(current,path,{**payload,'retained_backside_orientation':{}}) is None
