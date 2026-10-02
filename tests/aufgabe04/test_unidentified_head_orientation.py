"""ID-only misses retain an axis for another view, never a claimed backside."""

from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
from unittest.mock import Mock

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from scripts.aufgabe04.artifacts.backside_axis_observation import validated_backside_axis_observation
from scripts.aufgabe04.artifacts.retained_backside_orientation import (
    orientation_record, validate_retained_orientation,
)
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    load_backside_axis_frame_projection, write_backside_axis_frame_projection,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.head_backside_appearance import assess_current_head_backside_appearance
from scripts.aufgabe04.real_robot.observer.backside_head_crop import (
    review_current_head_crop, review_backside_head_crop,
)
from scripts.aufgabe04.real_robot.observer.bounded_head_observation import prepare_bounded_head
from scripts.aufgabe04.real_robot.observer.current_head_association import associate_current_measured_head
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from scripts.aufgabe04.real_robot.observer.tracked_head_registration import (
    register_current_tracked_head, tracked_head_selection,
)
from tests.aufgabe04 import test_bounded_head_observation as fixtures
from tests.aufgabe04 import test_current_head_association as association_fixtures
from tests.aufgabe04.backside_axis_fixture import write_candidate_frame_projection_fixture


@pytest.fixture
def observer():
    fixtures.BoundedHeadObservationTests.setUpClass()
    fixture = fixtures.BoundedHeadObservationTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def frame(observer, stamp, *, attempt_changes=None, associated=True, complete=True,
          age=.1, marker_seen=False, qr_conflict=False, pose=None, before_publish=None):
    """Use real head bounds, association, temporal gates, and durable publication."""
    adapter = observer.adapter
    observer.fixture.clock_sec = stamp + age
    pose = Pose2D(0., 0., 0.) if pose is None else pose
    current = replace(observer.current, debug=replace(observer.current.debug,
        qr_detected=None, qr_marker_verified=None), qr_observations=())
    appearance = assess_current_head_backside_appearance(
        current.estimate, current.debug, model_profile=observer.profile,
        camera=observer.camera,
        expected_center_u_px=sum(p.u_px for p in current.estimate.corners) / 4,
        expected_center_v_px=80., expected_height_px=90., identity_unresolved=True)
    assert appearance.accepted
    assert appearance.basis == "current_raw_head_identity_unresolved"
    current = replace(current, debug=replace(current.debug, head_backside_appearance=appearance))
    options = association_fixtures.CurrentHeadAssociationTests().options()
    options.update(estimate=current.estimate, debug=current.debug, attempt=current.attempt,
        profile_sha256=observer.profile.sha256, now_sec=stamp+.1,
        scan=replace(options["scan"], scan_stamp_sec=stamp, receipt_sec=stamp))
    association = associate_current_measured_head(**options)
    assert association.accepted
    if not associated:
        association = replace(association, accepted=False)
    selection = register_current_tracked_head(tracked_head_selection(current),
        association=association, observed_at_sec=stamp, now_sec=stamp+.1,
        max_age_sec=.5, expected_model_sha256=observer.profile.sha256)
    crop = review_current_head_crop(selection)
    appearance_crop = review_backside_head_crop(selection)
    if not complete:
        crop = replace(crop, accepted=False)
        appearance_crop = replace(appearance_crop, accepted=False)
    attempt = dict(attempted=True, accepted_crop=True, decoded_texts=[],
                   image_stamp_sec=stamp, scan_stamp_sec=stamp)
    attempt.update(attempt_changes or {})
    metadata = {"current_head_identity": attempt}
    adapter._pending_bounded_head = prepare_bounded_head(
        estimate=current.estimate, debug=current.debug, association=association,
        crop=crop, appearance_crop=appearance_crop,
        qr_binding=QrTargetBinding(False, "no_decoded_qr_identity"),
        marker_verified=None, marker_seen_in_epoch=marker_seen,
        image_stamp_sec=stamp, scan_stamp_sec=stamp, robot_pose=pose,
        camera_heading_rad=0., stand_x_m=.6, stand_y_m=0.,
        camera_signature=(640., 640., 400., 300.), roi=current.attempt.roi,
        metadata=metadata, projected_center_px=(400., 300.),
        expected_head_height_px=90., head_position_evidence=None)
    adapter._qr_marker_seen_in_stationary_epoch = marker_seen
    # Neutral geometry must not borrow a backside-confidence classification.
    adapter._pending_head_confidence = None
    update = adapter._record_observation_frame(
        robot_pose=pose, image_stamp_sec=stamp, scan_stamp_sec=stamp,
        observed_at_sec=stamp+age, lidar_associated=associated,
        axis_yaw_rad=None, axis_source=None,
        qr_texts=("Start", "QR_003") if qr_conflict else (),
        qr_symbol_count=2 if qr_conflict else 0)
    if before_publish is not None:
        before_publish(adapter)
    PassiveRealViewpointNode._write_status(adapter, "metric_model_measurement_unavailable")
    return update, metadata


def receipt(observer):
    path = observer.adapter.args.axis_observation_json
    return json.loads(path.read_text()) if path.exists() else None


def complete(observer):
    for index in range(7):
        update, metadata = frame(observer, 100. + index * .2)
    assert receipt(observer) is not None, metadata
    return receipt(observer), update


def test_seven_current_id_attempts_commit_orientation_without_claiming_backside(observer):
    for index in range(7):
        update, metadata = frame(observer, 100. + index * .2)
        if index < 6:
            assert receipt(observer) is None
    payload = receipt(observer)
    assert payload is not None, metadata
    observation = validated_backside_axis_observation(payload)
    assert payload["schema_version"] == 5
    assert payload["visible_face"] == "unidentified"
    assert payload["visible_face_source"] == "current_measured_head_unidentified"
    assert payload["model_evidence_state"] == "fresh_unidentified_head"
    assert payload["classification_basis"] == "measured_head_geometry_plus_repeated_identity_attempts"
    assert payload["qr_marker_detected"] is None
    assert payload["qr_texts"] == []
    assert payload["sample_gate_evidence"]["all_samples_identity_attempted"] is True
    assert payload["sample_gate_evidence"]["all_samples_identity_undecoded"] is True
    assert "all_samples_qr_marker_absent" not in payload["sample_gate_evidence"]
    assert "qr_absent_sample_count" not in payload
    assert payload["identity_undecoded_sample_count"] == 7
    assert payload["motion_capability"] == "none"
    assert observation.axis_sample_count == 7 and observation.axis_confidence == 0.
    assert observation.bounded_orientation["half_width_rad"] == pytest.approx(observer.proof.half_width_rad)
    assert math.isfinite(observation.opposite_face_normal_rad)
    assert update.resolved_qr_id is None
    assert observer.adapter.axis_observation_committed
    assert not observer.adapter.args.recommended_pose_json.exists()


def use_precise_geometry(observer):
    """Fit the same measured model at a pose passing the existing strict gate."""
    import cv2
    import numpy
    from scripts.aufgabe04.perception.stand_axis.head_model_quality import evaluate_head_model_quality
    from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import evaluate_current_head_orientation_bounds
    from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
    from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import estimate_planar_pose_ippe
    from tests.aufgabe04.test_head_model_admission import outer_boundary
    pixels = cv2.projectPoints(numpy.asarray([(p.x_m, p.y_m, p.z_m)
        for p in observer.profile.head_corners]), numpy.asarray((0., .2, 0.)),
        numpy.asarray((-.03, 0., .55)), numpy.asarray(((640., 0., 120.),
        (0., 640., 80.), (0., 0., 1.))), None)[0]
    corners = tuple(ImagePoint(float(u), float(v)) for u, v in pixels.reshape(-1, 2))
    poses = estimate_planar_pose_ippe(cv2, corners, observer.profile.head_corners, observer.camera)
    options = dict(profile=observer.profile, camera=observer.camera, corners=corners,
        pose_result=poses, raw_border_support_mean=.98, raw_corner_support_accepted=True,
        outer_border_verified=True)
    quality = evaluate_head_model_quality(cv2, **options, centered_neck_supported=False)
    assert quality.accepted
    proof = evaluate_current_head_orientation_bounds(cv2, **options, frame_shape=(160, 160))
    assert proof.accepted
    pose = poses.hypotheses[0]
    left, right = (math.hypot(corners[a].u_px-corners[b].u_px,
                            corners[a].v_px-corners[b].v_px) for a,b in ((0,3),(1,2)))
    observer.proof = proof
    observer.current = replace(observer.current,
        estimate=replace(observer.current.estimate, corners=corners, usable=True,
            yaw_deg=pose.yaw_deg, evidence_state='fresh_refined', left_height_px=left,
            right_height_px=right, reason=quality.reason),
        debug=replace(observer.current.debug, refined_corners=corners, model_pose=pose,
            evidence_state='fresh_refined', head_model_quality=quality,
            head_orientation_bounds=proof, head_outer_recovery=outer_boundary(corners, observer.profile.sha256),
            pose_reprojection_rmse_px=pose.reprojection_rmse_px))


@pytest.mark.parametrize('precise', [False, True])
def test_empty_decode_retains_precise_or_bounded_axis_ahead_of_centering(observer, monkeypatch, precise):
    if precise:
        use_precise_geometry(observer)
    for index in range(6):
        frame(observer, 100. + index*.2)
    centering = Mock(side_effect=AssertionError('usable retained geometry precedes advice'))
    monkeypatch.setattr('scripts.aufgabe04.real_robot.observer.node.commit_candidate_centering', centering)
    frame(observer, 101.2, before_publish=lambda adapter:
        setattr(adapter, '_candidate_centering_ready', object()))
    payload = receipt(observer)
    assert payload is not None
    assert payload['visible_face'] == 'unidentified'
    assert payload['bounded_orientation']['half_width_rad'] == pytest.approx(observer.proof.half_width_rad)
    centering.assert_not_called()


@pytest.mark.parametrize("changes", (
    {"attempted": False}, {"attempted": None}, {"accepted_crop": False},
    {"attempted": False, "decoder": {"events": [{"stage": "wechat", "reason": "decoder_error"}]}},
    {"decoded_texts": ["Start"]}, {"decoded_texts": None},
    {"image_stamp_sec": 99.}, {"scan_stamp_sec": 99.},
))
def test_skipped_unowned_or_wrong_tuple_attempts_never_supply_neutral_samples(observer, changes):
    for index in range(7):
        frame(observer, 100. + index * .2, attempt_changes=changes)
        assert observer.adapter._pending_bounded_head is None
    assert receipt(observer) is None


@pytest.mark.parametrize("changes", (
    {"associated": False}, {"complete": False}, {"age": .6}, {"marker_seen": True},
))
def test_neutral_samples_retain_existing_sensor_geometry_and_marker_vetoes(observer, changes):
    for index in range(7):
        frame(observer, 100. + index * .2, **changes)
    assert receipt(observer) is None


def test_conflicting_qr_epoch_cannot_commit_a_collected_neutral_axis(observer):
    for index in range(6):
        frame(observer, 100. + index * .2)
    update, _ = frame(observer, 101.2, qr_conflict=True)
    assert update.snapshot.poisoned
    assert receipt(observer) is None


def test_skipped_and_duplicate_frames_do_not_complete_seven_sample_collection(observer):
    for index in range(6):
        frame(observer, 100. + index * .2)
    frame(observer, 101.)  # Same source stamp cannot count twice.
    assert receipt(observer) is None
    frame(observer, 101.2, attempt_changes={"attempted": False})
    assert receipt(observer) is None
    frame(observer, 101.4)
    payload = receipt(observer)
    assert payload is not None
    stamps = payload["axis_measurement"]["bounded_orientation_window"]["source_stamps_sec"]
    assert len(stamps) == len(set(stamps)) == 7
    assert 101.2 not in stamps


def test_robot_motion_does_not_reuse_six_previous_head_samples(observer):
    for index in range(6):
        frame(observer, 100. + index * .2)
    update, _ = frame(observer, 101.2, pose=Pose2D(.2, 0., 0.))
    assert update.motion_epoch_reset
    assert receipt(observer) is None
    assert observer.adapter._bounded_head_window.metadata.get("sample_count", 0) <= 1


def test_neutral_receipt_roundtrips_projection_and_retention_without_backside_policy(observer):
    payload, _ = complete(observer)
    source = observer.root / "source_projection.json"
    target = observer.root / "target_projection.json"
    options = dict(candidate_uid=payload["stand_id"], canonical_x_m=.6, canonical_y_m=0.)
    source_sha, _, _ = write_candidate_frame_projection_fixture(source, **options,
        transform_x_m=0., transform_y_m=0., transform_yaw_rad=0.)
    target_sha, x, y = write_candidate_frame_projection_fixture(target, **options,
        transform_x_m=.2, transform_y_m=-.1, transform_yaw_rad=.4)
    projected_path = observer.root / "axis_projection.json"
    write_backside_axis_frame_projection(projected_path,
        axis_evidence_path=observer.adapter.args.axis_observation_json,
        source_candidate_projection_path=source, source_candidate_projection_sha256=source_sha,
        target_candidate_projection_path=target, target_candidate_projection_sha256=target_sha,
        target_candidate_x_m=x, target_candidate_y_m=y)
    projected = load_backside_axis_frame_projection(projected_path)
    orientation = orientation_record(projected_path)
    assert orientation["policy"] == "certified_unidentified_head_orientation_retained"
    assert orientation["axis_sample_count"] == 7
    assert orientation["bounded_orientation"]["half_width_rad"] == pytest.approx(observer.proof.half_width_rad)
    assert math.remainder(projected.stand_axis_rad - payload["stand_axis_rad"] - .4, math.pi) == pytest.approx(0.)
    assert validate_retained_orientation(orientation,
        candidate_uid=payload["stand_id"], planning_frame="map",
        stand_center=dict(x_m=x, y_m=y), model_sha256=observer.profile.sha256) == orientation


@pytest.mark.parametrize("mutation", (
    lambda p: p.update(qr_marker_detected=False),
    lambda p: p.update(visible_face="backside_candidate"),
    lambda p: p.update(bounded_orientation=None),
    lambda p: p.update(axis_sample_count=6),
    lambda p: p["sample_gate_evidence"].update(all_samples_identity_attempted=False),
    lambda p: p["sample_gate_evidence"].update(all_samples_identity_undecoded=False),
))
def test_neutral_receipt_rejects_missing_orientation_or_fabricated_side_proofs(observer, mutation):
    payload, _ = complete(observer)
    changed = deepcopy(payload)
    mutation(changed)
    with pytest.raises(ValueError):
        validated_backside_axis_observation(changed)
