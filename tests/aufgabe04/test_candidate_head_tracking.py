"""A complete-head hint carries search information across front/back images."""

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.head_model_quality import HeadModelQuality
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import classify_current_head_backside
from scripts.aufgabe04.perception.stand_axis.head_outer_border import HeadOuterBorderEvidence
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint, StandAxisEdgeDebugArtifacts, StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis, RectifiedCameraMatrix
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.candidate_head_tracking import (
    CandidateHeadContext, CandidateHeadTracking,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt


CONTEXT = CandidateHeadContext("candidate_001", "a" * 64, (640., 640., 400., 300.),
                               (600, 800, 3), 0)
ROBOT = Pose2D(0., 0., 0.)
NOMINAL = HeadRoiAttempt(ImageRoi(277, 213, 460, 396, 100.), "nominal_projection", 1.8,
                         367., 303., 100.)
WIDE = replace(NOMINAL, roi=ImageRoi(142, 78, 592, 528, 100.), padding_scale=4.5,
               source="target_centered_backside_reacquisition")


def measured_head(*, qr=True):
    corners = tuple(ImagePoint(u - 277, v - 213) for u, v in
                    ((290., 230.), (389., 237.), (388., 336.), (289., 335.)))
    quality = HeadModelQuality(True, "current_measured_head_observable", .95, True, False,
                              98., .3, 2., False, True, 1.5, 3., .75, 200.,
                              (.078, .078), CONTEXT.model_sha256, outer_border_verified=True)
    pose = PlanarPoseHypothesis((0., .4, 0.), (0., 0., .5), (0., 0., 1.), -23., .3, True)
    estimate = StandAxisImageEstimate(True, "axis_estimated_current_measured_head", "metric_model_only",
                                     corners, None, 105., 100., 1.05, None, -23., None, 10000.,
                                     source="model_current_measured_head",
                                     model_profile_sha256=CONTEXT.model_sha256,
                                     model_measurement_status="measured",
                                     visible_face="front" if qr else "backside_candidate")
    debug = StandAxisEdgeDebugArtifacts(None, evidence_state="fresh_refined",
                                       model_profile_sha256=CONTEXT.model_sha256,
                                       model_measurement_status="measured",
                                       model_pose_fit_source="model_current_measured_head",
                                       model_pose=pose, head_model_quality=quality,
                                       head_outer_recovery=HeadOuterBorderEvidence(
                                           True, "maximal_current_head_border", corners, corners,
                                           CONTEXT.model_sha256, neutral_proposal_corners=corners,
                                           current_raw_alternatives=(corners,), head_size_m=(.078, .078)),
                                       qr_detected=qr, qr_marker_verified=qr)
    if not qr:
        estimate, debug = classify_current_head_backside(
            estimate, debug, model_profile=SimpleNamespace(committable=True, environment="physical",
                sha256=CONTEXT.model_sha256, head_width_m=.078, head_height_m=.078),
            camera=RectifiedCameraMatrix(640., 640., 123., 87.),
            expected_center_u_px=sum(p.u_px for p in corners)/4,
            expected_center_v_px=sum(p.v_px for p in corners)/4, expected_height_px=102.)
    return HeadRoiEvaluation(NOMINAL, None, estimate, debug,
                            (DecodedQrObservation("QR_003", None, "test"),) if qr else ())


def seed(tracker, evaluation=None, **changes):
    arguments = dict(context=CONTEXT, observed_at_sec=10., now_sec=10.4,
                     max_age_sec=.5, robot_pose=ROBOT, candidate_associated=True)
    arguments.update(changes)
    return tracker.remember(measured_head() if evaluation is None else evaluation, **arguments)


def search(tracker, **changes):
    arguments = dict(context=CONTEXT, observed_at_sec=10.6, robot_pose=ROBOT)
    arguments.update(changes)
    return tracker.hint((NOMINAL, WIDE), **arguments)


@pytest.mark.parametrize("qr", (False, True))
def test_fresh_associated_complete_head_seeds_independently_of_qr_or_side(qr):
    tracker = CandidateHeadTracking()
    evaluation = measured_head(qr=qr)
    if not qr:
        assert evaluation.estimate.evidence_state == "fresh_backside"
    assert seed(tracker, evaluation)  # 400 ms acquisition can initialize 500 ms observer.
    hint = search(tracker)
    assert hint is not None
    assert hint.pose_hint == evaluation.debug.model_pose
    assert hint.full_image_corners == tuple(ImagePoint(p.u_px + 277, p.v_px + 213)
                                            for p in evaluation.estimate.corners)
    assert hint.attempt.expected_center_u_px == NOMINAL.expected_center_u_px
    assert hint.attempt.expected_center_v_px == NOMINAL.expected_center_v_px
    assert hint.attempt.expected_head_height_px == NOMINAL.expected_head_height_px
    assert tracker.last_metadata["measurement_reused"] is False
    assert tracker.last_metadata["motion_authorized"] is False
    crop = hint.attempt.roi
    assert all(crop.x0 + 2 < p.u_px < crop.x1 - 2
               and crop.y0 + 2 < p.v_px < crop.y1 - 2 for p in hint.full_image_corners)


@pytest.mark.parametrize("change", (
    {"candidate_associated": False}, {"now_sec": 10.50001},
    {"now_sec": 9.9}, {"robot_pose": Pose2D(math.nan, 0., 0.)},
))
def test_unassociated_stale_future_or_invalid_pose_cannot_seed(change):
    tracker = CandidateHeadTracking()
    assert not seed(tracker, **change)
    assert search(tracker) is None


@pytest.mark.parametrize("change", (
    {"usable": False}, {"evidence_state": "predicted_only"},
    {"source": "model_projection"}, {"model_profile_sha256": "other"},
    {"corners": None},
    {"corners": tuple(ImagePoint(u, v) for u, v in ((0., 5.), (50., 5.), (50., 55.), (0., 55.)))},
))
def test_rejected_or_clipped_geometry_cannot_seed(change):
    tracker = CandidateHeadTracking()
    evaluation = measured_head()
    assert not seed(tracker, replace(evaluation, estimate=replace(evaluation.estimate, **change)))
    assert search(tracker) is None


@pytest.mark.parametrize("field,value", (("accepted", False), ("outer_border_verified", False),
                                         ("axis_ambiguous", True), ("yaw_std_deg", 3.1)))
def test_invalid_current_head_quality_cannot_seed(field, value):
    tracker = CandidateHeadTracking()
    evaluation = measured_head()
    quality = replace(evaluation.debug.head_model_quality, **{field: value})
    assert not seed(tracker, replace(evaluation, debug=replace(evaluation.debug, head_model_quality=quality)))


@pytest.mark.parametrize("context", (
    replace(CONTEXT, target_key="candidate_002"), replace(CONTEXT, model_sha256="b" * 64),
    replace(CONTEXT, camera_signature=(640., 640., 410., 300.)),
    replace(CONTEXT, image_shape=(480, 640, 3)), replace(CONTEXT, stationary_epoch=1),
))
def test_target_model_calibration_image_or_stationary_epoch_change_invalidates(context):
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert search(tracker, context=context) is None
    assert search(tracker) is None  # Returning to the old key does not revive it.


@pytest.mark.parametrize("pose", (Pose2D(.011, 0., 0.), Pose2D(0., 0., math.radians(2.1))))
def test_motion_invalidates_hint(pose):
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert search(tracker, robot_pose=pose) is None


def test_anchor_does_not_walk_with_successive_small_movements():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert seed(tracker, observed_at_sec=10.6, now_sec=10.7, robot_pose=Pose2D(.006, 0., 0.))
    assert search(tracker, observed_at_sec=11.2, robot_pose=Pose2D(.012, 0., 0.)) is None


@pytest.mark.parametrize("stamp", (9.9, 10., 12.00001, math.nan))
def test_expired_nonadvancing_or_invalid_source_image_invalidates(stamp):
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert search(tracker, observed_at_sec=stamp) is None


def test_lookup_does_not_refresh_source_time_and_replayed_seed_does_not_extend_ttl():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert search(tracker, observed_at_sec=11.8) is not None
    assert search(tracker, observed_at_sec=12.01) is None
    assert seed(tracker)
    assert not seed(tracker, now_sec=10.45)
    assert search(tracker) is None


def test_current_projection_bound_and_crop_extent_still_apply():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    changed = replace(NOMINAL, expected_center_u_px=550.)
    assert tracker.hint((changed, WIDE), context=CONTEXT, observed_at_sec=10.6, robot_pose=ROBOT) is None
    assert seed(tracker)
    clipped = replace(NOMINAL, roi=ImageRoi(300, 230, 450, 400, 100.))
    assert tracker.hint((clipped,), context=CONTEXT, observed_at_sec=10.6, robot_pose=ROBOT) is None


def test_brief_current_head_loss_preserves_only_old_search_location():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    evaluation = measured_head()
    assert not seed(tracker, replace(evaluation, estimate=replace(evaluation.estimate, usable=False)),
                    observed_at_sec=10.6, now_sec=10.7)
    metadata = tracker.last_metadata
    assert metadata["current_measurement_accepted"] is False
    assert metadata["source_stamp_refreshed"] is False
    assert metadata["source_stamp_sec"] == 10.
    assert search(tracker, observed_at_sec=11.2) is not None
    assert search(tracker, observed_at_sec=12.01) is None


def test_misses_do_not_refresh_ttl_and_repeated_failure_returns_to_cold_search():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    failed = replace(measured_head(), estimate=replace(measured_head().estimate, usable=False))
    for stamp in (10.6, 10.8):
        assert not seed(tracker, failed, observed_at_sec=stamp, now_sec=stamp+.1,
                        candidate_associated=False)
        assert tracker.last_metadata["hint_retained"] is True
        assert tracker.last_metadata["source_stamp_sec"] == 10.
        assert search(tracker, observed_at_sec=stamp+.2) is not None
    assert not seed(tracker, failed, observed_at_sec=11., now_sec=11.1,
                    candidate_associated=False)
    assert search(tracker, observed_at_sec=11.2) is None


@pytest.mark.parametrize("changes", (
    {"robot_pose": Pose2D(.02, 0.)},
    {"context": replace(CONTEXT, target_key="different")},
    {"now_sec": 11.2},  # Current miss is itself stale.
    {"observed_at_sec": 10., "now_sec": 10.1},
))
def test_miss_cannot_preserve_hint_across_motion_context_staleness_or_replay(changes):
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    failed = replace(measured_head(), estimate=replace(measured_head().estimate, usable=False))
    assert not seed(tracker, failed, **{
        "observed_at_sec": 10.6, "now_sec": 10.7, "candidate_associated": False, **changes})
    assert search(tracker, observed_at_sec=11.3) is None


def test_verified_recovery_refreshes_locator_and_clears_miss_count():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert not seed(tracker, observed_at_sec=10.4, now_sec=10.5, candidate_associated=False)
    assert seed(tracker, observed_at_sec=10.6, now_sec=10.7)
    assert search(tracker, observed_at_sec=12.5) is not None
    assert tracker.last_metadata["consecutive_soft_misses"] == 0


def test_old_success_cannot_renew_locator_after_newer_missed_image():
    tracker = CandidateHeadTracking()
    assert seed(tracker)
    assert not seed(tracker, observed_at_sec=10.6, now_sec=10.7, candidate_associated=False)
    assert not seed(tracker, observed_at_sec=10.4, now_sec=10.8)
    assert tracker.last_metadata["reason"] == "candidate_head_seed_nonadvancing_image"
    assert search(tracker, observed_at_sec=11.) is None
