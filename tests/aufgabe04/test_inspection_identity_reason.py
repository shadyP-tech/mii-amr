"""Identity-only failures survive advisory classification and artifact binding."""

import math

import pytest

from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    build_candidate_inspection_observation,
    validate_candidate_inspection_observation,
)
from scripts.aufgabe04.real_robot.observer.inspection_progress import (
    InspectionProgress,
    classify_inspection_progress,
)


IDENTITY_PENDING = "measured_head_front_identity_unresolved"


def details(*, estimator_reason=False, bounded_hint=False):
    value = {"reason": IDENTITY_PENDING}
    if estimator_reason:
        value["estimator_reason"] = "model_head_yaw_weakly_observable"
    if bounded_hint:
        value["stand_axis_debug"] = {
            "bounded_head_view_hint": {
                "purpose": "orientation_disambiguation",
                "candidate_associated": True,
                "source_fresh": True,
                "camera_relative_yaw_rad": math.radians(17.317),
                "orientation_half_width_rad": math.radians(4.912),
            },
        }
    return value


@pytest.mark.parametrize("estimator_reason,bounded_hint", [
    (False, False), (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("front_classification", [None, "front_unreadable"])
def test_explicit_identity_failure_survives_diagnostic_reason_overrides(
        estimator_reason, bounded_hint, front_classification):
    source = details(estimator_reason=estimator_reason, bounded_hint=bounded_hint)
    if front_classification is not None:
        source["front_observation"] = {
            "axis_state": "unresolved", "classification": front_classification,
        }

    result = classify_inspection_progress("axis_observation_not_committable", source)

    assert result.reason == IDENTITY_PENDING
    assert result.classification == (front_classification or "unobservable")
    assert result.camera_relative_yaw_rad == (
        math.radians(17.317) if bounded_hint else None)


def test_identity_failure_remains_in_accumulated_hashed_advisory():
    progress = InspectionProgress()
    source = details(estimator_reason=True, bounded_hint=True)
    classification = classify_inspection_progress("axis_observation_not_committable", source)
    for index in range(7):
        fields = progress.record(
            frame_stamp_sec=100. + .5 * index,
            robot_pose={"x_m": 1., "y_m": 2., "yaw_rad": .1},
            frame_accepted=True, poisoned=False, classification=classification,
            now_monotonic_sec=float(index),
        )
    assert fields is not None

    artifact = build_candidate_inspection_observation(
        candidate_uid="survey_candidate_0003", stream_id="inspection_identity_reason",
        planning_frame="map", stand_center={"x_m": 1.5, "y_m": 2.},
        robot_profile_sha256="a" * 64,
        calibration_profile_sha256="b" * 64,
        stand_model_profile_sha256="c" * 64,
        **fields,
    )
    validated = validate_candidate_inspection_observation(artifact)

    assert validated["reasons"] == [IDENTITY_PENDING]
    assert validated["classification"] == "unobservable"
    assert validated["camera_relative_yaw_rad"] == math.radians(17.317)
    assert validated["angle_authority"] == "advisory_camera_relative_only"
    assert validated["motion_authorized"] is False


def test_other_failures_keep_existing_bounded_orientation_reason():
    source = details(estimator_reason=True, bounded_hint=True)
    source["reason"] = "current_head_geometry_unavailable"

    result = classify_inspection_progress("evidence_not_committable", source)

    assert result.reason == "bounded_head_orientation_disambiguation"
    assert result.camera_relative_yaw_rad == math.radians(17.317)
