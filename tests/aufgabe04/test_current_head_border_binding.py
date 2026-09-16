"""Accepted geometry must retain the physical border selected on this image."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.current_head_border_binding import (
    BORDER_BINDING_REJECTED, bind_selected_current_head,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.physical_head_pipeline import fit_physical_head_in_frame
from tests.aufgabe04.test_head_backside_classification import classified_head

PREFIX = "scripts.aufgabe04.perception.stand_axis.physical_head_pipeline."


def rectangle(low, high):
    return tuple(ImagePoint(float(x), float(y)) for x, y in (
        (low, low), (high, low), (high, high), (low, high)))


@pytest.fixture
def scene():
    # Two distinct, same-polarity physical boundaries. They are not the two
    # opposite gradients of one thin painted rail.
    frame = np.zeros((200, 200, 3), np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (80, 80, 80), -1)
    cv2.rectangle(frame, (55, 55), (145, 145), (160, 160, 160), -1)
    raw = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 20, 60)
    estimate, debug, _ = classified_head(u=100., v=100., height=100., yaw_deg=15.)
    return frame, raw, (estimate, debug, object())


def test_independent_inset_fit_cannot_inherit_selected_outer_border(scene):
    frame, raw, (estimate, debug, pose) = scene
    estimate = replace(estimate, corners=rectangle(55., 145.))
    result, binding = bind_selected_current_head((estimate, debug, pose), rectangle(50., 150.),
                                                raw_edges=raw, frame_bgr=frame)
    rejected, rejected_debug, rejected_pose = result
    assert binding["performed"] and not binding["accepted"]
    assert rejected.reason == BORDER_BINDING_REJECTED
    assert not rejected.usable
    assert rejected.yaw_deg is None
    assert rejected.camera_face_normal_xyz is None
    assert rejected.visible_face is None
    assert rejected_pose is None
    assert rejected_debug.model_pose is None
    assert rejected_debug.projected_landmarks is None
    assert rejected_debug.head_model_quality.accepted is False
    assert rejected_debug.head_model_quality.outer_border_verified is False
    assert rejected_debug.head_outer_recovery is None
    assert rejected_debug.head_orientation_bounds is None
    assert rejected_debug.head_pose_hypotheses is None


def test_subpixel_refinement_on_same_current_rails_preserves_fit(scene):
    frame, raw, (estimate, debug, pose) = scene
    result = (replace(estimate, corners=rectangle(50.4, 149.6)), debug, pose)
    bound, binding = bind_selected_current_head(result, rectangle(50., 150.),
                                               raw_edges=raw, frame_bgr=frame)
    assert bound is result
    assert binding["accepted"]
    assert binding["reason"] == "current_selected_border_preserved"
    assert binding["historical_measurement_reused"] is False


def test_incomplete_border_fit_and_missing_corners_keep_original_reason(scene):
    frame, raw, (estimate, debug, pose) = scene
    failed_debug = replace(debug, head_model_quality=replace(debug.head_model_quality,
        accepted=False, outer_border_verified=False, raw_corner_support_accepted=False))
    for changes in ({"usable": False, "reason": "model_corner_evidence_insufficient"},
                    {"corners": None}):
        result = (replace(estimate, **changes), failed_debug, pose)
        bound, binding = bind_selected_current_head(result, rectangle(50., 150.),
                                                   raw_edges=raw, frame_bgr=frame)
        assert bound is result
        assert not binding["performed"]


def test_ambiguous_angle_cannot_grant_appearance_or_interval_on_different_border(scene):
    frame, raw, (estimate, debug, pose) = scene
    estimate = replace(estimate, usable=False, corners=rectangle(55., 145.),
                       yaw_deg=None, reason="head_model_planar_axis_ambiguous")
    debug = replace(debug, head_model_quality=replace(debug.head_model_quality,
        accepted=False, reason="head_model_planar_axis_ambiguous"),
        head_orientation_bounds=object(), head_backside_appearance=object())
    result, binding = bind_selected_current_head((estimate, debug, pose), rectangle(50., 150.),
                                                raw_edges=raw, frame_bgr=frame)
    assert binding["performed"] and not binding["accepted"]
    assert result[0].reason == BORDER_BINDING_REJECTED
    assert result[1].head_orientation_bounds is None
    assert result[1].head_backside_appearance is None
    assert result[1].head_pose_hypotheses is None
    # Same selected borders retain the honest unresolved-angle reason.
    bound, binding = bind_selected_current_head((estimate, debug, pose), rectangle(55., 145.),
                                               raw_edges=raw, frame_bgr=frame)
    assert binding["accepted"]
    assert bound[0].reason == "head_model_planar_axis_ambiguous"


def invoke(scene, *, provided=False, verified=False):
    frame, raw, (estimate, debug, pose) = scene
    estimate = replace(estimate, corners=rectangle(55., 145.))
    selected = rectangle(50., 150.)
    proposal = HeadProposal(selected, (40, 40, 160, 180), (50., 50., 150., 150.),
                            100., 100., 100., 1., 0., 1., 1.)
    with patch(PREFIX + "fit_current_measured_head", return_value=(estimate, debug, pose)), \
         patch(PREFIX + "acquire_cold_head_proposal",
               return_value=HeadProposalResult(proposal, "current_head_proposal")) as acquire:
        result = fit_physical_head_in_frame(cv2, frame, raw,
            model_profile=SimpleNamespace(sha256=estimate.model_profile_sha256, measurement_status="measured"),
            camera=object(), timing=Mock(),
            current_head_proposal_corners=selected if provided else None,
            current_head_proposal_verified=verified)
    assert acquire.call_count == (0 if provided else 1)
    return result


def test_shared_cold_selection_always_binds_current_fitted_rails(scene):
    estimate, debug, pose = invoke(scene)
    assert estimate.reason == BORDER_BINDING_REJECTED
    assert pose is None
    assert debug.head_acquisition_diagnostics["selected_border_binding"]["accepted"] is False


def test_registered_current_proposal_binds_but_manual_locator_seed_does_not(scene):
    estimate, debug, pose = invoke(scene, provided=True, verified=True)
    assert estimate.reason == BORDER_BINDING_REJECTED
    assert pose is None
    assert debug.head_acquisition_diagnostics["selected_border_binding"]["performed"]
    manual, manual_debug, manual_pose = invoke(scene, provided=True)
    assert manual.usable
    assert manual_pose is not None
    assert "selected_border_binding" not in manual_debug.head_acquisition_diagnostics
