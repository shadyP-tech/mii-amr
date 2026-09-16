"""Viewer locator proposals keep candidate bounds and exact crop offsets."""

from dataclasses import replace
from unittest.mock import patch

import numpy
import pytest

from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposal, HeadProposalResult
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.head_acquisition_parity import acquire_viewer_candidate_head


MODULE = "scripts.aufgabe04.real_robot.observer.head_acquisition_parity."


def proposal():
    corners = tuple(ImagePoint(x, y) for x, y in ((250., 150.), (350., 150.), (350., 250.), (250., 250.)))
    return HeadProposal(corners, (238, 138, 363, 293), (250., 150., 350., 250.),
                        300., 200., 100., 1., 0., .99, .99)


def locate(result, **changes):
    diagnostics = {}
    options = dict(expected_center=(300., 200.), expected_height=100., max_center_offset_ratio=1.5,
                   edge_preprocess="channel_union", canny_low=20, canny_high=60,
                   deadline_monotonic_sec=10.4, diagnostics=diagnostics)
    with patch(MODULE + "time.monotonic", return_value=10.), patch(
            MODULE + "acquire_cold_head_proposal", return_value=result) as cold:
        found = acquire_viewer_candidate_head(object(), numpy.zeros((480, 640, 3)), **{**options, **changes})
    return found, diagnostics, cold


def test_candidate_bounds_precede_comparison_without_clipping_displaced_head():
    found, diagnostics, cold = locate(HeadProposalResult(proposal(), "current_head_proposal", 4, 3))
    assert diagnostics["crop_xyxy"] == (0, 0, 640, 480)
    assert cold.call_args.args[1].shape == (480, 640, 3)
    assert cold.call_args.kwargs["deadline_monotonic_sec"] == pytest.approx(10.375)
    assert cold.call_args.kwargs["expected_head_center_u_px"] == 300.
    assert cold.call_args.kwargs["expected_head_center_v_px"] == 200.
    assert cold.call_args.kwargs["expected_head_height_px"] == 100.
    assert cold.call_args.kwargs["expected_head_height_tolerance_ratio"] == .30
    assert cold.call_args.kwargs["max_center_offset_ratio"] == 1.5
    assert diagnostics["candidate_bounds_applied_before_comparison"] is True
    assert found.proposal.center_u_px == 300.
    assert found.proposal.center_v_px == 200.
    assert found.proposal.corners[0] == ImagePoint(250., 150.)
    assert found.proposal.bounds_xyxy == (238, 138, 363, 293)
    assert found.proposal.expected_height_ratio == 1.
    assert diagnostics["angle_authorized"] is False


def test_projection_size_still_rejects_an_unrelated_complete_head():
    found, diagnostics, _ = locate(HeadProposalResult(
        replace(proposal(), observed_height_px=150.), "current_head_proposal"))
    assert found.proposal is None
    assert diagnostics["reason"] == "viewer_head_outside_candidate_projection"


def test_insufficient_budget_does_not_start_another_locator():
    found, diagnostics, cold = locate(None, deadline_monotonic_sec=10.04)
    assert found is None
    assert diagnostics["performed"] is False
    cold.assert_not_called()


def test_partial_comparison_diagnostics_never_become_proposal_authority():
    result = HeadProposalResult(None, "head_acquisition_deadline_exceeded", 8, 3,
        joint_border_diagnostics={"deadline_stage": "cold_strict_verification", "comparison_complete": False})
    found, diagnostics, _ = locate(result)
    assert found is result
    assert found.proposal is None
    assert diagnostics["comparison"]["comparison_complete"] is False


@pytest.mark.parametrize("missing_side", (None, 0, 1, 2, 3))
def test_shared_locator_requires_current_complete_border_pixels(missing_side):
    cv2 = pytest.importorskip("cv2")
    from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
    frame = frame_with_heads(missing_side=missing_side)
    original = frame.copy()
    diagnostics = {}
    # Deterministic clock tests image evidence rather than machine speed.
    with patch(MODULE + "time.monotonic", return_value=10.):
        found = acquire_viewer_candidate_head(
            cv2, frame, expected_center=(190., 130.), expected_height=100.,
            max_center_offset_ratio=1.5, edge_preprocess="channel_union",
            canny_low=20, canny_high=60, deadline_monotonic_sec=10.5,
            diagnostics=diagnostics)
    numpy.testing.assert_array_equal(frame, original)
    assert diagnostics["performed"]
    assert diagnostics["motion_authorized"] is False
    if missing_side is None:
        assert found.proposal is not None, found.reason
        assert found.proposal.center_u_px == pytest.approx(190., abs=2.)
        assert found.proposal.center_v_px == pytest.approx(130., abs=2.)
    else:
        assert found.proposal is None


def test_off_center_head_outside_old_center_crop_is_acquired_before_any_pose():
    cv2 = pytest.importorskip("cv2")
    from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
    # The old nominal +/-1.4-height crop ended at x=330 and clipped this head's
    # right border (x=360), although the center is inside the admitted bound.
    frame = frame_with_heads((((310, 130), (100, 100), 0),))
    diagnostics = {}
    with patch(MODULE + "time.monotonic", return_value=10.):
        result = acquire_viewer_candidate_head(cv2, frame,
            expected_center=(190., 130.), expected_height=100.,
            max_center_offset_ratio=1.5, edge_preprocess="channel_union",
            canny_low=20, canny_high=60, deadline_monotonic_sec=10.5,
            diagnostics=diagnostics)
    assert result.proposal is not None, result.reason
    assert result.proposal.center_u_px == pytest.approx(310., abs=2.)
    assert max(p.u_px for p in result.proposal.corners) > 350.
    assert diagnostics["candidate_bounds_applied_before_comparison"] is True


@pytest.mark.parametrize("changes", (
    {"expected_center": (float("nan"), 200.)},
    {"expected_height": 0.}, {"expected_height": float("inf")},
    {"max_center_offset_ratio": -1.},
))
def test_invalid_candidate_projection_never_starts_locator(changes):
    found, diagnostics, locator = locate(None, **changes)
    assert found is None
    assert diagnostics["performed"] is False
    locator.assert_not_called()
