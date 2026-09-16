"""Candidate constraints reduce work without selecting a printed or wrong rail."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis import head_cold_acquisition as cold
from scripts.aufgabe04.perception.stand_axis.head_border_families import (
    rail_signature, same_current_border_family, same_supporting_rails,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal import _extent
from scripts.aufgabe04.perception.stand_axis.head_proposal_selection import select_verified_head
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from tests.aufgabe04.test_head_cold_acquisition import frame_with_heads
from tests.aufgabe04.test_head_texture_acquisition import _box, _normalized_proposals


def _bounds(**changes):
    values = dict(expected_head_center_u_px=190., expected_head_center_v_px=130.,
                  expected_head_height_px=100.)
    values.update(changes)
    return values


def test_candidate_scale_filters_room_and_small_texture_before_strict_comparison():
    frame = frame_with_heads()
    for x in (320, 410):
        for y in (25, 110, 195, 280):
            cv2.rectangle(frame, (x, y), (x + 30, y + 30), (200, 200, 200), 2)
    cv2.rectangle(frame, (285, 30), (495, 330), (200, 200, 200), 2)
    measured_inputs = []
    original = cold.refine_projected_head_border

    def refine(cv2, raw, corners, **kwargs):
        measured_inputs.append(_extent(corners))
        return original(cv2, raw, corners, **kwargs)

    with patch.object(cold, "refine_projected_head_border", side_effect=refine):
        result = cold.acquire_cold_head_proposal(cv2, frame, **_bounds())
    assert result.proposal is not None, result
    assert result.proposal.center_u_px == pytest.approx(190., abs=3.)
    assert result.raw_verifications <= cold.MAX_RAW_VERIFICATIONS
    assert result.joint_border_diagnostics["candidate_bounds_rejections"] > 0
    assert all(65. <= height <= 135. for _width, height, _center in measured_inputs)


def test_bounded_off_center_head_is_measured_not_replaced_by_projection():
    frame = frame_with_heads((((245, 130), (100, 100), 0),))
    result = cold.acquire_cold_head_proposal(cv2, frame, **_bounds())
    assert result.proposal is not None, result
    assert result.proposal.center_u_px == pytest.approx(245., abs=3.)
    assert result.proposal.center_offset_head_heights == pytest.approx(.55, abs=.04)


@pytest.mark.parametrize("invalid", ({"expected_head_height_px": 0.},
                                    {"expected_head_center_v_px": None},
                                    {"expected_head_height_tolerance_ratio": 1.},
                                    {"max_center_offset_ratio": float("nan")}))
def test_invalid_partial_or_unbounded_candidate_prior_does_not_start_search(invalid):
    with patch.object(cv2, "findContours") as contours:
        result = cold.acquire_cold_head_proposal(cv2, frame_with_heads(), **_bounds(**invalid))
    assert result.reason == "head_proposal_input_invalid"
    contours.assert_not_called()


def test_two_same_scale_heads_inside_candidate_bounds_remain_ambiguous():
    frame = frame_with_heads((((125, 130), (70, 70), 0), ((255, 130), (70, 70), 0)))
    result = cold.acquire_cold_head_proposal(cv2, frame,
        **_bounds(expected_head_height_px=70., max_center_offset_ratio=1.5))
    assert result.proposal is None
    assert result.reason == "head_proposal_ambiguous"


def test_current_association_filter_is_applied_before_head_selection():
    frame = frame_with_heads((((125, 130), (70, 70), 0), ((255, 130), (70, 70), 0)))
    result = cold.acquire_cold_head_proposal(cv2, frame,
        **_bounds(expected_head_height_px=70., max_center_offset_ratio=1.5),
        proposal_filter=lambda proposal: proposal.center_u_px > 190.)
    assert result.proposal is not None, result
    assert result.proposal.center_u_px == pytest.approx(255., abs=3.)
    assert result.joint_border_diagnostics["candidate_association_rejections"] > 0


def test_texture_cross_pairs_share_four_current_inset_rails_without_becoming_heads():
    rectangles = [_box(.10, .10), _box(.65, .10), _box(.10, .64)]
    rectangles.extend((np.array(((.10, .10), (.34, .10), (.34, .88), (.10, .88))),
                       np.array(((.10, .10), (.89, .10), (.89, .34), (.10, .34)))))
    proposals = _normalized_proposals(rectangles)
    selected, reason, _ = select_verified_head(cv2, proposals)
    assert selected is proposals[0]
    assert reason == "current_head_proposal"
    # A different complete internal boundary cannot be explained just because
    # three texture anchors happen to exist somewhere inside the outer head.
    proposals.extend(_normalized_proposals((_box(.18, .16, .61),))[1:])
    selected, reason, _ = select_verified_head(cv2, proposals)
    assert selected is None
    assert reason == "head_proposal_ambiguous"


def test_nearby_same_polarity_physical_rails_are_not_collapsed_by_head_size():
    frame = np.zeros((200, 200, 3), np.uint8)
    cv2.rectangle(frame, (40, 40), (140, 140), (90, 90, 90), -1)
    cv2.rectangle(frame, (44, 44), (136, 136), (180, 180, 180), -1)
    outer = tuple(ImagePoint(*point) for point in ((39., 39.), (140., 39.), (140., 140.), (39., 140.)))
    inner = tuple(ImagePoint(*point) for point in ((43., 43.), (136., 43.), (136., 136.), (43., 136.)))
    raw = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                  blur_kernel=1, canny_low=20, canny_high=60)
    assert not same_supporting_rails(rail_signature(raw, outer), rail_signature(raw, inner))
    assert not same_current_border_family(outer, inner, raw_edges=raw, frame_bgr=frame)
    selected, reason, _ = select_verified_head(cv2,
        (SimpleNamespace(corners=outer), SimpleNamespace(corners=inner)),
        raw_edges=raw, frame_bgr=frame)
    assert selected is None
    assert reason == "head_proposal_ambiguous"


def test_two_canny_edges_of_one_thin_current_rail_share_a_family():
    frame = frame_with_heads()
    raw = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                  blur_kernel=5, canny_low=20, canny_high=60)
    outer = tuple(ImagePoint(*point) for point in ((138., 78.), (242., 78.), (242., 182.), (138., 182.)))
    inner = tuple(ImagePoint(*point) for point in ((142., 82.), (238., 82.), (238., 178.), (142., 178.)))
    assert same_current_border_family(outer, inner, raw_edges=raw, frame_bgr=frame)
    assert not same_current_border_family(outer, inner)  # Similar geometry alone is insufficient.


def test_opposite_gradients_do_not_merge_two_separate_bright_stripes():
    from scripts.aufgabe04.perception.stand_axis.head_border_families import _single_thin_stripe
    frame = np.zeros((50, 120, 3), np.uint8)
    frame[20:22] = 255
    frame[24:26] = 255
    assert not _single_thin_stripe(frame, ImagePoint(10., 20.), ImagePoint(100., 20.),
                                  ImagePoint(10., 26.), ImagePoint(100., 26.))


def test_border_family_does_not_chain_across_alternating_bright_and_dark_rails():
    frame = np.zeros((200, 200, 3), np.uint8)
    proposals = []
    for offset, value in ((0, 255), (2, 0), (4, 255), (6, 0)):
        low, high = 40 + offset, 160 - offset
        cv2.rectangle(frame, (low, low), (high, high), (value, value, value), -1)
        proposals.append(SimpleNamespace(corners=tuple(ImagePoint(*point) for point in (
            (low-.5, low-.5), (high+.5, low-.5), (high+.5, high+.5), (low-.5, high+.5)))))
    raw = _canny_edges_from_frame(cv2, frame, edge_preprocess="channel_union",
                                  blur_kernel=1, canny_low=20, canny_high=60)
    selected, reason, _ = select_verified_head(cv2, proposals, raw_edges=raw, frame_bgr=frame)
    assert selected is None
    assert reason == "head_proposal_ambiguous"


def test_infinite_texture_rail_lines_do_not_explain_a_separate_central_box():
    proposals = _normalized_proposals((_box(.10, .10, .20), _box(.70, .10, .20),
                                       _box(.10, .70, .20), _box(.30, .30, .40)))
    selected, reason, _ = select_verified_head(cv2, proposals)
    assert selected is None
    assert reason == "head_proposal_ambiguous"


@pytest.mark.parametrize("missing_right", (False, True))
def test_four_current_rail_intersections_recover_fragmented_locator_endpoints(missing_right):
    from scripts.aufgabe04.perception.stand_axis.head_rail_intersections import candidate_rail_intersections
    from scripts.aufgabe04.perception.stand_axis.head_search_bounds import HeadSearchBounds
    from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
    raw = np.zeros((200, 200), np.uint8)
    points = np.array(((40, 40), (140, 40), (140, 140), (40, 140)))
    for index, (a, b) in enumerate(zip(points, np.roll(points, -1, axis=0))):
        if not missing_right or index != 1:
            cv2.line(raw, tuple(a), tuple(b), 255, 1)
    # Independent observed fragments do not provide complete endpoints; four
    # line intersections only locate the actual complete current corner pixels.
    groups = ([(50., (60., 40.), (110., 40.)), (65., (60., 140.), (125., 140.))],
              [(55., (40., 65.), (40., 120.)), (55., (140., 60.), (140., 115.))])
    bounds = HeadSearchBounds((90., 90.), 100., .70, .35)
    hints = candidate_rail_intersections(cv2, raw, groups, bounds)
    if missing_right:
        assert not hints
        return
    assert len(hints) == 1
    np.testing.assert_allclose(hints[0], points)
    measured = refine_projected_head_border(cv2, raw,
        tuple(ImagePoint(*point) for point in hints[0]), corridor_half_width_px=4.)
    assert measured.accepted


def test_cold_candidate_rescues_no_endpoint_proposal_with_same_strict_budget():
    frame = frame_with_heads()
    groups = ([(60., (160., 78.), (220., 78.)), (60., (160., 182.), (220., 182.))],
              [(60., (138., 100.), (138., 160.)), (60., (242., 100.), (242., 160.))])

    def fragments(*args, **kwargs):
        kwargs["rail_groups_out"].extend(groups)
        return (), (2, 2)

    with patch.object(cv2, "findContours", return_value=([], None)), \
            patch.object(cold, "_rail_endpoint_hints", side_effect=fragments):
        result = cold.acquire_cold_head_proposal(cv2, frame, **_bounds())
    assert result.proposal is not None, result
    assert result.joint_border_diagnostics["four_rail_rescue_attempted"]
    assert result.raw_verifications == 1
    assert result.joint_border_diagnostics["strict_verifications"][0]["locator"] == "four_current_rails"


def test_four_rail_rescue_cannot_replace_an_existing_verified_head():
    with patch.object(cold, "candidate_rail_intersections", side_effect=AssertionError("unexpected rescue")):
        result = cold.acquire_cold_head_proposal(cv2, frame_with_heads(), **_bounds())
    assert result.proposal is not None
    assert not result.joint_border_diagnostics.get("four_rail_rescue_attempted", False)


def test_four_rail_rescue_does_not_reset_an_exhausted_strict_verification_budget():
    failed = SimpleNamespace(accepted=False, reason="model_corner_evidence_insufficient", corners=None)
    with patch.object(cold, "MAX_RAW_VERIFICATIONS", 1), \
            patch.object(cold, "refine_projected_head_border", return_value=failed), \
            patch.object(cold, "candidate_rail_intersections", side_effect=AssertionError("budget reset")):
        result = cold.acquire_cold_head_proposal(cv2, frame_with_heads(), **_bounds())
    assert result.proposal is None
    assert result.raw_verifications == 1
    assert result.reason == "head_cold_acquisition_verification_budget_exceeded"


def test_candidate_bound_search_retains_deadline_and_does_not_return_partial_winner():
    clock = [0.]
    original = cold.refine_projected_head_border

    def refine(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 2.
        return result

    with patch("scripts.aufgabe04.perception.stand_axis.head_acquisition_budget.time.monotonic",
               side_effect=lambda: clock[0]), \
            patch.object(cold, "refine_projected_head_border", side_effect=refine):
        result = cold.acquire_cold_head_proposal(cv2, frame_with_heads(),
                                               deadline_monotonic_sec=1., **_bounds())
    assert result.reason == "head_acquisition_deadline_exceeded"
    assert result.proposal is None
