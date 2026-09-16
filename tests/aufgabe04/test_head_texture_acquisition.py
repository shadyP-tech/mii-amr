"""Complete current head borders survive internal texture without relaxing pose gates.

These are rendered current-pixel tests, not QR fixtures: the inset rectangles
have no finder-pattern structure or encoded payload. Their placement is defined
in the head plane so that containment also exercises perspective geometry.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import (
    MAX_RAW_VERIFICATIONS,
    acquire_cold_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis import head_cold_acquisition
from scripts.aufgabe04.perception.stand_axis.head_proposal_selection import select_verified_head
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from tests.aufgabe04.test_physical_head_pipeline import estimate, head_image, profile


def textured_head(profile, *, angle=25., count=3, missing_side=None):
    image, corners = head_image(profile, angle=angle, distance=.30)
    pixels = np.array([(point.u_px, point.v_px) for point in corners], np.float32)
    image[:] = 0
    for side, (first, second) in enumerate(zip(pixels, np.roll(pixels, -1, axis=0))):
        if side != missing_side:
            cv2.line(image, tuple(np.rint(first).astype(int)),
                     tuple(np.rint(second).astype(int)), (255, 255, 255), 1)
    transform = cv2.getPerspectiveTransform(
        np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), np.float32), pixels)
    for x, y in ((.10, .10), (.65, .12), (.12, .64), (.64, .64))[:count]:
        inset = np.array([((x, y), (x + .24, y),
                           (x + .24, y + .24), (x, y + .24))], np.float32)
        projected = cv2.perspectiveTransform(inset, transform)[0]
        cv2.polylines(image, [np.rint(projected).astype(np.int32)], True,
                      (255, 255, 255), 1)
    return image, corners


@pytest.mark.parametrize("count", (3, 4))
@pytest.mark.parametrize("angle", (25., 45.))
def test_complete_perspective_head_with_repeated_insets_acquires_with_bounded_work(
        profile, count, angle):
    image, corners = textured_head(profile, angle=angle, count=count)
    original = image.copy()
    acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.considered_proposals > MAX_RAW_VERIFICATIONS
    assert acquired.proposal is not None, acquired
    assert 0 < acquired.raw_verifications <= MAX_RAW_VERIFICATIONS
    # The measurement must follow the complete enclosing head, not a finder-
    # sized rectangle or a convenient union of several internal rectangles.
    for actual, expected in zip(acquired.proposal.corners, corners):
        assert abs(actual.u_px - expected.u_px) < 4.
        assert abs(actual.v_px - expected.v_px) < 4.
    fitted, debug = estimate(profile, image)
    assert fitted.usable, fitted.reason
    assert abs(fitted.yaw_deg + angle) < 3.
    assert fitted.source == "model_current_measured_head"
    assert debug.head_model_quality.outer_border_verified
    assert debug.head_neck_junction is None
    np.testing.assert_array_equal(image, original)


def test_independent_second_head_remains_ambiguous(profile):
    image, _ = head_image(profile, angle=25., distance=.30)
    other, _ = head_image(profile, angle=45., distance=.40)
    shifted = cv2.warpAffine(other, np.array(((1., 0., 220.), (0., 1., 0.))),
                             (image.shape[1], image.shape[0]))
    image = cv2.max(image, shifted)
    acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.proposal is None
    assert acquired.reason == "head_proposal_ambiguous"
    fitted, _ = estimate(profile, image)
    assert not fitted.usable
    assert fitted.yaw_deg is None


@pytest.mark.parametrize("shift", (-220., 220.))
@pytest.mark.parametrize("distance", (.26, .50))
@pytest.mark.parametrize("reverse_contours", (False, True))
def test_textured_head_does_not_hide_separate_larger_or_smaller_head(
        profile, shift, distance, reverse_contours):
    image, _ = textured_head(profile)
    other, _ = head_image(profile, angle=25., distance=distance)
    shifted = cv2.warpAffine(other, np.array(((1., 0., shift), (0., 1., 0.))),
                             (image.shape[1], image.shape[0]))
    image = cv2.max(image, shifted)
    original_find_contours = cv2.findContours

    def find_contours(*args, **kwargs):
        contours, hierarchy = original_find_contours(*args, **kwargs)
        # Acquisition uses RETR_LIST without hierarchy. Reversing that search
        # order must not convert a separate stand into the selected texture.
        return (tuple(reversed(contours)) if reverse_contours else contours), hierarchy

    with patch.object(cv2, "findContours", side_effect=find_contours):
        acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.proposal is None
    if acquired.reason == "head_cold_acquisition_verification_budget_exceeded":
        # Distinct raw rail families are no longer erased by approximate head
        # size. A complex two-head scene may exhaust its fixed comparison cap.
        assert acquired.raw_verifications == MAX_RAW_VERIFICATIONS
        assert acquired.joint_border_diagnostics["unverified_independent_hypotheses"] > 0
    else:
        assert acquired.reason == "head_proposal_ambiguous"
    assert acquired.raw_verifications <= MAX_RAW_VERIFICATIONS


def _normalized_proposals(rectangles):
    """Supply geometric selector inputs; this helper does not certify pixels."""
    outer_pixels = np.array(((100., 80.), (420., 60.),
                              (420., 400.), (100., 380.)), np.float32)
    transform = cv2.getPerspectiveTransform(
        np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), np.float32), outer_pixels)
    proposals = [SimpleNamespace(corners=tuple(ImagePoint(*point) for point in outer_pixels))]
    for points in rectangles:
        transformed = cv2.perspectiveTransform(np.array([points], np.float32), transform)[0]
        proposals.append(SimpleNamespace(corners=tuple(ImagePoint(*point) for point in transformed)))
    return proposals


def _box(x, y, size=.24):
    return np.array(((x, y), (x + size, y),
                     (x + size, y + size), (x, y + size)))


@pytest.mark.parametrize("kind", ("misaligned", "overlapping", "central"))
@pytest.mark.parametrize("reverse_order", (False, True))
def test_three_nested_shapes_need_distributed_disjoint_aligned_texture(kind, reverse_order):
    rectangles = [_box(.10, .10), _box(.65, .12), _box(.12, .64)]
    positive = _normalized_proposals(rectangles)
    selected, reason, _ = select_verified_head(cv2, positive)
    assert selected is positive[0]
    assert reason == "current_head_proposal"
    if kind == "misaligned":
        angle = np.deg2rad(25.)
        rotation = np.array(((np.cos(angle), -np.sin(angle)),
                             (np.sin(angle), np.cos(angle))))
        center = rectangles[0].mean(axis=0)
        rectangles[0] = (rectangles[0] - center) @ rotation.T + center
    elif kind == "overlapping":
        rectangles = [_box(.25, .07, .26), _box(.47, .07, .26), _box(.07, .62, .26)]
    else:
        rectangles[2] = _box(.40, .55)
    proposals = _normalized_proposals(rectangles)
    if reverse_order:
        proposals.reverse()
    selected, reason, diagnostics = select_verified_head(cv2, proposals)
    assert selected is None
    assert reason == "head_proposal_ambiguous"
    assert diagnostics["selection"] == "distinct_current_heads_ambiguous"


@pytest.mark.parametrize("count", (MAX_RAW_VERIFICATIONS + 1, MAX_RAW_VERIFICATIONS + 5))
def test_unverified_independent_corner_supported_families_exhaust_raw_budget(count):
    raw = np.zeros((360, 520), np.uint8)
    hints = []
    for index in range(count):
        x, y = 25 + 90 * (index % 5), 25 + 75 * (index // 5)
        corners = ((x, y), (x + 40, y), (x + 40, y + 40), (x, y + 40))
        cv2.polylines(raw, [np.array(corners, np.int32)], True, 255, 1)
        hints.append(corners)
    image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    # Supply exactly N independent locator families. Corner support and strict
    # four-border refinement still read their real rasterized current pixels.
    with patch.object(cv2, "findContours", return_value=([], None)), \
         patch.object(head_cold_acquisition, "_rail_endpoint_hints",
                      return_value=(tuple(hints), (0, 0))), \
         patch.object(head_cold_acquisition, "refine_projected_head_border",
                      wraps=head_cold_acquisition.refine_projected_head_border) as refine:
        acquired = acquire_cold_head_proposal(cv2, image, raw_edges=raw)
    assert acquired.proposal is None
    assert acquired.reason == "head_cold_acquisition_verification_budget_exceeded"
    assert acquired.raw_verifications == MAX_RAW_VERIFICATIONS
    assert refine.call_count == MAX_RAW_VERIFICATIONS
    diagnostics = acquired.joint_border_diagnostics
    assert diagnostics["border_families"] == count
    assert diagnostics["direct_corner_supported_hypotheses"] == count
    assert diagnostics["unverified_independent_hypotheses"] == count - MAX_RAW_VERIFICATIONS
    assert all(item["accepted"] for item in diagnostics["strict_verifications"])


def test_one_nested_head_cannot_be_dismissed_as_repeated_texture(profile):
    image, corners = head_image(profile, angle=25., distance=.30)
    points = np.array([(point.u_px, point.v_px) for point in corners])
    center = points.mean(axis=0)
    nested = center + .40 * (points - center)
    cv2.polylines(image, [np.rint(nested).astype(np.int32)], True,
                  (255, 255, 255), 1)
    acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.proposal is None
    assert acquired.reason == "head_proposal_ambiguous"
    fitted, _ = estimate(profile, image)
    assert not fitted.usable
    assert fitted.yaw_deg is None


@pytest.mark.parametrize("side", range(4))
def test_internal_rectangles_cannot_complete_a_missing_outer_border(profile, side):
    image, _ = textured_head(profile, missing_side=side)
    acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.proposal is None
    fitted, _ = estimate(profile, image)
    assert not fitted.usable
    assert fitted.yaw_deg is None


@pytest.mark.parametrize("clip_side", ("left", "right"))
def test_internal_rectangles_cannot_complete_a_clipped_outer_head(profile, clip_side):
    image, corners = textured_head(profile)
    if clip_side == "left":
        image = image[:, int(min(point.u_px for point in corners)) + 8:]
    else:
        image = image[:, :int(max(point.u_px for point in corners)) - 8]
    acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.proposal is None
    fitted, _ = estimate(profile, image)
    assert not fitted.usable
    assert fitted.yaw_deg is None


def test_decoded_qr_size_and_identity_do_not_change_textured_head_angle(profile):
    image, corners = textured_head(profile)
    baseline, _ = estimate(profile, image)
    assert baseline.usable, baseline.reason
    for scale, identity in ((.20, "QR_003"), (.95, "QR_003"), (1.4, "QR_004")):
        quad = tuple((400. + scale * (point.u_px - 400.),
                      300. + scale * (point.v_px - 300.)) for point in corners)
        fitted, debug = estimate(profile, image,
            qr_observations=(DecodedQrObservation(identity, quad, "test"),))
        assert fitted.usable, fitted.reason
        assert fitted.yaw_deg == baseline.yaw_deg
        assert fitted.corners == baseline.corners
        assert debug.head_marker_boundary is None
        assert debug.head_neck_junction is None


def test_complete_aligned_head_still_requires_a_constrained_physical_angle(profile):
    image, _ = head_image(profile, angle=0., distance=.70)
    acquired = acquire_cold_head_proposal(cv2, image)
    assert acquired.proposal is not None, acquired
    fitted, debug = estimate(profile, image)
    assert not fitted.usable
    assert fitted.yaw_deg is None
    assert debug.head_model_quality.outer_border_verified
    assert not debug.head_model_quality.accepted
