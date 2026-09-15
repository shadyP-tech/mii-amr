"""Bounded current-pixel head acquisition without QR, neck, or pose seeds.

This locator supplies neutral 2D corners to the measured physical head solver.
It does not assign a face, identity, angle, or navigation target. Closed contour
and paired rail endpoints are search hints only; every accepted proposal passes
the same current raw four-border and corner-arm checks as seeded acquisition.
"""

from __future__ import annotations

import math

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _distance, _polygon_area, _well_formed_quadrilateral, order_corners,
)
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MIN_HEAD_EDGE_PX
from scripts.aufgabe04.perception.stand_axis.head_proposal import (
    HeadProposalResult, _extent, _proposal,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal_selection import (
    rank_current_head_hypotheses, select_verified_head, uncovered_head_hypotheses,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame

MAX_IMAGE_PIXELS = 1920 * 1080
MAX_CONTOURS = 1024
MAX_RAILS_PER_DIRECTION = 24
MAX_LOCATOR_HYPOTHESES = 256
MAX_RAW_VERIFICATIONS = 12
MAX_HEAD_IMAGE_AREA_FRACTION = .80
MIN_LOCATOR_BORDER_SUPPORT = .80
_LOCATOR = "cold_current_borders"


def _bounded_quad(points, shape):
    corners = order_corners(tuple(ImagePoint(float(x), float(y)) for x, y in points))
    if not _well_formed_quadrilateral(corners):
        return None
    rows, cols = shape[:2]
    if not all(3 <= p.u_px < cols - 3 and 3 <= p.v_px < rows - 3 for p in corners):
        return None
    width, height, _center = _extent(corners)
    if (min(_distance(a, b) for a, b in zip(corners, corners[1:] + corners[:1]))
            < MIN_HEAD_EDGE_PX or not 0. < width / height <= 1.35):
        return None
    # Rectification introduces a strong closed border around the whole image.
    # A cold locator needs visible context, so that canvas boundary is excluded.
    if _polygon_area(corners) > MAX_HEAD_IMAGE_AREA_FRACTION * rows * cols:
        return None
    return corners


def _rail_endpoint_hints(cv2, gray, raw_edges):
    """Pair bounded opposite current line segments, independent of head scale."""
    import numpy as np

    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    # Equal-luminance colour boundaries can be absent from grayscale LSD while
    # the caller's channel-union Canny has complete current evidence.
    raw_lines = cv2.HoughLinesP(raw_edges, 1., np.pi / 180., threshold=8,
                               minLineLength=16, maxLineGap=2)
    groups = ([], [])
    segments = [segment for lines in (detected, raw_lines) if lines is not None
                for segment in lines.reshape(-1, 4)]
    for x0, y0, x1, y1 in segments:
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < MIN_HEAD_EDGE_PX * .70:
            continue
        direction = 0 if abs(dy) <= .50 * abs(dx) else 1 if abs(dx) <= .50 * abs(dy) else None
        if direction is None:
            continue
        first, last = (float(x0), float(y0)), (float(x1), float(y1))
        if first[direction] > last[direction]:
            first, last = last, first
        groups[direction].append((length, first, last))
    hints = []
    for direction, group in enumerate(groups):
        # LSD and Hough often locate the same rail with different endpoints.
        # Collapse those search duplicates before allocating the fixed budget;
        # the retained longest fragment still supplies only a locator line.
        distinct = []
        for item in sorted(group, key=lambda item: (-item[0], item[1:])):
            length, a, b = item
            delta = (b[0] - a[0], b[1] - a[1])
            if any(
                abs(delta[0] * (d[1] - c[1]) - delta[1] * (d[0] - c[0])) / (length * other_length) < .02
                and max(abs(delta[0] * (p[1] - a[1]) - delta[1] * (p[0] - a[0])) / length for p in (c, d)) < 1.5
                and min(b[direction], d[direction]) - max(a[direction], c[direction]) > .5 * min(length, other_length)
                for other_length, c, d in distinct
            ):
                continue
            distinct.append(item)
        # Long room boundaries must not consume every slot before a small
        # stand's rails are seen. Spread the fixed budget across measured line
        # scales and horizontal image thirds, without favouring the centre.
        strata = {}
        for item in distinct:
            length, a, b = item
            key = (max(0, int(math.log2(length / MIN_HEAD_EDGE_PX))),
                   min(2, int(3. * (a[0] + b[0]) / (2. * gray.shape[1]))))
            strata.setdefault(key, []).append(item)
        groups[direction][:] = [bucket[index] for index in range(max(map(len, strata.values()), default=0))
                                for _key, bucket in sorted(strata.items()) if index < len(bucket)][:MAX_RAILS_PER_DIRECTION]
        group = groups[direction]
        cross = 1 - direction
        for index, (length, first, last) in enumerate(group):
            for other_length, other_first, other_last in group[index + 1:]:
                if min(length, other_length) < .20 * max(length, other_length):
                    continue
                separation = abs((first[cross] + last[cross] - other_first[cross] - other_last[cross]) / 2.)
                if not MIN_HEAD_EDGE_PX * .70 <= separation <= max(gray.shape):
                    continue
                # A neck touching the lower rail splits its locator into two
                # short segments. Use their union extent only as a search hint;
                # unchanged raw rail/corner checks must establish every pixel.
                start = min(first[direction], other_first[direction])
                end = max(last[direction], other_last[direction])
                overlap = min(last[direction], other_last[direction]) - max(first[direction], other_first[direction])
                if overlap < .15 * min(length, other_length) or end - start > 1.4 * max(length, other_length):
                    continue
                hints.append((first, last, other_last, other_first))
                def extended(a, b, value):
                    fraction = (value - a[direction]) / (b[direction] - a[direction])
                    point = [0., 0.]
                    point[direction] = value
                    point[cross] = a[cross] + fraction * (b[cross] - a[cross])
                    return tuple(point)
                hints.append((extended(first, last, start), extended(first, last, end),
                              extended(other_first, other_last, end), extended(other_first, other_last, start)))
    return tuple(hints), tuple(map(len, groups))


def acquire_cold_head_proposal(
    cv2, frame_bgr, *, raw_edges=None, edge_preprocess="channel_union",
    canny_low=20, canny_high=60,
) -> HeadProposalResult:
    """Locate a unique complete head in a bounded image without a prior pose.

    Contours and line hints locate, but never certify, boundaries. A clipped or
    open rectangle cannot become a measurement through locator interpolation.
    Current corner evidence and border families allocate the fixed work budget.
    Unverified independent families remain unresolved on budget exhaustion.
    """
    import numpy as np

    diagnostics = {
        "method": _LOCATOR, "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_contours": MAX_CONTOURS, "max_rails_per_direction": MAX_RAILS_PER_DIRECTION,
        "max_locator_hypotheses": MAX_LOCATOR_HYPOTHESES,
        "max_raw_verifications": MAX_RAW_VERIFICATIONS,
        "max_head_image_area_fraction": MAX_HEAD_IMAGE_AREA_FRACTION,
        "min_locator_border_support": MIN_LOCATOR_BORDER_SUPPORT,
        "angle_authorized": False, "motion_authorized": False,
        "selection": "unavailable", "strict_verifications": [],
    }

    def result(reason, proposal=None):
        return HeadProposalResult(
            proposal, reason, diagnostics.get("considered_proposals", 0),
            len(diagnostics["strict_verifications"]), _LOCATOR, diagnostics,
        )

    if (frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
            or min(frame_bgr.shape[:2]) < MIN_HEAD_EDGE_PX + 6):
        return result("head_proposal_input_invalid")
    if frame_bgr.shape[0] * frame_bgr.shape[1] > MAX_IMAGE_PIXELS:
        return result("head_cold_acquisition_image_budget_exceeded")
    if raw_edges is None:
        raw_edges = _canny_edges_from_frame(
            cv2, frame_bgr, edge_preprocess=edge_preprocess, blur_kernel=5,
            canny_low=canny_low, canny_high=canny_high,
        )
    if raw_edges.ndim != 2 or raw_edges.shape != frame_bgr.shape[:2]:
        raise ValueError("raw_edges must match the cold search image")
    contours = cv2.findContours(raw_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    diagnostics["contours"] = len(contours)
    if len(contours) > MAX_CONTOURS:
        return result("head_cold_acquisition_contour_budget_exceeded")
    seeds = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 4 * MIN_HEAD_EDGE_PX:
            continue
        for fraction in (.015, .025, .04):
            quad = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                seeds.append((quad.reshape(-1, 2), "closed_contour"))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hints, rail_counts = _rail_endpoint_hints(cv2, gray, raw_edges)
    diagnostics["horizontal_rails"], diagnostics["vertical_rails"] = rail_counts
    seeds.extend((hint, "paired_current_rails") for hint in hints)
    distance = cv2.distanceTransform(np.where(raw_edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    fractions = np.linspace(.10, .90, 24)
    hypotheses = {}
    for points, locator in seeds:
        corners = _bounded_quad(points, raw_edges.shape)
        if corners is None:
            continue
        points = np.asarray([(p.u_px, p.v_px) for p in corners])
        pixels = np.rint(points[:, None, :] + fractions[None, :, None]
                         * (np.roll(points, -1, axis=0) - points)[:, None, :]).astype(np.int32)
        support = np.mean(distance[pixels[:, :, 1], pixels[:, :, 0]] <= 4., axis=1)
        # Long paired hints can bridge separate texture boxes. Each of the
        # four borders must already be substantially present, rather than
        # relying on the downstream mean support to hide a missing interval.
        if min(support) < MIN_LOCATOR_BORDER_SUPPORT:
            continue
        key = tuple(round(value / 2.) for value in points.ravel())
        score = .70 * float(min(support)) + .30 * float(np.mean(support))
        if key not in hypotheses or score > hypotheses[key][0]:
            hypotheses[key] = (score, corners, locator)
    diagnostics["considered_proposals"] = len(hypotheses)
    if len(hypotheses) > MAX_LOCATOR_HYPOTHESES:
        return result("head_cold_acquisition_locator_budget_exceeded")
    # Group duplicate rail contours before spending the strict verification
    # budget. Retain distinct border locations: choosing inward versus outward
    # rails remains a measurement decision, never a proximity-to-centre prior.
    ranked = sorted(hypotheses.values(), key=lambda item: (-item[0], -_polygon_area(item[1]),
                                                         tuple((p.u_px, p.v_px) for p in item[1])))
    unique = []
    for item in ranked:
        if not any(max(_distance(a, b) for a, b in zip(item[1], other[1])) < 3. for other in unique):
            unique.append(item)
    diagnostics["distinct_locator_hypotheses"] = len(unique)
    ordered, ranking = rank_current_head_hypotheses(cv2, raw_edges, unique)
    diagnostics.update(ranking)
    accepted = []
    for score, corners, locator in ordered[:MAX_RAW_VERIFICATIONS]:
        measured = refine_projected_head_border(cv2, raw_edges, corners, corridor_half_width_px=4.)
        diagnostics["strict_verifications"].append({
            "locator": locator, "score": score, "accepted": measured.accepted,
            "reason": measured.reason,
            "locator_corners": [(p.u_px, p.v_px) for p in corners],
            "corners": None if measured.corners is None else
                [(p.u_px, p.v_px) for p in measured.corners],
        })
        if measured.accepted:
            _width, height, center = _extent(measured.corners)
            accepted.append(_proposal(measured, raw_edges.shape, expected_height=height, expected_center=center))
    uncovered = uncovered_head_hypotheses(ordered[MAX_RAW_VERIFICATIONS:], accepted)
    diagnostics["unverified_independent_hypotheses"] = len(uncovered)
    if uncovered:
        return result("head_cold_acquisition_verification_budget_exceeded")
    selected, reason, selection = select_verified_head(cv2, accepted)
    diagnostics.update(selection)
    return result(reason, selected)
