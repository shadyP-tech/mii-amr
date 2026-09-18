"""Bounded current-pixel head acquisition without QR, neck, or pose seeds.

This locator supplies neutral 2D corners to the measured physical head solver.
It does not assign a face, identity, angle, or navigation target. Closed contour
and paired rail endpoints are search hints only; every accepted proposal passes
the same current raw four-border and corner-arm checks as seeded acquisition.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from dataclasses import asdict

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
from scripts.aufgabe04.perception.stand_axis.current_head_refinement import refine_current_physical_head
from scripts.aufgabe04.perception.stand_axis.current_head_refinement_proof import capture_current_head_refinement
from scripts.aufgabe04.perception.stand_axis.head_frame_resolution import (
    distinct_current_frames, resolved_current_frame, resolved_tested_head_hint,
    resolve_measured_current_frames,
    CurrentFrameResolutionRecord,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.head_search_bounds import HeadSearchBounds
from scripts.aufgabe04.perception.stand_axis.head_border_families import CurrentBorderFamilies
from scripts.aufgabe04.perception.stand_axis.head_rail_intersections import candidate_rail_intersections
from scripts.aufgabe04.perception.stand_axis.candidate_rail_hints import candidate_observed_rail_hints
from scripts.aufgabe04.perception.stand_axis.head_hint_neighborhood import observed_head_hint_neighborhood
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import (
    bounded_head_acquisition, check_head_acquisition_deadline,
)

MAX_IMAGE_PIXELS = 1920 * 1080
MAX_CONTOURS = 1024
MAX_RAILS_PER_DIRECTION = 24
MAX_LOCATOR_HYPOTHESES = 256
MAX_RAW_VERIFICATIONS = 12
MAX_GUIDED_NEIGHBOR_PARENTS = 32
MAX_HEAD_IMAGE_AREA_FRACTION = .80
MIN_LOCATOR_BORDER_SUPPORT = .80
_LOCATOR = "cold_current_borders"


def _distinct_rails(group, direction, *, deadline_monotonic_sec=None):
    """Keep the same longest-first representatives using bounded array work."""
    import numpy as np

    ordered = sorted(group, key=lambda item: (-item[0], item[1:]))
    kept = np.empty((len(ordered), 5), dtype=np.float64)
    distinct = []
    for item in ordered:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_rail_merge")
        length, a, b = item
        old = kept[:len(distinct)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        # Compare only retained representatives. A chain of overlapping
        # fragments must not merge two rails that fail the original predicate.
        parallel = (np.abs(dx * (old[:, 4] - old[:, 2])
                           - dy * (old[:, 3] - old[:, 1])) / (length * old[:, 0])) < .02
        nearby = np.maximum(np.abs(dx * (old[:, 2] - a[1]) - dy * (old[:, 1] - a[0])),
                            np.abs(dx * (old[:, 4] - a[1]) - dy * (old[:, 3] - a[0]))) / length < 1.5
        overlap = np.minimum(b[direction], old[:, 3 + direction]) - np.maximum(a[direction], old[:, 1 + direction])
        if np.any(parallel & nearby & (overlap > .5 * np.minimum(length, old[:, 0]))):
            continue
        kept[len(distinct)] = (length, *a, *b)
        distinct.append(item)
    return distinct


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


def _rail_endpoint_hints(cv2, gray, raw_edges, *, deadline_monotonic_sec=None,
                         search_bounds=None, rail_groups_out=None,
                         preferred_head_height_px=None, all_rail_groups_out=None,
                         candidate_search=None, guidance_diagnostics=None, source_support=None,
                         edge_region=None, edge_region_diagnostics=None, metric_search=None):
    """Pair current line segments, using candidate scale only to order hints."""
    import numpy as np

    preferred_height = (search_bounds.height if search_bounds is not None
                        else preferred_head_height_px)
    # Preserve LSD's original sampling and endpoints: cropping gray changes
    # even interior subpixel lines. Screen its results before the rail quota.
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_hough")
    # Equal-luminance colour boundaries can be absent from grayscale LSD while
    # the caller's channel-union Canny has complete current evidence.
    raw_lines = cv2.HoughLinesP(raw_edges, 1., np.pi / 180., threshold=8,
                               minLineLength=16, maxLineGap=2)
    groups = ([], [])
    segments = [segment for lines in (detected, raw_lines) if lines is not None
                for segment in lines.reshape(-1, 4)]
    if edge_region_diagnostics is not None:
        edge_region_diagnostics.update(input_line_segments=len(segments), rejected_line_segments=0)
    for x0, y0, x1, y1 in segments:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_rails")
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < MIN_HEAD_EDGE_PX * .70:
            continue
        direction = 0 if abs(dy) <= .50 * abs(dx) else 1 if abs(dx) <= .50 * abs(dy) else None
        if direction is None:
            continue
        first, last = (float(x0), float(y0)), (float(x1), float(y1))
        if (getattr(metric_search, "center_bounds_px", None) is None
                and edge_region is not None and not edge_region.contains(
                (ImagePoint(*first), ImagePoint(*last)))):
            if edge_region_diagnostics is not None:
                edge_region_diagnostics["rejected_line_segments"] += 1
            continue
        if source_support is not None and not source_support.segment(first, last):
            continue
        if metric_search is not None:
            bx0, by0, bx1, by1 = metric_search.image_bounds(gray.shape)
            midpoint = ((first[0]+last[0])/2., (first[1]+last[1])/2.)
            size = metric_search.pixel_size
            upper = size.max_width_px if direction == 0 else size.max_height_px
            if (length > 1.8*upper or (metric_search.center_bounds_px is None
                    and (not bx0 <= midpoint[0] < bx1 or not by0 <= midpoint[1] < by1))):
                continue
        if search_bounds is not None:
            # Before the fixed rail quota: remote room boundaries cannot displace
            # the selected candidate's rails simply by being longer.
            x0_bound, y0_bound, x1_bound, y1_bound = search_bounds.image_bounds(gray.shape)
            midpoint = ((first[0] + last[0]) / 2., (first[1] + last[1]) / 2.)
            if (not x0_bound <= midpoint[0] < x1_bound
                    or not y0_bound <= midpoint[1] < y1_bound
                    or length > 1.8 * search_bounds.height * (1. + search_bounds.height_tolerance_ratio)):
                continue
        if first[direction] > last[direction]:
            first, last = last, first
        groups[direction].append((length, first, last))
    distinct_groups = tuple(_distinct_rails(group, direction,
        deadline_monotonic_sec=deadline_monotonic_sec) for direction, group in enumerate(groups))
    guided_groups = []
    if candidate_search is not None and preferred_head_height_px is not None:
        candidate_observed_rail_hints(cv2, raw_edges, distinct_groups, candidate_search,
            deadline_monotonic_sec=deadline_monotonic_sec, diagnostics=guidance_diagnostics,
            prioritized_groups_out=guided_groups, rail_limit=MAX_RAILS_PER_DIRECTION,
            source_support=source_support)
    hints = []
    for direction, group in enumerate(groups):
        # LSD and Hough often locate the same rail with different endpoints.
        # Collapse those search duplicates before allocating the fixed budget;
        # the retained longest fragment still supplies only a locator line.
        distinct = distinct_groups[direction]
        if all_rail_groups_out is not None:
            all_rail_groups_out.append(tuple(distinct))
        if len(guided_groups) == 2 and guided_groups[direction]:
            groups[direction][:] = guided_groups[direction]
        elif preferred_height is not None:
            # The projected physical size already bounds this candidate. Give
            # its complete rails a slot before small printed-texture fragments;
            # position/scale remain hints, not measured corners or an angle.
            groups[direction][:] = sorted(distinct, key=lambda item: (
                abs(math.log(item[0] / preferred_height)), item[1:]))[:MAX_RAILS_PER_DIRECTION]
        else:
            # Without a projected candidate, spread the fixed budget across
            # line scales and image thirds so room boundaries cannot monopolize it.
            strata = {}
            for item in distinct:
                length, a, b = item
                key = (max(0, int(math.log2(length / MIN_HEAD_EDGE_PX))),
                       min(2, int(3. * (a[0] + b[0]) / (2. * gray.shape[1]))))
                strata.setdefault(key, []).append(item)
            groups[direction][:] = [bucket[index] for index in range(max(map(len, strata.values()), default=0))
                                    for _key, bucket in sorted(strata.items()) if index < len(bucket)][:MAX_RAILS_PER_DIRECTION]
        if getattr(metric_search, "center_bounds_px", None) is not None and edge_region is not None:
            # Do not refill slots vacated by the volume gate with ever more
            # inset/printed rails. That makes a tighter mask increase the
            # number of competing rectangles despite fewer input edge pixels.
            retained = [item for item in groups[direction] if edge_region.contains(
                (ImagePoint(*item[1]), ImagePoint(*item[2])))]
            if edge_region_diagnostics is not None:
                edge_region_diagnostics["rejected_line_segments"] += len(groups[direction])-len(retained)
            groups[direction][:] = retained
        group = groups[direction]
        cross = 1 - direction
        for index, (length, first, last) in enumerate(group):
            check_head_acquisition_deadline(deadline_monotonic_sec, "cold_endpoint_hypotheses")
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
    if rail_groups_out is not None:
        rail_groups_out.extend(tuple(group) for group in groups)
    return tuple(hints), tuple(map(len, groups))


@bounded_head_acquisition
def acquire_cold_head_proposal(
    cv2, frame_bgr, *, raw_edges=None, edge_preprocess="channel_union",
    canny_low=20, canny_high=60,
    deadline_monotonic_sec=None,
    expected_head_center_u_px=None, expected_head_center_v_px=None,
    expected_head_height_px=None, max_center_offset_ratio=.70,
    expected_head_height_tolerance_ratio=.35, proposal_filter=None,
    model_profile=None, refinement_out=None, candidate_search=None,
    color_support_mask=None,
    _preferred_rail_height_px=None, _verification_limit=None, _attempt_diagnostics=None,
    _guided_rail_search=False,
) -> HeadProposalResult:
    """Locate a unique complete head in a bounded image without a prior pose.

    Contours and line hints locate, but never certify, boundaries. A clipped or
    open rectangle cannot become a measurement through locator interpolation.
    Current corner evidence and border families allocate the fixed work budget.
    Unverified independent families remain unresolved on budget exhaustion.
    """
    import numpy as np

    if refinement_out is not None:
        refinement_out.clear()

    verification_limit = MAX_RAW_VERIFICATIONS if _verification_limit is None else _verification_limit
    if type(verification_limit) is not int or not 0 <= verification_limit <= MAX_RAW_VERIFICATIONS:
        raise ValueError("head verification limit must remain within the original work budget")
    diagnostics = {} if _attempt_diagnostics is None else _attempt_diagnostics
    diagnostics.update({
        "method": _LOCATOR, "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_contours": MAX_CONTOURS, "max_rails_per_direction": MAX_RAILS_PER_DIRECTION,
        "max_locator_hypotheses": MAX_LOCATOR_HYPOTHESES,
        "max_raw_verifications": verification_limit,
        "max_head_image_area_fraction": MAX_HEAD_IMAGE_AREA_FRACTION,
        "min_locator_border_support": MIN_LOCATOR_BORDER_SUPPORT,
        "angle_authorized": False, "motion_authorized": False,
        "selection": "unavailable", "strict_verifications": [],
        "candidate_bounds_rejections": 0, "candidate_association_rejections": 0,
        "candidate_association_previews": 0, "candidate_association_preview_rejections": 0,
        "physical_frame_refinement": model_profile is not None,
        "resolved_hint_aliases": 0, "raw_border_refinements": 0,
        "candidate_screen": None if candidate_search is None else candidate_search.diagnostics(),
        "candidate_screen_hint_rejections": 0, "candidate_screen_measurement_rejections": 0,
        "color_prior": {"applied": color_support_mask is not None,
                        "policy": "coherent_rail_ranking_only", "raw_edges_unchanged": True},
    })

    def result(reason, proposal=None):
        return HeadProposalResult(
            proposal, reason, diagnostics.get("considered_proposals", 0),
            len(diagnostics["strict_verifications"]), _LOCATOR, diagnostics,
        )

    if (frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
            or min(frame_bgr.shape[:2]) < MIN_HEAD_EDGE_PX + 6):
        return result("head_proposal_input_invalid")
    try:
        search_bounds = HeadSearchBounds.optional(
            expected_head_center_u_px, expected_head_center_v_px, expected_head_height_px,
            max_center_offset_ratio, expected_head_height_tolerance_ratio)
    except (TypeError, ValueError):
        return result("head_proposal_input_invalid")
    diagnostics["candidate_search_bounds"] = None if search_bounds is None else search_bounds.diagnostics()
    if frame_bgr.shape[0] * frame_bgr.shape[1] > MAX_IMAGE_PIXELS:
        return result("head_cold_acquisition_image_budget_exceeded")
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_preprocessing")
    if raw_edges is None:
        raw_edges = _canny_edges_from_frame(
            cv2, frame_bgr, edge_preprocess=edge_preprocess, blur_kernel=5,
            canny_low=canny_low, canny_high=canny_high,
        )
    if raw_edges.ndim != 2 or raw_edges.shape != frame_bgr.shape[:2]:
        raise ValueError("raw_edges must match the cold search image")
    if color_support_mask is not None and (
            color_support_mask.ndim != 2 or color_support_mask.shape != raw_edges.shape
            or str(color_support_mask.dtype) != "uint8"):
        raise ValueError("colour support must be a uint8 mask of the exact processing image")
    edge_region = getattr(candidate_search, "edge_region", None)
    locator_edges = raw_edges if edge_region is None else edge_region.locator_edges(raw_edges)
    if edge_region is not None:
        diagnostics["lidar_edge_region"] = dict(edge_region.diagnostics(),
            input_edge_pixels=int(np.count_nonzero(raw_edges)),
            retained_edge_pixels=int(np.count_nonzero(locator_edges)))
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_contours")
    contours = cv2.findContours(locator_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    diagnostics["input_contours"] = len(contours)
    metric_search = candidate_search if getattr(candidate_search, "pixel_size", None) is not None else None
    if metric_search is not None:
        bx0, by0, bx1, by1 = metric_search.image_bounds(raw_edges.shape)
        def metric_relevant(contour):
            x, y, w, h = cv2.boundingRect(contour)
            size = metric_search.pixel_size
            # A contour is only a locator. Leave refinement margin and keep
            # short texture loops available in the untouched raw evidence.
            return (x >= bx0 and y >= by0 and x+w <= bx1 and y+h <= by1
                    and h >= .85*size.min_height_px/(1.15*1.25)
                    and h <= 1.5*size.max_height_px
                    and w <= 1.5*size.max_width_px)
        contours = [contour for contour in contours if metric_relevant(contour)]
        diagnostics["metric_contours_rejected"] = diagnostics["input_contours"]-len(contours)
    if search_bounds is not None:
        x0, y0, x1, y1 = search_bounds.image_bounds(raw_edges.shape)
        def relevant(contour):
            if not len(contour):
                return False
            x, y, width, height = cv2.boundingRect(contour)
            return x >= x0 and y >= y0 and x + width <= x1 and y + height <= y1
        contours = [contour for contour in contours if relevant(contour)]
    eligible_contours = []
    for contour in contours:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_contour_hypotheses")
        if edge_region is not None:
            x0, y0, x1, y1 = edge_region.bounds
            x, y, w, h = cv2.boundingRect(contour)
            # Mask clipping is not a closed observed contour.
            if x <= x0 or y <= y0 or x+w >= x1 or y+h >= y1:
                continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 4 * MIN_HEAD_EDGE_PX:
            continue
        eligible_contours.append((contour, perimeter))
        if len(eligible_contours) > MAX_CONTOURS:
            diagnostics["contours"] = len(eligible_contours)
            return result("head_cold_acquisition_contour_budget_exceeded")
    # A contour shorter than the minimum four sides could never supply a
    # valid quad. Tiny blind/radiator/printed-texture loops do not spend the
    # bounded quadrilateral search budget, but remain in untouched raw edges.
    diagnostics["short_contours_rejected"] = len(contours) - len(eligible_contours)
    diagnostics["contours"] = len(eligible_contours)
    seeds = []
    for contour, perimeter in eligible_contours:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_contour_hypotheses")
        for fraction in (.015, .025, .04):
            quad = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                seeds.append((quad.reshape(-1, 2), "closed_contour"))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_line_detection")
    rail_groups = []
    all_rail_groups = []
    guidance = {}
    hints, rail_counts = _rail_endpoint_hints(cv2, gray, raw_edges,
                                            deadline_monotonic_sec=deadline_monotonic_sec,
                                            search_bounds=search_bounds, rail_groups_out=rail_groups,
                                            preferred_head_height_px=(metric_search.height if metric_search is not None
                                                                      else _preferred_rail_height_px),
                                            metric_search=metric_search,
                                            all_rail_groups_out=all_rail_groups,
                                            candidate_search=candidate_search if _guided_rail_search else None,
                                            source_support=getattr(proposal_filter, "source_support", None),
                                            edge_region=edge_region,
                                            edge_region_diagnostics=diagnostics.get("lidar_edge_region"),
                                            guidance_diagnostics=guidance)
    diagnostics["horizontal_rails"], diagnostics["vertical_rails"] = rail_counts
    seeds.extend((hint, "paired_current_rails") for hint in hints)
    if guidance:
        diagnostics["candidate_guided_rails"] = guidance
        if guidance["overflow"]:
            return result("head_candidate_guided_locator_budget_exceeded")
    if _guided_rail_search and model_profile is not None:
        parents = set()
        additions = []
        for points, _locator in seeds:
            check_head_acquisition_deadline(deadline_monotonic_sec, "candidate_hint_neighborhood")
            corners = _bounded_quad(points, raw_edges.shape)
            if (corners is None or corners in parents
                    or not candidate_search.accepts_hint(corners)):
                continue
            parents.add(corners)
            if len(parents) > MAX_GUIDED_NEIGHBOR_PARENTS:
                break
            additions.extend((tuple((p.u_px, p.v_px) for p in variant), "candidate_observed_offsets")
                for variant in observed_head_hint_neighborhood(cv2, raw_edges, corners,
                    model_profile=model_profile, deadline_monotonic_sec=deadline_monotonic_sec))
        diagnostics["candidate_hint_neighborhood"] = {
            "parents": min(len(parents), MAX_GUIDED_NEIGHBOR_PARENTS),
            "max_parents": MAX_GUIDED_NEIGHBOR_PARENTS,
            "variants": len(additions), "supplies_measurement": False}
        seeds.extend(additions)
    distance = cv2.distanceTransform(np.where(raw_edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    fractions = np.linspace(.10, .90, 24)
    hypotheses = {}
    texture_hints = {}
    association_previews = {}
    seen_exact_seeds = set()
    for points, locator in seeds:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_hypothesis_support")
        corners = _bounded_quad(points, raw_edges.shape)
        if corners is None:
            continue
        if corners in seen_exact_seeds:
            diagnostics["exact_duplicate_seeds"] = diagnostics.get("exact_duplicate_seeds", 0) + 1
            continue
        seen_exact_seeds.add(corners)
        texture_only = False
        if search_bounds is not None and not search_bounds.accepts(corners):
            diagnostics["candidate_bounds_rejections"] += 1
            _width, height, center = _extent(corners)
            # Small supported texture boxes still explain cross-paired internal
            # rails, but never spend a strict physical-head verification slot.
            texture_only = (height < .65 * search_bounds.height
                and math.dist(center, search_bounds.center)
                    <= (search_bounds.center_offset_ratio + .5) * search_bounds.height)
            if not texture_only:
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
        key = tuple(round(float(value), 2) for value in points.ravel())
        score = .70 * float(min(support)) + .30 * float(np.mean(support))
        if candidate_search is not None and not candidate_search.accepts_hint(corners):
            # Keep current supported texture for the unchanged topology graph.
            # Screening a head location cannot erase its smaller inset anchors.
            texture_only = True
            diagnostics["candidate_screen_hint_rejections"] += 1
        if proposal_filter is not None and not texture_only:
            if key not in association_previews:
                if len(association_previews) >= MAX_LOCATOR_HYPOTHESES:
                    return result("head_cold_acquisition_locator_budget_exceeded")
                _width, height, center = _extent(corners)
                # The callback sees a current supported 2D locator, with no
                # pose/angle authority. It is rechecked on measured corners
                # after strict refinement; projection never certifies pixels.
                preview = _proposal(SimpleNamespace(corners=corners,
                    support=SimpleNamespace(mean=float(np.mean(support)))), raw_edges.shape,
                    expected_height=height if search_bounds is None else search_bounds.height,
                    expected_center=center if search_bounds is None else search_bounds.center)
                association_previews[key] = bool(getattr(proposal_filter, "preview", proposal_filter)(preview))
                diagnostics["candidate_association_previews"] += 1
            if not association_previews[key]:
                diagnostics["candidate_association_preview_rejections"] += 1
                diagnostics["candidate_association_rejections"] += 1
                continue
        destination = texture_hints if texture_only else hypotheses
        if key not in destination or score > destination[key][0]:
            destination[key] = (score, corners, locator)
    diagnostics["texture_only_hypotheses"] = len(texture_hints)
    if len(texture_hints) > MAX_LOCATOR_HYPOTHESES:
        return result("head_cold_acquisition_locator_budget_exceeded")
    diagnostics["considered_proposals"] = len(hypotheses)
    if len(hypotheses) > MAX_LOCATOR_HYPOTHESES:
        return result("head_cold_acquisition_locator_budget_exceeded")
    # Group duplicate rail contours before spending the strict verification
    # budget. Retain distinct border locations: choosing inward versus outward
    # rails remains a measurement decision, never a proximity-to-centre prior.
    ranked = sorted(hypotheses.values(), key=lambda item: (-item[0], -_polygon_area(item[1]),
                                                         tuple((p.u_px, p.v_px) for p in item[1])))
    unique = []
    border_families = CurrentBorderFamilies(raw_edges, frame_bgr)
    for item in ranked:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_hypothesis_deduplication")
        if not any(max(_distance(a, b) for a, b in zip(item[1], other[1])) < 3.
                   and border_families.same(item[1], other[1])
                   for other in unique):
            unique.append(item)
    diagnostics["distinct_locator_hypotheses"] = len(unique)
    ordered, ranking = rank_current_head_hypotheses(cv2, raw_edges, unique,
        frame_bgr=frame_bgr, border_families=border_families,
        deadline_monotonic_sec=deadline_monotonic_sec)
    diagnostics.update(ranking)
    texture_pool = ordered + list(texture_hints.values())
    accepted = []
    verified_proposals = []
    untested = []
    tested_hints = set()
    tested_frame_resolutions = []
    measured_frames = {}
    def verify(items):
        for score, corners, locator in items:
            check_head_acquisition_deadline(deadline_monotonic_sec, "cold_strict_verification",
                considered_proposals=len(hypotheses), raw_verifications=len(diagnostics["strict_verifications"]))
            hint_key = tuple(corners)
            if hint_key in tested_hints:
                diagnostics["exact_duplicate_hints"] = diagnostics.get("exact_duplicate_hints", 0) + 1
                continue
            if model_profile is not None and resolved_current_frame(
                    corners, verified_proposals, border_families) is not None:
                diagnostics["resolved_hint_aliases"] += 1
                continue
            if model_profile is not None and resolved_tested_head_hint(
                    corners, tested_frame_resolutions, border_families) is not None:
                diagnostics["resolved_tested_hint_aliases"] = diagnostics.get("resolved_tested_hint_aliases", 0) + 1
                continue
            if len(diagnostics["strict_verifications"]) >= verification_limit:
                untested.append((score, corners, locator))
                continue
            tested_hints.add(hint_key)
            outer = None
            if model_profile is None:
                measured = refine_projected_head_border(cv2, raw_edges, corners, corridor_half_width_px=4.)
                diagnostics["raw_border_refinements"] += 1
            else:
                measured, outer, _seed = refine_current_physical_head(
                    cv2, raw_edges, model_profile=model_profile, proposal_corners=corners,
                    prefer_outer_metric_rail=getattr(metric_search, "center_bounds_px", None) is not None,
                    color_support_mask=color_support_mask,
                    deadline_monotonic_sec=deadline_monotonic_sec)
                diagnostics["raw_border_refinements"] += 1 + outer.attempted_raw_refinements
            diagnostics["strict_verifications"].append({
                "locator": locator, "score": score, "accepted": measured.accepted,
                "reason": measured.reason,
                "locator_corners": [(p.u_px, p.v_px) for p in corners],
                "corners": None if measured.corners is None else
                    [(p.u_px, p.v_px) for p in measured.corners],
                "physical_frame": None if outer is None else asdict(outer),
            })
            if measured.accepted and (outer is None or outer.accepted):
                _width, height, center = _extent(measured.corners)
                if search_bounds is not None and not search_bounds.accepts(measured.corners):
                    diagnostics["candidate_bounds_rejections"] += 1
                    continue
                proposal = _proposal(measured, raw_edges.shape,
                    expected_height=height if search_bounds is None else search_bounds.height,
                    expected_center=center if search_bounds is None else search_bounds.center)
                verified_proposals.append(proposal)
                if outer is not None:
                    tested_frame_resolutions.append(CurrentFrameResolutionRecord(
                        tuple(corners), tuple(outer.original_corners), proposal, outer))
                if candidate_search is not None and not candidate_search.accepts_measurement(measured.corners):
                    diagnostics["candidate_screen_measurement_rejections"] += 1
                    continue
                if proposal_filter is not None and not proposal_filter(proposal):
                    diagnostics["candidate_association_rejections"] += 1
                    continue
                accepted.append(proposal)
                if outer is not None:
                    measured_frames[proposal.corners] = (measured, outer, _seed)
    verify(ordered)
    used = len(diagnostics["strict_verifications"])
    # Fragment rescue cannot replace an already accepted/ambiguous physical
    # head and shares the original twelve strict checks, never a fresh budget.
    if not verified_proposals and search_bounds is not None and rail_groups and used < verification_limit:
        rescue = []
        diagnostics["four_rail_rescue_attempted"] = True
        hints = candidate_rail_intersections(cv2, raw_edges, rail_groups, search_bounds,
                                             deadline_monotonic_sec=deadline_monotonic_sec)
        diagnostics["four_rail_rescue_raw_hypotheses"] = len(hints)
        for points in hints:
            check_head_acquisition_deadline(deadline_monotonic_sec, "cold_fragment_corner_support")
            corners = _bounded_quad(points, raw_edges.shape)
            if corners is None or not search_bounds.accepts(corners):
                continue
            # Exact intersections need both current arms, without the endpoint
            # locator's ten-pixel proximity rescue to another proposal.
            if not metric_corner_arm_support(cv2, raw_edges, corners).accepted:
                continue
            if any(border_families.same(corners, other[1]) for other in rescue):
                continue
            if proposal_filter is not None:
                if diagnostics["candidate_association_previews"] >= MAX_LOCATOR_HYPOTHESES:
                    return result("head_cold_acquisition_locator_budget_exceeded")
                preview = _proposal(SimpleNamespace(corners=corners,
                    support=SimpleNamespace(mean=.80)), raw_edges.shape,
                    expected_height=search_bounds.height, expected_center=search_bounds.center)
                diagnostics["candidate_association_previews"] += 1
                if not getattr(proposal_filter, "preview", proposal_filter)(preview):
                    diagnostics["candidate_association_preview_rejections"] += 1
                    diagnostics["candidate_association_rejections"] += 1
                    continue
            rescue.append((.80, corners, "four_current_rails"))
            if len(rescue) > MAX_LOCATOR_HYPOTHESES:
                return result("head_cold_acquisition_locator_budget_exceeded")
        rescue.sort(key=lambda item: (-_polygon_area(item[1]),
            tuple((point.u_px, point.v_px) for point in item[1])))
        diagnostics["four_rail_rescue_hypotheses"] = len(rescue)
        ordered.extend(rescue)
        texture_pool.extend(rescue)
        diagnostics["considered_proposals"] += len(rescue)
        verify(rescue)
    if model_profile is not None:
        before_untested = len(untested)
        untested = [item for item in untested if resolved_current_frame(
            item[1], verified_proposals, border_families) is None
            and resolved_tested_head_hint(item[1], tested_frame_resolutions, border_families) is None]
        diagnostics["resolved_post_budget_hint_aliases"] = before_untested - len(untested)
        before = len(accepted)
        resolved = resolve_measured_current_frames(
            accepted, tested_frame_resolutions, border_families)
        diagnostics["resolved_measured_frame_recoveries"] = sum(
            original is not replacement for original, replacement in zip(accepted, resolved))
        accepted = distinct_current_frames(resolved, border_families)
        diagnostics["resolved_physical_frame_aliases"] = before - len(accepted)
    uncovered = uncovered_head_hypotheses(untested, verified_proposals,
        cv2=cv2, raw_edges=raw_edges, frame_bgr=frame_bgr, all_hypotheses=texture_pool,
        border_families=border_families, deadline_monotonic_sec=deadline_monotonic_sec)
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_head_selection",
        considered_proposals=len(hypotheses), raw_verifications=len(diagnostics["strict_verifications"]))
    diagnostics["unverified_independent_hypotheses"] = len(uncovered)
    if uncovered:
        return result("head_cold_acquisition_verification_budget_exceeded")
    selected, reason, selection = select_verified_head(cv2, accepted,
        raw_edges=raw_edges, frame_bgr=frame_bgr, texture_hypotheses=texture_pool,
        border_families=border_families, deadline_monotonic_sec=deadline_monotonic_sec)
    if selected is None and not accepted and diagnostics["candidate_association_rejections"]:
        reason = "head_proposal_candidate_association_rejected"
    diagnostics.update(selection)
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_complete_head_selection",
        considered_proposals=len(hypotheses), raw_verifications=len(diagnostics["strict_verifications"]))
    if selected is not None and refinement_out is not None and model_profile is not None:
        measured, outer, seed = measured_frames[selected.corners]
        refinement_out["selected"] = capture_current_head_refinement(
            frame_bgr, raw_edges, model_profile=model_profile,
            refinement=measured, outer_recovery=outer, seed=seed)
    used = len(diagnostics["strict_verifications"])
    if (reason in {"head_proposal_unavailable", "head_proposal_candidate_association_rejected"}
            and selected is None and not accepted and not uncovered
            and candidate_search is not None and search_bounds is None
            and not _guided_rail_search and used < verification_limit
            and (_preferred_rail_height_px is None or any(all_rail_groups))):
        # The original full-image distribution gets the first decision. Only a
        # completed candidate miss can spend the *remaining* strict checks on
        # the size-prioritized distribution, then observed candidate rail pairs.
        # Rejected background hints do not suppress recovery; successes and
        # unresolved ambiguity never retry. The deadline and pixels stay binding.
        retry_diagnostics = {}
        retried = acquire_cold_head_proposal(
            cv2, frame_bgr, raw_edges=raw_edges, edge_preprocess=edge_preprocess,
            canny_low=canny_low, canny_high=canny_high,
            deadline_monotonic_sec=deadline_monotonic_sec, proposal_filter=proposal_filter,
            model_profile=model_profile, refinement_out=refinement_out, candidate_search=candidate_search,
            color_support_mask=color_support_mask,
            _preferred_rail_height_px=candidate_search.height,
            _verification_limit=verification_limit-used, _attempt_diagnostics=retry_diagnostics,
            _guided_rail_search=_preferred_rail_height_px is not None)
        # Retain completed checks even if a cooperative deadline interrupts the
        # retry inside refinement before its decorator can return full details.
        combined = {**retry_diagnostics, **(retried.joint_border_diagnostics or {})}
        retry_checks = combined.get("strict_verifications", [])
        retry_refinements = combined.get("raw_border_refinements", 0)
        combined.update(
            max_raw_verifications=verification_limit,
            strict_verifications=diagnostics["strict_verifications"] + retry_checks,
            raw_border_refinements=diagnostics["raw_border_refinements"]
                + retry_refinements,
            candidate_size_retry={"performed": True, "initial_reason": reason,
                "initial_raw_verifications": used, "remaining_raw_verifications": verification_limit-used,
                "retry_raw_verifications": len(retry_checks),
                "preferred_rail_height_px": candidate_search.height,
                "guided_rails": _preferred_rail_height_px is not None,
                "same_raw_image": True, "deadline_extended": False})
        if (retried.joint_border_diagnostics or {}).get("candidate_size_retry"):
            combined["candidate_size_retry"]["next_attempt"] = retried.joint_border_diagnostics["candidate_size_retry"]
        considered = diagnostics.get("considered_proposals", 0) + max(
            retry_diagnostics.get("considered_proposals", 0), retried.considered_proposals)
        combined["considered_proposals"] = considered
        return HeadProposalResult(retried.proposal, retried.reason, considered,
                                  used + len(retry_checks), _LOCATOR, combined)
    return result(reason, selected)
