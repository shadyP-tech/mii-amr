"""Bounded current-border selection, independent of identity and fitted angles.

Search endpoints need current corner pixels before spending a strict fit. A
complete verified frame may contain repeated small, coplanar-looking texture;
that texture is not a set of competing heads. A lone nested head, or a separate
head, remains ambiguous. This is a 2D locator policy, not a motion certificate.
"""

from types import SimpleNamespace

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, _polygon_area
from scripts.aufgabe04.perception.stand_axis.head_proposal import _same_head
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline


def rank_current_head_hypotheses(cv2, raw_edges, hypotheses, *, corridor_half_width_px=4.,
                                 deadline_monotonic_sec=None):
    """Cover spatial/border families before spending spare work on variants.

    The inexpensive locator asks for one current pixel bin on each corner arm;
    the subsequent unchanged strict refinement still requires two, on its fitted
    corners. Infinite rail intersections without any local corner pixels cannot
    consume the verification budget. Neither check synthesizes missing borders.
    """
    directly_supported = []
    for item in hypotheses:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_corner_support")
        evidence = metric_corner_arm_support(cv2, raw_edges, item[1])
        if all(count >= 1 for arms in evidence.bins_by_corner.values() for count in arms.values()):
            directly_supported.append(item)
    # Paired segment endpoints can be a few pixels short of a current corner.
    # Keep such alternate corridors when all four endpoints are within the
    # strict fitter's maximum shift of another current-corner-supported hint.
    # This is a locator rescue; only a subsequent raw fit supplies corners.
    maximum_shift = 2. * corridor_half_width_px + 2.
    supported = [item for item in hypotheses if any(
        max(_distance(a, b) for a, b in zip(item[1], other[1])) <= maximum_shift
        for other in directly_supported)]
    # Complete enclosing borders are tried before their inset alternatives.
    # Selection still compares verified independent families; area is not an
    # authority to choose one stand over another.
    supported.sort(key=lambda item: (-_polygon_area(item[1]), -item[0],
                                    tuple((p.u_px, p.v_px) for p in item[1])))
    groups = []
    for item in supported:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_border_families")
        for group in groups:
            if _same_head(SimpleNamespace(corners=item[1]), SimpleNamespace(corners=group[0][1])):
                group.append(item)
                break
        else:
            groups.append([item])
    ordered = [group[index] for index in range(max(map(len, groups), default=0))
               for group in groups if index < len(group)]
    return ordered, {"corner_supported_hypotheses": len(supported),
                     "direct_corner_supported_hypotheses": len(directly_supported),
                     "corner_locator_rejections": len(hypotheses) - len(supported),
                     "border_families": len(groups)}


def uncovered_head_hypotheses(hypotheses, verified):
    """A failed representative cannot erase its untested border variants."""
    return [item for item in hypotheses if not any(
        _same_head(SimpleNamespace(corners=item[1]), proposal)
        and _polygon_area(item[1]) <= 1.03 * _polygon_area(proposal.corners)
        for proposal in verified)]


def repeated_inset_texture(cv2, outer, others):
    """Identify distributed inset texture inside an already verified frame.

    Use only current quadrilateral topology in the outer frame's projective
    coordinates. No QR API, QR dimensions, identity, neck, or solved angle enters
    this test. Three or more disjoint, similarly sized, small aligned rectangles
    must occupy at least three quadrants and span both axes. Arbitrary containment
    and a lone nested stand are deliberately insufficient.

    Pixels cannot distinguish every scene with identical projection. The returned
    locator remains conditional on the physical model and robot target association.
    """
    import numpy as np

    groups = []
    for proposal in sorted(others, key=lambda item: -_polygon_area(item.corners)):
        if not any(_same_head(proposal, other) for other in groups):
            groups.append(proposal)
    if len(groups) < 3:
        return False
    transform = cv2.getPerspectiveTransform(
        np.asarray([(p.u_px, p.v_px) for p in outer.corners], np.float32),
        np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), np.float32))
    boxes, centers, sizes = [], [], []
    for proposal in groups:
        points = cv2.perspectiveTransform(np.asarray(
            [(p.u_px, p.v_px) for p in proposal.corners], np.float32).reshape(1, 4, 2), transform)[0]
        if not np.isfinite(points).all() or points.min() < .02 or points.max() > .98:
            return False
        low, high = points.min(axis=0), points.max(axis=0)
        size, center = high - low, points.mean(axis=0)
        if size.min() <= 0. or size.max() > .42 or np.linalg.norm(center - .5) < .20:
            return False
        edges = np.roll(points, -1, axis=0) - points
        if any(abs(edge[1 - axis]) > .20 * abs(edge[axis])
               for edge, axis in zip(edges, (0, 1, 0, 1))):
            return False
        if any(np.all(np.minimum(high, other_high) > np.maximum(low, other_low))
               for other_low, other_high in boxes):
            return False
        boxes.append((low, high))
        centers.append(center)
        sizes.append(size)
    centers, sizes = np.asarray(centers), np.asarray(sizes)
    return bool(np.all(sizes.max(axis=0) <= 1.5 * sizes.min(axis=0))
                and np.all(np.ptp(centers, axis=0) >= .35)
                and len({tuple(center >= .5) for center in centers}) >= 3)


def select_verified_head(cv2, verified):
    """Return a complete outer frame or retain genuine current-head ambiguity."""
    diagnostics = {"selection": "unavailable", "texture_rectangles": 0}
    if not verified:
        return None, "head_proposal_unavailable", diagnostics
    selected = max(verified, key=lambda proposal: _polygon_area(proposal.corners))
    others = [proposal for proposal in verified if not _same_head(selected, proposal)]
    if others and not repeated_inset_texture(cv2, selected, others):
        diagnostics["selection"] = "distinct_current_heads_ambiguous"
        return None, "head_proposal_ambiguous", diagnostics
    diagnostics.update(selection="maximal_verified_current_head",
                       texture_rectangles=len(others),
                       selected_corners=[(p.u_px, p.v_px) for p in selected.corners])
    return selected, "current_head_proposal", diagnostics
