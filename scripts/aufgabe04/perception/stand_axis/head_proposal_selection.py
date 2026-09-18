"""Bounded current-border selection, independent of identity and fitted angles.

Search endpoints need current corner pixels before spending a strict fit. A
complete verified frame may contain repeated small, coplanar-looking texture;
that texture is not a set of competing heads. A lone nested head, or a separate
head, remains ambiguous. This is a 2D locator policy, not a motion certificate.
"""

from types import SimpleNamespace

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, _polygon_area
from scripts.aufgabe04.perception.stand_axis.head_border_families import (
    CurrentBorderFamilies,
)
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline


def rank_current_head_hypotheses(cv2, raw_edges, hypotheses, *, corridor_half_width_px=4.,
                                 deadline_monotonic_sec=None, frame_bgr=None, border_families=None):
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
    families = border_families or CurrentBorderFamilies(raw_edges, frame_bgr)
    for item in supported:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_border_families")
        for group in groups:
            if all(families.same(item[1], member[1]) for member in group):
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


def uncovered_head_hypotheses(hypotheses, verified, *, cv2=None, raw_edges=None,
                              frame_bgr=None, all_hypotheses=(), border_families=None,
                              deadline_monotonic_sec=None):
    """A failed representative cannot erase its untested border variants."""
    families = border_families or CurrentBorderFamilies(raw_edges, frame_bgr)
    uncovered = [item for item in hypotheses if not any(
        families.same(item[1], proposal.corners)
        and _polygon_area(item[1]) <= 1.03 * _polygon_area(proposal.corners)
        for proposal in verified)]
    if not uncovered or cv2 is None:
        return uncovered
    # Texture classification can use the already current-border/corner-supported
    # locator graph. Those hints never provide accepted physical head corners.
    pool = [SimpleNamespace(corners=item[1]) for item in all_hypotheses]
    covered = set()
    for outer in verified:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_uncovered_families")
        others = [item for item in pool if not families.same(outer.corners, item.corners)]
        covered.update(id(item) for item in texture_members(cv2, outer, others,
            deadline_monotonic_sec=deadline_monotonic_sec))
    covered_corners = {tuple(item.corners) for item in pool if id(item) in covered}
    return [item for item in uncovered if not any(
        families.same(item[1], corners) for corners in covered_corners)]


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
    return len(texture_members(cv2, outer, others)) == len(others) and len(others) >= 3


def texture_members(cv2, outer, others, *, deadline_monotonic_sec=None):
    """Find repeated inset boxes and rectangles assembled from their same rails.

    Three disjoint, aligned small boxes establish a distributed texture family.
    Additional overlapping/tall rectangles are not automatically competing heads
    when *each* of their four rails belongs to this same inset rail graph. A lone
    inset, a new unexplained rail, or another enclosing border stays ambiguous.
    This is a generic image-topology test with no QR size or identity input.
    """
    import numpy as np

    transform = cv2.getPerspectiveTransform(
        np.asarray([(p.u_px, p.v_px) for p in outer.corners], np.float32),
        np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), np.float32))
    normalized = []
    for proposal in others:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_texture_topology")
        points = cv2.perspectiveTransform(np.asarray(
            [(p.u_px, p.v_px) for p in proposal.corners], np.float32).reshape(1, 4, 2), transform)[0]
        if not np.isfinite(points).all() or points.min() < .02 or points.max() > .98:
            continue
        low, high = points.min(axis=0), points.max(axis=0)
        size, center = high - low, points.mean(axis=0)
        if size.min() <= 0.:
            continue
        edges = np.roll(points, -1, axis=0) - points
        if any(abs(edge[1 - axis]) > .20 * abs(edge[axis])
               for edge, axis in zip(edges, (0, 1, 0, 1))):
            continue
        normalized.append((proposal, low, high, center, size))
    anchors = []
    for item in sorted(normalized, key=lambda item: (-float(np.prod(item[4])),
                                                    tuple(item[1]))):
        if item[4].max() > .42 or np.linalg.norm(item[3] - .5) < .20:
            continue
        # Contour/raster variants of a small texture rail do not create extra
        # independent boxes. Physical-head alternatives are never merged here.
        if any(np.max(np.abs(item[1] - old[1])) < .035
               and np.max(np.abs(item[2] - old[2])) < .035 for old in anchors):
            continue
        anchors.append(item)
    if len(anchors) < 3 or len(anchors) > 32:
        return []
    best = []
    for start in anchors:
        check_head_acquisition_deadline(deadline_monotonic_sec, "cold_texture_rail_graph")
        group = [start]
        for item in anchors:
            if item is start:
                continue
            if any(np.any(np.maximum(item[4], old[4]) > 1.5 * np.minimum(item[4], old[4]))
                   or np.all(np.minimum(item[2], old[2]) > np.maximum(item[1], old[1]))
                   for old in group):
                continue
            group.append(item)
        centers = np.asarray([item[3] for item in group])
        if (len(group) < 3 or np.any(np.ptp(centers, axis=0) < .35)
                or len({tuple(center >= .5) for center in centers}) < 3):
            continue
        def explained(item):
            for axis in range(2):
                along = 1 - axis
                for value in (item[1][axis], item[2][axis]):
                    intervals = sorted((max(item[1][along], anchor[1][along]),
                                        min(item[2][along], anchor[2][along]))
                        for anchor in group if min(abs(value - anchor[1][axis]),
                                                  abs(value - anchor[2][axis])) <= .035)
                    coverage, previous = 0., -float("inf")
                    for low, high in intervals:
                        coverage += max(0., high - max(low, previous))
                        previous = max(previous, high)
                    # Collinearity alone is insufficient: the observed texture
                    # segments must explain most of every composite rail. A
                    # separate central box between them remains independent.
                    if coverage < .5 * float(item[2][along] - item[1][along]):
                        return False
            return True
        def concentric_inset(item):
            # Repeated printed motifs can contain concentric rails. Requiring
            # collinearity with the outer motif mistakes its inner ring for a
            # separate head. Only the already established three distributed,
            # disjoint anchors can own such an inset; ordinary containment in
            # the head (including a lone central rectangle) is insufficient.
            for anchor in group:
                scale = item[4] / anchor[4]
                center_error = np.abs(item[3] - anchor[3]) / anchor[4]
                if (np.all(item[1] > anchor[1]) and np.all(item[2] < anchor[2])
                        and np.all((.35 <= scale) & (scale <= .90))
                        and abs(float(scale[0] - scale[1])) <= .10
                        and np.all(center_error <= .08)):
                    return True
            return False
        members = [item[0] for item in normalized if explained(item) or concentric_inset(item)]
        if len(members) > len(best):
            best = members
    return best


def connected_border_family(selected, proposals, families):
    """Collect mutually evidenced aliases without transitive rail bridges."""
    members = [selected]
    remaining = [item for item in proposals if item is not selected]
    while remaining:
        new = []
        for item in remaining:
            if all(families.same(item.corners, member.corners) for member in members + new):
                new.append(item)
        if not new:
            break
        members.extend(new)
        remaining = [item for item in remaining if all(item is not added for added in new)]
    return members


def select_verified_head(cv2, verified, *, raw_edges=None, frame_bgr=None,
                         texture_hypotheses=(), border_families=None, deadline_monotonic_sec=None):
    """Return a complete outer frame or retain genuine current-head ambiguity."""
    diagnostics = {"selection": "unavailable", "texture_rectangles": 0}
    if not verified:
        return None, "head_proposal_unavailable", diagnostics
    selected = max(verified, key=lambda proposal: _polygon_area(proposal.corners))
    families = border_families or CurrentBorderFamilies(raw_edges, frame_bgr)
    same_frame = connected_border_family(selected, verified, families)
    others = [proposal for proposal in verified if all(proposal is not member for member in same_frame)]
    pool = others + [SimpleNamespace(corners=item[1]) for item in texture_hypotheses
                    if not families.same(selected.corners, item[1])]
    texture = texture_members(cv2, selected, pool,
        deadline_monotonic_sec=deadline_monotonic_sec) if others else []
    if any(all(proposal is not member for member in texture) for proposal in others):
        diagnostics["selection"] = "distinct_current_heads_ambiguous"
        return None, "head_proposal_ambiguous", diagnostics
    diagnostics.update(selection="maximal_verified_current_head",
                       texture_rectangles=len(others),
                       selected_corners=[(p.u_px, p.v_px) for p in selected.corners])
    return selected, "current_head_proposal", diagnostics
