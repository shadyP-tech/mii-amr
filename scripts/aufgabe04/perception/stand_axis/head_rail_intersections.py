"""Bounded four-observed-rail locators for fragmented candidate head borders."""

from itertools import combinations

from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline


def candidate_rail_intersections(cv2, raw_edges, groups, search_bounds, *, deadline_monotonic_sec=None):
    """Intersect current observed lines; every returned side needs current pixels.

    Segment endpoints can stop before a corner even when another segment sees
    the adjacent rail. Four independent measured lines provide a search quad,
    never accepted corners. At most 24 lines per direction enter from the cold
    locator. Vectorized scale/bearing checks precede raw support and the caller's
    unchanged bounded strict four-border/corner verification.
    """
    import numpy as np

    if search_bounds is None:
        return ()
    horizontal = [item for item in groups[0] if item[0] >= .20 * search_bounds.height]
    vertical = [item for item in groups[1] if item[0] >= .35 * search_bounds.height]
    if len(horizontal) < 2 or len(vertical) < 2:
        return ()
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_four_rail_intersections")
    horizontal.sort(key=lambda item: (item[1][1] + item[2][1]) / 2.)
    vertical.sort(key=lambda item: (item[1][0] + item[2][0]) / 2.)
    h = np.asarray([item[1:] for item in horizontal], float)
    v = np.asarray([item[1:] for item in vertical], float)
    hd, vd = h[:, 1] - h[:, 0], v[:, 1] - v[:, 0]
    delta = v[None, :, 0] - h[:, None, 0]
    denominator = hd[:, None, 0] * vd[None, :, 1] - hd[:, None, 1] * vd[None, :, 0]
    ratio = (delta[:, :, 0] * vd[None, :, 1] - delta[:, :, 1] * vd[None, :, 0]) / denominator
    intersections = h[:, None, 0] + ratio[:, :, None] * hd[:, None]
    hpairs = np.asarray(tuple(combinations(range(len(h)), 2)))
    vpairs = np.asarray(tuple(combinations(range(len(v)), 2)))
    top, bottom = hpairs[:, 0, None], hpairs[:, 1, None]
    left, right = vpairs[None, :, 0], vpairs[None, :, 1]
    corners = np.stack((intersections[top, left], intersections[top, right],
                        intersections[bottom, right], intersections[bottom, left]), axis=2).reshape(-1, 4, 2)
    side_lengths = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
    width = (side_lengths[:, 0] + side_lengths[:, 2]) / 2.
    height = (side_lengths[:, 1] + side_lengths[:, 3]) / 2.
    centers = np.mean(corners, axis=1)
    keep = ((abs(height / search_bounds.height - 1.) <= search_bounds.height_tolerance_ratio)
            & (np.linalg.norm(centers - search_bounds.center, axis=1)
               <= search_bounds.center_offset_ratio * search_bounds.height)
            & (width >= 12.) & (width <= 1.35 * height)
            & (corners[:, :, 0].min(axis=1) >= 3.)
            & (corners[:, :, 1].min(axis=1) >= 3.)
            & (corners[:, :, 0].max(axis=1) < raw_edges.shape[1] - 3.)
            & (corners[:, :, 1].max(axis=1) < raw_edges.shape[0] - 3.))
    corners = corners[keep]
    if not len(corners):
        return ()
    check_head_acquisition_deadline(deadline_monotonic_sec, "cold_four_rail_support")
    distance = cv2.distanceTransform(np.where(raw_edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    fractions = np.linspace(.10, .90, 24)
    pixels = np.rint(corners[:, :, None, :] + fractions[None, None, :, None]
                    * (np.roll(corners, -1, axis=1) - corners)[:, :, None, :]).astype(int)
    support = np.mean(distance[pixels[:, :, :, 1], pixels[:, :, :, 0]] <= 2., axis=2)
    return tuple(corners[np.min(support, axis=1) >= .80])
