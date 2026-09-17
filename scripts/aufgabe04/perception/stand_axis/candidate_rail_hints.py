"""Candidate-compatible observed rail pairs before the global line quota.

Only search locations are returned. No map-projected rectangle, accepted pixel,
pose, QR observation or previous image is supplied by this locator.
"""

import math

from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MIN_HEAD_EDGE_PX
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.geometry import order_corners
from scripts.aufgabe04.perception.stand_axis.metric_edge_association import metric_corner_arm_support


MAX_GUIDED_RAILS_PER_DIRECTION = 512
MAX_GUIDED_HINTS = 256


def candidate_observed_rail_hints(cv2, raw_edges, groups, candidate_search, *,
                                  deadline_monotonic_sec=None, diagnostics=None,
                                  prioritized_groups_out=None, rail_limit=24):
    """Screen complete pairs before charging the bounded candidate hint pool.

    Work is bounded by two 512-line pools and processed one line at a time.
    A pool overflow is explicit, never a claim that the retained head is unique.
    The caller still performs exact quad, corner, raw-fit and ambiguity checks.
    """
    import numpy as np

    info = {} if diagnostics is None else diagnostics
    info.update(input_rails=[len(group) for group in groups], eligible_rails=[],
                supported_hints=0, overflow=False, supplies_measurement=False)
    center = np.asarray(candidate_search.center)
    expected = candidate_search.height
    # A quad admitted by the ordinary locator has width <=1.35*height.
    # Its hint height can reach 1.35/.85 times the expected height. A side's
    # endpoints are within half its perimeter of the center; this loose bound
    # only removes rails that cannot contribute, not off-center foregrounds.
    radius = candidate_search.max_center_offset_ratio * expected + 10.
    reach = radius + 2.35 * (1.35/.85) * expected
    distance = None
    output = []
    seen = set()
    fractions = np.linspace(.10, .90, 24)
    rows, cols = raw_edges.shape
    for direction, group in enumerate(groups):
        priorities = {}
        eligible = [item for item in group
                    if math.dist(((item[1][0]+item[2][0])/2.,
                                  (item[1][1]+item[2][1])/2.), center) <= reach
                    and item[0] <= 2.7*(1.35/.85)*expected]
        info["eligible_rails"].append(len(eligible))
        if len(eligible) > MAX_GUIDED_RAILS_PER_DIRECTION:
            info["overflow"] = True
            return ()
        if len(eligible) < 2:
            if prioritized_groups_out is not None:
                prioritized_groups_out.append(tuple(eligible))
            continue
        lines = np.asarray([item[1:] for item in eligible], dtype=float)
        lengths = np.asarray([item[0] for item in eligible])
        cross = 1-direction
        for index in range(len(lines)-1):
            check_head_acquisition_deadline(deadline_monotonic_sec, "candidate_observed_rail_pairs")
            first, last = lines[index]
            others = lines[index+1:]
            shorter = np.minimum(lengths[index], lengths[index+1:])
            longer = np.maximum(lengths[index], lengths[index+1:])
            separation = np.abs((first[cross]+last[cross]-others[:, 0, cross]-others[:, 1, cross])/2.)
            starts = np.minimum(first[direction], others[:, 0, direction])
            ends = np.maximum(last[direction], others[:, 1, direction])
            overlap = np.minimum(last[direction], others[:, 1, direction])-np.maximum(
                first[direction], others[:, 0, direction])
            keep = ((shorter >= .20*longer) & (separation >= MIN_HEAD_EDGE_PX*.70)
                    & (overlap >= .15*shorter) & (ends-starts <= 1.4*longer))
            partners = np.arange(index+1, len(lines))[keep]
            others, starts, ends = others[keep], starts[keep], ends[keep]
            if not len(others):
                continue
            a = np.broadcast_to(first, (len(others), 2))
            b = np.broadcast_to(last, (len(others), 2))
            quads = np.stack((a, b, others[:, 1], others[:, 0]), axis=1)
            # Endpoint unions help when a neck interrupts the bottom segment.
            def extend(start, end, value):
                delta = end-start
                return start + ((value-start[:, direction])/delta[:, direction])[:, None]*delta
            extended = np.stack((extend(a, b, starts), extend(a, b, ends),
                                 extend(others[:, 0], others[:, 1], ends),
                                 extend(others[:, 0], others[:, 1], starts)), axis=1)
            quads = np.concatenate((quads, extended))
            partners = np.tile(partners, 2)
            if direction == 1:
                quads = quads[:, (0, 3, 2, 1)]
            side = np.linalg.norm(np.roll(quads, -1, axis=1)-quads, axis=2)
            height = (side[:, 1]+side[:, 3])/2.
            width = (side[:, 0]+side[:, 2])/2.
            positions = quads.mean(axis=1)
            keep = ((.85*height <= 1.35*expected) & (1.15*1.25*height >= .60*expected)
                    & (width <= 1.35*height) & (side.min(axis=1) >= MIN_HEAD_EDGE_PX)
                    & (np.linalg.norm(positions-center, axis=1) <= radius)
                    & (quads[:, :, 0].min(axis=1) >= 3.)
                    & (quads[:, :, 1].min(axis=1) >= 3.)
                    & (quads[:, :, 0].max(axis=1) < cols-3.)
                    & (quads[:, :, 1].max(axis=1) < rows-3.))
            quads, partners, height, positions = quads[keep], partners[keep], height[keep], positions[keep]
            if not len(quads):
                continue
            if distance is None:
                distance = cv2.distanceTransform(np.where(raw_edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
            pixels = np.rint(quads[:, :, None, :] + fractions[None, None, :, None]
                             *(np.roll(quads, -1, axis=1)-quads)[:, :, None, :]).astype(np.int32)
            support = np.mean(distance[pixels[:, :, :, 1], pixels[:, :, :, 0]] <= 4., axis=2)
            for number in np.flatnonzero(np.min(support, axis=1) >= .95):
                check_head_acquisition_deadline(deadline_monotonic_sec, "candidate_observed_corner_support")
                quad = quads[number]
                ordered = order_corners(tuple(ImagePoint(*point) for point in quad))
                if not metric_corner_arm_support(cv2, raw_edges, ordered).accepted:
                    continue
                # A complete observed pair, rather than its segment length,
                # earns rail priority. Short foreshortened/neck-split rails can
                # therefore retain a slot. Different image regions alternate.
                cell = tuple(np.floor(positions[number]/expected).astype(int))
                score = abs(math.log(height[number]/expected))
                for rail_index in (index, int(partners[number])):
                    item = eligible[rail_index]
                    bucket = priorities.setdefault(cell, {})
                    bucket[item] = min(score, bucket.get(item, math.inf))
                key = tuple(quad.ravel())
                if key in seen:
                    continue
                seen.add(key)
                output.append(tuple(map(tuple, quad)))
                if len(output) > MAX_GUIDED_HINTS:
                    info["overflow"] = True
                    info["supported_hints"] = len(output)
                    return ()
        if prioritized_groups_out is not None:
            buckets = [sorted(bucket, key=lambda item: (bucket[item], -item[0], item[1:]))
                       for _cell, bucket in sorted(priorities.items())]
            selected = []
            for offset in range(max(map(len, buckets), default=0)):
                for bucket in buckets:
                    if offset < len(bucket) and bucket[offset] not in selected:
                        selected.append(bucket[offset])
            prioritized_groups_out.append(tuple(selected[:rail_limit]))
    check_head_acquisition_deadline(deadline_monotonic_sec, "candidate_observed_rail_pairs_complete")
    info["supported_hints"] = len(output)
    return tuple(output)
