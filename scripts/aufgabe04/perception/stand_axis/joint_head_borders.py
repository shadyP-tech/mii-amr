"""Bounded four-rail localization scored on the current image.

Short line fragments locate independent sides of a closed head. Intersections
are search hypotheses only: callers must run the unchanged raw border/corner
verification and measured-head solver. Neither a projection nor a previous
pose supplies measurement pixels.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint

MAX_RAILS_PER_DIRECTION = 24
MAX_INPUT_RAILS_PER_DIRECTION = 128
MAX_JOINT_HYPOTHESES = 64
MAX_PAIR_COMBINATIONS = (MAX_RAILS_PER_DIRECTION * (MAX_RAILS_PER_DIRECTION - 1) // 2) ** 2
MAX_ENDPOINT_HINTS = 12


@dataclass(frozen=True)
class JointHeadBorderHypothesis:
    corners: tuple[ImagePoint, ...]
    score: float
    minimum_raw_support: float
    mean_raw_support: float
    mean_gradient: float
    locator: str = "independent_four_rails"


def _rails(segments, *, height, center):
    """Merge collinear fragments geometrically; do not bridge evidence gaps."""
    groups = ([], [])
    for segment in segments:
        x0, y0, x1, y1 = map(float, segment)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if not .20 * height <= length <= 1.50 * height:
            continue
        direction = 0 if abs(dy) <= .30 * abs(dx) else 1 if abs(dx) <= .30 * abs(dy) else None
        if direction is None:
            continue
        first, last = (x0, y0), (x1, y1)
        if first[direction] > last[direction]:
            first, last = last, first
        cross = 1 - direction
        slope = (last[cross] - first[cross]) / (last[direction] - first[direction])
        intercept = first[cross] - slope * first[direction]
        midpoint = tuple((a + b) / 2 for a, b in zip(first, last))
        if (abs(midpoint[0] - center[0]) > 2 * height
                or abs(midpoint[1] - center[1]) > 1.6 * height):
            continue
        groups[direction].append((slope, intercept, first[direction], last[direction], length))
    merged_groups = []
    for direction, group in enumerate(groups):
        merged = []
        # Longest current fragments define the locator direction; extending
        # its interval is not evidence for missing raw pixels.
        ranked = sorted(group, key=lambda item: (-item[4], item[:4]))
        for slope, intercept, start, end, length in ranked[:MAX_INPUT_RAILS_PER_DIRECTION]:
            for index, (m, b, low, high, strength) in enumerate(merged):
                overlap = min(end, high) - max(start, low)
                coordinate = (max(start, low) + min(end, high)) / 2
                if (overlap >= -.12 * height and abs(slope - m) <= .015
                        and abs((slope - m) * coordinate + intercept - b) <= 1.25):
                    merged[index] = (m, b, min(start, low), max(end, high), strength)
                    break
            else:
                merged.append((slope, intercept, start, end, length))
        merged.sort(key=lambda item: (-min(item[3] - item[2], height), item[:4]))
        merged_groups.append(tuple(merged[:MAX_RAILS_PER_DIRECTION]))
    return tuple(merged_groups)


def _pairs(rails, *, direction, height):
    pairs = []
    for index, first in enumerate(rails):
        for second in rails[index + 1:]:
            coordinate = (max(first[2], second[2]) + min(first[3], second[3])) / 2
            a, b = first[0] * coordinate + first[1], second[0] * coordinate + second[1]
            separation = abs(a - b)
            if not (.65 * height if direction == 0 else .30 * height) <= separation <= 1.35 * height:
                continue
            # Opposite fragments need not overlap: the other two current
            # rails establish their shared endpoints. Full four-side closure
            # and each fragment's overlap are checked on the joint hypothesis.
            pairs.append((first, second) if a < b else (second, first))
    return pairs


def _intersect(horizontal, vertical):
    mh, bh = horizontal[:2]
    mv, bv = vertical[:2]
    x = (mv * bh + bv) / (1 - mh * mv)
    return ImagePoint(x, mh * x + bh)


def rank_joint_head_borders(cv2, frame, raw_edges, *, expected_height, expected_center,
                           max_center_offset, locator_lines=None, hint_corners=()):
    """Rank bounded closed four-line hypotheses before expensive raw fitting."""
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lsd = (locator_lines if locator_lines is not None else
           cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0])
    hough = cv2.HoughLinesP(
        raw_edges, 1., np.pi / 180., threshold=max(8, round(.15 * expected_height)),
        minLineLength=max(6, round(.20 * expected_height)),
        maxLineGap=max(2, round(.08 * expected_height)),
    )
    segments = [tuple(p) for lines in (lsd, hough) if lines is not None
                for p in lines.reshape(-1, 4)]
    horizontal, vertical = _rails(segments, height=expected_height, center=expected_center)
    horizontal_pairs = _pairs(horizontal, direction=0, height=expected_height)
    vertical_pairs = _pairs(vertical, direction=1, height=expected_height)
    distance = cv2.distanceTransform(np.where(raw_edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    scale = max(float(np.percentile(magnitude[raw_edges > 0], 75)) if np.any(raw_edges) else 1., 1.)
    rows, cols = raw_edges.shape
    candidates = []
    considered = 0
    fractions = np.linspace(.10, .90, 24)
    for top, bottom in horizontal_pairs:
        for left, right in vertical_pairs:
            corners = (_intersect(top, left), _intersect(top, right),
                       _intersect(bottom, right), _intersect(bottom, left))
            if not all(2 <= p.u_px < cols - 2 and 2 <= p.v_px < rows - 2 for p in corners):
                continue
            width = (_distance(corners[0], corners[1]) + _distance(corners[2], corners[3])) / 2
            height = (_distance(corners[0], corners[3]) + _distance(corners[1], corners[2])) / 2
            center = (sum(p.u_px for p in corners) / 4, sum(p.v_px for p in corners) / 4)
            if (not .70 <= height / expected_height <= 1.30
                    or not .35 <= width / max(height, 1.) <= 1.35
                    or math.dist(center, expected_center) > max_center_offset * expected_height):
                continue
            # Every side must overlap observed locator fragments. Closure
            # cannot be inferred by intersecting four distant background lines.
            bounds = ((corners[0].u_px, corners[1].u_px), (corners[3].u_px, corners[2].u_px),
                      (corners[0].v_px, corners[3].v_px), (corners[1].v_px, corners[2].v_px))
            if any(min(rail[3], high) - max(rail[2], low) < .25 * (high - low)
                   for rail, (low, high) in zip((top, bottom, left, right), bounds)):
                continue
            considered += 1
            candidates.append((tuple(order_corners(corners)), "independent_four_rails"))
    candidates.extend((tuple(order_corners(corners)), "paired_locator_endpoints")
                      for corners in tuple(hint_corners)[:MAX_ENDPOINT_HINTS])
    by_pixels = {}
    for corners, locator in candidates:
        points = np.asarray([(p.u_px, p.v_px) for p in corners])
        pixels = np.rint(points[:, None, :] + fractions[None, :, None]
                         * (np.roll(points, -1, axis=0) - points)[:, None, :]).astype(np.int32)
        xs, ys = pixels[:, :, 0], pixels[:, :, 1]
        if np.any(xs < 0) or np.any(xs >= cols) or np.any(ys < 0) or np.any(ys >= rows):
            continue
        supports = np.mean(distance[ys, xs] <= 2., axis=1)
        minimum, mean = float(min(supports)), float(np.mean(supports))
        gradient = float(np.mean(np.minimum(magnitude[ys, xs] / scale, 1.)))
        score = .60 * minimum + .30 * mean + .10 * gradient
        if minimum < .35:
            continue
        # Quantized localization keys bound duplicate work; raw verification
        # still returns subpixel corners independently for every chosen seed.
        key = tuple(round(value / 1.5) for value in points.ravel()) + (locator,)
        old = by_pixels.get(key)
        if old is None or score > old.score:
            by_pixels[key] = JointHeadBorderHypothesis(corners, score, minimum, mean, gradient, locator)
    hypotheses = list(by_pixels.values())
    hypotheses.sort(key=lambda item: (-item.score, tuple((p.u_px, p.v_px) for p in item.corners), item.locator))
    # Preserve bounded endpoint hints alongside joint ranking. They locate
    # rare complete long rails when clutter consumes directional line slots.
    hints = [item for item in hypotheses if item.locator == "paired_locator_endpoints"][:4]
    selected = hypotheses[:MAX_JOINT_HYPOTHESES - len(hints)]
    selected.extend(item for item in hints if item not in selected)
    return tuple(selected), {
        "method": "joint_current_four_rail_support", "horizontal_rails": len(horizontal),
        "vertical_rails": len(vertical), "horizontal_pairs": len(horizontal_pairs),
        "vertical_pairs": len(vertical_pairs), "considered_closed_hypotheses": considered,
        "compared_pair_combinations": len(horizontal_pairs) * len(vertical_pairs),
        "scored_current_hypotheses": len(candidates),
        "max_pair_combinations": MAX_PAIR_COMBINATIONS,
        "max_scored_current_hypotheses": MAX_PAIR_COMBINATIONS + MAX_ENDPOINT_HINTS,
        "retained_hypotheses": len(selected),
        "max_rails_per_direction": MAX_RAILS_PER_DIRECTION,
        "max_input_rails_per_direction": MAX_INPUT_RAILS_PER_DIRECTION,
        "max_retained_hypotheses": MAX_JOINT_HYPOTHESES,
        "angle_authorized": False, "motion_authorized": False,
    }
