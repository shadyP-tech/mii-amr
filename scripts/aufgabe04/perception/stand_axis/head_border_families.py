"""Compare current supporting rails, not approximate head sizes or solved yaw."""

import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance
from scripts.aufgabe04.perception.stand_axis.current_rail_profiles import single_thin_stripe as _single_thin_stripe


def _rail_pixels(raw_edges, a, b):
    import numpy as np

    x0 = max(0, int(min(a.u_px, b.u_px)) - 2)
    x1 = min(raw_edges.shape[1], int(math.ceil(max(a.u_px, b.u_px))) + 3)
    y0 = max(0, int(min(a.v_px, b.v_px)) - 2)
    y1 = min(raw_edges.shape[0], int(math.ceil(max(a.v_px, b.v_px))) + 3)
    y, x = np.nonzero(raw_edges[y0:y1, x0:x1])
    points = np.column_stack((x + x0, y + y0))
    delta = np.array((b.u_px - a.u_px, b.v_px - a.v_px))
    length = float(np.linalg.norm(delta))
    tangent = delta / max(length, 1.)
    relative = points - (a.u_px, a.v_px)
    along = relative @ tangent
    offset = abs(relative @ np.array((-tangent[1], tangent[0])))
    keep = (along >= .12 * length) & (along <= .88 * length) & (offset <= 1.5)
    return frozenset((points[keep, 1] * raw_edges.shape[1] + points[keep, 0]).tolist())


def rail_signature(raw_edges, corners):
    return tuple(_rail_pixels(raw_edges, a, b)
                 for a, b in zip(corners, corners[1:] + corners[:1]))


def same_supporting_rails(first, second):
    return all(a and b and len(a & b) >= .70 * max(len(a), len(b))
               for a, b in zip(first, second))


def same_current_border_family(first, second, *, raw_edges=None, frame_bgr=None,
                               first_signature=None, second_signature=None):
    """Only measured pixel overlap or one evidenced thin ridge identifies a rail."""
    if max(_distance(a, b) for a, b in zip(first, second)) > 9.:
        return False
    if raw_edges is None:
        return max(_distance(a, b) for a, b in zip(first, second)) <= 1.5
    first_signature = first_signature or rail_signature(raw_edges, first)
    second_signature = second_signature or rail_signature(raw_edges, second)
    for index, (a, b) in enumerate(zip(first_signature, second_signature)):
        if max(_distance(first[index], second[index]),
               _distance(first[(index + 1) % 4], second[(index + 1) % 4])) <= 1.5 and a and b:
            continue
        if a and b and len(a & b) >= .70 * max(len(a), len(b)):
            continue
        if frame_bgr is None or not _single_thin_stripe(
                frame_bgr, first[index], first[(index + 1) % 4],
                second[index], second[(index + 1) % 4]):
            return False
    return True


class CurrentBorderFamilies:
    """Per-image cache; nothing survives into another camera observation."""

    def __init__(self, raw_edges=None, frame_bgr=None):
        self.raw_edges = raw_edges
        self.frame_bgr = frame_bgr
        self._signatures = {}
        self._comparisons = {}
        self._rail_signatures = {}
        self._rail_comparisons = {}

    def signature(self, corners):
        key = tuple(corners)
        if key not in self._signatures:
            self._signatures[key] = tuple(self._rail_signature(a, b)
                for a, b in zip(key, key[1:]+key[:1]))
        return self._signatures[key]

    def _rail_signature(self, a, b):
        key = (a, b)
        if key not in self._rail_signatures:
            self._rail_signatures[key] = _rail_pixels(self.raw_edges, a, b)
        return self._rail_signatures[key]

    def _same_rail(self, a, b, c, d):
        key = (a, b, c, d)
        if key not in self._rail_comparisons:
            first, second = self._rail_signature(a, b), self._rail_signature(c, d)
            result = bool(first and second and (
                max(_distance(a, c), _distance(b, d)) <= 1.5
                or len(first & second) >= .70 * max(len(first), len(second))))
            if not result and self.frame_bgr is not None:
                result = _single_thin_stripe(self.frame_bgr, a, b, c, d)
            self._rail_comparisons[key] = result
        return self._rail_comparisons[key]

    def same(self, first, second):
        key = (tuple(first), tuple(second))
        if key not in self._comparisons:
            if self.raw_edges is None:
                result = same_current_border_family(*key)
            elif max(_distance(a, b) for a, b in zip(*key)) > 9.:
                result = False
            else:
                result = all(self._same_rail(first[i], first[(i+1)%4],
                                             second[i], second[(i+1)%4]) for i in range(4))
            self._comparisons[key] = result
            self._comparisons[key[::-1]] = result
        return self._comparisons[key]
