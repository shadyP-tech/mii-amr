"""Compare current supporting rails, not approximate head sizes or solved yaw."""

import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance


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


def _single_thin_stripe(frame, a, b, c, d):
    """Prove two nearby Canny rails bound one thin colour ridge.

    Opposite image-gradient polarity distinguishes the two edges of one drawn
    rail from nearby, same-polarity physical boundaries. The test reads current
    colours at five positions; position/size similarity alone cannot merge rails.
    """
    import numpy as np

    tangent = np.array((b.u_px - a.u_px, b.v_px - a.v_px))
    tangent /= max(float(np.linalg.norm(tangent)), 1.)
    normal = np.array((-tangent[1], tangent[0]))
    first = np.array((a.u_px, a.v_px))
    last = np.array((b.u_px, b.v_px))
    other_first = np.array((c.u_px, c.v_px))
    other_last = np.array((d.u_px, d.v_px))
    offsets = ((other_first - first) @ normal, (other_last - last) @ normal)
    if max(abs(value) for value in offsets) > 6.:
        return False

    def sample(positions):
        low = np.floor(positions).astype(int)
        if (low[:, 0].min() < 0 or low[:, 1].min() < 0
                or low[:, 0].max() + 1 >= frame.shape[1]
                or low[:, 1].max() + 1 >= frame.shape[0]):
            return None
        fraction = positions - low
        fx, fy = fraction[:, 0, None], fraction[:, 1, None]
        x, y = low[:, 0], low[:, 1]
        return ((1.-fx)*(1.-fy)*frame[y, x] + fx*(1.-fy)*frame[y, x+1]
                + (1.-fx)*fy*frame[y+1, x] + fx*fy*frame[y+1, x+1])

    supported = 0
    opposite_gradients = 0
    multiple_ridges = 0
    for fraction in (.2, .35, .5, .65, .8):
        p = first + fraction * (last - first)
        q = other_first + fraction * (other_last - other_first)
        values_at_rails = sample(np.asarray((p - .75 * normal, p + .75 * normal,
                                             q - .75 * normal, q + .75 * normal)))
        if values_at_rails is not None:
            gradient_a = values_at_rails[1] - values_at_rails[0]
            gradient_b = values_at_rails[3] - values_at_rails[2]
            magnitude = float(np.linalg.norm(gradient_a) * np.linalg.norm(gradient_b))
            opposite_gradients += bool(magnitude >= 400.
                and float(gradient_a @ gradient_b) <= -.8 * magnitude)
        middle = (p + q) / 2.
        # A single measured colour ridge has matching surroundings and exactly
        # one excursion across the two Canny rails. This also tolerates oblique
        # one-pixel rasterization without treating nearby distinct rails as one.
        radius = abs(float((q-p) @ normal)) / 2. + 2.5
        positions = middle + np.linspace(-radius, radius, 25)[:, None] * normal
        values = sample(positions)
        if values is None:
            return False
        channel = int(np.argmax(np.ptp(values, axis=0)))
        profile = values[:, channel]
        baseline = (profile[0] + profile[-1]) / 2.
        if abs(profile[0] - profile[-1]) > 15. or np.ptp(profile) < 25.:
            continue
        positive = float(profile.max() - baseline) >= float(baseline - profile.min())
        excursion = profile - baseline if positive else baseline - profile
        if excursion.min() < -15.:
            continue
        # More than one separated excursion denotes separate raw image rails.
        active = excursion >= max(12., .25 * float(excursion.max()))
        starts = np.flatnonzero(active & ~np.roll(active, 1))
        if len(starts) > 1:
            multiple_ridges += 1
        if len(starts) != 1 or active[0] or active[-1]:
            continue
        supported += 1
    return multiple_ridges < 2 and (supported >= 4 or opposite_gradients >= 4)


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

    def signature(self, corners):
        key = tuple(corners)
        if key not in self._signatures:
            self._signatures[key] = rail_signature(self.raw_edges, key)
        return self._signatures[key]

    def same(self, first, second):
        key = (tuple(first), tuple(second))
        if key not in self._comparisons:
            if self.raw_edges is None:
                result = same_current_border_family(*key)
            elif max(_distance(a, b) for a, b in zip(*key)) > 9.:
                result = False
            else:
                result = same_current_border_family(*key, raw_edges=self.raw_edges,
                    frame_bgr=self.frame_bgr, first_signature=self.signature(key[0]),
                    second_signature=self.signature(key[1]))
            self._comparisons[key] = result
            self._comparisons[key[::-1]] = result
        return self._comparisons[key]
