"""Bounded paired-rail continuity measured only on current Canny pixels.

The small perspective-following corridors locate evidence; they never fill a
missing row, join disconnected pixels, or move the fitted head. This result
is a neck cue, not a head angle, face classification, or candidate identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners


MAX_RAIL_SLOPE_DX_DY = 0.25
_RECOVERY_SLOPE_OFFSETS = (0., .025, -.025, .05, -.05, .075, -.075,
                           .10, -.10, .125, -.125, .15, -.15)


@dataclass(frozen=True)
class RawNeckSupport:
    accepted: bool
    reason: str
    required_run: int = 0
    paired_run: int = 0
    start_gap_px: int | None = None
    maximum_start_gap_px: int = 0
    slope_dx_dy: float | None = None
    # Actual original-image edge pixels, never corridor centres or synthetic pixels.
    paths_px: tuple[tuple[tuple[int, int], ...], ...] = ()


def measure_raw_neck_support(edge_mask, corners) -> RawNeckSupport:
    """Require two separated, 8-connected raw rails beneath the head.

    Preserve the original neck extent, minimum paired run, rail separation,
    and start-gap bounds. Only the exact-column assumption changes: each
    straight search corridor has a one-pixel radius and follows a bounded
    slope near the observed head sides. Every accepted row contains two
    distinct, separated raw pixels connected to the preceding row.
    A late core is retained as rejected diagnostic evidence for the stricter
    physical head-boundary check; having paths alone does not mean acceptance.
    """

    import numpy as np

    if (not isinstance(edge_mask, np.ndarray) or edge_mask.ndim != 2
            or edge_mask.dtype.kind not in "buif" or edge_mask.size == 0
            or not np.isfinite(edge_mask).all()):
        return RawNeckSupport(False, "raw_neck_input_invalid")
    try:
        if corners is None or len(corners) != 4 or not all(
            math.isfinite(float(v)) for p in corners for v in (p.u_px, p.v_px)
        ):
            return RawNeckSupport(False, "raw_neck_corners_invalid")
        tl, tr, br, bl = order_corners(corners)
        width = (_distance(tl, tr) + _distance(bl, br)) / 2.0
        height = (_distance(tl, bl) + _distance(tr, br)) / 2.0
        if width <= 1.0 or height <= 1.0 or br.v_px <= tr.v_px or bl.v_px <= tl.v_px:
            return RawNeckSupport(False, "raw_neck_corners_invalid")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return RawNeckSupport(False, "raw_neck_corners_invalid")

    center_x = (bl.u_px + br.u_px) / 2.0
    bottom_y = (bl.v_px + br.v_px) / 2.0
    radius = max(3, int(round(0.16 * width)))
    y0 = max(0, int(math.floor(bottom_y + 1.0)))
    y1 = min(edge_mask.shape[0], int(math.ceil(bottom_y + 0.42 * height)))
    x0 = max(0, int(math.floor(center_x - radius)))
    x1 = min(edge_mask.shape[1], int(math.ceil(center_x + radius)) + 1)
    required = max(3, int(math.ceil(0.12 * height)))
    maximum_start_gap = max(3, int(math.ceil(0.08 * height)))
    if y1 - y0 < required or x1 <= x0:
        return RawNeckSupport(False, "raw_neck_window_unavailable", required,
                              maximum_start_gap_px=maximum_start_gap)
    raw = edge_mask[y0:y1, x0:x1] > 0
    min_gap = max(3, int(round(0.07 * width)))
    max_gap = max(min_gap + 1, int(round(0.34 * width)))
    # Candidate pairs include the one-pixel corridor margins. Actual witness
    # separation is checked independently on every row, so a thick single
    # edge cannot masquerade as two rails.
    left, right = np.triu_indices(raw.shape[1], k=max(1, min_gap - 2))
    keep = right - left <= max_gap + 2
    left, right = left[keep], right[keep]
    if not len(left):
        return RawNeckSupport(False, "raw_neck_pair_unavailable", required,
                              maximum_start_gap_px=maximum_start_gap)

    head_slope = ((bl.u_px - tl.u_px) / (bl.v_px - tl.v_px)
                  + (br.u_px - tr.u_px) / (br.v_px - tr.v_px)) / 2.0
    # Thirteen fixed proposals, nearest to the measured head direction first.
    # This is localization only; no tilt or continuity rejection is relaxed.
    slopes = tuple(dict.fromkeys(max(-MAX_RAIL_SLOPE_DX_DY,
                                     min(MAX_RAIL_SLOPE_DX_DY, head_slope + offset))
                                 for offset in _RECOVERY_SLOPE_OFFSETS))
    rows = np.arange(raw.shape[0])
    bases = np.arange(raw.shape[1])[:, None]
    best = 0
    best_core = None
    # Preserve existing measurable exact-column cores and their boundary-gap
    # diagnostics. Recovery is needed only when rasterized/slanted rails have
    # no such core, not to replace an existing rejected physical boundary.
    for slope, corridor_radius in ((0.0, 0), *((value, 1) for value in slopes)):
        centres = np.rint(bases + slope * rows).astype(int)
        witness = np.full(centres.shape, -1, dtype=int)
        # Prefer the nearest raw pixel, then the left/right one-pixel margin.
        for delta in ((0,) if corridor_radius == 0 else (0, -1, 1)):
            columns = centres + delta
            valid = (columns >= 0) & (columns < raw.shape[1])
            present = valid & raw[rows, np.clip(columns, 0, raw.shape[1] - 1)]
            selected = (witness < 0) & present
            witness[selected] = columns[selected]
        lx, rx = witness[left], witness[right]
        paired = ((lx >= 0) & (rx >= 0) & (rx - lx >= min_gap)
                  & (rx - lx <= max_gap))
        runs = np.zeros(len(left), dtype=int)
        for row in range(raw.shape[0]):
            connected = (np.zeros(len(left), dtype=bool) if row == 0 else
                         (np.abs(lx[:, row] - lx[:, row - 1]) <= 1)
                         & (np.abs(rx[:, row] - rx[:, row - 1]) <= 1))
            runs = np.where(paired[:, row], np.where(connected, runs + 1, 1), 0)
            starts = row - runs + 1
            eligible = starts <= maximum_start_gap
            if eligible.any():
                best = max(best, int(runs[eligible].max()))
            accepted = np.flatnonzero(runs >= required)
            if len(accepted):
                index = int(min(accepted, key=lambda i: (starts[i], left[i], right[i])))
                start = int(starts[index])
                paths = tuple(tuple((int(xs[index, y]) + x0, y + y0)
                                    for y in range(start, row + 1)) for xs in (lx, rx))
                core = RawNeckSupport(start <= maximum_start_gap,
                                      ("current_raw_paired_neck_rails" if start <= maximum_start_gap
                                       else "raw_neck_start_gap_too_large"), required,
                                      int(runs[index]), start, maximum_start_gap, float(slope), paths)
                if best_core is None or start < best_core.start_gap_px:
                    best_core = core
                if start == 0:
                    return core
                break  # Later runs on this slope cannot begin earlier.
        if corridor_radius == 0 and best_core is not None:
            return best_core
    if best_core is not None:
        return best_core
    return RawNeckSupport(False, "raw_neck_paired_continuity_insufficient", required,
                          best, maximum_start_gap_px=maximum_start_gap)
