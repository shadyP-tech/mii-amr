"""Geometry-only suppression of repeated radiator-rib edge families.

The stand head is defined by two parallel outer rails.  A heater creates a
larger family of similarly vertical rails at regular spacing.  This module
detects only that repeated family and returns a binary exclusion mask; it does
not use color, QR content, or stand coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _VerticalRail:
    x_px: float
    y0_px: float
    y1_px: float
    length_px: float


@dataclass(frozen=True)
class RadiatorRibMaskResult:
    """Diagnostic result for a repeated vertical-rib exclusion mask."""

    mask: object
    detected_rail_count: int
    suppressed_rail_count: int


def _vertical_overlap(first: _VerticalRail, second: _VerticalRail) -> float:
    overlap = max(0.0, min(first.y1_px, second.y1_px) - max(first.y0_px, second.y0_px))
    return overlap / max(1.0, min(first.length_px, second.length_px))


def _deduplicate_rails(rails: list[_VerticalRail]) -> list[_VerticalRail]:
    """Collapse duplicate Hough segments that describe one physical rail."""

    deduplicated: list[_VerticalRail] = []
    for rail in sorted(rails, key=lambda item: (-item.length_px, item.x_px)):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(deduplicated)
                if abs(existing.x_px - rail.x_px) <= 3.0
                and _vertical_overlap(existing, rail) >= 0.70
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(rail)
        elif rail.length_px > deduplicated[duplicate_index].length_px:
            deduplicated[duplicate_index] = rail
    return sorted(deduplicated, key=lambda item: item.x_px)


def _regular_rib_runs(
    rails: list[_VerticalRail],
    *,
    min_rib_count: int,
    max_gap_px: float,
) -> tuple[tuple[_VerticalRail, ...], ...]:
    """Return maximal runs of vertically aligned, near-periodic rails."""

    runs: list[tuple[_VerticalRail, ...]] = []
    for start in range(len(rails)):
        run = [rails[start]]
        gaps: list[float] = []
        for candidate in rails[start + 1 :]:
            previous = run[-1]
            gap = candidate.x_px - previous.x_px
            if gap <= 3.0 or gap > max_gap_px or _vertical_overlap(previous, candidate) < 0.70:
                break
            proposed_gaps = (*gaps, gap)
            median_gap = sorted(proposed_gaps)[len(proposed_gaps) // 2]
            if any(
                abs(item - median_gap) > max(2.0, 0.40 * median_gap)
                for item in proposed_gaps
            ):
                break
            run.append(candidate)
            gaps.append(gap)
        if len(run) >= min_rib_count:
            runs.append(tuple(run))

    # The same physical run is discovered from each of its early members.
    # Keep only maximal runs, then merge their rail membership below.
    maximal: list[tuple[_VerticalRail, ...]] = []
    for run in runs:
        members = {round(rail.x_px, 3) for rail in run}
        if any(members < {round(rail.x_px, 3) for rail in other} for other in runs):
            continue
        maximal.append(run)
    return tuple(maximal)


def repeated_vertical_rib_exclusion_mask(
    cv2,
    edges,
    *,
    min_rib_count: int = 4,
    max_angle_deg: float = 12.0,
    min_length_fraction: float = 0.20,
    max_gap_fraction: float = 0.08,
    mask_width_px: int = 5,
) -> RadiatorRibMaskResult:
    """Mask repeated radiator-like rails while preserving isolated stand sides.

    A valid family needs at least ``min_rib_count`` almost vertical, similarly
    tall rails with regular horizontal spacing.  The two parallel rails of the
    square stand cannot satisfy that cardinality requirement on their own.
    """

    import numpy

    frame_height, frame_width = edges.shape[:2]
    mask = numpy.zeros((frame_height, frame_width), dtype=numpy.uint8)
    if frame_height <= 0 or frame_width <= 0:
        return RadiatorRibMaskResult(mask, 0, 0)
    if min_rib_count < 4 or max_angle_deg <= 0.0 or min_length_fraction <= 0.0:
        raise ValueError("invalid repeated-rib suppression configuration")

    minimum_length = max(12, int(round(min_length_fraction * frame_height)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=numpy.pi / 180.0,
        threshold=max(12, int(round(0.04 * frame_height))),
        minLineLength=minimum_length,
        maxLineGap=max(4, int(round(0.02 * frame_height))),
    )
    if lines is None:
        return RadiatorRibMaskResult(mask, 0, 0)

    tangent_limit = math.tan(math.radians(max_angle_deg))
    rails = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < minimum_length or abs(dy) <= 1e-6:
            continue
        if abs(dx) > tangent_limit * abs(dy):
            continue
        rails.append(
            _VerticalRail(
                x_px=(float(x1) + float(x2)) / 2.0,
                y0_px=min(float(y1), float(y2)),
                y1_px=max(float(y1), float(y2)),
                length_px=length,
            )
        )

    rails = _deduplicate_rails(rails)
    max_gap_px = max(8.0, min(32.0, max_gap_fraction * frame_width))
    runs = _regular_rib_runs(
        rails,
        min_rib_count=min_rib_count,
        max_gap_px=max_gap_px,
    )
    suppressed = {
        (round(rail.x_px, 3), round(rail.y0_px, 3), round(rail.y1_px, 3))
        for run in runs
        for rail in run
    }
    # Mask the complete rail band rather than only individual Hough segments.
    # A rib can intermittently disappear from Hough because of compression or
    # Canny gaps; extending the verified periodic band by half a rib spacing
    # keeps that one missing background edge from re-entering next frame.
    line_width = max(1, int(mask_width_px))
    for run in runs:
        gaps = [right.x_px - left.x_px for left, right in zip(run, run[1:])]
        spacing = sorted(gaps)[len(gaps) // 2]
        x0 = max(0, int(math.floor(run[0].x_px - 0.5 * spacing - line_width / 2)))
        x1 = min(
            frame_width - 1,
            int(math.ceil(run[-1].x_px + 0.5 * spacing + line_width / 2)),
        )
        y0 = max(0, int(math.floor(min(rail.y0_px for rail in run))) - 2)
        y1 = min(
            frame_height - 1,
            int(math.ceil(max(rail.y1_px for rail in run))) + 2,
        )
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    return RadiatorRibMaskResult(mask, len(rails), len(suppressed))
