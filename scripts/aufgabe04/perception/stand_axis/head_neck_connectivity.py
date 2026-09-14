"""Bounded raw-pixel continuation of an already measured pair of neck rails.

This is not another neck detector. Measured paired raw runs remain the anchor;
only their short, rounded junction with the fitted head may change column. A
path uses one existing edge in every row, never fills a gap, and never leaves
its two-pixel anchor band. It supplies boundary evidence, not an angle.
"""

from dataclasses import dataclass
import math


MAX_BACKTRACK_ROWS_PX = 6
MAX_LATERAL_SHIFT_PX = 2
MAX_LATERAL_STEP_PX = 1

PixelPath = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RawNeckContinuation:
    accepted: bool
    reason: str
    start_gaps_px: tuple[int, int] | None = None
    paths_px: tuple[PixelPath, PixelPath] | None = None
    max_backtrack_rows_px: int = MAX_BACKTRACK_ROWS_PX
    max_lateral_shift_px: int = MAX_LATERAL_SHIFT_PX
    max_lateral_step_px: int = MAX_LATERAL_STEP_PX


def trace_raw_neck_junction(
    raw_edges, *, bottom_edge_px, rail_columns_px, core_start_row_px,
    core_run_length_px, min_rail_gap_px, max_rail_gap_px, max_start_gap_px,
    core_paths_px=None,
) -> RawNeckContinuation:
    """Trace two supported rails backward without relaxing their head gap.

    The bottom is evaluated at each observed pixel's column, including a
    sloping bottom edge. Only finite, in-image bottoms no steeper than 45
    degrees are eligible for this row-based continuation. Both paths must
    remain separated by the original rail-width bounds in every common row.
    Failed probes are diagnostic only; callers retain the original core gap.
    """

    invalid = RawNeckContinuation(False, "raw_neck_continuation_input_invalid")
    shape = getattr(raw_edges, "shape", ())
    try:
        (left_x, left_y), (right_x, right_y) = bottom_edge_px
        left_anchor, right_anchor = rail_columns_px
        integral = (left_anchor, right_anchor, core_start_row_px,
                    core_run_length_px, min_rail_gap_px, max_rail_gap_px,
                    max_start_gap_px)
        if (len(shape) != 2 or min(shape) <= 0
                or any(isinstance(v, bool) or not isinstance(v, int) for v in integral)
                or not all(math.isfinite(v) for v in (left_x, left_y, right_x, right_y))
                or not 0 <= left_x < right_x < shape[1]
                or not 0 <= min(left_y, right_y) <= max(left_y, right_y) < shape[0]
                or right_x - left_x < 1.0
                or abs(right_y - left_y) > right_x - left_x
                or not left_x <= left_anchor < right_anchor <= right_x
                or not 3 <= min_rail_gap_px <= right_anchor - left_anchor <= max_rail_gap_px
                or core_run_length_px < 3
                or not 0 <= core_start_row_px < core_start_row_px + core_run_length_px <= shape[0]
                or not 0 <= max_start_gap_px <= 2):
            return invalid
    except (TypeError, ValueError):
        return invalid

    slope = (right_y - left_y) / (right_x - left_x)

    def first_row(x):
        return int(math.floor(left_y + slope * (x - left_x) + 1.0))

    def supported(x, y):
        pixel = float(raw_edges[y, x])
        return math.isfinite(pixel) and pixel > 0.0

    if core_paths_px is None:
        core_paths_px = tuple(tuple((x, y) for y in range(
            core_start_row_px, core_start_row_px + core_run_length_px
        )) for x in rail_columns_px)
    try:
        if (len(core_paths_px) != 2
                or any(len(path) < core_run_length_px for path in core_paths_px)
                or any(path[0] != (anchor, core_start_row_px)
                       for path, anchor in zip(core_paths_px, rail_columns_px))):
            return invalid
        for index in range(core_run_length_px):
            for path in core_paths_px:
                x, y = path[index]
                if (type(x) is not int or type(y) is not int
                        or not 0 <= x < shape[1] or y != core_start_row_px + index
                        or not supported(x, y)
                        or (index and abs(x - path[index - 1][0]) > MAX_LATERAL_STEP_PX)):
                    return RawNeckContinuation(False, "raw_neck_continuation_core_unavailable")
            if not min_rail_gap_px <= core_paths_px[1][index][0] - core_paths_px[0][index][0] <= max_rail_gap_px:
                return RawNeckContinuation(False, "raw_neck_continuation_core_unavailable")
    except (TypeError, ValueError, IndexError):
        return invalid
    if any(core_start_row_px < first_row(x) for x in rail_columns_px):
        return invalid

    def paths_for(anchor):
        initial = ((anchor, core_start_row_px),)
        reachable = {anchor: initial}
        candidates = [(core_start_row_px - first_row(anchor), initial)]
        for depth in range(1, MAX_BACKTRACK_ROWS_PX + 1):
            y = core_start_row_px - depth
            if y < 0:
                break
            next_paths = {}
            for prior, path in sorted(reachable.items()):
                for x in range(prior - MAX_LATERAL_STEP_PX, prior + MAX_LATERAL_STEP_PX + 1):
                    if (not left_x <= x <= right_x
                            or abs(x - anchor) > MAX_LATERAL_SHIFT_PX
                            or y < first_row(x) or not supported(x, y)):
                        continue
                    # One deterministic witness per reachable pixel bounds
                    # work to five columns per rail per row.
                    next_paths.setdefault(x, path + ((x, y),))
            for x, path in next_paths.items():
                candidates.append((y - first_row(x), path))
            if not next_paths:
                break
            reachable = next_paths
        return candidates

    def separated(left_path, right_path):
        right_rows = {y: x for x, y in right_path}
        gaps = [right_rows[y] - x for x, y in left_path if y in right_rows]
        gaps.append(right_path[-1][0] - left_path[-1][0])
        return all(min_rail_gap_px <= gap <= max_rail_gap_px for gap in gaps)

    left_paths, right_paths = (paths_for(anchor) for anchor in rail_columns_px)
    best = None
    for left_gap, left_path in left_paths:
        for right_gap, right_path in right_paths:
            if not separated(left_path, right_path):
                continue
            score = (max(left_gap, right_gap), left_gap + right_gap,
                     len(left_path) + len(right_path), left_path, right_path)
            if best is None or score < best[0]:
                best = (score, (left_gap, right_gap), (left_path, right_path))
    if best is None:
        return RawNeckContinuation(False, "raw_neck_continuation_pair_unavailable")
    accepted = max(best[1]) <= max_start_gap_px
    return RawNeckContinuation(
        accepted,
        "raw_neck_continuation_verified" if accepted else "raw_neck_continuation_gap_too_large",
        start_gaps_px=best[1], paths_px=best[2],
    )
