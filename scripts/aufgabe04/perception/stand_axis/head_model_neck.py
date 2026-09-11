"""Verify a current outer-head boundary through its adjacent paired neck rails.

The generic head locator tolerates gaps proportional to head height. That is
useful for proposing a crop, but can mistake an inner paper border for the
measured head: the real neck may begin several pixels below that rectangle.
Physical head admission caps that gap at two pixels, and tightens the cap when
the measured paper-panel inset is only a few image rows. A two-pixel allowance
accounts for edge localization and row quantization. This is an engineering
resolution check, not an absolute proof of boundary identity or metrology.
QR symbol dimensions, decoded quads, pose history, interpolation and morphology
never supply boundary evidence.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal


MAX_HEAD_NECK_START_GAP_PX = 2
HEAD_PANEL_PIXEL_UNCERTAINTY_PX = 2.0


@dataclass(frozen=True)
class HeadNeckJunction:
    accepted: bool
    reason: str
    start_gap_px: int | None = None
    max_start_gap_px: int = MAX_HEAD_NECK_START_GAP_PX
    required_run_px: int | None = None
    rail_columns_px: tuple[int, int] | None = None
    run_start_row_px: int | None = None
    expected_panel_inset_px: float | None = None
    head_vertical_span_px: float | None = None
    measured_panel_inset_m: float | None = None
    pixel_uncertainty_allowance_px: float = HEAD_PANEL_PIXEL_UNCERTAINTY_PX


def measure_head_neck_junction(raw_edges, corners, profile) -> HeadNeckJunction:
    """Require adjacent raw neck rails at a resolvable physical-head boundary.

    The start gap counts missing rows after the first pixel row below the
    fitted bottom-edge midpoint. Both rails must then have a simultaneous
    uninterrupted run; the search never fills missing pixels. The diagnostic
    minimum is retained even when a distant neck fails admission.

    For the measured centered paper panel, its per-side physical inset is
    (head height - panel height) / 2. Scaling by the smaller vertical head-side
    span estimates the inset in image rows conservatively under perspective.
    The allowed integer gap is at most two and strictly smaller than the
    inset minus the two-pixel allowance. An inset no larger than that
    allowance is unresolved even with a zero-row gap.
    This calculation never changes the model or contributes an angle.
    """

    if not profile.committable or profile.environment != "physical":
        return HeadNeckJunction(False, "head_neck_junction_physical_profile_required")
    shape = getattr(raw_edges, "shape", ())
    if len(shape) != 2 or min(shape) <= 0:
        return HeadNeckJunction(False, "head_neck_junction_input_invalid")
    try:
        validated = validate_current_head_proposal(corners, frame_shape=shape)
    except ValueError:
        validated = None
    if validated is None:
        return HeadNeckJunction(False, "head_neck_junction_input_invalid")
    top_left, top_right, bottom_right, bottom_left = order_corners(validated)
    panel_height = profile.qr_panel_height_m
    if (panel_height is None or not math.isfinite(panel_height)
            or not 0.0 < panel_height < profile.head_height_m):
        return HeadNeckJunction(False, "head_neck_panel_dimensions_unavailable")
    width = (_distance(top_left, top_right) + _distance(bottom_left, bottom_right)) / 2.0
    height = (_distance(top_left, bottom_left) + _distance(top_right, bottom_right)) / 2.0
    vertical_span = min(abs(bottom_left.v_px - top_left.v_px),
                        abs(bottom_right.v_px - top_right.v_px))
    panel_inset_m = (profile.head_height_m - panel_height) / 2.0
    inset_px = vertical_span * panel_inset_m / profile.head_height_m
    resolution_qualified = inset_px > HEAD_PANEL_PIXEL_UNCERTAINTY_PX
    max_gap = min(MAX_HEAD_NECK_START_GAP_PX,
                  max(0, math.ceil(inset_px - HEAD_PANEL_PIXEL_UNCERTAINTY_PX) - 1))
    resolution_fields = dict(
        max_start_gap_px=max_gap, expected_panel_inset_px=inset_px,
        head_vertical_span_px=vertical_span, measured_panel_inset_m=panel_inset_m,
    )
    center_x = (bottom_left.u_px + bottom_right.u_px) / 2.0
    bottom_y = (bottom_left.v_px + bottom_right.v_px) / 2.0
    x_radius = max(3, int(round(0.16 * width)))
    x0 = max(0, int(math.floor(center_x - x_radius)))
    x1 = min(shape[1], int(math.ceil(center_x + x_radius)) + 1)
    y0 = max(0, int(math.floor(bottom_y + 1.0)))
    y1 = min(shape[0], int(math.ceil(bottom_y + 0.42 * height)))
    required_run = max(3, int(math.ceil(0.12 * height)))
    if y1 - y0 < required_run or x1 <= x0:
        return HeadNeckJunction(False, "head_neck_junction_rails_unavailable",
                                required_run_px=required_run, **resolution_fields)
    neck = raw_edges[y0:y1, x0:x1] > 0
    min_rail_gap = max(3, int(round(0.07 * width)))
    max_rail_gap = max(min_rail_gap + 1, int(round(0.34 * width)))
    best = None
    for left_column in range(neck.shape[1]):
        for right_column in range(left_column + min_rail_gap,
                                  min(left_column + max_rail_gap + 1, neck.shape[1])):
            run = 0
            for row_index, present in enumerate(neck[:, left_column] & neck[:, right_column]):
                run = run + 1 if present else 0
                if run >= required_run:
                    gap = row_index - run + 1
                    candidate = (gap, left_column + x0, right_column + x0)
                    if best is None or candidate < best:
                        best = candidate
                    break
    if best is None:
        return HeadNeckJunction(False, "head_neck_junction_rails_unavailable",
                                required_run_px=required_run, **resolution_fields)
    gap, left_column, right_column = best
    accepted = resolution_qualified and gap <= max_gap
    reason = (
        "head_neck_panel_separation_unresolved" if not resolution_qualified else
        "head_neck_junction_verified" if accepted else "head_neck_junction_gap_too_large"
    )
    return HeadNeckJunction(
        accepted, reason,
        start_gap_px=gap, required_run_px=required_run,
        rail_columns_px=(left_column, right_column), run_start_row_px=y0 + gap,
        **resolution_fields,
    )
