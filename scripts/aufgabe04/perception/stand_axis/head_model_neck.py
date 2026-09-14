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

from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _distance, order_corners
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_neck_connectivity import (
    RawNeckContinuation, trace_raw_neck_junction,
)
from scripts.aufgabe04.perception.stand_axis.raw_neck_support import measure_raw_neck_support


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
    core_start_gap_px: int | None = None
    raw_continuation: RawNeckContinuation | None = None
    core_paths_px: tuple[tuple[tuple[int, int], ...], ...] = ()


def measure_head_neck_junction(raw_edges, corners, profile) -> HeadNeckJunction:
    """Require adjacent raw neck rails at a resolvable physical-head boundary.

    The start gap counts missing rows after the first pixel row below the
    fitted bottom-edge midpoint. Both rails must then have a simultaneous
    uninterrupted raw core run. Exact-column cores retain precedence; a
    bounded perspective-following core is considered only if none exists.
    ``rail_columns_px`` are the core's starting columns, while ``core_paths_px``
    records every actual pixel along each possibly slanted rail.
    A rounded junction may continue backward through
    adjacent raw pixels within a bounded two-pixel band. The core run's gap,
    columns and start row remain separate from that connected path's gap.
    Rejected probes retain the original start gap for outer-border recovery.

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
    bottom_y = (bottom_left.v_px + bottom_right.v_px) / 2.0
    y0 = max(0, int(math.floor(bottom_y + 1.0)))
    required_run = max(3, int(math.ceil(0.12 * height)))
    min_rail_gap = max(3, int(round(0.07 * width)))
    max_rail_gap = max(min_rail_gap + 1, int(round(0.34 * width)))
    core = measure_raw_neck_support(raw_edges, validated)
    if not core.paths_px or core.start_gap_px is None:
        return HeadNeckJunction(False, "head_neck_junction_rails_unavailable",
                                required_run_px=required_run, **resolution_fields)
    # Core discovery retains diagnostic paths even beyond its permissive
    # locator gap. Physical-head admission still uses the stricter measured
    # panel-inset and at-most-two-pixel boundary checks below.
    gap = core.start_gap_px
    left_column, right_column = (path[0][0] for path in core.paths_px)
    accepted = resolution_qualified and gap <= max_gap
    reason = (
        "head_neck_panel_separation_unresolved" if not resolution_qualified else
        "head_neck_junction_verified" if accepted else "head_neck_junction_gap_too_large"
    )
    junction = HeadNeckJunction(
        accepted, reason,
        start_gap_px=gap, core_start_gap_px=gap, required_run_px=required_run,
        rail_columns_px=(left_column, right_column), run_start_row_px=y0 + gap,
        core_paths_px=core.paths_px,
        **resolution_fields,
    )
    if reason != "head_neck_junction_gap_too_large":
        return junction
    continuation = trace_raw_neck_junction(
        raw_edges,
        bottom_edge_px=((bottom_left.u_px, bottom_left.v_px),
                        (bottom_right.u_px, bottom_right.v_px)),
        rail_columns_px=junction.rail_columns_px, core_start_row_px=junction.run_start_row_px,
        core_run_length_px=required_run, min_rail_gap_px=min_rail_gap,
        max_rail_gap_px=max_rail_gap, max_start_gap_px=max_gap,
        core_paths_px=core.paths_px,
    )
    junction = replace(junction, raw_continuation=continuation)
    if not continuation.accepted:
        return junction
    return replace(junction, accepted=True, reason="head_neck_junction_verified",
                   start_gap_px=max(continuation.start_gaps_px))
