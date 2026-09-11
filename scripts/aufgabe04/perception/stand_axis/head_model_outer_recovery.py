"""At most two current-edge searches outside an apparent inner head border."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry import _polygon_area
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_model_neck import measure_head_neck_junction
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


@dataclass(frozen=True)
class OuterHeadRecovery:
    accepted: bool
    reason: str
    original_corners: tuple[ImagePoint, ...]
    original_neck_start_gap_px: int
    attempted_growth_factors: tuple[float, ...]
    recovered_corners: tuple[ImagePoint, ...] | None = None


def recover_outer_head_border(cv2, raw_edges, *, model_profile, refinement, junction, corridor_half_width_px):
    """Use a measured neck gap to relocate search, never to fill an edge.

    Half-gap and full-gap outward seeds are tried in that fixed order. Selection
    uses only complete raw rails/corners, increased extent and a directly joined
    neck. No QR, pose, residual ranking or altered model size participates.
    """

    if (not refinement.accepted or refinement.corners is None
            or junction.reason != "head_neck_junction_gap_too_large"
            or junction.start_gap_px is None):
        return refinement, junction, None
    corners = tuple(refinement.corners)
    height = sum(math.hypot(corners[a].u_px - corners[b].u_px,
                           corners[a].v_px - corners[b].v_px)
                 for a, b in ((0, 3), (1, 2))) / 2.0
    gap = junction.start_gap_px
    if not 0.0 < gap <= min(8.0, 0.10 * height):
        return refinement, junction, OuterHeadRecovery(
            False, "head_outer_recovery_gap_out_of_bounds", corners, gap, (),
        )
    center = (sum(p.u_px for p in corners) / 4.0, sum(p.v_px for p in corners) / 4.0)
    original_area = _polygon_area(corners)
    growths = []
    for padding in (gap / 2.0, float(gap)):
        growth = 1.0 + 2.0 * padding / height
        growths.append(growth)
        proposed = tuple(ImagePoint(
            center[0] + (p.u_px - center[0]) * growth,
            center[1] + (p.v_px - center[1]) * growth,
        ) for p in corners)
        try:
            proposed = validate_current_head_proposal(proposed, frame_shape=raw_edges.shape)
        except ValueError:
            continue
        current = refine_projected_head_border(
            cv2, raw_edges, proposed, corridor_half_width_px=corridor_half_width_px,
        )
        if not current.accepted or current.corners is None:
            continue
        area = _polygon_area(current.corners)
        # A rail search snapping back onto the paper is not outer recovery.
        if not 1.03 * original_area <= area <= 1.45 * original_area:
            continue
        current_junction = measure_head_neck_junction(raw_edges, current.corners, model_profile)
        if not current_junction.accepted:
            continue
        return current, current_junction, OuterHeadRecovery(
            True, "current_raw_outer_head_recovered", corners, gap,
            tuple(growths), tuple(current.corners),
        )
    return refinement, junction, OuterHeadRecovery(
        False, "head_outer_recovery_unavailable", corners, gap, tuple(growths),
    )
