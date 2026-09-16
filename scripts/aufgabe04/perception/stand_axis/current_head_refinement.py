"""Shared current-pixel physical-border refinement before any 3D pose solve.

Acquisition and fitting must compare the same enclosing physical rails. A
proposal only locates the existing bounded search; complete raw borders and
current corner arms supply the returned measurement. No identity, pose, neck,
historical angle, or candidate projection supplies accepted corners.
"""

from scripts.aufgabe04.perception.stand_axis.head_border_seed import (
    HeadBorderSeed, select_head_border_seed,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import (
    HeadOuterBorderEvidence, select_current_outer_head_border,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    RefinedHeadMeasurement, refine_projected_head_border,
)
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import check_head_acquisition_deadline


def refine_current_physical_head(
    cv2, raw_edges, *, model_profile, proposal_corners,
    deadline_monotonic_sec=None,
) -> tuple[RefinedHeadMeasurement, HeadOuterBorderEvidence, HeadBorderSeed]:
    """Apply the metric fitter's unchanged raw-border policy to a valid proposal.

    The caller owns proposal/crop validation and candidate association. Optional
    work deadlines stop between raw fits. Evidence has no 3D angle or admission authority;
    the usual fit, source freshness and observer gates remain separate.
    """
    seed = select_head_border_seed(
        model_profile=model_profile, projected_corners=None,
        pose_reprojection_rmse_px=None, current_head_proposal_corners=proposal_corners,
    )
    check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_raw_refinement")
    refinement = refine_projected_head_border(
        cv2, raw_edges, seed.corners, corridor_half_width_px=seed.corridor_half_width_px,
    )
    check_head_acquisition_deadline(deadline_monotonic_sec, "current_head_raw_refinement")
    refinement, outer_recovery = select_current_outer_head_border(
        cv2, raw_edges, model_profile=model_profile, refinement=refinement,
        corridor_half_width_px=seed.corridor_half_width_px,
        neutral_proposal_corners=seed.corners,
        deadline_monotonic_sec=deadline_monotonic_sec,
    )
    return refinement, outer_recovery, seed
