"""Use the shared current-border locator within a candidate's search image.

The caller has already bounded the search around the projected candidate. Its
position and physical scale prune hypotheses before comparison, not after an
unrestricted locator spends its budget. Every proposal still needs current
camera/LiDAR association and a new metric fit.
"""

import math
import time

from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposalResult


def acquire_viewer_candidate_head(
    cv2, frame, *, expected_center, expected_height, max_center_offset_ratio,
    edge_preprocess, canny_low, canny_high, deadline_monotonic_sec, diagnostics,
    proposal_filter=None,
):
    """Locate current complete borders using the viewer's candidate-aware path.

    Use the caller's whole bounded search image so a displaced head can retain
    all four borders. Reserve 25 ms for its strict fit and retain the original
    deadline; a failed comparison never starts another expensive locator.
    """
    now = time.monotonic()
    remaining = math.inf if deadline_monotonic_sec is None else deadline_monotonic_sec - now
    if (remaining <= .045 or frame is None or not hasattr(frame, "shape")
            or len(frame.shape) != 3):
        diagnostics.update(reason="viewer_candidate_trial_unavailable", performed=False)
        return None
    height, width = frame.shape[:2]
    cx, cy = expected_center
    if (not all(math.isfinite(v) for v in (cx, cy, expected_height, max_center_offset_ratio))
            or expected_height < 12 or max_center_offset_ratio <= 0):
        diagnostics.update(reason="viewer_candidate_trial_invalid_projection", performed=False)
        return None
    trial_deadline = now + .15 if deadline_monotonic_sec is None else deadline_monotonic_sec - .025
    result = acquire_cold_head_proposal(
        cv2, frame, edge_preprocess=edge_preprocess,
        canny_low=canny_low, canny_high=canny_high,
        expected_head_center_u_px=cx, expected_head_center_v_px=cy,
        expected_head_height_px=expected_height,
        expected_head_height_tolerance_ratio=.30,
        max_center_offset_ratio=max_center_offset_ratio,
        proposal_filter=proposal_filter,
        deadline_monotonic_sec=trial_deadline)
    diagnostics.update(performed=True, locator="shared_candidate_current_borders",
        crop_xyxy=(0, 0, width, height), reason=result.reason,
        elapsed_ms=(time.monotonic() - now) * 1000,
        comparison=result.joint_border_diagnostics,
        candidate_bounds_applied_before_comparison=True,
        considered_proposals=result.considered_proposals, raw_verifications=result.raw_verifications,
        angle_authorized=False, motion_authorized=False)
    proposal = result.proposal
    if proposal is None:
        return result
    ratio = proposal.observed_height_px / expected_height
    offset = math.dist((proposal.center_u_px, proposal.center_v_px), expected_center) / expected_height
    if (not .70 <= ratio <= 1.30 or offset > max_center_offset_ratio
            or abs(proposal.center_v_px - cy) > min(.75, max_center_offset_ratio) * expected_height):
        diagnostics["reason"] = "viewer_head_outside_candidate_projection"
        return HeadProposalResult(None, diagnostics["reason"], result.considered_proposals,
                                  result.raw_verifications, result.locator,
                                  result.joint_border_diagnostics)
    return result
