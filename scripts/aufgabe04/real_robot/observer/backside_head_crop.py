"""Require a current, candidate-associated complete head before backside use.

Negative QR evidence from a nominal projection may merely mean that the crop
cut off the QR. A current 2D head proposal must locate the complete head, bind
it to a unique scan cluster, and select the crop used by the strict fit/QR
checks. This is acquisition evidence, never an angle or motion authority.
"""

from dataclasses import asdict, dataclass, replace
import math

from scripts.aufgabe04.real_robot.observer.contract import BACKSIDE_AXIS_SAMPLE_SOURCE
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_metadata_is_unique


@dataclass(frozen=True)
class BacksideHeadCropReview:
    accepted: bool
    reason: str
    crop_xyxy: tuple | None = None
    head_bounds_full_image: tuple | None = None

    def metadata(self):
        return {**asdict(self), "basis": "current_candidate_head_proposal",
                "neck_required": False, "motion_authorized": False}


def review_current_head_crop(selection, *, require_marker_absence=False):
    """Bind the current fitted head to its proposal, crop and scan target.

    Geometric completeness is meaningful for either face. Backside consumers
    additionally require explicit marker absence through the wrapper below.
    """
    result = BacksideHeadCropReview(False, "current_complete_head_crop_required")
    acquisition = selection.head_acquisition or {}
    if (not selection.registered
            or acquisition.get("reason") != "current_head_proposal_strict_retry"
            or acquisition.get("candidate_associated") is not True):
        return result
    association = acquisition.get("lidar_association") or {}
    cluster = association.get("search_association") or {}
    unique = (registered_target_metadata_is_unique(association)
              if association.get("witnessed_fragmentation") is not None else
              association.get("associated") is True
              and type(cluster.get("eligible_cluster_count")) is int
              and cluster["eligible_cluster_count"] == 1)
    if not unique:
        return replace(result, reason="complete_head_unique_association_required")
    current = selection.selected
    if require_marker_absence and (current.qr_observations != ()
            or current.debug.qr_detected is not False
            or current.debug.qr_marker_verified is not False):
        return replace(result, reason="complete_head_current_marker_absence_required")
    roi = current.attempt.roi
    bounds = acquisition.get("head_bounds_full_image")
    corners = current.estimate.corners
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or corners is None or len(corners) != 4):
        return replace(result, reason="complete_head_bounds_unavailable")
    x0, y0, x1, y1 = bounds
    result = replace(result, crop_xyxy=(roi.x0, roi.y0, roi.x1, roi.y1),
                     head_bounds_full_image=tuple(bounds))
    margin = max(2., .03 * (y1 - y0))
    if not (x0 < x1 and y0 < y1
            and roi.x0 + margin <= x0 < x1 <= roi.x1 - margin
            and roi.y0 + margin <= y0 < y1 <= roi.y1 - margin):
        return replace(result, reason="complete_head_crop_clipped")
    points = [(p.u_px + roi.x0, p.v_px + roi.y0) for p in corners]
    if (any(not math.isfinite(v) for point in points for v in point)
            or not all(roi.x0 + 2 <= u <= roi.x1 - 2
                       and roi.y0 + 2 <= v <= roi.y1 - 2 for u, v in points)):
        return replace(result, reason="fitted_head_crop_clipped")
    # Refinement/outer-border selection can move proposal rails slightly.
    # A different rectangle in the same crop cannot inherit its association.
    fitted = (min(p[0] for p in points), min(p[1] for p in points),
              max(p[0] for p in points), max(p[1] for p in points))
    limit = max(3., .12 * (y1 - y0))
    if any(abs(a - b) > limit for a, b in zip(fitted, bounds)):
        return replace(result, reason="fitted_head_differs_from_current_proposal")
    return replace(result, accepted=True, reason="current_complete_head_crop_verified")


def review_backside_head_crop(selection):
    """Require complete current head pixels and explicit marker absence."""
    return review_current_head_crop(selection, require_marker_absence=True)


def gate_backside_head_crop(selection):
    """Withhold only backside use; retain all current-image QR veto evidence."""
    current = selection.selected
    if not current.estimate.usable or current.estimate.source != BACKSIDE_AXIS_SAMPLE_SOURCE:
        return selection, None
    review = review_backside_head_crop(selection)
    if review.accepted:
        return selection, review
    withheld = replace(
        current,
        estimate=replace(current.estimate, usable=False, yaw_deg=None,
                         reason="backside_complete_head_crop_unverified",
                         evidence_state="unobservable"),
        debug=replace(current.debug, model_reason="backside_complete_head_crop_unverified",
                      evidence_state="unobservable"),
    )
    return replace(selection, selected=withheld,
                   strict_retry=withheld if selection.strict_retry is current else selection.strict_retry,
                   evaluations=tuple(withheld if item is current else item
                                     for item in selection.evaluations)), review
