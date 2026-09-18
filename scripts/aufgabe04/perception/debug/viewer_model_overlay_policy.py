"""Render current model admission without promoting a diagnostic to a handoff."""

from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.perception.stand_axis.head_model_admission import (
    admit_measured_head_model,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.head_outer_border import current_head_boundary_eligible


def current_border_diagnostic(*, estimate, artifacts, local_result_fresh, source_fresh):
    """Display current-image border proof separately from live pose admission."""
    quality = None if artifacts is None else artifacts.head_model_quality
    verified = bool(local_result_fresh and estimate is not None and artifacts is not None
        and current_head_boundary_eligible(estimate, artifacts)
        and quality is not None and quality.raw_corner_support_accepted
        and quality.outer_border_verified)
    return {"visible": verified, "motion_authorized": False, "pose_authorized": False,
            "state": ("current_image_borders" if source_fresh else "delayed_image_borders")
                     if verified else "borders_unavailable",
            "reason": None if estimate is None else estimate.reason}


def estimate_in_full_image(crop_estimate, roi):
    """Translate only the display/association copy; retain exact proof pixels."""
    if crop_estimate.corners is None or roi is None:
        return crop_estimate
    return replace(crop_estimate, corners=tuple(
        ImagePoint(p.u_px + roi.x0, p.v_px + roi.y0) for p in crop_estimate.corners))


def current_crop_head_estimate(*, crop_estimate, displayed_estimate, roi,
                               current_accepted, held):
    """Use bound crop geometry only while this exact current fit is displayed.

    Never subtract display offsets to reconstruct a proof: floating-point
    round trips need not preserve exact selected border pixels. A held or
    replaced display estimate cannot borrow the new detector's quality either.
    """
    if held or not current_accepted:
        return replace(crop_estimate, usable=False, reason="current_head_selection_not_fresh")
    if displayed_estimate != estimate_in_full_image(crop_estimate, roi):
        return replace(crop_estimate, usable=False, reason="current_head_selection_mismatch")
    return crop_estimate


@dataclass(frozen=True)
class ModelOverlayState:
    state: str
    reason: str
    current_fit_accepted: bool
    geometry_color: tuple[int, int, int]
    status_color: tuple[int, int, int]


def current_model_overlay_state(
    *, inputs_ready, estimate, artifacts, result_fresh, freshness_reason=None,
) -> ModelOverlayState:
    """Use the final measured-head contract; never inspect landmarks as proof.

    Purple is reserved for a current admitted single-frame geometric fit.
    Even that state says nothing about target association, temporal consensus,
    directed face identity, or permission to move.
    """
    gray = (150, 150, 150)
    if not result_fresh:
        return ModelOverlayState(
            "obsolete_result", freshness_reason or "obsolete_detector_result",
            False, gray, gray,
        )
    if not inputs_ready or estimate is None or artifacts is None:
        return ModelOverlayState(
            "inputs_unavailable", "metric_inputs_unavailable", False, gray, gray,
        )
    if estimate.evidence_state == "predicted_only":
        return ModelOverlayState(
            "proposal_only", estimate.reason, False, (0, 190, 255), (0, 190, 255),
        )
    yaw = getattr(estimate, "yaw_deg", None)
    admission = admit_measured_head_model(
        estimate=estimate, debug=artifacts,
        yaw_rad=math.radians(yaw) if type(yaw) in (int, float) else math.nan,
    )
    if admission.accepted:
        return ModelOverlayState(
            "current_head_fit", admission.reason, True, (180, 0, 180), (0, 255, 0),
        )
    reason = admission.reason
    if estimate.usable is not True:
        reason = estimate.reason
    elif reason == "measured_head_quality_rejected":
        quality = getattr(artifacts, "head_model_quality", None)
        if quality is not None and not quality.accepted:
            reason = quality.reason
    return ModelOverlayState("rejected_fit", reason, False, gray, (0, 120, 255))
