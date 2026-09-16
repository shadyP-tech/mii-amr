"""Bind a freshly refitted tracked head to its current crop and scan target.

A previous head locates a bounded search only. This proof describes the new
image's measured geometry; it is deliberately separate from the two-pass 2D
proposal/retry protocol. QR identity and marker absence remain consumer gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.observation_freshness import observation_freshness
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    CameraTargetRegistrationSelection, HeadRoiEvaluation,
)
from scripts.aufgabe04.real_robot.observer.current_head_association import CurrentHeadCandidateAssociation
from scripts.aufgabe04.real_robot.observer.current_head_detection import current_head_detection_admission
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_is_unique


TRACKED_HEAD_SOURCE = "candidate_tracked_head_search"


@dataclass(frozen=True)
class CurrentMeasuredHeadRegistration:
    """Current in-memory binding; never serialize the image/evaluation objects."""

    accepted: bool
    reason: str
    evaluation: HeadRoiEvaluation = field(repr=False, compare=False)
    association: CurrentHeadCandidateAssociation | None = field(repr=False)
    observed_at_sec: float
    checked_at_sec: float
    max_age_sec: float
    expected_model_sha256: str
    head_bounds_full_image: tuple[float, ...] | None = None

    def metadata(self):
        return {"accepted": self.accepted, "reason": self.reason,
                "basis": "current_measured_candidate_head",
                "observed_at_sec": self.observed_at_sec,
                "checked_at_sec": self.checked_at_sec,
                "max_age_sec": self.max_age_sec,
                "head_bounds_full_image": self.head_bounds_full_image,
                "candidate_associated": self.accepted,
                "previous_measurement_reused": False,
                "proposal_retry_claimed": False, "neck_required": False,
                "motion_authorized": False}


def tracked_head_selection(evaluation: HeadRoiEvaluation) -> CameraTargetRegistrationSelection:
    """Wrap one newly evaluated tracked crop, with no registration authority."""
    return CameraTargetRegistrationSelection(
        selected=evaluation, evaluations=(evaluation,), proposal=None,
        decision=None, strict_retry=None, reacquisition_mode=None,
        search_hint_used=True,
    )


def _current_geometry_bounds(proof):
    current = proof.evaluation
    estimate, debug = current.estimate, current.debug
    if current.attempt.source != TRACKED_HEAD_SOURCE:
        return "current_tracked_head_crop_required", None
    if (not math.isfinite(proof.max_age_sec) or proof.max_age_sec <= 0
            or not observation_freshness(
                observed_at_sec=proof.observed_at_sec, now_sec=proof.checked_at_sec,
                max_age_sec=proof.max_age_sec).accepted):
        return "current_tracked_head_freshness_required", None
    admission, bounds = current_head_detection_admission(
        estimate, debug, profile_sha256=proof.expected_model_sha256)
    if not admission.accepted and bounds is None:
        return admission.reason, None
    if estimate.model_profile_sha256 != proof.expected_model_sha256:
        return "current_tracked_head_profile_mismatch", None
    association = proof.association
    if (not isinstance(association, CurrentHeadCandidateAssociation)
            or association.accepted is not True
            or association.reason != "current_head_unique_lidar_cluster"
            or association.head_admission != admission
            or association.head_orientation_bounds != bounds
            or association.roi_source != current.attempt.source
            or not association.scale_gate
            or association.scale_gate.get("accepted") is not True
            or not registered_target_is_unique(association.lidar_association)):
        return "complete_head_unique_association_required", None
    if not observation_freshness(
            observed_at_sec=association.lidar_association.search_association.scan_stamp_sec,
            now_sec=proof.checked_at_sec, max_age_sec=proof.max_age_sec).accepted:
        return "complete_head_current_scan_freshness_required", None
    roi = current.attempt.roi
    try:
        corners = validate_current_head_proposal(
            estimate.corners, frame_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0))
        points = tuple((p.u_px + roi.x0, p.v_px + roi.y0) for p in corners)
        center = tuple(sum(p[index] for p in points) / 4 for index in (0, 1))
        if (association.full_image_center_px is None
                or any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-9)
                       for a, b in zip(center, association.full_image_center_px))):
            return "current_head_association_geometry_mismatch", None
        bounds = (min(p[0] for p in points), min(p[1] for p in points),
                  max(p[0] for p in points), max(p[1] for p in points))
        margin = max(2., .03 * (bounds[3] - bounds[1]))
        if not (roi.x0 + margin <= bounds[0] < bounds[2] <= roi.x1 - margin
                and roi.y0 + margin <= bounds[1] < bounds[3] <= roi.y1 - margin):
            return "complete_head_crop_clipped", bounds
    except (TypeError, ValueError, ArithmeticError):
        return "complete_head_bounds_unavailable", None
    return "current_measured_head_crop_verified", bounds


def register_current_tracked_head(
    selection: CameraTargetRegistrationSelection, *, association,
    observed_at_sec: float, now_sec: float, max_age_sec: float,
    expected_model_sha256: str,
) -> CameraTargetRegistrationSelection:
    """Validate current geometry/association before any backside crop gate.

    The association must have been computed against the original current map
    projection, current exact-time scan and this selected fit. The observer
    still rechecks sensor freshness before publishing or accumulating evidence.
    """
    proof = CurrentMeasuredHeadRegistration(
        False, "current_complete_head_crop_required", selection.selected,
        association, observed_at_sec, now_sec, max_age_sec, expected_model_sha256)
    reason, bounds = _current_geometry_bounds(proof)
    return replace(selection, current_measured_head_registration=replace(
        proof, accepted=reason == "current_measured_head_crop_verified",
        reason=reason, head_bounds_full_image=bounds))


def review_current_tracked_head_crop(selection, *, require_marker_absence=False):
    """Review the explicit current proof without inheriting prior crop evidence."""
    from scripts.aufgabe04.real_robot.observer.backside_head_crop import BacksideHeadCropReview

    review = BacksideHeadCropReview(False, "current_complete_head_crop_required",
                                   basis="current_measured_candidate_head")
    proof = selection.current_measured_head_registration
    if (not isinstance(proof, CurrentMeasuredHeadRegistration)
            or proof.evaluation is not selection.selected):
        return replace(review, reason="current_head_crop_proof_evaluation_mismatch")
    reason, bounds = _current_geometry_bounds(proof)
    if (not proof.accepted or reason != "current_measured_head_crop_verified"
            or bounds != proof.head_bounds_full_image):
        return replace(review, reason=reason if reason != "current_measured_head_crop_verified"
                       else "current_head_crop_proof_rejected")
    current = selection.selected
    roi = current.attempt.roi
    review = replace(review, crop_xyxy=(roi.x0, roi.y0, roi.x1, roi.y1),
                     head_bounds_full_image=bounds)
    if require_marker_absence and (current.qr_observations != ()
            or current.debug.qr_detected is not False
            or current.debug.qr_marker_verified is not False):
        return replace(review, reason="complete_head_current_marker_absence_required")
    return replace(review, accepted=True, reason="current_complete_head_crop_verified")
