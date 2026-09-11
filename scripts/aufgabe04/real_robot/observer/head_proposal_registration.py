"""Associate a current 2D head proposal before spending time on metric fitting.

The bounded search has no pose prerequisite. Its pixels only choose a complete
head/neck crop; the caller still performs the ordinary strict metric fit and
publication freshness/consensus checks. No proposal is a measurement.
"""

from dataclasses import asdict, dataclass, replace
from typing import Callable

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis.head_proposal import (
    HeadProposal, acquire_head_proposal,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_handoff import rectified_pixel_bearing_in_scan
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, ImageRoi
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    BACKSIDE_REACQUISITION_MODE, QR_MODEL_REACQUISITION_MODE,
    CameraTargetRegistrationSelection, HeadRoiEvaluation,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt, HeadRoiRegistrationDecision,
    REGISTERED_BACKSIDE_REACQUISITION_SOURCE,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE, registered_head_roi_attempt,
)
from scripts.aufgabe04.real_robot.observer.camera_framing import build_camera_framing_hint


@dataclass(frozen=True)
class RegisteredHeadProposal:
    attempt: HeadRoiAttempt
    decision: HeadRoiRegistrationDecision
    corners: tuple[ImagePoint, ...]  # Coordinates in the new crop, offset once.
    metadata: dict[str, object]


def recenter_head_proposal(
    proposal: HeadProposal, search: HeadRoiAttempt, *,
    max_center_offset_ratio: float,
) -> RegisteredHeadProposal | None:
    """Shrink the certified search to the whole observed head, neck and margin."""
    decision = registered_head_roi_attempt(
        search, proposal.corners, max_center_offset_ratio=max_center_offset_ratio,
        registered_source=REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
    )
    if not decision.accepted or decision.attempt is None:
        return None
    x0, y0, x1, y1 = proposal.bounds_xyxy
    if not (0 <= x0 < x1 <= search.roi.x1 - search.roi.x0
            and 0 <= y0 < y1 <= search.roi.y1 - search.roi.y0):
        return None
    # Require every fitted border inside the crop. The proposal builder also
    # requires paired neck rails; clipping cannot invent that evidence.
    if not all(x0 < p.u_px < x1 - 1 and y0 < p.v_px < y1 - 1
               for p in proposal.corners):
        return None
    roi = ImageRoi(x0 + search.roi.x0, y0 + search.roi.y0,
                   x1 + search.roi.x0, y1 + search.roi.y0,
                   search.roi.expected_size_px)
    attempt = replace(decision.attempt, roi=roi)
    return RegisteredHeadProposal(
        attempt, replace(decision, attempt=attempt),
        tuple(ImagePoint(p.u_px - x0, p.v_px - y0) for p in proposal.corners),
        {"candidate_associated": False, "proposal": asdict(proposal),
         "search_roi": search.metadata(), "head_bounds_full_image": [
             proposal.head_bounds_xyxy[0] + search.roi.x0,
             proposal.head_bounds_xyxy[1] + search.roi.y0,
             proposal.head_bounds_xyxy[2] + search.roi.x0,
             proposal.head_bounds_xyxy[3] + search.roi.y0,
         ], "motion_authorized": False, "measurement_reused": False},
    )


def acquire_registered_head_measurement(
    cv2, frame, search: HeadRoiAttempt, *, intrinsics: CameraIntrinsics,
    scan_from_camera, scan, map_bearing_rad: float, cone_half_angle_rad: float,
    accepted_range_m: tuple[float, float], now_sec: float,
    max_scan_age_sec: float, min_cluster_sample_count: int,
    max_camera_map_bearing_delta_rad: float, max_center_offset_ratio: float,
    edge_preprocess: str, canny_low: int, canny_high: int,
    evaluate: Callable[[HeadRoiAttempt, tuple[ImagePoint, ...]], HeadRoiEvaluation],
    diagnostics: dict[str, object],
    primary: HeadRoiEvaluation | None = None,
) -> CameraTargetRegistrationSelection | None:
    """At most one geometric retry, after unique candidate/LiDAR association."""
    import time

    start = time.monotonic()

    def reject_proposal():
        # Once a physical proposal contradicts candidate association, another
        # expensive wide metric fit cannot repair that sensor evidence. Keep
        # the failed nominal result and its QR conflict/veto channel; do not
        # spend the current image's remaining freshness budget fitting it.
        if primary is None:
            return None
        return CameraTargetRegistrationSelection(
            selected=primary, evaluations=(), proposal=None, decision=None,
            strict_retry=None, reacquisition_mode=None,
            head_acquisition=dict(diagnostics),
        )

    roi = search.roi
    result = acquire_head_proposal(
        cv2, frame[roi.y0:roi.y1, roi.x0:roi.x1],
        expected_head_center_u_px=search.expected_center_u_px - roi.x0,
        expected_head_center_v_px=search.expected_center_v_px - roi.y0,
        expected_head_height_px=search.expected_head_height_px,
        edge_preprocess=edge_preprocess, canny_low=canny_low, canny_high=canny_high,
        max_center_offset_fraction=max_center_offset_ratio,
    )
    diagnostics.update(reason=result.reason, considered_proposals=result.considered_proposals,
                       raw_verifications=result.raw_verifications,
                       elapsed_ms=(time.monotonic() - start) * 1000.0,
                       candidate_associated=False)
    if result.proposal is None:
        if result.reason == "head_proposal_ambiguous":
            return reject_proposal()
        return None
    registered = recenter_head_proposal(
        result.proposal, search, max_center_offset_ratio=max_center_offset_ratio,
    )
    if registered is None:
        diagnostics["reason"] = "head_proposal_crop_registration_rejected"
        return reject_proposal()
    try:
        bearing = rectified_pixel_bearing_in_scan(
            u_px=result.proposal.center_u_px + roi.x0,
            v_px=result.proposal.center_v_px + roi.y0,
            fx_px=intrinsics.fx_px, fy_px=intrinsics.fy_px,
            cx_px=intrinsics.cx_px, cy_px=intrinsics.cy_px,
            scan_from_camera=scan_from_camera,
        )
    except ValueError:
        diagnostics["reason"] = "head_proposal_camera_bearing_unavailable"
        return reject_proposal()
    association = associate_camera_registered_candidate_lidar_target(
        scan, map_bearing_rad=map_bearing_rad, observed_camera_bearing_rad=bearing,
        cone_half_angle_rad=cone_half_angle_rad, accepted_range_m=accepted_range_m,
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        min_cluster_sample_count=min_cluster_sample_count,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
    )
    diagnostics.update(registered.metadata, candidate_associated=association.associated,
                       lidar_association=asdict(association))
    if not association.associated:
        diagnostics["reason"] = "head_proposal_candidate_association_rejected"
        return reject_proposal()
    # The current-image proposal is only a raw-border seed. It is never used
    # as a QR pose, temporal pose, accepted normal or cached angle.
    strict = evaluate(registered.attempt, registered.corners)
    source = (REGISTERED_QR_MODEL_REACQUISITION_SOURCE if strict.debug.qr_detected
              else REGISTERED_BACKSIDE_REACQUISITION_SOURCE)
    strict = replace(strict, attempt=replace(strict.attempt, source=source))
    decision = replace(registered.decision, attempt=strict.attempt)
    diagnostics["reason"] = "current_head_proposal_strict_retry"
    return CameraTargetRegistrationSelection(
        selected=strict, evaluations=(strict,), proposal=None, decision=decision,
        strict_retry=strict,
        reacquisition_mode=(QR_MODEL_REACQUISITION_MODE if strict.debug.qr_detected
                            else BACKSIDE_REACQUISITION_MODE),
        head_acquisition=dict(diagnostics),
    )


def unresolved_front_framing_hint(
    selection: CameraTargetRegistrationSelection, *, target_key: str,
    source_image_stamp_sec: float, source_fresh: bool, range_m: float,
    optical_depth_m: float, intrinsics: CameraIntrinsics,
) -> dict | None:
    """A failed strict joint fit may request another range, never an angle."""
    current, acquisition = selection.selected, selection.head_acquisition
    if (not source_fresh or not selection.registered or acquisition is None
            or acquisition.get("candidate_associated") is not True
            or current.estimate.usable
            or current.debug.qr_marker_verified is not True
            or current.debug.model_pose_fit_source != "joint_qr_head"):
        return None
    return build_camera_framing_hint(
        target_key=target_key, source_image_stamp_sec=source_image_stamp_sec,
        front_evidence_verified=True, candidate_associated=True,
        reason="head_qr_geometry_mismatch", range_m=range_m,
        optical_depth_m=optical_depth_m,
        image_size=(intrinsics.width_px, intrinsics.height_px),
        head_bounds=tuple(acquisition["head_bounds_full_image"]),
        fx_px=intrinsics.fx_px, fy_px=intrinsics.fy_px,
    )
