"""Bind a quality-admitted current head to a candidate independently of its ROI.

The crop's source remains acquisition provenance. A successful nominal fit
needs the same bounded image-to-map association as a reacquired fit, without
running another detector or inventing a recentering receipt. QR identity and
stationary evidence accumulation remain independent downstream gates.
"""

from dataclasses import asdict, dataclass, replace
import math

from scripts.aufgabe04.perception.candidate_lidar_association import (
    CameraRegisteredCandidateLidarAssociation,
    associate_camera_registered_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_model_admission import (
    HeadModelAdmission, admit_measured_head_model,
)
from scripts.aufgabe04.perception.stand_axis_handoff import rectified_pixel_bearing_in_scan
from scripts.aufgabe04.real_robot.configuration.geometry import validate_intrinsics
from scripts.aufgabe04.real_robot.observer.head_model_admission import (
    head_scale_gate, measured_head_lidar_rejection,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    validate_backside_registration_center_offset_ratio,
)


@dataclass(frozen=True)
class CurrentHeadCandidateAssociation:
    accepted: bool
    reason: str
    roi_source: str
    head_admission: HeadModelAdmission
    projected_center_px: tuple[float, float]
    max_center_offset_ratio: float
    scale_gate: dict | None = None
    full_image_center_px: tuple[float, float] | None = None
    center_offset_ratio: float | None = None
    fitted_head_bearing_rad: float | None = None
    lidar_association: CameraRegisteredCandidateLidarAssociation | None = None

    def metadata(self) -> dict:
        return {**asdict(self), "schema_version": 1,
                "association_source": "current_measured_head",
                "unique_eligible_cluster_required": True,
                "motion_authorized": False, "completion_authorized": False}


def associate_current_measured_head(
    *, estimate, debug, attempt, projection, expected_head_height_px, profile_sha256,
    intrinsics, scan_from_camera, scan, map_bearing_rad, cone_half_angle_rad,
    accepted_range_m, now_sec, max_scan_age_sec, min_cluster_sample_count,
    max_center_offset_ratio, max_camera_map_bearing_delta_rad,
    resolve_lidar_association=None,
) -> CurrentHeadCandidateAssociation:
    """Require current geometry, original projection bounds and a unique scan target.

    ``projection`` and ``expected_head_height_px`` must be the original map/TF
    projection, not the selected attempt's potentially recentered expectation.
    Intrinsics describe the full rectified image; the ROI offset is added once.
    The caller checks image freshness before this call and again at publication.
    """
    limit = validate_backside_registration_center_offset_ratio(max_center_offset_ratio)
    yaw = estimate.yaw_deg
    admission = admit_measured_head_model(
        estimate=estimate, debug=debug,
        yaw_rad=math.radians(yaw) if type(yaw) in (int, float) else math.nan,
    )
    result = CurrentHeadCandidateAssociation(
        False, admission.reason, attempt.source, admission,
        (projection.u_px, projection.v_px), limit,
    )
    if not admission.accepted:
        return result
    if estimate.model_profile_sha256 != profile_sha256:
        return replace(result, reason="current_head_profile_mismatch")
    try:
        scale = head_scale_gate(expected_size_px=expected_head_height_px,
                                left_height_px=estimate.left_height_px,
                                right_height_px=estimate.right_height_px)
    except (TypeError, ValueError, ArithmeticError):
        return replace(result, reason="current_head_scale_invalid")
    result = replace(result, scale_gate=scale)
    if not scale["accepted"]:
        return replace(result, reason=scale["reason"])
    roi = attempt.roi
    try:
        validate_intrinsics(intrinsics)
        if (not all(type(v) is int for v in (roi.x0, roi.x1, roi.y0, roi.y1))
                or not 0 <= roi.x0 < roi.x1 <= intrinsics.width_px
                or not 0 <= roi.y0 < roi.y1 <= intrinsics.height_px):
            raise ValueError("ROI must lie inside the full image")
        corners = validate_current_head_proposal(
            estimate.corners, frame_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0),
        )
        if (not all(math.isfinite(v) for v in (
                projection.u_px, projection.v_px, projection.depth_m, expected_head_height_px))
                or projection.depth_m <= 0 or expected_head_height_px <= 0):
            raise ValueError("original projection must be finite and in front of the camera")
        center = (sum(p.u_px for p in corners) / 4 + roi.x0,
                  sum(p.v_px for p in corners) / 4 + roi.y0)
        ratio = math.hypot(center[0] - projection.u_px,
                           center[1] - projection.v_px) / expected_head_height_px
        result = replace(result, full_image_center_px=center, center_offset_ratio=ratio)
        if ratio > limit:
            return replace(result, reason="current_head_outside_registration_window")
        bearing = rectified_pixel_bearing_in_scan(
            u_px=center[0], v_px=center[1], fx_px=intrinsics.fx_px,
            fy_px=intrinsics.fy_px, cx_px=intrinsics.cx_px, cy_px=intrinsics.cy_px,
            scan_from_camera=scan_from_camera,
        )
    except (TypeError, ValueError, ArithmeticError):
        return replace(result, reason="current_head_projection_invalid")
    association = associate_camera_registered_candidate_lidar_target(
        scan, map_bearing_rad=map_bearing_rad, observed_camera_bearing_rad=bearing,
        cone_half_angle_rad=cone_half_angle_rad, accepted_range_m=accepted_range_m,
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        min_cluster_sample_count=min_cluster_sample_count,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
    )
    if resolve_lidar_association is not None:
        association = resolve_lidar_association(association, scan)
    rejection = association.rejection_reason
    if association.associated:
        rejection = measured_head_lidar_rejection(
            association.search_association, registered=True,
            cone_half_angle_rad=cone_half_angle_rad,
            registered_association=association,
        )
    accepted = association.associated and not rejection
    return replace(
        result, accepted=accepted,
        reason="current_head_unique_lidar_cluster" if accepted else rejection,
        fitted_head_bearing_rad=bearing, lidar_association=association,
    )
