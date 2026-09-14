"""Current measured-head observability, independent of QR and view-side labels.

The covariance is a local pixel-noise model, not calibrated physical accuracy.
Its explicit noise floor and limits need hardware validation. It supplements
raw structural evidence and planar ambiguity checks, never replaces them.
"""

from dataclasses import dataclass, replace
import math

MEASURED_HEAD_AXIS_SOURCE = "model_current_measured_head"
MIN_HEAD_EDGE_PX = 24.0
MAX_HEAD_REPROJECTION_RMSE_PX = 2.0
MIN_HEAD_CORNER_SIGMA_PX = 0.75
MAX_HEAD_YAW_STD_DEG = 3.0
MAX_SCALED_JACOBIAN_CONDITION = 1.0e6


@dataclass(frozen=True)
class HeadModelQuality:
    accepted: bool
    reason: str
    raw_border_support_mean: float | None
    raw_corner_support_accepted: bool
    centered_neck_supported: bool  # Compatibility diagnostic; never an admission gate.
    minimum_edge_length_px: float | None
    reprojection_rmse_px: float | None
    ambiguity_gap_px: float | None
    axis_ambiguous: bool
    all_corners_positive_depth: bool
    yaw_std_deg: float | None
    max_yaw_std_deg: float
    corner_sigma_px: float | None
    jacobian_condition_number: float | None
    head_size_m: tuple[float, float]
    profile_sha256: str
    pose_model: str = "measured_head_only"
    face_semantics: str = "undirected_plane"
    uncertainty_policy: str = "local_pixel_noise_model_requires_hardware_validation"
    neck_junction_verified: bool = False  # Compatibility diagnostic only.
    outer_border_verified: bool = False


def validated_head_model_quality(quality) -> bool:
    """Shared, fixed admission contract for the distinct head-only channel."""

    if not isinstance(quality, HeadModelQuality):
        return False
    finite = (
        quality.raw_border_support_mean, quality.minimum_edge_length_px,
        quality.reprojection_rmse_px, quality.yaw_std_deg,
        quality.corner_sigma_px, quality.jacobian_condition_number,
    )
    if any(value is None or not math.isfinite(value) for value in finite):
        return False
    return bool(
        quality.accepted and quality.reason == "current_measured_head_observable"
        and quality.pose_model == "measured_head_only"
        and quality.face_semantics == "undirected_plane"
        and quality.raw_corner_support_accepted and quality.outer_border_verified
        and quality.all_corners_positive_depth and not quality.axis_ambiguous
        and 0.60 <= quality.raw_border_support_mean <= 1.0
        and quality.minimum_edge_length_px >= MIN_HEAD_EDGE_PX
        and 0.0 <= quality.reprojection_rmse_px <= MAX_HEAD_REPROJECTION_RMSE_PX
        and 0.0 <= quality.yaw_std_deg <= MAX_HEAD_YAW_STD_DEG
        and quality.max_yaw_std_deg == MAX_HEAD_YAW_STD_DEG
        and quality.corner_sigma_px >= MIN_HEAD_CORNER_SIGMA_PX
        and 1.0 <= quality.jacobian_condition_number <= MAX_SCALED_JACOBIAN_CONDITION
        and (quality.ambiguity_gap_px is None or
             math.isfinite(quality.ambiguity_gap_px) and quality.ambiguity_gap_px >= 0.0)
    )


def _local_yaw_uncertainty(cv2, *, profile, camera, pose, sigma):
    import numpy

    points = numpy.asarray([(p.x_m, p.y_m, p.z_m) for p in profile.head_corners])
    rotation = numpy.asarray(pose.rotation_vector, dtype=numpy.float64)
    translation = numpy.asarray(pose.translation_xyz_m, dtype=numpy.float64)
    matrix = numpy.asarray(((camera.fx_px, 0.0, camera.cx_px),
                            (0.0, camera.fy_px, camera.cy_px), (0.0, 0.0, 1.0)))
    rotation_matrix = cv2.Rodrigues(rotation)[0]
    depths = (points @ rotation_matrix.T + translation)[:, 2]
    if not numpy.isfinite(depths).all() or not numpy.all(depths > 1.0e-6):
        return None, None, False
    if abs(float(rotation_matrix[2, 2])) < 1.0e-6:
        return None, None, True
    _projected, jacobian = cv2.projectPoints(points, rotation, translation, matrix, numpy.zeros(4))
    # Scale translation parameters by measured head size, so rank/conditioning
    # does not compare radians with arbitrary meter-valued column magnitudes.
    scale = numpy.diag((1.0, 1.0, 1.0, profile.head_width_m,
                        profile.head_height_m, max(profile.head_width_m, profile.head_height_m)))
    design = numpy.asarray(jacobian[:, :6], dtype=numpy.float64) @ scale
    if design.shape != (8, 6) or not numpy.isfinite(design).all():
        return None, None, True
    _u, singular, vt = numpy.linalg.svd(design, full_matrices=False)
    if singular[-1] <= 0.0:
        return None, None, True
    condition = float(singular[0] / singular[-1])
    if not math.isfinite(condition) or condition > MAX_SCALED_JACOBIAN_CONDITION:
        return None, condition, True
    covariance = (vt.T * (sigma / singular) ** 2) @ vt

    def yaw(rvec):
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        return -math.atan2(float(normal[0]), abs(float(normal[2])))

    gradient = numpy.zeros(6)
    for index in range(3):
        delta = numpy.zeros(3)
        delta[index] = 1.0e-6
        difference = (yaw(rotation + delta) - yaw(rotation - delta) + math.pi / 2.0) % math.pi - math.pi / 2.0
        gradient[index] = difference / 2.0e-6
    variance = float(gradient @ covariance @ gradient)
    return (None if not math.isfinite(variance) or variance < 0.0 else
            math.degrees(math.sqrt(variance))), condition, True


def evaluate_head_model_quality(
    cv2, *, profile, camera, corners, pose_result,
    raw_border_support_mean, raw_corner_support_accepted, centered_neck_supported,
    neck_junction_verified=False,
    outer_border_verified=False,
) -> HeadModelQuality:
    """Evaluate one current fit; no QR geometry or historical pose is accepted."""

    camera.validate()
    minimum_edge = None if corners is None else min(
        math.hypot(a.u_px - b.u_px, a.v_px - b.v_px)
        for a, b in zip(corners, corners[1:] + corners[:1])
    )
    pose = None if pose_result is None else pose_result.best
    residual = None if pose is None else pose.reprojection_rmse_px
    sigma = None if residual is None else max(MIN_HEAD_CORNER_SIGMA_PX, residual)
    ambiguous = bool(pose_result is not None and pose_result.axis_ambiguous(
        max_residual_gap_px=MIN_HEAD_CORNER_SIGMA_PX if sigma is None else sigma,
    ))
    std, condition, depths = None, None, False
    reason = "current_measured_head_observable"
    if not profile.committable or profile.environment != "physical":
        reason = "head_model_measured_physical_profile_required"
    elif not raw_corner_support_accepted or raw_border_support_mean is None or raw_border_support_mean < 0.60:
        reason = "head_model_raw_border_evidence_insufficient"
    elif not outer_border_verified:
        reason = "head_model_outer_border_unverified"
    elif minimum_edge is None or not math.isfinite(minimum_edge) or minimum_edge < MIN_HEAD_EDGE_PX:
        reason = "head_model_pixel_span_insufficient"
    elif pose is None or residual > MAX_HEAD_REPROJECTION_RMSE_PX:
        reason = "head_model_pose_rejected"
    elif ambiguous:
        reason = "head_model_planar_axis_ambiguous"
    else:
        try:
            std, condition, depths = _local_yaw_uncertainty(
                cv2, profile=profile, camera=camera, pose=pose, sigma=sigma,
            )
        except (ArithmeticError, ValueError, TypeError):
            pass
        if not depths:
            reason = "head_model_corner_depth_invalid"
        elif std is None or condition is None:
            reason = "head_model_uncertainty_unavailable"
        elif std > MAX_HEAD_YAW_STD_DEG:
            reason = "head_model_yaw_uncertainty_too_high"
    quality = HeadModelQuality(
        reason == "current_measured_head_observable", reason,
        raw_border_support_mean, bool(raw_corner_support_accepted), bool(centered_neck_supported),
        minimum_edge, residual, None if pose_result is None else pose_result.ambiguity_gap_px,
        ambiguous, depths, std, MAX_HEAD_YAW_STD_DEG, sigma, condition,
        (profile.head_width_m, profile.head_height_m), profile.sha256,
        neck_junction_verified=bool(neck_junction_verified),
        outer_border_verified=bool(outer_border_verified),
    )
    if quality.accepted and not validated_head_model_quality(quality):
        return replace(quality, accepted=False, reason="head_model_quality_values_invalid")
    return quality
