"""Current-pixel head detection with an explicit, ambiguity-preserving axis bound.

This proof does not grant a unique angle or motion permission. The interval
encloses every plausible planar pose, expanded by an engineering pixel-noise
allowance. It is not a calibrated confidence interval. Downstream planning
must validate the entire interval, with fresh association and repeated images.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE,
    MAX_HEAD_REPROJECTION_RMSE_PX, MAX_SCALED_JACOBIAN_CONDITION,
    MIN_HEAD_CORNER_SIGMA_PX, MIN_HEAD_EDGE_PX, _local_yaw_uncertainty,
)
from scripts.aufgabe04.perception.stand_axis.head_outer_border import current_head_boundary_eligible
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint

NOISE_ALLOWANCE_MULTIPLIER = 3.0
BOUNDS_REASON = "current_head_orientation_bounded"
UNCERTAINTY_POLICY = "three_sigma_local_pixel_noise_engineering_allowance_not_calibrated"


@dataclass(frozen=True)
class HeadOrientationHypothesis:
    yaw_rad: float
    yaw_std_rad: float
    reprojection_rmse_px: float
    corner_sigma_px: float
    jacobian_condition_number: float
    all_corners_positive_depth: bool
    rotation_vector: tuple[float, float, float]
    translation_xyz_m: tuple[float, float, float]
    face_normal_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class CurrentHeadOrientationBounds:
    accepted: bool
    reason: str
    center_rad: float | None
    half_width_rad: float | None
    hypotheses: tuple[HeadOrientationHypothesis, ...]
    corners: tuple[ImagePoint, ...] | None
    frame_shape: tuple[int, int]
    camera_matrix: tuple[float, float, float, float]
    profile_sha256: str
    head_size_m: tuple[float, float]
    measured_physical_profile: bool
    raw_border_support_mean: float | None
    raw_corner_support_accepted: bool
    outer_border_verified: bool
    minimum_edge_length_px: float | None
    binding_sha256: str = ""
    noise_allowance_multiplier: float = NOISE_ALLOWANCE_MULTIPLIER
    uncertainty_policy: str = UNCERTAINTY_POLICY


def _finite(*values):
    try:
        return all(not isinstance(value, bool) and math.isfinite(value) for value in values)
    except (TypeError, ValueError):
        return False


def _complete_border(corners, shape):
    # A rail coinciding with the crop boundary cannot prove the whole head is
    # visible. Require one current pixel outside every extreme border.
    return bool(corners is not None and all(
        1.0 <= p.u_px <= shape[1] - 2.0 and 1.0 <= p.v_px <= shape[0] - 2.0
        for p in corners
    ))


def enclosing_axial_interval(hypotheses):
    """Smallest connected modulo-pi hull of all expanded hypotheses.

    Splitting and merging arcs before selecting a complementary gap avoids
    incorrectly narrowing intervals whose noise allowance crosses the wrap.
    A full-circle union has no useful bounded undirected orientation.
    """

    intervals = []
    for hypothesis in hypotheses:
        center = hypothesis.yaw_rad
        radius = NOISE_ALLOWANCE_MULTIPLIER * hypothesis.yaw_std_rad
        if not _finite(center, radius) or radius < 0.0 or radius >= math.pi / 2.0:
            return None
        start = (center - radius) % math.pi
        end = start + 2.0 * radius
        if end <= math.pi:
            intervals.append((start, end))
        else:
            intervals.extend(((start, math.pi), (0.0, end - math.pi)))
    if not intervals:
        return None
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    gaps = [(following[0] - current[1], following[0])
            for current, following in zip(merged, merged[1:])]
    gaps.append((merged[0][0] + math.pi - merged[-1][1], merged[0][0]))
    gap, start = max(gaps)
    if gap <= 1.0e-12:
        return None
    width = math.pi - gap
    return ((start + width / 2.0 + math.pi / 2.0) % math.pi - math.pi / 2.0,
            width / 2.0)


def _binding(bounds):
    payload = asdict(bounds)
    payload.pop("binding_sha256")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validated_current_head_orientation_bounds(
    bounds, *, estimate=None, debug=None, profile_sha256=None,
):
    """Validate an independent detection proof, optionally bound to its result.

    Freshness and candidate association belong to the observer. This function
    prevents an unrelated estimate, crop, profile, or altered interval from
    borrowing the proof; strict angle acceptance is deliberately unaffected.
    """

    if not isinstance(bounds, CurrentHeadOrientationBounds):
        return False
    try:
        valid = (
            bounds.accepted and bounds.reason == BOUNDS_REASON
            and bounds.measured_physical_profile
            and len(bounds.profile_sha256) == 64
            and all(c in "0123456789abcdef" for c in bounds.profile_sha256)
            and bounds.raw_corner_support_accepted and bounds.outer_border_verified
            and _finite(bounds.center_rad, bounds.half_width_rad,
                        bounds.raw_border_support_mean, bounds.minimum_edge_length_px,
                        *bounds.head_size_m, *bounds.camera_matrix)
            and 0.60 <= bounds.raw_border_support_mean <= 1.0
            and bounds.minimum_edge_length_px >= MIN_HEAD_EDGE_PX
            and len(bounds.head_size_m) == 2 and min(bounds.head_size_m) > 0.0
            and len(bounds.camera_matrix) == 4 and min(bounds.camera_matrix[:2]) > 0.0
            and -math.pi / 2.0 <= bounds.center_rad < math.pi / 2.0
            and 0.0 <= bounds.half_width_rad < math.pi / 2.0
            and bounds.noise_allowance_multiplier == NOISE_ALLOWANCE_MULTIPLIER
            and bounds.uncertainty_policy == UNCERTAINTY_POLICY
            and bool(bounds.hypotheses)
            and validate_current_head_proposal(bounds.corners, frame_shape=bounds.frame_shape) == bounds.corners
            and _complete_border(bounds.corners, bounds.frame_shape)
        )
        if not valid:
            return False
        best_residual = min(p.reprojection_rmse_px for p in bounds.hypotheses)
        if not _finite(best_residual) or not 0.0 <= best_residual <= MAX_HEAD_REPROJECTION_RMSE_PX:
            return False
        plausible_limit = best_residual + max(MIN_HEAD_CORNER_SIGMA_PX, best_residual)
        for pose in bounds.hypotheses:
            if not isinstance(pose, HeadOrientationHypothesis) or not (
                _finite(pose.yaw_rad, pose.yaw_std_rad, pose.reprojection_rmse_px,
                        pose.corner_sigma_px, pose.jacobian_condition_number,
                        *pose.rotation_vector, *pose.translation_xyz_m, *pose.face_normal_xyz)
                and pose.all_corners_positive_depth and pose.translation_xyz_m[2] > 0.0
                and 0.0 <= pose.yaw_std_rad < math.pi / 6.0
                and 0.0 <= pose.reprojection_rmse_px <= plausible_limit
                and pose.corner_sigma_px == max(MIN_HEAD_CORNER_SIGMA_PX, pose.reprojection_rmse_px)
                and 1.0 <= pose.jacobian_condition_number <= MAX_SCALED_JACOBIAN_CONDITION
                and len(pose.rotation_vector) == len(pose.translation_xyz_m) == len(pose.face_normal_xyz) == 3
            ):
                return False
        interval = enclosing_axial_interval(bounds.hypotheses)
        if interval is None or any(abs(a - b) > 1.0e-12 for a, b in zip(
                interval, (bounds.center_rad, bounds.half_width_rad))):
            return False
        if bounds.binding_sha256 != _binding(bounds):
            return False
        if profile_sha256 is not None and bounds.profile_sha256 != profile_sha256:
            return False
        if estimate is not None and (
            estimate.corners != bounds.corners or estimate.model_profile_sha256 != bounds.profile_sha256
            or estimate.model_measurement_status != "measured"
            or estimate.source not in {MEASURED_HEAD_AXIS_SOURCE, "model_backside_current_frame"}
            or estimate.evidence_state not in {"unobservable", "fresh_refined", "fresh_backside"}
        ):
            return False
        if debug is not None and (
            debug.refined_corners != bounds.corners or debug.model_profile_sha256 != bounds.profile_sha256
            or debug.model_measurement_status != "measured" or debug.head_orientation_bounds != bounds
            or debug.model_pose_fit_source != MEASURED_HEAD_AXIS_SOURCE
            or debug.evidence_state not in {"unobservable", "fresh_refined", "fresh_backside"}
        ):
            return False
        if estimate is not None and debug is not None and not current_head_boundary_eligible(estimate, debug):
            return False
        return True
    except (AttributeError, TypeError, ValueError, IndexError):
        return False


def evaluate_current_head_orientation_bounds(
    cv2, *, profile, camera, corners, pose_result, raw_border_support_mean,
    raw_corner_support_accepted, outer_border_verified, frame_shape,
):
    """Retain every plausible pose; one ill-conditioned alternative rejects all."""

    camera.validate()
    try:
        current = validate_current_head_proposal(corners, frame_shape=frame_shape)
    except ValueError:
        current = None
    minimum_edge = None if current is None else min(
        math.hypot(a.u_px - b.u_px, a.v_px - b.v_px)
        for a, b in zip(current, current[1:] + current[:1])
    )
    result = CurrentHeadOrientationBounds(
        False, "head_orientation_structure_unverified", None, None, (), current,
        tuple(frame_shape[:2]), (camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px),
        profile.sha256, (profile.head_width_m, profile.head_height_m),
        bool(profile.committable and profile.environment == "physical"),
        raw_border_support_mean, bool(raw_corner_support_accepted), bool(outer_border_verified), minimum_edge,
    )
    if not (result.measured_physical_profile and current is not None
            and _complete_border(current, frame_shape)
            and raw_corner_support_accepted and outer_border_verified
            and _finite(raw_border_support_mean, minimum_edge, *result.head_size_m)
            and 0.60 <= raw_border_support_mean <= 1.0 and minimum_edge >= MIN_HEAD_EDGE_PX
            and min(result.head_size_m) > 0.0):
        return result
    if pose_result is None or not pose_result.accepted or not pose_result.hypotheses:
        return replace(result, reason="head_orientation_pose_unavailable")
    poses = pose_result.hypotheses
    if any(not _finite(p.reprojection_rmse_px, p.yaw_deg) or p.reprojection_rmse_px < 0.0
           for p in poses):
        return replace(result, reason="head_orientation_pose_values_invalid")
    best_residual = min(p.reprojection_rmse_px for p in poses if p.positive_depth) if any(
        p.positive_depth for p in poses) else math.inf
    if best_residual > MAX_HEAD_REPROJECTION_RMSE_PX:
        return replace(result, reason="head_orientation_pose_rejected")
    limit = best_residual + max(MIN_HEAD_CORNER_SIGMA_PX, best_residual)
    hypotheses = []
    for pose in poses:
        if not pose.positive_depth or pose.reprojection_rmse_px > limit:
            continue
        if not (len(pose.rotation_vector) == len(pose.translation_xyz_m) == len(pose.face_normal_xyz) == 3
                and _finite(*pose.rotation_vector, *pose.translation_xyz_m, *pose.face_normal_xyz)):
            return replace(result, reason="head_orientation_pose_values_invalid")
        sigma = max(MIN_HEAD_CORNER_SIGMA_PX, pose.reprojection_rmse_px)
        try:
            std, condition, depths = _local_yaw_uncertainty(
                cv2, profile=profile, camera=camera, pose=pose, sigma=sigma,
            )
        except (ArithmeticError, ValueError, TypeError):
            std, condition, depths = None, None, False
        if not depths or not _finite(std, condition) or std < 0.0 or not (
                1.0 <= condition <= MAX_SCALED_JACOBIAN_CONDITION):
            return replace(result, reason="head_orientation_alternative_unbounded")
        hypotheses.append(HeadOrientationHypothesis(
            math.radians(pose.yaw_deg), math.radians(std), pose.reprojection_rmse_px,
            sigma, condition, depths, pose.rotation_vector, pose.translation_xyz_m, pose.face_normal_xyz,
        ))
    hypotheses = tuple(sorted(hypotheses, key=lambda p: p.reprojection_rmse_px))
    interval = enclosing_axial_interval(hypotheses)
    result = replace(result, hypotheses=hypotheses)
    if interval is None:
        return replace(result, reason="head_orientation_interval_unbounded")
    result = replace(result, accepted=True, reason=BOUNDS_REASON,
                     center_rad=interval[0], half_width_rad=interval[1])
    result = replace(result, binding_sha256=_binding(result))
    if not validated_current_head_orientation_bounds(result):
        return replace(result, accepted=False, reason="head_orientation_values_invalid")
    return result
