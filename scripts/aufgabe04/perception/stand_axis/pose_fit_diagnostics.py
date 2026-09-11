"""Diagnostic-only evidence for semantic head/QR border disagreement.

These measurements never admit a pose. In particular, an independently good
head fit cannot override a rejected joint fit or missing raw corner evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.geometry_contract import JointGeometryContract
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.model_projection import project_stand_model
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
    PlanarPoseResult,
    RectifiedCameraMatrix,
    estimate_planar_pose_ippe,
)

CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class IndependentPoseDiagnostic:
    accepted: bool
    reason: str
    axis_ambiguous: bool
    reprojection_rmse_px: float | None
    yaw_deg: float | None


@dataclass(frozen=True)
class MetricModelDiagnostics:
    observed_qr_corners: tuple[ImagePoint, ...] | None
    # Residuals describe the selected pose, or the best rejected hypothesis;
    # they are not independent fits of each group.
    group_reprojection_rmse_px: dict[str, float]
    corner_reprojection_error_px: dict[str, float]
    max_corner_error_px: float | None
    observed_head_qr_width_ratio: float | None
    observed_head_qr_height_ratio: float | None
    expected_head_qr_width_ratio: float
    expected_head_qr_height_ratio: float
    expected_panel_qr_width_ratio: float | None
    expected_panel_qr_height_ratio: float | None
    qr_only: IndependentPoseDiagnostic | None
    head_only: IndependentPoseDiagnostic | None
    geometry_contract: JointGeometryContract | None = None


def _independent_pose(result: PlanarPoseResult | None) -> IndependentPoseDiagnostic | None:
    if result is None:
        return None
    best = result.hypotheses[0] if result.hypotheses else None
    return IndependentPoseDiagnostic(
        result.accepted,
        result.reason,
        result.axis_ambiguous(),
        None if best is None else best.reprojection_rmse_px,
        None if best is None else best.yaw_deg,
    )


def rectified_head_qr_ratios(
    cv2,
    head_corners: tuple[ImagePoint, ...] | None,
    qr_corners: tuple[ImagePoint, ...] | None,
    profile: StandModelProfile,
) -> tuple[float | None, float | None]:
    """Measure the observed head in the QR plane, without pose-fit bias.

    The homography uses observed QR corners and their metric plane coordinates.
    Ratios are diagnostic only: localization errors or an incorrect QR/head
    pairing also distort them. Degenerate or horizon-crossing maps are omitted.
    """

    import numpy

    if head_corners is None or qr_corners is None:
        return None, None
    if len(head_corners) != 4 or len(qr_corners) != 4:
        return None, None
    head = numpy.asarray([(p.u_px, p.v_px) for p in head_corners], dtype=numpy.float32)
    qr = numpy.asarray([(p.u_px, p.v_px) for p in qr_corners], dtype=numpy.float32)
    if not numpy.isfinite(head).all() or not numpy.isfinite(qr).all():
        return None, None
    for polygon in (head, qr):
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1.0:
            return None, None
    plane = numpy.asarray([(p.x_m, p.y_m) for p in profile.qr_corners], dtype=numpy.float32)
    transform = cv2.getPerspectiveTransform(qr, plane)
    if not numpy.isfinite(transform).all() or numpy.linalg.matrix_rank(transform) < 3:
        return None, None
    homogeneous = numpy.column_stack((head, numpy.ones(4))) @ transform.T
    depths = homogeneous[:, 2]
    if not (numpy.all(depths > 1.0e-9) or numpy.all(depths < -1.0e-9)):
        return None, None
    mapped = homogeneous[:, :2] / depths[:, None]
    lengths = numpy.linalg.norm(numpy.roll(mapped, -1, axis=0) - mapped, axis=1)
    width = float((lengths[0] + lengths[2]) / (2.0 * profile.qr_symbol_width_m))
    height = float((lengths[1] + lengths[3]) / (2.0 * profile.qr_symbol_height_m))
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height)):
        return None, None
    return width, height


def collect_metric_model_diagnostics(
    cv2,
    *,
    profile: StandModelProfile,
    camera: RectifiedCameraMatrix,
    head_corners: tuple[ImagePoint, ...] | None,
    qr_corners: tuple[ImagePoint, ...] | None,
    diagnostic_pose: PlanarPoseHypothesis | None,
    qr_pose: PlanarPoseResult | None,
    head_pose: PlanarPoseResult | None = None,
    diagnose_head_only: bool = False,
    max_reprojection_rmse_px: float = 2.0,
) -> MetricModelDiagnostics:
    """Report disagreement, reusing existing solves whenever possible.

    Only rejected joint fits request the additional head-only diagnostic solve.
    The caller never consumes this diagnostic pose for acceptance or tracking.
    """

    if diagnose_head_only and head_corners is not None:
        head_pose = estimate_planar_pose_ippe(
            cv2, head_corners, profile.head_corners, camera,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
        )
    group_residuals: dict[str, float] = {}
    corner_residuals: dict[str, float] = {}
    if diagnostic_pose is not None:
        projected = project_stand_model(cv2, profile, diagnostic_pose, camera)
        for group, corners in (("head", head_corners), ("qr", qr_corners)):
            if corners is None:
                continue
            residuals = {}
            for suffix, observed in zip(CORNER_NAMES, corners):
                name = f"{group}_{suffix}"
                predicted = projected.landmarks[name]
                residuals[name] = math.hypot(
                    predicted.u_px - observed.u_px, predicted.v_px - observed.v_px
                )
            if residuals and all(math.isfinite(value) for value in residuals.values()):
                corner_residuals.update(residuals)
                group_residuals[group] = math.sqrt(
                    sum(value * value for value in residuals.values()) / len(residuals)
                )
    width_ratio, height_ratio = rectified_head_qr_ratios(
        cv2, head_corners, qr_corners, profile
    )
    return MetricModelDiagnostics(
        observed_qr_corners=qr_corners,
        group_reprojection_rmse_px=group_residuals,
        corner_reprojection_error_px=corner_residuals,
        max_corner_error_px=max(corner_residuals.values(), default=None),
        observed_head_qr_width_ratio=width_ratio,
        observed_head_qr_height_ratio=height_ratio,
        expected_head_qr_width_ratio=profile.head_width_m / profile.qr_symbol_width_m,
        expected_head_qr_height_ratio=profile.head_height_m / profile.qr_symbol_height_m,
        expected_panel_qr_width_ratio=(
            None if profile.qr_panel_width_m is None
            else profile.qr_panel_width_m / profile.qr_symbol_width_m
        ),
        expected_panel_qr_height_ratio=(
            None if profile.qr_panel_height_m is None
            else profile.qr_panel_height_m / profile.qr_symbol_height_m
        ),
        qr_only=_independent_pose(qr_pose),
        head_only=_independent_pose(head_pose),
    )
