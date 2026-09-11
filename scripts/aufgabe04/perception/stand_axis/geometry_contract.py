"""Classify incompatible current head/QR geometry without fitting a new pose.

Independent fits can explain a joint rejection, but cannot replace it. In
particular, this module never changes a profile dimension, chooses a semantic
border by its joint residual, or promotes a diagnostic angle into a pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
    from scripts.aufgabe04.perception.stand_axis.pose_fit_diagnostics import MetricModelDiagnostics


HEAD_QR_GEOMETRY_MISMATCH = "model_head_qr_geometry_mismatch"


@dataclass(frozen=True)
class JointGeometryContract:
    reason: str
    underlying_joint_reason: str
    joint_reprojection_rmse_px: float
    maximum_reprojection_rmse_px: float
    independent_head_rmse_px: float
    independent_qr_rmse_px: float
    diagnostic_head_yaw_deg: float
    diagnostic_qr_yaw_deg: float
    diagnostic_axis_separation_deg: float
    observed_head_qr_ratio: tuple[float, float]
    expected_head_qr_ratio: tuple[float, float]
    expected_head_panel_ratio: tuple[float | None, float | None]
    head_size_m: tuple[float, float]
    qr_symbol_size_m: tuple[float, float]
    qr_panel_size_m: tuple[float | None, float | None]
    profile_id: str
    profile_sha256: str
    profile_measurement_status: str
    # Preserve the descriptor verbatim. A profile marked "measured" can still
    # declare an image-estimated symbol size alongside measured head/paper.
    # Free text must not be reinterpreted into new metrology or acceptance.
    profile_dimension_source: str
    motion_authorized: bool = False
    measurement_authorized: bool = False
    profile_changed: bool = False


def classify_joint_geometry_contract(
    *,
    profile: StandModelProfile,
    diagnostics: MetricModelDiagnostics,
    joint_reason: str,
    joint_reprojection_rmse_px: float | None,
    max_reprojection_rmse_px: float,
    qr_marker_verified: bool,
) -> JointGeometryContract | None:
    """Recognize good separate fits that cannot satisfy one measured model.

    All inputs are existing solves of the same current image. Missing,
    ambiguous or inaccurate independent geometry retains the ordinary failure
    reason. No additional OpenCV work, fit, or candidate search is performed.
    """

    head, qr = diagnostics.head_only, diagnostics.qr_only
    if (
        joint_reason != "reprojection_error_too_high"
        or qr_marker_verified is not True
        or not profile.committable
        or head is None or qr is None
        or not head.accepted or not qr.accepted
        or head.axis_ambiguous or qr.axis_ambiguous
    ):
        return None
    values = (
        joint_reprojection_rmse_px, max_reprojection_rmse_px,
        head.reprojection_rmse_px, qr.reprojection_rmse_px,
        head.yaw_deg, qr.yaw_deg,
        diagnostics.observed_head_qr_width_ratio,
        diagnostics.observed_head_qr_height_ratio,
    )
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    if (
        max_reprojection_rmse_px <= 0.0
        or joint_reprojection_rmse_px <= max_reprojection_rmse_px
        or not 0.0 <= head.reprojection_rmse_px <= max_reprojection_rmse_px
        or not 0.0 <= qr.reprojection_rmse_px <= max_reprojection_rmse_px
        or diagnostics.observed_head_qr_width_ratio <= 0.0
        or diagnostics.observed_head_qr_height_ratio <= 0.0
    ):
        return None
    return JointGeometryContract(
        reason=HEAD_QR_GEOMETRY_MISMATCH,
        underlying_joint_reason=joint_reason,
        joint_reprojection_rmse_px=joint_reprojection_rmse_px,
        maximum_reprojection_rmse_px=max_reprojection_rmse_px,
        independent_head_rmse_px=head.reprojection_rmse_px,
        independent_qr_rmse_px=qr.reprojection_rmse_px,
        diagnostic_head_yaw_deg=head.yaw_deg,
        diagnostic_qr_yaw_deg=qr.yaw_deg,
        diagnostic_axis_separation_deg=abs((head.yaw_deg - qr.yaw_deg + 90.0) % 180.0 - 90.0),
        observed_head_qr_ratio=(diagnostics.observed_head_qr_width_ratio, diagnostics.observed_head_qr_height_ratio),
        expected_head_qr_ratio=(profile.head_width_m / profile.qr_symbol_width_m, profile.head_height_m / profile.qr_symbol_height_m),
        expected_head_panel_ratio=(
            None if profile.qr_panel_width_m is None else profile.head_width_m / profile.qr_panel_width_m,
            None if profile.qr_panel_height_m is None else profile.head_height_m / profile.qr_panel_height_m,
        ),
        head_size_m=(profile.head_width_m, profile.head_height_m),
        qr_symbol_size_m=(profile.qr_symbol_width_m, profile.qr_symbol_height_m),
        qr_panel_size_m=(profile.qr_panel_width_m, profile.qr_panel_height_m),
        profile_id=profile.profile_id, profile_sha256=profile.sha256,
        profile_measurement_status=profile.measurement_status,
        profile_dimension_source=profile.source,
    )
