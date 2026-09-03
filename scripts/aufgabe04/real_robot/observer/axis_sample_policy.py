"""Axis-sample admission policy for passive real-camera observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from scripts.aufgabe04.perception.stand_axis.models import (
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis_consensus import (
    AxisConditioning,
)


AXIS_SAMPLE_POLICY_VERSION = "real-camera-axis-sample-v2-qr-bound-35deg"
QR_BOUND_MODEL_AXIS_SAMPLE_SOURCE = "model_current_frame_qr_pose_refined"
MAX_QR_BOUND_MODEL_OBLIQUENESS_DEG = 35.0
DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_DEG = (
    MAX_QR_BOUND_MODEL_OBLIQUENESS_DEG
)
MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD = math.radians(
    MAX_QR_BOUND_MODEL_OBLIQUENESS_DEG
)
DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_RAD = (
    MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD
)
_LIMIT_TOLERANCE_RAD = 1.0e-12


@dataclass(frozen=True)
class AxisSampleAdmission:
    accepted: bool
    reason: str
    yaw_rad: float | None
    source: str | None
    conditioning: AxisConditioning
    qr_bound_model_fallback: bool

    def metadata(self) -> dict[str, object]:
        return {
            "policy_version": AXIS_SAMPLE_POLICY_VERSION,
            "accepted": self.accepted,
            "reason": self.reason,
            "yaw_rad": self.yaw_rad,
            "source": self.source,
            "conditioning": asdict(self.conditioning),
            "qr_bound_model_fallback": self.qr_bound_model_fallback,
        }


def admit_axis_sample(
    *,
    estimate: StandAxisImageEstimate,
    debug: StandAxisEdgeDebugArtifacts,
    conditioning: AxisConditioning,
    yaw_rad: float,
    qr_texts: tuple[str, ...],
    lidar_target_associated: bool,
    max_qr_bound_model_obliqueness_rad: float = (
        DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_RAD
    ),
) -> AxisSampleAdmission:
    """Decide whether one measured-model observation may enter consensus.

    The normal silhouette path keeps the configured obliqueness gate.  The
    fallback is deliberately narrower: it only admits a current-frame measured
    model pose jointly fit from QR corners and head borders, with a decoded QR
    text in the same frame, and only inside the bounded 30-35 degree band.
    """

    qr_bound_limit = normalize_qr_bound_model_obliqueness_limit(
        max_qr_bound_model_obliqueness_rad,
        generic_max_obliqueness_rad=conditioning.max_obliqueness_rad,
    )
    if lidar_target_associated is not True:
        return AxisSampleAdmission(
            accepted=False,
            reason="lidar_target_unassociated",
            yaw_rad=None,
            source=None,
            conditioning=conditioning,
            qr_bound_model_fallback=False,
        )
    if conditioning.accepted:
        return AxisSampleAdmission(
            accepted=True,
            reason=conditioning.reason,
            yaw_rad=yaw_rad,
            source=estimate.source,
            conditioning=conditioning,
            qr_bound_model_fallback=False,
        )
    if _qr_bound_model_axis_is_admissible(
        estimate=estimate,
        debug=debug,
        conditioning=conditioning,
        qr_texts=qr_texts,
        max_qr_bound_model_obliqueness_rad=qr_bound_limit,
    ):
        return AxisSampleAdmission(
            accepted=True,
            reason="qr_bound_model_axis_oblique_recovery",
            yaw_rad=yaw_rad,
            source=QR_BOUND_MODEL_AXIS_SAMPLE_SOURCE,
            conditioning=conditioning,
            qr_bound_model_fallback=True,
        )
    return AxisSampleAdmission(
        accepted=False,
        reason=conditioning.reason,
        yaw_rad=None,
        source=None,
        conditioning=conditioning,
        qr_bound_model_fallback=False,
    )


def _qr_bound_model_axis_is_admissible(
    *,
    estimate: StandAxisImageEstimate,
    debug: StandAxisEdgeDebugArtifacts,
    conditioning: AxisConditioning,
    qr_texts: tuple[str, ...],
    max_qr_bound_model_obliqueness_rad: float,
) -> bool:
    if (
        conditioning.accepted
        or conditioning.reason != "oblique_silhouette"
        or conditioning.obliqueness_rad > max_qr_bound_model_obliqueness_rad
    ):
        return False
    if (
        not estimate.usable
        or estimate.source != "model_current_frame_refined"
        or estimate.evidence_state != "fresh_refined"
        or estimate.model_measurement_status != "measured"
        or not estimate.model_profile_sha256
        or debug.evidence_state != "fresh_refined"
        or debug.model_pose_fit_source != "joint_qr_head"
        or debug.model_measurement_status != "measured"
        or debug.model_profile_sha256 != estimate.model_profile_sha256
        or not debug.qr_detected
        or not any(
            isinstance(text, str) and bool(text.strip())
            for text in qr_texts
        )
    ):
        return False
    if estimate.pose_reprojection_rmse_px is None:
        return False
    if (
        not math.isfinite(estimate.pose_reprojection_rmse_px)
        or estimate.pose_reprojection_rmse_px < 0.0
    ):
        return False
    if (
        estimate.pose_ambiguity_gap_px is None
        or not math.isfinite(estimate.pose_ambiguity_gap_px)
        or estimate.pose_ambiguity_gap_px < 0.0
    ):
        return False
    return True


def normalize_qr_bound_model_obliqueness_limit(
    value: float,
    *,
    generic_max_obliqueness_rad: float,
) -> float:
    """Validate and normalize the narrow QR-bound extension limit."""

    try:
        limit = float(value)
        generic_limit = float(generic_max_obliqueness_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("axis obliqueness limits must be finite numbers") from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            "max_qr_bound_model_obliqueness_rad must be finite and positive"
        )
    if not math.isfinite(generic_limit) or generic_limit <= 0.0:
        raise ValueError(
            "generic_max_obliqueness_rad must be finite and positive"
        )
    if limit > MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD + _LIMIT_TOLERANCE_RAD:
        raise ValueError(
            "QR-bound model obliqueness cannot exceed "
            f"{MAX_QR_BOUND_MODEL_OBLIQUENESS_DEG:g} degrees"
        )
    normalized = min(limit, MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD)
    if normalized + _LIMIT_TOLERANCE_RAD < generic_limit:
        raise ValueError(
            "QR-bound model obliqueness cannot be below the generic axis "
            "conditioning limit"
        )
    return normalized


__all__ = [
    "AXIS_SAMPLE_POLICY_VERSION",
    "AxisSampleAdmission",
    "DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_DEG",
    "DEFAULT_QR_BOUND_MODEL_MAX_OBLIQUENESS_RAD",
    "MAX_QR_BOUND_MODEL_OBLIQUENESS_DEG",
    "MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD",
    "QR_BOUND_MODEL_AXIS_SAMPLE_SOURCE",
    "admit_axis_sample",
    "normalize_qr_bound_model_obliqueness_limit",
]
