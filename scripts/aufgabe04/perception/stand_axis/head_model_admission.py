"""Admit a measured head plane independently of QR geometry and face identity."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE, validated_head_model_quality,
)


@dataclass(frozen=True)
class HeadModelAdmission:
    accepted: bool
    reason: str
    yaw_uncertainty_deg: float | None = None

    def metadata(self) -> dict:
        return {
            "accepted": self.accepted, "reason": self.reason,
            "source": MEASURED_HEAD_AXIS_SOURCE,
            "yaw_uncertainty_deg": self.yaw_uncertainty_deg,
            "uncertainty_basis": "conditional_current_pixel_noise",
            "face_semantics": "undirected_plane",
            "qr_geometry_required": False,
            "front_back_authorized": False, "motion_authorized": False,
        }


def admit_measured_head_model(*, estimate, debug, yaw_rad: float) -> HeadModelAdmission:
    """Require the new source's complete quality contract even below 30 degrees.

    Quality, including local yaw uncertainty and planar ambiguity, replaces
    the legacy silhouette obliqueness limit for this one source. Current head
    bearing association, freshness and stationary consensus remain external.
    """
    if (estimate.source != MEASURED_HEAD_AXIS_SOURCE
            or debug.model_pose_fit_source != MEASURED_HEAD_AXIS_SOURCE
            or estimate.usable is not True
            or estimate.evidence_state != "fresh_refined"
            or debug.evidence_state != "fresh_refined"
            or estimate.model_measurement_status != "measured"
            or debug.model_measurement_status != "measured"
            or not estimate.model_profile_sha256
            or debug.model_profile_sha256 != estimate.model_profile_sha256):
        return HeadModelAdmission(False, "measured_head_provenance_rejected")
    if (type(yaw_rad) not in (int, float) or not math.isfinite(yaw_rad)
            or type(estimate.yaw_deg) not in (int, float) or not math.isfinite(estimate.yaw_deg)
            or abs(math.remainder(yaw_rad - math.radians(estimate.yaw_deg), 2 * math.pi)) > 1e-9
            or estimate.corners is None or len(estimate.corners) != 4):
        return HeadModelAdmission(False, "measured_head_axis_unavailable")
    quality = getattr(debug, "head_model_quality", None)
    if (not validated_head_model_quality(quality)
            or quality.profile_sha256 != estimate.model_profile_sha256):
        return HeadModelAdmission(False, "measured_head_quality_rejected")
    return HeadModelAdmission(True, "measured_head_geometry_quality_accepted", quality.yaw_std_deg)

