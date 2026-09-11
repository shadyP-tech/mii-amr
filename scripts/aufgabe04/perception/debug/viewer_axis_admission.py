"""Apply measured-head quality to diagnostic camera consensus and handoff."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_model_admission import (
    admit_measured_head_model,
)
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning


def current_head_quality_ready(estimate, artifacts) -> bool:
    """Do not let a failed new-source quality check enter temporal consensus."""
    if estimate.source != MEASURED_HEAD_AXIS_SOURCE:
        return True
    if artifacts is None or estimate.yaw_deg is None:
        return False
    return admit_measured_head_model(
        estimate=estimate, debug=artifacts, yaw_rad=math.radians(estimate.yaw_deg),
    ).accepted


def viewer_color_side_allowed(estimate, *, requested: bool) -> bool:
    """A measured plane and a colored surface do not identify its QR side."""
    return bool(requested and estimate.source != MEASURED_HEAD_AXIS_SOURCE)


def viewer_face_export_allowed(estimate, side: str) -> bool:
    """Prevent the legacy observation format from flipping an unresolved plane."""
    return (estimate.source != MEASURED_HEAD_AXIS_SOURCE or side == "qr_code_side")


@dataclass(frozen=True)
class ViewerAxisAdmission:
    accepted: bool
    reason: str
    obliqueness_rad: float
    max_obliqueness_rad: float | None
    policy: str


def viewer_axis_admission(*, consensus, estimate, artifacts, max_obliqueness_rad):
    if consensus is None:
        return None
    legacy = axis_conditioning(consensus.yaw_rad, max_obliqueness_rad=max_obliqueness_rad)
    if consensus.source != MEASURED_HEAD_AXIS_SOURCE:
        return ViewerAxisAdmission(legacy.accepted, legacy.reason, legacy.obliqueness_rad,
                                   legacy.max_obliqueness_rad, "silhouette_obliqueness")
    valid = (estimate.source == consensus.source
             and math.isfinite(consensus.yaw_rad)
             and current_head_quality_ready(estimate, artifacts))
    return ViewerAxisAdmission(
        valid, "measured_head_geometry_quality_accepted" if valid else "measured_head_quality_rejected",
        legacy.obliqueness_rad, None, "current_measured_head_quality",
    )
