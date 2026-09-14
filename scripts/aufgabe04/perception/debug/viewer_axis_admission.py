"""Apply measured-head quality to diagnostic camera consensus and handoff."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.artifacts.backside_axis_observation import BACKSIDE_AXIS_SAMPLE_SOURCE
from scripts.aufgabe04.perception.stand_axis.head_model_admission import (
    admit_measured_head_model, requires_measured_head_admission,
)
from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    MEASURED_HEAD_AXIS_SOURCE,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning


_HEAD_SOURCES = frozenset({MEASURED_HEAD_AXIS_SOURCE, BACKSIDE_AXIS_SAMPLE_SOURCE})


def _uses_measured_head_policy(estimate, artifacts) -> bool:
    return (estimate.source in _HEAD_SOURCES
            or (artifacts is not None and requires_measured_head_admission(estimate, artifacts)))


def current_head_quality_ready(estimate, artifacts) -> bool:
    """Do not let a failed new-source quality check enter temporal consensus."""
    if not _uses_measured_head_policy(estimate, artifacts):
        return True
    if (artifacts is None or type(estimate.yaw_deg) not in (int, float)
            or not math.isfinite(estimate.yaw_deg)):
        return False
    return admit_measured_head_model(
        estimate=estimate, debug=artifacts, yaw_rad=math.radians(estimate.yaw_deg),
    ).accepted


def current_axis_evidence_ready(estimate, artifacts) -> bool:
    """Only current geometry, including a proved backside head, is measurable."""
    return bool(estimate is not None
                and estimate.evidence_state in {"fresh_refined", "fresh_backside"}
                and current_head_quality_ready(estimate, artifacts)
                and (estimate.evidence_state != "fresh_backside"
                     or estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE))


def viewer_color_side_allowed(estimate, *, requested: bool) -> bool:
    """A measured plane and a colored surface do not identify its QR side."""
    return bool(requested and estimate.source not in _HEAD_SOURCES)


def viewer_face_export_allowed(estimate, side: str) -> bool:
    """Prevent the legacy observation format from flipping an unresolved plane."""
    if estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE:
        # The viewer has no repeated, registered backside receipt. An async
        # QR label cannot turn this current QR-free plane into a directed face.
        return False
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
    if (consensus.source not in _HEAD_SOURCES
            and not _uses_measured_head_policy(estimate, artifacts)):
        return ViewerAxisAdmission(legacy.accepted, legacy.reason, legacy.obliqueness_rad,
                                   legacy.max_obliqueness_rad, "silhouette_obliqueness")
    valid = (estimate.source == consensus.source
             and math.isfinite(consensus.yaw_rad)
             and current_axis_evidence_ready(estimate, artifacts))
    return ViewerAxisAdmission(
        valid, "measured_head_geometry_quality_accepted" if valid else "measured_head_quality_rejected",
        legacy.obliqueness_rad, None, "current_measured_head_quality",
    )
