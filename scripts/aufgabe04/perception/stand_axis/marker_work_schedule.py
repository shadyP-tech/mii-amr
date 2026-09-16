"""Schedule optional current-image marker checks independently of head fitting.

No decision here changes geometric admission. A skipped check is unknown side,
never evidence that a head is the backside. Native OpenCV calls are atomic, so
their budget is cooperative and completion is checked against the same deadline.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_outer_border import current_head_boundary_eligible


MIN_NATIVE_MARKER_BUDGET_SEC = 0.070


@dataclass(frozen=True)
class MarkerWorkDecision:
    action: str
    reason: str


def current_head_available_for_markers(head_result):
    """Complete current borders can support side evidence without an angle.

    An uncertain planar orientation stays unadmitted. It must not suppress the
    separate appearance/marker channel when the physical border is verified.
    """
    if head_result is None:
        return False
    estimate, debug, _ = head_result
    quality = debug.head_model_quality
    return bool(current_head_boundary_eligible(estimate, debug)
                and quality is not None
                and quality.raw_corner_support_accepted is True
                and quality.outer_border_verified is True)


def schedule_marker_work(*, policy, positive_observations, physical_head,
                         head_available, now_monotonic_sec, deadline_monotonic_sec):
    if policy not in {"auto", "disabled", "supplied_only"}:
        raise ValueError("QR marker policy must be auto, disabled, or supplied_only")
    if not math.isfinite(now_monotonic_sec):
        raise ValueError("QR marker clock must be finite")
    if deadline_monotonic_sec is not None and not math.isfinite(deadline_monotonic_sec):
        raise ValueError("QR marker deadline must be finite")
    # Positive same-image decoder evidence needs no native acquisition. Its
    # identity/corner binding is still checked independently by the consumer.
    if positive_observations:
        return MarkerWorkDecision("supplied", "supplied_current_qr_observations")
    if policy != "auto":
        return MarkerWorkDecision("skip", "qr_marker_checks_disabled" if policy == "disabled"
                                  else "qr_marker_supplied_evidence_unavailable")
    if physical_head and not head_available:
        return MarkerWorkDecision("skip", "qr_marker_head_geometry_unavailable")
    if (deadline_monotonic_sec is not None
            and deadline_monotonic_sec - now_monotonic_sec < MIN_NATIVE_MARKER_BUDGET_SEC):
        return MarkerWorkDecision("skip", "qr_marker_processing_budget_exhausted")
    return MarkerWorkDecision("native", "current_native_marker_check")
