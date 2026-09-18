"""Current measured image borders and metric yaw are separate observations."""

import math

from scripts.aufgabe04.perception.stand_axis.head_outer_border import current_head_boundary_eligible
from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model


def head_frame_detection(estimate, artifacts):
    quality = None if artifacts is None else artifacts.head_model_quality
    detected = bool(estimate is not None and artifacts is not None
        and current_head_boundary_eligible(estimate, artifacts)
        and quality is not None and quality.raw_corner_support_accepted
        and quality.outer_border_verified)
    yaw = getattr(estimate, "yaw_deg", None)
    reliable = bool(detected and type(yaw) in (int, float) and math.isfinite(yaw)
        and admit_measured_head_model(estimate=estimate, debug=artifacts,
                                     yaw_rad=math.radians(yaw)).accepted)
    return {"head_frame_detected": detected, "yaw_reliable": reliable,
            "reason": None if estimate is None else estimate.reason,
            "motion_authorized": False}
