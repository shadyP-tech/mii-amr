"""Current-frame acquisition scheduling, without pose or identity authority.

Cold candidate search locates complete current borders once, before metric
fitting. A failed locator never falls through to another expensive crop on the
same image. The next image can reacquire; previous angles are never substituted.
"""

from dataclasses import replace
import math
import time

from scripts.aufgabe04.perception.stand_axis.geometry import _unusable
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MEASURED_HEAD_AXIS_SOURCE
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    CameraTargetRegistrationSelection, HeadRoiEvaluation,
)


class HeadProcessingDeadline:
    """Cooperative work deadline leaves time for association and publication.

    OpenCV calls are atomic. A deadline prevents starting additional stages;
    the observer must still reject results that overrun the actual source age.
    """

    def __init__(self, *, image_stamp_sec, scan_stamp_sec, started_ros_sec,
                 started_monotonic_sec, max_sensor_age_sec, reserve_sec=.05):
        values = (image_stamp_sec, scan_stamp_sec, started_ros_sec,
                  started_monotonic_sec, max_sensor_age_sec, reserve_sec)
        if any(not math.isfinite(v) for v in values) or max_sensor_age_sec <= 0 or reserve_sec < 0:
            raise ValueError("head processing requires finite timing and a positive source-age limit")
        remaining = max_sensor_age_sec - max(0., started_ros_sec - min(image_stamp_sec, scan_stamp_sec))
        self.deadline_monotonic_sec = started_monotonic_sec + max(0., remaining - reserve_sec)
        self.reserve_sec = reserve_sec
        self.decisions = []

    def allow(self, stage, *, minimum_work_sec=.01, now=None):
        remaining = self.deadline_monotonic_sec - (time.monotonic() if now is None else now)
        accepted = remaining >= minimum_work_sec
        self.decisions.append(dict(stage=stage, allowed=accepted, remaining_sec=remaining))
        return accepted

    def metadata(self):
        return dict(deadline_monotonic_sec=self.deadline_monotonic_sec,
                    publication_reserve_sec=self.reserve_sec,
                    cooperative_between_stages=True, decisions=list(self.decisions))


def unavailable_head_evaluation(attempt, frame, model_profile, reason, *, diagnostics=None):
    """Record an unmeasured frame without inventing negative QR evidence."""
    roi = attempt.roi
    estimate = replace(_unusable(reason, source=MEASURED_HEAD_AXIS_SOURCE),
        evidence_state="unobservable", model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status)
    debug = StandAxisEdgeDebugArtifacts(
        edges=None, model_reason=reason, model_pose_fit_source=MEASURED_HEAD_AXIS_SOURCE,
        evidence_state="unobservable", model_profile_sha256=model_profile.sha256,
        model_measurement_status=model_profile.measurement_status,
        head_acquisition_diagnostics=diagnostics)
    return HeadRoiEvaluation(attempt, frame[roi.y0:roi.y1, roi.x0:roi.x1], estimate,
        debug, qr_observations=None,
        qr_decode_metadata={"performed": False, "reason": reason})


def select_cold_candidate_head(roi_attempts, *, frame, model_profile, acquire_registered,
                               diagnostics, budget):
    """One bounded wide proposal, one associated recentered fit, or no fit."""
    search = roi_attempts[-1]
    selection = None
    if budget.allow("candidate_head_acquisition", minimum_work_sec=.04):
        selection = acquire_registered(search, None)
    else:
        diagnostics.update(reason="head_acquisition_deadline_exceeded", candidate_associated=False)
    if selection is not None:
        return selection
    reason = diagnostics.get("reason", "model_current_head_border_unavailable")
    failed = unavailable_head_evaluation(search, frame, model_profile, reason,
                                         diagnostics=dict(diagnostics))
    return CameraTargetRegistrationSelection(
        selected=failed, evaluations=(failed,), proposal=None, decision=None,
        strict_retry=None, reacquisition_mode="measured_head",
        head_acquisition=dict(diagnostics))
