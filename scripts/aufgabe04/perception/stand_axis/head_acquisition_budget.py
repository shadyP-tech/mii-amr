"""Cooperative bounds for current-image head acquisition.

OpenCV primitives cannot be preempted. Checkpoints stop the next primitive or
hypothesis loop from starting after expiry. A partially compared set of heads
never returns its first successful proposal as an unambiguous measurement.
"""

from functools import wraps
import math
import time


class HeadAcquisitionDeadlineExceeded(RuntimeError):
    def __init__(self, stage, *, considered_proposals=0, raw_verifications=0):
        super().__init__(stage)
        self.stage = stage
        self.considered_proposals = considered_proposals
        self.raw_verifications = raw_verifications


def check_head_acquisition_deadline(deadline_monotonic_sec, stage, **progress):
    if deadline_monotonic_sec is None:
        return
    if not math.isfinite(deadline_monotonic_sec):
        raise ValueError("head acquisition deadline must be finite")
    if time.monotonic() >= deadline_monotonic_sec:
        raise HeadAcquisitionDeadlineExceeded(stage, **progress)


def bounded_head_acquisition(acquire):
    """Translate expiry into a diagnostic result with no proposal authority."""
    @wraps(acquire)
    def bounded(*args, **kwargs):
        try:
            return acquire(*args, **kwargs)
        except HeadAcquisitionDeadlineExceeded as error:
            from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposalResult

            return HeadProposalResult(
                None, "head_acquisition_deadline_exceeded",
                error.considered_proposals, error.raw_verifications,
                joint_border_diagnostics={
                    "deadline_exceeded": True, "deadline_stage": error.stage,
                    "comparison_complete": False,
                    "angle_authorized": False, "motion_authorized": False,
                },
            )
    return bounded
