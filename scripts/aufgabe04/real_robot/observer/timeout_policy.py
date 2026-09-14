"""ROS-free policy for bounded recovery after an observer deadline.

This grants only a typed observation failure to the existing inspection loop.
It neither admits a perception result nor authorizes another robot movement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)

if TYPE_CHECKING:
    from scripts.aufgabe04.real_robot.observer.diagnostics import (
        PassiveObserverStatusEvidence,
    )


CANDIDATE_LOCAL_OBSERVER_TIMEOUT_STATES = frozenset(
    {
        "collecting_consensus",
        "evidence_not_committable",
        "head_size_projection_mismatch",
        "lidar_target_mismatch",
        "metric_model_measurement_unavailable",
        "target_outside_camera_gate",
    }
)
TRANSIENT_TF_OBSERVER_TIMEOUT_STATES = frozenset(
    {"tf_pending_exact_time", "tf_retry_exhausted"}
)


def _has_transform_ready_candidate_frames(
    status: PassiveObserverStatusEvidence,
) -> bool:
    # These counters advance only after exact-time TF and result freshness
    # succeed. A LiDAR rejection proves processing, not a usable observation.
    return any(
        isinstance(count, int) and not isinstance(count, bool) and count > 0
        for count in (status.accepted_frame_count, status.lidar_rejection_count)
    )


def candidate_local_observer_timeout_basis(
    status: PassiveObserverStatusEvidence,
) -> str | None:
    """Classify the attempt using accumulated evidence, not its last frame.

    The replaceable status may land on a TF retry, obsolete detector result
    or stale input just as the deadline expires. None invalidates earlier candidate
    processing. Missing/malformed evidence and identity conflicts remain
    terminal. Stale input and obsolete results require an explicit unpoisoned
    snapshot; it does not infer readiness from detector activity, soft misses
    or the stale frame itself.
    """

    if (
        status.load_error is not None
        or status.observation_evidence_poisoned is True
        or status.observation_evidence_poison_reason is not None
    ):
        return None
    if status.state in CANDIDATE_LOCAL_OBSERVER_TIMEOUT_STATES:
        return "final_candidate_local_state"
    if not _has_transform_ready_candidate_frames(status):
        return None
    if status.state in TRANSIENT_TF_OBSERVER_TIMEOUT_STATES:
        return "accumulated_transform_ready_candidate_frames"
    if (
        status.state in {"obsolete_detector_result", "stale_sensor_tuple"}
        and status.observation_evidence_poisoned is False
    ):
        return "accumulated_transform_ready_candidate_frames"
    return None


def _expected_deadline_cleanup_exit(
    process: PassiveObserverProcessEvidence,
) -> bool:
    # A crash can race the deadline before cleanup observes the child exit.
    # Accept normal exit or a signal exit attributable to bounded cleanup;
    # unrelated crash codes must not be hidden by an earlier quality status.
    if not isinstance(process.returncode, int) or isinstance(process.returncode, bool):
        return False
    if process.returncode == 0:
        return True
    signal_numbers = {"SIGINT": 2, "SIGTERM": 15, "SIGKILL": 9}
    return any(
        process.returncode in (-number, 128 + number)
        for name, number in signal_numbers.items()
        if name in process.signals_sent
    )


def is_candidate_local_observer_timeout(
    *,
    process: PassiveObserverProcessEvidence,
    status: PassiveObserverStatusEvidence,
) -> bool:
    """Allow only reaped quality deadlines into the existing bounded retry."""

    return (
        process.completion_kind == "deadline"
        and process.deadline_expired
        and _expected_deadline_cleanup_exit(process)
        and candidate_local_observer_timeout_basis(status) is not None
    )


__all__ = [
    "CANDIDATE_LOCAL_OBSERVER_TIMEOUT_STATES",
    "TRANSIENT_TF_OBSERVER_TIMEOUT_STATES",
    "candidate_local_observer_timeout_basis",
    "is_candidate_local_observer_timeout",
]
