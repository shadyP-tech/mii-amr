"""Fail-closed status diagnostics for one passive observer child.

The ROS observer publishes a replaceable latest-status snapshot and an
append-only event stream.  This module reads only the latest snapshot needed
for a concise mission failure, without making the autonomous runner depend on
the observer's full status schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.real_robot.observer.process import (
    PassiveObserverProcessEvidence,
)


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class PassiveObserverStatusEvidence:
    """Small stable view of the observer's final status snapshot."""

    state: str
    reason: str | None
    consensus_sample_count: int | None
    consensus_required_sample_count: int | None
    tf_retry_count: int | None
    tf_retry_elapsed_sec: float | None
    retry_exhausted: bool | None
    load_error: str | None
    peak_consensus_sample_count: int | None = None
    nearest_lidar_range_delta_m: float | None = None
    tf_retry_attempted_tuple_count: int | None = None
    tf_retry_exhausted_tuple_count: int | None = None
    accepted_frame_count: int | None = None
    lidar_rejection_count: int | None = None
    soft_miss_count: int | None = None
    last_soft_miss_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "consensus_sample_count": self.consensus_sample_count,
            "peak_consensus_sample_count": self.peak_consensus_sample_count,
            "consensus_required_sample_count": (
                self.consensus_required_sample_count
            ),
            "tf_retry_count": self.tf_retry_count,
            "tf_retry_elapsed_sec": self.tf_retry_elapsed_sec,
            "retry_exhausted": self.retry_exhausted,
            "nearest_lidar_range_delta_m": self.nearest_lidar_range_delta_m,
            "tf_retry_attempted_tuple_count": (
                self.tf_retry_attempted_tuple_count
            ),
            "tf_retry_exhausted_tuple_count": (
                self.tf_retry_exhausted_tuple_count
            ),
            "accepted_frame_count": self.accepted_frame_count,
            "lidar_rejection_count": self.lidar_rejection_count,
            "soft_miss_count": self.soft_miss_count,
            "last_soft_miss_reason": self.last_soft_miss_reason,
            "load_error": self.load_error,
        }


def load_passive_observer_status(
    path: Path,
) -> PassiveObserverStatusEvidence:
    """Load useful terminal fields without allowing corrupt status to pass."""

    status_path = Path(path)
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PassiveObserverStatusEvidence(
            state="no_status",
            reason=None,
            consensus_sample_count=None,
            consensus_required_sample_count=None,
            tf_retry_count=None,
            tf_retry_elapsed_sec=None,
            retry_exhausted=None,
            load_error=f"status file is missing: {status_path}",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return PassiveObserverStatusEvidence(
            state="invalid_status",
            reason=None,
            consensus_sample_count=None,
            consensus_required_sample_count=None,
            tf_retry_count=None,
            tf_retry_elapsed_sec=None,
            retry_exhausted=None,
            load_error=f"status file is unreadable: {type(exc).__name__}",
        )
    if not isinstance(payload, Mapping):
        return PassiveObserverStatusEvidence(
            state="invalid_status",
            reason=None,
            consensus_sample_count=None,
            consensus_required_sample_count=None,
            tf_retry_count=None,
            tf_retry_elapsed_sec=None,
            retry_exhausted=None,
            load_error="status root must be a JSON object",
        )

    state_value = payload.get("state")
    state = (
        state_value.strip()
        if isinstance(state_value, str) and state_value.strip()
        else "unknown_status"
    )
    reason_value = payload.get("reason")
    reason = (
        reason_value.strip()
        if isinstance(reason_value, str) and reason_value.strip()
        else None
    )
    consensus = _mapping(payload.get("axis_consensus"))
    tf_retry = _mapping(payload.get("tf_retry"))
    tf_retry_summary = _mapping(payload.get("tf_retry_attempt_summary"))
    lidar_association = _mapping(payload.get("candidate_lidar_association"))
    observation_evidence = _mapping(payload.get("observation_evidence"))
    retry_exhausted_value = payload.get("retry_exhausted")
    last_soft_miss_value = observation_evidence.get("last_soft_miss_reason")
    return PassiveObserverStatusEvidence(
        state=state,
        reason=reason,
        consensus_sample_count=_optional_nonnegative_int(
            consensus.get("sample_count")
        ),
        consensus_required_sample_count=_optional_nonnegative_int(
            consensus.get("required_sample_count")
        ),
        tf_retry_count=_optional_nonnegative_int(tf_retry.get("retry_count")),
        tf_retry_elapsed_sec=_optional_nonnegative_float(
            payload.get("tf_retry_elapsed_sec")
        ),
        retry_exhausted=(
            retry_exhausted_value
            if isinstance(retry_exhausted_value, bool)
            else None
        ),
        load_error=None,
        peak_consensus_sample_count=_optional_nonnegative_int(
            consensus.get("peak_sample_count")
        ),
        nearest_lidar_range_delta_m=_optional_nonnegative_float(
            lidar_association.get("nearest_range_delta_m")
        ),
        tf_retry_attempted_tuple_count=_optional_nonnegative_int(
            tf_retry_summary.get("attempted_tuple_count")
        ),
        tf_retry_exhausted_tuple_count=_optional_nonnegative_int(
            tf_retry_summary.get("exhausted_tuple_count")
        ),
        accepted_frame_count=_optional_nonnegative_int(
            observation_evidence.get("accepted_frame_count")
        ),
        lidar_rejection_count=_optional_nonnegative_int(
            observation_evidence.get("lidar_rejection_count")
        ),
        soft_miss_count=_optional_nonnegative_int(
            observation_evidence.get("soft_miss_count")
        ),
        last_soft_miss_reason=(
            last_soft_miss_value.strip()
            if isinstance(last_soft_miss_value, str)
            and last_soft_miss_value.strip()
            else None
        ),
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
    {
        "tf_pending_exact_time",
        "tf_retry_exhausted",
    }
)


def candidate_local_observer_timeout_basis(
    status: PassiveObserverStatusEvidence,
) -> str | None:
    """Explain why a status snapshot represents candidate-local failure.

    The final status is replaceable and can land on a transient exact-time TF
    retry just as the parent deadline expires.  ``accepted_frame_count`` is
    accumulated independently and increments only after a transform-ready,
    synchronized, LiDAR-associated frame reaches candidate processing.  It
    therefore proves that a trailing TF state did not starve the observer of
    all candidate evidence.
    """

    if status.state in CANDIDATE_LOCAL_OBSERVER_TIMEOUT_STATES:
        return "final_candidate_local_state"
    if (
        status.state in TRANSIENT_TF_OBSERVER_TIMEOUT_STATES
        and status.accepted_frame_count is not None
        and status.accepted_frame_count > 0
    ):
        return "accumulated_transform_ready_candidate_frames"
    return None


def is_candidate_local_observer_timeout(
    *,
    process: PassiveObserverProcessEvidence,
    status: PassiveObserverStatusEvidence,
) -> bool:
    """Classify only reaped, candidate-local quality deadlines as deferrable."""

    return (
        process.completion_kind == "deadline"
        and process.deadline_expired
        and status.load_error is None
        and candidate_local_observer_timeout_basis(status) is not None
    )


def format_passive_observer_failure(
    *,
    candidate_uid: str,
    process: PassiveObserverProcessEvidence,
    status: PassiveObserverStatusEvidence,
    process_evidence_path: Path,
) -> str:
    """Format a precise operator-facing failure from bounded evidence."""

    if process.completion_kind == "deadline":
        lead = "camera/LiDAR observation deadline expired"
    elif process.completion_kind == "child_exit":
        lead = "camera/LiDAR observer exited before producing a terminal artifact"
    else:
        lead = "camera/LiDAR terminal artifact became unavailable"

    details = [
        f"completion={process.completion_kind}",
        f"child_returncode={process.returncode}",
        f"state={status.state}",
    ]
    if status.reason is not None:
        details.append(f"reason={status.reason}")
    if (
        status.consensus_sample_count is not None
        and status.consensus_required_sample_count is not None
    ):
        details.append(
            "consensus="
            f"{status.consensus_sample_count}/"
            f"{status.consensus_required_sample_count}"
        )
    if status.peak_consensus_sample_count is not None:
        details.append(
            f"peak_consensus={status.peak_consensus_sample_count}"
        )
    if status.nearest_lidar_range_delta_m is not None:
        details.append(
            "nearest_lidar_range_delta_m="
            f"{status.nearest_lidar_range_delta_m:.3f}"
        )
    if status.tf_retry_count is not None:
        details.append(f"tf_retry_count={status.tf_retry_count}")
    if status.tf_retry_attempted_tuple_count is not None:
        details.append(
            "tf_retry_attempted_tuples="
            f"{status.tf_retry_attempted_tuple_count}"
        )
    if status.tf_retry_exhausted_tuple_count is not None:
        details.append(
            "tf_retry_exhausted_tuples="
            f"{status.tf_retry_exhausted_tuple_count}"
        )
    if status.accepted_frame_count is not None:
        details.append(
            f"accepted_candidate_frames={status.accepted_frame_count}"
        )
    if status.lidar_rejection_count is not None:
        details.append(f"lidar_rejections={status.lidar_rejection_count}")
    if status.soft_miss_count is not None:
        details.append(f"soft_misses={status.soft_miss_count}")
    if status.last_soft_miss_reason is not None:
        details.append(f"last_soft_miss={status.last_soft_miss_reason}")
    if status.tf_retry_elapsed_sec is not None:
        details.append(
            f"tf_retry_elapsed_sec={status.tf_retry_elapsed_sec:.3f}"
        )
    if status.retry_exhausted is not None:
        details.append(f"retry_exhausted={str(status.retry_exhausted).lower()}")
    if status.load_error is not None:
        details.append(f"status_load_error={status.load_error}")
    details.append(f"observer_process_evidence={process_evidence_path}")
    return f"{lead} for {candidate_uid} without a usable axis; " + "; ".join(
        details
    )


__all__ = [
    "CANDIDATE_LOCAL_OBSERVER_TIMEOUT_STATES",
    "TRANSIENT_TF_OBSERVER_TIMEOUT_STATES",
    "PassiveObserverStatusEvidence",
    "candidate_local_observer_timeout_basis",
    "format_passive_observer_failure",
    "is_candidate_local_observer_timeout",
    "load_passive_observer_status",
]
