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

from scripts.aufgabe04.real_robot.passive_observer_process import (
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
    retry_exhausted_value = payload.get("retry_exhausted")
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
    if status.tf_retry_count is not None:
        details.append(f"tf_retry_count={status.tf_retry_count}")
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
    "PassiveObserverStatusEvidence",
    "format_passive_observer_failure",
    "load_passive_observer_status",
]
