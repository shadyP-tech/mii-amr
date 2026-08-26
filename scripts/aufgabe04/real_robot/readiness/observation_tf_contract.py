"""ROS-free contract for exact-time LiDAR transform readiness.

This module owns only immutable evidence, validation, failure taxonomy, and the
pure readiness decision.  ROS collection belongs in
``autonomous_observation_tf_readiness`` so offline planning and tests can import
the contract without importing ROS2/tf2 adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

from scripts.aufgabe04.perception.stand_observation import (
    DEFAULT_OBSERVATION_TIMING_LIMITS,
    validated_observation_timing,
)


OBSERVATION_TF_READINESS_SCHEMA_VERSION = 1

FAILURE_ROS_UNAVAILABLE = "ros_unavailable"
FAILURE_OBSERVATION_EFFECT = "observation_effect_failed"
FAILURE_SCAN_TIMEOUT = "scan_timeout"
FAILURE_OBSERVER_CLOCK = "observer_clock_invalid"
FAILURE_SCAN_FRAME_EMPTY = "scan_frame_empty"
FAILURE_SCAN_FRAME_MISMATCH = "scan_frame_mismatch"
FAILURE_SCAN_STAMP_INVALID = "scan_stamp_invalid"
FAILURE_SCAN_STAMP_STALE = "scan_stamp_stale"
FAILURE_SCAN_STAMP_FUTURE = "scan_stamp_in_future"
FAILURE_TRANSFORM_UNAVAILABLE = "exact_time_transform_unavailable"
FAILURE_TRANSFORM_NOT_EXACT_TIME = "transform_query_not_exact_scan_time"
FAILURE_TRANSFORM_FRAME_MISMATCH = "transform_frame_mismatch"
FAILURE_TRANSFORM_PAYLOAD_INVALID = "transform_payload_invalid"
FAILURE_TRANSFORM_TIMING = "transform_timing_rejected"


@dataclass(frozen=True)
class ObservationTfReadinessConfig:
    """Bounded inputs for one passive LiDAR/TF readiness observation."""

    scan_topic: str
    expected_scan_frame: str
    target_frame: str
    timeout_sec: float = 3.0
    max_scan_age_sec: float = (
        DEFAULT_OBSERVATION_TIMING_LIMITS.max_scan_age_sec
    )
    max_future_timestamp_sec: float = (
        DEFAULT_OBSERVATION_TIMING_LIMITS.max_future_timestamp_sec
    )
    max_tf_age_sec: float = DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_age_sec
    max_tf_scan_skew_sec: float = (
        DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_scan_skew_sec
    )
    poll_interval_sec: float = 0.02

    def validated(self) -> "ObservationTfReadinessConfig":
        for name, value in (
            ("scan_topic", self.scan_topic),
            ("expected_scan_frame", self.expected_scan_frame),
            ("target_frame", self.target_frame),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty exact ROS name")
        for name, value in (
            ("timeout_sec", self.timeout_sec),
            ("max_scan_age_sec", self.max_scan_age_sec),
            ("max_tf_age_sec", self.max_tf_age_sec),
            ("poll_interval_sec", self.poll_interval_sec),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.max_future_timestamp_sec)
            or self.max_future_timestamp_sec < 0.0
        ):
            raise ValueError(
                "max_future_timestamp_sec must be finite and non-negative"
            )
        if (
            not math.isfinite(self.max_tf_scan_skew_sec)
            or self.max_tf_scan_skew_sec < 0.0
        ):
            raise ValueError(
                "max_tf_scan_skew_sec must be finite and non-negative"
            )
        if self.poll_interval_sec > self.timeout_sec:
            raise ValueError("poll_interval_sec must not exceed timeout_sec")
        return self


@dataclass(frozen=True)
class ObservationTfEvidence:
    """Serializable raw evidence collected without a policy decision."""

    observed_at_ns: int
    scan_received: bool
    scan_frame: str | None = None
    scan_stamp_ns: int | None = None
    transform_checked: bool = False
    transform_available: bool = False
    transform_target_frame: str | None = None
    transform_source_frame: str | None = None
    transform_query_stamp_ns: int | None = None
    transform_stamp_ns: int | None = None
    transform_x_m: float | None = None
    transform_y_m: float | None = None
    transform_z_m: float | None = None
    transform_yaw_rad: float | None = None
    transform_quaternion_norm: float | None = None
    transform_error: str | None = None
    timed_out: bool = False
    observer_failure_code: str | None = None
    observer_error: str | None = None


@dataclass(frozen=True)
class ObservationTfReadinessResult:
    """Typed fail-closed outcome suitable for direct JSON persistence."""

    ready: bool
    failure_code: str | None
    detail: str
    scan_age_sec: float | None
    tf_age_sec: float | None
    tf_scan_skew_sec: float | None
    config: ObservationTfReadinessConfig
    evidence: ObservationTfEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OBSERVATION_TF_READINESS_SCHEMA_VERSION,
            "kind": "autonomous_observation_tf_readiness",
            "ready": self.ready,
            "failure_code": self.failure_code,
            "detail": self.detail,
            "scan_age_sec": self.scan_age_sec,
            "tf_age_sec": self.tf_age_sec,
            "tf_scan_skew_sec": self.tf_scan_skew_sec,
            "motion_published": False,
            "operator_input_requested": False,
            "subprocess_started": False,
            "config": asdict(self.config),
            "evidence": asdict(self.evidence),
        }

    def to_failure_fields(self) -> dict[str, object]:
        if self.ready:
            raise ValueError("ready observation-TF result is not a failure")
        return {
            "failure_phase": "observation_tf_readiness",
            "observation_tf_failure_code": self.failure_code,
            "observation_tf_detail": self.detail,
            "observation_tf_scan_age_sec": self.scan_age_sec,
            "observation_tf_tf_age_sec": self.tf_age_sec,
            "observation_tf_tf_scan_skew_sec": self.tf_scan_skew_sec,
            "typed_run_requested": False,
            "motion_authorized": False,
            "motion_published": False,
        }


class ObservationTfReadinessError(RuntimeError):
    """A persisted passive LiDAR/TF gate rejected the requested phase."""

    def __init__(
        self,
        result: ObservationTfReadinessResult,
        *,
        evidence_path: str,
        evidence_sha256: str,
        phase: str,
        typed_run_already_issued: bool = False,
    ) -> None:
        if result.ready:
            raise ValueError("cannot raise readiness error for a ready result")
        self.result = result
        self.evidence_path = evidence_path
        self.evidence_sha256 = evidence_sha256
        self.phase = phase
        self.typed_run_already_issued = bool(typed_run_already_issued)
        super().__init__(
            f"{phase} rejected: {result.failure_code}: {result.detail}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            **self.result.to_failure_fields(),
            "failure_phase": self.phase,
            "observation_tf_readiness_json": self.evidence_path,
            "observation_tf_readiness_sha256": self.evidence_sha256,
            "typed_run_requested": self.typed_run_already_issued,
            "typed_run_already_issued": self.typed_run_already_issued,
        }


ObservationEffect = Callable[
    [ObservationTfReadinessConfig], ObservationTfEvidence
]


def _result(
    config: ObservationTfReadinessConfig,
    evidence: ObservationTfEvidence,
    *,
    ready: bool,
    failure_code: str | None,
    detail: str,
    scan_age_sec: float | None = None,
    tf_age_sec: float | None = None,
    tf_scan_skew_sec: float | None = None,
) -> ObservationTfReadinessResult:
    return ObservationTfReadinessResult(
        ready=ready,
        failure_code=failure_code,
        detail=detail,
        scan_age_sec=scan_age_sec,
        tf_age_sec=tf_age_sec,
        tf_scan_skew_sec=tf_scan_skew_sec,
        config=config,
        evidence=evidence,
    )


def evaluate_observation_tf_readiness(
    config: ObservationTfReadinessConfig,
    evidence: ObservationTfEvidence,
) -> ObservationTfReadinessResult:
    """Evaluate one passive observation using exact frame and timestamp identity."""

    selected = config.validated()
    if evidence.observer_failure_code is not None:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=evidence.observer_failure_code,
            detail=evidence.observer_error or "observation effect failed",
        )
    if not evidence.scan_received:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_SCAN_TIMEOUT,
            detail=(
                f"no LaserScan arrived on {selected.scan_topic!r} within "
                f"{selected.timeout_sec:.3f}s"
            ),
        )
    if (
        not isinstance(evidence.observed_at_ns, int)
        or isinstance(evidence.observed_at_ns, bool)
        or evidence.observed_at_ns <= 0
    ):
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_OBSERVER_CLOCK,
            detail="observer clock must be a positive nanosecond timestamp",
        )
    if not evidence.scan_frame:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_SCAN_FRAME_EMPTY,
            detail="LaserScan header.frame_id is empty",
        )
    if evidence.scan_frame != selected.expected_scan_frame:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_SCAN_FRAME_MISMATCH,
            detail=(
                f"LaserScan frame {evidence.scan_frame!r} does not exactly match "
                f"expected frame {selected.expected_scan_frame!r}"
            ),
        )
    if (
        not isinstance(evidence.scan_stamp_ns, int)
        or isinstance(evidence.scan_stamp_ns, bool)
        or evidence.scan_stamp_ns <= 0
    ):
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_SCAN_STAMP_INVALID,
            detail="LaserScan timestamp must be nonzero and positive",
        )

    scan_age_sec = (
        evidence.observed_at_ns - evidence.scan_stamp_ns
    ) / 1_000_000_000.0
    if scan_age_sec > selected.max_scan_age_sec:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_SCAN_STAMP_STALE,
            detail=(
                f"LaserScan is stale: age={scan_age_sec:.6f}s exceeds "
                f"{selected.max_scan_age_sec:.6f}s"
            ),
            scan_age_sec=scan_age_sec,
        )
    if scan_age_sec < -selected.max_future_timestamp_sec:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_SCAN_STAMP_FUTURE,
            detail=(
                f"LaserScan is future-dated: age={scan_age_sec:.6f}s is below "
                f"-{selected.max_future_timestamp_sec:.6f}s"
            ),
            scan_age_sec=scan_age_sec,
        )
    if not evidence.transform_checked or not evidence.transform_available:
        suffix = (
            ""
            if not evidence.transform_error
            else f": {evidence.transform_error}"
        )
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_TRANSFORM_UNAVAILABLE,
            detail=(
                f"exact-time {selected.target_frame}<-"
                f"{selected.expected_scan_frame} transform unavailable at "
                f"scan stamp {evidence.scan_stamp_ns}ns{suffix}"
            ),
            scan_age_sec=scan_age_sec,
        )
    if evidence.transform_query_stamp_ns != evidence.scan_stamp_ns:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_TRANSFORM_NOT_EXACT_TIME,
            detail=(
                "transform query timestamp does not exactly equal the "
                "LaserScan timestamp"
            ),
            scan_age_sec=scan_age_sec,
        )
    if (
        evidence.transform_target_frame != selected.target_frame
        or evidence.transform_source_frame != selected.expected_scan_frame
    ):
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_TRANSFORM_FRAME_MISMATCH,
            detail=(
                "returned transform frame identity does not exactly match "
                f"{selected.target_frame}<-{selected.expected_scan_frame}"
            ),
            scan_age_sec=scan_age_sec,
        )
    transform_values = (
        evidence.transform_x_m,
        evidence.transform_y_m,
        evidence.transform_z_m,
        evidence.transform_yaw_rad,
        evidence.transform_quaternion_norm,
    )
    if (
        not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in transform_values
        )
        or abs(float(evidence.transform_quaternion_norm) - 1.0) > 1.0e-3
    ):
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_TRANSFORM_PAYLOAD_INVALID,
            detail=(
                "exact-time transform pose must be finite and its quaternion "
                "must be normalized"
            ),
            scan_age_sec=scan_age_sec,
        )
    try:
        timing = validated_observation_timing(
            observer_clock_sec=evidence.observed_at_ns / 1_000_000_000.0,
            scan_stamp_sec=evidence.scan_stamp_ns / 1_000_000_000.0,
            tf_stamp_sec=evidence.transform_stamp_ns / 1_000_000_000.0,
            max_scan_age_sec=selected.max_scan_age_sec,
            max_future_timestamp_sec=selected.max_future_timestamp_sec,
            max_tf_age_sec=selected.max_tf_age_sec,
            max_tf_scan_skew_sec=selected.max_tf_scan_skew_sec,
        )
    except (TypeError, ValueError) as exc:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_TRANSFORM_TIMING,
            detail=f"exact-time transform timing rejected: {exc}",
            scan_age_sec=scan_age_sec,
        )
    return _result(
        selected,
        evidence,
        ready=True,
        failure_code=None,
        detail=(
            f"fresh LaserScan and exact-time {selected.target_frame}<-"
            f"{selected.expected_scan_frame} transform are ready"
        ),
        scan_age_sec=scan_age_sec,
        tf_age_sec=timing.tf_age_sec,
        tf_scan_skew_sec=timing.tf_scan_skew_sec,
    )


__all__ = [
    "FAILURE_OBSERVATION_EFFECT",
    "FAILURE_OBSERVER_CLOCK",
    "FAILURE_ROS_UNAVAILABLE",
    "FAILURE_SCAN_FRAME_EMPTY",
    "FAILURE_SCAN_FRAME_MISMATCH",
    "FAILURE_SCAN_STAMP_FUTURE",
    "FAILURE_SCAN_STAMP_INVALID",
    "FAILURE_SCAN_STAMP_STALE",
    "FAILURE_SCAN_TIMEOUT",
    "FAILURE_TRANSFORM_FRAME_MISMATCH",
    "FAILURE_TRANSFORM_NOT_EXACT_TIME",
    "FAILURE_TRANSFORM_PAYLOAD_INVALID",
    "FAILURE_TRANSFORM_TIMING",
    "FAILURE_TRANSFORM_UNAVAILABLE",
    "OBSERVATION_TF_READINESS_SCHEMA_VERSION",
    "ObservationEffect",
    "ObservationTfEvidence",
    "ObservationTfReadinessConfig",
    "ObservationTfReadinessError",
    "ObservationTfReadinessResult",
    "evaluate_observation_tf_readiness",
]
