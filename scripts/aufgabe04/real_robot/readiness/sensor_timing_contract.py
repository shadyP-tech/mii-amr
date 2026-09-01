"""ROS-free camera/LiDAR header-timing readiness contract.

The passive viewpoint observer can only judge freshness after it has assembled
a synchronized image/CameraInfo/LaserScan tuple. This module owns the earlier
fail-closed decision: all three publishers must be current in one ROS clock
domain and a real tuple must exist before any camera-enabled route may start.
ROS subscriptions and persistence deliberately live in separate adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable


SENSOR_TIMING_READINESS_SCHEMA_VERSION = 1

# Shared with the passive observer so the pre-motion and observation policies
# cannot silently drift apart.
DEFAULT_MAX_SENSOR_AGE_SEC = 0.5
DEFAULT_MAX_CAMERA_INFO_AGE_SEC = 1.0
DEFAULT_MAX_FUTURE_TIMESTAMP_SEC = 0.05
DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC = 0.10
DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC = 1.0
DEFAULT_SENSOR_TIMING_TIMEOUT_SEC = 5.0
DEFAULT_SENSOR_TIMING_SAMPLE_CAPACITY = 64

FAILURE_ROS_UNAVAILABLE = "ros_unavailable"
FAILURE_OBSERVATION_EFFECT = "sensor_timing_effect_failed"
FAILURE_OBSERVER_CLOCK = "observer_clock_invalid"
FAILURE_IMAGE_TIMEOUT = "camera_image_timeout"
FAILURE_CAMERA_INFO_TIMEOUT = "camera_info_timeout"
FAILURE_SCAN_TIMEOUT = "scan_timeout"
FAILURE_IMAGE_FRAME_EMPTY = "camera_image_frame_empty"
FAILURE_CAMERA_INFO_FRAME_EMPTY = "camera_info_frame_empty"
FAILURE_SCAN_FRAME_EMPTY = "scan_frame_empty"
FAILURE_IMAGE_FRAME_MISMATCH = "camera_image_frame_mismatch"
FAILURE_CAMERA_INFO_FRAME_MISMATCH = "camera_info_frame_mismatch"
FAILURE_SCAN_FRAME_MISMATCH = "scan_frame_mismatch"
FAILURE_IMAGE_STAMP_INVALID = "camera_image_stamp_invalid"
FAILURE_CAMERA_INFO_STAMP_INVALID = "camera_info_stamp_invalid"
FAILURE_SCAN_STAMP_INVALID = "scan_stamp_invalid"
FAILURE_IMAGE_STAMP_STALE = "camera_image_stamp_stale"
FAILURE_CAMERA_INFO_STAMP_STALE = "camera_info_stamp_stale"
FAILURE_SCAN_STAMP_STALE = "scan_stamp_stale"
FAILURE_IMAGE_STAMP_FUTURE = "camera_image_stamp_in_future"
FAILURE_CAMERA_INFO_STAMP_FUTURE = "camera_info_stamp_in_future"
FAILURE_SCAN_STAMP_FUTURE = "scan_stamp_in_future"
FAILURE_IMAGE_SCAN_SKEW = "camera_image_scan_skew_exceeded"
FAILURE_CAMERA_INFO_IMAGE_SKEW = "camera_info_image_skew_exceeded"
FAILURE_FRESH_TUPLE_UNAVAILABLE = "fresh_synchronized_header_tuple_unavailable"


@dataclass(frozen=True)
class SensorTimingReadinessConfig:
    """Sealed topics, frames, limits, and collection bounds for one gate."""

    image_topic: str
    camera_info_topic: str
    scan_topic: str
    expected_image_frame: str
    expected_camera_info_frame: str
    expected_scan_frame: str
    timeout_sec: float = DEFAULT_SENSOR_TIMING_TIMEOUT_SEC
    max_image_age_sec: float = DEFAULT_MAX_SENSOR_AGE_SEC
    max_camera_info_age_sec: float = DEFAULT_MAX_CAMERA_INFO_AGE_SEC
    max_scan_age_sec: float = DEFAULT_MAX_SENSOR_AGE_SEC
    max_future_timestamp_sec: float = DEFAULT_MAX_FUTURE_TIMESTAMP_SEC
    max_image_scan_skew_sec: float = DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC
    max_camera_info_image_skew_sec: float = (
        DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC
    )
    poll_interval_sec: float = 0.02
    sample_capacity: int = DEFAULT_SENSOR_TIMING_SAMPLE_CAPACITY

    def validated(self) -> "SensorTimingReadinessConfig":
        for name, value in (
            ("image_topic", self.image_topic),
            ("camera_info_topic", self.camera_info_topic),
            ("scan_topic", self.scan_topic),
            ("expected_image_frame", self.expected_image_frame),
            ("expected_camera_info_frame", self.expected_camera_info_frame),
            ("expected_scan_frame", self.expected_scan_frame),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty exact ROS name")
        for name, value in (
            ("timeout_sec", self.timeout_sec),
            ("max_image_age_sec", self.max_image_age_sec),
            ("max_camera_info_age_sec", self.max_camera_info_age_sec),
            ("max_scan_age_sec", self.max_scan_age_sec),
            ("poll_interval_sec", self.poll_interval_sec),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("max_future_timestamp_sec", self.max_future_timestamp_sec),
            ("max_image_scan_skew_sec", self.max_image_scan_skew_sec),
            (
                "max_camera_info_image_skew_sec",
                self.max_camera_info_image_skew_sec,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.poll_interval_sec > self.timeout_sec:
            raise ValueError("poll_interval_sec must not exceed timeout_sec")
        if type(self.sample_capacity) is not int or self.sample_capacity < 1:
            raise ValueError("sample_capacity must be a positive integer")
        return self


@dataclass(frozen=True)
class HeaderSample:
    """One normalized ROS header captured with its local receipt time."""

    stamp_ns: int | None
    frame_id: str | None
    receipt_ns: int


@dataclass(frozen=True)
class SensorTimingEvidence:
    """Serializable bounded samples collected without a policy decision."""

    observed_at_ns: int
    image_samples: tuple[HeaderSample, ...] = ()
    camera_info_samples: tuple[HeaderSample, ...] = ()
    scan_samples: tuple[HeaderSample, ...] = ()
    timed_out: bool = False
    observer_failure_code: str | None = None
    observer_error: str | None = None


@dataclass(frozen=True)
class SensorTimingReadinessResult:
    """Typed fail-closed outcome suitable for content-hashed persistence."""

    ready: bool
    failure_code: str | None
    detail: str
    image_age_sec: float | None
    camera_info_age_sec: float | None
    scan_age_sec: float | None
    image_scan_skew_sec: float | None
    camera_info_image_skew_sec: float | None
    selected_image_stamp_ns: int | None
    selected_camera_info_stamp_ns: int | None
    selected_scan_stamp_ns: int | None
    config: SensorTimingReadinessConfig
    evidence: SensorTimingEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SENSOR_TIMING_READINESS_SCHEMA_VERSION,
            "kind": "autonomous_camera_lidar_timing_readiness",
            "ready": self.ready,
            "failure_code": self.failure_code,
            "detail": self.detail,
            "image_age_sec": self.image_age_sec,
            "camera_info_age_sec": self.camera_info_age_sec,
            "scan_age_sec": self.scan_age_sec,
            "image_scan_skew_sec": self.image_scan_skew_sec,
            "camera_info_image_skew_sec": self.camera_info_image_skew_sec,
            "selected_image_stamp_ns": self.selected_image_stamp_ns,
            "selected_camera_info_stamp_ns": (
                self.selected_camera_info_stamp_ns
            ),
            "selected_scan_stamp_ns": self.selected_scan_stamp_ns,
            "selection_policy": "newest_fresh_complete_header_tuple",
            "motion_published": False,
            "operator_input_requested": False,
            "subprocess_started": False,
            "config": asdict(self.config),
            "evidence": asdict(self.evidence),
        }

    def to_failure_fields(self) -> dict[str, object]:
        if self.ready:
            raise ValueError("ready sensor-timing result is not a failure")
        return {
            "failure_phase": "sensor_timing_readiness",
            "sensor_timing_failure_code": self.failure_code,
            "sensor_timing_detail": self.detail,
            "sensor_timing_image_age_sec": self.image_age_sec,
            "sensor_timing_camera_info_age_sec": self.camera_info_age_sec,
            "sensor_timing_scan_age_sec": self.scan_age_sec,
            "sensor_timing_image_scan_skew_sec": self.image_scan_skew_sec,
            "sensor_timing_camera_info_image_skew_sec": (
                self.camera_info_image_skew_sec
            ),
            "typed_run_requested": False,
            "motion_authorized": False,
            "motion_published": False,
        }


class SensorTimingReadinessError(RuntimeError):
    """A persisted camera/LiDAR timing gate rejected the requested phase."""

    def __init__(
        self,
        result: SensorTimingReadinessResult,
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
            "sensor_timing_readiness_json": self.evidence_path,
            "sensor_timing_readiness_sha256": self.evidence_sha256,
            "typed_run_requested": self.typed_run_already_issued,
            "typed_run_already_issued": self.typed_run_already_issued,
        }


SensorTimingEffect = Callable[
    [SensorTimingReadinessConfig], SensorTimingEvidence
]


def _result(
    config: SensorTimingReadinessConfig,
    evidence: SensorTimingEvidence,
    *,
    ready: bool,
    failure_code: str | None,
    detail: str,
    image_age_sec: float | None = None,
    camera_info_age_sec: float | None = None,
    scan_age_sec: float | None = None,
    image_scan_skew_sec: float | None = None,
    camera_info_image_skew_sec: float | None = None,
    image_stamp_ns: int | None = None,
    camera_info_stamp_ns: int | None = None,
    scan_stamp_ns: int | None = None,
) -> SensorTimingReadinessResult:
    return SensorTimingReadinessResult(
        ready=ready,
        failure_code=failure_code,
        detail=detail,
        image_age_sec=image_age_sec,
        camera_info_age_sec=camera_info_age_sec,
        scan_age_sec=scan_age_sec,
        image_scan_skew_sec=image_scan_skew_sec,
        camera_info_image_skew_sec=camera_info_image_skew_sec,
        selected_image_stamp_ns=image_stamp_ns,
        selected_camera_info_stamp_ns=camera_info_stamp_ns,
        selected_scan_stamp_ns=scan_stamp_ns,
        config=config,
        evidence=evidence,
    )


def _stream_failure(
    config: SensorTimingReadinessConfig,
    evidence: SensorTimingEvidence,
    *,
    samples: tuple[HeaderSample, ...],
    label: str,
    expected_frame: str,
    timeout_code: str,
    empty_frame_code: str,
    frame_mismatch_code: str,
    invalid_stamp_code: str,
) -> SensorTimingReadinessResult | None:
    if not samples:
        return _result(
            config,
            evidence,
            ready=False,
            failure_code=timeout_code,
            detail=(
                f"no {label} header arrived within {config.timeout_sec:.3f}s"
            ),
        )
    sample = samples[-1]
    if not sample.frame_id:
        return _result(
            config,
            evidence,
            ready=False,
            failure_code=empty_frame_code,
            detail=f"newest {label} header.frame_id is empty",
        )
    if sample.frame_id != expected_frame:
        return _result(
            config,
            evidence,
            ready=False,
            failure_code=frame_mismatch_code,
            detail=(
                f"newest {label} frame {sample.frame_id!r} does not exactly "
                f"match expected frame {expected_frame!r}"
            ),
        )
    if (
        not isinstance(sample.stamp_ns, int)
        or isinstance(sample.stamp_ns, bool)
        or sample.stamp_ns <= 0
    ):
        return _result(
            config,
            evidence,
            ready=False,
            failure_code=invalid_stamp_code,
            detail=f"newest {label} header stamp must be nonzero and positive",
        )
    return None


def _selectable_samples(
    samples: tuple[HeaderSample, ...],
    *,
    expected_frame: str,
) -> tuple[HeaderSample, ...]:
    """Discard superseded malformed samples after the newest header is valid."""

    return tuple(
        sample
        for sample in samples
        if sample.frame_id == expected_frame
        and isinstance(sample.stamp_ns, int)
        and not isinstance(sample.stamp_ns, bool)
        and sample.stamp_ns > 0
    )


def _fresh_samples(
    samples: tuple[HeaderSample, ...],
    *,
    observed_at_ns: int,
    maximum_age_sec: float,
    maximum_future_sec: float,
) -> tuple[HeaderSample, ...]:
    """Keep valid headers whose stamps are current in the observer clock."""

    return tuple(
        sample
        for sample in samples
        if -maximum_future_sec
        <= (observed_at_ns - int(sample.stamp_ns)) / 1_000_000_000.0
        <= maximum_age_sec
    )


def _age_failure(
    config: SensorTimingReadinessConfig,
    evidence: SensorTimingEvidence,
    *,
    stamp_ns: int,
    maximum_age_sec: float,
    label: str,
    stale_code: str,
    future_code: str,
) -> tuple[float, SensorTimingReadinessResult | None]:
    age_sec = (evidence.observed_at_ns - stamp_ns) / 1_000_000_000.0
    age_fields = {
        "image_age_sec": age_sec if label == "camera image" else None,
        "camera_info_age_sec": age_sec if label == "CameraInfo" else None,
        "scan_age_sec": age_sec if label == "LaserScan" else None,
    }
    if age_sec > maximum_age_sec:
        return age_sec, _result(
            config,
            evidence,
            ready=False,
            failure_code=stale_code,
            detail=(
                f"{label} header is stale: age={age_sec:.6f}s exceeds "
                f"{maximum_age_sec:.6f}s"
            ),
            **age_fields,
        )
    if age_sec < -config.max_future_timestamp_sec:
        return age_sec, _result(
            config,
            evidence,
            ready=False,
            failure_code=future_code,
            detail=(
                f"{label} header is future-dated: age={age_sec:.6f}s is "
                f"below -{config.max_future_timestamp_sec:.6f}s"
            ),
            **age_fields,
        )
    return age_sec, None


def _nearest(
    samples: tuple[HeaderSample, ...],
    stamp_ns: int,
) -> HeaderSample:
    """Return a deterministic nearest sample (older sample wins exact ties)."""

    return min(
        samples,
        key=lambda sample: (
            abs(int(sample.stamp_ns) - stamp_ns),
            int(sample.stamp_ns) > stamp_ns,
            int(sample.stamp_ns),
        ),
    )


def evaluate_sensor_timing_readiness(
    config: SensorTimingReadinessConfig,
    evidence: SensorTimingEvidence,
) -> SensorTimingReadinessResult:
    """Find the newest fresh complete tuple and reject clock divergence."""

    selected = config.validated()
    if evidence.observer_failure_code is not None:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=evidence.observer_failure_code,
            detail=evidence.observer_error or "sensor timing effect failed",
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

    streams = (
        (
            evidence.image_samples,
            "camera image",
            selected.expected_image_frame,
            FAILURE_IMAGE_TIMEOUT,
            FAILURE_IMAGE_FRAME_EMPTY,
            FAILURE_IMAGE_FRAME_MISMATCH,
            FAILURE_IMAGE_STAMP_INVALID,
        ),
        (
            evidence.camera_info_samples,
            "CameraInfo",
            selected.expected_camera_info_frame,
            FAILURE_CAMERA_INFO_TIMEOUT,
            FAILURE_CAMERA_INFO_FRAME_EMPTY,
            FAILURE_CAMERA_INFO_FRAME_MISMATCH,
            FAILURE_CAMERA_INFO_STAMP_INVALID,
        ),
        (
            evidence.scan_samples,
            "LaserScan",
            selected.expected_scan_frame,
            FAILURE_SCAN_TIMEOUT,
            FAILURE_SCAN_FRAME_EMPTY,
            FAILURE_SCAN_FRAME_MISMATCH,
            FAILURE_SCAN_STAMP_INVALID,
        ),
    )
    for (
        samples,
        label,
        expected_frame,
        timeout_code,
        empty_frame_code,
        frame_mismatch_code,
        invalid_stamp_code,
    ) in streams:
        failure = _stream_failure(
            selected,
            evidence,
            samples=samples,
            label=label,
            expected_frame=expected_frame,
            timeout_code=timeout_code,
            empty_frame_code=empty_frame_code,
            frame_mismatch_code=frame_mismatch_code,
            invalid_stamp_code=invalid_stamp_code,
        )
        if failure is not None:
            return failure

    # Callback order, not numerical timestamp order, defines "newest". This
    # prevents a publisher whose clock just jumped backwards from borrowing a
    # formerly current header that remains in the bounded window.
    newest_image = evidence.image_samples[-1]
    newest_camera_info = evidence.camera_info_samples[-1]
    newest_scan = evidence.scan_samples[-1]
    newest_streams = (
        (
            newest_image,
            selected.max_image_age_sec,
            "camera image",
            FAILURE_IMAGE_STAMP_STALE,
            FAILURE_IMAGE_STAMP_FUTURE,
        ),
        (
            newest_camera_info,
            selected.max_camera_info_age_sec,
            "CameraInfo",
            FAILURE_CAMERA_INFO_STAMP_STALE,
            FAILURE_CAMERA_INFO_STAMP_FUTURE,
        ),
        (
            newest_scan,
            selected.max_scan_age_sec,
            "LaserScan",
            FAILURE_SCAN_STAMP_STALE,
            FAILURE_SCAN_STAMP_FUTURE,
        ),
    )
    for sample, maximum_age, label, stale_code, future_code in newest_streams:
        _age, failure = _age_failure(
            selected,
            evidence,
            stamp_ns=int(sample.stamp_ns),
            maximum_age_sec=maximum_age,
            label=label,
            stale_code=stale_code,
            future_code=future_code,
        )
        if failure is not None:
            return failure

    selectable_images = _fresh_samples(
        _selectable_samples(
            evidence.image_samples,
            expected_frame=selected.expected_image_frame,
        ),
        observed_at_ns=evidence.observed_at_ns,
        maximum_age_sec=selected.max_image_age_sec,
        maximum_future_sec=selected.max_future_timestamp_sec,
    )
    selectable_camera_infos = _fresh_samples(
        _selectable_samples(
            evidence.camera_info_samples,
            expected_frame=selected.expected_camera_info_frame,
        ),
        observed_at_ns=evidence.observed_at_ns,
        maximum_age_sec=selected.max_camera_info_age_sec,
        maximum_future_sec=selected.max_future_timestamp_sec,
    )
    selectable_scans = _fresh_samples(
        _selectable_samples(
            evidence.scan_samples,
            expected_frame=selected.expected_scan_frame,
        ),
        observed_at_ns=evidence.observed_at_ns,
        maximum_age_sec=selected.max_scan_age_sec,
        maximum_future_sec=selected.max_future_timestamp_sec,
    )

    diagnostic: tuple[
        HeaderSample,
        HeaderSample,
        HeaderSample,
        float,
        float,
    ] | None = None
    for image in reversed(selectable_images):
        scan = _nearest(selectable_scans, int(image.stamp_ns))
        camera_info = _nearest(
            selectable_camera_infos, int(image.stamp_ns)
        )
        image_scan_skew_sec = (
            abs(int(image.stamp_ns) - int(scan.stamp_ns)) / 1_000_000_000.0
        )
        camera_info_image_skew_sec = (
            abs(int(camera_info.stamp_ns) - int(image.stamp_ns))
            / 1_000_000_000.0
        )
        if diagnostic is None:
            diagnostic = (
                image,
                camera_info,
                scan,
                image_scan_skew_sec,
                camera_info_image_skew_sec,
            )
        image_age_sec = (
            evidence.observed_at_ns - int(image.stamp_ns)
        ) / 1_000_000_000.0
        camera_info_age_sec = (
            evidence.observed_at_ns - int(camera_info.stamp_ns)
        ) / 1_000_000_000.0
        scan_age_sec = (
            evidence.observed_at_ns - int(scan.stamp_ns)
        ) / 1_000_000_000.0
        if not (
            -selected.max_future_timestamp_sec
            <= image_age_sec
            <= selected.max_image_age_sec
            and -selected.max_future_timestamp_sec
            <= camera_info_age_sec
            <= selected.max_camera_info_age_sec
            and -selected.max_future_timestamp_sec
            <= scan_age_sec
            <= selected.max_scan_age_sec
            and image_scan_skew_sec <= selected.max_image_scan_skew_sec
            and camera_info_image_skew_sec
            <= selected.max_camera_info_image_skew_sec
        ):
            continue
        return _result(
            selected,
            evidence,
            ready=True,
            failure_code=None,
            detail=(
                "fresh camera image, CameraInfo, and LaserScan headers form "
                "a synchronized tuple"
            ),
            image_age_sec=image_age_sec,
            camera_info_age_sec=camera_info_age_sec,
            scan_age_sec=scan_age_sec,
            image_scan_skew_sec=image_scan_skew_sec,
            camera_info_image_skew_sec=camera_info_image_skew_sec,
            image_stamp_ns=int(image.stamp_ns),
            camera_info_stamp_ns=int(camera_info.stamp_ns),
            scan_stamp_ns=int(scan.stamp_ns),
        )

    assert diagnostic is not None
    image, camera_info, scan, image_scan_skew, info_image_skew = diagnostic
    image_age = (
        evidence.observed_at_ns - int(image.stamp_ns)
    ) / 1_000_000_000.0
    camera_info_age = (
        evidence.observed_at_ns - int(camera_info.stamp_ns)
    ) / 1_000_000_000.0
    scan_age = (
        evidence.observed_at_ns - int(scan.stamp_ns)
    ) / 1_000_000_000.0
    fields = {
        "image_age_sec": image_age,
        "camera_info_age_sec": camera_info_age,
        "scan_age_sec": scan_age,
        "image_scan_skew_sec": image_scan_skew,
        "camera_info_image_skew_sec": info_image_skew,
        "image_stamp_ns": int(image.stamp_ns),
        "camera_info_stamp_ns": int(camera_info.stamp_ns),
        "scan_stamp_ns": int(scan.stamp_ns),
    }
    if image_scan_skew > selected.max_image_scan_skew_sec:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_IMAGE_SCAN_SKEW,
            detail=(
                "no fresh image/LaserScan pair met the pre-motion skew limit: "
                f"nearest={image_scan_skew:.6f}s > "
                f"{selected.max_image_scan_skew_sec:.6f}s"
            ),
            **fields,
        )
    if info_image_skew > selected.max_camera_info_image_skew_sec:
        return _result(
            selected,
            evidence,
            ready=False,
            failure_code=FAILURE_CAMERA_INFO_IMAGE_SKEW,
            detail=(
                "no fresh CameraInfo/image pair met the pre-motion skew "
                f"limit: nearest={info_image_skew:.6f}s > "
                f"{selected.max_camera_info_image_skew_sec:.6f}s"
            ),
            **fields,
        )
    return _result(
        selected,
        evidence,
        ready=False,
        failure_code=FAILURE_FRESH_TUPLE_UNAVAILABLE,
        detail=(
            "the bounded header window contained current streams but no "
            "single image/CameraInfo/LaserScan tuple met all freshness and "
            "skew limits"
        ),
        **fields,
    )


__all__ = [
    "DEFAULT_MAX_CAMERA_INFO_AGE_SEC",
    "DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC",
    "DEFAULT_MAX_FUTURE_TIMESTAMP_SEC",
    "DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC",
    "DEFAULT_MAX_SENSOR_AGE_SEC",
    "DEFAULT_SENSOR_TIMING_SAMPLE_CAPACITY",
    "DEFAULT_SENSOR_TIMING_TIMEOUT_SEC",
    "FAILURE_CAMERA_INFO_FRAME_EMPTY",
    "FAILURE_CAMERA_INFO_FRAME_MISMATCH",
    "FAILURE_CAMERA_INFO_IMAGE_SKEW",
    "FAILURE_CAMERA_INFO_STAMP_FUTURE",
    "FAILURE_CAMERA_INFO_STAMP_INVALID",
    "FAILURE_CAMERA_INFO_STAMP_STALE",
    "FAILURE_CAMERA_INFO_TIMEOUT",
    "FAILURE_IMAGE_FRAME_EMPTY",
    "FAILURE_IMAGE_FRAME_MISMATCH",
    "FAILURE_IMAGE_SCAN_SKEW",
    "FAILURE_IMAGE_STAMP_FUTURE",
    "FAILURE_IMAGE_STAMP_INVALID",
    "FAILURE_IMAGE_STAMP_STALE",
    "FAILURE_IMAGE_TIMEOUT",
    "FAILURE_FRESH_TUPLE_UNAVAILABLE",
    "FAILURE_OBSERVATION_EFFECT",
    "FAILURE_OBSERVER_CLOCK",
    "FAILURE_ROS_UNAVAILABLE",
    "FAILURE_SCAN_FRAME_EMPTY",
    "FAILURE_SCAN_FRAME_MISMATCH",
    "FAILURE_SCAN_STAMP_FUTURE",
    "FAILURE_SCAN_STAMP_INVALID",
    "FAILURE_SCAN_STAMP_STALE",
    "FAILURE_SCAN_TIMEOUT",
    "HeaderSample",
    "SENSOR_TIMING_READINESS_SCHEMA_VERSION",
    "SensorTimingEffect",
    "SensorTimingEvidence",
    "SensorTimingReadinessConfig",
    "SensorTimingReadinessError",
    "SensorTimingReadinessResult",
    "evaluate_sensor_timing_readiness",
]
