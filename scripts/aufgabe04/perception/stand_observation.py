"""Map-frame stand observations for Aufgabe 04 LiDAR station discovery."""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.perception.models import StandCandidate


OBSERVATION_SCHEMA_VERSION = 2
LEGACY_OBSERVATION_SCHEMA_VERSION = 1
OBSERVER_CLOCK_ROS_SYSTEM_TIME = "ros_system_time"
OBSERVER_CLOCK_ROS_SIM_TIME = "ros_sim_time"
VALID_OBSERVER_CLOCKS = frozenset(
    {OBSERVER_CLOCK_ROS_SYSTEM_TIME, OBSERVER_CLOCK_ROS_SIM_TIME}
)
TF_LOOKUP_MODE_SCAN_TIME_EXACT = "scan_time_exact"
RUNTIME_TIMING_LIMITS_KEY = "observation_timing_limits"
OBSERVATION_ID_SCOPE_RUNTIME_KEY = "observation_id_scope"

_OBSERVATION_ID_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ObservationTimingLimits:
    """Admission limits shared by observation producers and consumers."""

    max_scan_age_sec: float = 1.0
    max_future_timestamp_sec: float = 0.25
    max_tf_age_sec: float = 1.0
    max_tf_scan_skew_sec: float = 0.02

    def validated(self) -> "ObservationTimingLimits":
        _finite_positive(self.max_scan_age_sec, "maximum scan age")
        _finite_nonnegative(
            self.max_future_timestamp_sec, "maximum future timestamp skew"
        )
        _finite_positive(self.max_tf_age_sec, "maximum TF age")
        _finite_nonnegative(
            self.max_tf_scan_skew_sec, "maximum TF/scan timestamp skew"
        )
        return self

    def as_dict(self) -> dict[str, float]:
        self.validated()
        return {
            "max_scan_age_sec": float(self.max_scan_age_sec),
            "max_future_timestamp_sec": float(self.max_future_timestamp_sec),
            "max_tf_age_sec": float(self.max_tf_age_sec),
            "max_tf_scan_skew_sec": float(self.max_tf_scan_skew_sec),
        }


DEFAULT_OBSERVATION_TIMING_LIMITS = ObservationTimingLimits()


@dataclass(frozen=True)
class PlanarTransform:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class ObservationProvenance:
    schema_version: int
    observer_version: str
    resolved_scan_topic: str
    scan_frame: str
    map_frame: str
    base_frame: str
    localization_source: str
    scan_stamp_sec: float
    tf_lookup_stamp_sec: float
    tf_age_sec: float
    runtime_config: Mapping[str, object]
    observer_clock: str = ""
    observer_clock_sec: float = 0.0
    scan_age_sec: float = 0.0
    tf_scan_skew_sec: float = 0.0
    tf_query_stamp_sec: float = 0.0
    tf_lookup_mode: str = ""
    map_yaml: str = ""
    map_yaml_sha256: str = ""
    map_image_sha256: str = ""
    map_bundle_sha256: str = ""


@dataclass(frozen=True)
class StandObservation:
    observation_id: str
    candidate_id: str
    x_m: float
    y_m: float
    bearing_rad: float
    distance_m: float
    approximate_width_m: float
    point_count: int
    confidence: float
    observed_at_sec: float
    provenance: ObservationProvenance


@dataclass(frozen=True)
class ObservationTiming:
    """Validated timing measurements in the observer's ROS clock domain."""

    scan_age_sec: float
    tf_age_sec: float
    tf_scan_skew_sec: float


def observer_clock_name(*, use_sim_time: bool) -> str:
    return (
        OBSERVER_CLOCK_ROS_SIM_TIME
        if use_sim_time
        else OBSERVER_CLOCK_ROS_SYSTEM_TIME
    )


def validated_provenance_observer_clock(
    provenance: ObservationProvenance,
    *,
    required_observer_clock: str | None = None,
) -> str:
    """Validate the explicit clock marker against the recorded ROS config."""

    if provenance.schema_version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported observation schema_version: {provenance.schema_version}"
        )
    if provenance.observer_clock not in VALID_OBSERVER_CLOCKS:
        raise ValueError("observation observer clock is invalid")
    if (
        required_observer_clock is not None
        and provenance.observer_clock != required_observer_clock
    ):
        raise ValueError("observation observer clock mismatch")
    use_sim_time = provenance.runtime_config.get("use_sim_time")
    if type(use_sim_time) is not bool:
        raise ValueError("observation runtime use_sim_time is invalid")
    if provenance.observer_clock != observer_clock_name(use_sim_time=use_sim_time):
        raise ValueError("observation observer clock/use_sim_time mismatch")
    return provenance.observer_clock


def validated_observation_stream_clock(
    observations: Iterable[StandObservation],
    *,
    required_observer_clock: str | None = None,
) -> str:
    """Reject mixed schema/clock evidence in one append-mode JSONL stream."""

    stream_clock = ""
    for observation in observations:
        clock = validated_provenance_observer_clock(
            observation.provenance,
            required_observer_clock=required_observer_clock,
        )
        if stream_clock and clock != stream_clock:
            raise ValueError("observation artifact mixes incompatible observer clocks")
        stream_clock = clock
    return stream_clock


def observation_timing_limits_from_runtime_config(
    runtime_config: Mapping[str, object],
) -> ObservationTimingLimits:
    """Load and validate the producer's persisted temporal admission policy."""

    payload = runtime_config.get(RUNTIME_TIMING_LIMITS_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"runtime_config.{RUNTIME_TIMING_LIMITS_KEY} must be an object"
        )
    return ObservationTimingLimits(
        max_scan_age_sec=_require_number(payload, "max_scan_age_sec"),
        max_future_timestamp_sec=_require_number(
            payload, "max_future_timestamp_sec"
        ),
        max_tf_age_sec=_require_number(payload, "max_tf_age_sec"),
        max_tf_scan_skew_sec=_require_number(
            payload, "max_tf_scan_skew_sec"
        ),
    ).validated()


def validated_scan_age_sec(
    *,
    observer_clock_sec: float,
    scan_stamp_sec: float,
    max_scan_age_sec: float,
    max_future_timestamp_sec: float,
) -> float:
    """Return signed scan age after rejecting invalid clock-domain evidence."""

    observer_clock_sec = _finite_positive(
        observer_clock_sec, "observer clock timestamp"
    )
    scan_stamp_sec = _finite_positive(scan_stamp_sec, "scan timestamp")
    max_scan_age_sec = _finite_positive(max_scan_age_sec, "maximum scan age")
    max_future_timestamp_sec = _finite_nonnegative(
        max_future_timestamp_sec, "maximum future timestamp skew"
    )
    scan_age_sec = observer_clock_sec - scan_stamp_sec
    if scan_age_sec > max_scan_age_sec:
        raise ValueError("scan timestamp is stale")
    if scan_age_sec < -max_future_timestamp_sec:
        raise ValueError("scan timestamp is in the future")
    return scan_age_sec


def validated_observation_timing(
    *,
    observer_clock_sec: float,
    scan_stamp_sec: float,
    tf_stamp_sec: float,
    max_scan_age_sec: float,
    max_future_timestamp_sec: float,
    max_tf_age_sec: float,
    max_tf_scan_skew_sec: float,
) -> ObservationTiming:
    """Validate scan and map<-scan TF stamps in one ROS clock domain.

    The map-to-scan chain contains the moving robot pose, so timestamp-zero TF
    is not accepted as a generic "static" escape hatch. A transform requested
    at the scan stamp must report a positive, nearby timestamp.
    """

    scan_age_sec = validated_scan_age_sec(
        observer_clock_sec=observer_clock_sec,
        scan_stamp_sec=scan_stamp_sec,
        max_scan_age_sec=max_scan_age_sec,
        max_future_timestamp_sec=max_future_timestamp_sec,
    )
    observer_clock_sec = _finite_positive(
        observer_clock_sec, "observer clock timestamp"
    )
    scan_stamp_sec = _finite_positive(scan_stamp_sec, "scan timestamp")
    tf_stamp_sec = _finite_positive(tf_stamp_sec, "TF timestamp")
    max_tf_age_sec = _finite_positive(max_tf_age_sec, "maximum TF age")
    max_tf_scan_skew_sec = _finite_nonnegative(
        max_tf_scan_skew_sec, "maximum TF/scan timestamp skew"
    )
    max_future_timestamp_sec = _finite_nonnegative(
        max_future_timestamp_sec, "maximum future timestamp skew"
    )

    signed_tf_age_sec = observer_clock_sec - tf_stamp_sec
    if signed_tf_age_sec > max_tf_age_sec:
        raise ValueError("TF timestamp is stale")
    if signed_tf_age_sec < -max_future_timestamp_sec:
        raise ValueError("TF timestamp is in the future")
    tf_scan_skew_sec = abs(tf_stamp_sec - scan_stamp_sec)
    if tf_scan_skew_sec > max_tf_scan_skew_sec:
        raise ValueError("TF/scan timestamp skew exceeds limit")
    return ObservationTiming(
        scan_age_sec=scan_age_sec,
        tf_age_sec=signed_tf_age_sec,
        tf_scan_skew_sec=tf_scan_skew_sec,
    )


def transform_point(x_m: float, y_m: float, transform: PlanarTransform) -> tuple[float, float]:
    cos_yaw = math.cos(transform.yaw_rad)
    sin_yaw = math.sin(transform.yaw_rad)
    return (
        transform.x_m + cos_yaw * x_m - sin_yaw * y_m,
        transform.y_m + sin_yaw * x_m + cos_yaw * y_m,
    )


def validated_observation_id_scope(scope: object) -> str | None:
    """Return a safe explicit identity scope, or ``None`` for legacy IDs."""

    if scope is None:
        return None
    if not isinstance(scope, str) or _OBSERVATION_ID_SCOPE_RE.fullmatch(scope) is None:
        raise ValueError(
            "observation ID scope must be a 1-64 character safe identifier "
            "starting with a letter or digit and containing only letters, "
            "digits, '.', '_' or '-'"
        )
    return scope


def observation_id_from_index(
    observation_index: int,
    *,
    observation_id_scope: str | None = None,
) -> str:
    """Build a stable observation ID with an optional process/epoch scope."""

    if isinstance(observation_index, bool) or not isinstance(observation_index, int):
        raise ValueError("observation index must be an integer")
    if observation_index <= 0:
        raise ValueError("observation index must be positive")
    scope = validated_observation_id_scope(observation_id_scope)
    legacy_id = f"stand_observation_{observation_index:06d}"
    if scope is None:
        return legacy_id
    return f"stand_observation_{scope}_{observation_index:06d}"


def observation_from_candidate(
    candidate: StandCandidate,
    *,
    transform_scan_to_map: PlanarTransform,
    observed_at_sec: float,
    provenance: ObservationProvenance,
    observation_index: int,
    observation_id_scope: str | None = None,
) -> StandObservation:
    x_m, y_m = transform_point(
        candidate.center_x_m,
        candidate.center_y_m,
        transform_scan_to_map,
    )
    return StandObservation(
        observation_id=observation_id_from_index(
            observation_index,
            observation_id_scope=observation_id_scope,
        ),
        candidate_id=candidate.candidate_id,
        x_m=x_m,
        y_m=y_m,
        bearing_rad=candidate.bearing_rad + transform_scan_to_map.yaw_rad,
        distance_m=candidate.distance_m,
        approximate_width_m=candidate.approximate_width_m,
        point_count=candidate.point_count,
        confidence=candidate.confidence,
        observed_at_sec=observed_at_sec,
        provenance=provenance,
    )


def observations_from_candidates(
    candidates: Iterable[StandCandidate],
    *,
    transform_scan_to_map: PlanarTransform,
    observed_at_sec: float,
    provenance: ObservationProvenance,
    start_index: int = 1,
    observation_id_scope: str | None = None,
) -> tuple[StandObservation, ...]:
    observation_id_scope = validated_observation_id_scope(observation_id_scope)
    return tuple(
        observation_from_candidate(
            candidate,
            transform_scan_to_map=transform_scan_to_map,
            observed_at_sec=observed_at_sec,
            provenance=provenance,
            observation_index=start_index + index,
            observation_id_scope=observation_id_scope,
        )
        for index, candidate in enumerate(candidates)
    )


def observation_to_payload(observation: StandObservation) -> dict[str, object]:
    return asdict(observation)


def observation_from_payload(payload: Mapping[str, object]) -> StandObservation:
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("observation provenance must be an object")
    return StandObservation(
        observation_id=_require_str(payload, "observation_id"),
        candidate_id=_require_str(payload, "candidate_id"),
        x_m=_require_number(payload, "x_m"),
        y_m=_require_number(payload, "y_m"),
        bearing_rad=_require_number(payload, "bearing_rad"),
        distance_m=_require_number(payload, "distance_m"),
        approximate_width_m=_require_number(payload, "approximate_width_m"),
        point_count=_require_integer(payload, "point_count"),
        confidence=_require_number(payload, "confidence"),
        observed_at_sec=_require_number(payload, "observed_at_sec"),
        provenance=provenance_from_payload(provenance_payload),
    )


def provenance_from_payload(payload: Mapping[str, object]) -> ObservationProvenance:
    runtime_config = payload.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise ValueError("provenance.runtime_config must be an object")
    schema_version = _require_integer(payload, "schema_version")
    timing_required = schema_version >= OBSERVATION_SCHEMA_VERSION
    return ObservationProvenance(
        schema_version=schema_version,
        observer_version=_require_str(payload, "observer_version"),
        resolved_scan_topic=_require_str(payload, "resolved_scan_topic"),
        scan_frame=_require_str(payload, "scan_frame"),
        map_frame=_require_str(payload, "map_frame"),
        base_frame=_require_str(payload, "base_frame"),
        localization_source=_require_str(payload, "localization_source"),
        scan_stamp_sec=_require_number(payload, "scan_stamp_sec"),
        tf_lookup_stamp_sec=_require_number(payload, "tf_lookup_stamp_sec"),
        tf_age_sec=_require_number(payload, "tf_age_sec"),
        runtime_config=dict(runtime_config),
        observer_clock=(
            _require_str(payload, "observer_clock") if timing_required else ""
        ),
        observer_clock_sec=(
            _require_number(payload, "observer_clock_sec")
            if timing_required
            else 0.0
        ),
        scan_age_sec=(
            _require_number(payload, "scan_age_sec") if timing_required else 0.0
        ),
        tf_scan_skew_sec=(
            _require_number(payload, "tf_scan_skew_sec")
            if timing_required
            else 0.0
        ),
        tf_query_stamp_sec=(
            _require_number(payload, "tf_query_stamp_sec")
            if timing_required
            else 0.0
        ),
        tf_lookup_mode=(
            _require_str(payload, "tf_lookup_mode") if timing_required else ""
        ),
        map_yaml=str(payload.get("map_yaml") or ""),
        map_yaml_sha256=str(payload.get("map_yaml_sha256") or ""),
        map_image_sha256=str(payload.get("map_image_sha256") or ""),
        map_bundle_sha256=str(payload.get("map_bundle_sha256") or ""),
    )


def write_observation_jsonl(path: Path, observations: Iterable[StandObservation]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = tuple(
        json.dumps(
            observation_to_payload(observation),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for observation in observations
    )
    with path.open("a") as file:
        file.writelines(lines)


def load_observation_jsonl(path: Path) -> tuple[StandObservation, ...]:
    observations, _sha256 = load_observation_jsonl_snapshot(path)
    return observations


def load_observation_jsonl_snapshot(
    path: Path,
) -> tuple[tuple[StandObservation, ...], str]:
    """Parse and hash one immutable read of an observation JSONL artifact.

    Returning the digest from the same bytes that were parsed prevents a
    candidate snapshot from claiming ancestry from a file revision different
    from the observations that actually produced its geometry.
    """

    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("observation JSONL must be UTF-8") from exc
    return _parse_observation_jsonl(text), hashlib.sha256(raw).hexdigest()


def _parse_observation_jsonl(text: str) -> tuple[StandObservation, ...]:
    observations = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(payload, Mapping):
                raise ValueError("line payload must be an object")
            observations.append(observation_from_payload(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid observation JSONL line {line_number}: {exc}") from exc
    return tuple(observations)


def _require_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _require_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON value {value!r}")


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
