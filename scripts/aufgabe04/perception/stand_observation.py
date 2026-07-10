"""Map-frame stand observations for Aufgabe 04 LiDAR station discovery."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.perception.models import StandCandidate


OBSERVATION_SCHEMA_VERSION = 1


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
    map_yaml: str = ""
    map_yaml_sha256: str = ""


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


def transform_point(x_m: float, y_m: float, transform: PlanarTransform) -> tuple[float, float]:
    cos_yaw = math.cos(transform.yaw_rad)
    sin_yaw = math.sin(transform.yaw_rad)
    return (
        transform.x_m + cos_yaw * x_m - sin_yaw * y_m,
        transform.y_m + sin_yaw * x_m + cos_yaw * y_m,
    )


def observation_from_candidate(
    candidate: StandCandidate,
    *,
    transform_scan_to_map: PlanarTransform,
    observed_at_sec: float,
    provenance: ObservationProvenance,
    observation_index: int,
) -> StandObservation:
    x_m, y_m = transform_point(
        candidate.center_x_m,
        candidate.center_y_m,
        transform_scan_to_map,
    )
    return StandObservation(
        observation_id=f"stand_observation_{observation_index:06d}",
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
) -> tuple[StandObservation, ...]:
    return tuple(
        observation_from_candidate(
            candidate,
            transform_scan_to_map=transform_scan_to_map,
            observed_at_sec=observed_at_sec,
            provenance=provenance,
            observation_index=start_index + index,
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
        point_count=int(_require_number(payload, "point_count")),
        confidence=_require_number(payload, "confidence"),
        observed_at_sec=_require_number(payload, "observed_at_sec"),
        provenance=provenance_from_payload(provenance_payload),
    )


def provenance_from_payload(payload: Mapping[str, object]) -> ObservationProvenance:
    runtime_config = payload.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise ValueError("provenance.runtime_config must be an object")
    return ObservationProvenance(
        schema_version=int(_require_number(payload, "schema_version")),
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
        map_yaml=str(payload.get("map_yaml") or ""),
        map_yaml_sha256=str(payload.get("map_yaml_sha256") or ""),
    )


def write_observation_jsonl(path: Path, observations: Iterable[StandObservation]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        for observation in observations:
            file.write(json.dumps(observation_to_payload(observation), sort_keys=True) + "\n")


def load_observation_jsonl(path: Path) -> tuple[StandObservation, ...]:
    observations = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
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
    return float(value)
