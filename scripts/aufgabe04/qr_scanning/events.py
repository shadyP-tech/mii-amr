"""Strict, ROS-free QR observation and station-identity contracts.

The camera node remains a passive producer.  Consumers must call
``validate_qr_observation`` before a scan is allowed to affect mission state.
The validation deliberately keeps wall-clock freshness, evidence binding, and
semantic identity separate from OpenCV and ROS message handling.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity as PersistedStationIdentity,
    StationIdentityRegistry as PersistedStationIdentityRegistry,
    validate_station_identity_registry,
)


Point2D = Tuple[float, float]
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SOURCE_LABEL_RE = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is not a valid identifier: {value!r}")
    return normalized


def _require_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_source_label(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not _SOURCE_LABEL_RE.fullmatch(normalized) or ".." in normalized:
        raise ValueError(f"{field_name} is not a valid source/frame label: {value!r}")
    return normalized


def _require_finite(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedStationIdentity:
    """Read-only QR-facing view of one persisted identity mapping."""

    persisted: PersistedStationIdentity

    @property
    def candidate_uid(self) -> str:
        return self.persisted.candidate_uid

    @property
    def qr_id(self) -> str:
        return self.persisted.qr_id

    @property
    def server_station_id(self) -> str:
        return self.persisted.server_station_id

    @property
    def station_id(self) -> str:
        """Compatibility name used by the controller for the server station."""

        return self.persisted.server_station_id


class StationIdentityRegistry:
    """QR-facing resolver backed only by a persisted station registry.

    This adapter intentionally stores no independent mappings and performs no
    alias guessing. QR IDs, candidate UIDs, and server station IDs are resolved
    through their corresponding fields in the canonical persisted artifact.
    """

    def __init__(self, persisted_registry: PersistedStationIdentityRegistry):
        validate_station_identity_registry(persisted_registry)
        self._persisted_registry = persisted_registry

    @property
    def persisted_registry(self) -> PersistedStationIdentityRegistry:
        return self._persisted_registry

    @property
    def identities(self) -> Tuple[ResolvedStationIdentity, ...]:
        return tuple(
            ResolvedStationIdentity(identity)
            for identity in self._persisted_registry.mappings
        )

    def resolve(self, server_station_id: str) -> ResolvedStationIdentity:
        """Resolve an exact server station ID; never reinterpret a QR alias."""

        identity = self._persisted_registry.for_server_station(server_station_id)
        if identity is None:
            raise ValueError(f"unknown server station identity: {server_station_id}")
        return ResolvedStationIdentity(identity)

    def validate_fields(
        self,
        *,
        qr_id: str,
        station_id: str,
        candidate_uid: str,
    ) -> ResolvedStationIdentity:
        qr_identity = self._persisted_registry.for_qr(qr_id)
        if qr_identity is None:
            raise ValueError(f"unknown QR identity: {qr_id}")
        station_identity = self._persisted_registry.for_server_station(station_id)
        if station_identity is None:
            raise ValueError(f"unknown server station identity: {station_id}")
        candidate_identity = self._persisted_registry.for_candidate(candidate_uid)
        if candidate_identity is None:
            raise ValueError(f"unknown candidate identity: {candidate_uid}")
        if not (qr_identity == station_identity == candidate_identity):
            raise ValueError(
                "QR identity fields disagree: "
                f"qr_id={qr_id!r}, station_id={station_id!r}, candidate_uid={candidate_uid!r}"
            )
        return ResolvedStationIdentity(qr_identity)

    def canonical_station_order(self, station_ids: Iterable[str]) -> Tuple[str, ...]:
        """Validate server station IDs while preserving every ordered element."""

        return tuple(self.resolve(station_id).server_station_id for station_id in station_ids)


def geometry_sha256(
    *,
    image_width_px: int,
    image_height_px: int,
    corners_px: Sequence[Point2D],
) -> str:
    return _canonical_sha256(
        {
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
            "corners_px": [[float(x), float(y)] for x, y in corners_px],
        }
    )


@dataclass(frozen=True)
class QRGeometryEvidence:
    image_width_px: int
    image_height_px: int
    corners_px: Tuple[Point2D, Point2D, Point2D, Point2D]
    geometry_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.image_width_px, bool) or not isinstance(self.image_width_px, int):
            raise ValueError("image_width_px must be an integer")
        if isinstance(self.image_height_px, bool) or not isinstance(self.image_height_px, int):
            raise ValueError("image_height_px must be an integer")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.corners_px) != 4:
            raise ValueError("QR geometry must contain exactly four corners")

        normalized = []
        for index, point in enumerate(self.corners_px):
            if len(point) != 2:
                raise ValueError(f"corner {index} must contain x and y")
            x = _require_finite(point[0], f"corners_px[{index}].x")
            y = _require_finite(point[1], f"corners_px[{index}].y")
            if x < 0.0 or x > float(self.image_width_px - 1):
                raise ValueError(f"corner {index} x coordinate is outside the image")
            if y < 0.0 or y > float(self.image_height_px - 1):
                raise ValueError(f"corner {index} y coordinate is outside the image")
            normalized.append((x, y))
        normalized_corners = tuple(normalized)
        object.__setattr__(self, "corners_px", normalized_corners)

        twice_area = 0.0
        corner_cross_products = []
        for index, (x1, y1) in enumerate(normalized_corners):
            x2, y2 = normalized_corners[(index + 1) % len(normalized_corners)]
            twice_area += x1 * y2 - x2 * y1
            x3, y3 = normalized_corners[(index + 2) % len(normalized_corners)]
            corner_cross_products.append((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2))
        if abs(twice_area) <= 1.0:
            raise ValueError("QR geometry quadrilateral has negligible area")
        if any(abs(value) <= 1.0e-9 for value in corner_cross_products):
            raise ValueError("QR geometry has collinear adjacent corners")
        if not (
            all(value > 0.0 for value in corner_cross_products)
            or all(value < 0.0 for value in corner_cross_products)
        ):
            raise ValueError("QR geometry corners must form an ordered convex quadrilateral")

        digest = _require_sha256(self.geometry_sha256, "geometry_sha256")
        expected = geometry_sha256(
            image_width_px=self.image_width_px,
            image_height_px=self.image_height_px,
            corners_px=normalized_corners,
        )
        if digest != expected:
            raise ValueError("geometry_sha256 does not match QR geometry")

    @classmethod
    def create(
        cls,
        *,
        image_width_px: int,
        image_height_px: int,
        corners_px: Sequence[Point2D],
    ) -> "QRGeometryEvidence":
        corners = tuple((float(point[0]), float(point[1])) for point in corners_px)
        return cls(
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            corners_px=corners,  # type: ignore[arg-type]
            geometry_sha256=geometry_sha256(
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                corners_px=corners,
            ),
        )


def consensus_sha256(
    *,
    qr_id: str,
    sample_ids: Sequence[str],
    agreeing_sample_ids: Sequence[str],
    window_start_sec: float,
    window_end_sec: float,
) -> str:
    return _canonical_sha256(
        {
            "qr_id": _require_identifier(qr_id, "qr_id").casefold(),
            "sample_ids": list(sample_ids),
            "agreeing_sample_ids": list(agreeing_sample_ids),
            "window_start_sec": float(window_start_sec),
            "window_end_sec": float(window_end_sec),
        }
    )


@dataclass(frozen=True)
class QRConsensusEvidence:
    qr_id: str
    sample_ids: Tuple[str, ...]
    agreeing_sample_ids: Tuple[str, ...]
    window_start_sec: float
    window_end_sec: float
    consensus_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "qr_id", _require_identifier(self.qr_id, "consensus qr_id"))
        sample_ids = tuple(_require_identifier(item, "sample_id") for item in self.sample_ids)
        agreeing_ids = tuple(
            _require_identifier(item, "agreeing_sample_id") for item in self.agreeing_sample_ids
        )
        if not sample_ids:
            raise ValueError("QR consensus requires at least one sample")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("QR consensus sample_ids must be unique")
        if len(set(agreeing_ids)) != len(agreeing_ids):
            raise ValueError("QR consensus agreeing_sample_ids must be unique")
        if not agreeing_ids:
            raise ValueError("QR consensus requires at least one agreeing sample")
        unknown = set(agreeing_ids) - set(sample_ids)
        if unknown:
            raise ValueError("agreeing_sample_ids must be a subset of sample_ids")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "agreeing_sample_ids", agreeing_ids)

        start = _require_finite(self.window_start_sec, "window_start_sec")
        end = _require_finite(self.window_end_sec, "window_end_sec")
        if start < 0.0 or end < start:
            raise ValueError("QR consensus window is invalid")
        object.__setattr__(self, "window_start_sec", start)
        object.__setattr__(self, "window_end_sec", end)

        digest = _require_sha256(self.consensus_sha256, "consensus_sha256")
        expected = consensus_sha256(
            qr_id=self.qr_id,
            sample_ids=sample_ids,
            agreeing_sample_ids=agreeing_ids,
            window_start_sec=start,
            window_end_sec=end,
        )
        if digest != expected:
            raise ValueError("consensus_sha256 does not match QR consensus evidence")

    @property
    def agreement_ratio(self) -> float:
        return len(self.agreeing_sample_ids) / float(len(self.sample_ids))

    @classmethod
    def create(
        cls,
        *,
        qr_id: str,
        sample_ids: Sequence[str],
        agreeing_sample_ids: Sequence[str],
        window_start_sec: float,
        window_end_sec: float,
    ) -> "QRConsensusEvidence":
        samples = tuple(sample_ids)
        agreeing = tuple(agreeing_sample_ids)
        return cls(
            qr_id=qr_id,
            sample_ids=samples,
            agreeing_sample_ids=agreeing,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            consensus_sha256=consensus_sha256(
                qr_id=qr_id,
                sample_ids=samples,
                agreeing_sample_ids=agreeing,
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
            ),
        )


@dataclass(frozen=True)
class QRObservationEvent:
    event_id: str
    robot_id: str
    qr_id: str
    station_id: str
    candidate_uid: str
    observed_at_sec: float
    received_at_sec: float
    clock_id: str
    source: str
    source_frame_id: str
    confidence: float
    geometry: QRGeometryEvidence
    consensus: QRConsensusEvidence
    calibration_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "event_id",
            "robot_id",
            "qr_id",
            "station_id",
            "candidate_uid",
            "clock_id",
        ):
            object.__setattr__(self, field_name, _require_identifier(getattr(self, field_name), field_name))
        for field_name in ("source", "source_frame_id"):
            object.__setattr__(
                self,
                field_name,
                _require_source_label(getattr(self, field_name), field_name),
            )
        observed = _require_finite(self.observed_at_sec, "observed_at_sec")
        received = _require_finite(self.received_at_sec, "received_at_sec")
        if observed < 0.0 or received < 0.0:
            raise ValueError("QR observation timestamps must be non-negative")
        if observed > received:
            raise ValueError("QR observation cannot be received before it was observed")
        object.__setattr__(self, "observed_at_sec", observed)
        object.__setattr__(self, "received_at_sec", received)
        confidence = _require_finite(self.confidence, "confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)
        _require_sha256(self.calibration_sha256, "calibration_sha256")
        if self.consensus.qr_id.casefold() != self.qr_id.casefold():
            raise ValueError("QR observation and consensus qr_id disagree")
        if self.consensus.window_end_sec > self.observed_at_sec:
            raise ValueError("QR consensus ends after the observation timestamp")


@dataclass(frozen=True)
class QRValidationPolicy:
    max_observation_age_sec: float = 1.0
    max_future_skew_sec: float = 0.1
    max_receive_latency_sec: float = 1.0
    min_confidence: float = 0.8
    min_consensus_samples: int = 3
    min_consensus_ratio: float = 2.0 / 3.0
    expected_calibration_sha256: Optional[str] = None
    expected_clock_id: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_observation_age_sec",
            "max_future_skew_sec",
            "max_receive_latency_sec",
            "min_confidence",
            "min_consensus_ratio",
        ):
            value = _require_finite(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        if self.max_observation_age_sec <= 0.0:
            raise ValueError("max_observation_age_sec must be positive")
        if self.max_future_skew_sec < 0.0 or self.max_receive_latency_sec < 0.0:
            raise ValueError("future skew and receive latency limits must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if isinstance(self.min_consensus_samples, bool) or self.min_consensus_samples <= 0:
            raise ValueError("min_consensus_samples must be positive")
        if not 0.0 < self.min_consensus_ratio <= 1.0:
            raise ValueError("min_consensus_ratio must be in (0, 1]")
        if self.expected_calibration_sha256 is not None:
            _require_sha256(self.expected_calibration_sha256, "expected_calibration_sha256")
        if self.expected_clock_id is not None:
            object.__setattr__(
                self,
                "expected_clock_id",
                _require_identifier(self.expected_clock_id, "expected_clock_id"),
            )


@dataclass(frozen=True)
class ValidatedQRObservation:
    event: QRObservationEvent
    identity: ResolvedStationIdentity
    age_sec: float
    receive_latency_sec: float
    consensus_ratio: float


def validate_qr_observation(
    event: QRObservationEvent,
    *,
    registry: StationIdentityRegistry,
    now_sec: float,
    policy: QRValidationPolicy = QRValidationPolicy(),
    expected_robot_id: Optional[str] = None,
    seen_event_ids: Iterable[str] = (),
) -> ValidatedQRObservation:
    """Validate freshness, replay, identity, calibration, and evidence quality."""

    now = _require_finite(now_sec, "now_sec")
    if expected_robot_id is not None and event.robot_id != expected_robot_id:
        raise ValueError(
            f"QR observation robot_id {event.robot_id!r} does not match {expected_robot_id!r}"
        )
    if policy.expected_clock_id is not None and event.clock_id != policy.expected_clock_id:
        raise ValueError(
            f"QR observation clock_id {event.clock_id!r} does not match "
            f"{policy.expected_clock_id!r}"
        )
    if event.event_id in set(seen_event_ids):
        raise ValueError(f"replayed QR observation event_id: {event.event_id}")
    if event.observed_at_sec > now + policy.max_future_skew_sec:
        raise ValueError("QR observation timestamp is in the future")
    if event.received_at_sec > now + policy.max_future_skew_sec:
        raise ValueError("QR receipt timestamp is in the future")
    age = now - event.observed_at_sec
    if age > policy.max_observation_age_sec:
        raise ValueError("QR observation is stale")
    receive_latency = event.received_at_sec - event.observed_at_sec
    if receive_latency > policy.max_receive_latency_sec:
        raise ValueError("QR observation receive latency exceeds the configured limit")
    if event.confidence < policy.min_confidence:
        raise ValueError("QR observation confidence is below the configured threshold")
    if len(event.consensus.sample_ids) < policy.min_consensus_samples:
        raise ValueError("QR observation has too few consensus samples")
    if event.consensus.agreement_ratio < policy.min_consensus_ratio:
        raise ValueError("QR observation consensus ratio is below the configured threshold")
    if (
        policy.expected_calibration_sha256 is not None
        and event.calibration_sha256 != policy.expected_calibration_sha256
    ):
        raise ValueError("QR observation calibration hash does not match the active calibration")

    identity = registry.validate_fields(
        qr_id=event.qr_id,
        station_id=event.station_id,
        candidate_uid=event.candidate_uid,
    )
    return ValidatedQRObservation(
        event=event,
        identity=identity,
        age_sec=age,
        receive_latency_sec=receive_latency,
        consensus_ratio=event.consensus.agreement_ratio,
    )
