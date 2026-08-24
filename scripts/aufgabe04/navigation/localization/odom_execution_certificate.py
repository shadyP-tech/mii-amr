"""Immutable dual-frame execution certificates and planar transforms.

``map_from_odom`` always means the transform whose value satisfies::

    p_map = R(map_from_odom.yaw_rad) @ p_odom
            + (map_from_odom.x_m, map_from_odom.y_m)

The route conversion helpers deliberately use the inverse of that transform
when converting a map-frame route for an odom-frame controller.  Everything in
this module is ROS-free so transform direction, route identity, and artifact
integrity can be tested before any motion-capable runtime imports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


ODOM_EXECUTION_CERTIFICATE_SCHEMA_VERSION = 1
POSE_ROUTE_HASH_SCHEMA_VERSION = 1

MAP_FROM_ODOM_CONVENTION = "p_map = R(yaw_rad) * p_odom + (x_m, y_m)"

_HASH_FIELD = "odom_execution_certificate_sha256"
_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version",
        "source_map_route_sha256",
        "source_map_execution_certificate_sha256",
        "transformed_odom_route_sha256",
        "map_frame",
        "odom_frame",
        "base_frame",
        "map_from_odom",
        "transform_stamp_sec",
        "transform_capture_time_sec",
        "waypoint_count",
        "tracking_tube_radius_m",
        "command_owner",
        "uncertainty_budget_sha256",
        "ambiguity_evidence_sha256",
    }
)
_TRANSFORM_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})


def normalize_yaw(yaw_rad: float) -> float:
    """Return a finite angle in the half-open interval ``[-pi, pi)``."""

    yaw = _finite_number(yaw_rad, "yaw_rad")
    normalized = (yaw + math.pi) % math.tau - math.pi
    # Modulo reduction of (canonical_yaw + 2*pi) can leave a few ulps of
    # arithmetic noise.  Fifteen decimal places preserve far more angular
    # precision than the platform can execute while giving equivalent wrapped
    # angles one canonical JSON representation.
    normalized = round(normalized, 14)
    # Avoid distinct hashes for the two IEEE representations of zero.
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class PlanarTransform2D:
    """A planar rigid transform value.

    In an :class:`OdomExecutionCertificate`, this value is specifically
    ``map_from_odom`` and follows :data:`MAP_FROM_ODOM_CONVENTION`.
    """

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _canonical_finite(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _canonical_finite(self.y_m, "y_m"))
        object.__setattr__(self, "yaw_rad", normalize_yaw(self.yaw_rad))

    @property
    def translation_x_m(self) -> float:
        return self.x_m

    @property
    def translation_y_m(self) -> float:
        return self.y_m

    def inverse(self) -> "PlanarTransform2D":
        """Return the inverse rigid transform value."""

        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        return PlanarTransform2D(
            x_m=-cosine * self.x_m - sine * self.y_m,
            y_m=sine * self.x_m - cosine * self.y_m,
            yaw_rad=-self.yaw_rad,
        )


@dataclass(frozen=True)
class OdomExecutionCertificate:
    """Bind one map route to its transform-specific odom execution route."""

    source_map_route_sha256: str
    source_map_execution_certificate_sha256: str
    transformed_odom_route_sha256: str
    map_frame: str
    odom_frame: str
    base_frame: str
    map_from_odom: PlanarTransform2D
    transform_stamp_sec: float
    transform_capture_time_sec: float
    waypoint_count: int
    tracking_tube_radius_m: float
    command_owner: str
    uncertainty_budget_sha256: str
    ambiguity_evidence_sha256: str
    schema_version: int = ODOM_EXECUTION_CERTIFICATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_certificate(self, canonicalize=True)

    @property
    def captured_at_sec(self) -> float:
        """Compatibility spelling for the transform capture clock value."""

        return self.transform_capture_time_sec


def odom_pose_to_map(
    pose_odom: Pose2D,
    map_from_odom: PlanarTransform2D,
) -> Pose2D:
    """Apply ``map_from_odom`` to an odom-frame pose."""

    pose = _validated_pose(pose_odom, "pose_odom")
    transform = _validated_transform(map_from_odom)
    cosine = math.cos(transform.yaw_rad)
    sine = math.sin(transform.yaw_rad)
    return Pose2D(
        x_m=_canonical_zero(
            cosine * pose.x_m - sine * pose.y_m + transform.x_m
        ),
        y_m=_canonical_zero(
            sine * pose.x_m + cosine * pose.y_m + transform.y_m
        ),
        yaw_rad=_transform_pose_yaw(pose.yaw_rad, transform.yaw_rad),
    )


def map_pose_to_odom(
    pose_map: Pose2D,
    map_from_odom: PlanarTransform2D,
) -> Pose2D:
    """Inverse-transform a map-frame pose into the odom frame."""

    pose = _validated_pose(pose_map, "pose_map")
    transform = _validated_transform(map_from_odom)
    delta_x = pose.x_m - transform.x_m
    delta_y = pose.y_m - transform.y_m
    cosine = math.cos(transform.yaw_rad)
    sine = math.sin(transform.yaw_rad)
    return Pose2D(
        x_m=_canonical_zero(cosine * delta_x + sine * delta_y),
        y_m=_canonical_zero(-sine * delta_x + cosine * delta_y),
        yaw_rad=_transform_pose_yaw(pose.yaw_rad, -transform.yaw_rad),
    )


def transform_map_route_to_odom(
    map_route: Sequence[Pose2D],
    map_from_odom: PlanarTransform2D,
) -> tuple[Pose2D, ...]:
    """Return the immutable odom route produced by the inverse transform."""

    route = _validated_route(map_route, "map_route")
    transform = _validated_transform(map_from_odom)
    return tuple(map_pose_to_odom(pose, transform) for pose in route)


def transform_odom_route_to_map(
    odom_route: Sequence[Pose2D],
    map_from_odom: PlanarTransform2D,
) -> tuple[Pose2D, ...]:
    """Return the immutable map route produced by the forward transform."""

    route = _validated_route(odom_route, "odom_route")
    transform = _validated_transform(map_from_odom)
    return tuple(odom_pose_to_map(pose, transform) for pose in route)


def pose_route_sha256(route: Sequence[Pose2D]) -> str:
    """Hash canonical Pose2D values independently of a route file's bytes."""

    return payload_sha256(_pose_route_payload(route))


def canonical_pose_route_sha256(route: Sequence[Pose2D]) -> str:
    """Descriptive alias for :func:`pose_route_sha256`."""

    return pose_route_sha256(route)


def odom_execution_certificate_sha256(
    certificate: OdomExecutionCertificate,
) -> str:
    return payload_sha256(_certificate_payload(certificate))


def write_odom_execution_certificate(
    path: Path,
    certificate: OdomExecutionCertificate,
) -> str:
    """Immutably publish a canonical, content-hashed certificate."""

    try:
        return write_content_hashed_json(
            path,
            _certificate_payload(certificate),
            hash_field=_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_odom_execution_certificate(path: Path) -> OdomExecutionCertificate:
    """Strictly load and validate a content-hashed certificate."""

    try:
        payload = load_content_hashed_json(path, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _CERTIFICATE_FIELDS:
        raise ValueError("odom execution certificate fields mismatch")
    transform_payload = payload["map_from_odom"]
    if not isinstance(transform_payload, Mapping):
        raise ValueError("odom execution certificate map_from_odom must be an object")
    if frozenset(transform_payload) != _TRANSFORM_FIELDS:
        raise ValueError("odom execution certificate transform fields mismatch")

    try:
        raw_yaw = _number(transform_payload["yaw_rad"], "map_from_odom.yaw_rad")
        normalized_yaw = normalize_yaw(raw_yaw)
        if raw_yaw != normalized_yaw:
            raise ValueError("map_from_odom.yaw_rad must be normalized")
        return OdomExecutionCertificate(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            source_map_route_sha256=_string(
                payload["source_map_route_sha256"],
                "source_map_route_sha256",
            ),
            source_map_execution_certificate_sha256=_string(
                payload["source_map_execution_certificate_sha256"],
                "source_map_execution_certificate_sha256",
            ),
            transformed_odom_route_sha256=_string(
                payload["transformed_odom_route_sha256"],
                "transformed_odom_route_sha256",
            ),
            map_frame=_string(payload["map_frame"], "map_frame"),
            odom_frame=_string(payload["odom_frame"], "odom_frame"),
            base_frame=_string(payload["base_frame"], "base_frame"),
            map_from_odom=PlanarTransform2D(
                x_m=_number(transform_payload["x_m"], "map_from_odom.x_m"),
                y_m=_number(transform_payload["y_m"], "map_from_odom.y_m"),
                yaw_rad=raw_yaw,
            ),
            transform_stamp_sec=_number(
                payload["transform_stamp_sec"], "transform_stamp_sec"
            ),
            transform_capture_time_sec=_number(
                payload["transform_capture_time_sec"],
                "transform_capture_time_sec",
            ),
            waypoint_count=_integer(payload["waypoint_count"], "waypoint_count"),
            tracking_tube_radius_m=_number(
                payload["tracking_tube_radius_m"], "tracking_tube_radius_m"
            ),
            command_owner=_string(payload["command_owner"], "command_owner"),
            uncertainty_budget_sha256=_string(
                payload["uncertainty_budget_sha256"],
                "uncertainty_budget_sha256",
            ),
            ambiguity_evidence_sha256=_string(
                payload["ambiguity_evidence_sha256"],
                "ambiguity_evidence_sha256",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid odom execution certificate: {exc}") from exc


def validate_odom_execution_identity(
    certificate: OdomExecutionCertificate,
    *,
    source_map_route: Sequence[Pose2D],
    transformed_odom_route: Sequence[Pose2D],
    source_map_execution_certificate_sha256: str,
    map_frame: str,
    odom_frame: str,
    base_frame: str,
    tracking_tube_radius_m: float,
    command_owner: str,
    waypoint_count: int | None = None,
    map_from_odom: PlanarTransform2D | None = None,
    transform_stamp_sec: float | None = None,
    transform_capture_time_sec: float | None = None,
    uncertainty_budget_sha256: str | None = None,
    ambiguity_evidence_sha256: str | None = None,
) -> None:
    """Reject any runtime route, frame, transform, or policy identity drift."""

    _validate_certificate(certificate, canonicalize=False)
    source_route = _validated_route(source_map_route, "source_map_route")
    odom_route = _validated_route(transformed_odom_route, "transformed_odom_route")
    if len(source_route) != len(odom_route):
        raise ValueError("source and transformed route waypoint counts differ")

    observed_waypoint_count = (
        len(source_route) if waypoint_count is None else waypoint_count
    )
    if (
        not isinstance(observed_waypoint_count, int)
        or isinstance(observed_waypoint_count, bool)
        or observed_waypoint_count < 2
    ):
        raise ValueError("waypoint_count must be an integer >= 2")
    if observed_waypoint_count != len(source_route):
        raise ValueError(
            "runtime waypoint_count differs from supplied route waypoint count"
        )

    checks: dict[str, tuple[object, object]] = {
        "source_map_route_sha256": (
            certificate.source_map_route_sha256,
            pose_route_sha256(source_route),
        ),
        "source_map_execution_certificate_sha256": (
            certificate.source_map_execution_certificate_sha256,
            _require_sha256(
                source_map_execution_certificate_sha256,
                "source_map_execution_certificate_sha256",
            ),
        ),
        "transformed_odom_route_sha256": (
            certificate.transformed_odom_route_sha256,
            pose_route_sha256(odom_route),
        ),
        "map_frame": (certificate.map_frame, map_frame),
        "odom_frame": (certificate.odom_frame, odom_frame),
        "base_frame": (certificate.base_frame, base_frame),
        "waypoint_count": (certificate.waypoint_count, observed_waypoint_count),
        "tracking_tube_radius_m": (
            certificate.tracking_tube_radius_m,
            _finite_number(
                tracking_tube_radius_m,
                "tracking_tube_radius_m",
            ),
        ),
        "command_owner": (certificate.command_owner, command_owner),
    }
    optional_checks = {
        "map_from_odom": (certificate.map_from_odom, map_from_odom),
        "transform_stamp_sec": (
            certificate.transform_stamp_sec,
            transform_stamp_sec,
        ),
        "transform_capture_time_sec": (
            certificate.transform_capture_time_sec,
            transform_capture_time_sec,
        ),
        "uncertainty_budget_sha256": (
            certificate.uncertainty_budget_sha256,
            uncertainty_budget_sha256,
        ),
        "ambiguity_evidence_sha256": (
            certificate.ambiguity_evidence_sha256,
            ambiguity_evidence_sha256,
        ),
    }
    for name, pair in optional_checks.items():
        if pair[1] is not None:
            checks[name] = pair

    for name, (certified, observed) in checks.items():
        if certified != observed:
            raise ValueError(
                f"odom execution certificate {name} mismatch: "
                f"certified={certified!r}, observed={observed!r}"
            )

    expected_odom_route = transform_map_route_to_odom(
        source_route,
        certificate.map_from_odom,
    )
    if pose_route_sha256(expected_odom_route) != pose_route_sha256(odom_route):
        raise ValueError(
            "transformed_odom_route geometry does not match inverse map_from_odom"
        )


def _validate_certificate(
    certificate: OdomExecutionCertificate,
    *,
    canonicalize: bool,
) -> None:
    if not isinstance(certificate, OdomExecutionCertificate):
        raise ValueError("certificate must be an OdomExecutionCertificate")
    _require_sha256(
        certificate.source_map_route_sha256,
        "source_map_route_sha256",
    )
    _require_sha256(
        certificate.source_map_execution_certificate_sha256,
        "source_map_execution_certificate_sha256",
    )
    _require_sha256(
        certificate.transformed_odom_route_sha256,
        "transformed_odom_route_sha256",
    )
    _require_sha256(
        certificate.uncertainty_budget_sha256,
        "uncertainty_budget_sha256",
    )
    _require_sha256(
        certificate.ambiguity_evidence_sha256,
        "ambiguity_evidence_sha256",
    )
    frames = (
        _require_frame_id(certificate.map_frame, "map_frame"),
        _require_frame_id(certificate.odom_frame, "odom_frame"),
        _require_frame_id(certificate.base_frame, "base_frame"),
    )
    if len(set(frames)) != len(frames):
        raise ValueError("map_frame, odom_frame, and base_frame must be distinct")
    if not isinstance(certificate.map_from_odom, PlanarTransform2D):
        raise ValueError("map_from_odom must be a PlanarTransform2D")

    transform_stamp = _nonnegative_finite(
        certificate.transform_stamp_sec,
        "transform_stamp_sec",
    )
    capture_time = _nonnegative_finite(
        certificate.transform_capture_time_sec,
        "transform_capture_time_sec",
    )
    # AMCL may deliberately future-date map->odom by its configured transform
    # tolerance.  The certificate binds both clock values exactly; freshness
    # and permitted future offset remain preflight-evidence decisions.
    if (
        not isinstance(certificate.waypoint_count, int)
        or isinstance(certificate.waypoint_count, bool)
        or certificate.waypoint_count < 2
    ):
        raise ValueError("waypoint_count must be an integer >= 2")
    tube_radius = _finite_number(
        certificate.tracking_tube_radius_m,
        "tracking_tube_radius_m",
    )
    if tube_radius <= 0.0:
        raise ValueError("tracking_tube_radius_m must be finite and positive")
    _require_command_owner(certificate.command_owner)
    if (
        not isinstance(certificate.schema_version, int)
        or isinstance(certificate.schema_version, bool)
        or certificate.schema_version != ODOM_EXECUTION_CERTIFICATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported odom execution certificate schema")

    if canonicalize:
        object.__setattr__(certificate, "transform_stamp_sec", transform_stamp)
        object.__setattr__(
            certificate,
            "transform_capture_time_sec",
            capture_time,
        )
        object.__setattr__(certificate, "tracking_tube_radius_m", tube_radius)


def _certificate_payload(
    certificate: OdomExecutionCertificate,
) -> dict[str, object]:
    _validate_certificate(certificate, canonicalize=False)
    return {
        "schema_version": certificate.schema_version,
        "source_map_route_sha256": certificate.source_map_route_sha256,
        "source_map_execution_certificate_sha256": (
            certificate.source_map_execution_certificate_sha256
        ),
        "transformed_odom_route_sha256": (
            certificate.transformed_odom_route_sha256
        ),
        "map_frame": certificate.map_frame,
        "odom_frame": certificate.odom_frame,
        "base_frame": certificate.base_frame,
        "map_from_odom": {
            "x_m": certificate.map_from_odom.x_m,
            "y_m": certificate.map_from_odom.y_m,
            "yaw_rad": certificate.map_from_odom.yaw_rad,
        },
        "transform_stamp_sec": certificate.transform_stamp_sec,
        "transform_capture_time_sec": certificate.transform_capture_time_sec,
        "waypoint_count": certificate.waypoint_count,
        "tracking_tube_radius_m": certificate.tracking_tube_radius_m,
        "command_owner": certificate.command_owner,
        "uncertainty_budget_sha256": certificate.uncertainty_budget_sha256,
        "ambiguity_evidence_sha256": certificate.ambiguity_evidence_sha256,
    }


def _pose_route_payload(route: Sequence[Pose2D]) -> dict[str, object]:
    validated = _validated_route(route, "route")
    return {
        "schema_version": POSE_ROUTE_HASH_SCHEMA_VERSION,
        "waypoints": [
            {
                "x_m": pose.x_m,
                "y_m": pose.y_m,
                # Route CSVs use quiet NaN for an unconstrained intermediate
                # heading.  JSON has no portable NaN value, so bind that state
                # canonically as null instead of enabling non-standard JSON.
                "yaw_rad": None if math.isnan(pose.yaw_rad) else pose.yaw_rad,
            }
            for pose in validated
        ],
    }


def _validated_route(
    route: Sequence[Pose2D],
    name: str,
) -> tuple[Pose2D, ...]:
    if isinstance(route, (str, bytes)) or not isinstance(route, Sequence):
        raise ValueError(f"{name} must be a Pose2D sequence")
    validated = tuple(
        _validated_pose(pose, f"{name}[{index}]")
        for index, pose in enumerate(route)
    )
    if len(validated) < 2:
        raise ValueError(f"{name} must contain at least two waypoints")
    return validated


def _validated_pose(pose: Pose2D, name: str) -> Pose2D:
    if not isinstance(pose, Pose2D):
        raise ValueError(f"{name} must be a Pose2D")
    return Pose2D(
        x_m=_canonical_finite(pose.x_m, f"{name}.x_m"),
        y_m=_canonical_finite(pose.y_m, f"{name}.y_m"),
        yaw_rad=_canonical_pose_yaw(pose.yaw_rad, f"{name}.yaw_rad"),
    )


def _canonical_pose_yaw(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    yaw = float(value)
    if math.isnan(yaw):
        return math.nan
    if math.isinf(yaw):
        raise ValueError(f"{name} must be finite or NaN (unconstrained)")
    return normalize_yaw(yaw)


def _transform_pose_yaw(yaw_rad: float, rotation_rad: float) -> float:
    if math.isnan(yaw_rad):
        return math.nan
    return normalize_yaw(yaw_rad + rotation_rad)


def _validated_transform(transform: PlanarTransform2D) -> PlanarTransform2D:
    if not isinstance(transform, PlanarTransform2D):
        raise ValueError("map_from_odom must be a PlanarTransform2D")
    # Reconstruct so a forged or subsequently corrupted object still fails.
    return PlanarTransform2D(transform.x_m, transform.y_m, transform.yaw_rad)


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_frame_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty frame id")
    if value != value.strip() or value.startswith("/") or any(
        character.isspace() for character in value
    ):
        raise ValueError(
            f"{name} must be an unprefixed frame id without whitespace"
        )
    return value


def _require_command_owner(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError("command_owner must be an absolute node identity")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_finite(value: object, name: str) -> float:
    return _canonical_zero(_finite_number(value, name))


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _nonnegative_finite(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return _canonical_zero(result)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)
