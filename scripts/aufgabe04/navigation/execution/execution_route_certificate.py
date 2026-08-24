"""Pure execution-contract checks for collision-certified Aufgabe 04 routes.

Planning certifies a polyline, not every shortcut a feedback controller might
choose between its vertices.  This module makes that distinction explicit:
physical stand-approach routes must pursue the next certified vertex and the
live base pose plus commanded pursuit chord must remain inside a narrow tube
around the currently certified segment.

The helpers are ROS-free so the same rules can be exercised in unit tests,
the simulation adapter, and a later real-robot adapter.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


EXECUTION_ROUTE_CERTIFICATE_SCHEMA_VERSION = 1
_HASH_FIELD = "execution_route_certificate_sha256"
_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version",
        "route_sha256",
        "planning_frame",
        "route_kind",
        "waypoint_count",
        "tracking_tube_radius_m",
        "exact_vertex_pursuit",
        "command_owner",
        "map_bundle_sha256",
        "candidate_snapshot_sha256",
    }
)


@dataclass(frozen=True)
class ExecutionRouteCertificate:
    """Identity and controller policy bound to one persisted route artifact."""

    route_sha256: str
    planning_frame: str
    route_kind: str
    waypoint_count: int
    tracking_tube_radius_m: float
    exact_vertex_pursuit: bool
    command_owner: str
    map_bundle_sha256: str = ""
    candidate_snapshot_sha256: str = ""
    schema_version: int = EXECUTION_ROUTE_CERTIFICATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.route_sha256, "route_sha256")
        if self.map_bundle_sha256:
            _require_sha256(self.map_bundle_sha256, "map_bundle_sha256")
        if self.candidate_snapshot_sha256:
            _require_sha256(
                self.candidate_snapshot_sha256, "candidate_snapshot_sha256"
            )
        if not str(self.planning_frame).strip():
            raise ValueError("planning_frame must be non-empty")
        if not str(self.route_kind).strip():
            raise ValueError("route_kind must be non-empty")
        if not str(self.command_owner).strip():
            raise ValueError("command_owner must be non-empty")
        if (
            not isinstance(self.waypoint_count, int)
            or isinstance(self.waypoint_count, bool)
            or self.waypoint_count < 2
        ):
            raise ValueError("waypoint_count must be an integer >= 2")
        if (
            not math.isfinite(self.tracking_tube_radius_m)
            or self.tracking_tube_radius_m <= 0.0
        ):
            raise ValueError("tracking_tube_radius_m must be finite and positive")
        if not self.exact_vertex_pursuit:
            raise ValueError(
                "physical execution certificates require exact vertex pursuit"
            )
        if self.schema_version != EXECUTION_ROUTE_CERTIFICATE_SCHEMA_VERSION:
            raise ValueError("unsupported execution route certificate schema")


@dataclass(frozen=True)
class ExecutionRouteCheck:
    ok: bool
    reason: str
    pose_distance_to_segment_m: float
    maximum_chord_distance_to_segment_m: float
    active_segment_start_index: int
    active_segment_end_index: int
    target_index: int
    pursuit_index: int
    tracking_tube_radius_m: float

    def to_log_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "source": "execution_route_certificate",
            "pose_distance_to_segment_m": self.pose_distance_to_segment_m,
            "maximum_chord_distance_to_segment_m": (
                self.maximum_chord_distance_to_segment_m
            ),
            "active_segment_start_index": self.active_segment_start_index,
            "active_segment_end_index": self.active_segment_end_index,
            "target_index": self.target_index,
            "pursuit_index": self.pursuit_index,
            "tracking_tube_radius_m": self.tracking_tube_radius_m,
            "fail_closed": not self.ok,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_route_certificate_sha256(
    certificate: ExecutionRouteCertificate,
) -> str:
    return payload_sha256(_certificate_payload(certificate))


def write_execution_route_certificate(
    path: Path, certificate: ExecutionRouteCertificate
) -> str:
    try:
        return write_content_hashed_json(
            path,
            _certificate_payload(certificate),
            hash_field=_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_execution_route_certificate(path: Path) -> ExecutionRouteCertificate:
    try:
        payload = load_content_hashed_json(path, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _CERTIFICATE_FIELDS:
        raise ValueError("execution route certificate fields mismatch")
    try:
        certificate = ExecutionRouteCertificate(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            route_sha256=_string(payload["route_sha256"], "route_sha256"),
            planning_frame=_string(payload["planning_frame"], "planning_frame"),
            route_kind=_string(payload["route_kind"], "route_kind"),
            waypoint_count=_integer(payload["waypoint_count"], "waypoint_count"),
            tracking_tube_radius_m=_number(
                payload["tracking_tube_radius_m"], "tracking_tube_radius_m"
            ),
            exact_vertex_pursuit=_boolean(
                payload["exact_vertex_pursuit"], "exact_vertex_pursuit"
            ),
            command_owner=_string(payload["command_owner"], "command_owner"),
            map_bundle_sha256=_string(
                payload["map_bundle_sha256"], "map_bundle_sha256"
            ),
            candidate_snapshot_sha256=_string(
                payload["candidate_snapshot_sha256"],
                "candidate_snapshot_sha256",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid execution route certificate: {exc}") from exc
    return certificate


def validate_execution_route_identity(
    certificate: ExecutionRouteCertificate,
    *,
    route_path: Path,
    planning_frame: str,
    route_kind: str,
    waypoint_count: int,
    command_owner: str,
    map_bundle_sha256: str = "",
    candidate_snapshot_sha256: str = "",
    route_snapshot_sha256: str | None = None,
) -> None:
    """Reject a route or runtime that differs from the certified inputs."""

    observed_route_sha256 = (
        file_sha256(route_path)
        if route_snapshot_sha256 is None
        else route_snapshot_sha256
    )
    checks = {
        "route_sha256": (certificate.route_sha256, observed_route_sha256),
        "planning_frame": (certificate.planning_frame, planning_frame),
        "route_kind": (certificate.route_kind, route_kind),
        "waypoint_count": (certificate.waypoint_count, waypoint_count),
        "command_owner": (certificate.command_owner, command_owner),
    }
    if certificate.map_bundle_sha256 or map_bundle_sha256:
        checks["map_bundle_sha256"] = (
            certificate.map_bundle_sha256,
            map_bundle_sha256,
        )
    if certificate.candidate_snapshot_sha256 or candidate_snapshot_sha256:
        checks["candidate_snapshot_sha256"] = (
            certificate.candidate_snapshot_sha256,
            candidate_snapshot_sha256,
        )
    for name, (certified, observed) in checks.items():
        if certified != observed:
            raise ValueError(
                f"execution route certificate {name} mismatch: "
                f"certified={certified!r}, observed={observed!r}"
            )


def point_to_segment_distance_m(
    point: Pose2D,
    start: Pose2D,
    end: Pose2D,
) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-18:
        return math.hypot(point.x_m - start.x_m, point.y_m - start.y_m)
    projection = (
        (point.x_m - start.x_m) * dx + (point.y_m - start.y_m) * dy
    ) / length_sq
    projection = max(0.0, min(1.0, projection))
    nearest_x = start.x_m + projection * dx
    nearest_y = start.y_m + projection * dy
    return math.hypot(point.x_m - nearest_x, point.y_m - nearest_y)


def check_execution_route_tube(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    *,
    target_index: int,
    pursuit_index: int,
    tracking_tube_radius_m: float,
    chord_sample_spacing_m: float = 0.01,
) -> ExecutionRouteCheck:
    """Check the live pose and pursuit chord against the active route segment."""

    if len(waypoints) < 2:
        raise ValueError("a certified execution route requires at least two waypoints")
    if not 0 <= target_index < len(waypoints):
        raise ValueError("target_index is outside the route")
    if not 0 <= pursuit_index < len(waypoints):
        raise ValueError("pursuit_index is outside the route")
    if not math.isfinite(tracking_tube_radius_m) or tracking_tube_radius_m <= 0.0:
        raise ValueError("tracking_tube_radius_m must be finite and positive")
    if not math.isfinite(chord_sample_spacing_m) or chord_sample_spacing_m <= 0.0:
        raise ValueError("chord_sample_spacing_m must be finite and positive")

    # Exact-vertex execution never commands a chord beyond the target.  Check
    # this independently from the controller implementation so a future
    # configuration regression stops before publishing velocity.
    if pursuit_index != target_index:
        return ExecutionRouteCheck(
            ok=False,
            reason="uncertified pursuit shortcut",
            pose_distance_to_segment_m=math.inf,
            maximum_chord_distance_to_segment_m=math.inf,
            active_segment_start_index=max(0, target_index - 1),
            active_segment_end_index=target_index,
            target_index=target_index,
            pursuit_index=pursuit_index,
            tracking_tube_radius_m=tracking_tube_radius_m,
        )

    segment_end_index = target_index
    segment_start_index = max(0, segment_end_index - 1)
    segment_start = waypoints[segment_start_index]
    segment_end = waypoints[segment_end_index]
    pose_distance = point_to_segment_distance_m(pose, segment_start, segment_end)

    pursuit = waypoints[pursuit_index]
    chord_length = math.hypot(pursuit.x_m - pose.x_m, pursuit.y_m - pose.y_m)
    sample_count = max(1, int(math.ceil(chord_length / chord_sample_spacing_m)))
    maximum_chord_distance = 0.0
    for sample_index in range(sample_count + 1):
        fraction = sample_index / sample_count
        sample = Pose2D(
            pose.x_m + fraction * (pursuit.x_m - pose.x_m),
            pose.y_m + fraction * (pursuit.y_m - pose.y_m),
            pose.yaw_rad,
        )
        maximum_chord_distance = max(
            maximum_chord_distance,
            point_to_segment_distance_m(sample, segment_start, segment_end),
        )

    tolerance = tracking_tube_radius_m + 1.0e-9
    reason = ""
    if pose_distance > tolerance:
        reason = "pose left certified route tube"
    elif maximum_chord_distance > tolerance:
        reason = "pursuit chord left certified route tube"
    return ExecutionRouteCheck(
        ok=not reason,
        reason=reason,
        pose_distance_to_segment_m=pose_distance,
        maximum_chord_distance_to_segment_m=maximum_chord_distance,
        active_segment_start_index=segment_start_index,
        active_segment_end_index=segment_end_index,
        target_index=target_index,
        pursuit_index=pursuit_index,
        tracking_tube_radius_m=tracking_tube_radius_m,
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _certificate_payload(
    certificate: ExecutionRouteCertificate,
) -> dict[str, object]:
    # Dataclass validation runs at construction; re-run the important identity
    # check here in case a forged object bypassed normal initialization.
    _require_sha256(certificate.route_sha256, "route_sha256")
    return {
        "schema_version": certificate.schema_version,
        "route_sha256": certificate.route_sha256,
        "planning_frame": certificate.planning_frame,
        "route_kind": certificate.route_kind,
        "waypoint_count": certificate.waypoint_count,
        "tracking_tube_radius_m": certificate.tracking_tube_radius_m,
        "exact_vertex_pursuit": certificate.exact_vertex_pursuit,
        "command_owner": certificate.command_owner,
        "map_bundle_sha256": certificate.map_bundle_sha256,
        "candidate_snapshot_sha256": certificate.candidate_snapshot_sha256,
    }


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


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value
