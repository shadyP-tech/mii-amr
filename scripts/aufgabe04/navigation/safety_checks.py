"""Pure Aufgabe 04 navigation safety gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    validate_start_egress_certificate,
)
from scripts.aufgabe04.navigation.waypoint_csv import SelectedRouteLeg
from scripts.aufgabe04.navigation.waypoint_csv import poses_from_waypoints
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    arrival_pose_catalog_sha256,
    load_arrival_pose_catalog,
)


@dataclass(frozen=True)
class PreflightStatus:
    ok: bool
    failures: List[str]


@dataclass(frozen=True)
class CatalogStartEgressCertificate:
    required: bool
    waypoint_index: int | None = None
    minimum_route_clearance_m: float | None = None
    start_join_clearance_m: float | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_required_topics(available_topics: Iterable[str], required_topics: Iterable[str]) -> PreflightStatus:
    available = set(available_topics)
    missing = [topic for topic in required_topics if topic not in available]
    return PreflightStatus(ok=not missing, failures=[f"missing topic: {topic}" for topic in missing])


def validate_route_diagnostics_json(
    path: Path,
    leg_index: int,
    *,
    csv_point_count: int,
    require_motion: bool = True,
) -> PreflightStatus:
    failures: List[str] = []
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return PreflightStatus(ok=False, failures=[f"invalid diagnostics JSON: {exc}"])

    legs = payload.get("legs")
    if not isinstance(legs, list):
        return PreflightStatus(ok=False, failures=["diagnostics JSON missing legs list"])
    if leg_index < 0 or leg_index >= len(legs):
        return PreflightStatus(ok=False, failures=[f"diagnostics missing leg_index {leg_index}"])

    leg = legs[leg_index]
    if not isinstance(leg, dict):
        return PreflightStatus(ok=False, failures=[f"diagnostics leg {leg_index} must be an object"])
    diagnostics = leg.get("diagnostics")
    if not isinstance(diagnostics, dict):
        failures.append(f"diagnostics leg {leg_index} missing diagnostics object")
    elif diagnostics.get("status") != "ok":
        failures.append(f"diagnostics leg {leg_index} status is not ok")
    if leg.get("failure") is not None:
        failures.append(f"diagnostics leg {leg_index} has failure")

    route_point_count = leg.get("route_point_count")
    if route_point_count != csv_point_count:
        failures.append(
            f"diagnostics leg {leg_index} route_point_count {route_point_count} "
            f"does not match CSV count {csv_point_count}"
        )
    route_length = leg.get("route_length_m")
    if not isinstance(route_length, (int, float)) or not math.isfinite(route_length):
        failures.append(f"diagnostics leg {leg_index} route_length_m must be finite")
    elif require_motion and route_length <= 0.0:
        failures.append(f"diagnostics leg {leg_index} route_length_m must be positive for motion")

    return PreflightStatus(ok=not failures, failures=failures)


def validate_catalog_route_binding_json(
    path: Path,
    leg: SelectedRouteLeg,
    *,
    position_tolerance_m: float = 1.0e-9,
    angle_tolerance_rad: float = 1.0e-9,
) -> PreflightStatus:
    """Bind one frozen catalog CSV leg to its planning diagnostics.

    The generic diagnostics gate intentionally supports legacy route formats.
    This stricter companion is only for ``catalog_face_approach`` and prevents
    a partial/stale artifact replacement from retaining an apparently valid
    point count while changing the semantic target or protected corridor.
    """

    failures: List[str] = []
    if leg.route_kind != "catalog_face_approach":
        return PreflightStatus(
            ok=False,
            failures=["catalog route binding requires catalog_face_approach"],
        )
    if (
        len(leg.catalog_sha256) != 64
        or any(character not in "0123456789abcdef" for character in leg.catalog_sha256)
    ):
        failures.append("catalog route CSV has invalid catalog_sha256")
    if not leg.source_arrival_id:
        failures.append("catalog route CSV is missing source_arrival_id")
    if not leg.target_arrival_id:
        failures.append("catalog route CSV is missing target_arrival_id")

    corridor = tuple(waypoint for waypoint in leg.raw_waypoints if waypoint.corridor)
    if not corridor:
        failures.append("catalog route CSV has no terminal corridor points")
    else:
        if any(not waypoint.protected for waypoint in corridor):
            failures.append("catalog route corridor contains an unprotected point")
        if any(not math.isfinite(waypoint.pose.yaw_rad) for waypoint in corridor):
            failures.append("catalog route corridor contains an unconstrained yaw")
        if leg.raw_waypoints[-1] != corridor[-1]:
            failures.append("catalog route final point is not the corridor target")

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid diagnostics JSON: {exc}")
        return PreflightStatus(ok=False, failures=failures)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        failures.append("catalog diagnostics missing metadata object")
    else:
        if metadata.get("route_kind") != leg.route_kind:
            failures.append("catalog route kind does not match diagnostics")
        if metadata.get("catalog_sha256") != leg.catalog_sha256:
            failures.append("catalog SHA-256 does not match diagnostics")
        expected_route_hash = metadata.get("route_csv_sha256")
        if (
            not isinstance(expected_route_hash, str)
            or len(expected_route_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_route_hash
            )
        ):
            failures.append("catalog diagnostics have invalid route_csv_sha256")
        else:
            try:
                actual_route_hash = _file_sha256(leg.source_path)
            except OSError as exc:
                failures.append(f"catalog route CSV hash unavailable: {exc}")
            else:
                if actual_route_hash != expected_route_hash:
                    failures.append("catalog route CSV SHA-256 does not match diagnostics")
        catalog_path_value = metadata.get("catalog_path")
        if not isinstance(catalog_path_value, str) or not catalog_path_value.strip():
            failures.append("catalog diagnostics are missing catalog_path")
        else:
            try:
                catalog = load_arrival_pose_catalog(Path(catalog_path_value))
            except (OSError, ValueError) as exc:
                failures.append(f"catalog snapshot validation failed: {exc}")
            else:
                if not catalog.frozen:
                    failures.append("catalog route references an unfrozen catalog")
                if arrival_pose_catalog_sha256(catalog) != leg.catalog_sha256:
                    failures.append("current catalog SHA-256 does not match route CSV")

    legs = payload.get("legs")
    if not isinstance(legs, list) or not 0 <= leg.leg_index < len(legs):
        failures.append(f"catalog diagnostics missing leg_index {leg.leg_index}")
        return PreflightStatus(ok=False, failures=failures)
    diagnostics_leg = legs[leg.leg_index]
    if not isinstance(diagnostics_leg, dict):
        failures.append(f"catalog diagnostics leg {leg.leg_index} must be an object")
        return PreflightStatus(ok=False, failures=failures)
    if diagnostics_leg.get("source_arrival_id") != leg.source_arrival_id:
        failures.append("catalog source arrival ID does not match diagnostics")
    if diagnostics_leg.get("target_arrival_id") != leg.target_arrival_id:
        failures.append("catalog target arrival ID does not match diagnostics")

    def compare_pose(field: str, csv_pose) -> None:
        value = diagnostics_leg.get(field)
        if not isinstance(value, dict):
            failures.append(f"catalog diagnostics missing {field}")
            return
        for coordinate, tolerance in (
            ("x_m", position_tolerance_m),
            ("y_m", position_tolerance_m),
            ("yaw_rad", angle_tolerance_rad),
        ):
            expected = getattr(csv_pose, coordinate)
            actual = value.get(coordinate)
            if (
                not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or not math.isfinite(expected)
                or abs(float(actual) - expected) > tolerance
            ):
                failures.append(
                    f"catalog {field}.{coordinate} does not match route CSV"
                )

    if leg.raw_waypoints:
        compare_pose("exact_target", leg.raw_waypoints[-1].pose)
    if corridor:
        compare_pose("corridor_entry", corridor[0].pose)
    try:
        catalog_start_egress_certificate(path, leg)
    except ValueError as exc:
        failures.append(f"catalog start-egress certificate is invalid: {exc}")
    return PreflightStatus(ok=not failures, failures=failures)


def catalog_start_egress_certificate(
    path: Path,
    leg: SelectedRouteLeg,
) -> CatalogStartEgressCertificate:
    """Validate and expose a frozen catalog leg's source-egress lock.

    A non-mission leg may begin in the one raster cell deliberately exempted
    around its source stand.  In that case the artifact's exact route—not a
    lookahead chord—is the clearance certificate.  Reuse the dynamic-route
    geometric validator so every reported stand clearance is remeasured from
    the immutable CSV before enabling waypoint-1 lock execution.
    """

    if leg.route_kind != "catalog_face_approach":
        raise ValueError("catalog start-egress requires catalog_face_approach")
    try:
        payload = json.loads(Path(path).read_text())
        diagnostics_leg = payload["legs"][leg.leg_index]
        overlay = diagnostics_leg["non_target_keepout_overlay"]
        clearances = diagnostics_leg["non_target_stand_clearances"]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"catalog egress evidence is unavailable: {exc}") from exc
    if not isinstance(overlay, Mapping):
        raise ValueError("non_target_keepout_overlay must be an object")
    raw_exempted = overlay.get("start_cell_exempted")
    if not isinstance(raw_exempted, bool):
        raise ValueError("start_cell_exempted must be boolean")
    if leg.source_arrival_id == "mission_start":
        if raw_exempted:
            raise ValueError("mission-start leg must not claim a source-arrival exemption")
        return CatalogStartEgressCertificate(False)
    if not raw_exempted:
        return CatalogStartEgressCertificate(False)
    if overlay.get("start_cell_was_rasterized") is not True:
        raise ValueError("exempted source cell was not recorded as rasterized")
    for field in (
        "exact_start_minimum_margin_m",
        "cell_center_minimum_margin_m",
        "start_connector_minimum_margin_m",
    ):
        value = overlay.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{field} must be finite and positive")
    if not isinstance(clearances, list) or not clearances:
        raise ValueError("continuous non-target stand clearances are missing")
    try:
        start_join_clearance_m = float(
            diagnostics_leg["diagnostics"]["fixed_arrival"][
                "start_join_clearance_m"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"start_join_clearance_m is unavailable: {exc}") from exc
    if not math.isfinite(start_join_clearance_m) or start_join_clearance_m <= 0.0:
        raise ValueError("start_join_clearance_m must be finite and positive")
    safety = {
        "known_stand_start_cell_exempted": True,
        "known_stand_start_cell": overlay.get("start_cell"),
        "known_stand_keepout_rasterized_cell_count": overlay.get(
            "rasterized_cell_count"
        ),
        "known_stand_keepout_cell_count": overlay.get("blocked_cell_count"),
        # Static catalog clearances already contain stand identity, radius, and
        # measured route clearance, so they serve as both lists expected by
        # the shared geometric certificate validator.
        "known_stand_keepouts": clearances,
        "known_stand_keepout_clearances": clearances,
    }
    waypoints = poses_from_waypoints(leg.raw_waypoints)
    geometric = validate_start_egress_certificate(
        safety,
        waypoints,
        waypoints[0],
    )
    return CatalogStartEgressCertificate(
        required=geometric.required,
        waypoint_index=geometric.waypoint_index,
        minimum_route_clearance_m=geometric.minimum_route_clearance_m,
        start_join_clearance_m=start_join_clearance_m,
    )


def validate_speed_limits(
    max_linear_mps: float,
    max_angular_radps: float,
    *,
    min_linear_mps: float = 0.0,
    max_allowed_linear_mps: float = 0.06,
    min_angular_radps: float = 0.0,
    max_allowed_angular_radps: float = 0.20,
) -> PreflightStatus:
    failures: List[str] = []
    values = {
        "max_linear_mps": max_linear_mps,
        "max_angular_radps": max_angular_radps,
    }
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{name} must be finite")
    if not failures:
        if max_linear_mps <= min_linear_mps:
            failures.append("max_linear_mps must be positive")
        if max_linear_mps > max_allowed_linear_mps:
            failures.append(f"max_linear_mps exceeds {max_allowed_linear_mps:.3f} m/s")
        if max_angular_radps <= min_angular_radps:
            failures.append("max_angular_radps must be positive")
        if max_angular_radps > max_allowed_angular_radps:
            failures.append(f"max_angular_radps exceeds {max_allowed_angular_radps:.3f} rad/s")
    return PreflightStatus(ok=not failures, failures=failures)
