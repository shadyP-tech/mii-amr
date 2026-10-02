"""Bounded geometry alternatives under one unchanged stopped uncertainty budget.

This search admits a route prefix, never motion. Ascending half-cell increments
retain the physical floor and select the first passing member of a finite set;
this is not a claim of globally minimum clearance or globally complete search.
"""
from dataclasses import asdict, dataclass
import math
from typing import Callable

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import (
    ReturnUncertaintyExhausted, return_route_geometry, select_admitted_return_prefix,
)

MAXIMUM_ROUTE_ALTERNATIVES = 5
ROUTE_ALTERNATIVES_POLICY = "ascending_half_cell_inflation_first_admitted_prefix_v1"


class AlternativeGeometryRejected(ValueError):
    """A geometric candidate failed after source authentication completed."""


class RouteAlternativesExhausted(ValueError):
    def __init__(self, evidence):
        super().__init__("return uncertainty budget exhausted: no meaningful admitted prefix after bounded route alternatives")
        self.evidence = evidence


@dataclass(frozen=True)
class StoredPoseRouteGeometry:
    result: object
    planning_costmap: object
    connector: object
    smoothing: object
    full_poses: tuple


def alternative_inflation_radii(base_radius_m: float, resolution_m: float) -> tuple[float, ...]:
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.
           for v in (base_radius_m, resolution_m)):
        raise ValueError("alternative inflation requires finite positive floor and resolution")
    return tuple(round(base_radius_m + index * resolution_m / 2., 12)
                 for index in range(MAXIMUM_ROUTE_ALTERNATIVES))


def uncertainty_context_evidence(context):
    return {"covariance": asdict(context.covariance), "config": asdict(context.admission_config),
            "source_evidence": dict(context.source_evidence)}


def select_stored_pose_route_alternative(
    build_candidate: Callable[[float], StoredPoseRouteGeometry], *, base_radius_m: float,
    base_costmap, uncertainty, target_evidence_sha256: str, identity: dict,
):
    """Retry geometry/uncertainty rejections only; unexpected/source errors escape."""
    radii = alternative_inflation_radii(base_radius_m, base_costmap.resolution)
    context = uncertainty_context_evidence(uncertainty)
    evidence = {
        "schema_version": 1, "artifact_kind": "stored_pose_route_alternatives", "motion_authorized": False,
        "policy": ROUTE_ALTERNATIVES_POLICY, "identity": dict(identity),
        "target_evidence_sha256": target_evidence_sha256,
        "uncertainty_context_sha256": payload_sha256(context),
        "base_inflation_radius_m": base_radius_m, "map_resolution_m": base_costmap.resolution,
        "candidate_inflation_radii_m": list(radii), "robot_radius_m": uncertainty.admission_config.robot_radius_m,
        "clearance_meaning": "robot_center_to_raw_obstacle; physical_floor_and_uncertainty_reserves_unchanged",
        "search_scope": "bounded_geometric_routes_and_stopped_prefixes; not_global_minimum_or_motion_permission",
        "attempts": [], "selected_attempt_index": None,
    }
    rejected_geometries = {}
    for index, radius in enumerate(radii):
        attempt = {"inflation_radius_m": radius}
        evidence["attempts"].append(attempt)
        try:
            geometry = build_candidate(radius)
        except AlternativeGeometryRejected as exc:
            attempt.update(status="geometry_rejected", reason=str(exc))
            continue
        attempt.update(full_route_geometry=return_route_geometry(geometry.full_poses))
        geometry_key = payload_sha256({**return_route_geometry(geometry.full_poses),
            "minimum_prefix_vertex_index": 1 if geometry.connector.required else 0})
        if geometry_key in rejected_geometries:
            prior_index, prior_evidence = rejected_geometries[geometry_key]
            attempt.update(status="uncertainty_rejected", admission_attempts=prior_evidence,
                reason="identical geometry rejected under the same frozen uncertainty context",
                uncertainty_evaluation_reused_from_attempt=prior_index)
            continue
        try:
            stage = select_admitted_return_prefix(
                full_poses=geometry.full_poses, base_costmap=base_costmap, uncertainty=uncertainty,
                target_evidence_sha256=target_evidence_sha256,
                minimum_prefix_vertex_index=1 if geometry.connector.required else 0,
            )
        except ReturnUncertaintyExhausted as exc:
            attempt.update(status="uncertainty_rejected", reason=str(exc), admission_attempts=exc.evidence)
            rejected_geometries[geometry_key] = (index, exc.evidence)
            continue
        attempt.update(status="accepted", selected_admission_sha256=payload_sha256(stage.evidence["selected_admission"]),
            selected_route_geometry_sha256=payload_sha256(return_route_geometry(stage.poses)),
            is_final_stage=stage.is_final_stage, stage_target_pose=asdict(stage.stage_target_pose))
        evidence["selected_attempt_index"] = index
        return geometry, stage, radius, evidence
    raise RouteAlternativesExhausted(evidence)


def validate_alternative_evidence(evidence, *, selected_radius_m, full_poses, selection, identity, target_sha256):
    """Bind the chosen geometry and unchanged context to the declared search."""
    if not isinstance(evidence, dict):
        raise ValueError("route alternatives evidence must be an object")
    expected = {"schema_version": 1, "artifact_kind": "stored_pose_route_alternatives", "motion_authorized": False,
                "policy": ROUTE_ALTERNATIVES_POLICY, "identity": identity, "target_evidence_sha256": target_sha256}
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"route alternatives {key} mismatch")
    radii = alternative_inflation_radii(evidence["base_inflation_radius_m"], evidence["map_resolution_m"])
    attempts, index = evidence.get("attempts"), evidence.get("selected_attempt_index")
    if (evidence.get("candidate_inflation_radii_m") != list(radii) or type(index) is not int
        or not 0 <= index < len(radii) or not isinstance(attempts, list) or len(attempts) != index + 1
        or selected_radius_m != radii[index]):
        raise ValueError("route alternatives selected radius or attempt order mismatch")
    for position, attempt in enumerate(attempts):
        if (not isinstance(attempt, dict) or attempt.get("inflation_radius_m") != radii[position]
            or attempt.get("status") not in ({"accepted"} if position == index else {"geometry_rejected", "uncertainty_rejected"})):
            raise ValueError("route alternatives must select the first admitted attempt")
    selected = attempts[-1]
    if (selected.get("full_route_geometry") != return_route_geometry(full_poses)
        or selected.get("selected_admission_sha256") != payload_sha256(selection["selected_admission"])
        or selected.get("selected_route_geometry_sha256") != selection["selected_route_geometry_sha256"]
        or selected.get("is_final_stage") != selection["is_final_stage"]):
        raise ValueError("route alternatives chosen geometry or admission mismatch")
    context = {key: selection[key] for key in ("covariance", "config", "source_evidence")}
    if evidence.get("uncertainty_context_sha256") != payload_sha256(context):
        raise ValueError("route alternatives uncertainty context mismatch")
