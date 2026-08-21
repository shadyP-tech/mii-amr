"""Fail-closed negative-visibility reconciliation for provisional candidates.

The decision is independent of the expected stand count.  A single-view
candidate may be rejected only when another visited, planned-visible viewpoint
provides repeated exact-time finite rays that clear the candidate's complete
radius-plus-uncertainty envelope.  Positive, nearer, occluded, malformed, or
insufficient evidence always retains the candidate for later camera handling.

This module is ROS-free.  It does not mutate the survey registry, select a
route, or authorize motion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.map_io import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.route_smoothing import (
    segment_is_collision_free,
    supercover_segment_cells,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    STATUS_PROVISIONAL,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    coverage_survey_plan_sha256,
    validate_stand_survey_registry,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    LidarVisibilityReceipt,
    validate_lidar_visibility_receipt,
    visibility_receipts_sha256,
)


COVERAGE_CANDIDATE_RECONCILIATION_SCHEMA_VERSION = 1
ACTION_RETAIN = "retain"
ACTION_REJECT_PROVISIONAL = "reject_provisional"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CoverageCandidateReconciliationConfig:
    """Conservative thresholds and the required receipt config identity."""

    observer_config_sha256: str
    minimum_distinct_clear_scan_count: int = 3
    minimum_clear_scan_separation_sec: float = 0.05
    far_edge_clearance_margin_m: float = 0.03
    matching_range_tolerance_m: float = 0.02

    def validated(self) -> "CoverageCandidateReconciliationConfig":
        if (
            not isinstance(self.observer_config_sha256, str)
            or _SHA256.fullmatch(self.observer_config_sha256) is None
        ):
            raise ValueError(
                "observer_config_sha256 must be a lowercase SHA-256"
            )
        if (
            type(self.minimum_distinct_clear_scan_count) is not int
            or self.minimum_distinct_clear_scan_count < 2
        ):
            raise ValueError(
                "minimum_distinct_clear_scan_count must be an integer >= 2"
            )
        _finite_nonnegative(
            self.minimum_clear_scan_separation_sec,
            "minimum_clear_scan_separation_sec",
        )
        _finite_nonnegative(
            self.far_edge_clearance_margin_m,
            "far_edge_clearance_margin_m",
        )
        _finite_nonnegative(
            self.matching_range_tolerance_m,
            "matching_range_tolerance_m",
        )
        return self

    def to_evidence_dict(self) -> dict[str, object]:
        self.validated()
        return {
            "observer_config_sha256": self.observer_config_sha256,
            "minimum_distinct_clear_scan_count": (
                self.minimum_distinct_clear_scan_count
            ),
            "minimum_clear_scan_separation_sec": (
                self.minimum_clear_scan_separation_sec
            ),
            "far_edge_clearance_margin_m": (
                self.far_edge_clearance_margin_m
            ),
            "matching_range_tolerance_m": (
                self.matching_range_tolerance_m
            ),
        }


@dataclass(frozen=True)
class CandidateVisibilityRayEvidence:
    receipt_id: str
    receipt_sha256: str
    viewpoint_id: str
    scan_stamp_sec: float
    classification: str
    reason: str
    candidate_distance_m: float
    candidate_envelope_radius_m: float
    near_edge_distance_m: float
    far_edge_distance_m: float
    conservative_distance_limit_m: float
    selected_ray_index: int | None
    selected_ray_bearing_rad: float | None
    selected_ray_offset_rad: float | None
    selected_range_m: float | None
    static_supercover_cells: tuple[GridCell, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "receipt_sha256": self.receipt_sha256,
            "viewpoint_id": self.viewpoint_id,
            "scan_stamp_sec": self.scan_stamp_sec,
            "classification": self.classification,
            "reason": self.reason,
            "candidate_distance_m": self.candidate_distance_m,
            "candidate_envelope_radius_m": self.candidate_envelope_radius_m,
            "near_edge_distance_m": self.near_edge_distance_m,
            "far_edge_distance_m": self.far_edge_distance_m,
            "conservative_distance_limit_m": (
                self.conservative_distance_limit_m
            ),
            "selected_ray_index": self.selected_ray_index,
            "selected_ray_bearing_rad": self.selected_ray_bearing_rad,
            "selected_ray_offset_rad": self.selected_ray_offset_rad,
            "selected_range_m": self.selected_range_m,
            "static_supercover_cells": [
                {"x": cell.x, "y": cell.y}
                for cell in self.static_supercover_cells
            ],
        }


@dataclass(frozen=True)
class CoverageCandidateReconciliationDecision:
    schema_version: int
    candidate_uid: str
    candidate_status: str
    action: str
    reject_provisional: bool
    reasons: tuple[str, ...]
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    config: CoverageCandidateReconciliationConfig
    input_receipt_set_sha256: str
    input_receipt_ids: tuple[str, ...]
    source_viewpoint_ids: tuple[str, ...]
    eligible_planned_viewpoint_ids: tuple[str, ...]
    receipt_viewpoint_ids: tuple[str, ...]
    distinct_clear_scan_stamps_sec: tuple[float, ...]
    ray_evidence: tuple[CandidateVisibilityRayEvidence, ...]

    @property
    def decision_sha256(self) -> str:
        return payload_sha256(self.to_evidence_dict())

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_uid": self.candidate_uid,
            "candidate_status": self.candidate_status,
            "action": self.action,
            "reject_provisional": self.reject_provisional,
            "reasons": list(self.reasons),
            "expected_stand_count_used": False,
            "survey": {
                "survey_id": self.survey_id,
                "planning_frame": self.planning_frame,
                "map_bundle_sha256": self.map_bundle_sha256,
                "plan_sha256": self.plan_sha256,
            },
            "config": self.config.to_evidence_dict(),
            "input_receipts": {
                "receipt_set_sha256": self.input_receipt_set_sha256,
                "receipt_ids": list(self.input_receipt_ids),
            },
            "viewpoints": {
                "source": list(self.source_viewpoint_ids),
                "eligible_planned": list(
                    self.eligible_planned_viewpoint_ids
                ),
                "with_receipts": list(self.receipt_viewpoint_ids),
            },
            "distinct_clear_scan_stamps_sec": list(
                self.distinct_clear_scan_stamps_sec
            ),
            "ray_evidence": [
                item.to_evidence_dict() for item in self.ray_evidence
            ],
        }


def reconcile_provisional_candidate_visibility(
    *,
    plan: CoverageSurveyPlan,
    candidate: SurveyCandidate,
    occupancy_grid: OccupancyGrid,
    receipts: Iterable[LidarVisibilityReceipt],
    config: CoverageCandidateReconciliationConfig,
) -> CoverageCandidateReconciliationDecision:
    """Return immutable evidence; never mutate ``candidate`` or ``plan``."""

    _validate_plan(plan)
    _validate_candidate(candidate, plan=plan)
    config.validated()
    static_costmap = _validated_static_costmap(occupancy_grid)
    supplied_receipts = tuple(receipts)
    for receipt in supplied_receipts:
        validate_lidar_visibility_receipt(receipt)
    receipt_items = tuple(
        sorted(
            supplied_receipts,
            key=lambda item: (
                item.viewpoint_id,
                item.scan_stamp_sec,
                item.receipt_id,
                item.receipt_sha256,
            ),
        )
    )
    receipt_ids_unique = len({item.receipt_id for item in receipt_items}) == len(
        receipt_items
    )

    source_viewpoint_ids = tuple(sorted(set(candidate.viewpoint_ids)))
    unknown_source_viewpoint_ids = tuple(
        viewpoint_id
        for viewpoint_id in source_viewpoint_ids
        if viewpoint_id not in plan.viewpoint_ids
    )
    candidate_pose = Pose2D(candidate.x_m, candidate.y_m, 0.0)
    candidate_cell = static_costmap.world_to_grid(candidate_pose)
    eligible_viewpoint_ids = tuple(
        viewpoint.viewpoint_id
        for viewpoint in plan.viewpoints
        if viewpoint.viewpoint_id not in source_viewpoint_ids
        and candidate_cell in viewpoint.visible_cells
    )
    plan_viewpoint_ids = set(plan.viewpoint_ids)
    identity_mismatch = any(
        receipt.survey_id != plan.survey_id
        or receipt.planning_frame != plan.planning_frame
        or receipt.map_bundle_sha256 != plan.map_bundle_sha256
        or receipt.observer_config_sha256 != config.observer_config_sha256
        or receipt.viewpoint_id not in plan_viewpoint_ids
        for receipt in receipt_items
    )
    eligible_receipts = tuple(
        sorted(
            (
                receipt
                for receipt in receipt_items
                if receipt.viewpoint_id in eligible_viewpoint_ids
            ),
            key=lambda item: (
                item.viewpoint_id,
                item.scan_stamp_sec,
                item.receipt_id,
            ),
        )
    )
    receipt_viewpoint_ids = tuple(
        sorted({receipt.viewpoint_id for receipt in eligible_receipts})
    )
    missing_viewpoint_receipts = tuple(
        viewpoint_id
        for viewpoint_id in eligible_viewpoint_ids
        if viewpoint_id not in receipt_viewpoint_ids
    )
    ray_evidence = tuple(
        _evaluate_receipt_ray(
            receipt,
            candidate=candidate,
            static_costmap=static_costmap,
            config=config,
            maximum_visibility_distance_m=plan.config.visibility_radius_m,
        )
        for receipt in eligible_receipts
    )
    clear_stamps = _separated_clear_scan_stamps(
        ray_evidence,
        minimum_separation_sec=config.minimum_clear_scan_separation_sec,
    )

    reasons: list[str] = []
    if candidate.status != STATUS_PROVISIONAL:
        reasons.append("candidate_not_provisional")
    if len(source_viewpoint_ids) != 1:
        reasons.append("candidate_not_single_view")
    if unknown_source_viewpoint_ids:
        reasons.append("candidate_source_viewpoint_unknown")
    if identity_mismatch:
        reasons.append("visibility_receipt_identity_mismatch")
    if not receipt_ids_unique:
        reasons.append("visibility_receipt_ids_not_unique")
    if not receipt_items:
        reasons.append("visibility_receipts_missing")
    if not eligible_viewpoint_ids:
        reasons.append("no_other_planned_visible_viewpoint")
    if missing_viewpoint_receipts:
        reasons.append("planned_visible_viewpoint_receipts_missing")
    for reason in (
        "actual_static_line_of_sight_blocked",
        "candidate_envelope_outside_conservative_range",
        "no_scan_ray_intersects_candidate_envelope",
        "selected_scan_ray_invalid",
        "nearer_return_occludes_candidate",
        "matching_return_supports_candidate",
    ):
        if any(item.reason == reason for item in ray_evidence):
            reasons.append(reason)
    if (
        len(clear_stamps)
        < config.minimum_distinct_clear_scan_count
    ):
        reasons.append("insufficient_distinct_clear_scan_times")

    reject = not reasons
    return CoverageCandidateReconciliationDecision(
        schema_version=COVERAGE_CANDIDATE_RECONCILIATION_SCHEMA_VERSION,
        candidate_uid=candidate.candidate_uid,
        candidate_status=candidate.status,
        action=ACTION_REJECT_PROVISIONAL if reject else ACTION_RETAIN,
        reject_provisional=reject,
        reasons=tuple(reasons),
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=coverage_survey_plan_sha256(plan),
        config=config,
        input_receipt_set_sha256=visibility_receipts_sha256(receipt_items),
        input_receipt_ids=tuple(item.receipt_id for item in receipt_items),
        source_viewpoint_ids=source_viewpoint_ids,
        eligible_planned_viewpoint_ids=eligible_viewpoint_ids,
        receipt_viewpoint_ids=receipt_viewpoint_ids,
        distinct_clear_scan_stamps_sec=clear_stamps,
        ray_evidence=ray_evidence,
    )


def _evaluate_receipt_ray(
    receipt: LidarVisibilityReceipt,
    *,
    candidate: SurveyCandidate,
    static_costmap: Costmap,
    config: CoverageCandidateReconciliationConfig,
    maximum_visibility_distance_m: float,
) -> CandidateVisibilityRayEvidence:
    target = Pose2D(candidate.x_m, candidate.y_m, 0.0)
    dx = target.x_m - receipt.scan_pose_map.x_m
    dy = target.y_m - receipt.scan_pose_map.y_m
    distance = math.hypot(dx, dy)
    envelope_radius = candidate.radius_m + candidate.uncertainty_m
    near_edge = max(0.0, distance - envelope_radius)
    far_edge = distance + envelope_radius
    conservative_distance_limit = min(
        receipt.range_max_m,
        maximum_visibility_distance_m,
    )
    line_of_sight_target = _far_envelope_pose(
        receipt.scan_pose_map,
        target,
        envelope_radius_m=envelope_radius,
    )
    supercover = supercover_segment_cells(
        static_costmap,
        receipt.scan_pose_map,
        line_of_sight_target,
    )
    common = {
        "receipt_id": receipt.receipt_id,
        "receipt_sha256": receipt.receipt_sha256,
        "viewpoint_id": receipt.viewpoint_id,
        "scan_stamp_sec": receipt.scan_stamp_sec,
        "candidate_distance_m": distance,
        "candidate_envelope_radius_m": envelope_radius,
        "near_edge_distance_m": near_edge,
        "far_edge_distance_m": far_edge,
        "conservative_distance_limit_m": conservative_distance_limit,
        "static_supercover_cells": supercover,
    }
    if not segment_is_collision_free(
        static_costmap,
        receipt.scan_pose_map,
        line_of_sight_target,
    ):
        return CandidateVisibilityRayEvidence(
            classification="blocked",
            reason="actual_static_line_of_sight_blocked",
            selected_ray_index=None,
            selected_ray_bearing_rad=None,
            selected_ray_offset_rad=None,
            selected_range_m=None,
            **common,
        )
    if (
        far_edge + config.far_edge_clearance_margin_m
        > conservative_distance_limit + _EPSILON
    ):
        return CandidateVisibilityRayEvidence(
            classification="out_of_range",
            reason="candidate_envelope_outside_conservative_range",
            selected_ray_index=None,
            selected_ray_bearing_rad=None,
            selected_ray_offset_rad=None,
            selected_range_m=None,
            **common,
        )
    target_bearing = math.atan2(dy, dx) - receipt.scan_pose_map.yaw_rad
    ray_index, ray_bearing, ray_offset = _nearest_intersecting_ray(
        receipt,
        target_bearing_rad=target_bearing,
        candidate_distance_m=distance,
        envelope_radius_m=envelope_radius,
    )
    if ray_index is None:
        return CandidateVisibilityRayEvidence(
            classification="no_intersection",
            reason="no_scan_ray_intersects_candidate_envelope",
            selected_ray_index=None,
            selected_ray_bearing_rad=None,
            selected_ray_offset_rad=None,
            selected_range_m=None,
            **common,
        )
    selected_range = receipt.ranges_m[ray_index]
    selected = {
        "selected_ray_index": ray_index,
        "selected_ray_bearing_rad": ray_bearing,
        "selected_ray_offset_rad": ray_offset,
        "selected_range_m": selected_range,
    }
    if selected_range is None:
        return CandidateVisibilityRayEvidence(
            classification="invalid",
            reason="selected_scan_ray_invalid",
            **selected,
            **common,
        )
    if (
        selected_range
        < near_edge - config.matching_range_tolerance_m
    ):
        return CandidateVisibilityRayEvidence(
            classification="nearer",
            reason="nearer_return_occludes_candidate",
            **selected,
            **common,
        )
    if (
        selected_range
        <= far_edge
        + config.far_edge_clearance_margin_m
        + config.matching_range_tolerance_m
    ):
        return CandidateVisibilityRayEvidence(
            classification="matching",
            reason="matching_return_supports_candidate",
            **selected,
            **common,
        )
    return CandidateVisibilityRayEvidence(
        classification="clear",
        reason="finite_ray_clears_candidate_far_edge",
        **selected,
        **common,
    )


def _far_envelope_pose(
    origin: Pose2D,
    center: Pose2D,
    *,
    envelope_radius_m: float,
) -> Pose2D:
    dx = center.x_m - origin.x_m
    dy = center.y_m - origin.y_m
    distance = math.hypot(dx, dy)
    if distance <= _EPSILON:
        return center
    scale = envelope_radius_m / distance
    return Pose2D(
        center.x_m + scale * dx,
        center.y_m + scale * dy,
        0.0,
    )


def _nearest_intersecting_ray(
    receipt: LidarVisibilityReceipt,
    *,
    target_bearing_rad: float,
    candidate_distance_m: float,
    envelope_radius_m: float,
) -> tuple[int | None, float | None, float | None]:
    best: tuple[float, int, float] | None = None
    for index in range(len(receipt.ranges_m)):
        ray_bearing = (
            receipt.angle_min_rad + index * receipt.angle_increment_rad
        )
        offset = _normalize_angle(ray_bearing - target_bearing_rad)
        perpendicular_distance = candidate_distance_m * abs(math.sin(offset))
        forward_projection = candidate_distance_m * math.cos(offset)
        if (
            forward_projection <= 0.0
            or perpendicular_distance > envelope_radius_m + _EPSILON
        ):
            continue
        candidate_key = (abs(offset), index, ray_bearing)
        if best is None or candidate_key < best:
            best = candidate_key
    if best is None:
        return None, None, None
    offset, index, ray_bearing = best
    signed_offset = _normalize_angle(ray_bearing - target_bearing_rad)
    return index, ray_bearing, signed_offset


def _separated_clear_scan_stamps(
    evidence: Iterable[CandidateVisibilityRayEvidence],
    *,
    minimum_separation_sec: float,
) -> tuple[float, ...]:
    clear = sorted(
        {
            item.scan_stamp_sec
            for item in evidence
            if item.classification == "clear"
        }
    )
    selected: list[float] = []
    for stamp in clear:
        if (
            not selected
            or stamp - selected[-1] + _EPSILON >= minimum_separation_sec
        ):
            selected.append(stamp)
    return tuple(selected)


def _validate_plan(plan: CoverageSurveyPlan) -> None:
    if not isinstance(plan, CoverageSurveyPlan):
        raise ValueError("plan must be a CoverageSurveyPlan")
    coverage_survey_plan_sha256(plan)
    if not plan.viewpoints:
        raise ValueError("plan must contain viewpoints")


def _validate_candidate(
    candidate: SurveyCandidate,
    *,
    plan: CoverageSurveyPlan,
) -> None:
    if not isinstance(candidate, SurveyCandidate):
        raise ValueError("candidate must be a SurveyCandidate")
    validate_stand_survey_registry(
        StandSurveyRegistry(
            schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
            survey_id=plan.survey_id,
            planning_frame=plan.planning_frame,
            map_bundle_sha256=plan.map_bundle_sha256,
            candidates=(candidate,),
        ),
        plan,
    )


def _validated_static_costmap(occupancy_grid: OccupancyGrid) -> Costmap:
    if not isinstance(occupancy_grid, OccupancyGrid):
        raise ValueError("occupancy_grid must be an OccupancyGrid")
    if (
        type(occupancy_grid.width) is not int
        or type(occupancy_grid.height) is not int
        or occupancy_grid.width <= 0
        or occupancy_grid.height <= 0
        or len(occupancy_grid.cells) != occupancy_grid.height
        or any(len(row) != occupancy_grid.width for row in occupancy_grid.cells)
    ):
        raise ValueError("occupancy_grid dimensions are invalid")
    if (
        not math.isfinite(occupancy_grid.metadata.resolution)
        or occupancy_grid.metadata.resolution <= 0.0
        or len(occupancy_grid.metadata.origin) != 3
        or not all(
            math.isfinite(value) for value in occupancy_grid.metadata.origin
        )
        or abs(occupancy_grid.metadata.origin[2]) > _EPSILON
    ):
        raise ValueError("occupancy_grid metadata is invalid")
    valid_cells = {CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN}
    if any(
        value not in valid_cells
        for row in occupancy_grid.cells
        for value in row
    ):
        raise ValueError("occupancy_grid contains an invalid cell value")
    return Costmap.from_occupancy_grid(occupancy_grid, block_unknown=True)


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _finite(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_nonnegative(value: float, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


__all__ = [
    "ACTION_REJECT_PROVISIONAL",
    "ACTION_RETAIN",
    "COVERAGE_CANDIDATE_RECONCILIATION_SCHEMA_VERSION",
    "CandidateVisibilityRayEvidence",
    "CoverageCandidateReconciliationConfig",
    "CoverageCandidateReconciliationDecision",
    "reconcile_provisional_candidate_visibility",
]
