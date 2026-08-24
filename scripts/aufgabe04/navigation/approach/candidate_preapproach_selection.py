"""Compose route previews with pure post-LiDAR camera candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateRouteOption,
    CameraCandidateSelection,
    CameraCandidateSelectionConfig,
    select_camera_candidate,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachPlan,
    CandidatePreapproachUnreachableError,
    compute_candidate_preapproach_plan,
    load_candidate_planning_context,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot


@dataclass(frozen=True)
class PlannedCameraCandidateSelection:
    """The winning score plus the exact no-write route that produced it."""

    selection: CameraCandidateSelection
    selected_plan: CandidatePreapproachPlan

    @property
    def selected_candidate_uid(self) -> str:
        return self.selection.selected_candidate_uid

    @property
    def motion_authorized(self) -> bool:
        return False

    def to_evidence(self) -> dict[str, object]:
        evidence = self.selection.to_evidence()
        evidence.update(
            {
                "selected_route_reused_for_materialization": True,
                "selected_map_bundle_sha256": (
                    self.selected_plan.map_bundle_sha256
                ),
                "motion_authorized": False,
            }
        )
        return evidence


def plan_and_select_camera_candidate(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    plan: CoverageSurveyPlan,
    snapshot: CandidateSnapshot,
    current_pose: Pose2D,
    unresolved: set[str] | frozenset[str],
    approach_offset_m: float,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
    physical_clearance: Mapping[str, float],
    selection_config: CameraCandidateSelectionConfig,
    support_class_by_uid: Mapping[str, str] | None = None,
) -> PlannedCameraCandidateSelection:
    """Preview all unresolved routes, rank them, and retain the winner."""

    unresolved_uids = frozenset(unresolved)
    if not unresolved_uids:
        raise ValueError("camera candidate selection requires unresolved candidates")
    unknown = sorted(unresolved_uids.difference(snapshot.candidate_uids))
    if unknown:
        raise ValueError(
            "unresolved candidates are absent from snapshot: " + ", ".join(unknown)
        )
    if support_class_by_uid is not None:
        missing_support = sorted(unresolved_uids.difference(support_class_by_uid))
        if missing_support:
            raise ValueError(
                "camera handoff support is missing for: "
                + ", ".join(missing_support)
            )

    context = load_candidate_planning_context(
        map_yaml,
        semantic_map_id=semantic_map_id,
        plan=plan,
        snapshot=snapshot,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
        physical_clearance=physical_clearance,
    )
    route_by_uid: dict[str, CandidatePreapproachPlan] = {}
    options: list[CameraCandidateRouteOption] = []
    candidates = sorted(
        (
            candidate
            for candidate in snapshot.candidates
            if candidate.candidate_uid in unresolved_uids
        ),
        key=lambda candidate: candidate.candidate_uid,
    )
    for candidate in candidates:
        support_class = (
            "coverage_admitted"
            if support_class_by_uid is None
            else support_class_by_uid[candidate.candidate_uid]
        )
        try:
            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id=semantic_map_id,
                plan=plan,
                snapshot=snapshot,
                candidate_uid=candidate.candidate_uid,
                start=current_pose,
                approach_offset_m=approach_offset_m,
                inflation_radius_m=inflation_radius_m,
                candidate_transit_radius_m=candidate_transit_radius_m,
                physical_clearance=physical_clearance,
                planning_context=context,
            )
        except CandidatePreapproachUnreachableError as exc:
            options.append(
                CameraCandidateRouteOption(
                    candidate_uid=candidate.candidate_uid,
                    feasible=False,
                    failure_reason=exc.reason,
                    route_length_m=None,
                    turn_burden_rad=None,
                    initial_turn_rad=None,
                    inside_requested_standoff=(
                        _inside_requested_standoff(
                            current_pose,
                            candidate.geometry.x_m,
                            candidate.geometry.y_m,
                            approach_offset_m,
                        )
                    ),
                    support_class=support_class,
                    confidence=candidate.confidence,
                    hit_count=candidate.hit_count,
                )
            )
            continue
        route_by_uid[candidate.candidate_uid] = prepared
        options.append(
            CameraCandidateRouteOption(
                candidate_uid=candidate.candidate_uid,
                feasible=True,
                failure_reason=None,
                route_length_m=prepared.route_length_m,
                turn_burden_rad=prepared.turn_burden_rad,
                initial_turn_rad=prepared.initial_turn_rad,
                inside_requested_standoff=(
                    prepared.inside_requested_standoff
                ),
                support_class=support_class,
                confidence=candidate.confidence,
                hit_count=candidate.hit_count,
            )
        )

    selection = select_camera_candidate(options, selection_config)
    selected_plan = route_by_uid.get(selection.selected_candidate_uid)
    if selected_plan is None:
        raise RuntimeError("selected camera candidate has no reusable route plan")
    return PlannedCameraCandidateSelection(
        selection=selection,
        selected_plan=selected_plan,
    )


def _inside_requested_standoff(
    current_pose: Pose2D,
    stand_x_m: float,
    stand_y_m: float,
    approach_offset_m: float,
) -> bool:
    import math

    return (
        math.hypot(
            current_pose.x_m - stand_x_m,
            current_pose.y_m - stand_y_m,
        )
        + 1.0e-9
        < approach_offset_m
    )


__all__ = [
    "PlannedCameraCandidateSelection",
    "plan_and_select_camera_candidate",
]
