"""Compose route previews with pure post-LiDAR camera candidate ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.approach.camera_head_alignment import (
    make_camera_alignment, camera_alignment_endpoint,
)
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateRouteOption,
    CameraCandidateSelection,
    CameraCandidateSelectionConfig,
    select_camera_candidate,
)
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
    NoUncertaintyAdmittedCameraCandidateError,
    _executable_route_poses,
    select_uncertainty_admitted_camera_candidate,
)
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import LidarInspectionHint
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachPlan,
    CandidatePreapproachUnreachableError,
    compute_candidate_preapproach_plan,
    load_candidate_planning_context,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
)
from scripts.aufgabe04.navigation.control.driving_behavior import (
    CommandSmoothingConfig,
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.control.segment_time_budget import (
    MAX_WAYPOINT_TIMEOUT_SEC,
    MIN_WAYPOINT_TIMEOUT_SEC,
    MOTION_TIME_MARGIN_FACTOR,
    SETTLING_ALLOWANCE_SEC,
    SegmentTimeBudgetError,
    route_time_budgets,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot


@dataclass(frozen=True)
class PlannedCameraCandidateSelection:
    """The winning score plus the exact no-write route that produced it."""

    selection: CameraCandidateSelection
    selected_plan: CandidatePreapproachPlan
    evidence: Mapping[str, object] | None = None

    @property
    def selected_candidate_uid(self) -> str:
        return self.selection.selected_candidate_uid

    @property
    def motion_authorized(self) -> bool:
        return False

    def to_evidence(self) -> dict[str, object]:
        evidence = dict(
            self.selection.to_evidence()
            if self.evidence is None
            else self.evidence
        )
        evidence.update(
            {
                "selected_candidate_uid": self.selected_candidate_uid,
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
    route_uncertainty_context: CandidateRouteUncertaintyContext | None = None,
    lidar_inspection_hints: Mapping[str, LidarInspectionHint] | None = None,
    lidar_hint_diagnostics: Mapping[str, object] | None = None,
    camera_calibration=None,
    camera_alignment_uncertainty: Mapping[str, float] | None = None,
) -> PlannedCameraCandidateSelection:
    """Preview all unresolved routes, admit/rank them, and retain the winner.

    When a stopped-localization uncertainty context is supplied, every exact
    preview route must pass it before the established camera ranking can select
    the candidate.  The downstream child dry-run remains authoritative.
    """

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
    view_evidence: dict[str, object] = {}
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
            compute_kwargs = dict(
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
            hint = (lidar_inspection_hints or {}).get(candidate.candidate_uid)
            prepared = None
            if hint is not None:
                prepared, view_evidence[candidate.candidate_uid] = _preview_lidar_views(
                    hint=hint, compute_kwargs=compute_kwargs, candidate=candidate,
                    support_class=support_class, selection_config=selection_config,
                    uncertainty=route_uncertainty_context, camera_calibration=camera_calibration,
                    camera_alignment_uncertainty=camera_alignment_uncertainty,
                )
            if prepared is None:
                view_evidence.setdefault(candidate.candidate_uid, {
                    "fallback": True, "reason": "usable_lidar_fit_unavailable",
                    "head_alignment_verified": False, "arrival_verification_required": True,
                })
                prepared = compute_candidate_preapproach_plan(**compute_kwargs)
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
        time_budget = _route_time_budget_evidence(prepared, selection_config)
        failure_reason = (
            str(time_budget["failure_reason"])
            if time_budget is not None and not time_budget["accepted"]
            else None
        )
        if failure_reason is None:
            route_by_uid[candidate.candidate_uid] = prepared
        options.append(
            CameraCandidateRouteOption(
                candidate_uid=candidate.candidate_uid,
                feasible=failure_reason is None,
                failure_reason=failure_reason,
                route_length_m=prepared.route_length_m,
                turn_burden_rad=prepared.turn_burden_rad,
                initial_turn_rad=prepared.initial_turn_rad,
                inside_requested_standoff=(
                    prepared.inside_requested_standoff
                ),
                support_class=support_class,
                confidence=candidate.confidence,
                hit_count=candidate.hit_count,
                route_time_budget=time_budget,
            )
        )

    if route_uncertainty_context is None:
        selection = select_camera_candidate(options, selection_config)
        selected_plan = route_by_uid.get(selection.selected_candidate_uid)
        evidence = selection.to_evidence()
    else:
        admitted = select_uncertainty_admitted_camera_candidate(
            base_costmap=context.costmaps.base_costmap,
            options=tuple(options),
            plans_by_uid=route_by_uid,
            selection_config=selection_config,
            uncertainty=route_uncertainty_context,
        )
        selection = admitted.selection
        selected_plan = admitted.selected_plan
        evidence = admitted.to_evidence()
    if selected_plan is None:
        raise RuntimeError("selected camera candidate has no reusable route plan")
    if lidar_inspection_hints is not None or lidar_hint_diagnostics is not None:
        evidence = {**evidence, "lidar_inspection_hints": {
            "diagnostics": dict(lidar_hint_diagnostics or {}),
            "candidate_views": view_evidence,
            "stand_axis_authorized": False, "motion_authorized": False,
        }}
    return PlannedCameraCandidateSelection(
        selection=selection,
        selected_plan=selected_plan,
        evidence=evidence,
    )


def _preview_lidar_views(*, hint, compute_kwargs, candidate, support_class,
                         selection_config, uncertainty, camera_calibration, camera_alignment_uncertainty):
    """Admit both perpendicular views before choosing one, then allow fallback."""
    normals = hint.normals(compute_kwargs["snapshot"], candidate.candidate_uid)
    evidence = {"hint": dict(hint.evidence), "views": [], "selected_normal_rad": None,
                "fallback": True, "head_alignment_verified": False,
                "arrival_verification_required": True}
    plans = []
    # Missing support is an ordinary acquisition route, explicitly unverified.
    try:
        make_camera_alignment(hint=hint, snapshot=compute_kwargs["snapshot"],
            candidate_uid=candidate.candidate_uid, normal_rad=normals[0],
            standoff_m=compute_kwargs["approach_offset_m"], calibration=camera_calibration,
            uncertainty=camera_alignment_uncertainty)
    except (ValueError, TypeError, AttributeError) as exc:
        evidence["reason"] = str(exc)
        return None, evidence
    for index, normal in enumerate(normals):
        row = {"view_index": index, "normal_rad": normal, "accepted": False}
        evidence["views"].append(row)
        alignment = make_camera_alignment(hint=hint, snapshot=compute_kwargs["snapshot"],
            candidate_uid=candidate.candidate_uid, normal_rad=normal,
            standoff_m=compute_kwargs["approach_offset_m"], calibration=camera_calibration,
            uncertainty=camera_alignment_uncertainty)
        try:
            prepared = compute_candidate_preapproach_plan(
                **compute_kwargs, inspection_view_normal_rad=normal, camera_alignment=alignment,
            )
        except CandidatePreapproachUnreachableError as exc:
            row["reason"] = exc.reason
            continue
        time_budget = _route_time_budget_evidence(prepared, selection_config)
        if time_budget is not None:
            row["route_time_budget"] = time_budget
            if not time_budget["accepted"]:
                row["reason"] = time_budget["failure_reason"]
                continue
        endpoint = camera_alignment_endpoint(alignment, prepared.selected_approach_pose)
        row["camera_alignment_endpoint"] = endpoint
        if not endpoint["accepted"]:
            row["reason"] = "camera_alignment_uncertainty_budget_exceeded"
            continue
        option = CameraCandidateRouteOption(
            candidate_uid=candidate.candidate_uid, feasible=True, failure_reason=None,
            route_length_m=prepared.route_length_m, turn_burden_rad=prepared.turn_burden_rad,
            initial_turn_rad=prepared.initial_turn_rad,
            inside_requested_standoff=prepared.inside_requested_standoff,
            support_class=support_class, confidence=candidate.confidence, hit_count=candidate.hit_count,
        )
        if uncertainty is not None:
            try:
                admitted = select_uncertainty_admitted_camera_candidate(
                    base_costmap=compute_kwargs["planning_context"].costmaps.base_costmap,
                    options=(option,), plans_by_uid={candidate.candidate_uid: prepared},
                    selection_config=selection_config, uncertainty=uncertainty,
                )
                row["route_uncertainty_selection"] = admitted.to_evidence()
            except NoUncertaintyAdmittedCameraCandidateError as exc:
                row.update(reason="route_uncertainty_rejected", route_uncertainty_selection=exc.to_evidence())
                continue
        rank = select_camera_candidate((option,), selection_config).ranked_candidates[0]
        row.update(accepted=True, reason="inspection_route_admitted",
                   route_length_m=prepared.route_length_m, turn_burden_rad=prepared.turn_burden_rad)
        plans.append(((rank.risk_tier, rank.estimated_duration_sec, index), prepared, normal))
    if not plans:
        return None, evidence
    _, prepared, normal = min(plans, key=lambda p: p[0])
    evidence.update(selected_normal_rad=normal, fallback=False)
    return prepared, evidence


def _route_time_budget_evidence(
    prepared: CandidatePreapproachPlan,
    config: CameraCandidateSelectionConfig,
) -> dict[str, object] | None:
    """Admit exact preview geometry with the execution deadline policy."""

    if not config.route_time_budget_enabled:
        return None
    controller = controller_config_for_route_kind(
        ControllerConfig(
            max_linear_mps=config.linear_speed_mps,
            max_angular_radps=config.angular_speed_radps,
        ),
        "detected_stand_preapproach",
        physical_waypoint_tolerance_m=0.02,
        physical_goal_tolerance_m=0.03,
    )
    smoothing = CommandSmoothingConfig()
    evidence: dict[str, object] = {
        "accepted": False,
        "policy": {
            "minimum_timeout_sec": MIN_WAYPOINT_TIMEOUT_SEC,
            "maximum_timeout_sec": MAX_WAYPOINT_TIMEOUT_SEC,
            "motion_time_margin_factor": MOTION_TIME_MARGIN_FACTOR,
            "settling_allowance_sec": SETTLING_ALLOWANCE_SEC,
        },
        "controller": asdict(controller),
        "smoothing": asdict(smoothing),
        "motion_authorized": False,
    }
    try:
        budgets = route_time_budgets(
            _executable_route_poses(prepared),
            start_pose=prepared.start,
            controller=controller,
            smoothing=smoothing,
        )
    except SegmentTimeBudgetError as exc:
        evidence["failure_reason"] = f"route_time_budget_rejected: {exc}"
        return evidence
    evidence.update(accepted=True, budgets=[budget.to_dict() for budget in budgets])
    return evidence


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
