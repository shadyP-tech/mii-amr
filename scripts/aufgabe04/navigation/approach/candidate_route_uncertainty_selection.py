"""Uncertainty admission before post-LiDAR camera candidate ranking.

Every geometrically feasible candidate route is evaluated against one frozen
stationary-localization envelope before the existing camera ranking policy may
select it.  This module is ROS-free, writes nothing, never changes a route, and
never authorizes motion.  The child process still performs a newer, authoritative
dry-run admission immediately before any motion permit can be issued.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CAMERA_CANDIDATE_SELECTION_SCHEMA_VERSION,
    CameraCandidateRouteOption,
    CameraCandidateSelection,
    CameraCandidateSelectionConfig,
    CameraCandidateSelectionError,
    select_camera_candidate,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePreapproachPlan,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    RouteUncertaintyAdmissionResult,
    evaluate_route_uncertainty_admission,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap


CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION = 1
CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_POLICY = (
    "exact-uncertainty-admission-before-existing-camera-ranking"
)


@dataclass(frozen=True)
class CandidateRouteUncertaintyContext:
    """One immutable stopped-localization envelope used only for selection."""

    covariance: PlanarCovariance
    admission_config: RouteUncertaintyAdmissionConfig
    source_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, PlanarCovariance):
            raise TypeError("candidate route covariance must be PlanarCovariance")
        if not isinstance(self.admission_config, RouteUncertaintyAdmissionConfig):
            raise TypeError(
                "candidate route admission config must be "
                "RouteUncertaintyAdmissionConfig"
            )
        if not isinstance(self.source_evidence, Mapping):
            raise TypeError("candidate route source evidence must be a mapping")


@dataclass(frozen=True)
class CandidateRouteUncertaintyEvaluation:
    """Exact admission result for one unchanged candidate route preview."""

    candidate_uid: str
    admission: RouteUncertaintyAdmissionResult
    admission_evidence_sha256: str

    @property
    def accepted(self) -> bool:
        return self.admission.decision.accepted

    def to_evidence(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "accepted": self.accepted,
            "reason": self.admission.decision.reason,
            "minimum_remaining_margin_m": (
                self.admission.decision.remaining_margin_m
            ),
            "limiting_segment_id": (
                self.admission.decision.limiting_segment_id
            ),
            "admission_evidence_sha256": self.admission_evidence_sha256,
            "admission_evidence": self.admission.to_evidence_dict(),
        }


@dataclass(frozen=True)
class UncertaintyAdmittedCameraCandidateSelection:
    """Existing candidate ranking restricted to exactly admitted routes."""

    selection: CameraCandidateSelection
    selected_plan: CandidatePreapproachPlan
    evaluations: tuple[CandidateRouteUncertaintyEvaluation, ...]
    evidence: Mapping[str, object]

    @property
    def selected_candidate_uid(self) -> str:
        return self.selection.selected_candidate_uid

    @property
    def motion_authorized(self) -> bool:
        return False

    def to_evidence(self) -> dict[str, object]:
        return dict(self.evidence)


class NoUncertaintyAdmittedCameraCandidateError(CameraCandidateSelectionError):
    """All geometrically reachable candidate routes failed exact admission."""

    def __init__(self, evidence: Mapping[str, object]) -> None:
        self.evidence = dict(evidence)
        super().__init__(
            "no_uncertainty_admitted_camera_candidate",
            "no geometrically reachable camera candidate route passed "
            "uncertainty admission",
        )

    def to_evidence(self) -> dict[str, object]:
        return {
            **dict(self.evidence),
            "error_code": self.code,
            "reason": str(self),
            "motion_authorized": False,
        }


def select_uncertainty_admitted_camera_candidate(
    *,
    base_costmap: Costmap,
    options: tuple[CameraCandidateRouteOption, ...],
    plans_by_uid: Mapping[str, CandidatePreapproachPlan],
    selection_config: CameraCandidateSelectionConfig,
    uncertainty: CandidateRouteUncertaintyContext,
) -> UncertaintyAdmittedCameraCandidateSelection:
    """Reject unsafe previews, then apply the established candidate ranking.

    Route uncertainty is a strict feasibility partition, not a score.  Every
    surviving option has passed the exact same pure admission calculation; the
    established turn-risk, LiDAR-support, duration, and confidence ordering is
    therefore preserved among admitted candidates.
    """

    if not isinstance(base_costmap, Costmap):
        raise TypeError("candidate route uncertainty requires a base Costmap")
    if not isinstance(uncertainty, CandidateRouteUncertaintyContext):
        raise TypeError("candidate route uncertainty context is required")
    if not isinstance(plans_by_uid, Mapping):
        raise TypeError("candidate route plans must be a mapping")

    # This evidence records what the prior policy would have selected.  It is
    # diagnostic only and cannot bypass the uncertainty partition below.
    geometric_selection = select_camera_candidate(options, selection_config)
    evaluations: list[CandidateRouteUncertaintyEvaluation] = []
    admitted_options: list[CameraCandidateRouteOption] = []
    rejected_options: list[CameraCandidateRouteOption] = []

    for option in options:
        if not option.feasible:
            rejected_options.append(option)
            continue
        plan = plans_by_uid.get(option.candidate_uid)
        if not isinstance(plan, CandidatePreapproachPlan):
            raise ValueError(
                "feasible camera candidate has no bound preapproach plan: "
                f"{option.candidate_uid}"
            )
        if plan.candidate_uid != option.candidate_uid:
            raise ValueError("candidate route plan UID binding mismatch")
        route = plan.result.route
        if route is None:
            raise ValueError("feasible candidate plan has no route")
        executable_poses = _executable_route_poses(plan)
        admission = evaluate_route_uncertainty_admission(
            base_costmap,
            executable_poses,
            uncertainty.covariance,
            uncertainty.admission_config,
        )
        evaluation = CandidateRouteUncertaintyEvaluation(
            candidate_uid=option.candidate_uid,
            admission=admission,
            admission_evidence_sha256=(
                route_uncertainty_admission_evidence_sha256(admission)
            ),
        )
        evaluations.append(evaluation)
        if evaluation.accepted:
            admitted_options.append(option)
        else:
            rejected_options.append(
                replace(
                    option,
                    feasible=False,
                    failure_reason=(
                        "route_uncertainty_rejected:"
                        f"{admission.decision.reason}"
                    ),
                )
            )

    evaluation_evidence = [
        item.to_evidence()
        for item in sorted(evaluations, key=lambda item: item.candidate_uid)
    ]
    common_evidence: dict[str, object] = {
        "schema_version": CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION,
        "policy": CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_POLICY,
        "policy_order": [
            "geometric_route_preview",
            "exact_route_uncertainty_admission",
            "existing_camera_candidate_ranking",
        ],
        "geometric_candidate_selection": geometric_selection.to_evidence(),
        "route_uncertainty_source": dict(uncertainty.source_evidence),
        "route_uncertainty_admission_config": (
            uncertainty.admission_config.to_evidence_dict()
        ),
        "route_uncertainty_evaluations": evaluation_evidence,
        "geometrically_feasible_candidate_count": len(evaluations),
        "uncertainty_admitted_candidate_count": len(admitted_options),
        "route_mutated": False,
        "child_dry_preflight_remains_authoritative": True,
        "motion_authorized": False,
    }

    if not admitted_options:
        uncertainty_evidence = {
            **common_evidence,
            "decision": {
                "ready": False,
                "selected_candidate_uid": None,
                "reason": "no_candidate_route_passed_uncertainty_admission",
                "fail_closed": True,
            },
        }
        evidence = {
            "schema_version": CAMERA_CANDIDATE_SELECTION_SCHEMA_VERSION,
            "selected_candidate_uid": None,
            "route_uncertainty_selection_applied": True,
            "route_uncertainty_selection": uncertainty_evidence,
            "route_uncertainty_selection_sha256": payload_sha256(
                uncertainty_evidence
            ),
            "rejected_candidates": [
                option.to_dict()
                for option in sorted(
                    rejected_options,
                    key=lambda item: item.candidate_uid,
                )
            ],
            "motion_authorized": False,
        }
        raise NoUncertaintyAdmittedCameraCandidateError(evidence)

    admitted_selection = select_camera_candidate(
        (*admitted_options, *rejected_options),
        selection_config,
    )
    selected_plan = plans_by_uid.get(admitted_selection.selected_candidate_uid)
    if not isinstance(selected_plan, CandidatePreapproachPlan):
        raise RuntimeError("uncertainty-admitted candidate has no reusable route")
    selected_evaluation = next(
        item
        for item in evaluations
        if item.candidate_uid == admitted_selection.selected_candidate_uid
    )
    uncertainty_evidence = {
        **common_evidence,
        "decision": {
            "ready": True,
            "selected_candidate_uid": admitted_selection.selected_candidate_uid,
            "selected_minimum_remaining_margin_m": (
                selected_evaluation.admission.decision.remaining_margin_m
            ),
            "selected_limiting_segment_id": (
                selected_evaluation.admission.decision.limiting_segment_id
            ),
            "fail_closed": False,
        },
    }
    evidence = {
        **admitted_selection.to_evidence(),
        "route_uncertainty_selection_applied": True,
        "route_uncertainty_selection": uncertainty_evidence,
        "route_uncertainty_selection_sha256": payload_sha256(
            uncertainty_evidence
        ),
        "motion_authorized": False,
    }
    return UncertaintyAdmittedCameraCandidateSelection(
        selection=admitted_selection,
        selected_plan=selected_plan,
        evaluations=tuple(evaluations),
        evidence=evidence,
    )


def validate_candidate_route_uncertainty_selection_binding(
    evidence: Mapping[str, object],
    prepared: CandidatePreapproachPlan,
) -> None:
    """Bind an admitted selection receipt to the exact retained route."""

    if not isinstance(evidence, Mapping):
        raise TypeError("candidate selection evidence must be a mapping")
    if evidence.get("route_uncertainty_selection_applied") is not True:
        raise ValueError("candidate route uncertainty selection marker is missing")
    selection = evidence.get("route_uncertainty_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("candidate route uncertainty selection evidence is missing")
    digest = evidence.get("route_uncertainty_selection_sha256")
    if digest != payload_sha256(selection):
        raise ValueError("candidate route uncertainty selection hash mismatch")
    if selection.get("policy") != CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_POLICY:
        raise ValueError("candidate route uncertainty selection policy mismatch")
    if (
        selection.get("motion_authorized") is not False
        or selection.get("route_mutated") is not False
    ):
        raise ValueError(
            "candidate route uncertainty selection is not motion neutral"
        )
    source = selection.get("route_uncertainty_source")
    if (
        not isinstance(source, Mapping)
        or source.get("motion_authorized") is not False
    ):
        raise ValueError(
            "candidate route uncertainty source is missing motion-neutral evidence"
        )
    decision = selection.get("decision")
    if not isinstance(decision, Mapping) or decision.get("ready") is not True:
        raise ValueError("candidate route uncertainty selection was not admitted")
    if decision.get("selected_candidate_uid") != prepared.candidate_uid:
        raise ValueError("candidate route uncertainty selected UID mismatch")

    evaluations = selection.get("route_uncertainty_evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("candidate route uncertainty evaluations are missing")
    selected = tuple(
        item
        for item in evaluations
        if isinstance(item, Mapping)
        and item.get("candidate_uid") == prepared.candidate_uid
    )
    if len(selected) != 1 or selected[0].get("accepted") is not True:
        raise ValueError("selected candidate lacks one accepted route admission")
    admission_evidence = selected[0].get("admission_evidence")
    if not isinstance(admission_evidence, Mapping):
        raise ValueError("selected candidate route admission evidence is missing")
    if selected[0].get("admission_evidence_sha256") != (
        route_uncertainty_admission_evidence_sha256(admission_evidence)
    ):
        raise ValueError("selected candidate route admission hash mismatch")
    if admission_evidence.get("config") != selection.get(
        "route_uncertainty_admission_config"
    ):
        raise ValueError("selected candidate route admission config mismatch")
    admission_scope = admission_evidence.get("scope")
    admission_validation = admission_evidence.get("validation")
    budget_evidence = admission_evidence.get("decision")
    budget_decision = (
        budget_evidence.get("decision")
        if isinstance(budget_evidence, Mapping)
        else None
    )
    if (
        not isinstance(admission_scope, Mapping)
        or admission_scope.get("generates_commands") is not False
        or admission_scope.get("mutates_route") is not False
        or not isinstance(admission_validation, Mapping)
        or admission_validation.get("ok") is not True
        or not isinstance(budget_decision, Mapping)
        or budget_decision.get("accepted") is not True
    ):
        raise ValueError(
            "selected candidate route admission is not an accepted pure decision"
        )

    route = prepared.result.route
    if route is None:
        raise ValueError("selected candidate prepared route is missing")
    admitted_route = admission_evidence.get("route")
    if not isinstance(admitted_route, Mapping):
        raise ValueError("selected candidate admission route is missing")
    expected_poses = [
        _route_pose_evidence(pose)
        for pose in _executable_route_poses(prepared)
    ]
    if admitted_route.get("poses") != expected_poses:
        raise ValueError(
            "selected candidate route differs from uncertainty-admitted route"
        )
    expected_route_payload = {"frame": "map", "poses": expected_poses}
    if admitted_route.get("route_sha256") != payload_sha256(
        expected_route_payload
    ):
        raise ValueError("selected candidate uncertainty route hash mismatch")


def _executable_route_poses(
    prepared: CandidatePreapproachPlan,
) -> tuple[Pose2D, ...]:
    """Mirror the materialized CSV route, including its protected final yaw."""

    route = prepared.result.route
    if route is None or not route.points:
        raise ValueError("candidate preapproach route is missing")
    poses = [point.pose for point in route.points]
    final = poses[-1]
    poses[-1] = Pose2D(
        final.x_m,
        final.y_m,
        prepared.terminal_yaw_rad,
    )
    return tuple(poses)


def _route_pose_evidence(pose: Pose2D) -> dict[str, object]:
    yaw = float(pose.yaw_rad)
    return {
        "x_m": float(pose.x_m),
        "y_m": float(pose.y_m),
        "yaw_rad": None if math.isnan(yaw) else yaw,
        "yaw_mode": "unconstrained_nan" if math.isnan(yaw) else "constrained",
    }


__all__ = [
    "CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_POLICY",
    "CANDIDATE_ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION",
    "CandidateRouteUncertaintyContext",
    "CandidateRouteUncertaintyEvaluation",
    "NoUncertaintyAdmittedCameraCandidateError",
    "UncertaintyAdmittedCameraCandidateSelection",
    "select_uncertainty_admitted_camera_candidate",
    "validate_candidate_route_uncertainty_selection_binding",
]
