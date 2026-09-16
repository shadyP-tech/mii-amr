"""Pure bounded state and view diversity for one unresolved candidate.

Angles describe the robot's position around a stand in canonical odometry.
Advisory perception can prioritize a search, but cannot assert a directed
stand normal or satisfy the joint QR/axis completion contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


MINIMUM_VIEW_SEPARATION_RAD = math.radians(20.0)


def validate_inspection_budget(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 16:
        raise ValueError("max_candidate_inspection_views must be an integer in [1, 16]")


def novel_view(normal: float, previous: list[float]) -> bool:
    return all(abs(math.remainder(normal - old, 2.0 * math.pi)) >=
               MINIMUM_VIEW_SEPARATION_RAD - 1e-12 for old in previous)


def candidate_view_options(
    current_normal_rad: float,
    *,
    classification: str,
    achieved_normals: list[float],
    attempted_normals: list[float],
    advisory_yaw_rad: float | None = None,
    exhausted_normals: list[float] | None = None,
) -> tuple[float, ...]:
    """Offer diverse hypotheses, without assigning either side as QR front."""

    offsets = (45, -45, 90, -90, 180, 135, -135)
    if classification in {"edge_on", "backside_unresolved", "unobservable"}:
        offsets = (90, -90, 180, 45, -45, 135, -135)
    hypotheses = [math.radians(degrees) for degrees in offsets]
    if classification in {"oblique", "front_readable", "front_unreadable"} and (
        type(advisory_yaw_rad) in (int, float) and math.isfinite(advisory_yaw_rad)
    ):
        angle = min(math.pi / 2, max(MINIMUM_VIEW_SEPARATION_RAD, abs(advisory_yaw_rad)))
        hypotheses = [angle, -angle, *hypotheses]
    if classification == "certified_backside" or (
        classification == "backside_unresolved"
        and type(advisory_yaw_rad) in (int, float) and math.isfinite(advisory_yaw_rad)
        and abs(advisory_yaw_rad) <= math.pi
    ):
        # A bounded current head suggests resolving orientation locally. A
        # certified opposite route that proved infeasible also benefits from
        # a small view change before a quarter-turn search. Both signs remain
        # hypotheses and every resulting route still needs normal admission.
        hypotheses = [MINIMUM_VIEW_SEPARATION_RAD, -MINIMUM_VIEW_SEPARATION_RAD, *hypotheses]
    # Legacy callers supply fully attempted directions. The live controller
    # separately tracks pose proposals and only excludes a failed direction
    # once all its bounded standoffs have been considered.
    previous = achieved_normals + (attempted_normals if exhausted_normals is None else exhausted_normals)
    selected: list[float] = []
    for offset in hypotheses:
        normal = math.remainder(current_normal_rad + offset, 2.0 * math.pi)
        if novel_view(normal, previous + selected):
            selected.append(normal)
    return tuple(selected)


@dataclass
class CandidateInspectionState:
    candidate_uid: str
    max_views: int
    history: list[dict[str, object]] = field(default_factory=list)
    attempted_normals: list[float] = field(default_factory=list)
    achieved_normals: list[float] = field(default_factory=list)
    exhausted_normals: list[float] = field(default_factory=list)
    termination_reason: str | None = None
    provisional_qr_ids: set[str] = field(default_factory=set)
    route_failures: list[dict[str, object]] = field(default_factory=list)
    camera_distance_recovery_attempted: bool = False

    def __post_init__(self) -> None:
        validate_inspection_budget(self.max_views)

    def record(
        self, *, outcome: str, normal: float | None,
        observation: Mapping[str, object] | None = None,
        reason: str | None = None,
    ) -> None:
        if len(self.history) >= self.max_views:
            raise RuntimeError("candidate local inspection view budget exhausted")
        evidence = {} if observation is None else dict(observation)
        qr_id = evidence.get("qr_id")
        if isinstance(qr_id, str) and qr_id:
            if self.provisional_qr_ids and qr_id not in self.provisional_qr_ids:
                raise ValueError("candidate inspection QR identity conflict across views")
            self.provisional_qr_ids.add(qr_id)
        self.history.append({
            "view_index": len(self.history), "outcome": outcome,
            "canonical_view_normal_rad": normal,
            "reason": reason, "observation": evidence,
        })
        if normal is not None:
            self.achieved_normals.append(normal)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1, "progress_kind": "candidate_local_inspection",
            "candidate_uid": self.candidate_uid, "max_views": self.max_views,
            "view_history": list(self.history),
            "attempted_view_normals_rad": list(self.attempted_normals),
            "achieved_view_normals_rad": list(self.achieved_normals),
            "exhausted_view_normals_rad": list(self.exhausted_normals),
            "local_view_count": len(self.history),
            "termination_reason": self.termination_reason,
            "view_budget_exhausted": self.termination_reason == "view_budget_exhausted",
            "proposal_search_exhausted": self.termination_reason in {
                "view_proposals_exhausted", "route_proposal_budget_exhausted",
            },
            "provisional_qr_ids": sorted(self.provisional_qr_ids),
            "route_failures": list(self.route_failures),
            "camera_distance_recovery_attempted": self.camera_distance_recovery_attempted,
            "joint_observation_ready": bool(self.history and
                                            self.history[-1]["outcome"] == "resolved"),
            "completion_authorized": False, "motion_authorized": False,
        }
