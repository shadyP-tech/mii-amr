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
               MINIMUM_VIEW_SEPARATION_RAD for old in previous)


def candidate_view_options(
    current_normal_rad: float,
    *,
    classification: str,
    achieved_normals: list[float],
    attempted_normals: list[float],
    advisory_yaw_rad: float | None = None,
) -> tuple[float, ...]:
    """Offer diverse hypotheses, without assigning either side as QR front."""

    offsets = (45, -45, 90, -90, 180, 135, -135)
    if classification in {"edge_on", "backside_unresolved", "unobservable"}:
        offsets = (90, -90, 180, 45, -45, 135, -135)
    hypotheses = [math.radians(degrees) for degrees in offsets]
    if classification in {"oblique", "front_readable", "front_unreadable"} and (
        advisory_yaw_rad is not None and math.isfinite(advisory_yaw_rad)
    ):
        angle = min(math.pi / 2, max(MINIMUM_VIEW_SEPARATION_RAD, abs(advisory_yaw_rad)))
        hypotheses = [angle, -angle, *hypotheses]
    previous = achieved_normals + attempted_normals
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
    provisional_qr_ids: set[str] = field(default_factory=set)
    route_failures: list[dict[str, object]] = field(default_factory=list)

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
            "provisional_qr_ids": sorted(self.provisional_qr_ids),
            "route_failures": list(self.route_failures),
            "joint_observation_ready": bool(self.history and
                                            self.history[-1]["outcome"] == "resolved"),
            "completion_authorized": False, "motion_authorized": False,
        }
