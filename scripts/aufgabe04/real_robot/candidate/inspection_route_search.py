"""Bounded pose proposals for a generic candidate inspection direction.

A rejected pose is not an observed view. Each direction may try fresh routes at
smaller standoffs, down to the unchanged physical and raster keepout limits.
Only typed static/no-motion failures permit another proposal; every other
failure propagates. This module neither publishes motion nor issues permits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import TypeVar

from scripts.aufgabe04.real_robot.candidate.opposite_face_route_fallback import (
    bounded_approach_offsets,
)


MAX_STANDOFF_PROPOSALS_PER_DIRECTION = 9
MAX_GENERIC_ROUTE_PROPOSALS = 64
STANDOFF_STEP_M = 0.05
Frame = TypeVar("Frame")


class CandidateInspectionRouteUnavailableError(RuntimeError):
    """A pose proposal failed static or strictly no-motion admission."""

    def __init__(self, message: str, *, reason_code: str = "route_unavailable",
                 evidence: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.evidence = {} if evidence is None else dict(evidence)


def bounded_inspection_standoffs(
    requested_m: float, *, minimum_active_standoff_m: float,
    candidate_transit_radius_m: float, map_resolution_m: float | None,
) -> tuple[float, ...]:
    """Use the opposite-face step, with finite work and a raster-safe floor.

    No map resolution means the legacy injected offline adapter cannot prove a
    smaller standoff. It retains just the requested proposal. Production maps
    provide the resolution; the actual planner still validates the full map,
    all candidate keepouts, continuous clearance and each route certificate.
    """
    for value in (requested_m, minimum_active_standoff_m, candidate_transit_radius_m):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("inspection standoff inputs must be finite and positive")
    if requested_m < minimum_active_standoff_m:
        raise ValueError("requested inspection standoff is below its physical minimum")
    if map_resolution_m is None:
        return (float(requested_m),)
    if (isinstance(map_resolution_m, bool) or not isinstance(map_resolution_m, (int, float))
            or not math.isfinite(map_resolution_m) or map_resolution_m <= 0):
        raise ValueError("inspection map resolution must be finite and positive")
    raster_floor = candidate_transit_radius_m + map_resolution_m / math.sqrt(2.0)
    # The planner requires strict separation above the raster margin. Round up
    # before using bounded_approach_offsets, whose public contract rounds to 1um.
    minimum = max(minimum_active_standoff_m,
                  math.ceil((raster_floor + 1.0e-6) * 1.0e6) / 1.0e6)
    if requested_m < minimum:
        raise ValueError("requested inspection standoff violates raster keepout minimum")
    # Clip the interval before invoking the shared stepping helper, so even a
    # malformed very large requested radius cannot allocate unbounded options.
    bounded_minimum = max(minimum, requested_m - STANDOFF_STEP_M * (MAX_STANDOFF_PROPOSALS_PER_DIRECTION - 2))
    values = list(bounded_approach_offsets(requested_m, bounded_minimum, step_m=STANDOFF_STEP_M))
    if values[-1] > minimum + 1.0e-9:
        values.append(minimum)
    return tuple(values)


@dataclass
class CandidateInspectionRouteSearch:
    """A finite candidate-local route ledger, separate from camera view count."""

    candidate_uid: str
    event_sink: Callable[[Mapping[str, object]], None]
    max_proposals: int = MAX_GENERIC_ROUTE_PROPOSALS
    proposals: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.max_proposals) is not int or not 1 <= self.max_proposals <= MAX_GENERIC_ROUTE_PROPOSALS:
            raise ValueError("generic inspection route proposal limit must be in [1, 64]")

    @property
    def budget_exhausted(self) -> bool:
        return len(self.proposals) >= self.max_proposals

    def to_dict(self) -> dict[str, object]:
        return {
            "max_generic_route_proposals": self.max_proposals,
            "generic_route_proposal_count": len(self.proposals),
            "generic_route_proposal_budget_exhausted": self.budget_exhausted,
            "generic_route_proposals": [dict(item) for item in self.proposals],
        }

    def move_direction(
        self, *, requested_normal_rad: float, standoffs: tuple[float, ...],
        output_root: Path, move: Callable[[float, Path], Frame],
    ) -> Frame:
        if not math.isfinite(requested_normal_rad):
            raise ValueError("inspection direction must be finite")
        if not standoffs or len(standoffs) > MAX_STANDOFF_PROPOSALS_PER_DIRECTION:
            raise ValueError("inspection direction has an invalid standoff proposal set")
        direction_attempts = []
        for radius_index, standoff in enumerate(standoffs):
            if self.budget_exhausted:
                raise CandidateInspectionRouteUnavailableError(
                    "candidate generic route proposal budget exhausted",
                    reason_code="route_proposal_budget_exhausted", evidence=self.to_dict(),
                )
            root = output_root if radius_index == 0 else output_root / f"standoff_{radius_index:03d}"
            record: dict[str, object] = {
                "candidate_uid": self.candidate_uid,
                "proposal_index": len(self.proposals),
                "standoff_attempt_index": radius_index,
                "requested_normal_rad": requested_normal_rad,
                "approach_offset_m": standoff,
                "output_root": str(root),
                "outcome": "proposal_started",
                "route_limits_unchanged": True,
                "motion_authorized": False,
            }
            self.proposals.append(record)
            direction_attempts.append(record)
            self.event_sink({**record, "event": "inspection_route_proposal_started"})
            try:
                result = move(standoff, root)
            except CandidateInspectionRouteUnavailableError as exc:
                record.update(outcome="proposal_rejected", reason=str(exc),
                              reason_code=exc.reason_code, evidence=exc.evidence)
                self.event_sink({**record, "event": "inspection_route_proposal_rejected"})
                continue
            except Exception as exc:
                record.update(outcome="terminal_failure", reason=str(exc),
                              error_type=type(exc).__name__)
                self.event_sink({**record, "event": "inspection_route_proposal_terminal"})
                raise
            record["outcome"] = "route_completed"
            self.event_sink({**record, "event": "inspection_route_proposal_completed"})
            return result
        raise CandidateInspectionRouteUnavailableError(
            "all bounded standoff proposals for inspection direction are unavailable",
            reason_code="standoff_proposals_exhausted",
            evidence={"direction_proposals": [dict(item) for item in direction_attempts]},
        )
