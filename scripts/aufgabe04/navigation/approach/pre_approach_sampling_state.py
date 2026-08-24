"""Persistent orientation-blind pre-approach sampling state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RejectedInspectionCandidate:
    candidate_index: int
    reason: str


@dataclass(frozen=True)
class PreApproachSamplingState:
    schema_version: int
    stand_id: str
    reference_x_m: float
    reference_y_m: float
    candidate_index: int
    candidate_count: int
    rejected: tuple[RejectedInspectionCandidate, ...] = ()

    def reject_current(self, reason: str) -> "PreApproachSamplingState":
        cleaned_reason = reason.strip()
        if not cleaned_reason:
            raise ValueError("pre-approach rejection reason is required")
        next_index = self.candidate_index + 1
        if next_index >= self.candidate_count:
            raise ValueError("all pre-approach inspection candidates are exhausted")
        return PreApproachSamplingState(
            schema_version=self.schema_version,
            stand_id=self.stand_id,
            reference_x_m=self.reference_x_m,
            reference_y_m=self.reference_y_m,
            candidate_index=next_index,
            candidate_count=self.candidate_count,
            rejected=self.rejected + (
                RejectedInspectionCandidate(self.candidate_index, cleaned_reason),
            ),
        )


def initial_sampling_state(
    *, stand_id: str, reference_x_m: float, reference_y_m: float, candidate_count: int
) -> PreApproachSamplingState:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return PreApproachSamplingState(
        schema_version=1,
        stand_id=stand_id,
        reference_x_m=reference_x_m,
        reference_y_m=reference_y_m,
        candidate_index=0,
        candidate_count=candidate_count,
    )


def load_sampling_state(path: Path) -> PreApproachSamplingState:
    payload = json.loads(Path(path).read_text())
    rejected = tuple(
        RejectedInspectionCandidate(**item) for item in payload.pop("rejected", [])
    )
    state = PreApproachSamplingState(rejected=rejected, **payload)
    if state.schema_version != 1:
        raise ValueError("unsupported pre-approach sampling state schema")
    if not 0 <= state.candidate_index < state.candidate_count:
        raise ValueError("invalid pre-approach candidate index in state")
    return state


def write_sampling_state(path: Path, state: PreApproachSamplingState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
