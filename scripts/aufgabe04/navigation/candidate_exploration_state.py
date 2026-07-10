"""Persistent state for sequential detected-candidate exploration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand


STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
DEFAULT_MATCH_RADIUS_M = 0.12


@dataclass(frozen=True)
class CandidateDecision:
    status: str
    stand_id: str
    x_m: float
    y_m: float
    confidence: float
    hit_count: int
    source_observation_ids: tuple[str, ...]

    def matches(self, stand: ConfirmedStand, *, match_radius_m: float = DEFAULT_MATCH_RADIUS_M) -> bool:
        if self.source_observation_ids and set(self.source_observation_ids).intersection(stand.source_observation_ids):
            return True
        return math.hypot(self.x_m - stand.x_m, self.y_m - stand.y_m) <= match_radius_m

    def to_json_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "stand_id": self.stand_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "confidence": self.confidence,
            "hit_count": self.hit_count,
            "source_observation_ids": list(self.source_observation_ids),
        }


@dataclass(frozen=True)
class CandidateExplorationState:
    decisions: tuple[CandidateDecision, ...] = ()
    legacy_confirmed_stand_ids: frozenset[str] = frozenset()
    legacy_rejected_stand_ids: frozenset[str] = frozenset()

    def status_for(self, stand: ConfirmedStand, *, match_radius_m: float = DEFAULT_MATCH_RADIUS_M) -> str:
        for decision in reversed(self.decisions):
            if decision.matches(stand, match_radius_m=match_radius_m):
                return decision.status
        if stand.stand_id in self.legacy_confirmed_stand_ids:
            return STATUS_CONFIRMED
        if stand.stand_id in self.legacy_rejected_stand_ids:
            return STATUS_REJECTED
        return STATUS_PENDING

    def with_decisions(
        self,
        stands: Iterable[ConfirmedStand],
        *,
        confirmed_stand_ids: Iterable[str] = (),
        rejected_stand_ids: Iterable[str] = (),
    ) -> "CandidateExplorationState":
        by_id = {stand.stand_id: stand for stand in stands}
        decisions = list(self.decisions)
        legacy_confirmed = set(self.legacy_confirmed_stand_ids)
        legacy_rejected = set(self.legacy_rejected_stand_ids)

        for status, stand_ids in (
            (STATUS_CONFIRMED, confirmed_stand_ids),
            (STATUS_REJECTED, rejected_stand_ids),
        ):
            for stand_id in stand_ids:
                cleaned = stand_id.strip()
                if not cleaned:
                    continue
                stand = by_id.get(cleaned)
                if stand is None:
                    if status == STATUS_CONFIRMED:
                        legacy_confirmed.add(cleaned)
                        legacy_rejected.discard(cleaned)
                    else:
                        legacy_rejected.add(cleaned)
                        legacy_confirmed.discard(cleaned)
                    continue
                decisions = [decision for decision in decisions if not decision.matches(stand)]
                decisions.append(_decision_from_stand(stand, status=status))
                legacy_confirmed.discard(cleaned)
                legacy_rejected.discard(cleaned)

        return CandidateExplorationState(
            decisions=tuple(decisions),
            legacy_confirmed_stand_ids=frozenset(legacy_confirmed),
            legacy_rejected_stand_ids=frozenset(legacy_rejected),
        )

    def pending_stands(self, stands: Iterable[ConfirmedStand]) -> tuple[ConfirmedStand, ...]:
        return tuple(stand for stand in stands if self.status_for(stand) == STATUS_PENDING)

    def count(self, status: str, stands: Iterable[ConfirmedStand]) -> int:
        return sum(1 for stand in stands if self.status_for(stand) == status)

    def to_json_dict(self) -> dict[str, object]:
        confirmed_legacy = sorted(self.legacy_confirmed_stand_ids)
        rejected_legacy = sorted(self.legacy_rejected_stand_ids)
        return {
            "schema_version": 2,
            "match_radius_m": DEFAULT_MATCH_RADIUS_M,
            "decisions": [decision.to_json_dict() for decision in self.decisions],
            "confirmed_stand_ids": confirmed_legacy,
            "rejected_stand_ids": rejected_legacy,
        }


def _decision_from_stand(stand: ConfirmedStand, *, status: str) -> CandidateDecision:
    return CandidateDecision(
        status=status,
        stand_id=stand.stand_id,
        x_m=stand.x_m,
        y_m=stand.y_m,
        confidence=stand.confidence,
        hit_count=stand.hit_count,
        source_observation_ids=stand.source_observation_ids,
    )


def _load_decision(payload: object) -> CandidateDecision:
    if not isinstance(payload, dict):
        raise ValueError("candidate decision must be a JSON object")
    status = str(payload.get("status", ""))
    if status not in {STATUS_CONFIRMED, STATUS_REJECTED}:
        raise ValueError(f"invalid candidate decision status: {status!r}")
    return CandidateDecision(
        status=status,
        stand_id=str(payload.get("stand_id", "")),
        x_m=float(payload["x_m"]),
        y_m=float(payload["y_m"]),
        confidence=float(payload.get("confidence", 0.0)),
        hit_count=int(payload.get("hit_count", 0)),
        source_observation_ids=tuple(str(item) for item in payload.get("source_observation_ids", ())),
    )


def load_candidate_exploration_state(path: Path) -> CandidateExplorationState:
    path = Path(path)
    if not path.exists():
        return CandidateExplorationState()
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate exploration state: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate exploration state must be a JSON object")
    return CandidateExplorationState(
        decisions=tuple(_load_decision(item) for item in payload.get("decisions", ())),
        legacy_confirmed_stand_ids=frozenset(str(item) for item in payload.get("confirmed_stand_ids", ())),
        legacy_rejected_stand_ids=frozenset(str(item) for item in payload.get("rejected_stand_ids", ())),
    )


def write_candidate_exploration_state(path: Path, state: CandidateExplorationState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_json_dict(), indent=2, sort_keys=True) + "\n")
