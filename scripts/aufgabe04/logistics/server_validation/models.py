"""Validated FastAPI task models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.logistics.navigation_request import StationNavigationRequest


@dataclass(frozen=True)
class ValidatedServerTask:
    robot_id: str
    mission_id: str
    state: str
    last_qr: str
    resolved_current_station: str
    target_station: str
    cargo: str
    plan_step_index: int
    evidence: Mapping[str, object]

    def to_navigation_request(self) -> StationNavigationRequest:
        return StationNavigationRequest(
            robot_id=self.robot_id,
            mission_id=self.mission_id,
            current_station_id=self.resolved_current_station,
            target_station_id=self.target_station,
            server_state=self.state,
            cargo=self.cargo,
            evidence=self.evidence,
        )

