"""Controller-neutral navigation request produced by task validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class StationNavigationRequest:
    robot_id: str
    mission_id: str
    current_station_id: str
    target_station_id: str
    server_state: str
    cargo: str
    evidence: Mapping[str, object]

