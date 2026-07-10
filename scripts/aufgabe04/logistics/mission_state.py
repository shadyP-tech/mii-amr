"""Pure single-robot mission state transitions."""

from dataclasses import dataclass, replace
from typing import Tuple

from .models import MissionState, PuckState


@dataclass(frozen=True)
class MissionSnapshot:
    state: MissionState = MissionState.IDLE
    station_ids: Tuple[str, ...] = ()
    current_index: int = 0
    puck_state: PuckState = PuckState.UNKNOWN
    failure_reason: str = ""

    @property
    def current_station_id(self) -> str:
        if self.current_index >= len(self.station_ids):
            return ""
        return self.station_ids[self.current_index]


def start_from_station_order(station_ids: Tuple[str, ...]) -> MissionSnapshot:
    if not station_ids:
        raise ValueError("mission requires at least one station")
    return MissionSnapshot(
        state=MissionState.ROUTING,
        station_ids=station_ids,
        current_index=0,
        puck_state=PuckState.NOT_HELD,
    )


def mark_visit_complete(snapshot: MissionSnapshot) -> MissionSnapshot:
    next_index = snapshot.current_index + 1
    if next_index >= len(snapshot.station_ids):
        return replace(snapshot, state=MissionState.COMPLETED, current_index=next_index)
    return replace(snapshot, state=MissionState.ROUTING, current_index=next_index)


def fail_mission(snapshot: MissionSnapshot, reason: str) -> MissionSnapshot:
    return replace(snapshot, state=MissionState.FAILED, failure_reason=reason)

