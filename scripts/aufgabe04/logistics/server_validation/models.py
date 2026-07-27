"""Validated FastAPI task models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from scripts.aufgabe04.logistics.navigation_request import StationNavigationRequest


def server_order_sha256(
    *,
    robot_id: str,
    mission_id: str,
    target_station: str,
    plan_step_index: int,
    ordered_station_ids: Sequence[str],
    plan_generated_at_sec: float,
) -> str:
    """Bind the exact remaining server order and the plan version that produced it."""

    payload = {
        "mission_id": mission_id,
        "ordered_station_ids": list(ordered_station_ids),
        "plan_generated_at_sec": float(plan_generated_at_sec),
        "plan_step_index": plan_step_index,
        "robot_id": robot_id,
        "target_station": target_station,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    ordered_station_ids: Tuple[str, ...] = ()
    status_observed_at_sec: Optional[float] = None
    plan_generated_at_sec: Optional[float] = None
    validated_at_sec: Optional[float] = None
    order_sha256: str = ""
    source_plan_sha256: str = ""

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
