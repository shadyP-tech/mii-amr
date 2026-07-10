"""In-memory FastAPI client test double for Aufgabe 04 dry runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FakeFastApiClient:
    health_payload: Any
    status_payload: Any
    plans_payload: Any

    def health(self) -> Any:
        return self.health_payload

    def fetch_admin_status(self) -> Any:
        return self.status_payload

    def fetch_robot_plans(self) -> Any:
        return self.plans_payload

