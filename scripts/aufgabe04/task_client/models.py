"""Typed FastAPI task-client models for Aufgabe 04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class FastApiConfig:
    base_url: str
    robot_id: str
    timeout_sec: float = 3.0
    scanned_qr_endpoint_template: Optional[str] = None


@dataclass(frozen=True)
class RobotStatus:
    robot_id: str
    mission_id: str
    state: str
    target: str
    last_qr: str
    cargo: str
    completed_jobs: int
    score: int
    penalties: int
    last_seen_at: str
    charging_visits: int
    raw: Mapping[str, object]


@dataclass(frozen=True)
class QrMapping:
    robot_id: str
    qr_code_id: str
    station_id: str
    station_type: str
    display_name: str
    raw: Mapping[str, object]


@dataclass(frozen=True)
class RobotPlan:
    robot_id: str
    mode: str
    processing_sequence: Tuple[str, ...]
    plan_steps: Tuple[str, ...]
    expanded_path: Tuple[str, ...]
    qr_mappings: Tuple[QrMapping, ...]
    next_job_index: int
    next_step_index: int
    generated_at: str
    raw: Mapping[str, object]


@dataclass(frozen=True)
class ServerTaskSnapshot:
    robot_id: str
    status: RobotStatus
    plan: RobotPlan
    scanned_qr_id: str
    resolved_station_id: str

