"""Pure validation for FastAPI Aufgabe 04 task snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from scripts.aufgabe04.task_client.models import RobotPlan, RobotStatus, ServerTaskSnapshot

from .models import ValidatedServerTask


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp field {field_name}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _select_one_robot(items: Iterable[object], robot_id: str, item_name: str):
    matches = [item for item in items if getattr(item, "robot_id") == robot_id]
    if not matches:
        raise ValueError(f"no {item_name} found for robot_id {robot_id}")
    if len(matches) > 1:
        raise ValueError(f"multiple {item_name} entries found for robot_id {robot_id}")
    return matches[0]


def resolve_scanned_station(plan: RobotPlan, qr_id: str) -> str:
    normalized = qr_id.strip().upper()
    for mapping in plan.qr_mappings:
        if mapping.qr_code_id == normalized:
            return mapping.station_id
    known_stations = set(plan.expanded_path) | {mapping.station_id for mapping in plan.qr_mappings}
    if normalized in known_stations:
        return normalized
    raise ValueError(f"unknown QR or station id: {normalized}")


def build_server_task_snapshot(
    *,
    robot_id: str,
    scanned_qr_id: str,
    statuses: Sequence[RobotStatus],
    plans: Sequence[RobotPlan],
) -> ServerTaskSnapshot:
    plan = _select_one_robot(plans, robot_id, "robot plan")
    status = _select_one_robot(statuses, robot_id, "robot status")
    resolved_station = resolve_scanned_station(plan, scanned_qr_id)
    return ServerTaskSnapshot(
        robot_id=robot_id,
        status=status,
        plan=plan,
        scanned_qr_id=scanned_qr_id.strip().upper(),
        resolved_station_id=resolved_station,
    )


def validate_server_task(
    snapshot: ServerTaskSnapshot,
    *,
    local_station_ids: Iterable[str],
    now: datetime | None = None,
    max_status_age_sec: float = 300.0,
    max_plan_age_sec: float = 3600.0,
) -> ValidatedServerTask:
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local_stations = {station.strip().upper() for station in local_station_ids}
    if not local_stations:
        raise ValueError("local station set must not be empty")
    if snapshot.status.robot_id != snapshot.robot_id:
        raise ValueError("status robot_id does not match configured robot")
    if snapshot.plan.robot_id != snapshot.robot_id:
        raise ValueError("plan robot_id does not match configured robot")
    status_age = (now_utc - _parse_timestamp(snapshot.status.last_seen_at, field_name="last_seen_at")).total_seconds()
    if status_age < 0:
        raise ValueError("status timestamp is in the future")
    if status_age > max_status_age_sec:
        raise ValueError("robot status is stale")
    plan_age = (now_utc - _parse_timestamp(snapshot.plan.generated_at, field_name="generated_at")).total_seconds()
    if plan_age < 0:
        raise ValueError("plan timestamp is in the future")
    if plan_age > max_plan_age_sec:
        raise ValueError("robot plan is stale")
    if snapshot.status.target not in local_stations:
        raise ValueError(f"target station is not in local station map: {snapshot.status.target}")
    known_server_stations = set(snapshot.plan.expanded_path) | {
        mapping.station_id for mapping in snapshot.plan.qr_mappings
    }
    if snapshot.status.target not in known_server_stations:
        raise ValueError(f"target station is not in server plan: {snapshot.status.target}")
    if snapshot.resolved_station_id not in known_server_stations:
        raise ValueError(f"resolved scanned station is not in server plan: {snapshot.resolved_station_id}")
    evidence = {
        "scanned_qr_id": snapshot.scanned_qr_id,
        "resolved_station_id": snapshot.resolved_station_id,
        "server_target": snapshot.status.target,
        "status_age_sec": round(status_age, 3),
        "plan_age_sec": round(plan_age, 3),
        "admin_observed_endpoints": True,
    }
    return ValidatedServerTask(
        robot_id=snapshot.robot_id,
        mission_id=snapshot.status.mission_id,
        state=snapshot.status.state,
        last_qr=snapshot.status.last_qr,
        resolved_current_station=snapshot.resolved_station_id,
        target_station=snapshot.status.target,
        cargo=snapshot.status.cargo,
        plan_step_index=snapshot.plan.next_step_index,
        evidence=evidence,
    )

