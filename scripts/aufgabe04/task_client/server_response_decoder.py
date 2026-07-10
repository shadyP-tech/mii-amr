"""Pure FastAPI response normalization for Aufgabe 04."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from .models import QrMapping, RobotPlan, RobotStatus


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid string field: {key}")
    return value.strip()


def _int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing or invalid integer field: {key}")
    return value


def _str_tuple(payload: Mapping[str, object], key: str) -> Tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"missing or invalid list field: {key}")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"invalid string item in field: {key}")
        result.append(item.strip().upper())
    return tuple(result)


def decode_robot_statuses(payload: Iterable[Mapping[str, object]]) -> Tuple[RobotStatus, ...]:
    statuses = []
    for item in payload:
        statuses.append(
            RobotStatus(
                robot_id=_required_str(item, "robot_id"),
                mission_id=_required_str(item, "mission_id"),
                state=_required_str(item, "state"),
                target=_required_str(item, "target").upper(),
                last_qr=_required_str(item, "last_qr").upper(),
                cargo=_required_str(item, "cargo"),
                completed_jobs=_int_field(item, "completed_jobs"),
                score=_int_field(item, "score"),
                penalties=_int_field(item, "penalties"),
                last_seen_at=_required_str(item, "last_seen_at"),
                charging_visits=_int_field(item, "charging_visits"),
                raw=dict(item),
            )
        )
    return tuple(statuses)


def _decode_qr_mappings(payload: object) -> Tuple[QrMapping, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("missing or invalid list field: qr_mappings")
    mappings = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("invalid qr_mappings item")
        mappings.append(
            QrMapping(
                robot_id=_required_str(item, "robot_id"),
                qr_code_id=_required_str(item, "qr_code_id").upper(),
                station_id=_required_str(item, "station_id").upper(),
                station_type=_required_str(item, "station_type"),
                display_name=_required_str(item, "display_name"),
                raw=dict(item),
            )
        )
    return tuple(mappings)


def decode_robot_plans(payload: Iterable[Mapping[str, object]]) -> Tuple[RobotPlan, ...]:
    plans = []
    for item in payload:
        plans.append(
            RobotPlan(
                robot_id=_required_str(item, "robot_id"),
                mode=_required_str(item, "mode"),
                processing_sequence=_str_tuple(item, "processing_sequence"),
                plan_steps=_str_tuple(item, "plan_steps"),
                expanded_path=_str_tuple(item, "expanded_path"),
                qr_mappings=_decode_qr_mappings(item.get("qr_mappings")),
                next_job_index=_int_field(item, "next_job_index"),
                next_step_index=_int_field(item, "next_step_index"),
                generated_at=_required_str(item, "generated_at"),
                raw=dict(item),
            )
        )
    return tuple(plans)

