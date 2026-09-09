"""Pure FastAPI response normalization for Aufgabe 04."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from scripts.aufgabe04.stations.station_ids import canonical_qr_id, canonical_server_station_id

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
        result.append(canonical_server_station_id(item) if key == "expanded_path" else item.strip())
    return tuple(result)


def decode_robot_statuses(payload: Iterable[Mapping[str, object]]) -> Tuple[RobotStatus, ...]:
    statuses = []
    for item in payload:
        statuses.append(
            RobotStatus(
                robot_id=_required_str(item, "robot_id"),
                mission_id=_required_str(item, "mission_id"),
                state=_required_str(item, "state"),
                target=_required_str(item, "target"),
                last_qr=_required_str(item, "last_qr"),
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


def _decode_qr_mappings(payload: object, *, robot_id: str) -> Tuple[QrMapping, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("missing or invalid list field: qr_mappings")
    mappings = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("invalid qr_mappings item")
        mappings.append(
            QrMapping(
                robot_id=_required_str(item, "robot_id"),
                qr_code_id=canonical_qr_id(_required_str(item, "qr_code_id")),
                station_id=canonical_server_station_id(_required_str(item, "station_id")),
                station_type=_required_str(item, "station_type"),
                display_name=_required_str(item, "display_name"),
                raw=dict(item),
            )
        )
    if any(mapping.robot_id != robot_id for mapping in mappings):
        raise ValueError("qr_mappings entry belongs to another robot")
    if len({mapping.qr_code_id for mapping in mappings}) != len(mappings):
        raise ValueError("ambiguous duplicate qr_mappings QR identifier")
    if len({mapping.station_id for mapping in mappings}) != len(mappings):
        raise ValueError("ambiguous duplicate qr_mappings station identifier")
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
                qr_mappings=_decode_qr_mappings(item.get("qr_mappings"), robot_id=_required_str(item, "robot_id")),
                next_job_index=_int_field(item, "next_job_index"),
                next_step_index=_int_field(item, "next_step_index"),
                generated_at=_required_str(item, "generated_at"),
                raw=dict(item),
            )
        )
    return tuple(plans)

