"""Immutable persistence for one freshly validated task-server snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.logistics.server_validation.models import (
    ValidatedServerTask,
    server_order_sha256,
)


VALIDATED_TASK_SNAPSHOT_SCHEMA_VERSION = 1
_HASH_FIELD = "task_snapshot_sha256"
_ROOT_FIELDS = frozenset({"schema_version", "task"})
_TASK_FIELDS = frozenset(
    {
        "robot_id",
        "mission_id",
        "state",
        "last_qr",
        "resolved_current_station",
        "target_station",
        "cargo",
        "plan_step_index",
        "evidence",
        "ordered_station_ids",
        "status_observed_at_sec",
        "plan_generated_at_sec",
        "validated_at_sec",
        "order_sha256",
        "source_plan_sha256",
    }
)


def validated_task_snapshot_sha256(task: ValidatedServerTask) -> str:
    return payload_sha256(_payload(task))


def write_validated_task_snapshot(path: Path, task: ValidatedServerTask) -> str:
    try:
        return write_content_hashed_json(
            path,
            _payload(task),
            hash_field=_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_validated_task_snapshot(path: Path) -> ValidatedServerTask:
    try:
        payload = load_content_hashed_json(path, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _ROOT_FIELDS:
        raise ValueError("validated task snapshot fields mismatch")
    task_payload = payload["task"]
    if not isinstance(task_payload, Mapping) or frozenset(task_payload) != _TASK_FIELDS:
        raise ValueError("validated task fields mismatch")
    evidence = task_payload["evidence"]
    ordered = task_payload["ordered_station_ids"]
    if not isinstance(evidence, Mapping):
        raise ValueError("validated task evidence must be an object")
    if not isinstance(ordered, list) or not all(
        isinstance(item, str) and item for item in ordered
    ):
        raise ValueError("ordered_station_ids must be a non-empty string array")
    try:
        task = ValidatedServerTask(
            robot_id=_string(task_payload["robot_id"], "robot_id"),
            mission_id=_string(task_payload["mission_id"], "mission_id"),
            state=_string(task_payload["state"], "state"),
            last_qr=_string(task_payload["last_qr"], "last_qr"),
            resolved_current_station=_string(
                task_payload["resolved_current_station"],
                "resolved_current_station",
            ),
            target_station=_string(task_payload["target_station"], "target_station"),
            cargo=_string(task_payload["cargo"], "cargo"),
            plan_step_index=_integer(
                task_payload["plan_step_index"], "plan_step_index"
            ),
            evidence=dict(evidence),
            ordered_station_ids=tuple(ordered),
            status_observed_at_sec=_number(
                task_payload["status_observed_at_sec"], "status_observed_at_sec"
            ),
            plan_generated_at_sec=_number(
                task_payload["plan_generated_at_sec"], "plan_generated_at_sec"
            ),
            validated_at_sec=_number(
                task_payload["validated_at_sec"], "validated_at_sec"
            ),
            order_sha256=_digest(task_payload["order_sha256"], "order_sha256"),
            source_plan_sha256=_digest(
                task_payload["source_plan_sha256"], "source_plan_sha256"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid validated task snapshot: {exc}") from exc
    expected_order_sha256 = server_order_sha256(
        robot_id=task.robot_id,
        mission_id=task.mission_id,
        target_station=task.target_station,
        plan_step_index=task.plan_step_index,
        ordered_station_ids=task.ordered_station_ids,
        plan_generated_at_sec=float(task.plan_generated_at_sec),
    )
    if task.order_sha256 != expected_order_sha256:
        raise ValueError("validated task order SHA-256 is inconsistent")
    return task


def _payload(task: ValidatedServerTask) -> dict[str, object]:
    if (
        task.status_observed_at_sec is None
        or task.plan_generated_at_sec is None
        or task.validated_at_sec is None
        or not task.ordered_station_ids
    ):
        raise ValueError("task must contain complete validation timestamps and order")
    expected_order_sha256 = server_order_sha256(
        robot_id=task.robot_id,
        mission_id=task.mission_id,
        target_station=task.target_station,
        plan_step_index=task.plan_step_index,
        ordered_station_ids=task.ordered_station_ids,
        plan_generated_at_sec=task.plan_generated_at_sec,
    )
    if task.order_sha256 != expected_order_sha256:
        raise ValueError("validated task order SHA-256 is inconsistent")
    _digest(task.source_plan_sha256, "source_plan_sha256")
    return {
        "schema_version": VALIDATED_TASK_SNAPSHOT_SCHEMA_VERSION,
        "task": {
            "robot_id": task.robot_id,
            "mission_id": task.mission_id,
            "state": task.state,
            "last_qr": task.last_qr,
            "resolved_current_station": task.resolved_current_station,
            "target_station": task.target_station,
            "cargo": task.cargo,
            "plan_step_index": task.plan_step_index,
            "evidence": dict(task.evidence),
            "ordered_station_ids": list(task.ordered_station_ids),
            "status_observed_at_sec": task.status_observed_at_sec,
            "plan_generated_at_sec": task.plan_generated_at_sec,
            "validated_at_sec": task.validated_at_sec,
            "order_sha256": task.order_sha256,
            "source_plan_sha256": task.source_plan_sha256,
        },
    }


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _digest(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text
