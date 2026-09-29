"""Authenticate a spent tour execution without changing its terminal status."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, write_content_hashed_json

TERMINAL_HASH_FIELD = "tour_terminal_evidence_sha256"


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def is_replannable_tour_stop(outcome) -> bool:
    """Only geometric blockage can request a fresh stopped detour plan."""
    if outcome.status != "stopped" or type(outcome.returncode) is not int or outcome.returncode == 0:
        return False
    details = outcome.stop_details
    if not isinstance(details, Mapping):
        return False
    if outcome.stop_reason == "stored pose tour route blocked":
        point = details.get("blocked_point_odom", {})
        return (
            details.get("source") == "stored_pose_tour_obstacle_monitor"
            and details.get("fault_code") == "stored_pose_tour_route_blocked"
            and isinstance(details.get("execution_frame"), str) and bool(details["execution_frame"])
            and type(details.get("confirmed_distinct_scans")) is int
            and details["confirmed_distinct_scans"] >= 2
            and _positive(details.get("scan_stamp_sec"))
            and _positive(details.get("previous_scan_stamp_sec"))
            and details["scan_stamp_sec"] > details["previous_scan_stamp_sec"]
            and isinstance(point, Mapping)
            and all(type(point.get(key)) in (int, float) and math.isfinite(point[key]) for key in ("x_m", "y_m"))
        )
    if outcome.stop_reason == "obstacle too close":
        return (
            details.get("source") == "global_scan"
            and _valid_ranges(details)
            and _positive(details.get("threshold_m"))
            and details["nearest_valid_range_m"] < details["threshold_m"]
        )
    if outcome.stop_reason == "clearance-limited motion floor":
        front = details.get("front_clearance", {})
        return (
            details.get("source") == "linear_motion_floor"
            and details.get("zero_hold_required") is True
            and isinstance(front, Mapping) and front.get("source") == "front_sector"
            and _valid_ranges(front)
            and type(details.get("front_clearance_scale")) in (int, float)
            and 0 <= details["front_clearance_scale"] < 1
        )
    return False


def _valid_ranges(value: Mapping) -> bool:
    return (type(value.get("valid_sample_count")) is int and value["valid_sample_count"] > 0
            and _positive(value.get("nearest_valid_range_m")))


def write_tour_terminal_evidence(path: Path, *, outcome, request, stage_index: int,
                                 replan_count: int, final_stage: bool,
                                 arrival_path: Path | None = None,
                                 scan_capture_path: Path | None = None) -> tuple[Path, str]:
    """Seal one real child outcome and its source-log byte interval."""
    from .mission_leg_motion_consumption import default_mission_leg_motion_consumption_receipt_path
    from .mission_leg_motion_permit import file_sha256, load_mission_leg_motion_permit, mission_leg_motion_permit_sha256
    permit_path = Path(request.permit_json_path)
    permit = load_mission_leg_motion_permit(permit_path)
    for name in ("run_id", "session_id", "mission_leg_kind", "mission_leg_index", "target_id"):
        if getattr(request, name) != getattr(permit, name):
            raise ValueError(f"tour terminal request {name} differs from permit")
    if outcome.run_id != permit.run_id:
        raise ValueError("tour terminal outcome run differs from permit")
    receipt_path = default_mission_leg_motion_consumption_receipt_path(permit_path)
    log_path = Path(outcome.semantic_log_path)
    # Hash the original bounded byte interval: appending subsequent child logs
    # is legal, changing or truncating this execution's evidence is not.
    file_sha256(log_path)
    log_bytes = log_path.read_bytes()
    start = outcome.semantic_log_start_offset
    if type(start) is not int or not 0 <= start < len(log_bytes):
        raise ValueError("tour terminal semantic log offset is invalid")
    payload = {
        "schema_version": 1, "artifact_kind": "stored_pose_tour_terminal",
        "tour_id": permit.session_id, "visit_index": permit.mission_leg_index // 6,
        "execution_index": permit.mission_leg_index % 6,
        "stage_index": stage_index, "replan_count": replan_count, "final_stage": final_stage,
        "run_id": outcome.run_id, "target_id": permit.target_id,
        "status": outcome.status, "returncode": outcome.returncode,
        "stop_reason": outcome.stop_reason, "stop_details": outcome.stop_details,
        "motion_published": outcome.motion_published,
        "permit_json": str(permit_path), "permit_sha256": mission_leg_motion_permit_sha256(permit),
        "receipt_json": str(receipt_path), "receipt_sha256": file_sha256(receipt_path),
        "semantic_log_jsonl": str(log_path), "semantic_log_start_offset": start,
        "semantic_log_end_offset": len(log_bytes),
        "semantic_log_slice_sha256": hashlib.sha256(log_bytes[start:]).hexdigest(),
        "arrival_json": str(arrival_path) if arrival_path is not None else "",
        "arrival_sha256": file_sha256(arrival_path) if arrival_path is not None else "",
        "scan_capture_json": str(scan_capture_path) if scan_capture_path is not None else "",
        "scan_capture_sha256": file_sha256(scan_capture_path) if scan_capture_path is not None else "",
    }
    _validate_terminal(payload)
    path = Path(path)
    write_content_hashed_json(path, payload, hash_field=TERMINAL_HASH_FIELD)
    return path, file_sha256(path)


def load_tour_terminal_evidence(path: Path) -> dict[str, object]:
    payload = load_content_hashed_json(Path(path), hash_field=TERMINAL_HASH_FIELD)
    _validate_terminal(payload)
    return payload


def _validate_terminal(value: Mapping) -> None:
    from .mission_leg_motion_consumption import load_mission_leg_motion_consumption_receipt
    from .mission_leg_motion_permit import (
        MissionLegKind, TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, file_sha256,
        load_mission_leg_motion_authorization, load_mission_leg_motion_permit,
        mission_leg_motion_permit_sha256,
    )
    from .tour_replan_binding import validate_tour_navigation
    if value.get("schema_version") != 1 or value.get("artifact_kind") != "stored_pose_tour_terminal":
        raise ValueError("tour terminal artifact identity mismatch")
    permit = load_mission_leg_motion_permit(Path(value["permit_json"]))
    master = load_mission_leg_motion_authorization(Path(permit.master_authorization_path))
    if master.scope_text != TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE or permit.mission_leg_kind is not MissionLegKind.STORED_POSE_TOUR:
        raise ValueError("tour terminal requires explicit obstacle-detour authority")
    if mission_leg_motion_permit_sha256(permit) != value["permit_sha256"]:
        raise ValueError("tour terminal permit hash mismatch")
    if file_sha256(Path(value["receipt_json"])) != value["receipt_sha256"]:
        raise ValueError("tour terminal consumed receipt hash mismatch")
    receipt = load_mission_leg_motion_consumption_receipt(Path(value["receipt_json"]))
    if receipt.mission_leg_motion_permit_sha256 != value["permit_sha256"]:
        raise ValueError("tour terminal receipt belongs to a different permit")
    for name, expected in (("tour_id", permit.session_id), ("run_id", permit.run_id), ("target_id", permit.target_id),
                           ("visit_index", permit.mission_leg_index // 6), ("execution_index", permit.mission_leg_index % 6)):
        if value.get(name) != expected:
            raise ValueError(f"tour terminal {name} identity mismatch")
    if file_sha256(Path(permit.diagnostics_path)) != permit.diagnostics_sha256:
        raise ValueError("tour terminal route diagnostics changed")
    metadata = json.loads(Path(permit.diagnostics_path).read_text())["metadata"]
    nav = validate_tour_navigation(metadata.get("tour_navigation"))
    for name in ("stage_index", "replan_count", "execution_index"):
        if type(value.get(name)) is not int or value[name] != nav[name]:
            raise ValueError(f"tour terminal {name} differs from the sealed execution")
    if type(value.get("final_stage")) is not bool or metadata["return_to_start_stage"]["final_stage"] != value["final_stage"]:
        raise ValueError("tour terminal final stage differs from the sealed route")
    finished = _validate_source_events(value, permit, receipt)
    if value["status"] == "completed":
        _validate_arrival(value)
    elif is_replannable_tour_stop(SimpleNamespace(**value)):
        _validate_stopped_capture(value, finished)
    else:
        raise ValueError("tour terminal is not an eligible completed arrival or obstacle stop")


def _validate_source_events(value: Mapping, permit, receipt) -> Mapping:
    from .mission_leg_motion_consumption import mission_leg_motion_consumption_receipt_sha256
    path = Path(value["semantic_log_jsonl"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("tour terminal semantic log must be a normal file")
    raw = path.read_bytes()
    start, end = value["semantic_log_start_offset"], value["semantic_log_end_offset"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
        raise ValueError("tour terminal semantic log interval is invalid")
    data = raw[start:end]
    if hashlib.sha256(data).hexdigest() != value["semantic_log_slice_sha256"]:
        raise ValueError("tour terminal source log bytes changed")
    events = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("tour terminal semantic log contains a non-object")
    events = [event for event in events if event.get("run_id") == permit.run_id]
    names = [event.get("event") for event in events]
    required = ("mission_leg_motion_permit_consumed", "motion_started", "run_finished")
    if any(names.count(name) != 1 for name in required):
        raise ValueError("tour terminal requires one consumed, started and finished source event")
    terminal_indices = [i for i, name in enumerate(names) if name in {"motion_completed", "safety_stop", "preflight_failed"}]
    if len(terminal_indices) != 1:
        raise ValueError("tour terminal source outcome is ambiguous")
    consumed_i, started_i, finished_i = (names.index(name) for name in required)
    terminal_i = terminal_indices[0]
    if not consumed_i < started_i < terminal_i < finished_i:
        raise ValueError("tour terminal child event ordering mismatch")
    consumed, started, terminal, finished = (events[i] for i in (consumed_i, started_i, terminal_i, finished_i))
    for event in (consumed, started, terminal):
        if (event.get("mission_leg_kind") != "stored_pose_tour"
                or type(event.get("mission_leg_index")) is not int or event["mission_leg_index"] != permit.mission_leg_index
                or event.get("target_id") != permit.target_id):
            raise ValueError("tour terminal source event mission identity mismatch")
    if (consumed.get("mission_leg_motion_permit_sha256") != value["permit_sha256"]
            or consumed.get("mission_leg_motion_consumption_receipt_sha256") != mission_leg_motion_consumption_receipt_sha256(receipt)):
        raise ValueError("tour terminal source consumed event differs from actual receipt")
    if started.get("motion_published") is not False or started.get("event_semantics") != "child_execution_attempt_started_before_follower":
        raise ValueError("tour terminal child execution boundary is invalid")
    for name in ("status", "stop_reason", "motion_published"):
        if terminal.get(name) != value[name]:
            raise ValueError(f"tour terminal {name} differs from original child event")
    if (type(value["motion_published"]) is not bool or type(value["returncode"]) is not int
            or (value["status"] == "completed") != (value["returncode"] == 0)
            or terminal.get("event") != ("motion_completed" if value["status"] == "completed" else "safety_stop")
            or terminal.get("stop_details", {}) != value["stop_details"]
            or finished.get("final_status") != value["status"]):
        raise ValueError("tour terminal genuine child outcome mismatch")
    return finished


def _validate_arrival(value: Mapping) -> None:
    from .mission_leg_motion_permit import file_sha256
    path = Path(value["arrival_json"])
    if file_sha256(path) != value["arrival_sha256"]:
        raise ValueError("tour terminal arrival evidence hash mismatch")
    arrival = load_content_hashed_json(path, hash_field="stored_pose_tour_leg_arrival_sha256")
    for name in ("tour_id", "visit_index", "stage_index", "run_id", "final_stage", "status", "returncode", "motion_published"):
        if arrival.get(name) != value[name]:
            raise ValueError(f"tour terminal arrival {name} mismatch")
    if arrival.get("artifact_kind") != "stored_pose_tour_leg_arrival" or arrival.get("arrival_verified") is not True:
        raise ValueError("tour terminal needs a verified intermediate or final arrival")
    if arrival.get("candidate_uid") != value["target_id"]:
        raise ValueError("tour terminal arrival candidate differs from the stored target")
    for name, maximum in (("position_error_m", .08), ("heading_error_rad", .15)):
        error = arrival.get(name)
        if type(error) not in (float, int) or not math.isfinite(error) or not 0 <= error <= maximum:
            raise ValueError("tour terminal arrival is outside the admitted tolerance")
    progress = arrival.get("odom_progress_m")
    if not value["final_stage"] and (
        type(progress) not in (float, int) or not math.isfinite(progress) or progress < .10
    ):
        raise ValueError("tour intermediate arrival lacks measured odom progress")


def _validate_stopped_capture(value: Mapping, finished: Mapping) -> None:
    from .mission_leg_motion_permit import file_sha256
    from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import capture_payload, HASH_FIELD
    path = Path(value["scan_capture_json"])
    if file_sha256(path) != value["scan_capture_sha256"]:
        raise ValueError("tour terminal stopped scan capture hash mismatch")
    capture = load_content_hashed_json(path, hash_field=HASH_FIELD)
    replay = capture_payload(capture["scans"], **{name: capture[name] for name in (
        "tour_id", "odom_frame", "base_frame", "scan_frame", "captured_at_unix_sec",
    )})
    if capture != replay or capture["tour_id"] != value["tour_id"]:
        raise ValueError("tour terminal stopped scan capture identity mismatch")
    if value["stop_reason"] == "stored pose tour route blocked" and (
        value["stop_details"].get("execution_frame") != capture["odom_frame"]
        or value["stop_details"].get("scan_frame") != capture["scan_frame"]
    ):
        raise ValueError("tour obstacle monitor and stopped scan capture frame mismatch")
    finished_time = datetime.fromisoformat(finished["timestamp"])
    if finished_time.tzinfo is None or capture["scans"][0]["stamp_sec"] <= finished_time.timestamp():
        raise ValueError("tour detour scans must be captured after the prior child stopped")
