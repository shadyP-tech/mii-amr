"""Content-bound authorization for one small, current-candidate inspection turn."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MissionLegKind,
    load_mission_leg_motion_authorization,
    mission_leg_motion_authorization_sha256,
)


PERMIT_HASH = "candidate_centering_permit_sha256"
RESULT_HASH = "candidate_centering_result_sha256"
MAX_TURN_RAD = math.radians(6.0)
MAX_TOTAL_TRAVEL_RAD = math.radians(12.0)
STOP_TOLERANCE_RAD = math.radians(0.3)
MAX_ADVISORY_AGE_SEC = 5.0


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_centering_permit(payload: Mapping[str, object]) -> dict[str, object]:
    """Revalidate master scope, observation identity, and the spent view budget."""
    permit = dict(payload)
    if permit.get("schema_version") != 1 or permit.get("purpose") != "candidate_centering":
        raise ValueError("invalid candidate centering permit contract")
    master = load_mission_leg_motion_authorization(Path(str(permit["master_authorization_path"])))
    if (master.scope_text != MISSION_LEG_MOTION_AUTHORIZATION_SCOPE
            or MissionLegKind.CANDIDATE_PREAPPROACH not in master.allowed_leg_kinds):
        raise ValueError("mission RUN does not authorize candidate centering")
    if permit.get("master_authorization_sha256") != mission_leg_motion_authorization_sha256(master):
        raise ValueError("candidate centering master authorization changed")
    for field in ("session_id", "robot_id", "namespace", "cmd_vel_topic"):
        if permit.get(field) != getattr(master, field):
            raise ValueError(f"candidate centering {field} mismatch")
    for field in ("candidate_id", "view_id", "run_id", "result_path", "controller_trace_path"):
        if not isinstance(permit.get(field), str) or not permit[field]:
            raise ValueError(f"candidate centering {field} is missing")
    advisory = permit.get("advisory")
    if not isinstance(advisory, dict) or advisory.get("candidate_uid") != permit["candidate_id"]:
        raise ValueError("candidate centering advisory candidate mismatch")
    # The geometry layer checks calibration and the complete numerical derivation.
    from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory
    validate_camera_centering_advisory(advisory)
    if advisory.get("motion_authorized") is not False:
        raise ValueError("observation must not itself authorize motion")
    turn = finite_number(permit.get("signed_turn_rad"), "signed_turn_rad")
    requested = finite_number(advisory.get("requested_yaw_rad"), "requested_yaw_rad")
    if not STOP_TOLERANCE_RAD < abs(turn) <= MAX_TURN_RAD + 1e-12:
        raise ValueError("centering turn must be at most six degrees")
    if turn * requested <= 0.0 or abs(turn) > abs(requested) + 1e-12:
        raise ValueError("centering turn exceeds the measured advisory")
    if type(permit.get("turn_index")) is not int or permit["turn_index"] not in (0, 1):
        raise ValueError("centering allows at most two turns per view")
    spent = 0.0
    if permit["turn_index"] == 1:
        previous_path = Path(str(permit.get("previous_result_path", "")))
        previous = load_candidate_centering_result(previous_path)
        if payload_sha256(previous) != permit.get("previous_result_sha256"):
            raise ValueError("previous centering result changed")
        for field in ("session_id", "candidate_id", "view_id"):
            if previous.get(field) != permit[field]:
                raise ValueError(f"previous centering {field} mismatch")
        if previous.get("turn_index") != 0 or previous.get("status") != "completed":
            raise ValueError("previous centering turn did not complete")
        if advisory["image_stamp_sec"] <= previous["stopped_at_sec"] or advisory["scan_stamp_sec"] <= previous["stopped_at_sec"]:
            raise ValueError("centering advisory predates the previous turn stop")
        spent = finite_number(previous["total_angular_travel_rad"], "previous travel")
    elif permit.get("previous_result_path") or permit.get("previous_result_sha256"):
        raise ValueError("first centering turn must not have a predecessor")
    remaining = finite_number(permit.get("remaining_travel_rad"), "remaining_travel_rad")
    if remaining <= 0 or remaining > MAX_TOTAL_TRAVEL_RAD - spent + 1e-12:
        raise ValueError("centering cumulative travel budget is invalid")
    if abs(turn) + STOP_TOLERANCE_RAD > remaining + 1e-12:
        raise ValueError("centering turn leaves no stopping travel reserve")
    clearance = finite_number(permit.get("minimum_clearance_m"), "minimum_clearance_m")
    if clearance < 0.20:
        raise ValueError("centering clearance cannot be below 0.20 m")
    if permit.get("previous_angular_travel_rad") != spent:
        raise ValueError("centering spent budget mismatch")
    if not 5.0 <= finite_number(permit.get("timeout_sec"), "timeout_sec") <= 15.0:
        raise ValueError("centering timeout must be between five and fifteen seconds")
    return permit


def write_candidate_centering_permit(path: Path, payload: Mapping[str, object]) -> str:
    return write_content_hashed_json(path, validate_centering_permit(payload), hash_field=PERMIT_HASH)


def load_candidate_centering_permit(path: Path) -> dict[str, object]:
    return validate_centering_permit(load_content_hashed_json(path, hash_field=PERMIT_HASH))


def load_candidate_centering_result(path: Path, *, permit_path: Path | None = None) -> dict[str, object]:
    result = load_content_hashed_json(path, hash_field=RESULT_HASH)
    if result.get("schema_version") != 1 or result.get("purpose") != "candidate_centering":
        raise ValueError("invalid centering result contract")
    for key in ("actual_angular_travel_rad", "total_angular_travel_rad", "maximum_translation_m", "stopped_at_sec"):
        if finite_number(result.get(key), key) < 0:
            raise ValueError(f"negative centering result {key}")
    if result.get("translation_commanded") is not False:
        raise ValueError("centering result commanded translation")
    if result.get("status") == "completed":
        if (result.get("stationary_odom", {}).get("accepted") is not True
                or result["maximum_translation_m"] > 0.01
                or result["total_angular_travel_rad"] > MAX_TOTAL_TRAVEL_RAD + 1e-12
                or abs(finite_number(result.get("final_yaw_error_rad"), "final_yaw_error_rad")) > STOP_TOLERANCE_RAD
                or int(result.get("zero_command_count", 0)) < 10):
            raise ValueError("centering result lacks a bounded stopped pose")
    if permit_path is not None:
        permit = load_candidate_centering_permit(permit_path)
        if result.get("permit_sha256") != payload_sha256(permit):
            raise ValueError("centering result permit hash mismatch")
        for field in ("session_id", "candidate_id", "view_id", "turn_index", "signed_turn_rad", "run_id"):
            if result.get(field) != permit[field]:
                raise ValueError(f"centering result {field} mismatch")
        if abs(result["total_angular_travel_rad"] - result["actual_angular_travel_rad"] - permit["previous_angular_travel_rad"]) > 1e-12:
            raise ValueError("centering result total travel mismatch")
    return result
