"""Content-bound authorization for one small, current-candidate inspection turn."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    LEGACY_BOUNDED_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_CENTERING_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
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
    if (master.scope_text not in {
            MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            LEGACY_BOUNDED_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            LEGACY_CENTERING_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
        }
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
    validated_advisory = validate_camera_centering_advisory(advisory)
    recovery = validated_advisory.arrival_recovery
    from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import RECOVERY_STEP_RAD, RECOVERY_TRAVEL_RAD, RECOVERY_TURNS
    previous = None
    index = permit.get('turn_index')
    if type(index) is not int or not 0 <= index < RECOVERY_TURNS:
        raise ValueError('invalid centering turn index')
    if index > 0:
        previous_path = Path(str(permit.get('previous_result_path', '')))
        # Check the decreasing index before following receipt links.
        header = load_content_hashed_json(previous_path, hash_field=RESULT_HASH)
        if header.get('turn_index') != index-1:
            raise ValueError('previous centering turn index does not decrease')
        previous = load_candidate_centering_result(previous_path)
        recovery = recovery or previous.get('arrival_recovery') is True
    if recovery and master.scope_text != MISSION_LEG_MOTION_AUTHORIZATION_SCOPE:
        raise ValueError('mission RUN does not authorize extended arrival recovery')
    if permit.get('arrival_recovery', False) != recovery:
        raise ValueError('centering arrival recovery authority mismatch')
    total_limit = RECOVERY_TRAVEL_RAD if recovery else MAX_TOTAL_TRAVEL_RAD
    step_limit = RECOVERY_STEP_RAD if recovery and permit.get("turn_index") == 0 else MAX_TURN_RAD
    if advisory.get("motion_authorized") is not False:
        raise ValueError("observation must not itself authorize motion")
    from scripts.aufgabe04.navigation.foundation.ros_runtime_config import RuntimeConfig, resolve_runtime_config
    config = permit.get("runtime_config")
    if not isinstance(config, dict) or config.get("use_sim_time") is not False:
        raise ValueError("centering requires a physical runtime configuration")
    resolved = resolve_runtime_config(RuntimeConfig(**config))
    if (resolved.namespace != permit["namespace"] or resolved.cmd_vel_topic != permit["cmd_vel_topic"]
            or resolved.base_frame != advisory["base_from_camera"]["parent_frame"]):
        raise ValueError("centering runtime topic/frame binding mismatch")
    if finite_number(permit.get("maximum_angular_speed_radps"), "maximum angular speed") <= 0:
        raise ValueError("centering angular speed must be positive")
    turn = finite_number(permit.get("signed_turn_rad"), "signed_turn_rad")
    requested = finite_number(advisory.get("requested_yaw_rad"), "requested_yaw_rad")
    if not STOP_TOLERANCE_RAD < abs(turn) <= step_limit + 1e-12:
        raise ValueError("centering turn exceeds its authorized step bound")
    if turn * requested <= 0.0 or abs(turn) > abs(requested) + 1e-12:
        raise ValueError("centering turn exceeds the measured advisory")
    if type(permit.get("turn_index")) is not int or not 0 <= permit["turn_index"] < (RECOVERY_TURNS if recovery else 2):
        raise ValueError("centering exceeds its authorized turn count")
    spent = 0.0
    if permit["turn_index"] > 0:
        previous_path = Path(str(permit.get("previous_result_path", "")))
        if payload_sha256(previous) != permit.get("previous_result_sha256"):
            raise ValueError("previous centering result changed")
        for field in ("session_id", "candidate_id", "view_id"):
            if previous.get(field) != permit[field]:
                raise ValueError(f"previous centering {field} mismatch")
        if previous.get("turn_index") != permit["turn_index"]-1 or previous.get("status") != "completed":
            raise ValueError("previous centering turn did not complete")
        if advisory["image_stamp_sec"] <= previous["stopped_at_sec"] or advisory["scan_stamp_sec"] <= previous["stopped_at_sec"]:
            raise ValueError("centering advisory predates the previous turn stop")
        spent = finite_number(previous["total_angular_travel_rad"], "previous travel")
    elif permit.get("previous_result_path") or permit.get("previous_result_sha256"):
        raise ValueError("first centering turn must not have a predecessor")
    remaining = finite_number(permit.get("remaining_travel_rad"), "remaining_travel_rad")
    if remaining <= 0 or remaining > total_limit - spent + 1e-12:
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
    result_limit = MAX_TOTAL_TRAVEL_RAD
    if result.get("arrival_recovery") is True:
        from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import RECOVERY_TRAVEL_RAD
        bound_path = Path(result['arrival_recovery_permit_path'])
        raw_bound = load_content_hashed_json(bound_path, hash_field=PERMIT_HASH)
        if raw_bound.get('turn_index') != result.get('turn_index'):
            raise ValueError('arrival recovery result turn index mismatch')
        bound = validate_centering_permit(raw_bound)
        if (bound['result_path'] != str(Path(path).resolve())
                or result.get('permit_sha256') != payload_sha256(bound)):
            raise ValueError('arrival recovery result permit binding mismatch')
        if bound.get('arrival_recovery') is not True:
            raise ValueError('arrival recovery result lacks validated epoch proof')
        result_limit = RECOVERY_TRAVEL_RAD
        permit_path = bound_path if permit_path is None else permit_path
    if result.get("status") == "completed":
        if (result.get("stationary_odom", {}).get("accepted") is not True
                or result["maximum_translation_m"] > 0.01
                or result["total_angular_travel_rad"] > result_limit + 1e-12
                or abs(finite_number(result.get("final_yaw_error_rad"), "final_yaw_error_rad")) > STOP_TOLERANCE_RAD
                or type(result.get("zero_command_count")) is not int
                or result["zero_command_count"] < 10):
            raise ValueError("centering result lacks a bounded stopped pose")
    if permit_path is not None:
        permit = (bound if result.get("arrival_recovery") is True and Path(permit_path) == bound_path
                  else load_candidate_centering_permit(permit_path))
        if result.get("permit_sha256") != payload_sha256(permit):
            raise ValueError("centering result permit hash mismatch")
        for field in ("session_id", "candidate_id", "view_id", "turn_index", "signed_turn_rad", "run_id"):
            if result.get(field) != permit[field]:
                raise ValueError(f"centering result {field} mismatch")
        if abs(result["total_angular_travel_rad"] - result["actual_angular_travel_rad"] - permit["previous_angular_travel_rad"]) > 1e-12:
            raise ValueError("centering result total travel mismatch")
        if result.get("status") == "completed" and (
                result["actual_angular_travel_rad"] > permit["remaining_travel_rad"] + 1e-12
                or result["actual_angular_travel_rad"] < abs(permit["signed_turn_rad"])-STOP_TOLERANCE_RAD
                or result["stopped_at_sec"] <= max(permit["advisory"]["image_stamp_sec"],
                                                  permit["advisory"]["scan_stamp_sec"])):
            raise ValueError("centering result exceeds its budget or predates observation")
    return result
