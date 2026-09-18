"""Sealed parent/child boundary for a single camera-centering turn."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.aufgabe04.artifacts.content_store import payload_sha256, write_content_hashed_json
from scripts.aufgabe04.navigation.execution.candidate_centering_permit import (
    MAX_ADVISORY_AGE_SEC, RESULT_HASH, load_candidate_centering_permit,
    load_candidate_centering_result, write_candidate_centering_permit, finite_number,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    load_mission_leg_motion_authorization, mission_leg_motion_authorization_sha256,
)
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import RuntimeConfig, resolve_runtime_config
from scripts.aufgabe04.real_robot.configuration.profile import real_robot_profile_sha256
from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory


@dataclass(frozen=True)
class CandidateCenteringChildRequest:
    session_id: str
    output_dir: Path
    profile: object
    master_authorization_path: Path
    candidate_id: str
    view_id: str
    turn_index: int
    advisory: dict
    signed_turn_rad: float
    remaining_travel_rad: float
    minimum_clearance_m: float
    previous_result_path: Path | None = None


@dataclass(frozen=True)
class CandidateCenteringChildOutcome:
    result: dict
    result_path: Path
    permit_path: Path
    controller_trace_path: Path
    returncode: int


def validate_candidate_centering_dependencies():
    """Resolve both execution edges before a camera mission can start moving."""
    from scripts.aufgabe04.navigation.waypoint_follower.runtime import run_candidate_centering_motion
    entrypoint = Path(__file__).resolve().parents[4] / "scripts/aufgabe04/navigation/entrypoints/run_candidate_centering.py"
    if not entrypoint.is_file() or not callable(run_candidate_centering_motion):
        raise RuntimeError("candidate centering execution dependencies are unavailable")
    return entrypoint


def build_candidate_centering_permit(request):
    profile = request.profile
    validate_camera_centering_advisory(request.advisory,
        candidate_uid=request.candidate_id,
        stream_id=f"{request.session_id}_{request.candidate_id}",
        robot_profile_sha256=real_robot_profile_sha256(profile),
        calibration_profile_sha256=profile.calibration_profile_sha256,
        now_sec=time.time(), max_receipt_age_sec=MAX_ADVISORY_AGE_SEC)
    master_path = Path(request.master_authorization_path).resolve()
    master = load_mission_leg_motion_authorization(master_path)
    config = RuntimeConfig(use_sim_time=False, **{name: getattr(profile, name) for name in (
        "namespace", "scan_topic", "odom_topic", "cmd_vel_topic", "amcl_topic",
        "map_frame", "odom_frame", "base_frame", "localization_source")})
    resolved = resolve_runtime_config(config)
    previous_path = None if request.previous_result_path is None else Path(request.previous_result_path).resolve()
    previous = None if previous_path is None else load_candidate_centering_result(previous_path)
    root = Path(request.output_dir).resolve()
    return dict(schema_version=1, purpose="candidate_centering",
        master_authorization_path=str(master_path),
        master_authorization_sha256=mission_leg_motion_authorization_sha256(master),
        session_id=request.session_id, robot_id=profile.robot_id,
        namespace=resolved.namespace, cmd_vel_topic=resolved.cmd_vel_topic,
        runtime_config=asdict(config), maximum_angular_speed_radps=profile.max_angular_speed_radps,
        candidate_id=request.candidate_id, view_id=request.view_id, turn_index=request.turn_index,
        run_id=f"{request.session_id}_centering_{payload_sha256({'view': request.view_id})[:12]}_{request.turn_index}",
        result_path=str(root / "candidate_centering_result.json"),
        controller_trace_path=str(root / "controller_trace.jsonl"),
        advisory=request.advisory, signed_turn_rad=request.signed_turn_rad,
        remaining_travel_rad=request.remaining_travel_rad,
        previous_angular_travel_rad=0. if previous is None else previous["total_angular_travel_rad"],
        previous_result_path=None if previous_path is None else str(previous_path),
        previous_result_sha256=None if previous is None else payload_sha256(previous),
        minimum_clearance_m=max(.20, finite_number(request.minimum_clearance_m, "minimum clearance")), timeout_sec=15.)


def run_candidate_centering_child(request, *, run_process=None):
    run_process = subprocess.run if run_process is None else run_process
    entrypoint = validate_candidate_centering_dependencies()
    root = Path(request.output_dir).resolve()
    permit_path = root / "candidate_centering_permit.json"
    payload = build_candidate_centering_permit(request)
    result_path, trace_path = Path(payload["result_path"]), Path(payload["controller_trace_path"])
    for path in (permit_path, result_path, trace_path, root / "candidate_centering_failure.json"):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to reuse candidate centering artifacts: {path}")
    write_candidate_centering_permit(permit_path, payload)
    command = [sys.executable, str(entrypoint),
               "--permit", str(permit_path)]
    try:
        completed = run_process(command, check=False)
        result = load_candidate_centering_result(result_path, permit_path=permit_path)
        returncode = int(completed.returncode)
        if returncode != 0 or result.get("status") != "completed":
            raise RuntimeError("candidate centering stopped: " + str(result.get("stop_reason")))
        if result.get("motion_published") is not True or not trace_path.is_file() or trace_path.stat().st_size == 0:
            raise RuntimeError("candidate centering lacks motion/trace evidence")
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"candidate centering child failed; artifacts: {root}: {exc}") from exc
    return CandidateCenteringChildOutcome(result, result_path, permit_path, trace_path, returncode)


def _claim_turn(permit):
    # Identity-based, not output-path-based: copying a permit or changing the
    # output directory cannot replay a spent turn in the same mission/view.
    identity = {k: permit[k] for k in ("session_id", "candidate_id", "view_id", "turn_index")}
    directory = Path(permit["master_authorization_path"]).parent / "candidate_centering_claims"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (payload_sha256(identity) + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump({**identity, "permit_sha256": payload_sha256(permit)}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def execute_candidate_centering_permit(permit_path, *, motion=None):
    """Child entry, with an offline motion seam. No operator prompt is added."""
    permit = load_candidate_centering_permit(Path(permit_path))
    result_path, trace_path = Path(permit["result_path"]), Path(permit["controller_trace_path"])
    failure_path = result_path.parent / "candidate_centering_failure.json"
    if any(p.exists() or p.is_symlink() for p in (result_path, trace_path, failure_path)):
        raise RuntimeError("refusing to reuse candidate centering child outputs")
    if motion is None:
        from scripts.aufgabe04.navigation.waypoint_follower.runtime import run_candidate_centering_motion
        motion = run_candidate_centering_motion
    # Claim before ROS initialization, publisher creation or any motion. A
    # failed attempt consumes its slot and must not be silently retried.
    _claim_turn(permit)
    entered_motion = False
    try:
        age = time.time() - permit["advisory"]["created_at_sec"]
        if not 0. <= age <= MAX_ADVISORY_AGE_SEC:
            raise ValueError("candidate centering advisory expired before child execution")
        entered_motion = True
        measured = motion(permit)
        result = {**measured, **{k: permit[k] for k in (
            "schema_version", "purpose", "run_id", "session_id", "candidate_id",
            "view_id", "turn_index", "signed_turn_rad")}, "permit_sha256": payload_sha256(permit)}
        write_content_hashed_json(result_path, result, hash_field=RESULT_HASH)
        load_candidate_centering_result(result_path, permit_path=Path(permit_path))
        return 0 if result["status"] == "completed" else 1
    except BaseException as exc:
        write_content_hashed_json(failure_path, dict(schema_version=1, purpose="candidate_centering",
            status="failed_closed", reason=f"{type(exc).__name__}: {exc}",
            permit_sha256=payload_sha256(permit), motion_continues_authorized=False,
            motion_published="unknown_after_exception" if entered_motion else False),
            hash_field="candidate_centering_failure_sha256")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permit", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return execute_candidate_centering_permit(args.permit)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: candidate centering failed closed: {exc}", file=sys.stderr)
        return 1
