"""Strict semantic evidence for rejection before a follower or permit claim.

The odom admission classifier validates the geometric proof. This module
additionally binds it to the completed child attempt and exact mission leg.
Neither a generic preflight failure nor absence of motion events is a proof.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256


def validate_odom_startup_rejection_log(
    path: Path,
    *,
    rejected_run_id: str,
    mission_leg_kind: str,
    mission_leg_index: int,
    target_id: str,
) -> Mapping[str, object]:
    from scripts.aufgabe04.navigation.localization.startup_route_admission import (
        evaluate_odom_startup_route_rejection,
    )

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("odom startup rejection log must be a normal file")
    if (
        not rejected_run_id or not target_id
        or type(mission_leg_index) is not int or mission_leg_index < 0
    ):
        raise ValueError("odom startup rejection identity is invalid")
    try:
        events = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError("invalid odom startup rejection log") from exc
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("invalid odom startup rejection log objects")
    same_run = [event for event in events if event.get("run_id") == rejected_run_id]
    required = ("odom_execution_admission_failed", "safety_stop", "run_finished")
    matches = []
    attempt_start = max(
        (index for index, event in enumerate(same_run) if event.get("event") == "run_started"),
        default=-1,
    )
    for index, event in enumerate(same_run):
        name = event.get("event")
        motion = event.get("motion_published")
        if (
            name in {"motion_started", "motion_completed", "follower_started"}
            or (isinstance(name, str) and name.endswith("motion_permit_consumed"))
            or motion is True
            or ("motion_published" in event and type(motion) is not bool)
        ):
            raise ValueError("odom startup rejection log contains motion, follower, or consumption")
        if index <= attempt_start or name not in required:
            continue
        if (
            event.get("mission_leg_kind") != mission_leg_kind
            or type(event.get("mission_leg_index")) is not int
            or event.get("mission_leg_index") != mission_leg_index
            or event.get("target_id") != target_id
        ):
            raise ValueError("odom startup rejection mission leg identity mismatch")
        decision = evaluate_odom_startup_route_rejection(
            status=event.get("status"),
            motion_published=motion,
            stop_reason=event.get("stop_reason"),
            stop_details=event.get("stop_details"),
        )
        if not decision.eligible:
            raise ValueError(f"odom startup rejection is not eligible: {decision.reason}")
        if name == "run_finished" and event.get("final_status") != "preflight_failed":
            raise ValueError("odom startup rejection final status mismatch")
        matches.append(event)
    if tuple(event["event"] for event in matches) != required:
        raise ValueError("odom startup rejection requires one completed same-run rejection sequence")
    details = matches[0]["stop_details"]
    if any(event["stop_details"] != details for event in matches[1:]):
        raise ValueError("odom startup rejection event evidence mismatch")
    if attempt_start < 0:
        raise ValueError("odom startup rejection requires its child run_started event")
    if any(
        event.get("dry_run") is not details["dry_run"]
        for event in (same_run[attempt_start], *matches)
    ):
        raise ValueError("odom startup rejection dry/execute attempt identity mismatch")
    _validate_source_artifacts(same_run[attempt_start], matches[-1], details)
    return details


def _validate_source_artifacts(start: Mapping, end: Mapping, details: Mapping) -> None:
    """Bind the recomputed geometry to saved PASS observations and certified CSV."""
    from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
        execution_route_certificate_sha256, load_execution_route_certificate,
    )
    from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import file_sha256
    from scripts.aufgabe04.navigation.localization.odom_execution_certificate import pose_route_sha256
    from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg, poses_from_waypoints

    proof = details["startup_route_admission"]
    preflight_path = _source_path(end.get("preflight_json_path"))
    if str(preflight_path) != start.get("preflight_json_path"):
        raise ValueError("odom startup source preflight path mismatch")
    preflight = json.loads(preflight_path.read_text())
    if (
        not isinstance(preflight, Mapping)
        or payload_sha256(preflight) != proof["source_preflight_sha256"]
        or preflight.get("ok") is not True or preflight.get("failures") != []
    ):
        raise ValueError("odom startup source preflight was not the hashed PASS evidence")
    runtime = preflight.get("runtime_config")
    if (
        not isinstance(runtime, Mapping) or runtime.get("localization_source") != "amcl"
        or runtime.get("use_sim_time") is not False
    ):
        raise ValueError("odom startup source preflight is not physical AMCL")
    for source_key, proof_key in (("route_pose", "chained_map_pose"), ("odom_pose", "odom_pose"),
                                  ("map_from_odom", "map_from_odom")):
        if preflight.get(source_key) != proof[proof_key]:
            raise ValueError("odom startup source pose evidence mismatch")
    observations = preflight.get("observations")
    if not isinstance(observations, list) or any(not isinstance(item, dict) for item in observations):
        raise ValueError("odom startup source observations are malformed")
    for name, frame in (("map", proof["map_frame"]), ("odom", proof["odom_frame"])):
        observed = [item for item in observations if item.get("name") == f"tf {frame}->{proof['base_frame']}"]
        if (len(observed) != 1 or observed[0].get("ok") is not True
                or observed[0].get("data") != proof["pose_tf_observations"][name]):
            raise ValueError("odom startup source TF observation mismatch")
    certificate_path = _source_path(end.get("map_route_certificate_json_path"))
    certificate = load_execution_route_certificate(certificate_path)
    route_path = _source_path(start.get("authoritative_route_csv"))
    route_index = start.get("leg_index")
    if type(route_index) is not int or route_index < 0:
        raise ValueError("odom startup source route leg index is invalid")
    route = load_route_leg(route_path, route_index, require_motion=False, thinning_min_spacing_m=0.0)
    if (
        execution_route_certificate_sha256(certificate) != proof["source_map_execution_certificate_sha256"]
        or certificate.route_sha256 != file_sha256(route_path)
        or certificate.planning_frame != proof["map_frame"]
        or certificate.tracking_tube_radius_m != proof["tracking_tube_radius_m"]
        or certificate.waypoint_count != len(route.executable_waypoints)
        or pose_route_sha256(poses_from_waypoints(route.executable_waypoints)) != proof["source_map_route_sha256"]
    ):
        raise ValueError("odom startup source certified route binding mismatch")


def _source_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("odom startup source artifact path is missing")
    path = Path(value)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or Path(os.path.normpath(str(path))) != path):
        raise ValueError("odom startup source artifact must be an absolute canonical normal file")
    return path
