"""Bounded tour continuation identities, separate from exploration recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256

MAX_TOUR_DETOUR_REPLANS = 2
MAX_TOUR_EXECUTIONS_PER_VISIT = 6
_NAVIGATION_FIELDS = frozenset({
    "execution_index", "stage_index", "replan_count",
    "previous_terminal_json", "previous_terminal_sha256",
})


def tour_mission_leg_index(visit_index: int, execution_index: int) -> int:
    if type(visit_index) is not int or visit_index < 0:
        raise ValueError("tour visit_index must be a nonnegative integer")
    if type(execution_index) is not int or not 0 <= execution_index < MAX_TOUR_EXECUTIONS_PER_VISIT:
        raise ValueError("tour execution_index exceeds its six-execution budget")
    return visit_index * MAX_TOUR_EXECUTIONS_PER_VISIT + execution_index


def validate_tour_navigation(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _NAVIGATION_FIELDS:
        raise ValueError("tour_navigation fields mismatch")
    for name, maximum in (("stage_index", 3), ("replan_count", MAX_TOUR_DETOUR_REPLANS), ("execution_index", 5)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            raise ValueError(f"tour_navigation {name} exceeds its bounded budget")
    if value["execution_index"] != value["stage_index"] + value["replan_count"]:
        raise ValueError("tour execution_index must equal stage_index + replan_count")
    for name in ("previous_terminal_json", "previous_terminal_sha256"):
        if not isinstance(value[name], str):
            raise ValueError(f"tour_navigation {name} must be a string")
    if bool(value["previous_terminal_json"]) != (value["execution_index"] > 0):
        raise ValueError("tour predecessor must exist exactly after execution zero")
    if bool(value["previous_terminal_sha256"]) != bool(value["previous_terminal_json"]):
        raise ValueError("tour predecessor path and hash must be paired")
    return dict(value)


def tour_execution_slot_path(master_path: Path, master_sha256: str, visit: int, execution: int) -> Path:
    tour_mission_leg_index(visit, execution)
    return Path(master_path).parent / f"stored_pose_tour_execution_consumption_{master_sha256}_{visit}_{execution}.json"


def stored_tour_source_identity(evidence: Mapping[str, object]) -> str:
    return payload_sha256({key: evidence.get(key) for key in (
        "tour_id", "visit_index", "candidate_uid", "qr_id", "catalog_sha256",
        "stored_pose", "source_planning_frame", "source_artifacts",
    )})


def validate_tour_navigation_binding(permit, metadata: Mapping, evidence: Mapping, *, scope_text: str) -> tuple[int, int]:
    """Validate immutable predecessor evidence; never claim or release a permit."""
    from .mission_leg_motion_permit import TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, file_sha256
    dynamic = scope_text == TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE
    if not dynamic:
        if "tour_navigation" in metadata or "tour_navigation" in evidence or "temporary_obstacle_overlay_json" in metadata:
            raise ValueError("legacy tour scope does not authorize obstacle detours")
        return divmod(permit.mission_leg_index, 4)
    nav = validate_tour_navigation(metadata.get("tour_navigation"))
    if evidence.get("tour_navigation") != nav:
        raise ValueError("tour_navigation differs from sealed target evidence")
    visit, execution = divmod(permit.mission_leg_index, MAX_TOUR_EXECUTIONS_PER_VISIT)
    if nav["execution_index"] != execution:
        raise ValueError("tour execution identity differs from permit index")
    overlay_path = Path(metadata.get("temporary_obstacle_overlay_json", ""))
    if file_sha256(overlay_path) != metadata.get("temporary_obstacle_overlay_sha256"):
        raise ValueError("tour temporary obstacle overlay hash mismatch")
    overlay = load_content_hashed_json(overlay_path, hash_field="temporary_obstacle_overlay_sha256")
    if (
        overlay.get("artifact_kind") != "stored_pose_tour_temporary_obstacle_overlay"
        or overlay.get("schema_version") != 1
        or overlay.get("tour_id") != permit.session_id
        or overlay.get("planning_frame_admission") != evidence.get("planning_frame_admission")
        or not overlay.get("capture_sources")
    ):
        raise ValueError("tour temporary obstacle overlay identity mismatch")
    if execution:
        _validate_predecessor(permit, nav, evidence, visit, overlay)
    return visit, nav["stage_index"]


def _validate_predecessor(permit, nav: Mapping, evidence: Mapping, visit: int, overlay: Mapping) -> None:
    from .mission_leg_motion_consumption import (
        MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
        load_mission_leg_motion_consumption_receipt,
    )
    from .mission_leg_motion_permit import file_sha256, load_mission_leg_motion_permit
    from .tour_terminal_evidence import load_tour_terminal_evidence
    path = Path(nav["previous_terminal_json"])
    if file_sha256(path) != nav["previous_terminal_sha256"]:
        raise ValueError("tour predecessor terminal hash mismatch")
    terminal = load_tour_terminal_evidence(path)
    previous = load_mission_leg_motion_permit(Path(terminal["permit_json"]))
    if (
        previous.master_authorization_sha256 != permit.master_authorization_sha256
        or previous.session_id != permit.session_id or previous.target_id != permit.target_id
        or previous.mission_leg_kind != permit.mission_leg_kind
        or previous.mission_leg_index != permit.mission_leg_index - 1
        or terminal["visit_index"] != visit
    ):
        raise ValueError("tour predecessor must be the immediately preceding same-target execution")
    slot = tour_execution_slot_path(Path(permit.master_authorization_path), permit.master_authorization_sha256, visit, nav["execution_index"] - 1)
    slot_receipt = load_content_hashed_json(slot, hash_field=MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD)
    receipt = load_mission_leg_motion_consumption_receipt(Path(terminal["receipt_json"]))
    if slot_receipt != receipt.to_payload():
        raise ValueError("tour predecessor execution slot differs from its consumed permit")
    prior_metadata = json.loads(Path(previous.diagnostics_path).read_text())["metadata"]
    prior_evidence_path = Path(prior_metadata["target_evidence_json"])
    if file_sha256(prior_evidence_path) != prior_metadata["target_evidence_sha256"]:
        raise ValueError("tour predecessor target evidence changed")
    prior_evidence = json.loads(prior_evidence_path.read_text())
    if stored_tour_source_identity(prior_evidence) != stored_tour_source_identity(evidence):
        raise ValueError("tour continuation must keep the same stored source target")
    completed = terminal["status"] == "completed"
    if not completed and not any(
        isinstance(source, Mapping)
        and source.get("sha256") == terminal["scan_capture_sha256"]
        and isinstance(source.get("path"), str)
        and Path(source["path"]).resolve() == Path(terminal["scan_capture_json"]).resolve()
        for source in overlay["capture_sources"]
    ):
        raise ValueError("tour detour overlay must include the authenticated post-stop scan capture")
    if completed and terminal["final_stage"]:
        raise ValueError("tour cannot continue after genuine final arrival")
    if (
        nav["stage_index"] != terminal["stage_index"] + int(completed)
        or nav["replan_count"] != terminal["replan_count"] + int(not completed)
    ):
        raise ValueError("tour continuation counters do not match the genuine previous outcome")


def validate_tour_obstacle_monitor_admission(args, metadata: Mapping) -> None:
    """Bind the runtime monitor to explicit new authority, including dry evidence."""
    from .mission_leg_motion_permit import TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, file_sha256, load_mission_leg_motion_authorization
    enabled = getattr(args, "stored_pose_tour_obstacle_monitor", False)
    has_navigation = "tour_navigation" in metadata
    if not enabled and not has_navigation:
        return
    if (
        not enabled or not has_navigation
        or metadata.get("route_kind") != "admitted_candidate_pose"
        or metadata.get("route_purpose") != "stored_pose_tour"
        or getattr(args, "allow_sim_time", False)
    ):
        raise ValueError("stored-pose tour obstacle monitor requires the sealed dynamic tour route")
    validate_tour_navigation(metadata["tour_navigation"])
    if not metadata.get("temporary_obstacle_overlay_json") or not metadata.get("temporary_obstacle_overlay_sha256"):
        raise ValueError("stored-pose tour obstacle monitor requires a frozen obstacle overlay")
    overlay_path = Path(metadata["temporary_obstacle_overlay_json"])
    if file_sha256(overlay_path) != metadata["temporary_obstacle_overlay_sha256"]:
        raise ValueError("stored-pose tour obstacle monitor overlay hash mismatch")
    overlay = load_content_hashed_json(overlay_path, hash_field="temporary_obstacle_overlay_sha256")
    if not overlay.get("capture_sources") or not overlay.get("scan_frame") or (
        getattr(args, "stored_pose_tour_scan_frame", None) != overlay["scan_frame"]
    ):
        raise ValueError("stored-pose tour monitor scan frame differs from the frozen scan capture")
    master_path = getattr(args, "mission_leg_motion_authorization_json", None)
    if master_path:
        master = load_mission_leg_motion_authorization(Path(master_path))
        if master.scope_text != TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE:
            raise ValueError("legacy tour scope cannot enable obstacle monitoring or detours")
    elif not getattr(args, "dry_run", False):
        raise ValueError("stored-pose tour obstacle monitor requires explicit tour authorization")


# Public orchestration entry points are kept here while terminal evidence has
# its own module so the permit implementation stays small.
from .tour_terminal_evidence import is_replannable_tour_stop, write_tour_terminal_evidence
