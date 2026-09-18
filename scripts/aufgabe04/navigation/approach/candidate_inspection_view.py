"""Motion-neutral, snapshot-bound requests for additional inspection views.

These requests select a robot viewing direction, never a measured stand axis.
Their separate bearing mode prevents advisory perception from masquerading as
the certified backside evidence required by the opposite-face contract.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.route_context import file_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
)


INSPECTION_VIEW_BEARING_MODE = "candidate-inspection-view"
HASH_FIELD = "candidate_inspection_view_sha256"


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"inspection view {name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"inspection view {name} must be finite")
    return float(value)


def load_candidate_inspection_view(path: Path) -> dict[str, object]:
    unhashed = load_content_hashed_json(path, hash_field=HASH_FIELD)
    payload = {**unhashed, HASH_FIELD: payload_sha256(unhashed)}
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1 or payload.get("view_kind") != (
        "candidate_local_inspection_view"
    ):
        raise ValueError("unsupported inspection view schema")
    if payload.get("purpose") not in {
        "diverse_inspection", "arrival_alignment", "camera_distance_recovery", "lidar_axis_hint",
    }:
        raise ValueError("unsupported inspection view purpose")
    if payload.get("motion_authorized") is not False or payload.get(
        "stand_axis_authorized"
    ) is not False:
        raise ValueError("inspection view must not authorize motion or stand axis")
    normal = _number(payload.get("view_normal_rad"), "normal")
    if not -math.pi <= normal <= math.pi:
        raise ValueError("inspection view normal must be normalized")
    if type(payload.get("view_index")) is not int or not 0 <= payload["view_index"] < 16:
        raise ValueError("inspection view index must be in [0, 15]")
    for field, coordinates in (("stand_center", ("x_m", "y_m")),
                               ("start_pose", ("x_m", "y_m", "yaw_rad"))):
        value = payload.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"inspection view {field} is missing")
        for coordinate in coordinates:
            _number(value.get(coordinate), f"{field}.{coordinate}")
    for field in ("source_observation", "source_view"):
        source = payload.get(field)
        if source is not None:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise ValueError(f"inspection view {field} is malformed")
            if file_sha256(Path(source["path"])) != source.get("sha256"):
                raise ValueError(f"inspection view {field} content changed")
    return payload


def validate_candidate_inspection_view_binding(
    view: Mapping[str, object],
    *,
    snapshot: CandidateSnapshot,
    candidate_uid: str,
    start: Pose2D | None = None,
    view_normal_rad: float | None = None,
) -> None:
    candidate = snapshot.candidate_for(candidate_uid)
    if candidate is None or view.get("candidate_uid") != candidate_uid:
        raise ValueError("inspection view candidate binding mismatch")
    if view.get("candidate_snapshot_sha256") != candidate_snapshot_sha256(snapshot):
        raise ValueError("inspection view snapshot binding mismatch")
    if view.get("planning_frame") != snapshot.planning_frame or view.get(
        "map_bundle_sha256"
    ) != snapshot.map_bundle_sha256:
        raise ValueError("inspection view frame/map binding mismatch")
    center = view["stand_center"]
    if math.hypot(center["x_m"] - candidate.geometry.x_m,
                  center["y_m"] - candidate.geometry.y_m) > 1.0e-9:
        raise ValueError("inspection view stand center binding mismatch")
    if start is not None:
        pose = view["start_pose"]
        if any(abs(pose[key] - getattr(start, key)) > 1.0e-9
               for key in ("x_m", "y_m", "yaw_rad")):
            raise ValueError("inspection view start pose binding mismatch")
    if view_normal_rad is not None and abs(math.remainder(
        float(view["view_normal_rad"]) - view_normal_rad, 2.0 * math.pi
    )) > 1.0e-9:
        raise ValueError("inspection view direction binding mismatch")


def write_candidate_inspection_view(
    path: Path,
    *,
    snapshot: CandidateSnapshot,
    candidate_uid: str,
    start: Pose2D,
    view_normal_rad: float,
    purpose: str,
    view_index: int,
    source_observation_path: Path | None = None,
    source_view_path: Path | None = None,
) -> dict[str, object]:
    candidate = snapshot.candidate_for(candidate_uid)
    if candidate is None:
        raise ValueError("inspection view candidate absent from snapshot")
    normal = _number(view_normal_rad, "normal")
    def source(value: Path | None) -> dict[str, str] | None:
        return None if value is None else {
            "path": str(Path(value).resolve()), "sha256": file_sha256(value)
        }
    payload = {
        "schema_version": 1,
        "view_kind": "candidate_local_inspection_view",
        "candidate_uid": candidate_uid,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
        "planning_frame": snapshot.planning_frame,
        "map_bundle_sha256": snapshot.map_bundle_sha256,
        "stand_center": {"x_m": candidate.geometry.x_m, "y_m": candidate.geometry.y_m},
        "start_pose": {key: getattr(start, key) for key in ("x_m", "y_m", "yaw_rad")},
        "view_normal_rad": math.remainder(normal, 2.0 * math.pi),
        "purpose": purpose,
        "view_index": view_index,
        "source_observation": source(source_observation_path),
        "source_view": source(source_view_path),
        "motion_authorized": False,
        "stand_axis_authorized": False,
    }
    write_content_hashed_json(path, payload, hash_field=HASH_FIELD)
    return load_candidate_inspection_view(path)
