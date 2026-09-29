"""Reconstruct and conservatively rasterize an immutable tour occupancy layer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.foundation.models import GridCell
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.temporary_obstacle_overlay import (
    TemporaryObstacleMap, OVERLAY_HASH_FIELD, validate_capture_source_count,
)
from scripts.aufgabe04.navigation.planning.temporary_scan_capture import RESOLUTION_M, RAY_CAP_M, TTL_SEC, number


def _intersects(a, b):
    for polygon in (a, b):
        for p, q in zip(polygon, (*polygon[1:], polygon[0])):
            normal = (p[1]-q[1], q[0]-p[0])
            ap = [x*normal[0]+y*normal[1] for x, y in a]
            bp = [x*normal[0]+y*normal[1] for x, y in b]
            if max(ap) < min(bp)-1e-12 or max(bp) < min(ap)-1e-12:
                return False
    return True


def load_validated_projection(path: Path) -> dict:
    data = load_content_hashed_json(path, hash_field=OVERLAY_HASH_FIELD)
    for key, expected in (("schema_version", 1), ("artifact_kind", "stored_pose_tour_temporary_obstacle_overlay"),
                          ("resolution_m", RESOLUTION_M), ("ray_cap_m", RAY_CAP_M), ("ttl_sec", TTL_SEC),
                          ("clearance_confirmation_scan_count", 2), ("motion_authorized", False),
                          ("semantic_survey_evidence", False)):
        if type(data.get(key)) is not type(expected) or data[key] != expected:
            raise ValueError(f"temporary occupancy overlay {key} mismatch")
    frame = CandidatePlanningFrame.from_evidence(data.get("planning_frame_admission"))
    if (frame.map_frame, frame.odom_frame) != (data.get("planning_frame"), data.get("odom_frame")):
        raise ValueError("temporary occupancy overlay frame mismatch")
    reconstructed = TemporaryObstacleMap(data.get("tour_id"), data.get("odom_frame"), data.get("map_bundle_sha256"))
    sources = data.get("capture_sources")
    if not isinstance(sources, list):
        raise ValueError("temporary occupancy overlay capture sources are invalid")
    validate_capture_source_count(len(sources))
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256", "admitted_at_unix_sec"}:
            raise ValueError("temporary occupancy overlay capture reference is invalid")
        path = Path(source["path"])
        if not path.is_absolute() or file_sha256(path) != source["sha256"]:
            raise ValueError("temporary occupancy source capture hash mismatch")
        reconstructed.update_from_capture(path, now_sec=source["admitted_at_unix_sec"])
    for key in ("base_frame", "scan_frame"):
        expected = None if reconstructed._last_capture is None else reconstructed._last_capture[key]
        if data.get(key) != expected:
            raise ValueError(f"temporary occupancy overlay {key} differs from source captures")
    selected = number(data.get("selected_at_unix_sec"), "selection time")
    if selected <= 0. or data.get("active_odom_cells") != reconstructed.active_cells(selected):
        raise ValueError("temporary occupancy cells differ from replayed stopped scans")
    return data


def apply_bound_overlay(base, metadata: Mapping, *, execution_map_from_odom=None):
    path_value = metadata.get("temporary_obstacle_overlay_json")
    hash_value = metadata.get("temporary_obstacle_overlay_sha256")
    if path_value is None and hash_value is None:
        if "tour_navigation" in metadata:
            raise ValueError("dynamic tour requires an immutable temporary occupancy overlay")
        return base
    if metadata.get("route_purpose") != "stored_pose_tour" or not isinstance(path_value, str):
        raise ValueError("temporary occupancy overlay is restricted to stored pose tours")
    path = Path(path_value)
    if file_sha256(path) != hash_value:
        raise ValueError("temporary occupancy overlay file hash mismatch")
    data = load_validated_projection(path)
    for key in ("tour_id", "map_bundle_sha256", "planning_frame"):
        if data.get(key) != metadata.get(key):
            raise ValueError(f"temporary occupancy overlay {key} binding mismatch")
    target_path = metadata.get("target_evidence_json")
    if target_path is not None:
        import json
        target_file = Path(target_path)
        if file_sha256(target_file) != metadata.get("target_evidence_sha256"):
            raise ValueError("temporary occupancy target evidence hash mismatch")
        evidence = json.loads(target_file.read_text())
        expected_frame = evidence.get("planning_frame_admission")
    else:
        expected_frame = metadata.get("planning_frame_admission")
    if not isinstance(expected_frame, Mapping) or payload_sha256(expected_frame) != payload_sha256(data["planning_frame_admission"]):
        raise ValueError("temporary occupancy projection differs from route planning frame")
    transform = CandidatePlanningFrame.from_evidence(data["planning_frame_admission"]).map_from_odom
    if execution_map_from_odom is not None:
        # The artifact remains bound to the parent's admitted projection. A
        # child that freezes a newer map<-odom must test these same physical
        # ODOM cells in its own MAP coordinates before converting the route.
        if not isinstance(execution_map_from_odom, PlanarTransform2D):
            raise ValueError("temporary occupancy execution transform must be planar")
        for value in (execution_map_from_odom.x_m, execution_map_from_odom.y_m, execution_map_from_odom.yaw_rad):
            number(value, "execution transform")
        transform = execution_map_from_odom
    cosine, sine = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    blocked = set()
    for record in data["active_odom_cells"]:
        x, y = record["x"]*RESOLUTION_M, record["y"]*RESOLUTION_M
        square = [(transform.x_m+cosine*px-sine*py, transform.y_m+sine*px+cosine*py)
                  for px, py in ((x,y), (x+RESOLUTION_M,y), (x+RESOLUTION_M,y+RESOLUTION_M), (x,y+RESOLUTION_M))]
        low = base.world_to_grid(min(p[0] for p in square)-1e-10, min(p[1] for p in square)-1e-10)
        high = base.world_to_grid(max(p[0] for p in square)+1e-10, max(p[1] for p in square)+1e-10)
        for row in range(max(0, low.y), min(base.height-1, high.y)+1):
            for col in range(max(0, low.x), min(base.width-1, high.x)+1):
                cell = GridCell(col, row)
                if cell in base.blocked_cells:
                    continue
                center = base.grid_to_world(cell)
                r = base.resolution/2
                other = [(center.x_m-r,center.y_m-r), (center.x_m+r,center.y_m-r),
                         (center.x_m+r,center.y_m+r), (center.x_m-r,center.y_m+r)]
                if _intersects(square, other):
                    blocked.add(cell)
    return base.with_blocked_cells(blocked, source="temporary_obstacle")
