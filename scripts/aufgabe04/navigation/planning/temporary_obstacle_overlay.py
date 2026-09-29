"""Tour-local odom occupancy; immutable snapshots never update a moving leg."""

from __future__ import annotations

import math
from pathlib import Path
import time

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.planning.temporary_scan_capture import (
    RESOLUTION_M, RAY_CAP_M, TTL_SEC, number, pose, scan_cells, validate_capture,
)


OVERLAY_HASH_FIELD = "temporary_obstacle_overlay_sha256"
# The CLI supports 100 production visits, each potentially expanded into
# pickup, processing, dropoff and charging, plus both Start visits and up to
# ten supplemental stations followed by Start. At six executions per visit,
# that needs at most 2478 captures under that server expansion. Retain a
# bounded 4096-reference margin; exceeding it fails before state mutation.
MAX_TOUR_CAPTURE_SOURCES = 4096


def validate_capture_source_count(count: int) -> None:
    if type(count) is not int or not 0 <= count <= MAX_TOUR_CAPTURE_SOURCES:
        raise ValueError("temporary occupancy bounded capture storage exhausted")


class TemporaryObstacleMap:
    """A bounded occupancy grid private to one running tour's odom origin."""

    def __init__(self, tour_id: str, odom_frame: str, map_bundle_sha256: str):
        if not isinstance(tour_id, str) or not tour_id or not isinstance(odom_frame, str) or not odom_frame:
            raise ValueError("temporary occupancy requires tour and odom identities")
        if not isinstance(map_bundle_sha256, str) or len(map_bundle_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in map_bundle_sha256
        ):
            raise ValueError("temporary occupancy requires a map bundle hash")
        self.tour_id, self.odom_frame, self.map_bundle_sha256 = tour_id, odom_frame, map_bundle_sha256
        self._occupied: dict[tuple[int, int], tuple[float, str]] = {}
        self._sources: list[dict] = []
        self._last_capture = None
        self._last_selection = 0.

    def update_from_capture(self, path: Path, now_sec: float | None = None) -> None:
        validate_capture_source_count(len(self._sources)+1)
        now = time.time() if now_sec is None else number(now_sec, "update time")
        if now < self._last_selection:
            raise ValueError("temporary occupancy clock moved backwards")
        path = Path(path).resolve(strict=True)
        data = validate_capture(path, tour_id=self.tour_id, odom_frame=self.odom_frame, now_sec=now)
        last = self._last_capture
        if last is not None:
            if (data["base_frame"], data["scan_frame"]) != (last["base_frame"], last["scan_frame"]):
                raise ValueError("temporary occupancy sensor frame identity changed")
            delta = data["scans"][0]["stamp_sec"]-last["scans"][-1]["stamp_sec"]
            if delta <= 0. or data["captured_at_unix_sec"] <= last["captured_at_unix_sec"]:
                raise ValueError("temporary occupancy source capture reused or time reset")
            a = pose(last["scans"][-1]["base_pose_odom"], "previous base")
            b = pose(data["scans"][0]["base_pose_odom"], "current base")
            if (math.hypot(a[0]-b[0], a[1]-b[1]) > .30*delta+.03
                or abs(math.remainder(a[2]-b[2], math.tau)) > 1.2*delta+.05):
                raise ValueError("temporary occupancy odom discontinuity exceeds physical motion bounds")
        source_sha = file_sha256(path)
        occupied, clearing_votes = set(), {}
        for scan in data["scans"]:
            hits, frees = scan_cells(scan)
            occupied.update(hits)
            for cell in frees:
                clearing_votes[cell] = clearing_votes.get(cell, 0)+1
        retained = {cell: entry for cell, entry in self._occupied.items()
                    if now-entry[0] < TTL_SEC and (clearing_votes.get(cell, 0) < 2 or cell in occupied)}
        observed = data["scans"][-1]["stamp_sec"]
        retained.update({cell: (observed, source_sha) for cell in occupied})
        if len(retained) > 20000:
            raise ValueError("temporary occupancy bounded storage exhausted")
        self._occupied = retained
        self._sources.append({"path": str(path), "sha256": source_sha, "admitted_at_unix_sec": now})
        self._last_capture = data
        self._last_selection = now

    def active_cells(self, now_sec: float) -> list[dict]:
        now = number(now_sec, "selection time")
        if now < self._last_selection:
            raise ValueError("temporary occupancy selection time moved backwards")
        return [{"x": x, "y": y, "observed_at_unix_sec": timestamp, "source_sha256": source}
                for (x, y), (timestamp, source) in sorted(self._occupied.items()) if now-timestamp < TTL_SEC]

    def write_projection(self, path: Path, frame: CandidatePlanningFrame, now_sec: float | None = None) -> Path:
        now = time.time() if now_sec is None else number(now_sec, "selection time")
        if not isinstance(frame, CandidatePlanningFrame) or frame.odom_frame != self.odom_frame:
            raise ValueError("temporary occupancy planning frame identity changed")
        active = self.active_cells(now)
        payload = {
            "schema_version": 1, "artifact_kind": "stored_pose_tour_temporary_obstacle_overlay",
            "tour_id": self.tour_id, "odom_frame": self.odom_frame, "planning_frame": frame.map_frame,
            "map_bundle_sha256": self.map_bundle_sha256, "planning_frame_admission": frame.to_evidence(),
            "resolution_m": RESOLUTION_M, "ray_cap_m": RAY_CAP_M, "ttl_sec": TTL_SEC,
            "clearance_confirmation_scan_count": 2, "selected_at_unix_sec": now,
            "active_odom_cells": active, "capture_sources": list(self._sources),
            "scan_frame": None if self._last_capture is None else self._last_capture["scan_frame"],
            "base_frame": None if self._last_capture is None else self._last_capture["base_frame"],
            "motion_authorized": False, "semantic_survey_evidence": False,
        }
        result = Path(path).resolve()
        write_content_hashed_json(result, payload, hash_field=OVERLAY_HASH_FIELD)
        self._last_selection = now
        return result


def apply_bound_temporary_obstacles(base_costmap, metadata, *, execution_map_from_odom=None):
    """Shared raw occupancy loader for planner, route binding and live child."""
    from scripts.aufgabe04.navigation.planning.temporary_obstacle_projection import apply_bound_overlay
    return apply_bound_overlay(base_costmap, metadata, execution_map_from_odom=execution_map_from_odom)
