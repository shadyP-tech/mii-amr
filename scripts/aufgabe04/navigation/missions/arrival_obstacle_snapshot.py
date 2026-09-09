"""Keep unresolved LiDAR hypotheses in promoted catalog route validation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import DynamicApproachConfig
from scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint import _known_stand_keepout_costmap
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, load_candidate_snapshot


def load_bound_obstacle_snapshot(path: Path | None, *, catalog, confirmed_snapshot, map_bundle_sha256: str):
    digest = catalog.provenance.obstacle_candidate_snapshot_sha256
    if not digest:
        if path is not None:
            raise ValueError("obstacle candidate snapshot is not bound by catalog provenance")
        return None
    if path is None:
        raise ValueError("catalog requires --obstacle-candidate-snapshot for the complete obstacle pool")
    if confirmed_snapshot is None:
        raise ValueError("obstacle pool requires the bound confirmed candidate snapshot")
    snapshot = load_candidate_snapshot(path, required_map_bundle_sha256=map_bundle_sha256)
    if candidate_snapshot_sha256(snapshot) != digest:
        raise ValueError("obstacle candidate snapshot hash differs from catalog provenance")
    if any(snapshot.candidate_for(candidate.candidate_uid) != candidate for candidate in confirmed_snapshot.candidates):
        raise ValueError("confirmed candidates are not an unchanged subset of the obstacle snapshot")
    if snapshot.planning_frame != confirmed_snapshot.planning_frame:
        raise ValueError("obstacle snapshot frame differs from confirmed candidates")
    return snapshot


def unconfirmed_obstacle_keepouts(snapshot, confirmed_uids, *, config: DynamicApproachConfig):
    if snapshot is None:
        return ()
    return tuple({
        "x_m": candidate.geometry.x_m, "y_m": candidate.geometry.y_m,
        "radius_m": replace(
            config, stand_radius_m=candidate.geometry.radius_m,
            stand_position_uncertainty_m=candidate.geometry.uncertainty_m,
            minimum_non_target_keepout_radius_m=candidate.geometry.keepout_radius_m,
        ).non_target_stand_keepout_radius_m,
    } for candidate in snapshot.candidates if candidate.candidate_uid not in confirmed_uids)


def overlay_unconfirmed_obstacles(costmap, keepouts):
    # No exemption of static/inflated cells; an unreachable start fails closed.
    return _known_stand_keepout_costmap(costmap, tuple((item["x_m"], item["y_m"], item["radius_m"]) for item in keepouts)).costmap
