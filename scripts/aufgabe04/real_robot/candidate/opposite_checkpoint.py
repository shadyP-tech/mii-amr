"""Bounded checkpoint recovery after verified no-motion opposite vetoes.

Only one prefix is executed. Any failure after dispatch is terminal here, so
neither a no-motion retry nor a stale local-view fallback can follow it.
"""
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.navigation.approach.opposite_checkpoint_route import materialize_opposite_checkpoint
from scripts.aufgabe04.navigation.approach.opposite_checkpoint_selection import select_opposite_checkpoint
from scripts.aufgabe04.navigation.execution.route_context import file_sha256
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest


class OppositeCheckpointExecutionError(RuntimeError):
    """A checkpoint dispatch must never escape as a no-motion route failure."""


def try_opposite_checkpoint(*, rejected_routes, config, effects, planning_frame,
                            candidate_root, execute_prefix, continue_from_checkpoint, event_sink):
    """Return None if ineligible; otherwise complete one prefix then a fresh suffix.

    Entries are collected only after evaluate_opposite_face_route_fallback has
    verified exact initial-child, no-motion, no-permit uncertainty rejection.
    """
    if (not rejected_routes or planning_frame is None or config.robot_radius_m is None
            or effects.load_route_uncertainty_readiness is None):
        return None
    uncertainty = effects.load_route_uncertainty_readiness(CandidateRouteUncertaintyReadinessRequest(
        preflight_json=candidate_root/"opposite_face_planning_localization.json",
        expected_start=planning_frame.current_pose, planning_frame=config.planning_frame,
        odom_frame=planning_frame.odom_frame, robot_radius_m=config.robot_radius_m,
        sigma_multiplier=config.uncertainty_sigma_multiplier,
    ))
    costmap = Costmap.from_occupancy_grid(load_occupancy_grid(config.map_yaml)).with_arena_bounds(config.plan.arena_bounds)
    options = []
    for request, sealed, run_id in rejected_routes:
        leg = load_route_leg(Path(sealed["route_csv"]), 0, thinning_min_spacing_m=0.)
        choice = select_opposite_checkpoint(
            poses=tuple(w.pose for w in leg.raw_waypoints), start_pose=planning_frame.current_pose,
            costmap=costmap, uncertainty=uncertainty, source_sha256=file_sha256(Path(sealed["route_csv"])),
        )
        if choice is not None:
            options.append((choice, request, sealed, run_id))
    if not options:
        event_sink({"event":"opposite_checkpoint_unavailable", "motion_authorized":False,
                    "reason":"no_two_part_uncertainty_admitted_existing_waypoint"})
        return None
    choice, request, sealed, run_id = max(options, key=lambda item:item[0].minimum_margin_m)
    root = candidate_root/"opposite_checkpoint"
    sealed_prefix = materialize_opposite_checkpoint(sealed_parent=sealed,
        snapshot_path=request.output_dir/"candidate_snapshot.json", choice=choice, output_dir=root)
    prefix_request = replace(request, output_dir=root)
    event_sink({"event":"opposite_checkpoint_selected", "source_run_id":run_id,
        "checkpoint_run_id":run_id+"_checkpoint", "vertex_index":choice.vertex_index,
        "minimum_forecast_margin_m":choice.minimum_margin_m,
        "maximum_checkpoint_count":1, "fresh_suffix_localization_required":True,
        "motion_authorized":False})
    try:
        outcome = execute_prefix(prefix_request, sealed_prefix, run_id+"_checkpoint")
        if getattr(outcome, "status", None) != "completed":
            raise RuntimeError("checkpoint prefix did not report completed arrival")
        event_sink({"event":"opposite_checkpoint_arrived", "camera_arrival":False,
            "fresh_suffix_localization_required":True, "motion_authorized":False})
        # This callback MUST admit fresh stopped localization and reproject the
        # original observer receipt before planning. No forecast anchor is used.
        result = continue_from_checkpoint()
        if result is None:
            raise RuntimeError("checkpoint continuation returned no arrival frame")
        return result
    except Exception as exc:
        event_sink({"event":"opposite_checkpoint_failed", "reason":str(exc),
                    "motion_authorized":False, "further_checkpoint_retries_allowed":False})
        raise OppositeCheckpointExecutionError(f"opposite checkpoint stopped: {exc}") from exc
