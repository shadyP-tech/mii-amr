"""Exact metric Start connectors cannot relax obstacles or their sealed binding."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from scripts.aufgabe04.navigation.approach.admitted_pose_route import (
    STORED_POSE_TOUR_ROUTE_PURPOSE, plan_admitted_pose_route,
    validate_admitted_pose_route_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.route_smoothing import segment_is_collision_free
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, write_candidate_snapshot
from scripts.aufgabe04.stations.models import Station, StationPose
from tests.aufgabe04 import test_admitted_pose_route as route_fixtures
from tests.aufgabe04 import test_candidate_preapproach_planning as candidate_fixtures
from tests.aufgabe04.test_admitted_return_uncertainty import _context
from tests.aufgabe04.test_detected_station_exploration import write_free_map
from tests.aufgabe04.test_stored_pose_tour_authorization import catalog_evidence


def _bind_sources(args):
    _, bundle = load_occupancy_grid_with_bundle(args["map_yaml"],
        semantic_map_id=args["semantic_map_id"], planning_frame="map")
    args["plan"] = replace(args["plan"], map_bundle_sha256=bundle.bundle_sha256)
    args["snapshot"] = replace(args["snapshot"], map_bundle_sha256=bundle.bundle_sha256)
    args["snapshot_path"].unlink(missing_ok=True)
    write_candidate_snapshot(args["snapshot_path"], args["snapshot"])
    args["target_evidence"]["candidate_snapshot_sha256"] = candidate_snapshot_sha256(args["snapshot"])


@pytest.fixture
def boundary_goal(tmp_path):
    # Exact distance .3417 > .34, but the seven-cell keepout on a 5 cm map
    # covers the endpoint's grid cell. This is the recorded failure mechanism.
    args = route_fixtures.AdmittedPoseRouteTest()._fixture(
        tmp_path, target=Pose2D(.6583, 0., -.71))
    args["map_yaml"] = write_free_map(tmp_path, width=60, height=60, resolution=.05)
    candidate = args["snapshot"].candidates[0]
    candidate = replace(candidate, geometry=replace(candidate.geometry, keepout_radius_m=.34))
    args["snapshot"] = replace(args["snapshot"], candidates=(candidate,))
    args["physical_clearance"] = {
        **args["physical_clearance"], "minimum_candidate_transit_radius_m": .34,
        "minimum_active_standoff_m": .33, "minimum_collision_standoff_m": .28,
    }
    _bind_sources(args)
    grid, _ = load_occupancy_grid_with_bundle(args["map_yaml"],
        semantic_map_id="arena", planning_frame="map")
    static = Costmap.from_occupancy_grid(grid).with_arena_bounds(args["plan"].arena_bounds).with_inflation(.25)
    planning = static.with_station_keepouts((Station("start_stand", StationPose(1., 0., 0.), 0., .34),))
    assert segment_is_collision_free(static, args["target"], args["target"])
    assert not segment_is_collision_free(planning, args["target"], args["target"])
    return args


def _sealed(args):
    result = plan_admitted_pose_route(**args)
    leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
    payload = json.loads(Path(result["diagnostics_json"]).read_text())
    return result, leg, payload


def _binding(result, leg, payload=None):
    return validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), leg,
        candidate_snapshot_path=Path(result["candidate_snapshot"]), diagnostics_payload=payload)


def test_raster_only_boundary_goal_keeps_exact_pose_and_sealed_terminal_connector(boundary_goal):
    result, leg, payload = _sealed(boundary_goal)
    assert _binding(result, leg).ok
    proof = payload["metadata"]["exact_goal_connector"]
    assert proof["policy"] == "candidate_raster_exact_stored_goal_connector"
    assert proof["target"] == asdict(boundary_goal["target"])
    assert proof["candidate_snapshot_sha256"] == candidate_snapshot_sha256(boundary_goal["snapshot"])
    assert proof["map_bundle_sha256"] == boundary_goal["snapshot"].map_bundle_sha256
    assert proof["candidate_keepouts_continuously_validated"]
    assert proof["static_inflated_connector_grid_validated"]
    assert proof["target_blocked_only_by_candidate_raster"]
    assert proof["motion_authorized"] is False
    assert leg.raw_waypoints[-1].pose == boundary_goal["target"]
    assert leg.raw_waypoints[-1].protected and leg.raw_waypoints[-2].protected
    assert (leg.raw_waypoints[-2].pose.x_m, leg.raw_waypoints[-2].pose.y_m) == (
        proof["anchor"]["x_m"], proof["anchor"]["y_m"])
    assert 0 < proof["connector_length_m"] <= proof["maximum_connector_length_m"] <= .15
    thinned = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=10.)
    kept_anchor = thinned.executable_waypoints[-2]
    assert kept_anchor.protected and kept_anchor.point_index == leg.raw_waypoints[-2].point_index
    assert (kept_anchor.pose.x_m, kept_anchor.pose.y_m) == (proof["anchor"]["x_m"], proof["anchor"]["y_m"])


def test_child_requires_protected_terminal_anchor(boundary_goal):
    result, leg, _ = _sealed(boundary_goal)
    unprotected = replace(leg.raw_waypoints[-2], protected=False)
    changed = replace(leg, raw_waypoints=(*leg.raw_waypoints[:-2], unprotected, leg.raw_waypoints[-1]))
    status = _binding(result, changed)
    assert not status.ok
    assert "anchor must remain protected" in status.failures[0]


@pytest.mark.parametrize("mutation", (
    lambda m: m.pop("exact_goal_connector"),
    lambda m: m["exact_goal_connector"]["anchor"].update(x_m=0.),
    lambda m: m["exact_goal_connector"]["target"].update(yaw_rad=0.),
    lambda m: m["exact_goal_connector"].update(candidate_keepouts_continuously_validated=False),
    lambda m: m["exact_goal_connector"].update(candidate_snapshot_sha256="f" * 64),
    lambda m: m["exact_goal_connector"]["static_clearance"].update(minimum_margin_m=9.),
))
def test_missing_or_tampered_goal_proof_is_rejected_without_trusting_claims(boundary_goal, mutation):
    result, leg, original = _sealed(boundary_goal)
    changed = deepcopy(original)
    mutation(changed["metadata"])
    assert not _binding(result, leg, changed).ok


def test_a_real_static_obstacle_cannot_use_the_candidate_raster_exception(boundary_goal):
    image = boundary_goal["map_yaml"].with_name("map.pgm")
    tokens = image.read_text().split()
    width, height = int(tokens[1]), int(tokens[2])
    cells = tokens[4:]
    # PGM rows are stored top to bottom; target is map cell (33,20).
    cells[(height - 1 - 20) * width + 33] = "0"
    image.write_text(f"P2\n{width} {height}\n255\n" + " ".join(cells) + "\n")
    _bind_sources(boundary_goal)
    with pytest.raises(ValueError):
        plan_admitted_pose_route(**boundary_goal)
    assert not boundary_goal["output_dir"].exists()


@pytest.mark.parametrize("blocker", ("original_candidate", "neighbor", "measured_center"))
def test_full_continuous_candidate_and_measured_clearances_remain_required(boundary_goal, blocker):
    if blocker == "original_candidate":
        candidate = boundary_goal["snapshot"].candidates[0]
        candidate = replace(candidate, geometry=replace(candidate.geometry, keepout_radius_m=.342))
        boundary_goal["snapshot"] = replace(boundary_goal["snapshot"], candidates=(candidate,))
    elif blocker == "neighbor":
        target = boundary_goal["target"]
        neighbor = candidate_fixtures.CandidatePreapproachPlanningTest._candidate("neighbor", target.x_m, target.y_m)
        boundary_goal["snapshot"] = replace(boundary_goal["snapshot"],
            candidates=tuple(sorted((*boundary_goal["snapshot"].candidates, neighbor),
                                    key=lambda c: c.candidate_uid)))
    else:
        boundary_goal["target_evidence"]["measured_target_center"] = {
            "x_m": .8, "y_m": 0., "uncertainty_m": .02,
        }
    _bind_sources(boundary_goal)
    with pytest.raises(ValueError):
        plan_admitted_pose_route(**boundary_goal)
    assert not boundary_goal["output_dir"].exists()


@pytest.mark.parametrize("update_both_receipts", (False, True))
def test_stage_proof_binds_full_exact_goal_not_intermediate_stop(boundary_goal, update_both_receipts):
    boundary_goal["route_uncertainty_context"] = _context(boundary_goal["start"], heading_sigma_rad=.4)
    result, leg, payload = _sealed(boundary_goal)
    assert not result["is_final_stage"]
    assert _binding(result, leg).ok
    metadata = payload["metadata"]
    full_path = Path(result["full_return_route_json"])
    full = json.loads(full_path.read_text())
    assert full["exact_goal_connector"] == metadata["exact_goal_connector"]
    assert full["poses"][-1] == asdict(boundary_goal["target"])
    assert metadata["exact_goal_connector"]["target"] != asdict(leg.raw_waypoints[-1].pose)
    # Refreshing the outer file hash must not make a changed final proof valid.
    full["exact_goal_connector"]["anchor"]["x_m"] += .01
    full_path.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n")
    metadata["return_to_start_stage"]["full_return_route_sha256"] = file_sha256(full_path)
    if update_both_receipts:
        metadata["exact_goal_connector"] = deepcopy(full["exact_goal_connector"])
    assert not _binding(result, leg, payload).ok


def test_blocked_raster_stationary_heading_does_not_gain_a_travel_connector(boundary_goal):
    start = replace(boundary_goal["target"], yaw_rad=.3)
    boundary_goal["start"] = start
    frame = CandidatePlanningFrame.from_evidence(boundary_goal["target_evidence"]["planning_frame_admission"])
    boundary_goal["target_evidence"]["planning_frame_admission"] = replace(frame, current_pose=start).to_evidence()
    with pytest.raises(ValueError):
        plan_admitted_pose_route(**boundary_goal)
    assert not boundary_goal["output_dir"].exists()


def test_stored_pose_tour_retains_original_blocked_goal_policy(boundary_goal):
    root = boundary_goal["map_yaml"].parent
    boundary_goal["target_evidence"].update({
        **catalog_evidence(root, uid=boundary_goal["candidate_uid"], pose=asdict(boundary_goal["target"])),
        "tour_id": "existing-tour", "visit_index": 2,
    })
    with pytest.raises(ValueError):
        plan_admitted_pose_route(**boundary_goal, purpose=STORED_POSE_TOUR_ROUTE_PURPOSE)
    assert not boundary_goal["output_dir"].exists()
