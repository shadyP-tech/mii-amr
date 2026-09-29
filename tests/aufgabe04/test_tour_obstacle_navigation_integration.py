"""Offline complete detour binding, with only sensor and child-process effects."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.admitted_pose_route import validate_admitted_pose_route_binding
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import CandidateFrameProvenance, CandidatePoint2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import stand_survey_registry_sha256
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind, TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    file_sha256, mission_leg_motion_permit_sha256, write_mission_leg_motion_authorization,
    write_mission_leg_motion_permit,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import (
    consume_mission_leg_motion_permit, mission_leg_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.execution.tour_terminal_evidence import load_tour_terminal_evidence
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.temporary_obstacle_overlay import TemporaryObstacleMap
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.execution.child_runner import parse_motion_leg_outcome
from scripts.aufgabe04.real_robot.mission.stored_start_pose import StoredAdmittedPose
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import StoredPoseNavigationEffects
from scripts.aufgabe04.real_robot.mission.tour_obstacle_navigation import execute_tour_obstacle_navigation
from scripts.aufgabe04.stations.candidate_snapshot import new_candidate_snapshot, write_candidate_snapshot
from tests.aufgabe04 import test_admitted_pose_route as route_fixtures
from tests.aufgabe04 import test_candidate_frame_projection as projection_fixtures
from tests.aufgabe04 import test_mission_leg_motion_permit as permit_fixtures
from tests.aufgabe04.test_admitted_return_uncertainty import _context
from tests.aufgabe04.test_stored_pose_tour_authorization import catalog_evidence
from tests.aufgabe04.test_temporary_obstacle_overlay import capture_payload


class TourObstacleNavigationIntegrationTest(unittest.TestCase):
    def test_real_stopped_detour_preserves_stored_goal_through_final_authorization(self):
        fixture = permit_fixtures.MissionLegMotionPermitTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        root = fixture.root
        args = route_fixtures.AdmittedPoseRouteTest()._fixture(root, fine_grid=True)
        transform = PlanarTransform2D(0., 0., 0.)
        frame = CandidatePlanningFrame(args["start"], transform)
        provenance = CandidateFrameProvenance.from_frozen_map_observation(
            map_frame="map", odom_frame="odom", frozen_map_point=CandidatePoint2D(1., 0.),
            frozen_map_from_odom=transform, source_evidence_id="d" * 64)
        registry = replace(projection_fixtures._registry(1., 0., provenance), map_bundle_sha256=args["plan"].map_bundle_sha256)
        candidate = projection_fixtures._frozen_candidate(1., 0., source_registry_sha256=stand_survey_registry_sha256(registry))
        snapshot = new_candidate_snapshot(snapshot_id="original-admitted-pool", created_unix_sec=3., planning_frame="map",
            map_bundle_sha256=args["plan"].map_bundle_sha256, candidates=(candidate,))
        snapshot_path = root / "bound-source-pool.json"
        write_candidate_snapshot(snapshot_path, snapshot)
        evidence = {**catalog_evidence(root, uid=candidate.candidate_uid, pose=asdict(args["target"])),
                    "source_planning_frame": frame.to_evidence()}
        stored = StoredAdmittedPose(candidate.candidate_uid, args["target"], frame, registry, evidence)
        master = replace(fixture.authorization, session_id="tour", semantic_map_id="arena",
            allowed_leg_kinds=(MissionLegKind.STORED_POSE_TOUR,), scope_text=TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE)
        master_path = root / "tour-master.json"
        master_sha = write_mission_leg_motion_authorization(master_path, master)
        config = SimpleNamespace(
            snapshot=snapshot, snapshot_path=snapshot_path, planning_frame="map", map_yaml=args["map_yaml"],
            plan=args["plan"], semantic_map_id="arena", inflation_radius_m=args["inflation_radius_m"],
            physical_clearance=args["physical_clearance"], robot_radius_m=.105, uncertainty_sigma_multiplier=2.,
            localization_branch_proof_id=master.localization_branch_proof_id, mission_leg_motion_authorization_json=master_path)
        visit_root = root / "visit"
        clock = [100.23]
        captures = []
        requests = []
        routes = []
        frames = iter((frame, frame, CandidatePlanningFrame(args["target"], transform)))

        def admit_frame(path):
            selected = next(frames)
            path.write_text(json.dumps(selected.to_evidence()))
            return selected

        def capture(path):
            start = 100. + 2 * len(captures)
            clock[0] = start + .23
            base = (args["start"].x_m, args["start"].y_m, args["start"].yaw_rad)
            # First route is clear. A new occupied cell near its center is
            # observed only after the authentic first child stops.
            origin = (base[0], base[1], math.atan2(.025 - base[1], .025 - base[0]))
            distance = math.hypot(.025 - base[0], .025 - base[1]) if captures else 5.
            payload = capture_payload(start=start, ranges=(distance, None), base=base, origin=origin)
            write_content_hashed_json(path, payload, hash_field="tour_scan_capture_sha256")
            captures.append(path)
            return path

        def execute_child(request):
            self.assertFalse((visit_root / "arrival.json").exists())
            leg = load_route_leg(Path(request.sealed["route_csv"]), 0, thinning_min_spacing_m=0.)
            self.assertTrue(validate_admitted_pose_route_binding(Path(request.sealed["diagnostics_json"]), leg,
                candidate_snapshot_path=request.candidate_snapshot_path).ok)
            routes.append(leg)
            permit = replace(fixture.permit, master_authorization_path=str(master_path), master_authorization_sha256=master_sha,
                session_id="tour", semantic_map_id="arena", run_id=request.run_id, mission_leg_kind=request.mission_leg_kind,
                mission_leg_index=request.mission_leg_index, target_id=request.target_id,
                route_csv_path=request.sealed["route_csv"], route_csv_sha256=file_sha256(Path(request.sealed["route_csv"])),
                diagnostics_path=request.sealed["diagnostics_json"], diagnostics_sha256=file_sha256(Path(request.sealed["diagnostics_json"])),
                map_route_certificate_path=request.sealed["route_certificate_json"],
                map_route_certificate_sha256=file_sha256(Path(request.sealed["route_certificate_json"])))
            write_mission_leg_motion_permit(request.permit_json_path, permit)
            receipt = consume_mission_leg_motion_permit(permit_path=request.permit_json_path, permit=permit,
                session_id="tour", run_id=permit.run_id, mission_leg_kind=permit.mission_leg_kind,
                mission_leg_index=permit.mission_leg_index, target_id=permit.target_id)
            stopped = not requests
            status = "stopped" if stopped else "completed"
            reason = "obstacle too close" if stopped else "route completed"
            details = {"source": "global_scan", "valid_sample_count": 15, "nearest_valid_range_m": .18, "threshold_m": .2} if stopped else {}
            identity = {"run_id": permit.run_id, "mission_leg_kind": "stored_pose_tour", "mission_leg_index": permit.mission_leg_index,
                        "target_id": permit.target_id}
            events = [
                {**identity, "event": "mission_leg_motion_permit_consumed", "mission_leg_motion_permit_sha256": mission_leg_motion_permit_sha256(permit),
                 "mission_leg_motion_consumption_receipt_sha256": mission_leg_motion_consumption_receipt_sha256(receipt)},
                {**identity, "event": "motion_started", "motion_published": False, "event_semantics": "child_execution_attempt_started_before_follower"},
                {**identity, "event": "safety_stop" if stopped else "motion_completed", "status": status, "stop_reason": reason,
                 "stop_details": details, "motion_published": True},
                {"run_id": permit.run_id, "event": "run_finished", "final_status": status,
                 "timestamp": datetime.fromtimestamp(clock[0] + .1, timezone.utc).isoformat()},
            ]
            log_path = visit_root / "child-events.jsonl"
            offset = log_path.stat().st_size if log_path.exists() else 0
            with log_path.open("a") as stream:
                stream.write("".join(json.dumps(event) + "\n" for event in events))
            requests.append(request)
            return parse_motion_leg_outcome(log_path, run_id=request.run_id, returncode=1 if stopped else 0, start_offset=offset)

        effects = StoredPoseNavigationEffects(admit_frame, execute_child,
            load_route_uncertainty_readiness=lambda request: _context(request.expected_start, heading_sigma_rad=.001))
        with patch("scripts.aufgabe04.navigation.planning.temporary_obstacle_overlay.time.time", side_effect=lambda: clock[0]):
            result = execute_tour_obstacle_navigation(stored, config, effects, tour_session_id="tour", visit_index=2,
                output_root=visit_root, obstacle_map=TemporaryObstacleMap("tour", "odom", config.plan.map_bundle_sha256), capture_scan=capture)
        self.assertEqual([request.mission_leg_index for request in requests], [12, 13])
        self.assertEqual([len(leg.raw_waypoints) for leg in routes[:1]], [2])
        self.assertGreater(len(routes[1].raw_waypoints), 2)
        self.assertEqual(routes[1].raw_waypoints[-1].pose, args["target"])
        self.assertEqual(result["qr_id"], "Werkbank")
        self.assertTrue(result["target_pose_reached"])
        self.assertEqual(result["replan_count"], 1)
        self.assertFalse(result["legs"][0]["arrival_verified"])
        first = load_tour_terminal_evidence(visit_root / "legs/000/terminal.json")
        final = load_tour_terminal_evidence(visit_root / "legs/001/terminal.json")
        self.assertEqual((first["status"], final["status"]), ("stopped", "completed"))
        self.assertEqual((first["stage_index"], final["stage_index"]), (0, 0))
        self.assertEqual(Path(first["scan_capture_json"]), captures[1])
        self.assertTrue(Path(first["receipt_json"]).exists())


if __name__ == "__main__":
    unittest.main()
