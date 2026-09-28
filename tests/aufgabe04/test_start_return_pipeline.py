"""Recorded geometry through real planning/admission with simulated stopped poses.

Only target loading and the ROS motion/capture effects are substituted. Every
stage uses the production covariance loader, route planner, sealed binding and
pure child admission. No robot, server, or ignored audit directory is used.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.admitted_pose_route import validate_admitted_pose_route_binding
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import evaluate_admitted_return_stage_uncertainty
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import load_stand_survey_registry
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import evaluate_route_uncertainty_admission
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.candidate_planning_pose import admitted_candidate_planning_pose
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.real_robot.mission.start_return import StartReturnEffects, execute_start_return
from scripts.aufgabe04.real_robot.mission.start_return_readiness import load_start_return_readiness
from scripts.aufgabe04.real_robot.mission.stored_start_pose import StoredStartPose
from scripts.aufgabe04.stations.candidate_snapshot import CandidateGeometry
from tests.aufgabe04 import test_autonomous_candidate_approach as candidate_fixture
from tests.aufgabe04.test_candidate_route_uncertainty_readiness import _preflight_payload


FIXTURE = Path(__file__).with_name("fixtures") / "start_return_uncertainty_20260923.json"


def _write_recorded_map(root, recorded):
    width, height = recorded["width"], recorded["height"]
    cells = [value for value, count in recorded["cells_rle"] for _ in range(count)]
    pixels = {0: 255, 1: 0, 2: 205}
    rows = [cells[i * width:(i + 1) * width] for i in range(height)]
    (root / "map.pgm").write_text(f"P2\n{width} {height}\n255\n" + "\n".join(
        " ".join(str(pixels[v]) for v in row) for row in reversed(rows)) + "\n")
    path = root / "map.yaml"
    path.write_text("image: map.pgm\n" + "\n".join(
        f"{key}: {recorded[key]}" for key in
        ("resolution", "origin", "negate", "occupied_thresh", "free_thresh", "mode")) + "\n")
    return path


class StartReturnPipelineTest(unittest.TestCase):
    def _run(self, root, *, worsen_after_stop=False):
        fixture = json.loads(FIXTURE.read_text())
        helper = candidate_fixture.AutonomousCandidateApproachTest()
        candidates = tuple(replace(
            helper._candidate(entry["candidate_uid"], entry["geometry"]["x_m"], entry["geometry"]["y_m"]),
            geometry=CandidateGeometry(**entry["geometry"]),
        ) for entry in fixture["candidate_geometries"])
        config = helper._config(root, candidates)
        path = _write_recorded_map(root, fixture["map"])
        grid, bundle = load_occupancy_grid_with_bundle(path, semantic_map_id="arena", planning_frame="map")
        config = replace(config, map_yaml=path, robot_radius_m=.105,
            plan=replace(config.plan, map_bundle_sha256=bundle.bundle_sha256,
                         arena_bounds=ArenaBounds(**fixture["map"]["arena"])),
            snapshot=replace(config.snapshot, map_bundle_sha256=bundle.bundle_sha256),
            physical_clearance=fixture["planner_parameters"]["physical_clearance"])
        config = helper._write_frame_registry(config, frozen_map_from_odom=PlanarTransform2D(0., 0., 0.))
        registry = load_stand_survey_registry(config.survey_root / "stand_registry.json")
        target = Pose2D(**fixture["full_poses"][-1])
        original_frame = CandidatePlanningFrame(target, PlanarTransform2D(0., 0., 0.))
        source = root / "admitted_start.json"
        source.write_text(json.dumps({"qr_id": "Start", "pose": asdict(target)}))
        stored = StoredStartPose("survey_candidate_0001", target, original_frame, registry, {
            "qr_id": "Start", "candidate_uid": "survey_candidate_0001",
            "pose_kind": "qr_verified_observation_pose", "stored_pose": asdict(target),
            "source_planning_frame": original_frame.to_evidence(),
            "source_artifacts": [{"path": str(source), "sha256": file_sha256(source)}],
        })
        state = {"pose": Pose2D(**fixture["full_poses"][0]), "stages": [], "captures": []}
        costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(config.plan.arena_bounds)

        def capture(path):
            payload = _preflight_payload(state["pose"])
            for sample in payload["stationary_amcl_samples"]:
                cov = sample["covariance"]
                cov[0] = cov[7] = .20 if worsen_after_stop and state["stages"] else fixture["covariance_m2"]["xx_m2"]
                cov[35] = fixture["admission_config"]["heading_sigma_rad"] ** 2
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            pose, provenance = admitted_candidate_planning_pose(payload, map_frame="map", odom_frame="odom")
            frame = CandidatePlanningFrame(pose, PlanarTransform2D(0., 0., 0.), pose_provenance=provenance)
            state.update(frame=frame, preflight=path)
            state["captures"].append(path)
            return frame

        def motion(request):
            sealed = request.sealed
            leg = load_route_leg(Path(sealed["route_csv"]), 0, thinning_min_spacing_m=0.)
            binding = validate_admitted_pose_route_binding(Path(sealed["diagnostics_json"]), leg,
                candidate_snapshot_path=Path(sealed["candidate_snapshot"]))
            self.assertTrue(binding.ok, binding.failures)
            context = load_start_return_readiness(CandidateRouteUncertaintyReadinessRequest(
                state["preflight"], state["frame"].current_pose, "map", "odom", .105, 2.,
            ))
            metadata = json.loads(Path(sealed["diagnostics_json"]).read_text())["metadata"]
            admission = evaluate_admitted_return_stage_uncertainty(
                costmap, tuple(w.pose for w in leg.raw_waypoints), context.covariance, context.admission_config,
                start_pose=state["frame"].current_pose, target_evidence_sha256=metadata["target_evidence_sha256"],
                is_final_stage=metadata["return_to_start_stage"]["final_stage"],
            )
            self.assertTrue(admission.decision.accepted)
            self.assertFalse((config.session_root / "return_to_start/arrival.json").exists())
            state["stages"].append({"request": request, "leg": leg, "admission": admission, "config": context.admission_config})
            state["pose"] = leg.raw_waypoints[-1].pose
            return helper._completed(request)

        effects = StartReturnEffects(capture, motion, load_target=lambda *_: stored)
        if worsen_after_stop:
            with self.assertRaisesRegex(ValueError, "uncertainty|admitted"):
                execute_start_return(None, config, effects)
            failure = json.loads((config.session_root / "return_to_start/failure.json").read_text())
            self.assertFalse(failure["fastapi_request_ready"])
            self.assertEqual(len(state["stages"]), 1)
            return
        result = execute_start_return(None, config, effects)
        self.assertEqual(len(state["stages"]), 2)
        self.assertEqual(len(state["captures"]), 3)
        self.assertTrue(result["fastapi_request_ready"])
        self.assertFalse(result["fastapi_request_sent"])
        self.assertLess(result["start_arrival_position_error_m"], 1e-9)
        self.assertLess(result["start_arrival_heading_error_rad"], 1e-9)
        self.assertEqual(state["stages"][-1]["leg"].raw_waypoints[-1].pose, target)
        first = state["stages"][0]["request"].sealed
        full = json.loads(Path(first["full_return_route_json"]).read_text())
        original = evaluate_route_uncertainty_admission(costmap,
            tuple(Pose2D(**p) for p in full["poses"]), context_covariance(fixture),
            state["stages"][0]["config"])
        self.assertFalse(original.decision.accepted)

    def test_recorded_route_completes_through_two_real_planned_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run(Path(folder))

    def test_fresh_worse_covariance_prevents_second_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run(Path(folder), worsen_after_stop=True)


def context_covariance(fixture):
    from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
    return PlanarCovariance(**fixture["covariance_m2"])
