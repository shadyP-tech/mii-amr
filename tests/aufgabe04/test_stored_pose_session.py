"""Load a real artifact chain from a saved session, without ROS or motion."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.admitted_pose_route import validate_admitted_pose_route_binding
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import evaluate_admitted_return_stage_uncertainty
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import write_coverage_survey_plan
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegMotionAuthorization, ROUTINE_MISSION_LEG_KINDS, MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    write_mission_leg_motion_authorization,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.candidate.approach import execute_candidate_approach_phase
from scripts.aufgabe04.real_robot.configuration.profile import load_real_robot_profile, real_robot_profile_sha256, write_real_robot_profile
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.real_robot.execution.artifact_paths import resolve_child_artifact_paths
from scripts.aufgabe04.real_robot.mission.start_return_readiness import load_stored_pose_tour_readiness
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import StoredPoseNavigationEffects, execute_stored_pose_navigation
from scripts.aufgabe04.real_robot.mission.stored_pose_session import load_stored_pose_session
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
from tests.aufgabe04 import test_qr_pose_discovery_execution as fixtures
from tests.aufgabe04.test_candidate_route_uncertainty_readiness import _preflight_payload


REPO = Path(__file__).resolve().parents[2]


class StoredPoseSessionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.QrPoseDiscoveryExecutionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def source(self):
        case = self.fixture.case
        root, helper = case.root, case.fixture
        site_payload = json.loads((REPO / "docs/setups/aufgabe04_lab_20260817.json").read_text())
        site_payload["physical_site_id"] = "fixture_site"
        site_payload["station_setup"]["expected_stand_count"] = 1
        site_path = root / "fixture_site.json"
        site_path.write_text(json.dumps(site_payload))
        profile = replace(load_real_robot_profile(REPO / "configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json"),
            physical_site_id="fixture_site", physical_site_sha256=file_sha256(site_path))
        profile_path = root / "profile.json"
        write_real_robot_profile(profile_path, profile)
        map_yaml = REPO / site_payload["map_measurement"]["map_yaml"]
        _, bundle = load_occupancy_grid_with_bundle(map_yaml,
            semantic_map_id=site_payload["map_measurement"]["semantic_map_id"], planning_frame="map")
        config = helper._config(root, (helper._candidate("candidate_a", 2., 0.), helper._candidate("unvisited", 3., 1.)))
        config = replace(config, expected_stand_count=1, robot_radius_m=profile.robot_radius_m,
            semantic_map_id=bundle.semantic_map_id, map_yaml=map_yaml,
            robot_profile_sha256=real_robot_profile_sha256(profile), calibration_profile_sha256=profile.calibration_profile_sha256,
            plan=replace(config.plan, map_bundle_sha256=bundle.bundle_sha256,
                config=replace(config.plan.config, expected_stand_count=1)),
            snapshot=replace(config.snapshot, map_bundle_sha256=bundle.bundle_sha256))
        config = helper._write_frame_registry(config, frozen_map_from_odom=PlanarTransform2D(1., 0., 0.))
        frames = iter((CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(0., 0., 0.)),
                       CandidatePlanningFrame(Pose2D(.5, 0., 0.), PlanarTransform2D(.2, 0., 0.))))
        effects = self.fixture.effects(config, qr_only={"candidate_a"}, identities={"candidate_a": "QR002"},
            mutate=lambda data: data.update(robot_pose=asdict(Pose2D(.5, 0., .1)),
                robot_profile_sha256=config.robot_profile_sha256, calibration_profile_sha256=config.calibration_profile_sha256))
        completed = execute_candidate_approach_phase(config, replace(effects, admit_planning_frame=lambda _: next(frames)))
        write_coverage_survey_plan(config.survey_root / "coverage_plan.json", config.plan)
        auth_path = config.session_root / "motion_authorization" / "mission_leg_motion_authorization.json"
        write_mission_leg_motion_authorization(auth_path, MissionLegMotionAuthorization(
            session_id=config.session_id, robot_id=profile.robot_id, namespace=profile.resolved_runtime().namespace,
            cmd_vel_topic=profile.resolved_runtime().cmd_vel_topic, semantic_map_id=config.semantic_map_id,
            localization_branch_proof_id="known_start", allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS,
            scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, operator_confirmation="RUN"))
        model_path = REPO / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
        summary = {**completed.to_mission_summary_fields(), "session_id": config.session_id,
            "status": "failed_closed", "camera_validation_complete": True,
            "survey_root": str(config.survey_root), "candidate_snapshot": str(config.snapshot_path),
            "candidate_snapshot_sha256": candidate_snapshot_sha256(config.snapshot),
            "stand_model_profile": str(model_path), "stand_model_profile_sha256": load_measured_physical_stand_model(model_path).sha256}
        (config.session_root / "mission_summary.json").write_text(json.dumps(summary))
        return config, profile_path, site_path

    def test_loads_completed_camera_session_even_if_old_return_failed(self):
        config, profile_path, site_path = self.source()
        session = load_stored_pose_session(config.session_root, profile_path, physical_site_path=site_path)
        self.assertEqual(set(session.poses_by_qr), {"QR002"})
        self.assertEqual(session.config.session_id, config.session_id)
        self.assertEqual(session.config.map_yaml, config.map_yaml)
        self.assertEqual(session.config.snapshot, config.snapshot)
        self.assertEqual(session.poses_by_qr["QR002"].pose, Pose2D(.5, 0., .1))
        self.assertFalse(session.completed.motion_authorized)
        self.assertEqual(session.config.mission_leg_motion_authorization_json,
            (config.session_root / "motion_authorization/mission_leg_motion_authorization.json").resolve())
        self.assertEqual(session.poses_by_qr["QR002"].evidence["source_session_id"], config.session_id)

    def test_changed_catalog_and_incomplete_camera_session_are_rejected(self):
        config, profile_path, site_path = self.source()
        summary_path = config.session_root / "mission_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["camera_validation_complete"] = False
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "completed camera"):
            load_stored_pose_session(config.session_root, profile_path, physical_site_path=site_path)
        summary["camera_validation_complete"] = True
        summary_path.write_text(json.dumps(summary))
        catalog = Path(summary["qr_observation_pose_catalog"])
        payload = json.loads(catalog.read_text())
        payload["records"][0]["robot_observation_pose"]["x_m"] = .6
        catalog.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "hash"):
            load_stored_pose_session(config.session_root, profile_path, physical_site_path=site_path)

    def test_reconstructed_session_plans_and_admits_exact_qr_visit(self):
        config, profile_path, site_path = self.source()
        session = load_stored_pose_session(config.session_root, profile_path, physical_site_path=site_path)
        config = session.config
        state = {"pose": Pose2D(0., 0., 0.), "legs": []}
        grid, _ = load_occupancy_grid_with_bundle(config.map_yaml,
            semantic_map_id=config.semantic_map_id, planning_frame=config.planning_frame)
        costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(config.plan.arena_bounds)

        def capture(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_preflight_payload(state["pose"])))
            state["preflight"] = path
            return CandidatePlanningFrame(state["pose"], PlanarTransform2D(0., 0., 0.))

        def motion(request):
            sealed = request.sealed
            resolve_child_artifact_paths(session_root=request.session_root, sealed=sealed)
            self.assertEqual(set(sealed), {"route_csv", "diagnostics_json", "route_certificate_json"})
            leg = load_route_leg(Path(sealed["route_csv"]), 0, thinning_min_spacing_m=0.)
            binding = validate_admitted_pose_route_binding(Path(sealed["diagnostics_json"]), leg,
                candidate_snapshot_path=request.candidate_snapshot_path)
            self.assertTrue(binding.ok, binding.failures)
            context = load_stored_pose_tour_readiness(CandidateRouteUncertaintyReadinessRequest(
                state["preflight"], state["pose"], "map", "odom", config.robot_radius_m,
                config.uncertainty_sigma_multiplier))
            metadata = json.loads(Path(sealed["diagnostics_json"]).read_text())["metadata"]
            self.assertEqual(metadata["route_purpose"], "stored_pose_tour")
            evidence = json.loads(Path(metadata["target_evidence_json"]).read_text())
            self.assertEqual(evidence["qr_id"], "QR002")
            admission = evaluate_admitted_return_stage_uncertainty(
                costmap, tuple(w.pose for w in leg.raw_waypoints), context.covariance, context.admission_config,
                start_pose=state["pose"], target_evidence_sha256=metadata["target_evidence_sha256"],
                is_final_stage=metadata["return_to_start_stage"]["final_stage"])
            self.assertTrue(admission.decision.accepted)
            state["legs"].append(leg)
            state["pose"] = leg.raw_waypoints[-1].pose
            return self.fixture.case.fixture._completed(request)

        result = execute_stored_pose_navigation(session.poses_by_qr["QR002"], config,
            StoredPoseNavigationEffects(capture, motion), tour_session_id="tour_real_planner", visit_index=0,
            output_root=config.session_root.parent / "fresh_tour_visit")
        self.assertTrue(result["arrival_verified"])
        self.assertEqual(result["qr_id"], "QR002")
        self.assertEqual(len(state["legs"]), 1)
        self.assertEqual(state["pose"], Pose2D(.3, 0., .1))


if __name__ == "__main__":
    unittest.main()
