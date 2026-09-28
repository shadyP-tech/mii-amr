"""Stored discovery -> exact Start navigation -> server readiness ordering."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind, MAX_RETURN_TO_START_LEGS
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import RouteUncertaintyAdmissionConfig
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.approach import execute_candidate_approach_phase
from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_to_dict, load_recommendation
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.real_robot.mission.start_return import StartReturnEffects, execute_start_return
from scripts.aufgabe04.real_robot.mission.stored_start_pose import load_stored_start_pose
from tests.aufgabe04 import test_qr_pose_discovery_execution as fixtures


class StartReturnTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.QrPoseDiscoveryExecutionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def completed(self, qr="Start", *, facing=False):
        case = self.fixture.case
        helper = case.fixture
        config = helper._config(case.root, (
            helper._candidate("candidate_a", 2., 0.), helper._candidate("unvisited", 3., 1.),
        ))
        config = replace(config, expected_stand_count=1, robot_radius_m=.105,
                         robot_profile_sha256="a" * 64, calibration_profile_sha256="b" * 64,
                         plan=replace(config.plan, config=replace(config.plan.config, expected_stand_count=1)))
        config = helper._write_frame_registry(config, frozen_map_from_odom=PlanarTransform2D(1., 0., 0.))
        frames = iter((
            CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(0., 0., 0.)),
            CandidatePlanningFrame(Pose2D(.5, 0., 0.), PlanarTransform2D(.2, 0., 0.)),
        ))
        effects = self.fixture.effects(config, qr_only={"candidate_a"}, identities={"candidate_a": qr},
            mutate=lambda data: data.update(robot_pose={"x_m": .5, "y_m": 0., "yaw_rad": .1}))
        if facing:
            def capture(request):
                candidate = request.candidate
                recommendation = build_real_viewpoint_recommendation(
                    stream_id=f"{config.session_id}_{candidate.candidate_uid}", stand_id=candidate.candidate_uid,
                    planning_frame="map", stand_center=Pose2D(candidate.geometry.x_m, candidate.geometry.y_m),
                    stand_radius_m=candidate.geometry.radius_m, stand_uncertainty_m=candidate.geometry.uncertainty_m,
                    robot_pose=Pose2D(.5, 0., 0.), stand_axis_rad=1.5707963267948966,
                    axis_confidence=.95, axis_sample_count=7, sensor_stamp_sec=9.9,
                    expected_qr_id=qr, observed_qr_ids=(qr,), target_distance_m=.35, observation_unix_sec=9.9,
                )
                request.output_dir.mkdir(parents=True, exist_ok=True)
                path = request.output_dir / "recommendation.json"
                path.write_text(json.dumps(recommendation_to_dict(recommendation)))
                return CandidateObservation(path, qr, None)
            def validate(request):
                return {"candidate_uid": request.candidate.candidate_uid,
                        "facing_pose": asdict(load_recommendation(request.recommendation_path).material_target.pose),
                        "recommendation_json": str(request.recommendation_path),
                        "camera_recommendation_sha256": file_sha256(request.recommendation_path)}
            effects = replace(effects, capture_observation=capture, validate_facing=validate)
        completed = execute_candidate_approach_phase(config, replace(effects, admit_planning_frame=lambda _: next(frames)))
        return config, completed

    def test_loads_stored_qr_pose_with_complete_artifact_chain(self):
        config, completed = self.completed()
        stored = load_stored_start_pose(completed, config)
        self.assertEqual(stored.candidate_uid, "candidate_a")
        self.assertEqual(stored.pose, Pose2D(.5, 0., .1))
        self.assertEqual(stored.source_frame.map_from_odom, PlanarTransform2D(.2, 0., 0.))
        self.assertEqual(stored.evidence["pose_kind"], "qr_verified_observation_pose")

    def test_start_qr_is_case_sensitive(self):
        config, completed = self.completed("start")
        with self.assertRaisesRegex(ValueError, "exactly one.*'Start'"):
            load_stored_start_pose(completed, config)

    def test_geometry_target_uses_admitted_facing_pose_instead_of_observer_pose(self):
        config, completed = self.completed(facing=True)
        stored = load_stored_start_pose(completed, config)
        self.assertAlmostEqual(stored.pose.x_m, .85)
        self.assertAlmostEqual(stored.pose.y_m, 0.)
        self.assertAlmostEqual(stored.pose.yaw_rad, 0.)
        self.assertEqual(stored.evidence["pose_kind"], "geometry_validated_facing_pose")

    def test_changed_catalog_is_rejected_before_motion(self):
        config, completed = self.completed()
        path = completed.qr_observation_catalog_path
        payload = json.loads(path.read_text())
        payload["records"][0]["robot_observation_pose"]["x_m"] = 1.5
        path.write_text(json.dumps(payload))
        motion = Mock()
        with self.assertRaisesRegex(ValueError, "hash"):
            execute_start_return(completed, config, StartReturnEffects(Mock(), motion))
        motion.assert_not_called()
        self.assertFalse(json.loads((config.session_root / "return_to_start/failure.json").read_text())["fastapi_request_ready"])

    def test_even_rehashed_source_pose_cannot_replace_catalog_pose(self):
        config, completed = self.completed()
        record = completed.qr_observation_records[0]
        path = Path(record["qr_observation_pose_json"])
        payload = json.loads(path.read_text())
        payload.pop("qr_verified_observation_pose_sha256")
        payload["robot_pose"]["yaw_rad"] = .2
        path.unlink()
        write_content_hashed_json(path, payload, hash_field="qr_verified_observation_pose_sha256")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_stored_start_pose(completed, config)

    def test_duplicate_stored_start_pose_is_ambiguous_even_with_valid_hash(self):
        config, completed = self.completed()
        path = completed.qr_observation_catalog_path
        payload = json.loads(path.read_text())
        payload.pop("qr_observation_pose_catalog_sha256")
        payload["records"].append(dict(payload["records"][0]))
        path.unlink()
        digest = write_content_hashed_json(path, payload, hash_field="qr_observation_pose_catalog_sha256")
        with self.assertRaisesRegex(ValueError, "ambiguous stored poses"):
            load_stored_start_pose(replace(completed, qr_observation_catalog_sha256=digest), config)

    @staticmethod
    def uncertainty(request):
        return CandidateRouteUncertaintyContext(
            PlanarCovariance(.0001, 0., .0001),
            RouteUncertaintyAdmissionConfig(robot_radius_m=.105, collision_margin_m=.02,
                fixed_odom_tracking_bound_m=.03, empirical_odom_drift_bound_m=.02,
                localization_sigma_multiplier=2., heading_lever_arm_m=.105, sampling_spacing_m=.005,
                braking_latency_distance_m=.075, heading_sigma_rad=.01,
                heading_reference_x_m=request.expected_start.x_m,
                heading_reference_y_m=request.expected_start.y_m),
            {"source_preplanning_localization_json": str(request.preflight_json)},
        )

    def effects(self, config, *, arrival=Pose2D(.7, 0., .1), status="completed"):
        frames = iter((
            CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.4, 0., 0.)),
            CandidatePlanningFrame(arrival, PlanarTransform2D(.4, 0., 0.)),
        ))
        order = []
        def plan(**kwargs):
            order.append("plan")
            self.assertEqual(kwargs["snapshot"].candidate_uids, ("candidate_a", "unvisited"))
            self.assertAlmostEqual(kwargs["target"].x_m, .7)
            self.assertEqual(kwargs["target"].yaw_rad, .1)
            self.assertTrue(Path(kwargs["target_evidence"]["catalog_path"]).is_file())
            return {"candidate_snapshot": str(kwargs["snapshot_path"]),
                    "is_final_stage": True, "stage_target_pose": asdict(kwargs["target"])}
        def motion(request):
            order.append("motion")
            self.assertEqual(request.mission_leg_kind, MissionLegKind.RETURN_TO_START)
            self.assertEqual(request.target_id, "candidate_a")
            self.assertFalse((config.session_root / "return_to_start/arrival.json").exists())
            return replace(self.fixture.case.fixture._completed(request), status=status)
        def admit(_):
            order.append("stationary_frame")
            return next(frames)
        return StartReturnEffects(admit, motion, plan_route=plan,
                                 load_route_uncertainty_readiness=self.uncertainty), order

    def test_reprojects_pose_and_full_pool_before_return_and_checks_arrival(self):
        config, completed = self.completed()
        effects, order = self.effects(config)
        result = execute_start_return(completed, config, effects)
        self.assertEqual(order, ["stationary_frame", "plan", "motion", "stationary_frame"])
        self.assertTrue(result["start_pose_reached"])
        self.assertTrue(result["fastapi_request_ready"])
        self.assertFalse(result["fastapi_request_sent"])

    def test_failed_child_does_not_enable_server_request(self):
        config, completed = self.completed()
        effects, order = self.effects(config, status="stopped")
        with self.assertRaisesRegex(RuntimeError, "motion failed"):
            execute_start_return(completed, config, effects)
        self.assertEqual(order, ["stationary_frame", "plan", "motion"])
        self.assertFalse((config.session_root / "return_to_start/arrival.json").exists())

    def test_false_child_completion_cannot_skip_arrival_check(self):
        config, completed = self.completed()
        effects, _ = self.effects(config, arrival=Pose2D(.7, 0., .8))
        with self.assertRaisesRegex(RuntimeError, "arrival tolerance"):
            execute_start_return(completed, config, effects)

    def test_frame_identity_change_stops_before_motion(self):
        config, completed = self.completed()
        frame = CandidatePlanningFrame(Pose2D(.7, 0., .1), PlanarTransform2D(.4, 0., 0.), odom_frame="other_odom")
        motion = Mock()
        with self.assertRaisesRegex(ValueError, "frame identities changed"):
            execute_start_return(completed, config, StartReturnEffects(lambda _: frame, motion))
        motion.assert_not_called()

    def test_already_at_start_requires_stationary_admission_without_new_motion(self):
        config, completed = self.completed()
        frame = CandidatePlanningFrame(Pose2D(.7, 0., .1), PlanarTransform2D(.4, 0., 0.))
        motion = Mock()
        plan = Mock()
        result = execute_start_return(completed, config, StartReturnEffects(lambda _: frame, motion, plan_route=plan))
        self.assertEqual(result["return_to_start_status"], "already_at_start")
        self.assertTrue(result["fastapi_request_ready"])
        motion.assert_not_called()
        plan.assert_not_called()

    def staged_effects(self, config, targets, frames, *, fail_index=None):
        frame_values = iter(frames)
        planned = []
        requests = []
        readiness = []
        def admit(path):
            value = next(frame_values)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value.to_evidence()))
            return value
        def load(request):
            readiness.append(request)
            self.assertTrue(request.preflight_json.is_file())
            return self.uncertainty(request)
        def plan(**kwargs):
            index = kwargs["return_stage_index"]
            planned.append(kwargs)
            target, final = targets[index]
            return {"candidate_snapshot": str(kwargs["snapshot_path"]),
                    "is_final_stage": final, "stage_target_pose": asdict(target)}
        def motion(request):
            requests.append(request)
            self.assertFalse((config.session_root / "return_to_start/arrival.json").exists())
            if request.mission_leg_index:
                prior = config.session_root / "return_to_start/legs" / f"{request.mission_leg_index-1:03d}" / "arrival.json"
                self.assertFalse(json.loads(prior.read_text())["fastapi_request_ready"])
            outcome = self.fixture.case.fixture._completed(request)
            return replace(outcome, status="preflight_failed" if fail_index == request.mission_leg_index else "completed")
        return StartReturnEffects(admit, motion, plan_route=plan,
                                  load_route_uncertainty_readiness=load), planned, requests, readiness

    def test_stages_reproject_arrival_and_full_pool_into_each_fresh_epoch(self):
        config, completed = self.completed()
        frames = [
            CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.4, 0., 0.)),
            CandidatePlanningFrame(Pose2D(.5, 0., 0.), PlanarTransform2D(.5, 0., 0.)),
            CandidatePlanningFrame(Pose2D(.9, 0., .1), PlanarTransform2D(.6, 0., 0.)),
        ]
        effects, plans, requests, readiness = self.staged_effects(config,
            [(Pose2D(.4, 0., 0.), False), (Pose2D(.8, 0., .1), True)], frames)
        result = execute_start_return(completed, config, effects)
        self.assertTrue(result["fastapi_request_ready"])
        self.assertEqual(result["start_return_leg_count"], 2)
        self.assertEqual([r.mission_leg_index for r in requests], [0, 1])
        self.assertEqual(len({r.run_id for r in requests}), 2)
        self.assertEqual(len({r.permit_json_path for r in requests}), 2)
        self.assertEqual([r.target_id for r in requests], ["candidate_a"] * 2)
        self.assertEqual(plans[0]["snapshot"].candidate_uids, plans[1]["snapshot"].candidate_uids)
        self.assertAlmostEqual(plans[1]["snapshot"].candidates[0].geometry.x_m -
                               plans[0]["snapshot"].candidates[0].geometry.x_m, .1)
        self.assertAlmostEqual(plans[1]["target"].x_m, .8)
        self.assertEqual(readiness[1].expected_start, frames[1].current_pose)
        self.assertEqual(readiness[1].preflight_json.name, "arrival_localization.json")
        self.assertEqual(result["start_return_legs"][0]["arrival_target_pose"]["x_m"], .5)
        self.assertAlmostEqual(result["start_arrival_target_pose"]["x_m"], .9)

    def test_later_child_failure_retains_discovery_and_does_not_retry(self):
        config, completed = self.completed()
        frames = [CandidatePlanningFrame(Pose2D(x, 0., 0.), PlanarTransform2D(.4, 0., 0.)) for x in (0., .4)]
        effects, plans, requests, _ = self.staged_effects(config,
            [(Pose2D(.4, 0., 0.), False), (Pose2D(.7, 0., .1), True)], frames, fail_index=1)
        with self.assertRaisesRegex(RuntimeError, "preflight_failed"):
            execute_start_return(completed, config, effects)
        self.assertEqual(len(requests), 2)
        failure = json.loads((config.session_root / "return_to_start/failure.json").read_text())
        self.assertFalse(failure["fastapi_request_ready"])
        self.assertTrue(failure["start_return_legs"][0]["arrival_verified"])
        self.assertFalse(failure["start_return_legs"][1]["arrival_verified"])
        self.assertTrue(completed.qr_observation_catalog_path.is_file())

    def test_false_intermediate_completion_cannot_start_next_leg(self):
        config, completed = self.completed()
        frame = CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.4, 0., 0.))
        effects, _, requests, _ = self.staged_effects(config, [(Pose2D(.4, 0., 0.), False)], [frame, frame])
        with self.assertRaisesRegex(RuntimeError, "arrival tolerance"):
            execute_start_return(completed, config, effects)
        self.assertEqual(len(requests), 1)

    def test_map_correction_cannot_fabricate_intermediate_progress(self):
        config, completed = self.completed()
        frames = [
            CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.4, 0., 0.)),
            CandidatePlanningFrame(Pose2D(.4, 0., 0.), PlanarTransform2D(.8, 0., 0.)),
        ]
        effects, _, requests, _ = self.staged_effects(config, [(Pose2D(.4, 0., 0.), False)], frames)
        with self.assertRaisesRegex(RuntimeError, "arrival tolerance"):
            execute_start_return(completed, config, effects)
        self.assertEqual(len(requests), 1)

    def test_leg_budget_blocks_nonfinal_fourth_leg_before_motion(self):
        config, completed = self.completed()
        frames = [CandidatePlanningFrame(Pose2D(0., y, 0.), PlanarTransform2D(.4, 0., 0.))
                  for y in (0., .3, .6, .9)]
        effects, _, requests, _ = self.staged_effects(config,
            [(Pose2D(0., .3 * (i + 1), 0.), False) for i in range(MAX_RETURN_TO_START_LEGS)], frames)
        with self.assertRaisesRegex(RuntimeError, "leg budget exhausted"):
            execute_start_return(completed, config, effects)
        self.assertEqual(len(requests), MAX_RETURN_TO_START_LEGS - 1)

    def test_missing_uncertainty_evidence_stops_before_motion(self):
        config, completed = self.completed()
        frame = CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.4, 0., 0.))
        motion = Mock()
        with self.assertRaisesRegex(ValueError, "preflight evidence is unavailable"):
            execute_start_return(completed, config, StartReturnEffects(lambda _: frame, motion))
        motion.assert_not_called()

    def test_arrival_publication_failure_never_persists_ready_failure(self):
        config, completed = self.completed()
        frame = CandidatePlanningFrame(Pose2D(.7, 0., .1), PlanarTransform2D(.4, 0., 0.))
        def write(path, *args, **kwargs):
            if path.name == "arrival.json":
                raise OSError("arrival publication failed")
            return write_content_hashed_json(path, *args, **kwargs)
        with patch("scripts.aufgabe04.real_robot.mission.start_return.write_content_hashed_json", side_effect=write):
            with self.assertRaisesRegex(OSError, "publication failed"):
                execute_start_return(completed, config, StartReturnEffects(lambda _: frame, Mock()))
        failure = json.loads((config.session_root / "return_to_start/failure.json").read_text())
        self.assertEqual(failure["return_to_start_status"], "failed_closed")
        self.assertFalse(failure["fastapi_request_ready"])


if __name__ == "__main__":
    unittest.main()
