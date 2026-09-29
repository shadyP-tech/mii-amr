"""The selection retry must reacquire all stopped-frame evidence, without motion."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateRouteOption,
    NoFeasibleCameraCandidateError,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
    NoUncertaintyAdmittedCameraCandidateError,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import RouteUncertaintyAdmissionConfig
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.approach import (
    CameraCandidateInitialSelection,
    CandidateApproachEffects,
    execute_candidate_approach_phase,
)
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures


class SelectedBeforeMotion(RuntimeError):
    """Stop at materialization so a successful selection cannot hide motion."""


class CandidateSelectionLocalizationRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture = fixtures.AutonomousCandidateApproachTest()
        candidates = (fixture._candidate("candidate_1", 2.0, 0.0),
                      fixture._candidate("candidate_2", 3.0, 0.0))
        self.config = fixture._write_frame_registry(
            replace(fixture._config(self.root, candidates),
                    require_uncertainty_aware_selection=True, robot_radius_m=0.105),
            frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0),
        )
        self.frames = (
            CandidatePlanningFrame(Pose2D(0.0, 0.0, 0.0), PlanarTransform2D(0.0, 0.0, 0.0)),
            CandidatePlanningFrame(Pose2D(0.03, 0.20, 0.01), PlanarTransform2D(0.0, 0.20, 0.01)),
        )
        self.admissions, self.readiness, self.requests, self.events = [], [], [], []
        self.motion, self.capture, self.commit = Mock(), Mock(), Mock()
        self.plan = Mock(side_effect=SelectedBeforeMotion())
        self.effects = CandidateApproachEffects(
            read_current_pose=Mock(side_effect=AssertionError("must use admitted pose")),
            admit_planning_frame=self.admit,
            load_route_uncertainty_readiness=self.load_readiness,
            select_initial_preapproach=self.reject,
            plan_preapproach=self.plan,
            run_motion_leg=self.motion,
            capture_observation=self.capture,
            commit_decision=self.commit,
            event_sink=lambda _path, event: self.events.append(event),
            clock=lambda: 10.0,
        )

    def admit(self, path):
        epoch = len(self.admissions)
        self.admissions.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump({"covariance": (0.0055, 0.0035)[epoch]}, stream)
        # No candidate may be discarded between the two no-motion attempts.
        self.assertEqual(self.dispositions(), ["pending", "pending"])
        return self.frames[epoch]

    def load_readiness(self, request):
        self.readiness.append(request)
        variance = json.loads(request.preflight_json.read_text())["covariance"]
        return CandidateRouteUncertaintyContext(
            covariance=PlanarCovariance(variance, 0.0, variance),
            admission_config=RouteUncertaintyAdmissionConfig(
                robot_radius_m=request.robot_radius_m, collision_margin_m=0.02,
                fixed_odom_tracking_bound_m=0.03, empirical_odom_drift_bound_m=0.02,
                braking_latency_distance_m=0.01, localization_sigma_multiplier=request.sigma_multiplier,
                heading_sigma_rad=0.01, heading_lever_arm_m=0.105, sampling_spacing_m=0.005,
            ),
            source_evidence={"source_preplanning_localization_json": str(request.preflight_json)},
        )

    def reject(self, request):
        self.requests.append(request)
        raise NoUncertaintyAdmittedCameraCandidateError({
            "route_uncertainty_selection_applied": True,
            "source": dict(request.route_uncertainty_context.source_evidence),
            "recorded_margin_m": -0.017,
            "motion_authorized": False,
        })

    def dispositions(self):
        goal = json.loads((self.config.session_root / "candidate_goal_progress.json").read_text())
        return [row["disposition"] for row in goal["candidate_dispositions"]]

    def assert_no_motion(self):
        self.motion.assert_not_called()
        self.capture.assert_not_called()
        self.commit.assert_not_called()

    def test_second_epoch_replans_from_fresh_pose_covariance_and_source_projection(self):
        def select(request):
            if not self.requests:
                return self.reject(request)
            self.requests.append(request)
            return CameraCandidateInitialSelection("candidate_1", None, {
                "selected_candidate_uid": "candidate_1",
                "route_uncertainty_selection_applied": True,
                "source": dict(request.route_uncertainty_context.source_evidence),
            })

        with self.assertRaises(SelectedBeforeMotion):
            execute_candidate_approach_phase(self.config, replace(self.effects, select_initial_preapproach=select))

        self.assertEqual(len(self.admissions), 2)
        first, second = self.requests
        self.assertEqual(first.unresolved, second.unresolved)
        self.assertIs(first.source_registry, second.source_registry)
        self.assertEqual(second.current_pose, self.frames[1].current_pose)
        self.assertIs(second.planning_frame, self.frames[1])
        self.assertEqual(self.readiness[1].expected_start, self.frames[1].current_pose)
        self.assertEqual(first.route_uncertainty_context.covariance.xx_m2, 0.0055)
        self.assertEqual(second.route_uncertainty_context.covariance.xx_m2, 0.0035)
        self.assertEqual(first.route_uncertainty_context.admission_config, second.route_uncertainty_context.admission_config)
        self.assertNotEqual(first.config.snapshot, second.config.snapshot)
        self.assertNotEqual(first.config.snapshot_path, second.config.snapshot_path)
        self.assertTrue(first.config.snapshot_path.exists())
        self.assertTrue(second.config.snapshot_path.exists())
        self.assertEqual(self.admissions[0].read_text(), '{"covariance": 0.0055}')
        self.assertIn("epoch_001", str(self.admissions[1]))
        self.assertEqual(self.dispositions(), ["pending", "pending"])
        materialized = self.plan.call_args.args[0]
        self.assertEqual(materialized.start, second.current_pose)
        self.assertEqual(materialized.snapshot_path, second.config.snapshot_path)
        evidence = materialized.selection_evidence
        self.assertEqual(evidence["selection_localization_epoch"], 1)
        self.assertEqual(evidence["selection_planning_frame_evidence_path"], str(self.admissions[1]))
        self.assertIn("selection_000_epoch_001", evidence["candidate_frame_projection_path"])
        self.assertEqual(self.events[0]["event"], "camera_candidate_selection_localization_refresh")
        self.assertEqual(self.events[0]["recorded_margin_m"], -0.017)
        self.assertFalse(self.events[0]["motion_authorized"])
        self.assert_no_motion()

    def test_two_rejections_exhaust_exactly_one_refresh_without_motion(self):
        with self.assertRaises(NoUncertaintyAdmittedCameraCandidateError) as caught:
            execute_candidate_approach_phase(self.config, self.effects)
        self.assertEqual(len(self.admissions), 2)
        self.assertEqual(len(self.readiness), 2)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0].unresolved, self.requests[1].unresolved)
        self.assertTrue(caught.exception.to_evidence()["localization_refresh_exhausted"])
        self.assertEqual(self.dispositions(), ["route_admission_exhausted"] * 2)
        self.assertEqual([event["event"] for event in self.events], [
            "camera_candidate_selection_localization_refresh", "camera_candidate_selection_failed"])
        self.plan.assert_not_called()
        self.assert_no_motion()

    def test_accepted_fresh_frame_reaches_motion_and_camera_handoff(self):
        def select(request):
            if not self.requests:
                return self.reject(request)
            self.requests.append(request)
            return CameraCandidateInitialSelection("candidate_1", None, {
                "selected_candidate_uid": "candidate_1",
                "route_uncertainty_selection_applied": True,
            })

        def plan(request):
            request.output_dir.mkdir(parents=True, exist_ok=True)
            (request.output_dir / "candidate_snapshot.json").write_text(request.snapshot_path.read_text())
            return {"route_csv": str(request.output_dir / "route.csv")}

        def camera_handoff(**kwargs):
            frame = kwargs["observation_frame"]
            self.assertIs(frame.planning_frame, self.frames[1])
            self.assertEqual(frame.observation_pose, self.frames[1].current_pose)
            self.assertEqual(frame.config.snapshot_path, self.requests[1].config.snapshot_path)
            raise SelectedBeforeMotion("stop after verifying actual motion handoff")

        motion = Mock(side_effect=fixtures.AutonomousCandidateApproachTest._completed)
        with patch("scripts.aufgabe04.real_robot.candidate.approach._capture_candidate_camera_result",
                   side_effect=camera_handoff), self.assertRaises(SelectedBeforeMotion):
            execute_candidate_approach_phase(self.config, replace(
                self.effects, select_initial_preapproach=select,
                plan_preapproach=plan, run_motion_leg=motion))
        motion.assert_called_once()
        motion_request = motion.call_args.args[0]
        self.assertEqual(motion_request.candidate_snapshot_path.read_text(),
                         self.requests[1].config.snapshot_path.read_text())
        self.assertEqual(len(self.admissions), 2)

    def test_first_epoch_acceptance_does_not_reacquire(self):
        select = Mock(return_value=CameraCandidateInitialSelection("candidate_1", None, {
            "selected_candidate_uid": "candidate_1",
            "route_uncertainty_selection_applied": True,
        }))
        with self.assertRaises(SelectedBeforeMotion):
            execute_candidate_approach_phase(self.config, replace(self.effects, select_initial_preapproach=select))
        self.assertEqual(len(self.admissions), 1)
        self.assertEqual(len(self.readiness), 1)
        select.assert_called_once()
        evidence = self.plan.call_args.args[0].selection_evidence
        self.assertEqual(evidence["selection_localization_epoch"], 0)
        self.assertFalse(any(event["event"] == "camera_candidate_selection_localization_refresh"
                             for event in self.events))
        self.assert_no_motion()

    def test_static_infeasibility_never_refreshes(self):
        option = CameraCandidateRouteOption(
            candidate_uid="candidate_1", feasible=False, failure_reason="no_path",
            route_length_m=None, turn_burden_rad=None, initial_turn_rad=None,
            inside_requested_standoff=False, support_class="coverage_admitted",
            confidence=0.9, hit_count=4,
        )
        select = Mock(side_effect=NoFeasibleCameraCandidateError((option,)))
        with self.assertRaises(NoFeasibleCameraCandidateError):
            execute_candidate_approach_phase(self.config, replace(self.effects, select_initial_preapproach=select))
        self.assertEqual(len(self.admissions), 1)
        select.assert_called_once()
        self.assertEqual(self.dispositions(), ["no_feasible_route"] * 2)
        self.plan.assert_not_called()
        self.assert_no_motion()

    def test_malformed_or_failed_readiness_never_refreshes(self):
        for value in (None, ValueError("malformed stationary envelope")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                self.config = replace(self.config, session_root=Path(tmp))
                self.admissions.clear()
                loader = (Mock(side_effect=value) if isinstance(value, Exception)
                          else Mock(return_value=value))
                select = Mock()
                with self.assertRaises((TypeError, ValueError)):
                    execute_candidate_approach_phase(self.config, replace(
                        self.effects, load_route_uncertainty_readiness=loader,
                        select_initial_preapproach=select))
                self.assertEqual(len(self.admissions), 1)
                select.assert_not_called()
                self.assertEqual(self.dispositions(), ["pending"] * 2)
                self.assert_no_motion()

    def test_unbound_legacy_uncertainty_error_does_not_refresh(self):
        select = Mock(side_effect=NoUncertaintyAdmittedCameraCandidateError({}))
        effects = replace(self.effects, admit_planning_frame=None,
                          read_current_pose=lambda: Pose2D(0.0, 0.0, 0.0),
                          select_initial_preapproach=select)
        with self.assertRaises(NoUncertaintyAdmittedCameraCandidateError):
            execute_candidate_approach_phase(replace(self.config, require_uncertainty_aware_selection=False), effects)
        select.assert_called_once()
        self.assertEqual(self.admissions, [])
        self.assertEqual(len(self.events), 1)
        self.plan.assert_not_called()
        self.assert_no_motion()
