"""Runtime regressions for target admission independently of route admission."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import load_stand_survey_registry
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, CandidateObservation,
    _admit_camera_arrival_geometry, _select_initial_preapproach,
    execute_candidate_approach_phase,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import CandidateQrGoalIncompleteError
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
from tests.aufgabe04 import test_autonomous_candidate_approach as approach_fixtures
from scripts.aufgabe04.navigation.approach import candidate_preapproach_selection


class CameraTargetGateRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.fixture = approach_fixtures.AutonomousCandidateApproachTest()

    def config(self, root, positions, *, expected=None, narrow_arena=False):
        candidates = tuple(self.fixture._candidate(uid, x, y) for uid, x, y in positions)
        config = self.fixture._config(root, candidates)
        plan = config.plan
        if expected is not None:
            plan = replace(plan, config=replace(plan.config, expected_stand_count=expected))
            config = replace(config, expected_stand_count=expected)
        if narrow_arena:
            plan = replace(plan, arena_bounds=ArenaBounds(length_m=4., width_m=4.))
        return replace(config, plan=plan, max_candidate_inspection_views=1)

    def effects(self, **overrides):
        values = dict(
            select_initial_preapproach=self.fixture._nearest_selection,
            read_current_pose=lambda: Pose2D(0., 0., 0.),
            plan_preapproach=Mock(return_value={"route_csv": "route.csv"}),
            run_motion_leg=Mock(side_effect=self.fixture._completed),
            capture_observation=Mock(), validate_facing=Mock(),
            commit_decision=Mock(), clock=lambda: 10.,
        )
        values.update(overrides)
        return CandidateApproachEffects(**values)

    def test_invalid_target_cannot_enter_unframed_arrival_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root, [("wall", 4.98, 0.)])
            effects = self.effects()
            with self.assertRaises(CandidateObservationUnavailableError) as caught:
                _admit_camera_arrival_geometry(
                    source_config=config, effects=effects, source_registry=None,
                    candidate_uid="wall", candidate_root=root / "arrival",
                    observation_attempt_index=0, allow_centering_acquisition=True,
                )
            self.assertEqual(caught.exception.reason, "candidate_target_ineligible")
            self.assertIn("target_static_map_incompatible", str(caught.exception.status_evidence))
            effects.capture_observation.assert_not_called()
            effects.run_motion_leg.assert_not_called()

    def test_fresh_arrival_rechecks_reprojected_target_before_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture._write_frame_registry(
                self.config(root, [("stand", 1., 0.)], narrow_arena=True),
                frozen_map_from_odom=PlanarTransform2D(0., 0., 0.),
            )
            effects = self.effects(admit_planning_frame=lambda _path: CandidatePlanningFrame(
                Pose2D(1.28, 0., 0.), PlanarTransform2D(.98, 0., 0.)))
            with self.assertRaises(CandidateObservationUnavailableError) as caught:
                _admit_camera_arrival_geometry(
                    source_config=config, effects=effects,
                    source_registry=load_stand_survey_registry(config.survey_root / "stand_registry.json"),
                    candidate_uid="stand", candidate_root=root / "arrival",
                    observation_attempt_index=0,
                )
            self.assertEqual(caught.exception.reason, "candidate_target_ineligible")
            self.assertIn("target_static_map_incompatible", str(caught.exception.status_evidence))
            effects.capture_observation.assert_not_called()
            effects.run_motion_leg.assert_not_called()
            self.assertEqual(config.snapshot.candidate_for("stand").geometry.x_m, 1.)

    def test_six_candidate_pool_resolves_five_valid_targets_and_retains_all_keepouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positions = [(f"stand_{i}", x, y) for i, (x, y) in enumerate(
                ((.8, 0.), (1.8, 0.), (-1., 0.), (0., 1.5), (0., -1.5)))]
            positions.append(("wall", 4.98, 0.))
            config = self.config(root, positions, expected=5)
            original_hash = candidate_snapshot_sha256(config.snapshot)
            all_uids = set(config.snapshot.candidate_uids)
            selections, plans, captures, events = [], [], [], []

            def select(request):
                selections.append(request)
                self.assertEqual(set(request.config.snapshot.candidate_uids), all_uids)
                return _select_initial_preapproach(request)

            def plan(request):
                plans.append(request)
                self.assertEqual(set(request.snapshot.candidate_uids), all_uids)
                self.assertNotEqual(request.candidate_uid, "wall")
                return {"route_csv": "route.csv"}

            def capture(request):
                uid = request.candidate.candidate_uid
                captures.append(uid)
                return CandidateObservation(request.output_dir / "recommendation.json", f"QR_{uid}", None)

            with patch.object(candidate_preapproach_selection, "compute_candidate_preapproach_plan",
                              wraps=candidate_preapproach_selection.compute_candidate_preapproach_plan) as preview:
                result = execute_candidate_approach_phase(config, self.effects(
                    select_initial_preapproach=select, plan_preapproach=plan,
                    capture_observation=capture,
                    validate_facing=lambda request: {"candidate_uid": request.candidate.candidate_uid},
                    event_sink=lambda _path, event: events.append(event),
                ))
            self.assertTrue(preview.called)
            self.assertNotIn("wall", [call.kwargs["candidate_uid"] for call in preview.call_args_list])
            self.assertEqual(result.stand_count, 5)
            self.assertEqual(set(captures), all_uids - {"wall"})
            self.assertEqual(len(plans), 5)
            self.assertEqual(len(selections), 5)
            self.assertEqual(candidate_snapshot_sha256(config.snapshot), original_hash)
            progress = json.loads((config.session_root / "candidate_goal_progress.json").read_text())
            self.assertTrue(progress["goal_completed"])
            self.assertEqual(set(progress["keepout_candidate_uids"]), all_uids)
            wall = next(row for row in progress["candidate_dispositions"] if row["candidate_uid"] == "wall")
            self.assertEqual(wall["disposition"], "target_reconciliation_required")
            target_evidence = [e["candidate_target_admission"] for e in events
                               if "candidate_target_admission" in e]
            self.assertTrue(target_evidence)
            self.assertTrue(any(e.get("candidate_uid") == "wall" and e["accepted"] is False
                                for e in target_evidence))

    def test_all_invalid_targets_leave_goal_incomplete_without_planning_or_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root, [("wall_a", 4.98, 0.), ("wall_b", -4.98, 0.)], expected=1)
            selector = Mock(wraps=_select_initial_preapproach)
            effects = self.effects(select_initial_preapproach=selector)
            with patch.object(candidate_preapproach_selection, "compute_candidate_preapproach_plan",
                              side_effect=AssertionError("ineligible target reached route preview")) as preview:
                with self.assertRaises(CandidateQrGoalIncompleteError):
                    execute_candidate_approach_phase(config, effects)
            selector.assert_called_once()
            preview.assert_not_called()
            effects.plan_preapproach.assert_not_called()
            effects.run_motion_leg.assert_not_called()
            effects.capture_observation.assert_not_called()
            effects.commit_decision.assert_not_called()
            progress = json.loads((config.session_root / "candidate_goal_progress.json").read_text())
            self.assertFalse(progress["goal_completed"])
            self.assertEqual(progress["confirmed_stand_count"], 0)
            self.assertEqual(set(progress["keepout_candidate_uids"]), {"wall_a", "wall_b"})
            self.assertEqual({r["disposition"] for r in progress["candidate_dispositions"]},
                             {"target_reconciliation_required"})
            self.assertFalse((config.session_root / "observed_stand_identities.json").exists())

    def test_post_centering_reprojection_cannot_bypass_target_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture._write_frame_registry(
                self.config(root, [("stand", 1., 0.)], narrow_arena=True),
                frozen_map_from_odom=PlanarTransform2D(0., 0., 0.),
            )
            frames = iter((
                CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(0., 0., 0.)),
                CandidatePlanningFrame(Pose2D(.30, 0., 0.), PlanarTransform2D(0., 0., 0.)),
                # The turned robot remains .7 m from the projected target. A
                # range-only post-turn check would allow a second capture.
                CandidatePlanningFrame(Pose2D(1.28, 0., 0.), PlanarTransform2D(.98, 0., 0.)),
            ))
            captures = []

            def capture(request):
                captures.append(request)
                return CandidateObservation(None, None, None,
                    centering_advisory_path=root / "injected_turn_advice.json")

            turn = Mock(return_value=SimpleNamespace(result_path=root / "turn_result.json",
                result={"actual_angular_travel_rad": .05, "stopped_at_sec": 12.}))
            effects = self.effects(
                admit_planning_frame=lambda _path: next(frames),
                read_current_pose=lambda: Pose2D(.3, 0., 0.),
                capture_observation=capture, run_centering_turn=turn,
            )
            with self.assertRaises(CandidateQrGoalIncompleteError):
                execute_candidate_approach_phase(config, effects)
            self.assertEqual(len(captures), 1)
            self.assertEqual(captures[0].candidate.geometry.x_m, 1.)
            self.assertEqual(turn.call_count, 1)
            self.assertEqual(effects.run_motion_leg.call_count, 1)
            effects.commit_decision.assert_not_called()
            progress = json.loads((config.session_root / "candidate_goal_progress.json").read_text())
            self.assertFalse(progress["goal_completed"])
            stand = progress["candidate_dispositions"][0]
            self.assertEqual(stand["disposition"], "target_reconciliation_required")
            self.assertIn("candidate_target_ineligible", json.dumps(stand))


if __name__ == "__main__":
    unittest.main()
