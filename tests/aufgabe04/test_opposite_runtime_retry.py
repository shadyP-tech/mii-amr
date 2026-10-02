"""Post-motion opposite-route recovery through the real motion dispatcher.

Only child execution, planning and fresh localization are effect fixtures.  The
runtime-stop validator, recovery handoff, projections, route rejection and
opposite-view coordinators all run as production code.
"""

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    BacksideAxisFrameProjection, load_backside_axis_planning_observation,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext, evaluate_map_odom_continuity,
)
from scripts.aufgabe04.real_robot.candidate import approach
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import CandidateInspectionRouteUnavailableError
from scripts.aufgabe04.real_robot.candidate.opposite_runtime_retry import OppositeRuntimeRouteRejected
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import CandidateRuntimeRecoveryError
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_autonomous_candidate_runtime_recovery import _runtime_stop


class OppositeRuntimeRetryTest(unittest.TestCase):
    @contextmanager
    def case(self, *, alternate="success", replacement_overrides=None,
             replacement_permit=False, runtime_exception=None):
        factory = fixtures.AutonomousCandidateApproachTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authorization = root / "mission_authorization.json"
            authorization.write_text("{}\n")
            source_config = replace(factory._config(root, (
                factory._candidate("candidate_a", 2.0, 0.0),)),
                approach_offset_m=.5, max_startup_reseals_per_leg=2,
                max_runtime_localization_reseals_per_leg=1,
                mission_motion_authorization_json=authorization,
                startup_reseal_motion_authorization_json=root / "startup_authorization.json")
            source_config = factory._write_frame_registry(source_config,
                frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0))
            registry = approach.load_stand_survey_registry(
                source_config.survey_root / "stand_registry.json", source_config.plan)
            observer_frame = CandidatePlanningFrame(Pose2D(1.0, .7, 0.0),
                                                    PlanarTransform2D(0., 0., 0.))
            observer_projection = approach._materialize_candidate_frame_projection(
                source_config=source_config, source_registry=registry,
                planning_frame=observer_frame, output_root=root / "observer_frame")
            frame = approach._CandidateObservationFrame(
                config=observer_projection.config,
                candidate=observer_projection.config.snapshot.candidate_for("candidate_a"),
                planning_frame=observer_frame,
                decision_binding=observer_projection.camera_decision_binding())
            source_axis = root / "original_observer_axis.json"
            source_axis.write_text(json.dumps(backside_axis_payload(
                stand_id="candidate_a", stand_x_m=1., robot_x_m=1., robot_y_m=.7)))
            original_axis_bytes = source_axis.read_bytes()
            planning_frames = (
                CandidatePlanningFrame(Pose2D(.30, .20, 0.), PlanarTransform2D(0., .20, 0.)),
                CandidatePlanningFrame(Pose2D(.40, .20, .08), PlanarTransform2D(.05, .20, .08)),
                CandidatePlanningFrame(Pose2D(.50, .30, .12), PlanarTransform2D(.10, .30, .12)),
            )
            if alternate.startswith("checkpoint"):
                planning_frames += (CandidatePlanningFrame(Pose2D(.6, .3, .14),
                                                            PlanarTransform2D(.10, .30, .14)),)
            admissions, plans, children, replacements, events, execution_configs = [], [], [], [], [], []

            def admit(path):
                index = len(admissions)
                self.assertLess(index, len(planning_frames), "recovery restarted its localization budget")
                admissions.append(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
                return planning_frames[index]

            def plan(request):
                plans.append(request)
                request.output_dir.mkdir(parents=True, exist_ok=True)
                return {"route_csv": str(request.output_dir / "route.csv")}

            def stopped(request):
                outcome = _runtime_stop(root, request.run_id,
                    continuity_reason="map_from_odom_yaw_drift")
                context = OdomExecutionContext(map_frame="map", odom_frame="odom",
                    base_frame="base_footprint", frozen_map_from_odom=PlanarTransform2D(0., 0., 0.),
                    certificate_sha256="a" * 64, max_map_from_odom_translation_drift_m=.1,
                    max_map_from_odom_yaw_drift_rad=math.radians(3.120))
                continuity = evaluate_map_odom_continuity(context,
                    PlanarTransform2D(0., 0., math.radians(3.232)))
                self.assertFalse(continuity.accepted)
                details = {**outcome.stop_details, "continuity": continuity.to_evidence()}
                permit = outcome.startup_reseal_motion_permit_path
                return replace(outcome, stop_details=details,
                    startup_reseal_motion_permit_path=None, startup_reseal_motion_permit_sha256="",
                    mission_leg_motion_permit_path=permit,
                    mission_leg_motion_permit_sha256=sha256(permit.read_bytes()).hexdigest())

            def rejected(request, *, overrides=None, permit=False):
                reason = ("odom execution admission failed: route uncertainty budget exhausted: "
                          "limiting_segment=segment:0002:0092 remaining_margin=-0.003759 m")
                return factory._route_uncertainty_rejection(request, stop_reason=reason,
                    stop_detail_overrides={"route_uncertainty_remaining_margin_m": -.003759,
                                           **(overrides or {})},
                    report_mission_leg_permit=permit)

            def run(request):
                children.append(request)
                if "_post_motion_001_" not in request.run_id:
                    return stopped(request)
                if alternate == "stop":
                    return stopped(request)
                if alternate.startswith("checkpoint"):
                    if "_after_checkpoint_" in request.run_id and alternate == "checkpoint_stop":
                        return stopped(request)
                    if request.run_id.endswith("_checkpoint_001") or "_after_checkpoint_" in request.run_id:
                        return factory._completed(request)
                    return rejected(request)
                if alternate == "exhausted" or (alternate == "standoff" and
                                                  not request.run_id.endswith("_standoff_001")):
                    return rejected(request)
                return factory._completed(request)

            def runtime(request, attempt):
                replacements.append((request, attempt))
                if runtime_exception is not None:
                    raise runtime_exception
                return rejected(request, overrides=replacement_overrides, permit=replacement_permit)

            effects = approach.CandidateApproachEffects(
                read_current_pose=lambda: planning_frames[-1].current_pose,
                admit_planning_frame=admit, plan_preapproach=plan, run_motion_leg=run,
                admit_runtime_localization=Mock(side_effect=AssertionError("planning frame owns admission")),
                run_runtime_localization_reseal_motion_leg=runtime,
                run_startup_reseal_motion_leg=Mock(side_effect=AssertionError("startup budget restarted")),
                capture_observation=Mock(), event_sink=lambda _, event: events.append(event),
                clock=lambda: 10.)
            real_execute = approach._execute_candidate_motion

            def execute(**kwargs):
                execution_configs.append(kwargs["config"])
                return real_execute(**kwargs)

            arrived = object()
            with patch.object(approach, "_admit_camera_arrival_geometry", return_value=arrived) as arrival, \
                 patch.object(approach, "_execute_candidate_motion", side_effect=execute), \
                 patch.object(approach, "try_opposite_checkpoint", return_value=None) as checkpoint:
                def invoke(*, observed_frame=frame):
                    return approach._move_certified_opposite_face(observation_frame=observed_frame,
                        observation=approach.CandidateObservation(None, None, source_axis),
                        source_config=source_config, effects=effects, source_registry=registry,
                        candidate_root=root / "opposite", candidate_run_id="mission_inspect",
                        candidate_index=0)
                yield SimpleNamespace(root=root, invoke=invoke, source_axis=source_axis,
                    original_axis_bytes=original_axis_bytes, plans=plans, children=children,
                    replacements=replacements, admissions=admissions, events=events,
                    configs=execution_configs, arrived=arrived, arrival=arrival,
                    checkpoint=checkpoint, planning_frames=planning_frames, effects=effects,
                    frame=frame)

    def test_recorded_stop_and_negative_margin_replans_from_fresh_frame(self):
        with self.case() as case:
            self.assertIs(case.invoke(), case.arrived)
            self.assertEqual(len(case.children), 2)
            self.assertEqual(len(case.replacements), 1)
            self.assertEqual(len(case.admissions), 3)
            self.assertEqual(case.admissions[-1], case.root / "opposite/post_motion_001/opposite_face_planning_localization.json")
            self.assertEqual(case.children[-1].run_id, "mission_inspect_post_motion_001_opposite")
            self.assertEqual(case.plans[-1].start, case.planning_frames[-1].current_pose)
            self.assertEqual(case.source_axis.read_bytes(), case.original_axis_bytes)
            self.assertEqual(case.configs[-1].max_startup_reseals_per_leg, 0)
            self.assertEqual(case.configs[-1].max_runtime_localization_reseals_per_leg, 0)
            for request in case.plans:
                axis = load_backside_axis_planning_observation(request.axis_observation_path)
                candidate = request.snapshot.candidates[0]
                self.assertAlmostEqual(axis.stand_x_m, candidate.geometry.x_m)
                self.assertAlmostEqual(axis.stand_y_m, candidate.geometry.y_m)
                self.assertAlmostEqual(axis.opposite_face_normal_rad, request.approach_normal_rad)
                self.assertEqual(axis.source_axis_observation_path.resolve(), case.source_axis.resolve())
            final_axis = load_backside_axis_planning_observation(case.plans[-1].axis_observation_path)
            self.assertAlmostEqual(final_axis.stand_axis_rad, .12)
            all_children = [*case.children, case.replacements[0][0]]
            self.assertEqual(len({c.run_id for c in all_children}), 3)
            self.assertEqual(len({c.permit_json_path for c in all_children}), 3)
            self.assertEqual(len({p.output_dir for p in case.plans}), 3)
            self.assertFalse(any(e["event"] == "opposite_localization_refresh_requested" for e in case.events))
            self.assertEqual(case.arrival.call_args.kwargs["candidate_root"], case.root / "opposite/post_motion_001")
            case.effects.capture_observation.assert_not_called()

    def test_alternate_standoff_retains_existing_limits_and_distinct_identity(self):
        with self.case(alternate="standoff") as case:
            self.assertIs(case.invoke(), case.arrived)
            self.assertEqual(len(case.children), 3)
            self.assertEqual([p.approach_offset_m for p in case.plans], [.5, .5, .5, .45])
            self.assertEqual(case.children[-1].run_id, "mission_inspect_post_motion_001_opposite_standoff_001")
            self.assertTrue(all(c.max_startup_reseals_per_leg == 0 and
                                c.max_runtime_localization_reseals_per_leg == 0
                                for c in case.configs[1:]))
            self.assertTrue(all(p.physical_clearance == case.plans[0].physical_clearance for p in case.plans))
            self.assertTrue(all(p.candidate_transit_radius_m == case.plans[0].candidate_transit_radius_m
                                for p in case.plans))

    def test_exhausted_alternatives_are_terminal_after_motion(self):
        with self.case(alternate="exhausted") as case:
            with self.assertRaises(CandidateRuntimeRecoveryError) as caught:
                case.invoke()
            self.assertEqual(caught.exception.phase, "opposite_routes_exhausted")
            self.assertNotIsInstance(caught.exception, CandidateInspectionRouteUnavailableError)
            self.assertEqual(len(case.admissions), 3)
            self.assertEqual(len(case.replacements), 1)
            self.assertFalse(any(e["event"] == "opposite_localization_refresh_requested" for e in case.events))
            self.assertFalse(any("post_motion_002" in child.run_id for child in case.children))
            case.arrival.assert_not_called()

    def test_post_motion_checkpoint_suffix_cannot_restart_recovery(self):
        for alternate in ("checkpoint", "checkpoint_stop"):
            with self.subTest(alternate=alternate), self.case(alternate=alternate) as case:
                self.exercise_checkpoint_suffix(case, should_complete=alternate == "checkpoint")

    def exercise_checkpoint_suffix(self, case, *, should_complete):
        def checkpoint(**kwargs):
            request, sealed, run_id = kwargs["rejected_routes"][0]
            outcome = kwargs["execute_prefix"](request, sealed, run_id + "_checkpoint_001")
            self.assertEqual(outcome.status, "completed")
            return kwargs["continue_from_checkpoint"]()
        case.checkpoint.side_effect = checkpoint
        real_epoch = approach._move_certified_opposite_face_epoch
        with patch.object(BacksideAxisFrameProjection, "validated_target_center",
                          new_callable=PropertyMock,
                          return_value={"x_m": 1., "y_m": .3, "uncertainty_m": .024}), \
             patch.object(approach, "_move_certified_opposite_face_epoch", wraps=real_epoch) as epochs:
            if should_complete:
                self.assertIs(case.invoke(), case.arrived)
            else:
                with self.assertRaises(RuntimeError):
                    case.invoke()
                case.arrival.assert_not_called()
        case.checkpoint.assert_called_once()
        self.assertEqual(len(case.admissions), 4)
        self.assertEqual(len(case.replacements), 1)
        self.assertTrue(all(c.max_startup_reseals_per_leg == 0 and
                            c.max_runtime_localization_reseals_per_leg == 0
                            for c in case.configs[1:]))
        suffix = epochs.call_args_list[-1].kwargs
        self.assertEqual(suffix["candidate_root"], case.root / "opposite/post_motion_001/after_checkpoint")
        self.assertFalse(suffix["allow_checkpoint"])
        self.assertFalse(suffix["allow_runtime_route_retry"])
        self.assertEqual(suffix["source_config"].max_startup_reseals_per_leg, 0)
        self.assertEqual(suffix["source_config"].max_runtime_localization_reseals_per_leg, 0)
        self.assertEqual(suffix["observation"].axis_observation_path, case.source_axis)
        final_axis = load_backside_axis_planning_observation(case.plans[-1].axis_observation_path)
        self.assertAlmostEqual(final_axis.stand_axis_rad, .14)

    def test_existing_post_motion_epoch_cannot_be_reused(self):
        with self.case() as case:
            (case.root / "opposite/post_motion_001").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                case.invoke()
            self.assertEqual(len(case.children), 1)
            self.assertEqual(len(case.admissions), 2)
            case.arrival.assert_not_called()

    def test_unbound_observer_cannot_use_an_otherwise_valid_runtime_rejection(self):
        with self.case() as case:
            # Obtain the marker through the production dispatcher and stop the
            # outer coordinator before it can run the fresh epoch.
            with patch.object(approach, "with_opposite_runtime_retry",
                              side_effect=lambda **kwargs: kwargs["attempt"]()):
                with self.assertRaises(OppositeRuntimeRouteRejected) as caught:
                    case.invoke()
            for missing in ({"planning_frame": None}, {"decision_binding": None}):
                with self.subTest(missing=missing), \
                     patch.object(approach, "_move_certified_opposite_face_epoch",
                                  side_effect=caught.exception) as epoch:
                    with self.assertRaises(CandidateRuntimeRecoveryError) as terminal:
                        case.invoke(observed_frame=replace(case.frame, **missing))
                    self.assertIs(terminal.exception, caught.exception.error)
                    epoch.assert_called_once()
            self.assertEqual(len(case.admissions), 2)
            case.arrival.assert_not_called()

    def test_second_motion_stop_cannot_reset_runtime_or_startup_budget(self):
        with self.case(alternate="stop") as case:
            with self.assertRaises(RuntimeError):
                case.invoke()
            self.assertEqual(len(case.children), 2)
            self.assertEqual(len(case.replacements), 1)
            self.assertEqual(len(case.admissions), 3)
            case.effects.run_startup_reseal_motion_leg.assert_not_called()
            case.arrival.assert_not_called()

    def test_malformed_or_non_uncertainty_replacement_never_starts_an_epoch(self):
        for overrides, permit in [({"route_uncertainty_remaining_margin_m": 0.}, False),
                                  ({"route_uncertainty_remaining_margin_m": float("nan")}, False),
                                  ({"motion_published": True}, False), ({}, True)]:
            with self.subTest(overrides=overrides, permit=permit), \
                 self.case(replacement_overrides=overrides, replacement_permit=permit) as case:
                with self.assertRaises(RuntimeError):
                    case.invoke()
                self.assertEqual(len(case.children), 1)
                self.assertEqual(len(case.admissions), 2)
                case.arrival.assert_not_called()

    def test_untyped_child_failure_does_not_request_a_new_epoch(self):
        failure = RuntimeError("unstructured route failure")
        with self.case(runtime_exception=failure) as case:
            with self.assertRaises(RuntimeError):
                case.invoke()
            self.assertEqual(len(case.children), 1)
            self.assertEqual(len(case.admissions), 2)
            case.arrival.assert_not_called()


if __name__ == "__main__":
    unittest.main()
