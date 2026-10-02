"""Survey approach and local target refinement through real orchestration.

Only the sensor support capture/replay boundary is substituted. Real frame
projection, registry preservation, selection handoff, arrival and both recovery
coordinators run against temporary artifacts.
"""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_inspection_view import (
    load_candidate_inspection_view, write_candidate_inspection_view,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import load_stand_survey_registry
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.candidate import approach
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import HASH_FIELD
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import (
    assess_current_lidar_targets, capture_current_lidar_targets,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import capture_candidate_lidar_view
from scripts.aufgabe04.real_robot.candidate.lidar_head_capture import (
    HASH_FIELD as CAPTURE_HASH_FIELD, head_capture_payload,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import CandidateQrGoalIncompleteError
from scripts.aufgabe04.real_robot.candidate.recovery_failure import CandidateStartupRecoveryError
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import CandidateRuntimeRecoveryError
from scripts.aufgabe04.real_robot.candidate.target_admission import (
    SURVEY_OBSERVATION_POLICY, load_survey_observation_support,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_autonomous_candidate_runtime_recovery import _runtime_stop, _outcome
from tests.aufgabe04.test_lidar_alignment_arrival import calibration
from tests.aufgabe04.test_current_lidar_targets import fixture as lidar_target_fixture, raw_scans


MODULE = "scripts.aufgabe04.real_robot.candidate.approach"
SENSOR = "scripts.aufgabe04.real_robot.candidate.current_lidar_targets"


class CurrentLidarApproachIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixtures = fixtures.AutonomousCandidateApproachTest()
        self.events = []

    def config(self, *, count=1):
        candidates = tuple(self.fixtures._candidate(f"candidate_{i}", 1.0 + i, 0.0)
                           for i in range(count))
        config = self.fixtures._write_frame_registry(
            self.fixtures._config(self.root, candidates),
            frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0))
        return replace(config, require_current_lidar_support=True,
            camera_calibration=calibration(), lidar_scan_frame="laser", lidar_scan_topic="/scan",
            measured_stand_model=load_measured_physical_stand_model(Path(__file__).parents[2] /
                "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"))

    def frame(self, x=.3):
        return CandidatePlanningFrame(Pose2D(x, 0.0, 0.0), PlanarTransform2D(0.0, 0.0, 0.0))

    def effects(self, **kwargs):
        values = dict(read_current_pose=lambda: Pose2D(.3, 0.0, 0.0),
            admit_planning_frame=lambda path: self.frame(),
            select_initial_preapproach=Mock(), plan_preapproach=Mock(),
            run_motion_leg=Mock(), capture_observation=Mock(),
            capture_lidar_view=Mock(),
            event_sink=lambda path, event: self.events.append(event), clock=lambda: 10.0)
        values.update(kwargs)
        return approach.CandidateApproachEffects(**values)

    def registry(self, config):
        return load_stand_survey_registry(config.survey_root / "stand_registry.json", config.plan)

    def support(self, config, frame, candidate_uids, root, estimates):
        """Persist the support boundary's decision so view binding stays real."""
        path = Path(root) / "current_lidar_targets.json"
        evidence = {
            "eligible_candidate_uids": sorted(estimates),
            "excluded_candidate_uids": sorted(set(candidate_uids)-set(estimates)),
            "candidate_decisions": {uid: {"candidate_uid": uid, "accepted": uid in estimates,
                "reasons": [] if uid in estimates else ["insufficient_current_lidar_support"]}
                for uid in candidate_uids},
            "planning_frame": frame.to_evidence(), "motion_authorized": False,
            "keepouts_changed": False,
        }
        digest = write_content_hashed_json(path, evidence, hash_field=HASH_FIELD)
        return estimates, {**evidence, "evidence_path": str(path), "evidence_sha256": digest}

    @staticmethod
    def target(x=1.06):
        return {"x_m": x, "y_m": 0.0, "uncertainty_m": .08,
                "policy": "current_stopped_lidar_surface"}

    def real_support(self, config, planning_frame, output_dir, *, shift=.12):
        """Produce and replay the actual accepted eight-scan support artifact."""
        transform = planning_frame.map_from_odom
        self.assertEqual(transform.yaw_rad, 0.)
        base_x = planning_frame.current_pose.x_m-transform.x_m
        base_y = planning_frame.current_pose.y_m-transform.y_m
        scan_x = base_x+.04
        target_x = config.snapshot.candidates[0].geometry.x_m-transform.x_m+shift
        def capture(request):
            scans = raw_scans(shifts=[target_x-scan_x-1.]*8)
            for scan in scans:
                scan["scan_pose_odom"] = {"x_m": scan_x, "y_m": base_y, "yaw_rad": 0.}
                scan["base_pose_odom"] = {"x_m": base_x, "y_m": base_y, "yaw_rad": 0.}
            payload = head_capture_payload(scans, tour_id=request.viewpoint_id,
                odom_frame="odom", base_frame=request.base_frame, scan_frame=request.scan_frame,
                captured_at_unix_sec=100.71)
            path = request.output_dir / "scan_cohort.json"
            write_content_hashed_json(path, payload, hash_field=CAPTURE_HASH_FIELD)
            clock = iter((100., 100.72))
            return capture_candidate_lidar_view(request, capture_cohort=lambda _: path,
                                                clock=lambda: next(clock))
        times = iter((100., 100.72))
        effects = self.effects(capture_lidar_view=capture, clock=lambda: next(times))
        return capture_current_lidar_targets(config, effects, planning_frame, {"candidate_0"}, output_dir)

    def retained_source(self, *, name="source", shift=.12):
        from scripts.aufgabe04.real_robot.candidate.target_admission import bind_current_lidar_target
        config = self.config()
        config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
        planning_frame = self.frame(-.04)
        artifacts = approach._materialize_candidate_frame_projection(source_config=config,
            source_registry=self.registry(config), planning_frame=planning_frame,
            output_root=self.root / name / "projection")
        estimates, evidence = self.real_support(artifacts.config, planning_frame,
                                                self.root / name / "support", shift=shift)
        frame = approach._CandidateObservationFrame(artifacts.config, artifacts.config.snapshot.candidates[0],
            planning_frame, artifacts.camera_decision_binding(), observation_pose=planning_frame.current_pose)
        return config, bind_current_lidar_target(frame, evidence_path=Path(evidence["evidence_path"])), evidence

    def test_survey_selection_keeps_all_candidates_without_distant_lidar_gating(self):
        config = self.config(count=2)
        order, requests = [], []
        registry_path = config.survey_root / "stand_registry.json"
        original_registry, original_snapshot = registry_path.read_bytes(), config.snapshot_path.read_bytes()
        def select(request):
            order.append("select")
            requests.append(request)
            return approach.CameraCandidateInitialSelection("candidate_1",
                SimpleNamespace(candidate_uid="candidate_1", validated_target_center=None,
                                camera_alignment=None, approach_bearing_mode="robot-to-stand"),
                {"selected_candidate_uid": "candidate_1", "motion_authorized": False})
        effects = self.effects(select_initial_preapproach=select)
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError(
                "initial survey selection must not require visible head returns")) as capture:
            planned_config, _, selection, planning = approach._select_initial_candidate_with_localization_refresh(
                config=config, effects=effects, source_registry=self.registry(config), candidate_index=0,
                eligible=set(config.snapshot.candidate_uids), exact_two_support_by_uid=None)
        capture.assert_not_called()
        self.assertEqual(order, ["select"])
        self.assertEqual(requests[0].unresolved, frozenset(config.snapshot.candidate_uids))
        self.assertEqual(requests[0].current_target_estimates, {})
        self.assertEqual(requests[0].config.snapshot.candidate_uids, config.snapshot.candidate_uids)
        self.assertEqual(requests[0].config.snapshot.candidate_for("candidate_0").geometry,
                         config.snapshot.candidate_for("candidate_0").geometry)
        self.assertNotIn("current_lidar_support", selection.evidence)
        self.assertEqual(selection.evidence["observation_target_source"], SURVEY_OBSERVATION_POLICY)
        proof = load_survey_observation_support(Path(selection.evidence["survey_observation_support_path"]),
            candidate_uid="candidate_1", snapshot=planned_config.snapshot)
        self.assertEqual(proof["planning_frame"], planning.to_evidence())
        self.assertFalse(proof["precise_motion_authorized"])
        self.assertEqual(registry_path.read_bytes(), original_registry)
        self.assertEqual(config.snapshot_path.read_bytes(), original_snapshot)
        effects.run_motion_leg.assert_not_called()
        effects.capture_observation.assert_not_called()

    def test_survey_selection_cannot_claim_a_precise_target_or_alignment(self):
        config = self.config()
        survey = SimpleNamespace(candidate_uid="candidate_0", validated_target_center=None,
                                 camera_alignment=None, approach_bearing_mode="robot-to-stand")
        for index, prepared in enumerate((None,
                SimpleNamespace(**{**vars(survey), "validated_target_center": self.target()}),
                SimpleNamespace(**{**vars(survey), "camera_alignment": {"untrusted": True}}),
                SimpleNamespace(**{**vars(survey), "approach_bearing_mode": "candidate-inspection-view"}))):
            with self.subTest(case=index):
                current = replace(config, session_root=self.root / f"bad_selection_{index}")
                effects = self.effects(select_initial_preapproach=Mock(return_value=
                    approach.CameraCandidateInitialSelection("candidate_0", prepared,
                        {"selected_candidate_uid": "candidate_0", "motion_authorized": False})))
                with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("selection rescan")):
                    with self.assertRaises(RuntimeError):
                        approach._select_initial_candidate_with_localization_refresh(
                            config=current, effects=effects, source_registry=self.registry(current), candidate_index=0,
                            eligible={"candidate_0"}, exact_two_support_by_uid=None)
                effects.plan_preapproach.assert_not_called()
                effects.run_motion_leg.assert_not_called()
                effects.capture_observation.assert_not_called()

    def test_default_selector_passes_current_centers_to_route_selection(self):
        config = self.config()
        estimate = self.target()
        request = approach.CameraCandidateSelectionRequest(config, Pose2D(.3, 0., 0.),
            frozenset({"candidate_0"}), None, current_target_estimates={"candidate_0": estimate})
        planned = SimpleNamespace(selected_candidate_uid="candidate_0", selected_plan=None,
            to_evidence=lambda: {"selected_candidate_uid": "candidate_0", "motion_authorized": False})
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_inspection_hints.load_camera_lidar_hints",
                   return_value=({}, {})), patch(MODULE + ".plan_and_select_camera_candidate", return_value=planned) as planner:
            approach._select_initial_preapproach(request)
        self.assertEqual(planner.call_args.kwargs["current_target_estimates"], {"candidate_0": estimate})

    def test_enabled_gate_fails_closed_when_sensor_or_frame_dependencies_are_missing(self):
        config, effects = self.config(), self.effects()
        cases = (
            ("admit_planning_frame", config, replace(effects, admit_planning_frame=None)),
            ("capture_lidar_view", config, replace(effects, capture_lidar_view=None)),
            ("camera_calibration", replace(config, camera_calibration=None), effects),
            ("measured_stand_model", replace(config, measured_stand_model=None), effects),
            ("lidar_scan_frame", replace(config, lidar_scan_frame=None), effects),
            ("lidar_scan_topic", replace(config, lidar_scan_topic=None), effects),
        )
        for missing, current, current_effects in cases:
            with self.subTest(missing=missing), patch(SENSOR + ".capture_current_lidar_targets") as capture:
                with self.assertRaisesRegex(RuntimeError, "current LiDAR target validation dependencies"):
                    approach.execute_candidate_approach_phase(current, current_effects)
                capture.assert_not_called()
                current_effects.select_initial_preapproach.assert_not_called()
                current_effects.plan_preapproach.assert_not_called()
                current_effects.run_motion_leg.assert_not_called()
                current_effects.capture_observation.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "current LiDAR arrival requires a fresh planning frame"):
            approach._admit_camera_arrival_geometry(source_config=config,
                effects=replace(effects, admit_planning_frame=None), source_registry=self.registry(config),
                candidate_uid="candidate_0", candidate_root=self.root / "missing_frame", observation_attempt_index=0)

    def test_camera_arrival_retains_accepted_target_without_another_lidar_vote(self):
        config, frame, support = self.retained_source()
        order = []
        effects = self.effects(capture_observation=Mock(side_effect=lambda request: (
            order.append("camera") or approach.CandidateObservation(
                request.output_dir / "recommendation.json", "QR_A", None))))
        # The run's second cohort was 6/8 supported with an ambiguity veto.
        # It must not be requested just to admit the already reached view.
        estimates, ambiguous = assess_current_lidar_targets(**lidar_target_fixture(
            kinds=["stand"]*6+["ambiguous"]*2, shifts=[.12]*8))
        self.assertFalse(estimates)
        rejection = ambiguous["candidate_decisions"]["candidate_1"]
        self.assertEqual(rejection["supported_scan_count"], 6)
        self.assertIn("ambiguous_current_target_correspondence", rejection["reasons"])
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError(
                "arrival must not reacquire the preapproach target")) as recapture:
            _, admitted = approach._capture_candidate_camera_result(
                observation_frame=frame, source_config=config, effects=effects,
                source_registry=self.registry(config), candidate_root=self.root / "arrival",
                candidate_run_id="candidate_arrival", candidate_index=0)
        recapture.assert_not_called()
        self.assertEqual(order, ["camera"])
        self.assertEqual(support["candidate_decisions"]["candidate_0"]["supported_scan_count"], 8)
        self.assertGreater(frame.camera_target_geometry.x_m-frame.candidate.geometry.x_m, .08)
        self.assertAlmostEqual(admitted.camera_target_geometry.x_m, 1.12)
        self.assertAlmostEqual(admitted.candidate.geometry.x_m, 1.0)
        self.assertEqual(admitted.current_lidar_target_path, Path(support["evidence_path"]))
        effects.run_motion_leg.assert_not_called()

    def test_full_phase_selects_and_moves_before_local_refinement_and_camera(self):
        config = self.config()
        config = replace(config, camera_calibration=replace(config.camera_calibration, base_frame="base_footprint"))
        frames = iter((self.frame(-.04), self.frame(.3)))
        order, support_records = [], []
        def capture(cfg, effects, planning_frame, uids, output):
            order.append("scan")
            self.assertEqual(order, ["select", "motion", "scan"])
            self.assertEqual(planning_frame.current_pose, self.frame(.3).current_pose)
            self.assertFalse(support_records, "local refinement must use one cohort")
            estimates, evidence = self.real_support(cfg, planning_frame, output)
            support_records.append(evidence)
            return estimates, evidence
        def select(request):
            order.append("select")
            self.assertEqual(request.current_target_estimates, {})
            plan = SimpleNamespace(candidate_uid="candidate_0", validated_target_center=None,
                approach_bearing_mode="robot-to-stand", approach_bearing_rad=0., camera_alignment=None)
            return approach.CameraCandidateInitialSelection("candidate_0", plan,
                {"selected_candidate_uid": "candidate_0", "motion_authorized": False})
        def motion(request):
            order.append("motion")
            return self.fixtures._completed(request)
        def camera(request):
            order.append("camera")
            return approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
        effects = self.effects(admit_planning_frame=lambda path: next(frames),
            select_initial_preapproach=select, plan_preapproach=lambda request: {"route_csv": "route.csv"},
            run_motion_leg=motion, capture_observation=camera,
            validate_facing=lambda request: {"candidate_uid": request.candidate.candidate_uid},
            commit_decision=lambda request: None)
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture) as scans:
            result = approach.execute_candidate_approach_phase(config, effects)
        self.assertEqual(result.stand_count, 1)
        self.assertEqual(order, ["select", "motion", "scan", "camera"])
        scans.assert_called_once()
        self.assertEqual(support_records[0]["candidate_decisions"]["candidate_0"]["supported_scan_count"], 8)

    def test_retained_target_survives_map_frame_refresh_without_compounding(self):
        config, source, _ = self.retained_source()
        current = source
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("no translation planned")):
            for serial, angle in enumerate((.4, -.25, .4)):
                transform = PlanarTransform2D(.2, -.1, angle)
                pose = Pose2D(.2+.3*math.cos(angle), -.1+.3*math.sin(angle), angle)
                effects = self.effects(admit_planning_frame=lambda path, p=pose, t=transform:
                    CandidatePlanningFrame(p, t))
                current = approach._admit_camera_arrival_geometry(source_config=config, effects=effects,
                    source_registry=self.registry(config), candidate_uid="candidate_0",
                    candidate_root=self.root / f"refresh_{serial}", observation_attempt_index=0,
                    target_source_frame=current)
                self.assertAlmostEqual(current.camera_target_geometry.x_m, .2+1.12*math.cos(angle))
                self.assertAlmostEqual(current.camera_target_geometry.y_m, -.1+1.12*math.sin(angle))
                self.assertAlmostEqual(current.camera_target_geometry.uncertainty_m, .08)
                self.assertEqual(current.retained_lidar_target, source.retained_lidar_target)
                effects.capture_observation.assert_not_called()
                effects.run_motion_leg.assert_not_called()

    def test_camera_centering_turn_refreshes_pose_without_reacquiring_lidar(self):
        config, source, _ = self.retained_source()
        frames = iter((self.frame(), CandidatePlanningFrame(
            Pose2D(.3, 0., .02), PlanarTransform2D(0., 0., 0.))))
        order = []
        def camera(request):
            order.append("camera")
            if order == ["camera"]:
                return approach.CandidateObservation(None, None, None,
                    centering_advisory_path=self.root / "injected_advisory.json")
            self.assertEqual(request.observation_not_before_sec, 100.8)
            return approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
        def turn(**kwargs):
            order.append("centering_turn")
            return SimpleNamespace(result_path=self.root / "injected_turn.json",
                result={"actual_angular_travel_rad": .02, "stopped_at_sec": 100.8})
        effects = self.effects(admit_planning_frame=lambda path: next(frames),
            capture_observation=camera, run_centering_turn=turn)
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("yaw-only turn must not rescan")) as scans:
            _, arrived = approach._capture_candidate_camera_result(
                observation_frame=source, source_config=config, effects=effects,
                source_registry=self.registry(config), candidate_root=self.root / "centered_arrival",
                candidate_run_id="candidate_centering", candidate_index=0)
        self.assertEqual(order, ["camera", "centering_turn", "camera"])
        self.assertAlmostEqual(arrived.planning_frame.current_pose.yaw_rad, .02)
        self.assertAlmostEqual(arrived.camera_target_geometry.x_m, 1.12)
        self.assertEqual(arrived.retained_lidar_target, source.retained_lidar_target)
        scans.assert_not_called()
        effects.run_motion_leg.assert_not_called()

    def test_retaining_target_does_not_bypass_range_frame_or_map_admission(self):
        for fault in ("range", "frame", "map", "proof"):
            with self.subTest(fault=fault):
                config, frame, support = self.retained_source(name=f"source_{fault}")
                effects = self.effects(admit_planning_frame=(lambda path: None) if fault == "frame"
                    else (lambda path: self.frame(-.4 if fault == "range" else .3)))
                if fault == "map":
                    original_map = config.map_yaml.read_text()
                    config.map_yaml.write_text(original_map+"\n# changed bound map\n")
                elif fault == "proof":
                    path = Path(support["evidence_path"])
                    payload = json.loads(path.read_text())
                    payload["assessed_at_unix_sec"] += .1
                    path.write_text(json.dumps(payload))
                with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("no rescan fallback")):
                    with self.assertRaises((CandidateObservationUnavailableError, TypeError, ValueError)):
                        approach._capture_candidate_camera_result(observation_frame=frame, source_config=config,
                            effects=effects, source_registry=self.registry(config),
                            candidate_root=self.root / f"rejected_{fault}", candidate_run_id="candidate_arrival",
                            candidate_index=0)
                effects.capture_observation.assert_not_called()
                effects.run_motion_leg.assert_not_called()
                if fault == "map":
                    config.map_yaml.write_text(original_map)

    def test_required_passive_arrival_cannot_invent_missing_preapproach_proof(self):
        config, effects = self.config(), self.effects()
        source = approach._CandidateObservationFrame(config, config.snapshot.candidates[0],
            self.frame(), None, observation_pose=self.frame().current_pose)
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("missing proof cannot rescan")) as recapture:
            with self.assertRaises((CandidateObservationUnavailableError, RuntimeError, ValueError)):
                approach._capture_candidate_camera_result(observation_frame=source, source_config=config,
                    effects=effects, source_registry=self.registry(config), candidate_root=self.root / "missing_proof",
                    candidate_run_id="candidate_missing_proof", candidate_index=0)
        recapture.assert_not_called()
        effects.capture_observation.assert_not_called()
        effects.run_motion_leg.assert_not_called()

    def test_startup_and_runtime_reseals_reacquire_before_planning_replacement(self):
        for owner in ("startup", "runtime"):
            for supported in (True, False):
                with self.subTest(owner=owner, supported=supported):
                    self._check_reseal(owner, supported)

    def test_startup_then_runtime_replacements_emit_only_the_completed_target_frame(self):
        self._check_reseal("startup_runtime", True)

    def test_inspection_reseals_check_retained_axis_against_the_fresh_target_before_planning(self):
        for owner in ("startup", "runtime"):
            for compatible in (True, False):
                with self.subTest(owner=owner, compatible=compatible):
                    self._check_reseal(owner, True, retained_axis_compatible=compatible)

    def test_accepted_preapproach_support_reaches_camera_then_opposite_without_arrival_rescan(self):
        config, frame, support = self.retained_source()
        order = []
        def camera(request):
            order.append("camera")
            return approach.CandidateObservation(None, None, self.root / "camera_axis.json")
        class OppositeDispatchReached(RuntimeError):
            pass
        def opposite(**kwargs):
            order.append("opposite")
            arrived = kwargs["observation_frame"]
            self.assertEqual(arrived.current_lidar_target_path, Path(support["evidence_path"]))
            self.assertAlmostEqual(arrived.camera_target_geometry.x_m, 1.12)
            raise OppositeDispatchReached()
        effects = self.effects(capture_observation=Mock(side_effect=camera))
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=AssertionError("arrival rescan")) as recapture, \
                patch(MODULE + "._move_certified_opposite_face", side_effect=opposite):
            with self.assertRaises(OppositeDispatchReached):
                approach._capture_candidate_camera_result(observation_frame=frame, source_config=config,
                    effects=effects, source_registry=self.registry(config), candidate_root=self.root / "opposite_arrival",
                    candidate_run_id="candidate_opposite", candidate_index=0)
        self.assertEqual(order, ["camera", "opposite"])
        recapture.assert_not_called()
        effects.run_motion_leg.assert_not_called()

    def test_typed_startup_target_deferral_preserves_goal_and_visits_next_candidate(self):
        self._check_startup_target_deferral()

    def test_typed_startup_target_deferral_does_not_expand_camera_pilot(self):
        self._check_startup_target_deferral(pilot=True)

    def test_startup_target_deferral_cannot_change_parent_routine_identity(self):
        self._check_startup_target_deferral(wrong_identity=True)

    def _check_startup_target_deferral(self, *, pilot=False, wrong_identity=False):
        from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
            CandidateStartupTargetUnavailableError, RejectedChildFailure,
        )
        from scripts.aufgabe04.real_robot.candidate.startup_recovery import CandidateRoutineIdentity

        # Sensor/certificate details of this subtype are covered by the real
        # coordinator tests. Here exercise the parent ledger and candidate loop.
        config = replace(self.config(count=2), require_current_lidar_support=False,
                         stop_after_camera_candidates=1 if pilot else None)
        attempted, observed = [], []

        def unavailable(uid):
            return CandidateObservationUnavailableError(
                candidate_uid=uid, observation_attempt_index=0, reason="candidate_target_ineligible",
                process_evidence={"observer_started": False, "motion_authorized": False},
                status_evidence={"reason": "current_lidar_target_unavailable"})

        def execute(**kwargs):
            uid = kwargs["target_id"]
            attempted.append(uid)
            if uid == "candidate_0":
                stopped = MotionLegOutcome(run_id=kwargs["run_id"], status="preflight_failed",
                    stop_reason="certified start mismatch", stop_details={}, motion_published=False,
                    returncode=1, semantic_log_path=self.root / "stopped.jsonl")
                raise CandidateStartupTargetUnavailableError(
                    observation_error=unavailable(uid),
                    initial_identity=CandidateRoutineIdentity(config.session_id, config.semantic_map_id,
                        "candidate_preapproach", kwargs["candidate_index"], uid,
                        "foreign_run" if wrong_identity else kwargs["run_id"]),
                    rejected_child=RejectedChildFailure.from_outcome(stopped,
                        policy_reason="current_target_unavailable", preserve_child_reason=True),
                    completed_startup_reseal_count=0, startup_reseal_index=1,
                    startup_target_deferral_evidence={"motion_published": False,
                                                      "test_consumed_authority_verified": True})
            request = kwargs["plan_request"]
            kwargs["completed_frame_sink"](approach._CandidateObservationFrame(
                kwargs["config"], request.snapshot.candidate_for(uid), kwargs["plan_planning_frame"], None,
                observation_pose=request.start))
            return replace(stopped_template, run_id=kwargs["run_id"], status="completed", returncode=0)

        stopped_template = MotionLegOutcome(run_id="template", status="stopped", stop_reason="",
            stop_details={}, motion_published=False, returncode=1, semantic_log_path=self.root / "test.jsonl")

        def observe(**kwargs):
            uid = kwargs["observation_frame"].candidate.candidate_uid
            observed.append(uid)
            raise unavailable(uid)

        effects = self.effects(admit_planning_frame=None,
            select_initial_preapproach=self.fixtures._nearest_selection,
            plan_preapproach=lambda request: {"route_csv": "injected_route.csv"})
        with patch(MODULE + "._execute_candidate_motion", side_effect=execute), \
                patch(MODULE + "._capture_candidate_camera_result", side_effect=observe):
            if wrong_identity:
                with self.assertRaisesRegex(RuntimeError, "changed candidate routine identity"):
                    approach.execute_candidate_approach_phase(config, effects)
                self.assertEqual(attempted, ["candidate_0"])
                self.assertFalse(observed)
                return
            with self.assertRaises(CandidateQrGoalIncompleteError):
                approach.execute_candidate_approach_phase(config, effects)
        self.assertEqual(attempted, ["candidate_0"] if pilot else ["candidate_0", "candidate_1"])
        self.assertEqual(observed, [] if pilot else ["candidate_1"])
        progress = json.loads((config.session_root / "candidate_goal_progress.json").read_text())
        self.assertFalse(progress["goal_completed"])
        self.assertEqual(progress["confirmed_stand_count"], 0)
        self.assertEqual(progress["keepout_candidate_uids"], ["candidate_0", "candidate_1"])
        self.assertEqual(progress["inspection_order"], [] if pilot else ["candidate_1"])
        first = next(d for d in progress["candidate_dispositions"] if d["candidate_uid"] == "candidate_0")
        self.assertEqual(first["disposition"], "target_reconciliation_required")
        event = next(e for e in self.events if e.get("event") == "camera_candidate_startup_target_deferred")
        self.assertEqual(event["candidate_uid"], "candidate_0")
        self.assertFalse(event["motion_authorized"])
        self.assertFalse(event["retry_eligible"])
        self.assertFalse(event["keepouts_changed"])

    def _check_reseal(self, owner, supported, *, retained_axis_compatible=None):
        root = self.root / f"{owner}_{supported}_{retained_axis_compatible}"
        root.mkdir()
        config = self.config()
        authorization = root / "authorization.json"
        authorization.write_text("{}")
        config = replace(config, max_startup_reseals_per_leg=1,
            max_runtime_localization_reseals_per_leg=1 if owner != "startup" else 0,
            mission_motion_authorization_json=authorization)
        original_frame, replacement_frame = self.frame(0.0), self.frame(.3)
        old_target, new_target = self.target(1.02), self.target(1.06)
        retained_axis_path = None
        axis_projections = []
        if retained_axis_compatible is not None:
            old_target = {**old_target, "uncertainty_m": .02}
            new_target = {**self.target(1.05 if retained_axis_compatible else 1.15),
                          "uncertainty_m": .02}
            retained_axis_path = root / "retained_axis.json"
            retained_axis_path.write_text("{}")
        order, plans, replacements, target_cohorts, completed_frames = [], [], [], [], []
        known_targets = {}
        def replay(path, *, candidate_uid, snapshot):
            return known_targets[str(path)]
        def capture(cfg, effects, frame, uids, output):
            self.assertEqual(frame, replacement_frame)
            order.append("scan")
            target = self.target(1.10) if target_cohorts else new_target
            target_cohorts.append(target)
            estimates, evidence = self.support(cfg, frame, uids, output,
                {"candidate_0": target} if supported else {})
            if supported:
                known_targets[evidence["evidence_path"]] = target
            return estimates, evidence
        def admit(path):
            order.append("localize")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            return replacement_frame
        def initial(request):
            order.append("initial_motion")
            if owner == "runtime":
                stopped = _runtime_stop(root, request.run_id)
                return replace(stopped,
                    mission_leg_motion_permit_path=stopped.startup_reseal_motion_permit_path,
                    mission_leg_motion_permit_sha256=stopped.startup_reseal_motion_permit_sha256,
                    startup_reseal_motion_permit_path=None, startup_reseal_motion_permit_sha256="")
            return MotionLegOutcome(run_id=request.run_id, status="stopped",
                stop_reason="pose outside certified startup segment",
                stop_details={"source": "execution_route_certificate", "phase": "before_motion_confirmation",
                    "reason": "pose outside certified startup segment", "fail_closed": True,
                    "route_pose": {"x_m": .3, "y_m": 0.0, "yaw_rad": 0.0}},
                motion_published=False, returncode=1, semantic_log_path=root / "initial.jsonl")
        def plan(request):
            order.append("plan")
            plans.append(request)
            view = load_candidate_inspection_view(request.inspection_view_path)
            self.assertEqual(view["validated_target_center"], target_cohorts[-1])
            self.assertNotEqual(view["current_lidar_targets_path"], old_support["evidence_path"])
            self.assertEqual(view["start_pose"], {"x_m": .3, "y_m": 0., "yaw_rad": 0.})
            self.assertAlmostEqual(abs(view["view_normal_rad"]), math.pi)
            self.assertIsNone(request.prepared_plan)
            self.assertIsNone(request.selection_evidence)
            self.assertIsNone(request.axis_observation_path)
            return {"route_csv": "replacement.csv"}
        def replacement(request, attempt):
            order.append("replacement_motion")
            replacements.append(request)
            if owner == "startup_runtime" and len(replacements) == 1:
                return _runtime_stop(root, request.run_id)
            if owner == "startup":
                return self.fixtures._startup_completed(request)
            return _outcome(root, run_id=request.run_id, status="completed", motion_published=True,
                permit_name="runtime_replacement.json", permit_digest="d" * 64)
        effects = self.effects(admit_planning_frame=admit, run_motion_leg=initial,
            plan_preapproach=plan, run_startup_reseal_motion_leg=replacement,
            admit_runtime_localization=lambda path: replacement_frame.current_pose,
            run_runtime_localization_reseal_motion_leg=replacement)
        _, old_support = self.support(config, original_frame, {"candidate_0"}, root / "old_support",
                                      {"candidate_0": old_target})
        known_targets[old_support["evidence_path"]] = old_target
        view_path = root / "initial_view.json"
        def completed_frame(*, config, plan_request, planning_frame):
            view = load_candidate_inspection_view(plan_request.inspection_view_path)
            return SimpleNamespace(config=config, plan_request=plan_request, planning_frame=planning_frame,
                                   validated_target_center=view["validated_target_center"])
        def project_retained_axis(path, **kwargs):
            order.append("axis_project")
            self.assertEqual(kwargs["axis_evidence_path"], retained_axis_path)
            projection = json.loads(kwargs["target_candidate_projection_path"].read_text())
            self.assertEqual(projection["planning_frame_admission"], replacement_frame.to_evidence())
            self.assertAlmostEqual(kwargs["target_candidate_x_m"], 1.)
            self.assertAlmostEqual(kwargs["target_candidate_y_m"], 0.)
            axis_projections.append(path)
        def load_retained_axis(path):
            order.append("axis_load")
            self.assertEqual(path, axis_projections[-1])
            return SimpleNamespace(validated_target_center={
                "x_m": 1.02, "y_m": 0., "uncertainty_m": .02,
                "policy": "reconciled_metric_head_position_engineering_bound"})
        with patch(SENSOR + ".load_current_lidar_target", side_effect=replay), \
                patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture), \
                patch(MODULE + "._observation_frame_for_plan", side_effect=completed_frame) as frame_builder, \
                ExitStack() as retained_patches:
            if retained_axis_path is not None:
                retained_patches.enter_context(patch(MODULE + ".write_backside_axis_frame_projection",
                    side_effect=project_retained_axis))
                retained_patches.enter_context(patch(MODULE + ".load_backside_axis_planning_observation",
                    side_effect=load_retained_axis))
            write_candidate_inspection_view(view_path, snapshot=config.snapshot, candidate_uid="candidate_0",
                start=original_frame.current_pose, view_normal_rad=math.pi, purpose="current_lidar_target",
                view_index=0, validated_target_center=old_target,
                current_lidar_targets_path=Path(old_support["evidence_path"]))
            request = approach.CandidatePreapproachRequest(config.map_yaml, config.semantic_map_id,
                config.plan, config.snapshot, config.snapshot_path, "candidate_0", original_frame.current_pose,
                root / "initial_route", config.approach_offset_m, config.inflation_radius_m,
                config.candidate_transit_radius_m, config.physical_clearance, inspection_view_path=view_path)
            kwargs = dict(config=config, effects=effects, candidate_root=root, plan_request=request,
                initial_sealed={"route_csv": "initial.csv"}, run_id=f"candidate_{owner}_{supported}",
                leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH, candidate_index=0, target_id="candidate_0",
                frame_source_config=config, source_registry=self.registry(config),
                plan_planning_frame=original_frame, completed_frame_sink=completed_frames.append,
                retained_backside_axis_path=retained_axis_path)
            retained_steps = [] if retained_axis_path is None else ["axis_project", "axis_load"]
            if supported and retained_axis_compatible is not False:
                outcome = approach._execute_candidate_motion(**kwargs)
                self.assertEqual(outcome.status, "completed")
                self.assertEqual(order, ["initial_motion"] +
                                 (["localize", "scan"] + retained_steps + ["plan", "replacement_motion"]) *
                                 (2 if owner == "startup_runtime" else 1))
                self.assertEqual(len(completed_frames), 1)
                frame_builder.assert_called_once()
                self.assertIs(completed_frames[0].plan_request, plans[-1])
                self.assertEqual(completed_frames[0].validated_target_center, target_cohorts[-1])
                self.assertNotEqual(completed_frames[0].validated_target_center, old_target)
                self.assertEqual(completed_frames[0].planning_frame, replacement_frame)
                if owner == "startup_runtime":
                    self.assertEqual([target["x_m"] for target in target_cohorts], [1.06, 1.10])
                    self.assertIn("startup_reseal_001_runtime_localization_reseal_001", outcome.run_id)
            else:
                with self.assertRaises((CandidateStartupRecoveryError, CandidateRuntimeRecoveryError)) as caught:
                    approach._execute_candidate_motion(**kwargs)
                self.assertEqual(order, ["initial_motion", "localize", "scan"] + retained_steps)
                self.assertFalse(plans)
                self.assertFalse(replacements)
                self.assertFalse(completed_frames)
                frame_builder.assert_not_called()
                if retained_axis_compatible is False:
                    cause = caught.exception.__cause__
                    self.assertIsInstance(cause, CandidateObservationUnavailableError)
                    self.assertEqual(cause.status_evidence["reason"],
                                     "current_lidar_disagrees_with_retained_target")
            if retained_axis_path is not None:
                self.assertEqual(len(axis_projections), 1)


if __name__ == "__main__":
    unittest.main()
