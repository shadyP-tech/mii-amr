"""Current-scan target gating through the actual approach/reseal orchestration.

Only the sensor support capture/replay boundary is substituted. Real frame
projection, registry preservation, selection handoff, arrival and both recovery
coordinators run against temporary artifacts.
"""
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
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import CandidateQrGoalIncompleteError
from scripts.aufgabe04.real_robot.candidate.recovery_failure import CandidateStartupRecoveryError
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import CandidateRuntimeRecoveryError
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_autonomous_candidate_runtime_recovery import _runtime_stop, _outcome
from tests.aufgabe04.test_lidar_alignment_arrival import calibration


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

    def test_fresh_capture_precedes_selection_and_filters_only_target_eligibility(self):
        config = self.config(count=2)
        order, requests = [], []
        estimate = self.target(2.06)
        def capture(cfg, effects, frame, uids, output):
            order.append("scan")
            self.assertEqual(uids, set(config.snapshot.candidate_uids))
            return self.support(cfg, frame, uids, output, {"candidate_1": estimate})
        def select(request):
            order.append("select")
            requests.append(request)
            return approach.CameraCandidateInitialSelection("candidate_1",
                SimpleNamespace(candidate_uid="candidate_1", validated_target_center=estimate),
                {"selected_candidate_uid": "candidate_1", "motion_authorized": False})
        effects = self.effects(select_initial_preapproach=select)
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
            _, _, selection, _ = approach._select_initial_candidate_with_localization_refresh(
                config=config, effects=effects, source_registry=self.registry(config), candidate_index=0,
                eligible=set(config.snapshot.candidate_uids), exact_two_support_by_uid=None)
        self.assertEqual(order, ["scan", "select"])
        self.assertEqual(requests[0].unresolved, frozenset({"candidate_1"}))
        self.assertEqual(requests[0].current_target_estimates, {"candidate_1": estimate})
        self.assertEqual(requests[0].config.snapshot.candidate_uids, config.snapshot.candidate_uids)
        self.assertEqual(requests[0].config.snapshot.candidate_for("candidate_0").geometry,
                         config.snapshot.candidate_for("candidate_0").geometry)
        self.assertEqual(selection.evidence["current_lidar_support"]["excluded_candidate_uids"], ["candidate_0"])
        effects.run_motion_leg.assert_not_called()
        effects.capture_observation.assert_not_called()

    def test_supported_selection_cannot_drop_or_replace_the_measured_target(self):
        config = self.config()
        estimate = self.target()
        for index, prepared in enumerate((None,
                SimpleNamespace(candidate_uid="candidate_0", validated_target_center=None),
                SimpleNamespace(candidate_uid="candidate_0", validated_target_center=self.target(1.10)))):
            with self.subTest(case=index):
                current = replace(config, session_root=self.root / f"bad_selection_{index}")
                def capture(cfg, effects, frame, uids, output):
                    return self.support(cfg, frame, uids, output, {"candidate_0": estimate})
                effects = self.effects(select_initial_preapproach=Mock(return_value=
                    approach.CameraCandidateInitialSelection("candidate_0", prepared,
                        {"selected_candidate_uid": "candidate_0", "motion_authorized": False})))
                with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
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

    def test_all_unsupported_stops_before_planning_motion_and_camera_with_incomplete_progress(self):
        config = self.config(count=2)
        effects = self.effects()
        registry_path = config.survey_root / "stand_registry.json"
        original_registry, original_snapshot = registry_path.read_bytes(), config.snapshot_path.read_bytes()
        def capture(cfg, effects, frame, uids, output):
            return self.support(cfg, frame, uids, output, {})
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
            with self.assertRaises(CandidateQrGoalIncompleteError) as rejected:
                approach.execute_candidate_approach_phase(config, effects)
        for effect in (effects.select_initial_preapproach, effects.plan_preapproach,
                       effects.run_motion_leg, effects.capture_observation):
            effect.assert_not_called()
        progress = rejected.exception.progress
        self.assertFalse(progress["goal_completed"])
        self.assertEqual(progress["confirmed_stand_count"], 0)
        self.assertEqual(progress["keepout_candidate_uids"], list(config.snapshot.candidate_uids))
        self.assertTrue(all(item["disposition"] == "target_reconciliation_required"
                            for item in progress["candidate_dispositions"]))
        self.assertEqual(registry_path.read_bytes(), original_registry)
        self.assertEqual(config.snapshot_path.read_bytes(), original_snapshot)
        self.assertEqual(self.events[-1]["event"], "camera_candidates_current_lidar_unavailable")

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

    def test_camera_arrival_recaptures_support_before_starting_observer(self):
        for supported in (True, False):
            with self.subTest(supported=supported):
                config = self.config()
                order = []
                def capture(cfg, effects, frame, uids, output):
                    order.append("scan")
                    return self.support(cfg, frame, uids, output,
                        {"candidate_0": self.target()} if supported else {})
                def camera(request):
                    order.append("camera")
                    return approach.CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
                effects = self.effects(capture_observation=Mock(side_effect=camera))
                frame = approach._CandidateObservationFrame(config, config.snapshot.candidates[0],
                    self.frame(), None, observation_pose=Pose2D(.3, 0.0, 0.0))
                args = dict(observation_frame=frame, source_config=config, effects=effects,
                    source_registry=self.registry(config), candidate_root=self.root / f"arrival_{supported}",
                    candidate_run_id="candidate_arrival", candidate_index=0)
                with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
                    if supported:
                        _, admitted = approach._capture_candidate_camera_result(**args)
                        self.assertEqual(order, ["scan", "camera"])
                        self.assertAlmostEqual(admitted.camera_target_geometry.x_m, 1.06)
                        self.assertAlmostEqual(admitted.candidate.geometry.x_m, 1.0)
                    else:
                        with self.assertRaises(CandidateObservationUnavailableError):
                            approach._capture_candidate_camera_result(**args)
                        self.assertEqual(order, ["scan"])
                        effects.capture_observation.assert_not_called()
                effects.run_motion_leg.assert_not_called()

    def test_startup_and_runtime_reseals_reacquire_before_planning_replacement(self):
        for owner in ("startup", "runtime"):
            for supported in (True, False):
                with self.subTest(owner=owner, supported=supported):
                    self._check_reseal(owner, supported)

    def test_ambiguous_arrival_reacquires_before_camera_and_opposite_dispatch(self):
        config = self.config()
        order, paths = [], []

        def capture(cfg, effects, frame, uids, output):
            index = len(paths)
            paths.append(Path(output))
            order.append("scan")
            accepted = index == 1
            evidence = {
                "eligible_candidate_uids": ["candidate_0"] if accepted else [],
                "excluded_candidate_uids": [] if accepted else ["candidate_0"],
                "candidate_decisions": {"candidate_0": {
                    "candidate_uid": "candidate_0", "accepted": accepted,
                    "reasons": [] if accepted else ["ambiguous_current_target_correspondence"],
                    "scans": [{"scan_stamp_sec": 100.0 + index * 3 + i * .1,
                               "reason": "ambiguous_clusters" if not accepted and i == 5 else "supported"}
                              for i in range(8)],
                }},
                "observation_not_before_sec": 100.0 + index * 3,
                "planning_frame": frame.to_evidence(), "motion_authorized": False,
                "keepouts_changed": False,
            }
            path = Path(output) / "current_lidar_targets.json"
            digest = write_content_hashed_json(path, evidence, hash_field=HASH_FIELD)
            return ({"candidate_0": self.target()} if accepted else {},
                    {**evidence, "evidence_path": str(path), "evidence_sha256": digest})

        def camera(request):
            order.append("camera")
            return approach.CandidateObservation(None, None, self.root / "camera_axis.json")

        class OppositeDispatchReached(RuntimeError):
            pass

        def opposite(**kwargs):
            order.append("opposite")
            self.assertEqual(kwargs["observation_frame"].current_lidar_target_path,
                             paths[1] / "current_lidar_targets.json")
            # The test stops at the existing certified route boundary; it
            # never substitutes an unsafe route or claims executed motion.
            raise OppositeDispatchReached()

        effects = self.effects(capture_observation=Mock(side_effect=camera),
                               wait_for_lidar_reacquisition=lambda seconds: order.append("pause"))
        frame = approach._CandidateObservationFrame(config, config.snapshot.candidates[0],
            self.frame(), None, observation_pose=self.frame().current_pose)
        with patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture), \
                patch(MODULE + "._move_certified_opposite_face", side_effect=opposite):
            with self.assertRaises(OppositeDispatchReached):
                approach._capture_candidate_camera_result(
                    observation_frame=frame, source_config=config, effects=effects,
                    source_registry=self.registry(config), candidate_root=self.root / "arrival_retry",
                    candidate_run_id="candidate_arrival_retry", candidate_index=0)
        self.assertEqual(order, ["scan", "pause", "scan", "camera", "opposite"])
        self.assertEqual(paths[1], paths[0] / "reacquire_001")
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

    def _check_reseal(self, owner, supported):
        root = self.root / f"{owner}_{supported}"
        root.mkdir()
        config = self.config()
        authorization = root / "authorization.json"
        authorization.write_text("{}")
        config = replace(config, max_startup_reseals_per_leg=1,
            max_runtime_localization_reseals_per_leg=1 if owner == "runtime" else 0,
            mission_motion_authorization_json=authorization)
        original_frame, replacement_frame = self.frame(0.0), self.frame(.3)
        old_target, new_target = self.target(1.02), self.target(1.06)
        order, plans, replacements = [], [], []
        known_targets = {}
        def replay(path, *, candidate_uid, snapshot):
            return known_targets[str(path)]
        def capture(cfg, effects, frame, uids, output):
            self.assertEqual(frame, replacement_frame)
            order.append("scan")
            estimates, evidence = self.support(cfg, frame, uids, output,
                {"candidate_0": new_target} if supported else {})
            if supported:
                known_targets[evidence["evidence_path"]] = new_target
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
            self.assertEqual(view["validated_target_center"], new_target)
            self.assertNotEqual(view["current_lidar_targets_path"], old_support["evidence_path"])
            self.assertEqual(view["start_pose"], {"x_m": .3, "y_m": 0., "yaw_rad": 0.})
            self.assertAlmostEqual(abs(view["view_normal_rad"]), math.pi)
            self.assertIsNone(request.prepared_plan)
            self.assertIsNone(request.selection_evidence)
            return {"route_csv": "replacement.csv"}
        def replacement(request, attempt):
            order.append("replacement_motion")
            replacements.append(request)
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
        with patch(SENSOR + ".load_current_lidar_target", side_effect=replay), \
                patch(SENSOR + ".capture_current_lidar_targets", side_effect=capture):
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
                plan_planning_frame=original_frame)
            if supported:
                outcome = approach._execute_candidate_motion(**kwargs)
                self.assertEqual(outcome.status, "completed")
                self.assertEqual(order, ["initial_motion", "localize", "scan", "plan", "replacement_motion"])
            else:
                with self.assertRaises((CandidateStartupRecoveryError, CandidateRuntimeRecoveryError)):
                    approach._execute_candidate_motion(**kwargs)
                self.assertEqual(order, ["initial_motion", "localize", "scan"])
                self.assertFalse(plans)
                self.assertFalse(replacements)


if __name__ == "__main__":
    unittest.main()
