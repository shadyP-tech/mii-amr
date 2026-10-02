"""Candidate-local deferral must preserve stopped-child authority and lineage."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind, mission_leg_motion_permit_sha256, write_mission_leg_motion_permit,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import (
    default_mission_leg_motion_consumption_receipt_path,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
    startup_reseal_motion_authorization_sha256, startup_reseal_motion_permit_sha256,
    write_startup_reseal_motion_authorization, write_startup_reseal_motion_permit,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_permit_retirement import (
    DISPOSITION_HASH_FIELD, validate_odom_startup_rejected_permit_disposition,
)
from scripts.aufgabe04.navigation.localization.ros_preflight_evidence_contract import (
    ros_preflight_requirements_evidence,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.recovery_dispatch import (
    CandidateRecoveryHandoff, CandidateRecoveryState,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import (
    CandidateRoutineIdentity, CandidateStartupRecoveryError,
    CandidateStartupTargetUnavailableError, execute_candidate_motion_with_startup_recovery,
)
from scripts.aufgabe04.real_robot.candidate.startup_permit_retirement import retire_candidate_startup_permit
from tests.aufgabe04 import test_autonomous_candidate_startup_recovery as startup_fixture
from tests.aufgabe04 import test_mission_leg_motion_consumption as mission_fixture
from tests.aufgabe04 import test_startup_reseal_motion_consumption as receipt_fixture
from tests.aufgabe04 import test_startup_reseal_permit_retirement as retirement_fixture
from tests.aufgabe04.test_initial_map_tf_recovery import initial_map_tf_stop


class StartupTargetDeferralTest(unittest.TestCase):
    def setUp(self):
        self.h = startup_fixture.CandidateStartupRecoveryTest()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.h.identity = CandidateRoutineIdentity(
            "mission-001", "arena", "candidate_preapproach", 3,
            "survey_candidate_0002", "mission-001_candidate_002",
        )

    def error(self, **overrides):
        values = dict(
            candidate_uid=self.h.identity.target_id, observation_attempt_index=0,
            reason="candidate_target_ineligible",
            process_evidence={"observer_started": False, "motion_authorized": False},
            status_evidence={"reason": "current_lidar_target_unavailable",
                             "current_lidar_support": {"evidence_path": "/captured/target.json"}},
        )
        values.update(overrides)
        return CandidateObservationUnavailableError(**values)

    def stopped(self, identity=None, **fields):
        identity = identity or self.h.identity
        return startup_fixture._outcome(
            self.h.root, run_id=identity.run_id, status="stopped",
            stop_reason="TF transform unavailable: map <- odom",
            stop_details=initial_map_tf_stop(), **fields,
        )

    def mission_permit(self, *, consume=True):
        fixture = mission_fixture.MissionLegMotionConsumptionTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        identity = self.h.identity
        fixture.permit = replace(
            fixture.permit, run_id=identity.run_id,
            mission_leg_kind=MissionLegKind(identity.routine_kind),
            mission_leg_index=identity.routine_index, target_id=identity.target_id,
        )
        fixture.permit_path = fixture.root / "candidate_permit.json"
        write_mission_leg_motion_permit(fixture.permit_path, fixture.permit)
        if consume:
            fixture._consume()
        return fixture

    def startup_permit(self):
        """Reuse execution artifact fixtures with this candidate's exact identity."""
        fixture = receipt_fixture.StartupResealMotionConsumptionTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        initial, identity = self.h.identity, self.h.identity.replacement(1)
        fixture.authorization = replace(
            fixture.authorization, semantic_map_id=identity.semantic_map_id,
            allowed_mission_leg_kinds=(MissionLegKind.CANDIDATE_PREAPPROACH,),
        )
        fixture.master_path = fixture.root / "candidate_master.json"
        write_startup_reseal_motion_authorization(fixture.master_path, fixture.authorization)
        changes = {
            "rejected_semantic_log": {
                "run_id": initial.run_id, "mission_leg_kind": identity.routine_kind,
                "mission_leg_index": identity.routine_index, "target_id": identity.target_id,
                "coverage_leg_index": None, "target_viewpoint_id": "",
            },
            "startup_reseal_summary": {
                "rejected_run_id": initial.run_id, "mission_leg_kind": identity.routine_kind,
                "mission_leg_index": identity.routine_index,
                "target_id": identity.target_id, "target_viewpoint_id": identity.target_id,
                "recovery_source_kind": STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
            },
            "fresh_stationary_localization_evidence": {
                "preflight_requirements": ros_preflight_requirements_evidence(
                    stationary_map_from_odom_pairing_requested=True,
                    stationary_map_from_odom_pairing_required=True,
                ),
            },
        }
        for name, values in changes.items():
            path = fixture.artifacts[name]
            raw = json.loads(path.read_text())
            path.write_text(json.dumps({**raw, **values}) + "\n")
        binding = changes["rejected_semantic_log"]
        events = [
            {**binding, "event": "mission_leg_motion_permit_consumed",
             "covered_by_initial_mission_run": True, "additional_typed_run_required": False},
            {**binding, "event": "motion_started", "motion_published": False,
             "event_semantics": "child_execution_attempt_started_before_follower"},
            {**binding, "event": "safety_stop", "status": "stopped", "motion_published": False,
             "stop_reason": "TF transform unavailable: map <- odom", "stop_details": initial_map_tf_stop()},
        ]
        fixture.artifacts["rejected_semantic_log"].write_text(
            "".join(json.dumps(event) + "\n" for event in events))
        fixture.permit = replace(
            fixture.permit, run_id=identity.run_id, rejected_run_id=initial.run_id,
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            target_id=identity.target_id, target_viewpoint_id=identity.target_id,
            master_authorization_path=str(fixture.master_path),
            master_authorization_sha256=startup_reseal_motion_authorization_sha256(fixture.authorization),
            rejected_semantic_log_sha256=fixture._sha("rejected_semantic_log"),
            startup_reseal_summary_sha256=fixture._sha("startup_reseal_summary"),
            fresh_stationary_localization_evidence_sha256=fixture._sha("fresh_stationary_localization_evidence"),
            recovery_source_kind=STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
        )
        fixture.permit_path = fixture.root / "candidate_permit.json"
        write_startup_reseal_motion_permit(fixture.permit_path, fixture.permit)
        fixture._consume(mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                         mission_leg_index=identity.routine_index, target_id=identity.target_id)
        return fixture

    def execute(self, error=None, *, initial=None, replan=None, state=None, resumed=None,
                replacements=(), effects_mutator=None):
        def reject(attempt):
            raise error or self.error()
        effects, calls = self.h._effects(
            initial=initial or self.stopped(), replacements=list(replacements),
            replan=replan or reject,
        )
        if effects_mutator is not None:
            effects = effects_mutator(effects)
        return execute_candidate_motion_with_startup_recovery(
            startup_fixture._Request(self.h.identity), config=self.h._config(3),
            effects=effects, recovery_state=state, resumed_handoff=resumed,
        )

    def test_initial_consumed_child_defers_and_keeps_exact_lineage(self):
        permit = self.mission_permit()
        outcome = self.stopped(
            mission_leg_motion_permit_path=permit.permit_path,
            mission_leg_motion_permit_sha256=mission_leg_motion_permit_sha256(permit.permit),
        )
        with self.assertRaises(CandidateStartupTargetUnavailableError) as raised:
            self.execute(initial=outcome, state=CandidateRecoveryState())
        error = raised.exception
        self.assertEqual(error.initial_identity, self.h.identity)
        self.assertEqual(error.initial_run_id, outcome.run_id)
        self.assertEqual(error.target_id, self.h.identity.target_id)
        self.assertEqual(error.startup_reseal_index, 1)
        self.assertEqual(error.completed_startup_reseal_count, 0)
        evidence = error.observation_error.process_evidence["candidate_startup_target_deferral"]
        child = evidence["rejected_children"][0]
        self.assertEqual(child["stop_details"], outcome.stop_details)
        self.assertEqual(child["closed_motion_authority"]["disposition"], "consumed_before_motion")
        self.assertEqual(child["issued_motion_permits"]["routine_mission_leg"]["path"], str(permit.permit_path))
        self.assertTrue(Path(evidence["fresh_localization_evidence_json"]).is_file())
        self.assertFalse(evidence["replacement_started"])
        self.assertFalse(evidence["replacement_permit_issued"])
        self.assertFalse(self.h.replacement_attempts)
        self.assertEqual(self.h.events[-1]["event"], "candidate_startup_target_deferred")
        with self.assertRaisesRegex(ValueError, "already consumed"):
            permit._consume()

    def test_repeated_before_motion_stops_defer_without_resetting_counter(self):
        permit, replacement = self.mission_permit(), self.startup_permit()
        initial = self.stopped(
            mission_leg_motion_permit_path=permit.permit_path,
            mission_leg_motion_permit_sha256=mission_leg_motion_permit_sha256(permit.permit),
        )
        initial = replace(initial, semantic_log_path=replacement.artifacts["rejected_semantic_log"])
        stopped = self.stopped(
            self.h.identity.replacement(1), startup_reseal_motion_permit_path=replacement.permit_path,
            startup_reseal_motion_permit_sha256=startup_reseal_motion_permit_sha256(replacement.permit),
        )
        def replan(attempt):
            if attempt.reseal_index == 2:
                raise self.error()
            return self.h._replan(attempt)
        state = CandidateRecoveryState()
        with self.assertRaises(CandidateStartupTargetUnavailableError) as raised:
            self.execute(initial=initial, replacements=[stopped], replan=replan, state=state)
        evidence = raised.exception.startup_target_deferral_evidence
        self.assertEqual(state.startup_reseal_count, 1)
        self.assertEqual(evidence["startup_reseal_index"], 2)
        self.assertEqual(evidence["completed_startup_reseal_count"], 1)
        self.assertEqual([child["child_run_id"] for child in evidence["rejected_children"]],
                         [initial.run_id, stopped.run_id])
        self.assertEqual(len(self.h.replacement_attempts), 1)
        self.assertEqual(len(self.h.admitted_paths), 2)

    def test_wrong_target_reason_or_observer_flags_remain_terminal(self):
        variants = (
            {"candidate_uid": "survey_candidate_0001"},
            {"reason": "candidate_arrival_geometry_rejected"},
            {"status_evidence": {"reason": "static_map_target_rejected"}},
            {"process_evidence": {"observer_started": True, "motion_authorized": False}},
            {"process_evidence": {"observer_started": False, "motion_authorized": True}},
            {"process_evidence": {"observer_started": False}},
        )
        for index, values in enumerate(variants):
            with self.subTest(values=values):
                self.h.root = self.h.root / str(index)
                with self.assertRaises(CandidateStartupRecoveryError) as raised:
                    self.execute(self.error(**values))
                self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
                self.assertEqual(raised.exception.phase, "same_routine_replan")

    def test_generic_and_integrity_replan_failures_remain_terminal(self):
        for index, error in enumerate((RuntimeError("scan unavailable"), ValueError("hash mismatch"))):
            with self.subTest(error=error):
                self.h.root = self.h.root / str(index)
                with self.assertRaises(CandidateStartupRecoveryError) as raised:
                    self.execute(error)
                self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
                self.assertIs(raised.exception.__cause__, error)

    def test_opposite_face_runtime_history_and_resumption_never_defer(self):
        for index, variant in enumerate(("opposite", "runtime", "startup", "resumed")):
            with self.subTest(variant=variant):
                self.h.root = self.h.root / str(index)
                state, handoff = None, None
                if variant == "opposite":
                    self.h.identity = replace(self.h.identity, routine_kind="opposite_face")
                else:
                    self.h.identity = replace(self.h.identity, routine_kind="candidate_preapproach")
                    state = CandidateRecoveryState(
                        runtime_reseal_count=int(variant == "runtime"),
                        startup_reseal_count=int(variant == "startup"),
                    )
                    if variant == "resumed":
                        handoff = CandidateRecoveryHandoff(self.stopped(), "startup")
                with self.assertRaises(CandidateStartupRecoveryError) as raised:
                    self.execute(state=state, resumed=handoff)
                self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
                self.assertEqual(raised.exception.phase, "same_routine_replan")

    def test_post_motion_child_and_invalid_startup_source_never_reach_replan(self):
        for index, outcome in enumerate((
            replace(self.stopped(), motion_published=True),
            replace(self.stopped(), stop_details={"source": "scan", "reason": "stale_scan"}),
        )):
            with self.subTest(outcome=outcome):
                self.h.root = self.h.root / str(index)
                with self.assertRaises(CandidateStartupRecoveryError) as raised:
                    self.execute(initial=outcome)
                self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
                self.assertFalse(self.h.admitted_paths)

    def test_missing_unclaimed_or_corrupted_consumption_is_terminal(self):
        unclaimed = self.mission_permit(consume=False)
        corrupt = self.mission_permit()
        default_mission_leg_motion_consumption_receipt_path(corrupt.permit_path).write_text("{}\n")
        cases = [self.stopped()]
        for fixture in (unclaimed, corrupt):
            cases.append(self.stopped(
                mission_leg_motion_permit_path=fixture.permit_path,
                mission_leg_motion_permit_sha256=mission_leg_motion_permit_sha256(fixture.permit),
            ))
        for index, outcome in enumerate(cases):
            with self.subTest(index=index):
                self.h.root = self.h.root / str(index)
                with self.assertRaises(CandidateStartupRecoveryError) as raised:
                    self.execute(initial=outcome)
                self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
                self.assertEqual(raised.exception.phase, "target_deferral_authority_evidence")

    def retirement_case(self, *, dry_run=False):
        fixture = retirement_fixture.StartupResealPermitRetirementTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.h.identity = replace(self.h.identity, semantic_map_id=fixture.old_permit.semantic_map_id)
        identity = self.h.identity
        fixture.run_id = identity.run_id
        fixture.identity.update(
            session_id=identity.session_id, rejected_run_id=identity.run_id,
            mission_leg_kind=identity.routine_kind, mission_leg_index=identity.routine_index,
            target_id=identity.target_id,
        )
        fixture.old_permit = replace(
            fixture.old_permit, run_id=identity.run_id,
            mission_leg_kind=MissionLegKind(identity.routine_kind),
            mission_leg_index=identity.routine_index, target_id=identity.target_id,
        )
        fixture.old_permit_path = fixture.root / "candidate_old_permit.json"
        write_mission_leg_motion_permit(fixture.old_permit_path, fixture.old_permit)
        fixture._write_log(dry_run=dry_run)
        details = fixture._events(dry_run=dry_run)[-1]["stop_details"]
        fields = {} if dry_run else {
            "mission_leg_motion_permit_path": fixture.old_permit_path,
            "mission_leg_motion_permit_sha256": mission_leg_motion_permit_sha256(fixture.old_permit),
        }
        outcome = startup_fixture._outcome(
            self.h.root, run_id=self.h.identity.run_id, status="preflight_failed",
            stop_reason=details["reason"], stop_details=details, **fields,
        )
        return fixture, replace(outcome, semantic_log_path=fixture.log_path)

    def retirement_effects(self, effects, *, retire=retire_candidate_startup_permit):
        def admit(path):
            # The real dry-disposition effect already creates this directory.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"accepted":true}\n')
            self.h.admitted_paths.append(path)
            return Pose2D(0.987, -0.070, 0.01)
        return replace(effects, retire_rejected_permit=retire,
                       admit_fresh_stationary_localization=admit)

    def test_genuine_dry_and_live_odom_closures_precede_deferral(self):
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                self.h.root = self.h.root / str(dry_run)
                fixture, outcome = self.retirement_case(dry_run=dry_run)
                order = []
                def retire(*args):
                    order.append("retire")
                    return retire_candidate_startup_permit(*args)
                def replan(attempt):
                    order.append("replan")
                    raise self.error()
                with self.assertRaises(CandidateStartupTargetUnavailableError) as raised:
                    self.execute(initial=outcome, replan=replan,
                                 effects_mutator=lambda effects: self.retirement_effects(effects, retire=retire))
                self.assertEqual(order, ["retire", "replan"])
                child = raised.exception.startup_target_deferral_evidence["rejected_children"][0]
                closure = child["closed_motion_authority"]
                self.assertEqual(closure["disposition"], "closed_by_startup_route_admission")
                if not dry_run:
                    self.assertEqual(closure["evidence_path"], str(
                        default_mission_leg_motion_consumption_receipt_path(fixture.old_permit_path)))
                    with self.assertRaisesRegex(ValueError, "already consumed"):
                        fixture._consume_old()

    def test_retirement_contents_identity_map_and_exclusive_claim_are_revalidated(self):
        cases = ("malformed", "foreign_session", "foreign_run", "foreign_target", "foreign_index",
                 "wrong_digest", "wrong_permit_path", "wrong_map", "copied_claim", "dry_foreign_path")
        for case in cases:
            with self.subTest(case=case):
                self.h.root = self.h.root / case
                fixture, outcome = self.retirement_case(dry_run=case == "dry_foreign_path")
                if case == "wrong_map":
                    self.h.identity = replace(self.h.identity, semantic_map_id="foreign_map")
                elif case == "wrong_digest":
                    outcome = replace(outcome, mission_leg_motion_permit_sha256="f" * 64)
                elif case == "wrong_permit_path":
                    copied = fixture.root / "copied_permit.json"
                    copied.write_bytes(fixture.old_permit_path.read_bytes())
                    outcome = replace(outcome, mission_leg_motion_permit_path=copied)
                def retire(child, identity, index, root):
                    # Build a real tombstone independently of a deliberately
                    # wrong callback binding, then challenge the coordinator.
                    original = replace(child,
                        mission_leg_motion_permit_path=(None if case == "dry_foreign_path" else fixture.old_permit_path),
                        mission_leg_motion_permit_sha256=("" if case == "dry_foreign_path" else mission_leg_motion_permit_sha256(fixture.old_permit)))
                    path = retire_candidate_startup_permit(original, identity, index, root)
                    if case == "malformed":
                        path.write_text("{}\n")
                    elif case.startswith("foreign_"):
                        field = {"foreign_session": "session_id", "foreign_run": "rejected_run_id",
                                 "foreign_target": "target_id", "foreign_index": "mission_leg_index"}[case]
                        payload = json.loads(path.read_text())
                        payload.pop(DISPOSITION_HASH_FIELD)
                        payload[field] = 99 if field == "mission_leg_index" else "foreign"
                        payload[DISPOSITION_HASH_FIELD] = payload_sha256(payload)
                        path.write_text(json.dumps(payload))
                    elif case in {"copied_claim", "dry_foreign_path"}:
                        copied_path = fixture.root / "copied_disposition.json"
                        copied_path.write_bytes(path.read_bytes())
                        return copied_path
                    return path
                with self.assertRaises(CandidateStartupRecoveryError) as raised:
                    self.execute(initial=outcome,
                                 effects_mutator=lambda effects: self.retirement_effects(effects, retire=retire))
                self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
                self.assertEqual(raised.exception.phase, "target_deferral_authority_evidence")

    def test_changed_odom_disposition_cannot_be_used_to_defer(self):
        fixture, outcome = self.retirement_case()
        paths = []
        def retire(*args):
            path = retire_candidate_startup_permit(*args)
            paths.append(path)
            return path
        def replan(attempt):
            paths[0].write_text('{"disposition":"changed"}')
            raise self.error()
        with self.assertRaises(CandidateStartupRecoveryError) as raised:
            self.execute(initial=outcome, replan=replan,
                         effects_mutator=lambda effects: self.retirement_effects(effects, retire=retire))
        self.assertNotIsInstance(raised.exception, CandidateStartupTargetUnavailableError)
        self.assertEqual(raised.exception.phase, "target_deferral_authority_evidence")

    def test_retired_startup_replacement_requires_cumulative_reseal_index(self):
        fixture = retirement_fixture.StartupResealPermitRetirementTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        permit_path, permit = fixture._issue_replacement()
        fixture.run_id = permit.run_id
        fixture.identity.update(rejected_run_id=permit.run_id, reseal_index=2)
        fixture.route_path = Path(permit.route_csv_path)
        fixture.certificate_path = Path(permit.map_route_certificate_path)
        route = retirement_fixture.load_route_leg(fixture.route_path, 0, require_motion=False)
        previous = fixture.evidence
        fixture.evidence = retirement_fixture.build_odom_startup_route_admission_evidence(
            **{key: previous[key] for key in (
                "odom_pose", "chained_map_pose", "map_from_odom", "pose_tf_observations",
                "map_frame", "odom_frame", "base_frame", "tracking_tube_radius_m", "max_tf_age_sec",
                "max_composition_yaw_error_rad", "source_preflight_sha256",
            )},
            map_route=retirement_fixture.poses_from_waypoints(route.executable_waypoints),
            source_map_execution_certificate_sha256=payload_sha256({
                key: value for key, value in json.loads(fixture.certificate_path.read_text()).items()
                if key != "execution_route_certificate_sha256"
            }),
        )
        fixture.log_path = fixture.root / "replacement_startup_rejection.jsonl"
        fixture._write_log()
        permit_bindings = dict(
            permit_path=permit_path, permit_kind="startup_reseal",
            expected_permit_sha256=startup_reseal_motion_permit_sha256(permit),
        )
        disposition = fixture._retire(**permit_bindings)
        inputs = dict(
            **permit_bindings, **fixture.identity,
            expected_sha256=retirement_fixture.file_sha256(disposition),
            semantic_map_id=fixture.replacement.authorization.semantic_map_id,
            rejected_semantic_log_path=fixture.log_path,
            no_permit_disposition_path=fixture.root / "unused_dry_disposition.json",
        )
        validate_odom_startup_rejected_permit_disposition(disposition, **inputs)
        with self.assertRaisesRegex(ValueError, "cumulative reseal index"):
            validate_odom_startup_rejected_permit_disposition(
                disposition, **{**inputs, "reseal_index": 1})


if __name__ == "__main__":
    unittest.main()
