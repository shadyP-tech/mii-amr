from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json, payload_sha256
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCertificate, execution_route_certificate_sha256,
    write_execution_route_certificate,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import (
    consume_mission_leg_motion_permit, default_mission_leg_motion_consumption_receipt_path,
    load_mission_leg_motion_consumption_receipt,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind, MissionLegMotionAuthorization, MissionLegMotionPermit,
    write_mission_leg_motion_authorization, write_mission_leg_motion_permit,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_ODOM_STARTUP_ROUTE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    _validate_fresh_stationary_localization_evidence,
    file_sha256, write_startup_reseal_motion_authorization, write_startup_reseal_motion_permit,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_permit_retirement import (
    DISPOSITION_HASH_FIELD, retire_odom_startup_rejected_permit,
)
from scripts.aufgabe04.navigation.execution.startup_route_rejection_evidence import (
    validate_odom_startup_rejection_log,
)
from scripts.aufgabe04.navigation.localization.startup_route_admission import (
    OdomStartupRouteAdmissionRejected, ODOM_STARTUP_ROUTE_REJECTION_REASON,
    build_odom_startup_route_admission_evidence,
)
from scripts.aufgabe04.navigation.localization.candidate_planning_pose import admitted_candidate_planning_pose
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg, poses_from_waypoints
from tests.aufgabe04.test_startup_reseal_motion_consumption import StartupResealMotionConsumptionTest
from tests.aufgabe04 import test_startup_reseal_motion_authorization as authorization_fixtures
from tests.aufgabe04.test_startup_route_admission import FIXTURE, recorded_evidence


class StartupResealPermitRetirementTests(unittest.TestCase):
    def setUp(self):
        self.replacement = StartupResealMotionConsumptionTest()
        self.replacement.setUp()
        self.addCleanup(self.replacement.tearDown)
        self.root = self.replacement.root
        self.run_id = self.replacement.permit.rejected_run_id
        self.identity = dict(
            session_id="mission-001", rejected_run_id=self.run_id,
            mission_leg_kind=self.replacement.permit.mission_leg_kind.value,
            mission_leg_index=self.replacement.permit.mission_leg_index,
            target_id=self.replacement.permit.target_id, reseal_index=1,
        )
        fixture = json.loads(FIXTURE.read_text())
        self.preflight = fixture["stages"]["execute"]["preflight"]
        self.preflight_path = self.root / "failed_execute_preflight.json"
        self.preflight_path.write_text(json.dumps(self.preflight))
        self.route_path = self.root / "old_route.csv"
        rows = ["leg_index,point_index,world_x_m,world_y_m,yaw_rad,cumulative_length_m"]
        for index, pose in enumerate(fixture["map_route"]):
            yaw = "" if pose["yaw_rad"] is None else str(pose["yaw_rad"])
            rows.append(f"0,{index},{pose['x_m']},{pose['y_m']},{yaw},{index * .2}")
        self.route_path.write_text("\n".join(rows) + "\n")
        self.certificate_path = self.root / "old_map_certificate.json"
        self.certificate = ExecutionRouteCertificate(
            route_sha256=file_sha256(self.route_path), planning_frame="map",
            route_kind="stand_discovery", waypoint_count=len(fixture["map_route"]),
            tracking_tube_radius_m=.03, exact_vertex_pursuit=True, command_owner="/cmd_vel",
        )
        write_execution_route_certificate(self.certificate_path, self.certificate)
        self.evidence = recorded_evidence()
        self.evidence["source_map_execution_certificate_sha256"] = execution_route_certificate_sha256(self.certificate)
        self.log_path = self.root / "typed_rejection.jsonl"
        self._write_log()
        auth = self.replacement.authorization
        self.old_master_path = self.root / "old_routine_master.json"
        old_master = MissionLegMotionAuthorization(
            **{name: getattr(auth, name) for name in (
                "session_id", "robot_id", "namespace", "cmd_vel_topic", "semantic_map_id",
                "localization_branch_proof_id",
            )},
            allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS,
            scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, operator_confirmation="RUN",
        )
        old_master_sha = write_mission_leg_motion_authorization(self.old_master_path, old_master)
        route_bindings = {}
        for name in ("diagnostics", "dry_preflight", "dry_odom_certificate", "dry_uncertainty_budget"):
            path = self.replacement.artifacts[name]
            route_bindings[f"{name}_path"] = str(path)
            route_bindings[f"{name}_sha256"] = file_sha256(path)
        self.old_permit = MissionLegMotionPermit(
            master_authorization_sha256=old_master_sha,
            master_authorization_path=str(self.old_master_path),
            **{name: getattr(auth, name) for name in (
                "session_id", "robot_id", "namespace", "cmd_vel_topic", "semantic_map_id",
                "localization_branch_proof_id",
            )},
            run_id=self.run_id, mission_leg_kind=self.replacement.permit.mission_leg_kind,
            mission_leg_index=self.identity["mission_leg_index"], target_id=self.identity["target_id"],
            route_csv_path=str(self.route_path), route_csv_sha256=file_sha256(self.route_path),
            map_route_certificate_path=str(self.certificate_path),
            map_route_certificate_sha256=file_sha256(self.certificate_path),
            **route_bindings, dry_run_passed=True, additional_typed_run_required=False,
        )
        self.old_permit_path = self.root / "old_routine_permit.json"
        write_mission_leg_motion_permit(self.old_permit_path, self.old_permit)

    def _events(self, *, dry_run=False):
        details = OdomStartupRouteAdmissionRejected(evidence=self.evidence, dry_run=dry_run).to_stop_details()
        start = dict(event="run_started", run_id=self.run_id, leg_index=0, dry_run=dry_run,
                     authoritative_route_csv=str(self.route_path), preflight_json_path=str(self.preflight_path))
        events = [start]
        for name in ("odom_execution_admission_failed", "safety_stop", "run_finished"):
            event = dict(event=name, run_id=self.run_id, status="preflight_failed",
                         motion_published=False, stop_reason=ODOM_STARTUP_ROUTE_REJECTION_REASON,
                         stop_details=details, dry_run=dry_run,
                         **{key: self.identity[key] for key in ("mission_leg_kind", "mission_leg_index", "target_id")})
            if name == "run_finished":
                event.update(final_status="preflight_failed", preflight_json_path=str(self.preflight_path),
                             map_route_certificate_json_path=str(self.certificate_path))
            events.append(event)
        return events

    def _write_log(self, events=None, *, dry_run=False):
        self.log_path.write_text("\n".join(json.dumps(event) for event in (events or self._events(dry_run=dry_run))) + "\n")

    def _retire(self, **changes):
        values = dict(permit_path=self.old_permit_path, permit_kind="mission_leg",
                      expected_permit_sha256=payload_sha256(self.old_permit.to_payload()),
                      rejected_semantic_log_path=self.log_path, **self.identity)
        values.update(changes)
        return retire_odom_startup_rejected_permit(**values)

    def _consume_old(self, path=None):
        return consume_mission_leg_motion_permit(
            permit_path=path or self.old_permit_path, permit=self.old_permit,
            session_id=self.identity["session_id"], run_id=self.run_id,
            mission_leg_kind=self.old_permit.mission_leg_kind,
            mission_leg_index=self.old_permit.mission_leg_index, target_id=self.old_permit.target_id,
        )

    def test_retirement_occupies_old_claim_slot_without_false_consumption(self):
        path = self._retire()
        self.assertEqual(path, default_mission_leg_motion_consumption_receipt_path(self.old_permit_path))
        payload = load_content_hashed_json(path, hash_field=DISPOSITION_HASH_FIELD)
        self.assertEqual(payload["disposition"], "retired_before_motion")
        self.assertFalse(payload["motion_authorization_consumed"])
        self.assertEqual(path, self._retire())
        with self.assertRaises(ValueError):
            load_mission_leg_motion_consumption_receipt(path)
        copy = self.root / "copied_permit.json"; shutil.copyfile(self.old_permit_path, copy)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume_old(copy)

    def test_consume_retire_race_allows_exactly_one_terminal_state(self):
        def attempt(action):
            try:
                action(); return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(attempt, action) for action in (self._retire, self._consume_old)]
            self.assertEqual(sum(future.result() for future in futures), 1)

    def test_already_consumed_and_wrong_run_target_or_digest_reject(self):
        for change in (
            {"rejected_run_id": "another-run"}, {"target_id": "another-target"},
            {"session_id": "another-session"}, {"expected_permit_sha256": "f" * 64},
            {"permit_path": None, "permit_kind": None, "expected_permit_sha256": ""},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self._retire(**change)
        self._consume_old()
        with self.assertRaises(ValueError):
            self._retire()

    def test_successful_dry_history_does_not_hide_execute_rejection(self):
        events = self._events()
        prior = [deepcopy(events[0]), dict(event="run_finished", run_id=self.run_id, final_status="dry_run_ok")]
        self._write_log(prior + events)
        self._retire()

    def test_generic_duplicate_incomplete_cross_target_or_consumed_history_reject(self):
        original = self._events()
        mutations = [original[:-1], original + [deepcopy(original[-1])]]
        wrong = deepcopy(original); wrong[1]["target_id"] = "neighbor"; mutations.append(wrong)
        generic = deepcopy(original)
        for event in generic[1:]: event["stop_details"] = {"source": "odom_execution_admission"}
        mutations.append(generic)
        consumed = [dict(event="mission_leg_motion_permit_consumed", run_id=self.run_id)] + original
        mutations.append(consumed)
        for events in mutations:
            with self.subTest(events=events):
                self._write_log(events)
                with self.assertRaises(ValueError): self._retire()

    def test_source_pass_payload_and_original_certificate_are_revalidated(self):
        altered = deepcopy(self.preflight); altered["ok"] = False
        self.preflight_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "source preflight"):
            self._retire()
        self.preflight_path.write_text(json.dumps(self.preflight))
        self.route_path.write_text(self.route_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "certified route binding"):
            self._retire()

    def test_dry_rejection_requires_explicit_no_permit_disposition(self):
        self._write_log(dry_run=True)
        with self.assertRaises(ValueError): self._retire()
        with self.assertRaisesRegex(ValueError, "mission session"):
            self._retire(permit_path=None, permit_kind=None, expected_permit_sha256="",
                         session_id="other-mission", disposition_path=self.root / "wrong_dry.json")
        path = self._retire(permit_path=None, permit_kind=None, expected_permit_sha256="",
                            disposition_path=self.root / "dry_no_permit.json")
        self.assertEqual(load_content_hashed_json(path, hash_field=DISPOSITION_HASH_FIELD)["disposition"], "no_permit_issued")

    def test_replacement_issues_and_consumes_only_with_bound_terminal_disposition(self):
        path, permit = self._issue_replacement()
        self.replacement._consume(permit_path=path, permit=permit)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.replacement._consume(permit_path=path, permit=permit)

    def _issue_replacement(self, *, kind=None, fresh_pose=None):
        disposition = self._retire()
        summary_path = self.root / "typed_summary.json"
        summary = json.loads(self.replacement.artifacts["startup_reseal_summary"].read_text())
        summary.update(recovery_source_kind=STARTUP_RESEAL_RECOVERY_SOURCE_ODOM_STARTUP_ROUTE_MISMATCH,
                       rejected_permit_disposition_path=str(disposition),
                       rejected_permit_disposition_sha256=file_sha256(disposition))
        if fresh_pose is not None:
            summary.update(fresh_start_pose=fresh_pose,
                           fresh_start_pose_basis="latest_direct_map_from_odom_composed_with_odom_base")
        if kind is not None:
            summary["mission_leg_kind"] = kind.value
        summary_path.write_text(json.dumps(summary))
        next_route = self.replacement.artifacts["route_csv"]
        next_certificate_path = self.root / "replacement_map_certificate.json"
        next_certificate = replace(self.certificate, route_sha256=file_sha256(next_route))
        write_execution_route_certificate(next_certificate_path, next_certificate)
        permit = replace(self.replacement.permit,
                         mission_leg_kind=kind or self.replacement.permit.mission_leg_kind,
                         recovery_source_kind=STARTUP_RESEAL_RECOVERY_SOURCE_ODOM_STARTUP_ROUTE_MISMATCH,
                         rejected_semantic_log_path=str(self.log_path), rejected_semantic_log_sha256=file_sha256(self.log_path),
                         startup_reseal_summary_path=str(summary_path), startup_reseal_summary_sha256=file_sha256(summary_path),
                         map_route_certificate_path=str(next_certificate_path),
                         map_route_certificate_sha256=file_sha256(next_certificate_path),
                         route_csv_sha256=file_sha256(next_route),
                         diagnostics_sha256=file_sha256(self.replacement.artifacts["diagnostics"]))
        path = self.root / "typed_replacement.json"
        write_startup_reseal_motion_permit(path, permit)
        return path, permit

    def test_repeated_startup_rejection_retires_replacement_and_preserves_index(self):
        path, permit = self._issue_replacement()
        self.run_id = permit.run_id
        self.identity.update(rejected_run_id=self.run_id, reseal_index=2)
        self.route_path = Path(permit.route_csv_path)
        self.certificate_path = Path(permit.map_route_certificate_path)
        route = load_route_leg(self.route_path, 0, require_motion=False)
        values = dict(self.evidence)
        self.evidence = build_odom_startup_route_admission_evidence(
            **{key: values[key] for key in (
                "odom_pose", "chained_map_pose", "map_from_odom", "pose_tf_observations", "map_frame",
                "odom_frame", "base_frame", "tracking_tube_radius_m", "max_tf_age_sec",
                "max_composition_yaw_error_rad", "source_preflight_sha256",
            )},
            map_route=poses_from_waypoints(route.executable_waypoints),
            source_map_execution_certificate_sha256=payload_sha256(
                {key: value for key, value in json.loads(self.certificate_path.read_text()).items()
                 if key != "execution_route_certificate_sha256"}),
        )
        self.log_path = self.root / "second_typed_rejection.jsonl"; self._write_log()
        args = dict(permit_path=path, permit_kind="startup_reseal",
                    expected_permit_sha256=payload_sha256(permit.to_payload()))
        with self.assertRaisesRegex(ValueError, "cumulative reseal index"):
            self._retire(**args, reseal_index=3)
        disposition = self._retire(**args)
        self.assertEqual(load_content_hashed_json(disposition, hash_field=DISPOSITION_HASH_FIELD)["replacement_startup_reseal_index"], 2)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.replacement._consume(permit_path=path, permit=permit)

    def test_candidate_full_issue_and_claim_uses_captured_composed_pose(self):
        kind = MissionLegKind.CANDIDATE_PREAPPROACH
        self.identity["mission_leg_kind"] = kind.value
        self.old_permit = replace(self.old_permit, mission_leg_kind=kind)
        self.old_permit_path = self.root / "candidate_routine_permit.json"
        write_mission_leg_motion_permit(self.old_permit_path, self.old_permit)
        self._write_log()
        fresh = deepcopy(self.preflight)
        # Model a new explicit stopped-localization request using saved captures.
        fresh["preflight_requirements"]["stationary_map_from_odom_pairing_requested"] = True
        pose, _ = admitted_candidate_planning_pose(fresh, map_frame="map", odom_frame="odom")
        pose_dict = dict(x_m=pose.x_m, y_m=pose.y_m, yaw_rad=pose.yaw_rad)
        self.assertNotEqual(pose_dict, {key: fresh["route_pose"][key] for key in pose_dict})
        fresh_path = self.root / "candidate_fresh.json"; fresh_path.write_text(json.dumps(fresh))
        auth = replace(self.replacement.authorization, allowed_mission_leg_kinds=ROUTINE_MISSION_LEG_KINDS)
        master_path = self.root / "candidate_startup_master.json"
        master_sha = write_startup_reseal_motion_authorization(master_path, auth)
        self.replacement.permit = replace(
            self.replacement.permit, master_authorization_path=str(master_path),
            master_authorization_sha256=master_sha,
            fresh_stationary_localization_evidence_path=str(fresh_path),
            fresh_stationary_localization_evidence_sha256=file_sha256(fresh_path),
        )
        authorization_fixtures.StartupResealMotionAuthorizationTest._write_route_and_diagnostics(
            self.replacement, route_start_pose=pose_dict, exact_start_pose=pose_dict,
            provenance_pose=pose_dict,
        )
        path, permit = self._issue_replacement(kind=kind, fresh_pose=pose_dict)
        self.replacement._consume(permit_path=path, permit=permit)
        legacy = replace(permit, recovery_source_kind=STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH)
        _validate_fresh_stationary_localization_evidence(
            legacy, expected_route_pose=(pose.x_m, pose.y_m, pose.yaw_rad),
            fresh_start_pose_basis="latest_direct_map_from_odom_composed_with_odom_base",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_fresh_stationary_localization_evidence(legacy, expected_route_pose=(pose.x_m, pose.y_m, pose.yaw_rad))


if __name__ == "__main__":
    unittest.main()
