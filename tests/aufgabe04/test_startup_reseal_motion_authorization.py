import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION,
    STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD,
    STARTUP_RESEAL_MOTION_PERMIT_SCHEMA_VERSION,
    STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION,
    STARTUP_RESEAL_RECOVERY_KIND,
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
    STARTUP_RESEAL_RUN_CONFIRMATION,
    StartupResealMotionAuthorization,
    StartupResealMotionPermit,
    file_sha256,
    load_startup_reseal_motion_authorization,
    load_startup_reseal_motion_permit,
    startup_reseal_motion_authorization_sha256,
    startup_reseal_motion_permit_sha256,
    validate_startup_reseal_motion_authorization,
    validate_startup_reseal_motion_permit,
    validate_startup_reseal_motion_permit_for_execution,
    write_startup_reseal_motion_authorization,
    write_startup_reseal_motion_permit,
)
from scripts.aufgabe04.real_robot.autonomous_startup_reseal import (
    StartupResealPermitContext,
    write_startup_reseal_permit_summary,
)


class StartupResealMotionAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.master_path = self.root / "startup_master.json"
        self.permit_path = self.root / "startup_permit.json"
        self.authorization = StartupResealMotionAuthorization(
            session_id="mission-001",
            robot_id="tb3_0",
            namespace="",
            cmd_vel_topic="/tb3_0/cmd_vel",
            semantic_map_id="arena-map-v3",
            localization_branch_proof_id="amcl-proof-001",
            max_startup_reseals_per_leg=2,
            scope_text=STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
            operator_confirmation=STARTUP_RESEAL_RUN_CONFIRMATION,
            allowed_recovery_kind=STARTUP_RESEAL_RECOVERY_KIND,
        )
        self.master_sha256 = write_startup_reseal_motion_authorization(
            self.master_path,
            self.authorization,
        )
        self.artifacts = {}
        for name in (
            "fresh_stationary_localization_evidence",
            "route_csv",
            "diagnostics",
            "map_route_certificate",
            "dry_preflight",
            "dry_odom_certificate",
            "dry_uncertainty_budget",
        ):
            path = self.root / f"{name}.artifact"
            path.write_text(f"sealed {name}\n", encoding="utf-8")
            self.artifacts[name] = path
        self._write_route_and_diagnostics()
        self.artifacts["rejected_semantic_log"] = (
            self.root / "rejected_run.jsonl"
        )
        self._write_rejected_log()
        self.artifacts["startup_reseal_summary"] = (
            self.root / "startup_reseal_summary.json"
        )
        self._write_summary()
        self._write_fresh_localization_evidence()
        self.permit = self._new_permit()

    def tearDown(self):
        self.temporary.cleanup()

    def _path(self, name):
        return str(self.artifacts[name].absolute())

    def _sha(self, name):
        return file_sha256(self.artifacts[name])

    def _write_route_and_diagnostics(
        self,
        *,
        route_start_pose=None,
        exact_start_pose=None,
        provenance_pose=None,
        include_provenance=True,
        first_yaw="",
    ):
        route_start = dict(
            route_start_pose
            or {"x_m": 0.1, "y_m": 0.2, "yaw_rad": 0.3}
        )
        exact_start = dict(exact_start_pose or route_start)
        provenance = dict(provenance_pose or exact_start)
        anchor = {
            "x_m": 0.2,
            "y_m": 0.2,
            "yaw_rad": exact_start["yaw_rad"],
        }
        length_m = (
            (anchor["x_m"] - exact_start["x_m"]) ** 2
            + (anchor["y_m"] - exact_start["y_m"]) ** 2
        ) ** 0.5
        sample_count = 21
        sample_spacing_m = length_m / (sample_count - 1)
        minimum_sampled_clearance_m = 0.5
        minimum_continuous_clearance_m = (
            minimum_sampled_clearance_m - sample_spacing_m / 2.0
        )
        inflation_radius_m = 0.01
        self.artifacts["route_csv"].write_text(
            "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
            "yaw_rad,segment_length_m,cumulative_length_m\n"
            f"0,0,0,0,{route_start['x_m']},{route_start['y_m']},"
            f"{first_yaw},0.0,0.0\n"
            f"0,1,1,0,{anchor['x_m']},{anchor['y_m']},0.3,"
            f"{length_m},{length_m}\n",
            encoding="utf-8",
        )
        metadata = {
            "planning_frame": "map",
            "inflation_radius_m": inflation_radius_m,
            "exact_start_connector": {
                "required": True,
                "validated": True,
                "exact_start": exact_start,
                "anchor": anchor,
                "connector_length_m": length_m,
                "required_clearance_m": inflation_radius_m,
                "minimum_sampled_clearance_m": (
                    minimum_sampled_clearance_m
                ),
                "minimum_continuous_clearance_m": (
                    minimum_continuous_clearance_m
                ),
                "minimum_margin_m": (
                    minimum_continuous_clearance_m - inflation_radius_m
                ),
                "sample_spacing_m": sample_spacing_m,
                "sample_count": sample_count,
            },
        }
        if include_provenance:
            metadata["route_start_pose_provenance"] = {
                "source": "autonomous_candidate_current_pose",
                "planning_frame": "map",
                "pose": provenance,
            }
        self.artifacts["diagnostics"].write_text(
            json.dumps({"metadata": metadata}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def _write_rejected_log(self, *extra_events, **event_replacements):
        event = {
            "timestamp": "2026-08-17T12:00:00+00:00",
            "event": "startup_route_rejected",
            "run_id": "mission-001-coverage-003",
            "leg_index": 0,
            "coverage_leg_index": 3,
            "target_viewpoint_id": "survey-vp-007",
            "status": "stopped",
            "stop_reason": "pose outside certified startup segment",
            "motion_published": False,
            "stop_details": {
                "reason": "pose outside certified startup segment",
                "source": "execution_route_certificate",
                "phase": "before_motion_confirmation",
                "route_pose": {"x_m": 0.1, "y_m": 0.2, "yaw_rad": 0.3},
                "fail_closed": True,
            },
        }
        event.update(event_replacements)
        events = (event, *extra_events)
        self.artifacts["rejected_semantic_log"].write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
            encoding="utf-8",
        )

    def _write_prestart_rejected_log(
        self,
        *extra_events,
        **event_replacements,
    ):
        reason = "global localization consistency requires zero and reseal"
        details = {
            "reason": reason,
            "fault_code": "localization_reseal_required",
            "source": "global_consistency_monitor",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "monitor_action": "FORCE_ZERO_RESEAL",
            "monitor_reason": "reseal_required",
            "monitor_warning": "",
            "motion_published": False,
            "execution_phase": "before_motion",
            "phase": "initial_runtime_input_wait",
            "continuity": {
                "schema_version": 1,
                "accepted": False,
                "decision": "force_zero_reseal",
                "reason": "map_from_odom_translation_drift",
                "fail_closed": True,
                "requires_zero_cycle": True,
                "requires_reseal": True,
                "threshold_semantics": (
                    "accept_if_observed_less_than_or_equal_to_limit"
                ),
                "certificate_sha256": "a" * 64,
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_footprint",
                "frozen_map_from_odom": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "yaw_rad": 0.0,
                },
                "live_map_from_odom": {
                    "x_m": 0.2,
                    "y_m": 0.0,
                    "yaw_rad": 0.0,
                },
                "relative_translation_x_m": 0.2,
                "relative_translation_y_m": 0.0,
                "translation_drift_m": 0.2,
                "relative_yaw_rad": 0.0,
                "absolute_yaw_drift_rad": 0.0,
                "max_translation_drift_m": 0.1,
                "max_yaw_drift_rad": 0.1,
                "validation_error": None,
            },
            "fail_closed": True,
        }
        event = {
            "timestamp": "2026-08-19T14:09:01+00:00",
            "event": "safety_stop",
            "run_id": "mission-001-coverage-003",
            "leg_index": 0,
            "coverage_leg_index": 3,
            "target_viewpoint_id": "survey-vp-007",
            "status": "stopped",
            "stop_reason": reason,
            "motion_published": False,
            "stop_details": details,
        }
        event.update(event_replacements)
        permit_consumed = {
            "timestamp": "2026-08-19T14:09:00+00:00",
            "event": "mission_leg_motion_permit_consumed",
            "run_id": "mission-001-coverage-003",
            "leg_index": 0,
            "mission_leg_kind": "coverage",
            "mission_leg_index": 3,
            "target_id": "survey-vp-007",
            "coverage_leg_index": 3,
            "target_viewpoint_id": "survey-vp-007",
            "covered_by_initial_mission_run": True,
            "additional_typed_run_required": False,
        }
        motion_started = {
            "timestamp": "2026-08-19T14:09:00.500000+00:00",
            "event": "motion_started",
            "run_id": "mission-001-coverage-003",
            "leg_index": 0,
            "coverage_leg_index": 3,
            "target_viewpoint_id": "survey-vp-007",
            "motion_published": False,
            "event_semantics": (
                "child_execution_attempt_started_before_follower"
            ),
            "resolved_cmd_vel_topic": "/tb3_0/cmd_vel",
        }
        events = (permit_consumed, motion_started, event, *extra_events)
        self.artifacts["rejected_semantic_log"].write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
            encoding="utf-8",
        )

    def _write_summary(self, **replacements):
        summary = {
            "schema_version": STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION,
            "status": "startup_route_replanned",
            "motion_published": False,
            "reseal_kind": "startup",
            "leg_index": 3,
            "mission_leg_kind": MissionLegKind.COVERAGE.value,
            "mission_leg_index": 3,
            "startup_reseal_index": 1,
            "rejected_run_id": "mission-001-coverage-003",
            "target_viewpoint_id": "survey-vp-007",
            "target_id": "survey-vp-007",
            "fresh_start_pose": {"x_m": 0.1, "y_m": 0.2, "yaw_rad": 0.3},
            "route_csv": self._path("route_csv"),
            "diagnostics_json": self._path("diagnostics"),
            "same_target_verified": True,
            "additional_typed_run_required": False,
            "recovery_source_kind": (
                STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
            ),
        }
        summary.update(replacements)
        self.artifacts["startup_reseal_summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_fresh_localization_evidence(self, **replacements):
        pose = {"x_m": 0.1, "y_m": 0.2, "yaw_rad": 0.3}
        evidence = {
            "ok": True,
            "failures": [],
            "observations": [
                {
                    "name": "stationary AMCL stability",
                    "ok": True,
                    "detail": "samples=2/2",
                    "data": {
                        "sample_count": 2,
                        "required_sample_count": 2,
                        "service_request_count": 2,
                        "position_covariance_complete": True,
                        "yaw_covariance_complete": True,
                    },
                }
            ],
            "runtime_config": {
                "localization_source": "amcl",
                "use_sim_time": False,
            },
            "route_pose": {
                "frame_id": "map",
                "child_frame_id": "base_footprint",
                **pose,
            },
            "odom_pose": None,
            "map_from_odom": None,
            "stationary_amcl_samples": [
                {**pose, "covariance": [0.0] * 36},
                {**pose, "covariance": [0.0] * 36},
            ],
            "stationary_map_from_odom_samples": [],
        }
        evidence.update(replacements)
        self.artifacts["fresh_stationary_localization_evidence"].write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _new_permit(self, **replacements):
        values = {
            "master_authorization_sha256": self.master_sha256,
            "master_authorization_path": str(self.master_path.absolute()),
            "run_id": "mission-001-coverage-003-startup-reseal-001",
            "leg_index": 3,
            "target_viewpoint_id": "survey-vp-007",
            "reseal_index": 1,
            "max_startup_reseals_per_leg": 2,
            "rejected_run_id": "mission-001-coverage-003",
            "rejected_semantic_log_path": self._path(
                "rejected_semantic_log"
            ),
            "rejected_semantic_log_sha256": self._sha(
                "rejected_semantic_log"
            ),
            "startup_reseal_summary_path": self._path(
                "startup_reseal_summary"
            ),
            "startup_reseal_summary_sha256": self._sha(
                "startup_reseal_summary"
            ),
            "fresh_stationary_localization_evidence_path": self._path(
                "fresh_stationary_localization_evidence"
            ),
            "fresh_stationary_localization_evidence_sha256": self._sha(
                "fresh_stationary_localization_evidence"
            ),
            "route_csv_path": self._path("route_csv"),
            "route_csv_sha256": self._sha("route_csv"),
            "diagnostics_path": self._path("diagnostics"),
            "diagnostics_sha256": self._sha("diagnostics"),
            "map_route_certificate_path": self._path(
                "map_route_certificate"
            ),
            "map_route_certificate_sha256": self._sha(
                "map_route_certificate"
            ),
            "dry_preflight_path": self._path("dry_preflight"),
            "dry_preflight_sha256": self._sha("dry_preflight"),
            "dry_odom_certificate_path": self._path("dry_odom_certificate"),
            "dry_odom_certificate_sha256": self._sha(
                "dry_odom_certificate"
            ),
            "dry_uncertainty_budget_path": self._path(
                "dry_uncertainty_budget"
            ),
            "dry_uncertainty_budget_sha256": self._sha(
                "dry_uncertainty_budget"
            ),
            "same_target_verified": True,
            "rejected_motion_published": False,
            "dry_run_passed": True,
            "additional_typed_run_required": False,
            "recovery_source_kind": (
                STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
            ),
        }
        values.update(replacements)
        return StartupResealMotionPermit(**values)

    def _execution_kwargs(self):
        return {
            "master_authorization_path": self.master_path,
            "run_id": self.permit.run_id,
            "session_id": self.authorization.session_id,
            "robot_id": self.authorization.robot_id,
            "namespace": self.authorization.namespace,
            "cmd_vel_topic": self.authorization.cmd_vel_topic,
            "semantic_map_id": self.authorization.semantic_map_id,
            "target_viewpoint_id": self.permit.target_viewpoint_id,
            "leg_index": self.permit.leg_index,
            "localization_branch_proof_id": (
                self.authorization.localization_branch_proof_id
            ),
            "route_csv_path": self.artifacts["route_csv"],
            "diagnostics_path": self.artifacts["diagnostics"],
            "map_route_certificate_path": self.artifacts[
                "map_route_certificate"
            ],
        }

    def test_master_is_frozen_hashed_round_trip_and_live_scope_bound(self):
        loaded = load_startup_reseal_motion_authorization(self.master_path)
        self.assertEqual(loaded, self.authorization)
        self.assertEqual(
            self.master_sha256,
            startup_reseal_motion_authorization_sha256(self.authorization),
        )
        with self.assertRaises(FrozenInstanceError):
            loaded.robot_id = "other"
        validated = validate_startup_reseal_motion_authorization(
            self.master_path,
            session_id=self.authorization.session_id,
            robot_id=self.authorization.robot_id,
            namespace=self.authorization.namespace,
            cmd_vel_topic=self.authorization.cmd_vel_topic,
            semantic_map_id=self.authorization.semantic_map_id,
            localization_branch_proof_id=(
                self.authorization.localization_branch_proof_id
            ),
        )
        self.assertEqual(validated, loaded)
        self.assertEqual(
            loaded.schema_version,
            STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION,
        )
        self.assertEqual(STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION, 3)
        self.assertEqual(STARTUP_RESEAL_MOTION_PERMIT_SCHEMA_VERSION, 3)
        self.assertIn("certified start pose mismatch", loaded.scope_text)
        self.assertIn("prestart localization-continuity", loaded.scope_text)
        with self.assertRaisesRegex(ValueError, "unsupported.*schema"):
            replace(self.authorization, schema_version=1)
        with self.assertRaisesRegex(ValueError, "unsupported.*schema"):
            replace(self.permit, schema_version=1)

    def test_candidate_identity_is_bound_and_cannot_collide_with_other_kinds(self):
        candidate_master = self.root / "candidate_startup_master.json"
        candidate_authorization = replace(
            self.authorization,
            allowed_mission_leg_kinds=(
                MissionLegKind.COVERAGE,
                MissionLegKind.CANDIDATE_PREAPPROACH,
                MissionLegKind.OPPOSITE_FACE,
            ),
        )
        candidate_master_sha = write_startup_reseal_motion_authorization(
            candidate_master,
            candidate_authorization,
        )
        self._write_rejected_log(
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH.value,
            mission_leg_index=3,
            target_id="survey-vp-007",
            coverage_leg_index=None,
            target_viewpoint_id="",
        )
        self._write_summary(
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH.value,
            mission_leg_index=3,
            target_id="survey-vp-007",
        )
        candidate = self._new_permit(
            master_authorization_path=str(candidate_master.absolute()),
            master_authorization_sha256=candidate_master_sha,
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=3,
            target_id="survey-vp-007",
            rejected_semantic_log_sha256=self._sha(
                "rejected_semantic_log"
            ),
            startup_reseal_summary_sha256=self._sha(
                "startup_reseal_summary"
            ),
        )
        candidate_path = self.root / "candidate_startup_permit.json"
        write_startup_reseal_motion_permit(candidate_path, candidate)
        kwargs = {
            **self._execution_kwargs(),
            "master_authorization_path": candidate_master,
            "mission_leg_kind": MissionLegKind.CANDIDATE_PREAPPROACH,
            "mission_leg_index": 3,
            "target_id": "survey-vp-007",
        }
        loaded = validate_startup_reseal_motion_permit_for_execution(
            candidate_path,
            **kwargs,
        )
        self.assertEqual(
            loaded.mission_leg_kind,
            MissionLegKind.CANDIDATE_PREAPPROACH,
        )
        opposite = replace(
            candidate,
            mission_leg_kind=MissionLegKind.OPPOSITE_FACE,
        )
        self.assertNotEqual(
            startup_reseal_motion_permit_sha256(candidate),
            startup_reseal_motion_permit_sha256(opposite),
        )

        mismatches = {
            "mission_leg_kind": MissionLegKind.OPPOSITE_FACE,
            "mission_leg_index": 4,
            "target_id": "other-target",
        }
        for name, value in mismatches.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, rf"{name}.*mismatch"
            ):
                validate_startup_reseal_motion_permit_for_execution(
                    candidate_path,
                    **{**kwargs, name: value},
                )

        self._write_route_and_diagnostics(include_provenance=False)
        without_provenance = replace(
            candidate,
            route_csv_sha256=self._sha("route_csv"),
            diagnostics_sha256=self._sha("diagnostics"),
        )
        with self.assertRaisesRegex(ValueError, "provenance is required"):
            write_startup_reseal_motion_permit(
                self.root / "candidate-without-start-provenance.json",
                without_provenance,
            )

    def test_coverage_default_master_does_not_authorize_candidate_reseal(self):
        candidate = self._new_permit(
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=3,
            target_id="survey-vp-007",
        )
        with self.assertRaisesRegex(ValueError, "kind is not authorized"):
            write_startup_reseal_motion_permit(
                self.root / "disallowed_candidate.json",
                candidate,
            )
    def test_master_rejects_wrong_scope_run_kind_and_budget(self):
        cases = (
            ("scope_text", "generic", "scope_text mismatch"),
            ("operator_confirmation", "yes", "confirmation RUN"),
            ("allowed_recovery_kind", "generic", "recovery kind mismatch"),
            ("max_startup_reseals_per_leg", -1, "non-negative integer"),
            (
                "allowed_mission_leg_kinds",
                (MissionLegKind.COVERAGE, MissionLegKind.COVERAGE),
                "duplicates",
            ),
            (
                "allowed_mission_leg_kinds",
                (MissionLegKind.STARTUP_RESEAL,),
                "routine mission leg kind",
            ),
            (
                "allowed_mission_leg_kinds",
                (
                    MissionLegKind.OPPOSITE_FACE,
                    MissionLegKind.CANDIDATE_PREAPPROACH,
                ),
                "canonical order",
            ),
        )
        for name, value, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                replace(self.authorization, **{name: value})

    def test_permit_round_trip_and_execution_validation_rehash_everything(self):
        stored = write_startup_reseal_motion_permit(
            self.permit_path,
            self.permit,
        )
        loaded = load_startup_reseal_motion_permit(self.permit_path)
        self.assertEqual(stored, startup_reseal_motion_permit_sha256(loaded))
        self.assertEqual(loaded, self.permit)
        self.assertEqual(
            validate_startup_reseal_motion_permit_for_execution(
                self.permit_path,
                **self._execution_kwargs(),
            ),
            self.permit,
        )

        original = self.artifacts["fresh_stationary_localization_evidence"].read_bytes()
        self.artifacts["fresh_stationary_localization_evidence"].write_bytes(
            original + b"tampered"
        )
        with self.assertRaisesRegex(ValueError, "localization_evidence hash mismatch"):
            validate_startup_reseal_motion_permit_for_execution(
                self.permit_path,
                **self._execution_kwargs(),
            )

    def test_permit_requires_exact_flags_budget_and_distinct_replacement(self):
        cases = (
            ("same_target_verified", False, "same_target_verified=true"),
            (
                "rejected_motion_published",
                True,
                "rejected_motion_published=false",
            ),
            ("dry_run_passed", False, "dry_run_passed=true"),
            (
                "additional_typed_run_required",
                True,
                "additional_typed_run_required=false",
            ),
            ("reseal_index", 0, "positive integer"),
            ("reseal_index", 3, "budget exceeded"),
            (
                "run_id",
                self.permit.rejected_run_id,
                "must differ from replacement run_id",
            ),
            (
                "recovery_source_kind",
                "generic_stop",
                "recovery_source_kind is not authorized",
            ),
        )
        for name, value, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                replace(self.permit, **{name: value})

    def test_rejected_log_must_prove_exact_pre_motion_rejection(self):
        invalid = (
            (
                {"run_id": "other-run"},
                (),
                "exactly one same-run",
            ),
            (
                {"coverage_leg_index": 4},
                (),
                "exactly one same-run",
            ),
            (
                {"target_viewpoint_id": "other-target"},
                (),
                "exactly one same-run",
            ),
            (
                {"motion_published": True},
                (),
                "published or started motion",
            ),
            (
                {},
                (
                    {
                        "event": "motion_started",
                        "run_id": "mission-001-coverage-003",
                    },
                ),
                "published or started motion",
            ),
            (
                {},
                (
                    {
                        "event": "motion_completed",
                        "run_id": "mission-001-coverage-003",
                    },
                ),
                "published or started motion",
            ),
            (
                {
                    "stop_details": {
                        "reason": "pose outside certified startup segment",
                        "source": "execution_route_certificate",
                        "phase": "after_motion",
                        "fail_closed": True,
                    }
                },
                (),
                "exactly one same-run",
            ),
        )
        for replacements, extras, message in invalid:
            with self.subTest(replacements=replacements):
                self._write_rejected_log(*extras, **replacements)
                candidate = self._new_permit()
                with self.assertRaisesRegex(ValueError, message):
                    write_startup_reseal_motion_permit(
                        self.root / f"bad-log-{len(message)}.json",
                        candidate,
                    )

    def test_prestart_localization_source_round_trips_only_with_eligible_stop(self):
        source_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )
        self._write_prestart_rejected_log()
        self._write_summary(recovery_source_kind=source_kind)
        candidate = self._new_permit(recovery_source_kind=source_kind)

        path = self.root / "prestart-permit.json"
        write_startup_reseal_motion_permit(path, candidate)
        self.assertEqual(
            load_startup_reseal_motion_permit(path).recovery_source_kind,
            source_kind,
        )
        self.assertEqual(
            validate_startup_reseal_motion_permit_for_execution(
                path,
                **{
                    **self._execution_kwargs(),
                    "run_id": candidate.run_id,
                },
            ),
            candidate,
        )

        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop(STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD)
        raw["recovery_source_kind"] = (
            STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
        )
        raw[STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD] = payload_sha256(raw)
        tampered = self.root / "prestart-permit-cross-kind-tamper.json"
        tampered.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "published or started motion"):
            validate_startup_reseal_motion_permit_for_execution(
                tampered,
                **{
                    **self._execution_kwargs(),
                    "run_id": candidate.run_id,
                },
            )

    def test_prestart_tf_warmup_source_also_requires_a_new_sealed_permit(self):
        source_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )
        self._write_prestart_rejected_log()
        events = [
            json.loads(line)
            for line in self.artifacts["rejected_semantic_log"]
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        event = events[-1]
        details = event["stop_details"]
        details["monitor_warning"] = "stale_map_from_odom"
        continuity = details["continuity"]
        continuity.update(
            {
                "reason": "map_from_odom_missing",
                "live_map_from_odom": None,
                "relative_translation_x_m": None,
                "relative_translation_y_m": None,
                "translation_drift_m": None,
                "relative_yaw_rad": None,
                "absolute_yaw_drift_rad": None,
                "validation_error": "live map_from_odom is missing",
            }
        )
        self.artifacts["rejected_semantic_log"].write_text(
            "".join(json.dumps(item) + "\n" for item in events),
            encoding="utf-8",
        )
        self._write_summary(recovery_source_kind=source_kind)
        candidate = self._new_permit(recovery_source_kind=source_kind)

        path = self.root / "prestart-warmup-permit.json"
        write_startup_reseal_motion_permit(path, candidate)
        self.assertEqual(
            load_startup_reseal_motion_permit(path).recovery_source_kind,
            source_kind,
        )

    def test_prestart_requires_exact_consumed_started_stopped_sequence(self):
        source_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )
        self._write_prestart_rejected_log()
        base_events = [
            json.loads(line)
            for line in self.artifacts["rejected_semantic_log"]
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self._write_summary(recovery_source_kind=source_kind)

        missing_consumption = base_events[1:]
        missing_started = [base_events[0], base_events[2]]
        wrong_order = [base_events[1], base_events[0], base_events[2]]
        duplicate_started = [
            base_events[0],
            base_events[1],
            dict(base_events[1]),
            base_events[2],
        ]
        motion_completed = [
            *base_events,
            {
                "event": "motion_completed",
                "run_id": "mission-001-coverage-003",
            },
        ]
        wrong_consumption_identity = json.loads(json.dumps(base_events))
        wrong_consumption_identity[0]["coverage_leg_index"] = 4
        wrong_started_identity = json.loads(json.dumps(base_events))
        wrong_started_identity[1]["target_viewpoint_id"] = "other-target"
        wrong_started_semantics = json.loads(json.dumps(base_events))
        wrong_started_semantics[1]["event_semantics"] = "nonzero_motion_started"
        wrong_scope = json.loads(json.dumps(base_events))
        wrong_scope[0]["covered_by_initial_mission_run"] = False

        cases = (
            (
                "missing-consumption",
                missing_consumption,
                "exactly one same-run motion permit consumption",
            ),
            (
                "missing-started",
                missing_started,
                "exactly one same-run child execution attempt",
            ),
            ("wrong-order", wrong_order, "event ordering mismatch"),
            (
                "duplicate-started",
                duplicate_started,
                "exactly one same-run child execution attempt",
            ),
            (
                "motion-completed",
                motion_completed,
                "completed or published motion",
            ),
            (
                "wrong-consumption-identity",
                wrong_consumption_identity,
                "motion permit consumption identity mismatch",
            ),
            (
                "wrong-started-identity",
                wrong_started_identity,
                "child execution attempt identity mismatch",
            ),
            (
                "wrong-started-semantics",
                wrong_started_semantics,
                "child execution-attempt semantics mismatch",
            ),
            ("wrong-scope", wrong_scope, "motion permit scope mismatch"),
        )
        for name, events, message in cases:
            with self.subTest(name=name):
                self.artifacts["rejected_semantic_log"].write_text(
                    "".join(json.dumps(item) + "\n" for item in events),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    write_startup_reseal_motion_permit(
                        self.root / f"prestart-sequence-{name}.json",
                        self._new_permit(recovery_source_kind=source_kind),
                    )

    def test_prestart_accepts_each_exact_prior_coverage_permit_family(self):
        source_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )
        self._write_prestart_rejected_log()
        base_events = [
            json.loads(line)
            for line in self.artifacts["rejected_semantic_log"]
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self._write_summary(recovery_source_kind=source_kind)

        for prior_event_name in (
            "mission_leg_motion_permit_consumed",
            "startup_reseal_motion_permit_consumed",
            "runtime_localization_motion_permit_consumed",
        ):
            with self.subTest(prior_event_name=prior_event_name):
                events = json.loads(json.dumps(base_events))
                events[0]["event"] = prior_event_name
                if prior_event_name == "startup_reseal_motion_permit_consumed":
                    events[0]["recovery_source_kind"] = (
                        STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
                    )
                self.artifacts["rejected_semantic_log"].write_text(
                    "".join(json.dumps(item) + "\n" for item in events),
                    encoding="utf-8",
                )
                write_startup_reseal_motion_permit(
                    self.root / f"prestart-prior-{prior_event_name}.json",
                    self._new_permit(recovery_source_kind=source_kind),
                )

    def test_recovery_source_kinds_cannot_be_cross_applied(self):
        prestart_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )

        self._write_rejected_log()
        self._write_summary(recovery_source_kind=prestart_kind)
        with self.assertRaisesRegex(
            ValueError,
            "eligible prestart localization-continuity safety stop",
        ):
            write_startup_reseal_motion_permit(
                self.root / "pose-as-prestart.json",
                self._new_permit(recovery_source_kind=prestart_kind),
            )

        self._write_prestart_rejected_log()
        self._write_summary()
        with self.assertRaisesRegex(ValueError, "published or started motion"):
            write_startup_reseal_motion_permit(
                self.root / "prestart-as-pose.json",
                self._new_permit(),
            )

    def test_prestart_source_rejects_motion_conflict_and_summary_kind_tamper(self):
        source_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )
        self._write_prestart_rejected_log(motion_published=True)
        self._write_summary(recovery_source_kind=source_kind)
        with self.assertRaisesRegex(ValueError, "completed or published motion"):
            write_startup_reseal_motion_permit(
                self.root / "prestart-motion-conflict.json",
                self._new_permit(recovery_source_kind=source_kind),
            )

        self._write_prestart_rejected_log()
        self._write_summary(
            recovery_source_kind=(
                STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "summary recovery_source_kind mismatch",
        ):
            write_startup_reseal_motion_permit(
                self.root / "prestart-summary-cross-kind.json",
                self._new_permit(recovery_source_kind=source_kind),
            )

        for identity_field, value in (
            ("coverage_leg_index", 4),
            ("target_viewpoint_id", "other-target"),
        ):
            with self.subTest(identity_field=identity_field):
                self._write_prestart_rejected_log(**{identity_field: value})
                self._write_summary(recovery_source_kind=source_kind)
                with self.assertRaisesRegex(
                    ValueError,
                    "eligible prestart localization-continuity safety stop",
                ):
                    write_startup_reseal_motion_permit(
                        self.root / f"prestart-wrong-{identity_field}.json",
                        self._new_permit(recovery_source_kind=source_kind),
                    )

    def test_adapter_binds_recovery_source_to_context_and_summary(self):
        source_kind = (
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        )
        summary_path = self.root / "adapter-summary.json"
        write_startup_reseal_permit_summary(
            summary_path,
            leg_index=3,
            target_viewpoint_id="survey-vp-007",
            reseal_index=1,
            rejected_run_id="mission-001-coverage-003",
            fresh_start_x_m=0.1,
            fresh_start_y_m=0.2,
            fresh_start_yaw_rad=0.3,
            route_csv=self.artifacts["route_csv"],
            diagnostics_json=self.artifacts["diagnostics"],
            recovery_source_kind=source_kind,
        )
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            stored["schema_version"],
            STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION,
        )
        self.assertEqual(stored["recovery_source_kind"], source_kind)

        context = StartupResealPermitContext(
            mission_authorization_json=self.master_path,
            session_id=self.authorization.session_id,
            semantic_map_id=self.authorization.semantic_map_id,
            leg_index=3,
            target_viewpoint_id="survey-vp-007",
            reseal_index=1,
            max_startup_reseals_per_leg=2,
            rejected_run_id="mission-001-coverage-003",
            rejected_semantic_log_path=self.artifacts[
                "rejected_semantic_log"
            ],
            startup_reseal_summary_path=summary_path,
            fresh_localization_evidence_path=self.artifacts[
                "fresh_stationary_localization_evidence"
            ],
            permit_json_path=self.root / "adapter-permit.json",
            recovery_source_kind=source_kind,
        )
        self.assertEqual(context.recovery_source_kind, source_kind)
        with self.assertRaisesRegex(ValueError, "not authorized"):
            replace(context, recovery_source_kind="generic_stop")

    def test_summary_binds_no_motion_reseal_rejected_run_target_and_route(self):
        cases = (
            ("motion_published", True),
            ("leg_index", 4),
            ("startup_reseal_index", 2),
            ("rejected_run_id", "other-run"),
            ("target_viewpoint_id", "other-target"),
            ("route_csv", self._path("diagnostics")),
            ("diagnostics_json", self._path("route_csv")),
            ("same_target_verified", False),
            ("additional_typed_run_required", True),
            ("status", "generic_replan"),
        )
        for name, value in cases:
            with self.subTest(name=name):
                self._write_rejected_log()
                self._write_summary(**{name: value})
                candidate = self._new_permit()
                with self.assertRaisesRegex(ValueError, rf"summary {name} mismatch"):
                    write_startup_reseal_motion_permit(
                        self.root / f"bad-summary-{name}.json",
                        candidate,
                    )

    def test_fresh_localization_evidence_is_semantically_bound(self):
        cases = (
            ({"ok": False}, "was not admitted"),
            ({"failures": ["stale AMCL"]}, "was not admitted"),
            ({"route_pose": None}, "route_pose is missing"),
            (
                {
                    "route_pose": {
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": 9.0,
                        "y_m": 0.2,
                        "yaw_rad": 0.3,
                    }
                },
                "does not match the startup reseal summary",
            ),
            (
                {"stationary_amcl_samples": []},
                "lacks an AMCL sample window",
            ),
            (
                {"observations": []},
                "lacks an admitted stationary AMCL observation",
            ),
        )
        for index, (replacements, message) in enumerate(cases):
            with self.subTest(replacements=replacements):
                self._write_rejected_log()
                self._write_summary()
                self._write_fresh_localization_evidence(**replacements)
                candidate = self._new_permit()
                with self.assertRaisesRegex(ValueError, message):
                    write_startup_reseal_motion_permit(
                        self.root / f"bad-localization-{index}.json",
                        candidate,
                    )

        self.artifacts["fresh_stationary_localization_evidence"].write_text(
            "not JSON\n",
            encoding="utf-8",
        )
        candidate = self._new_permit()
        with self.assertRaisesRegex(ValueError, "invalid startup reseal JSON"):
            write_startup_reseal_motion_permit(
                self.root / "bad-localization-malformed.json",
                candidate,
            )

    def test_fresh_pose_is_bound_to_replacement_route_content(self):
        cases = (
            (
                {
                    "route_start_pose": {
                        "x_m": 0.11,
                        "y_m": 0.2,
                        "yaw_rad": 0.3,
                    },
                    "exact_start_pose": {
                        "x_m": 0.11,
                        "y_m": 0.2,
                        "yaw_rad": 0.3,
                    },
                    "provenance_pose": {
                        "x_m": 0.11,
                        "y_m": 0.2,
                        "yaw_rad": 0.3,
                    },
                },
                "replacement exact start differs",
            ),
            (
                {
                    "exact_start_pose": {
                        "x_m": 0.1,
                        "y_m": 0.2,
                        "yaw_rad": 0.31,
                    },
                    "provenance_pose": {
                        "x_m": 0.1,
                        "y_m": 0.2,
                        "yaw_rad": 0.31,
                    },
                },
                "replacement exact start differs",
            ),
            (
                {
                    "provenance_pose": {
                        "x_m": 0.1,
                        "y_m": 0.2,
                        "yaw_rad": 0.31,
                    },
                },
                "provenance differs from fresh stationary pose",
            ),
            (
                {"first_yaw": "0.31"},
                "waypoint 0 yaw differs",
            ),
        )
        for index, (route_replacements, message) in enumerate(cases):
            with self.subTest(route_replacements=route_replacements):
                self._write_route_and_diagnostics(**route_replacements)
                candidate = self._new_permit()
                with self.assertRaisesRegex(ValueError, message):
                    write_startup_reseal_motion_permit(
                        self.root / f"bad-route-binding-{index}.json",
                        candidate,
                    )

    def test_coverage_route_may_omit_start_provenance_but_keeps_connector(self):
        self._write_route_and_diagnostics(include_provenance=False)
        candidate = self._new_permit()
        path = self.root / "coverage-without-start-provenance.json"

        write_startup_reseal_motion_permit(path, candidate)

        self.assertEqual(load_startup_reseal_motion_permit(path), candidate)

    def test_full_validator_rejects_substitution_and_false_supplied_hash(self):
        write_startup_reseal_motion_permit(self.permit_path, self.permit)
        kwargs = {
            **self._execution_kwargs(),
            "rejected_semantic_log_path": self.artifacts[
                "rejected_semantic_log"
            ],
            "rejected_semantic_log_sha256": self._sha(
                "rejected_semantic_log"
            ),
            "startup_reseal_summary_path": self.artifacts[
                "startup_reseal_summary"
            ],
            "startup_reseal_summary_sha256": self._sha(
                "startup_reseal_summary"
            ),
            "fresh_stationary_localization_evidence_path": self.artifacts[
                "fresh_stationary_localization_evidence"
            ],
            "fresh_stationary_localization_evidence_sha256": self._sha(
                "fresh_stationary_localization_evidence"
            ),
            "route_csv_sha256": self._sha("route_csv"),
            "diagnostics_sha256": self._sha("diagnostics"),
            "map_route_certificate_sha256": self._sha(
                "map_route_certificate"
            ),
            "dry_preflight_path": self.artifacts["dry_preflight"],
            "dry_preflight_sha256": self._sha("dry_preflight"),
            "dry_odom_certificate_path": self.artifacts[
                "dry_odom_certificate"
            ],
            "dry_odom_certificate_sha256": self._sha(
                "dry_odom_certificate"
            ),
            "dry_uncertainty_budget_path": self.artifacts[
                "dry_uncertainty_budget"
            ],
            "dry_uncertainty_budget_sha256": self._sha(
                "dry_uncertainty_budget"
            ),
        }
        kwargs["route_csv_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "route_csv supplied hash mismatch"):
            validate_startup_reseal_motion_permit(self.permit_path, **kwargs)

        replacement = self.root / "replacement-route.csv"
        replacement.write_bytes(self.artifacts["route_csv"].read_bytes())
        kwargs["route_csv_sha256"] = self._sha("route_csv")
        kwargs["route_csv_path"] = replacement
        with self.assertRaisesRegex(ValueError, "route_csv path mismatch"):
            validate_startup_reseal_motion_permit(self.permit_path, **kwargs)

    def test_symlink_noncanonical_and_unknown_stored_fields_fail_closed(self):
        write_startup_reseal_motion_permit(self.permit_path, self.permit)
        link = self.root / "permit-link.json"
        link.symlink_to(self.permit_path)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            validate_startup_reseal_motion_permit_for_execution(
                link,
                **self._execution_kwargs(),
            )

        with self.assertRaisesRegex(ValueError, "canonical absolute"):
            replace(
                self.permit,
                route_csv_path=str(
                    self.root / "unused" / ".." / "route_csv.artifact"
                ),
            )

        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw.pop("startup_reseal_motion_permit_sha256")
        raw["unexpected"] = True
        raw["startup_reseal_motion_permit_sha256"] = payload_sha256(raw)
        altered = self.root / "extra-field.json"
        altered.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "permit fields mismatch"):
            load_startup_reseal_motion_permit(altered)


if __name__ == "__main__":
    unittest.main()
