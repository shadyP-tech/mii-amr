import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_RECOVERY_KIND,
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

    def _write_rejected_log(self, *extra_events, **event_replacements):
        event = {
            "timestamp": "2026-08-17T12:00:00+00:00",
            "event": "startup_route_rejected",
            "run_id": "mission-001-coverage-003",
            "leg_index": 3,
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

    def _write_summary(self, **replacements):
        summary = {
            "schema_version": 1,
            "status": "startup_route_replanned",
            "motion_published": False,
            "reseal_kind": "startup",
            "leg_index": 3,
            "startup_reseal_index": 1,
            "rejected_run_id": "mission-001-coverage-003",
            "target_viewpoint_id": "survey-vp-007",
            "fresh_start_pose": {"x_m": 0.1, "y_m": 0.2, "yaw_rad": 0.3},
            "route_csv": self._path("route_csv"),
            "diagnostics_json": self._path("diagnostics"),
            "same_target_verified": True,
            "additional_typed_run_required": False,
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

    def test_master_rejects_wrong_scope_run_kind_and_budget(self):
        cases = (
            ("scope_text", "generic", "scope_text mismatch"),
            ("operator_confirmation", "yes", "confirmation RUN"),
            ("allowed_recovery_kind", "generic", "recovery kind mismatch"),
            ("max_startup_reseals_per_leg", -1, "non-negative integer"),
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
