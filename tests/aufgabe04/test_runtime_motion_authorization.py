import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
    MISSION_MOTION_AUTHORIZATION_SCOPE,
    MISSION_RUN_CONFIRMATION,
    RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND,
    MissionMotionAuthorization,
    RuntimeLocalizationMotionPermit,
    file_sha256,
    load_mission_motion_authorization,
    load_runtime_localization_motion_permit,
    mission_motion_authorization_sha256,
    runtime_localization_motion_permit_sha256,
    validate_mission_motion_authorization,
    validate_runtime_localization_motion_permit,
    validate_runtime_localization_motion_permit_for_execution,
    write_mission_motion_authorization,
    write_runtime_localization_motion_permit,
)


def _decision():
    return {
        "schema_version": 1,
        "eligible": True,
        "reason": "runtime_localization_reseal_required",
        "execution_phase": "after_motion",
        "motion_published": True,
        "continuity_reason": "map_from_odom_yaw_drift",
        "requires_fresh_localization": True,
        "requires_new_route_certificate": True,
        "requires_fresh_typed_run": True,
        "automatic_motion_authorized": False,
    }


class RuntimeMotionAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.master_path = self.root / "mission_authorization.json"
        self.permit_path = self.root / "runtime_permit.json"
        self.authorization = MissionMotionAuthorization(
            session_id="mission-session-001",
            robot_id="tb3_0",
            namespace="",
            cmd_vel_topic="/tb3_0/cmd_vel",
            semantic_map_id="arena-map-v3",
            localization_branch_proof_id="amcl-branch-proof-001",
            max_runtime_reseals_per_leg=2,
            scope_text=MISSION_MOTION_AUTHORIZATION_SCOPE,
            operator_confirmation=MISSION_RUN_CONFIRMATION,
            allowed_recovery_kind=RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND,
        )
        self.master_sha256 = write_mission_motion_authorization(
            self.master_path, self.authorization
        )
        self.artifacts = {}
        for name in (
            "fresh_localization_evidence",
            "route_csv",
            "diagnostics",
            "map_route_certificate",
            "dry_odom_certificate",
            "dry_uncertainty_budget",
            "dry_preflight",
        ):
            path = self.root / f"{name}.artifact"
            path.write_text(f"sealed {name}\n", encoding="utf-8")
            self.artifacts[name] = path
        decision = _decision()
        self.permit = RuntimeLocalizationMotionPermit(
            master_authorization_sha256=self.master_sha256,
            master_authorization_path=str(self.master_path.absolute()),
            run_id="mission-session-001-leg-3-reseal-1",
            leg_index=3,
            target_viewpoint_id="viewpoint-007",
            reseal_index=1,
            max_runtime_reseals_per_leg=2,
            rejected_run_id="mission-session-001-leg-3-attempt-0",
            runtime_reseal_decision_evidence=decision,
            runtime_reseal_decision_sha256=payload_sha256(decision),
            fresh_localization_evidence_path=self._path(
                "fresh_localization_evidence"
            ),
            fresh_localization_evidence_sha256=self._sha(
                "fresh_localization_evidence"
            ),
            route_csv_path=self._path("route_csv"),
            route_csv_sha256=self._sha("route_csv"),
            diagnostics_path=self._path("diagnostics"),
            diagnostics_sha256=self._sha("diagnostics"),
            map_route_certificate_path=self._path("map_route_certificate"),
            map_route_certificate_sha256=self._sha("map_route_certificate"),
            dry_odom_certificate_path=self._path("dry_odom_certificate"),
            dry_odom_certificate_sha256=self._sha("dry_odom_certificate"),
            dry_uncertainty_budget_path=self._path("dry_uncertainty_budget"),
            dry_uncertainty_budget_sha256=self._sha("dry_uncertainty_budget"),
            dry_preflight_path=self._path("dry_preflight"),
            dry_preflight_sha256=self._sha("dry_preflight"),
            same_target_verified=True,
            dry_run_passed=True,
            additional_typed_run_required=False,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _path(self, name):
        return str(self.artifacts[name].absolute())

    def _sha(self, name):
        return file_sha256(self.artifacts[name])

    def _write_permit(self):
        return write_runtime_localization_motion_permit(
            self.permit_path, self.permit
        )

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

    def test_master_round_trip_is_frozen_content_hashed_and_scope_bound(self):
        loaded = load_mission_motion_authorization(self.master_path)

        self.assertEqual(loaded, self.authorization)
        self.assertEqual(
            self.master_sha256,
            mission_motion_authorization_sha256(self.authorization),
        )
        with self.assertRaises(FrozenInstanceError):
            loaded.robot_id = "other"
        validated = validate_mission_motion_authorization(
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

    def test_master_rejects_wrong_run_confirmation_scope_kind_and_budget(self):
        replacements = (
            (
                {"operator_confirmation": "yes"},
                "requires operator confirmation RUN",
            ),
            ({"scope_text": "generic recovery"}, "scope_text mismatch"),
            ({"allowed_recovery_kind": "generic_stop"}, "recovery kind mismatch"),
            ({"max_runtime_reseals_per_leg": -1}, "must be a non-negative integer"),
        )
        for fields, message in replacements:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, message):
                    replace(self.authorization, **fields)

        disabled = replace(self.authorization, max_runtime_reseals_per_leg=0)
        self.assertEqual(disabled.max_runtime_reseals_per_leg, 0)
        self.assertEqual(self.authorization.namespace, "")
        disabled_path = self.root / "disabled-master.json"
        write_mission_motion_authorization(disabled_path, disabled)
        self.assertEqual(
            load_mission_motion_authorization(disabled_path), disabled
        )

    def test_master_validation_rejects_each_live_identity_mismatch(self):
        good = {
            "session_id": self.authorization.session_id,
            "robot_id": self.authorization.robot_id,
            "namespace": self.authorization.namespace,
            "cmd_vel_topic": self.authorization.cmd_vel_topic,
            "semantic_map_id": self.authorization.semantic_map_id,
            "localization_branch_proof_id": (
                self.authorization.localization_branch_proof_id
            ),
        }
        for name in good:
            with self.subTest(name=name):
                wrong = dict(good)
                wrong[name] = "wrong"
                with self.assertRaisesRegex(ValueError, rf"{name} mismatch"):
                    validate_mission_motion_authorization(self.master_path, **wrong)

    def test_permit_round_trip_hash_and_deep_immutable_decision(self):
        stored_sha256 = self._write_permit()
        loaded = load_runtime_localization_motion_permit(self.permit_path)

        self.assertEqual(
            stored_sha256, runtime_localization_motion_permit_sha256(loaded)
        )
        self.assertEqual(loaded.to_payload(), self.permit.to_payload())
        with self.assertRaises(TypeError):
            loaded.runtime_reseal_decision_evidence["eligible"] = False

    def test_write_is_idempotent_but_refuses_different_content(self):
        first = self._write_permit()
        second = self._write_permit()
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "refusing to replace immutable"):
            write_runtime_localization_motion_permit(
                self.permit_path,
                replace(self.permit, run_id="another-run"),
            )

    def test_permit_rejects_startup_generic_or_incomplete_decision(self):
        replacements = {
            "motion_published": False,
            "execution_phase": "before_motion",
            "reason": "generic_stop",
            "automatic_motion_authorized": True,
            "requires_new_route_certificate": False,
        }
        for field, value in replacements.items():
            with self.subTest(field=field):
                decision = _decision()
                decision[field] = value
                with self.assertRaisesRegex(
                    ValueError, rf"decision evidence {field} mismatch"
                ):
                    replace(
                        self.permit,
                        runtime_reseal_decision_evidence=decision,
                        runtime_reseal_decision_sha256=payload_sha256(decision),
                    )

    def test_permit_rejects_decision_hash_and_field_set_mismatch(self):
        with self.assertRaisesRegex(ValueError, "decision evidence hash mismatch"):
            replace(self.permit, runtime_reseal_decision_sha256="0" * 64)
        extra = _decision()
        extra["extra"] = True
        with self.assertRaisesRegex(ValueError, "decision evidence fields mismatch"):
            replace(
                self.permit,
                runtime_reseal_decision_evidence=extra,
                runtime_reseal_decision_sha256=payload_sha256(extra),
            )
        wrong_type = _decision()
        wrong_type["schema_version"] = 1.0
        with self.assertRaisesRegex(
            ValueError, "decision evidence schema_version mismatch"
        ):
            replace(
                self.permit,
                runtime_reseal_decision_evidence=wrong_type,
                runtime_reseal_decision_sha256=payload_sha256(wrong_type),
            )

    def test_permit_requires_same_target_dry_run_and_no_additional_typed_run(self):
        cases = (
            ("same_target_verified", False, "same_target_verified=true"),
            ("dry_run_passed", False, "dry_run_passed=true"),
            (
                "additional_typed_run_required",
                True,
                "additional_typed_run_required=false",
            ),
        )
        for name, value, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    replace(self.permit, **{name: value})

    def test_permit_budget_is_one_based_and_bound_to_master(self):
        for value in (0, 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reseal"):
                    replace(self.permit, reseal_index=value)
        with self.assertRaisesRegex(ValueError, "reseal maximum mismatch"):
            write_runtime_localization_motion_permit(
                self.root / "bad-budget.json",
                replace(
                    self.permit,
                    max_runtime_reseals_per_leg=3,
                ),
            )

    def test_execution_validator_accepts_exact_same_leg_same_target_scope(self):
        self._write_permit()
        validated = validate_runtime_localization_motion_permit_for_execution(
            self.permit_path, **self._execution_kwargs()
        )
        self.assertEqual(validated.to_payload(), self.permit.to_payload())

    def test_execution_validator_rejects_every_live_identity_mismatch(self):
        self._write_permit()
        kwargs = self._execution_kwargs()
        for name in (
            "run_id",
            "robot_id",
            "namespace",
            "cmd_vel_topic",
            "target_viewpoint_id",
            "localization_branch_proof_id",
            "session_id",
            "semantic_map_id",
        ):
            with self.subTest(name=name):
                wrong = dict(kwargs)
                wrong[name] = "wrong"
                with self.assertRaisesRegex(ValueError, rf"{name} mismatch"):
                    validate_runtime_localization_motion_permit_for_execution(
                        self.permit_path, **wrong
                    )
        wrong = dict(kwargs)
        wrong["leg_index"] += 1
        with self.assertRaisesRegex(ValueError, "leg_index mismatch"):
            validate_runtime_localization_motion_permit_for_execution(
                self.permit_path, **wrong
            )

    def test_execution_validator_rejects_live_path_substitution(self):
        self._write_permit()
        replacement_path = self.root / "replacement.csv"
        replacement_path.write_bytes(self.artifacts["route_csv"].read_bytes())
        kwargs = self._execution_kwargs()
        kwargs["route_csv_path"] = replacement_path
        with self.assertRaisesRegex(ValueError, "route_csv path mismatch"):
            validate_runtime_localization_motion_permit_for_execution(
                self.permit_path, **kwargs
            )

    def test_execution_validator_rehashes_live_and_hidden_dry_artifacts(self):
        self._write_permit()
        for name in (
            "route_csv",
            "diagnostics",
            "map_route_certificate",
            "fresh_localization_evidence",
            "dry_odom_certificate",
            "dry_uncertainty_budget",
            "dry_preflight",
        ):
            with self.subTest(name=name):
                original = self.artifacts[name].read_bytes()
                self.artifacts[name].write_bytes(original + b"tampered")
                with self.assertRaisesRegex(ValueError, rf"{name} hash mismatch"):
                    validate_runtime_localization_motion_permit_for_execution(
                        self.permit_path, **self._execution_kwargs()
                    )
                self.artifacts[name].write_bytes(original)

    def test_full_validator_rejects_supplied_hash_even_when_file_is_unchanged(self):
        self._write_permit()
        kwargs = self._execution_kwargs()
        full = {
            **kwargs,
            "route_csv_sha256": self._sha("route_csv"),
            "diagnostics_sha256": self._sha("diagnostics"),
            "map_route_certificate_sha256": self._sha("map_route_certificate"),
            "dry_odom_certificate_path": self.artifacts["dry_odom_certificate"],
            "dry_odom_certificate_sha256": self._sha("dry_odom_certificate"),
            "dry_uncertainty_budget_path": self.artifacts[
                "dry_uncertainty_budget"
            ],
            "dry_uncertainty_budget_sha256": self._sha(
                "dry_uncertainty_budget"
            ),
            "dry_preflight_path": self.artifacts["dry_preflight"],
            "dry_preflight_sha256": self._sha("dry_preflight"),
        }
        full["route_csv_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "route_csv supplied hash mismatch"):
            validate_runtime_localization_motion_permit(self.permit_path, **full)

    def test_missing_non_file_and_symlink_artifacts_fail_closed(self):
        missing = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "must be a normal file"):
            file_sha256(missing)
        with self.assertRaisesRegex(ValueError, "must be a normal file"):
            file_sha256(self.root)
        symlink = self.root / "route-link"
        symlink.symlink_to(self.artifacts["route_csv"])
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            file_sha256(symlink)

    def test_symlink_permit_and_master_fail_closed(self):
        self._write_permit()
        permit_link = self.root / "permit-link.json"
        permit_link.symlink_to(self.permit_path)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            validate_runtime_localization_motion_permit_for_execution(
                permit_link, **self._execution_kwargs()
            )
        master_link = self.root / "master-link.json"
        master_link.symlink_to(self.master_path)
        kwargs = self._execution_kwargs()
        kwargs["master_authorization_path"] = master_link
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            validate_runtime_localization_motion_permit_for_execution(
                self.permit_path, **kwargs
            )

    def test_corrupt_or_malformed_hashed_artifacts_fail_closed(self):
        self._write_permit()
        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw["run_id"] = "tampered"
        self.permit_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            load_runtime_localization_motion_permit(self.permit_path)

        malformed = self.root / "malformed.json"
        malformed.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid artifact JSON"):
            load_runtime_localization_motion_permit(malformed)

    def test_unknown_stored_fields_fail_closed_even_with_recomputed_hash(self):
        self._write_permit()
        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw.pop("runtime_localization_motion_permit_sha256")
        raw["unexpected"] = True
        raw["runtime_localization_motion_permit_sha256"] = payload_sha256(raw)
        altered = self.root / "extra-field.json"
        altered.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "permit fields mismatch"):
            load_runtime_localization_motion_permit(altered)


if __name__ == "__main__":
    unittest.main()
