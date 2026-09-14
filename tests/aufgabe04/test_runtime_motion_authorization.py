import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256, write_content_hashed_json
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.map_odom_drift_reference import RouteDriftAnchor
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    write_odom_execution_certificate,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext,
    evaluate_map_odom_continuity,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind,
)
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


def _anchored_decision():
    frozen = PlanarTransform2D(-5.0, 0.0, 0.0)
    context = OdomExecutionContext(
        map_frame="map", odom_frame="odom", base_frame="base_footprint",
        frozen_map_from_odom=frozen, certificate_sha256="a" * 64,
        max_map_from_odom_translation_drift_m=0.1,
        max_map_from_odom_yaw_drift_rad=0.1,
        drift_reference=RouteDriftAnchor.from_route_start(Pose2D(0.0, 0.0, 0.0), frozen),
    )
    result = evaluate_map_odom_continuity(context, PlanarTransform2D(-4.8, 0.0, 0.0))
    return {**_decision(), "schema_version": 2, "continuity_reason": result.reason,
            "continuity_evidence": result.to_evidence()}


def _dry_certificate(*, reference=None, budget_hash="b" * 64):
    return OdomExecutionCertificate(
        source_map_route_sha256="1" * 64,
        source_map_execution_certificate_sha256="2" * 64,
        transformed_odom_route_sha256="3" * 64,
        map_frame="map", odom_frame="odom", base_frame="base_footprint",
        map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
        transform_stamp_sec=10.0, transform_capture_time_sec=10.0,
        waypoint_count=2, tracking_tube_radius_m=0.15,
        command_owner="/follower", uncertainty_budget_sha256=budget_hash,
        ambiguity_evidence_sha256="c" * 64,
        schema_version=1 if reference is None else 2,
        drift_reference=reference,
    )


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
            allowed_mission_leg_kinds=(
                MissionLegKind.COVERAGE,
                MissionLegKind.CANDIDATE_PREAPPROACH,
                MissionLegKind.OPPOSITE_FACE,
            ),
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
        self.artifacts["dry_odom_certificate"].unlink()
        self.artifacts["dry_uncertainty_budget"].unlink()
        legacy_budget_hash = write_content_hashed_json(
            self.artifacts["dry_uncertainty_budget"],
            {"schema_version": 1, "runtime_map_odom_continuity_allocation": {}},
            hash_field="route_uncertainty_artifact_sha256",
        )
        write_odom_execution_certificate(
            self.artifacts["dry_odom_certificate"],
            _dry_certificate(budget_hash=legacy_budget_hash),
        )
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

    def _install_anchored_artifacts(self, *, budget_edit=None, certificate_budget_hash=None):
        reference = RouteDriftAnchor(1.0, 2.0, 1.0, 2.0)
        budget = {
            "schema_version": 2,
            "runtime_map_odom_continuity_allocation": {"drift_reference": reference.to_evidence()},
            "admission": {"config": {"heading_reference_x_m": 1.0, "heading_reference_y_m": 2.0}},
        }
        if budget_edit is not None:
            budget_edit(budget)
        self.artifacts["dry_uncertainty_budget"].unlink()
        budget_hash = write_content_hashed_json(
            self.artifacts["dry_uncertainty_budget"], budget,
            hash_field="route_uncertainty_artifact_sha256",
        )
        self.artifacts["dry_odom_certificate"].unlink()
        write_odom_execution_certificate(
            self.artifacts["dry_odom_certificate"],
            _dry_certificate(reference=reference, budget_hash=certificate_budget_hash or budget_hash),
        )
        decision = _anchored_decision()
        self.permit = replace(
            self.permit, runtime_reseal_decision_evidence=decision,
            runtime_reseal_decision_sha256=payload_sha256(decision),
            dry_odom_certificate_sha256=self._sha("dry_odom_certificate"),
            dry_uncertainty_budget_sha256=self._sha("dry_uncertainty_budget"),
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

    def test_legacy_coverage_authorization_and_permit_remain_loadable(self):
        legacy_authorization = replace(
            self.authorization,
            schema_version=1,
            allowed_mission_leg_kinds=(MissionLegKind.COVERAGE,),
        )
        legacy_master_path = self.root / "legacy-master.json"
        legacy_master_sha256 = write_mission_motion_authorization(
            legacy_master_path,
            legacy_authorization,
        )
        loaded_master = load_mission_motion_authorization(legacy_master_path)
        self.assertEqual(loaded_master, legacy_authorization)

        legacy_permit = replace(
            self.permit,
            schema_version=1,
            master_authorization_path=str(legacy_master_path.absolute()),
            master_authorization_sha256=legacy_master_sha256,
        )
        legacy_permit_path = self.root / "legacy-permit.json"
        write_runtime_localization_motion_permit(
            legacy_permit_path,
            legacy_permit,
        )
        loaded_permit = load_runtime_localization_motion_permit(
            legacy_permit_path
        )
        self.assertEqual(loaded_permit, legacy_permit)
        self.assertEqual(
            loaded_permit.mission_leg_kind,
            MissionLegKind.COVERAGE,
        )

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

    def test_candidate_runtime_permit_binds_generic_routine_identity(self):
        candidate_permit = replace(
            self.permit,
            run_id="mission-candidate-004-runtime-reseal-001",
            leg_index=4,
            target_viewpoint_id="survey-candidate-004",
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=4,
            target_id="survey-candidate-004",
        )
        candidate_path = self.root / "candidate-runtime-permit.json"
        write_runtime_localization_motion_permit(
            candidate_path,
            candidate_permit,
        )
        kwargs = self._execution_kwargs()
        kwargs.update(
            {
                "run_id": candidate_permit.run_id,
                "leg_index": 4,
                "target_viewpoint_id": "survey-candidate-004",
                "mission_leg_kind": MissionLegKind.CANDIDATE_PREAPPROACH,
                "mission_leg_index": 4,
                "target_id": "survey-candidate-004",
            }
        )

        loaded = validate_runtime_localization_motion_permit_for_execution(
            candidate_path,
            **kwargs,
        )

        self.assertEqual(
            loaded.mission_leg_kind,
            MissionLegKind.CANDIDATE_PREAPPROACH,
        )
        for field, value in (
            ("mission_leg_kind", MissionLegKind.OPPOSITE_FACE),
            ("mission_leg_index", 5),
            ("target_id", "another-candidate"),
        ):
            with self.subTest(field=field):
                wrong = dict(kwargs)
                wrong[field] = value
                with self.assertRaisesRegex(ValueError, rf"{field}.*mismatch"):
                    validate_runtime_localization_motion_permit_for_execution(
                        candidate_path,
                        **wrong,
                    )

    def test_master_leg_scope_rejects_candidate_runtime_permit(self):
        coverage_only = replace(
            self.authorization,
            allowed_mission_leg_kinds=(MissionLegKind.COVERAGE,),
        )
        coverage_master = self.root / "coverage-only-master.json"
        coverage_hash = write_mission_motion_authorization(
            coverage_master,
            coverage_only,
        )
        candidate_permit = replace(
            self.permit,
            master_authorization_path=str(coverage_master.absolute()),
            master_authorization_sha256=coverage_hash,
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
        )
        with self.assertRaisesRegex(ValueError, "kind is not authorized"):
            write_runtime_localization_motion_permit(
                self.root / "unauthorized-candidate.json",
                candidate_permit,
            )

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

    def test_anchored_recovery_round_trip_preserves_old_stop_and_new_route_references(self):
        self._install_anchored_artifacts()
        self._write_permit()
        loaded = validate_runtime_localization_motion_permit_for_execution(
            self.permit_path, **self._execution_kwargs()
        )
        decision = loaded.to_payload()["runtime_reseal_decision_evidence"]
        self.assertEqual(decision, _anchored_decision())
        # The rejected route started at (0, 0); the recovery starts at (1, 2).
        self.assertEqual(decision["continuity_evidence"]["drift_reference"]["map_anchor"],
                         {"x_m": 0.0, "y_m": 0.0})
        with self.assertRaises(TypeError):
            loaded.runtime_reseal_decision_evidence["continuity_evidence"]["drift_reference"]["map_anchor"]["x_m"] = 99
        decision["continuity_evidence"]["drift_reference"]["map_anchor"]["x_m"] = 99
        self.assertEqual(loaded.to_payload(), self.permit.to_payload())

    def test_anchored_decision_rejects_rehashed_tampered_or_downgraded_continuity(self):
        edits = (
            lambda value: value["continuity_evidence"].pop("drift_reference"),
            lambda value: value["continuity_evidence"].update(schema_version=1),
            lambda value: value["continuity_evidence"].update(translation_drift_m=0.0),
            lambda value: value["continuity_evidence"]["drift_reference"]["map_anchor"].update(x_m=10.0),
            lambda value: value.update(continuity_reason="map_from_odom_yaw_drift"),
            lambda value: value.update(schema_version=1),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                decision = _anchored_decision()
                edit(decision)
                with self.assertRaises(ValueError):
                    replace(self.permit, runtime_reseal_decision_evidence=decision,
                            runtime_reseal_decision_sha256=payload_sha256(decision))

    def test_anchored_recovery_cannot_downgrade_stop_to_legacy_decision(self):
        self._install_anchored_artifacts()
        decision = _decision()
        self.permit = replace(self.permit, runtime_reseal_decision_evidence=decision,
                              runtime_reseal_decision_sha256=payload_sha256(decision))
        with self.assertRaisesRegex(ValueError, "requires runtime reseal decision v2"):
            self._write_permit()
        # An externally resealed permit must fail the execution boundary too.
        write_content_hashed_json(self.permit_path, self.permit.to_payload(),
                                  hash_field="runtime_localization_motion_permit_sha256")
        with self.assertRaisesRegex(ValueError, "requires runtime reseal decision v2"):
            validate_runtime_localization_motion_permit_for_execution(
                self.permit_path, **self._execution_kwargs()
            )

    def test_anchored_recovery_cannot_use_legacy_dry_certificate(self):
        decision = _anchored_decision()
        self.permit = replace(self.permit, runtime_reseal_decision_evidence=decision,
                              runtime_reseal_decision_sha256=payload_sha256(decision))
        with self.assertRaisesRegex(ValueError, "cannot use a legacy dry certificate"):
            self._write_permit()

    def test_anchored_recovery_rejects_changed_budget_reference_even_when_rehashed(self):
        edits = (
            (lambda value: value.update(schema_version=1), "versions mismatch"),
            (lambda value: value["runtime_map_odom_continuity_allocation"].pop("drift_reference"), "drift_reference mismatch"),
            (lambda value: value["runtime_map_odom_continuity_allocation"]["drift_reference"]["map_anchor"].update(x_m=9.0), "drift_reference mismatch"),
            (lambda value: value["runtime_map_odom_continuity_allocation"]["drift_reference"]["map_anchor"].update(x_m=True), "drift_reference mismatch"),
            (lambda value: value["admission"]["config"].update(heading_reference_x_m=9.0), "heading reference mismatch"),
            (lambda value: value["admission"]["config"].update(heading_reference_x_m=True), "heading reference mismatch"),
            (lambda value: value["admission"]["config"].pop("heading_reference_y_m"), "heading reference mismatch"),
        )
        for edit, message in edits:
            with self.subTest(message=message):
                self._install_anchored_artifacts(budget_edit=edit)
                with self.assertRaisesRegex(ValueError, message):
                    self._write_permit()

    def test_anchored_recovery_requires_certificate_budget_content_binding(self):
        self._install_anchored_artifacts(certificate_budget_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "uncertainty budget hash mismatch"):
            self._write_permit()

    def test_legacy_certificate_cannot_authorize_anchored_budget_or_allocation(self):
        for budget_version in (1, 2):
            with self.subTest(budget_version=budget_version):
                self._install_anchored_artifacts(
                    budget_edit=lambda value: value.update(schema_version=budget_version)
                )
                budget = json.loads(self.artifacts["dry_uncertainty_budget"].read_text())
                budget_hash = budget.pop("route_uncertainty_artifact_sha256")
                self.artifacts["dry_odom_certificate"].unlink()
                write_odom_execution_certificate(
                    self.artifacts["dry_odom_certificate"], _dry_certificate(budget_hash=budget_hash)
                )
                decision = _decision()
                self.permit = replace(
                    self.permit,
                    runtime_reseal_decision_evidence=decision,
                    runtime_reseal_decision_sha256=payload_sha256(decision),
                    dry_odom_certificate_sha256=self._sha("dry_odom_certificate"),
                )
                with self.assertRaisesRegex(ValueError, "versions mismatch|cannot contain drift_reference"):
                    self._write_permit()

    def test_anchored_certificate_cannot_drop_reference_even_when_rehashed(self):
        self._install_anchored_artifacts()
        certificate_path = self.artifacts["dry_odom_certificate"]
        certificate = json.loads(certificate_path.read_text())
        certificate.pop("odom_execution_certificate_sha256")
        certificate.pop("drift_reference")
        certificate_path.unlink()
        write_content_hashed_json(certificate_path, certificate,
                                  hash_field="odom_execution_certificate_sha256")
        self.permit = replace(self.permit, dry_odom_certificate_sha256=self._sha("dry_odom_certificate"))
        with self.assertRaisesRegex(ValueError, "certificate fields mismatch"):
            self._write_permit()


if __name__ == "__main__":
    unittest.main()
