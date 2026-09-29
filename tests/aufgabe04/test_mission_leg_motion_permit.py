import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256, write_content_hashed_json
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    LEGACY_CENTERING_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_POSITION_EPOCH_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MAX_RETURN_TO_START_LEGS,
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MISSION_LEG_MOTION_PERMIT_HASH_FIELD,
    MISSION_LEG_RUN_CONFIRMATION,
    RECOVERABLE_MISSION_LEG_KINDS,
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
    MissionLegMotionAuthorization,
    MissionLegMotionPermit,
    file_sha256,
    load_mission_leg_motion_authorization,
    load_mission_leg_motion_permit,
    mission_leg_motion_authorization_sha256,
    mission_leg_motion_permit_sha256,
    validate_mission_leg_motion_authorization,
    validate_mission_leg_motion_permit,
    validate_mission_leg_motion_permit_for_execution,
    write_mission_leg_motion_authorization,
    write_mission_leg_motion_permit,
)


class MissionLegMotionPermitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).absolute()
        self.master_path = self.root / "mission_leg_authorization.json"
        self.permit_path = self.root / "mission_leg_permit.json"
        self.authorization = MissionLegMotionAuthorization(
            session_id="mission-session-001",
            robot_id="tb3_0",
            namespace="",
            cmd_vel_topic="/tb3_0/cmd_vel",
            semantic_map_id="arena-map-v3",
            localization_branch_proof_id="amcl-branch-proof-001",
            allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS,
            scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            operator_confirmation=MISSION_LEG_RUN_CONFIRMATION,
        )
        self.master_sha256 = write_mission_leg_motion_authorization(
            self.master_path, self.authorization
        )
        self.artifacts = {}
        for name in (
            "route_csv",
            "diagnostics",
            "map_route_certificate",
            "dry_preflight",
            "dry_odom_certificate",
            "dry_uncertainty_budget",
        ):
            path = self.root / f"{name}.artifact"
            path.write_text(json.dumps({"metadata": {}}) if name == "diagnostics" else f"sealed {name}\n", encoding="utf-8")
            self.artifacts[name] = path
        self.permit = MissionLegMotionPermit(
            master_authorization_sha256=self.master_sha256,
            master_authorization_path=str(self.master_path),
            session_id=self.authorization.session_id,
            robot_id=self.authorization.robot_id,
            namespace=self.authorization.namespace,
            cmd_vel_topic=self.authorization.cmd_vel_topic,
            semantic_map_id=self.authorization.semantic_map_id,
            localization_branch_proof_id=(
                self.authorization.localization_branch_proof_id
            ),
            run_id="mission-session-001-coverage-003",
            mission_leg_kind=MissionLegKind.COVERAGE,
            mission_leg_index=3,
            target_id="viewpoint-007",
            route_csv_path=self._path("route_csv"),
            route_csv_sha256=self._sha("route_csv"),
            diagnostics_path=self._path("diagnostics"),
            diagnostics_sha256=self._sha("diagnostics"),
            map_route_certificate_path=self._path("map_route_certificate"),
            map_route_certificate_sha256=self._sha("map_route_certificate"),
            dry_preflight_path=self._path("dry_preflight"),
            dry_preflight_sha256=self._sha("dry_preflight"),
            dry_odom_certificate_path=self._path("dry_odom_certificate"),
            dry_odom_certificate_sha256=self._sha("dry_odom_certificate"),
            dry_uncertainty_budget_path=self._path("dry_uncertainty_budget"),
            dry_uncertainty_budget_sha256=self._sha(
                "dry_uncertainty_budget"
            ),
            dry_run_passed=True,
            additional_typed_run_required=False,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _path(self, name):
        return str(self.artifacts[name])

    def _sha(self, name):
        return file_sha256(self.artifacts[name])

    def _write_permit(self):
        return write_mission_leg_motion_permit(
            self.permit_path, self.permit
        )

    def _bind_return_stage(self, permit, *, final_stage=True, staged=True, changes=None):
        target = {"x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.5}
        evidence_path = self.root / "return-target.json"
        evidence_path.write_text(json.dumps({
            "qr_id": "Start", "candidate_uid": permit.target_id, "target_pose": target,
        }))
        metadata = {
            "route_kind": "admitted_candidate_pose", "route_purpose": "return_to_start",
            "selected_candidate_stand_id": permit.target_id,
            "route_csv_sha256": permit.route_csv_sha256,
            "selected_approach_pose": target if final_stage else {**target, "x_m": 0.5},
            "stored_start_target_pose": target,
            "target_evidence_json": str(evidence_path),
            "target_evidence_sha256": file_sha256(evidence_path),
        }
        if staged:
            metadata["return_to_start_stage"] = {
                "stage_index": permit.mission_leg_index, "final_stage": final_stage,
                "start_candidate_uid": permit.target_id,
            }
        metadata.update(changes or {})
        self.artifacts["diagnostics"].write_text(json.dumps({"metadata": metadata}))
        return replace(permit, diagnostics_sha256=self._sha("diagnostics"))

    def _execution_kwargs(self):
        return {
            "master_authorization_path": self.master_path,
            "session_id": self.permit.session_id,
            "robot_id": self.permit.robot_id,
            "namespace": self.permit.namespace,
            "cmd_vel_topic": self.permit.cmd_vel_topic,
            "semantic_map_id": self.permit.semantic_map_id,
            "localization_branch_proof_id": (
                self.permit.localization_branch_proof_id
            ),
            "run_id": self.permit.run_id,
            "mission_leg_kind": self.permit.mission_leg_kind,
            "mission_leg_index": self.permit.mission_leg_index,
            "target_id": self.permit.target_id,
            "route_csv_path": self.artifacts["route_csv"],
            "diagnostics_path": self.artifacts["diagnostics"],
            "map_route_certificate_path": self.artifacts[
                "map_route_certificate"
            ],
            "dry_preflight_path": self.artifacts["dry_preflight"],
            "dry_odom_certificate_path": self.artifacts[
                "dry_odom_certificate"
            ],
            "dry_uncertainty_budget_path": self.artifacts[
                "dry_uncertainty_budget"
            ],
        }

    def _full_kwargs(self):
        kwargs = self._execution_kwargs()
        for name in (
            "route_csv",
            "diagnostics",
            "map_route_certificate",
            "dry_preflight",
            "dry_odom_certificate",
            "dry_uncertainty_budget",
        ):
            kwargs[f"{name}_sha256"] = self._sha(name)
        return kwargs

    def test_master_round_trip_is_frozen_hashed_and_identity_bound(self):
        loaded = load_mission_leg_motion_authorization(self.master_path)

        self.assertEqual(loaded, self.authorization)
        self.assertEqual(
            self.master_sha256,
            mission_leg_motion_authorization_sha256(self.authorization),
        )
        self.assertIsInstance(loaded.allowed_leg_kinds, tuple)
        with self.assertRaises(FrozenInstanceError):
            loaded.robot_id = "other"
        validated = validate_mission_leg_motion_authorization(
            self.master_path,
            session_id=self.authorization.session_id,
            robot_id=self.authorization.robot_id,
            namespace=self.authorization.namespace,
            cmd_vel_topic=self.authorization.cmd_vel_topic,
            semantic_map_id=self.authorization.semantic_map_id,
            localization_branch_proof_id=(
                self.authorization.localization_branch_proof_id
            ),
            required_leg_kind="candidate_preapproach",
        )
        self.assertEqual(validated, loaded)

    def test_master_requires_exact_run_scope_and_canonical_routine_kinds(self):
        cases = (
            (
                {"operator_confirmation": "yes"},
                "requires operator confirmation RUN",
            ),
            ({"scope_text": "generic motion"}, "scope_text mismatch"),
            ({"allowed_leg_kinds": ()}, "at least one routine leg"),
            (
                {
                    "allowed_leg_kinds": (
                        MissionLegKind.CANDIDATE_PREAPPROACH,
                        MissionLegKind.COVERAGE,
                    )
                },
                "canonical routine order",
            ),
            (
                {
                    "allowed_leg_kinds": (
                        MissionLegKind.COVERAGE,
                        MissionLegKind.COVERAGE,
                    )
                },
                "must be unique",
            ),
            (
                {"allowed_leg_kinds": (MissionLegKind.STARTUP_RESEAL,)},
                "requires a separate typed RUN",
            ),
        )
        for fields, message in cases:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, message):
                    replace(self.authorization, **fields)

        with self.assertRaisesRegex(ValueError, "requires a separate typed RUN"):
            validate_mission_leg_motion_authorization(
                self.master_path,
                session_id=self.authorization.session_id,
                robot_id=self.authorization.robot_id,
                namespace=self.authorization.namespace,
                cmd_vel_topic=self.authorization.cmd_vel_topic,
                semantic_map_id=self.authorization.semantic_map_id,
                localization_branch_proof_id=(
                    self.authorization.localization_branch_proof_id
                ),
                required_leg_kind="startup_reseal",
            )

    def test_legacy_scope_hashes_remain_readable_without_return_authority(self):
        for index, scope in enumerate((
            LEGACY_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            LEGACY_CENTERING_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
        )):
            with self.subTest(scope=scope):
                legacy = replace(
                    self.authorization,
                    allowed_leg_kinds=RECOVERABLE_MISSION_LEG_KINDS,
                    scope_text=scope,
                )
                path = self.root / f"legacy-master-{index}.json"
                digest = write_mission_leg_motion_authorization(path, legacy)
                self.assertEqual(digest, payload_sha256(legacy.to_payload()))
                self.assertEqual(load_mission_leg_motion_authorization(path), legacy)
                with self.assertRaisesRegex(ValueError, "explicit return-to-Start scope"):
                    replace(legacy, allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS)

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
                    validate_mission_leg_motion_authorization(
                        self.master_path, **wrong
                    )

    def test_permit_round_trip_hash_and_frozen_leg_identity(self):
        stored_sha256 = self._write_permit()
        loaded = load_mission_leg_motion_permit(self.permit_path)

        self.assertEqual(
            stored_sha256, mission_leg_motion_permit_sha256(loaded)
        )
        self.assertEqual(loaded.to_payload(), self.permit.to_payload())
        self.assertIs(loaded.mission_leg_kind, MissionLegKind.COVERAGE)
        with self.assertRaises(FrozenInstanceError):
            loaded.target_id = "another-target"

    def test_each_routine_kind_can_be_bound_but_startup_reseal_cannot(self):
        cases = (
            (MissionLegKind.COVERAGE, "viewpoint-001"),
            (MissionLegKind.CANDIDATE_PREAPPROACH, "candidate-001"),
            (MissionLegKind.OPPOSITE_FACE, "candidate-001"),
            (MissionLegKind.RETURN_TO_START, "candidate-start"),
        )
        for index, (kind, target_id) in enumerate(cases):
            with self.subTest(kind=kind):
                permit = replace(
                    self.permit,
                    run_id=f"mission-leg-{index}",
                    mission_leg_kind=kind.value,
                    mission_leg_index=index,
                    target_id=target_id,
                )
                if kind is MissionLegKind.RETURN_TO_START:
                    permit = self._bind_return_stage(permit)
                path = self.root / f"permit-{index}.json"
                write_mission_leg_motion_permit(path, permit)
                self.assertIs(
                    load_mission_leg_motion_permit(path).mission_leg_kind,
                    kind,
                )

        with self.assertRaisesRegex(ValueError, "requires a separate typed RUN"):
            replace(self.permit, mission_leg_kind="startup_reseal")

    def test_opposite_checkpoint_requires_current_scope_and_same_candidate(self):
        self.artifacts['diagnostics'].write_text(json.dumps({'metadata': {
            'approach_bearing_mode': 'opposite-localization-checkpoint',
            'selected_candidate_stand_id': 'candidate-001',
        }}))
        permit = replace(self.permit, mission_leg_kind=MissionLegKind.OPPOSITE_FACE,
                         target_id='candidate-001', diagnostics_sha256=self._sha('diagnostics'))
        write_mission_leg_motion_permit(self.permit_path, permit)
        with self.assertRaisesRegex(ValueError, 'candidate differs'):
            write_mission_leg_motion_permit(self.root/'wrong-target.json', replace(permit,target_id='neighbor'))
        legacy = replace(self.authorization, scope_text=LEGACY_POSITION_EPOCH_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE)
        legacy_path = self.root/'legacy-master.json'
        digest = write_mission_leg_motion_authorization(legacy_path,legacy)
        old_permit = replace(permit,master_authorization_path=str(legacy_path),master_authorization_sha256=digest)
        with self.assertRaisesRegex(ValueError, 'explicit checkpoint mission scope'):
            write_mission_leg_motion_permit(self.root/'legacy-permit.json',old_permit)

    def test_permit_cannot_broaden_master_allowed_leg_kinds(self):
        narrow_master = replace(
            self.authorization,
            allowed_leg_kinds=(MissionLegKind.COVERAGE,),
        )
        narrow_path = self.root / "narrow-master.json"
        narrow_hash = write_mission_leg_motion_authorization(
            narrow_path, narrow_master
        )
        broadened = replace(
            self.permit,
            master_authorization_path=str(narrow_path),
            master_authorization_sha256=narrow_hash,
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            target_id="candidate-002",
        )

        with self.assertRaisesRegex(ValueError, "not authorized by master"):
            write_mission_leg_motion_permit(
                self.root / "broadened.json", broadened
            )

    def test_write_is_idempotent_but_refuses_different_content(self):
        first = self._write_permit()
        second = self._write_permit()
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "refusing to replace immutable"):
            write_mission_leg_motion_permit(
                self.permit_path,
                replace(self.permit, run_id="another-run"),
            )

    def test_permit_requires_passed_dry_run_and_no_additional_typed_run(self):
        cases = (
            ("dry_run_passed", False, "dry_run_passed=true"),
            (
                "additional_typed_run_required",
                True,
                "additional_typed_run_required=false",
            ),
            ("mission_leg_index", -1, "non-negative integer"),
            ("mission_leg_index", True, "non-negative integer"),
            ("target_id", " ", "non-empty canonical string"),
        )
        for name, value, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    replace(self.permit, **{name: value})

    def test_write_rejects_master_identity_or_hash_mismatch(self):
        for name in (
            "session_id",
            "robot_id",
            "namespace",
            "cmd_vel_topic",
            "semantic_map_id",
            "localization_branch_proof_id",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"{name} mismatch"):
                    write_mission_leg_motion_permit(
                        self.root / f"bad-{name}.json",
                        replace(self.permit, **{name: "wrong"}),
                    )
        with self.assertRaisesRegex(ValueError, "master authorization hash mismatch"):
            write_mission_leg_motion_permit(
                self.root / "bad-master-hash.json",
                replace(self.permit, master_authorization_sha256="0" * 64),
            )

    def test_execution_validator_accepts_exact_path_first_scope(self):
        self._write_permit()
        validated = validate_mission_leg_motion_permit_for_execution(
            self.permit_path, **self._execution_kwargs()
        )
        self.assertEqual(validated.to_payload(), self.permit.to_payload())

    def test_return_to_start_execution_binds_the_exact_admitted_target(self):
        self.permit = self._bind_return_stage(replace(
            self.permit,
            mission_leg_kind=MissionLegKind.RETURN_TO_START,
            mission_leg_index=0,
            target_id="candidate-start",
        ))
        self._write_permit()
        kwargs = self._execution_kwargs()
        self.assertEqual(
            validate_mission_leg_motion_permit_for_execution(self.permit_path, **kwargs),
            self.permit,
        )
        kwargs["target_id"] = "another-candidate"
        with self.assertRaisesRegex(ValueError, "target_id mismatch"):
            validate_mission_leg_motion_permit_for_execution(self.permit_path, **kwargs)

    def test_return_stage_indices_are_bounded_and_the_last_stage_must_be_final(self):
        base = replace(self.permit, mission_leg_kind=MissionLegKind.RETURN_TO_START)
        with self.assertRaisesRegex(ValueError, "must be less than 4"):
            replace(base, mission_leg_index=MAX_RETURN_TO_START_LEGS)
        for index in range(MAX_RETURN_TO_START_LEGS):
            permit = self._bind_return_stage(replace(base, mission_leg_index=index))
            write_mission_leg_motion_permit(self.root / f"final-stage-{index}.json", permit)
        intermediate = self._bind_return_stage(base, final_stage=False)
        with self.assertRaisesRegex(ValueError, "last return_to_start stage must reach"):
            write_mission_leg_motion_permit(self.root / "invalid-last-stage.json", intermediate)

    def test_single_return_legacy_scope_allows_only_final_index_zero(self):
        master = replace(self.authorization, scope_text=LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE)
        path = self.root / "single-return-master.json"
        digest = write_mission_leg_motion_authorization(path, master)
        self.assertEqual(load_mission_leg_motion_authorization(path), master)
        for index, final_stage, staged in ((0, True, False), (0, True, True), (0, False, True), (1, True, True)):
            with self.subTest(index=index, final_stage=final_stage, staged=staged):
                permit = self._bind_return_stage(replace(
                    self.permit, mission_leg_kind=MissionLegKind.RETURN_TO_START,
                    mission_leg_index=index, master_authorization_path=str(path),
                    master_authorization_sha256=digest,
                ), final_stage=final_stage, staged=staged)
                output = self.root / f"legacy-{index}-{final_stage}-{staged}.json"
                if index == 0 and final_stage:
                    write_mission_leg_motion_permit(output, permit)
                else:
                    with self.assertRaisesRegex(ValueError, "legacy single-return scope"):
                        write_mission_leg_motion_permit(output, permit)
                    # A manually sealed artifact cannot bypass the live child
                    # validator by avoiding the normal issue-time checks.
                    write_content_hashed_json(
                        output, permit.to_payload(), hash_field=MISSION_LEG_MOTION_PERMIT_HASH_FIELD,
                    )
                    kwargs = self._execution_kwargs()
                    kwargs.update(
                        master_authorization_path=path,
                        mission_leg_kind=MissionLegKind.RETURN_TO_START,
                        mission_leg_index=index,
                    )
                    with self.assertRaisesRegex(ValueError, "legacy single-return scope"):
                        validate_mission_leg_motion_permit_for_execution(output, **kwargs)

    def test_return_stage_target_and_index_must_match_route_metadata(self):
        base = replace(self.permit, mission_leg_kind=MissionLegKind.RETURN_TO_START, mission_leg_index=0)
        for changes, reason in (
            ({"selected_candidate_stand_id": "other-start"}, "selected_candidate_stand_id mismatch"),
            ({"return_to_start_stage": {"stage_index": 1, "final_stage": True, "start_candidate_uid": base.target_id}}, "stage identity mismatch"),
            ({"return_to_start_stage": {"stage_index": 0, "final_stage": True, "start_candidate_uid": "other-start"}}, "stage identity mismatch"),
            ({"selected_approach_pose": {"x_m": 0., "y_m": 0., "yaw_rad": 0.}}, "must target the admitted Start pose"),
        ):
            with self.subTest(changes=changes):
                permit = self._bind_return_stage(base, changes=changes)
                with self.assertRaisesRegex(ValueError, reason):
                    write_mission_leg_motion_permit(self.root / "mismatch.json", permit)

    def test_return_target_evidence_tamper_rejected_by_live_validation(self):
        self.permit = self._bind_return_stage(replace(
            self.permit, mission_leg_kind=MissionLegKind.RETURN_TO_START, mission_leg_index=0,
        ))
        self._write_permit()
        (self.root / "return-target.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "target evidence hash mismatch"):
            validate_mission_leg_motion_permit_for_execution(self.permit_path, **self._execution_kwargs())

    def test_execution_validator_rejects_every_live_identity_mismatch(self):
        self._write_permit()
        kwargs = self._execution_kwargs()
        for name in (
            "session_id",
            "robot_id",
            "namespace",
            "cmd_vel_topic",
            "semantic_map_id",
            "localization_branch_proof_id",
            "run_id",
            "target_id",
        ):
            with self.subTest(name=name):
                wrong = dict(kwargs)
                wrong[name] = "wrong"
                with self.assertRaisesRegex(ValueError, rf"{name} mismatch"):
                    validate_mission_leg_motion_permit_for_execution(
                        self.permit_path, **wrong
                    )

        wrong = dict(kwargs)
        wrong["mission_leg_kind"] = "candidate_preapproach"
        with self.assertRaisesRegex(ValueError, "mission_leg_kind mismatch"):
            validate_mission_leg_motion_permit_for_execution(
                self.permit_path, **wrong
            )
        wrong = dict(kwargs)
        wrong["mission_leg_index"] += 1
        with self.assertRaisesRegex(ValueError, "mission_leg_index mismatch"):
            validate_mission_leg_motion_permit_for_execution(
                self.permit_path, **wrong
            )

    def test_execution_validator_rejects_path_substitution_with_same_bytes(self):
        self._write_permit()
        replacement_path = self.root / "replacement.csv"
        replacement_path.write_bytes(self.artifacts["route_csv"].read_bytes())
        kwargs = self._execution_kwargs()
        kwargs["route_csv_path"] = replacement_path
        with self.assertRaisesRegex(ValueError, "route_csv path mismatch"):
            validate_mission_leg_motion_permit_for_execution(
                self.permit_path, **kwargs
            )

    def test_execution_validator_rehashes_all_six_artifacts(self):
        self._write_permit()
        for name in (
            "route_csv",
            "diagnostics",
            "map_route_certificate",
            "dry_preflight",
            "dry_odom_certificate",
            "dry_uncertainty_budget",
        ):
            with self.subTest(name=name):
                original = self.artifacts[name].read_bytes()
                self.artifacts[name].write_bytes(original + b"tampered")
                with self.assertRaisesRegex(ValueError, rf"{name} hash mismatch"):
                    validate_mission_leg_motion_permit_for_execution(
                        self.permit_path, **self._execution_kwargs()
                    )
                self.artifacts[name].write_bytes(original)

    def test_full_validator_rejects_supplied_hash_mismatch(self):
        self._write_permit()
        kwargs = self._full_kwargs()
        kwargs["dry_preflight_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "supplied hash mismatch"):
            validate_mission_leg_motion_permit(self.permit_path, **kwargs)

    def test_canonical_normal_path_and_symlink_gates_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "canonical absolute path"):
            replace(self.permit, route_csv_path="relative.csv")
        with self.assertRaisesRegex(ValueError, "must be a normal file"):
            file_sha256(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "must be a normal file"):
            file_sha256(self.root)
        route_link = self.root / "route-link"
        route_link.symlink_to(self.artifacts["route_csv"])
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            file_sha256(route_link)

        self._write_permit()
        permit_link = self.root / "permit-link.json"
        permit_link.symlink_to(self.permit_path)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            validate_mission_leg_motion_permit_for_execution(
                permit_link, **self._execution_kwargs()
            )
        master_link = self.root / "master-link.json"
        master_link.symlink_to(self.master_path)
        kwargs = self._execution_kwargs()
        kwargs["master_authorization_path"] = master_link
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            validate_mission_leg_motion_permit_for_execution(
                self.permit_path, **kwargs
            )

    def test_corrupt_unknown_or_noncanonical_stored_fields_fail_closed(self):
        self._write_permit()
        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw["run_id"] = "tampered"
        self.permit_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            load_mission_leg_motion_permit(self.permit_path)

        unknown = self.root / "unknown-field.json"
        raw = self.permit.to_payload()
        raw["unexpected"] = True
        raw["mission_leg_motion_permit_sha256"] = payload_sha256(raw)
        unknown.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "permit fields mismatch"):
            load_mission_leg_motion_permit(unknown)

        relative = self.root / "relative-field.json"
        raw = self.permit.to_payload()
        raw["route_csv_path"] = "route.csv"
        raw["mission_leg_motion_permit_sha256"] = payload_sha256(raw)
        relative.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "canonical absolute path"):
            load_mission_leg_motion_permit(relative)


if __name__ == "__main__":
    unittest.main()
