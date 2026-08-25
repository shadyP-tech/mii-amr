import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    payload_sha256,
    write_content_hashed_json,
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
    mission_motion_authorization_sha256,
    write_mission_motion_authorization,
    write_runtime_localization_motion_permit,
)
from scripts.aufgabe04.navigation.execution.runtime_motion_consumption import (
    RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
    RuntimeMotionConsumptionReceipt,
    consume_runtime_motion_permit,
    default_runtime_motion_consumption_receipt_path,
    derive_runtime_motion_consumption_receipt_path,
    load_runtime_motion_consumption_receipt,
    runtime_motion_consumption_receipt_sha256,
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


class RuntimeMotionConsumptionTest(unittest.TestCase):
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
        write_mission_motion_authorization(
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
            master_authorization_sha256=mission_motion_authorization_sha256(
                self.authorization
            ),
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
        write_runtime_localization_motion_permit(
            self.permit_path, self.permit
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _path(self, name):
        return str(self.artifacts[name].absolute())

    def _sha(self, name):
        return file_sha256(self.artifacts[name])

    def _consume(self, **replacements):
        arguments = {
            "permit_path": self.permit_path,
            "permit": self.permit,
            "session_id": self.authorization.session_id,
            "run_id": self.permit.run_id,
            "leg_index": self.permit.leg_index,
            "target_viewpoint_id": self.permit.target_viewpoint_id,
            "reseal_index": self.permit.reseal_index,
        }
        arguments.update(replacements)
        return consume_runtime_motion_permit(**arguments)

    def test_first_use_passes_round_trips_and_second_identical_use_rejects(self):
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        self.assertEqual(
            receipt_path,
            derive_runtime_motion_consumption_receipt_path(self.permit_path),
        )
        self.assertEqual(receipt_path.parent, self.master_path.parent)

        receipt = self._consume()

        self.assertTrue(receipt_path.is_file())
        self.assertEqual(
            load_runtime_motion_consumption_receipt(receipt_path), receipt
        )
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            stored[RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD],
            runtime_motion_consumption_receipt_sha256(receipt),
        )
        self.assertEqual(
            receipt.runtime_localization_motion_permit_path,
            str(self.permit_path.absolute()),
        )
        with self.assertRaises(FrozenInstanceError):
            receipt.run_id = "other"
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()

    def test_candidate_runtime_permit_consumes_with_generic_identity(self):
        candidate = replace(
            self.permit,
            run_id="mission-candidate-003-runtime-reseal-001",
            leg_index=3,
            target_viewpoint_id="survey-candidate-003",
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=3,
            target_id="survey-candidate-003",
        )
        candidate_path = self.root / "candidate-runtime-permit.json"
        write_runtime_localization_motion_permit(candidate_path, candidate)

        receipt = consume_runtime_motion_permit(
            permit_path=candidate_path,
            permit=candidate,
            session_id=self.authorization.session_id,
            run_id=candidate.run_id,
            leg_index=candidate.leg_index,
            target_viewpoint_id=candidate.target_viewpoint_id,
            reseal_index=candidate.reseal_index,
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=3,
            target_id="survey-candidate-003",
        )

        self.assertEqual(
            receipt.mission_leg_kind,
            MissionLegKind.CANDIDATE_PREAPPROACH,
        )
        self.assertEqual(receipt.mission_leg_index, 3)
        self.assertEqual(receipt.target_id, "survey-candidate-003")
        loaded = load_runtime_motion_consumption_receipt(
            default_runtime_motion_consumption_receipt_path(candidate_path)
        )
        self.assertEqual(loaded, receipt)

    def test_byte_identical_permit_copy_converges_on_existing_claim(self):
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        self._consume()
        copy_dir = self.root / "copied"
        copy_dir.mkdir()
        copied_permit = copy_dir / "copied-permit.json"
        shutil.copyfile(self.permit_path, copied_permit)

        self.assertEqual(
            default_runtime_motion_consumption_receipt_path(copied_permit),
            receipt_path,
        )
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume(permit_path=copied_permit)

    def test_concurrent_claims_have_exactly_one_winner(self):
        def claim():
            try:
                return self._consume()
            except ValueError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: claim(), range(2)))

        receipts = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, RuntimeMotionConsumptionReceipt)
        ]
        failures = [outcome for outcome in outcomes if isinstance(outcome, str)]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("already consumed", failures[0])

    def test_preexisting_malformed_receipt_rejects_without_replacement(self):
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        receipt_path.write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()
        self.assertEqual(receipt_path.read_text(encoding="utf-8"), "{")
        with self.assertRaisesRegex(ValueError, "invalid artifact JSON"):
            load_runtime_motion_consumption_receipt(receipt_path)

    def test_preexisting_receipt_symlink_rejects_as_consumed(self):
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        receipt_path.symlink_to(self.artifacts["diagnostics"])

        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            load_runtime_motion_consumption_receipt(receipt_path)

    def test_preexisting_content_hashed_identity_mismatch_fails_closed(self):
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        wrong = RuntimeMotionConsumptionReceipt(
            runtime_localization_motion_permit_path=str(
                self.permit_path.absolute()
            ),
            runtime_localization_motion_permit_sha256=(
                self._permit_payload_sha256()
            ),
            run_id="different-run",
            session_id=self.authorization.session_id,
            leg_index=self.permit.leg_index,
            target_viewpoint_id=self.permit.target_viewpoint_id,
            reseal_index=self.permit.reseal_index,
        )
        write_content_hashed_json(
            receipt_path,
            wrong.to_payload(),
            hash_field=RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
        )

        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()
        with self.assertRaisesRegex(ValueError, "run_id mismatch"):
            load_runtime_motion_consumption_receipt(receipt_path)

    def test_every_live_identity_mismatch_rejects_before_claim(self):
        wrong_values = {
            "session_id": "wrong-session",
            "run_id": "wrong-run",
            "leg_index": self.permit.leg_index + 1,
            "target_viewpoint_id": "wrong-viewpoint",
            "reseal_index": self.permit.reseal_index + 1,
        }
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        for name, value in wrong_values.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"{name} mismatch"):
                    self._consume(**{name: value})
                self.assertFalse(receipt_path.exists())

    def test_changed_validated_permit_or_tampered_file_rejects(self):
        altered = replace(self.permit, run_id="another-valid-run")
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            self._consume(permit=altered)

        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw["run_id"] = "tampered"
        self.permit_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self._consume()

    def test_receipt_content_tampering_is_hash_rejected(self):
        receipt_path = default_runtime_motion_consumption_receipt_path(
            self.permit_path
        )
        self._consume()
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw["session_id"] = "tampered-session"
        receipt_path.write_text(json.dumps(raw), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            load_runtime_motion_consumption_receipt(receipt_path)

    def test_symlink_and_noncanonical_permit_paths_reject(self):
        symlink = self.root / "permit-link.json"
        symlink.symlink_to(self.permit_path)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            self._consume(permit_path=symlink)

        noncanonical = self.root / "unused" / ".." / self.permit_path.name
        with self.assertRaisesRegex(ValueError, "canonical absolute"):
            self._consume(permit_path=noncanonical)

    def _permit_payload_sha256(self):
        from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
            runtime_localization_motion_permit_sha256,
        )

        return runtime_localization_motion_permit_sha256(self.permit)


if __name__ == "__main__":
    unittest.main()
