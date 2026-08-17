import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_RECOVERY_KIND,
    STARTUP_RESEAL_RUN_CONFIRMATION,
    StartupResealMotionAuthorization,
    StartupResealMotionPermit,
    file_sha256,
    startup_reseal_motion_authorization_sha256,
    startup_reseal_motion_permit_sha256,
    write_startup_reseal_motion_authorization,
    write_startup_reseal_motion_permit,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_consumption import (
    STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
    StartupResealMotionConsumptionReceipt,
    consume_startup_reseal_motion_permit,
    default_startup_reseal_motion_consumption_receipt_path,
    derive_startup_reseal_motion_consumption_receipt_path,
    load_startup_reseal_motion_consumption_receipt,
    startup_reseal_motion_consumption_receipt_sha256,
)


class StartupResealMotionConsumptionTest(unittest.TestCase):
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
        write_startup_reseal_motion_authorization(
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
        pose = {"x_m": 0.1, "y_m": 0.2, "yaw_rad": 0.3}
        localization_evidence = {
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
        self.artifacts["fresh_stationary_localization_evidence"].write_text(
            json.dumps(localization_evidence) + "\n",
            encoding="utf-8",
        )
        self.artifacts["rejected_semantic_log"] = self.root / "rejected.jsonl"
        rejected = {
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
                "fail_closed": True,
            },
        }
        self.artifacts["rejected_semantic_log"].write_text(
            json.dumps(rejected) + "\n",
            encoding="utf-8",
        )
        self.artifacts["startup_reseal_summary"] = (
            self.root / "startup_summary.json"
        )
        summary = {
            "schema_version": 1,
            "status": "startup_route_replanned",
            "motion_published": False,
            "reseal_kind": "startup",
            "leg_index": 3,
            "startup_reseal_index": 1,
            "rejected_run_id": "mission-001-coverage-003",
            "target_viewpoint_id": "survey-vp-007",
            "fresh_start_pose": pose,
            "route_csv": self._path("route_csv"),
            "diagnostics_json": self._path("diagnostics"),
            "same_target_verified": True,
            "additional_typed_run_required": False,
        }
        self.artifacts["startup_reseal_summary"].write_text(
            json.dumps(summary) + "\n",
            encoding="utf-8",
        )
        self.permit = StartupResealMotionPermit(
            master_authorization_sha256=(
                startup_reseal_motion_authorization_sha256(self.authorization)
            ),
            master_authorization_path=str(self.master_path.absolute()),
            run_id="mission-001-coverage-003-startup-reseal-001",
            leg_index=3,
            target_viewpoint_id="survey-vp-007",
            reseal_index=1,
            max_startup_reseals_per_leg=2,
            rejected_run_id="mission-001-coverage-003",
            rejected_semantic_log_path=self._path("rejected_semantic_log"),
            rejected_semantic_log_sha256=self._sha("rejected_semantic_log"),
            startup_reseal_summary_path=self._path("startup_reseal_summary"),
            startup_reseal_summary_sha256=self._sha("startup_reseal_summary"),
            fresh_stationary_localization_evidence_path=self._path(
                "fresh_stationary_localization_evidence"
            ),
            fresh_stationary_localization_evidence_sha256=self._sha(
                "fresh_stationary_localization_evidence"
            ),
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
            dry_uncertainty_budget_sha256=self._sha("dry_uncertainty_budget"),
            same_target_verified=True,
            rejected_motion_published=False,
            dry_run_passed=True,
            additional_typed_run_required=False,
        )
        write_startup_reseal_motion_permit(self.permit_path, self.permit)

    def tearDown(self):
        self.temporary.cleanup()

    def _path(self, name):
        return str(self.artifacts[name].absolute())

    def _sha(self, name):
        return file_sha256(self.artifacts[name])

    def _consume(self, **replacements):
        values = {
            "permit_path": self.permit_path,
            "permit": self.permit,
            "session_id": self.authorization.session_id,
            "run_id": self.permit.run_id,
            "leg_index": self.permit.leg_index,
            "target_viewpoint_id": self.permit.target_viewpoint_id,
            "reseal_index": self.permit.reseal_index,
        }
        values.update(replacements)
        return consume_startup_reseal_motion_permit(**values)

    def test_first_claim_round_trips_and_identical_replay_rejects(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        self.assertEqual(
            receipt_path,
            derive_startup_reseal_motion_consumption_receipt_path(
                self.permit_path
            ),
        )
        receipt = self._consume()
        self.assertEqual(
            load_startup_reseal_motion_consumption_receipt(receipt_path),
            receipt,
        )
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            stored[STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD],
            startup_reseal_motion_consumption_receipt_sha256(receipt),
        )
        with self.assertRaises(FrozenInstanceError):
            receipt.run_id = "other"
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()

    def test_byte_identical_permit_copy_converges_on_one_claim(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        copy_dir = self.root / "copy"
        copy_dir.mkdir()
        copied = copy_dir / "permit.json"
        shutil.copyfile(self.permit_path, copied)
        self.assertEqual(
            default_startup_reseal_motion_consumption_receipt_path(copied),
            receipt_path,
        )
        self._consume()
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume(permit_path=copied)

    def test_concurrent_claims_have_exactly_one_winner(self):
        def claim():
            try:
                return self._consume()
            except ValueError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: claim(), range(2)))
        winners = [
            value
            for value in outcomes
            if isinstance(value, StartupResealMotionConsumptionReceipt)
        ]
        failures = [value for value in outcomes if isinstance(value, str)]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("already consumed", failures[0])

    def test_each_live_identity_mismatch_rejects_before_claim(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        wrong = {
            "session_id": "wrong-session",
            "run_id": "wrong-run",
            "leg_index": 4,
            "target_viewpoint_id": "wrong-target",
            "reseal_index": 2,
        }
        for name, value in wrong.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"{name} mismatch"):
                    self._consume(**{name: value})
                self.assertFalse(receipt_path.exists())

    def test_artifact_tamper_revalidates_and_rejects_before_claim(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        self.artifacts["dry_preflight"].write_text(
            "tampered\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "dry_preflight hash mismatch"):
            self._consume()
        self.assertFalse(receipt_path.exists())

    def test_changed_validated_permit_and_tampered_permit_file_reject(self):
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            self._consume(permit=replace(self.permit, run_id="another-run"))
        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw["run_id"] = "tampered"
        self.permit_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self._consume()

    def test_preexisting_malformed_claim_is_permanent_replay(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        receipt_path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()
        self.assertEqual(receipt_path.read_text(encoding="utf-8"), "{")
        with self.assertRaisesRegex(ValueError, "invalid artifact JSON"):
            load_startup_reseal_motion_consumption_receipt(receipt_path)

    def test_preexisting_receipt_symlink_is_permanent_replay(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        receipt_path.symlink_to(self.artifacts["diagnostics"])
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            load_startup_reseal_motion_consumption_receipt(receipt_path)

    def test_wrong_preexisting_hashed_receipt_is_never_replaced(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        wrong = StartupResealMotionConsumptionReceipt(
            startup_reseal_motion_permit_path=str(self.permit_path.absolute()),
            startup_reseal_motion_permit_sha256=(
                startup_reseal_motion_permit_sha256(self.permit)
            ),
            session_id=self.authorization.session_id,
            run_id="another-run",
            leg_index=self.permit.leg_index,
            target_viewpoint_id=self.permit.target_viewpoint_id,
            reseal_index=self.permit.reseal_index,
        )
        write_content_hashed_json(
            receipt_path,
            wrong.to_payload(),
            hash_field=STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
        )
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()
        with self.assertRaisesRegex(ValueError, "run_id mismatch"):
            load_startup_reseal_motion_consumption_receipt(receipt_path)

    def test_receipt_tamper_and_noncanonical_or_symlink_permit_reject(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        self._consume()
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw["session_id"] = "tampered"
        receipt_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            load_startup_reseal_motion_consumption_receipt(receipt_path)

        link = self.root / "permit-link.json"
        link.symlink_to(self.permit_path)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            self._consume(permit_path=link)
        noncanonical = self.root / "unused" / ".." / self.permit_path.name
        with self.assertRaisesRegex(ValueError, "canonical absolute"):
            self._consume(permit_path=noncanonical)

    def test_receipt_reload_revalidates_referenced_artifacts(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        self._consume()
        self.artifacts["route_csv"].write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "route_csv hash mismatch"):
            load_startup_reseal_motion_consumption_receipt(receipt_path)


if __name__ == "__main__":
    unittest.main()
