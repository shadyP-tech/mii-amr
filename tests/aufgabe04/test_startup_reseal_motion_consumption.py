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
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD,
    STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION,
    STARTUP_RESEAL_RECOVERY_KIND,
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
    STARTUP_RESEAL_RUN_CONFIRMATION,
    StartupResealMotionAuthorization,
    StartupResealMotionPermit,
    file_sha256,
    load_startup_reseal_motion_permit,
    startup_reseal_motion_authorization_sha256,
    startup_reseal_motion_permit_sha256,
    write_startup_reseal_motion_authorization,
    write_startup_reseal_motion_permit,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_consumption import (
    STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
    StartupResealMotionConsumptionReceipt,
    consume_startup_reseal_motion_permit,
    default_startup_reseal_motion_consumption_receipt_path,
    derive_startup_reseal_motion_consumption_receipt_path,
    load_startup_reseal_motion_consumption_receipt,
    startup_reseal_motion_consumption_receipt_sha256,
)
from scripts.aufgabe04.navigation.localization.ros_preflight_evidence_contract import (
    ros_preflight_requirements_evidence,
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
        self._write_route_and_diagnostics()
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
            "preflight_requirements": ros_preflight_requirements_evidence(
                stationary_map_from_odom_pairing_requested=False,
                stationary_map_from_odom_pairing_required=False,
            ),
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
            "fresh_start_pose": pose,
            "route_csv": self._path("route_csv"),
            "diagnostics_json": self._path("diagnostics"),
            "same_target_verified": True,
            "additional_typed_run_required": False,
            "recovery_source_kind": (
                STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
            ),
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
            recovery_source_kind=(
                STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
            ),
        )
        write_startup_reseal_motion_permit(self.permit_path, self.permit)

    def tearDown(self):
        self.temporary.cleanup()

    def _path(self, name):
        return str(self.artifacts[name].absolute())

    def _sha(self, name):
        return file_sha256(self.artifacts[name])

    def _write_route_and_diagnostics(self, *, start_x_m=0.1):
        anchor_x_m = 0.2
        length_m = anchor_x_m - start_x_m
        sample_count = 21
        sample_spacing_m = length_m / (sample_count - 1)
        minimum_sampled_clearance_m = 0.5
        minimum_continuous_clearance_m = (
            minimum_sampled_clearance_m - sample_spacing_m / 2.0
        )
        self.artifacts["route_csv"].write_text(
            "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
            "yaw_rad,segment_length_m,cumulative_length_m\n"
            f"0,0,0,0,{start_x_m},0.2,,0.0,0.0\n"
            f"0,1,1,0,{anchor_x_m},0.2,0.3,{length_m},{length_m}\n",
            encoding="utf-8",
        )
        pose = {"x_m": start_x_m, "y_m": 0.2, "yaw_rad": 0.3}
        diagnostics = {
            "metadata": {
                "planning_frame": "map",
                "inflation_radius_m": 0.01,
                "exact_start_connector": {
                    "required": True,
                    "validated": True,
                    "exact_start": pose,
                    "anchor": {
                        "x_m": anchor_x_m,
                        "y_m": 0.2,
                        "yaw_rad": 0.3,
                    },
                    "connector_length_m": length_m,
                    "required_clearance_m": 0.01,
                    "minimum_sampled_clearance_m": (
                        minimum_sampled_clearance_m
                    ),
                    "minimum_continuous_clearance_m": (
                        minimum_continuous_clearance_m
                    ),
                    "minimum_margin_m": (
                        minimum_continuous_clearance_m - 0.01
                    ),
                    "sample_spacing_m": sample_spacing_m,
                    "sample_count": sample_count,
                },
                "route_start_pose_provenance": {
                    "source": "autonomous_candidate_current_pose",
                    "planning_frame": "map",
                    "pose": pose,
                },
            }
        }
        self.artifacts["diagnostics"].write_text(
            json.dumps(diagnostics, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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

    def _rewrite_fresh_localization_and_permit(self, payload):
        self.artifacts["fresh_stationary_localization_evidence"].write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw.pop(STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD)
        raw["fresh_stationary_localization_evidence_sha256"] = self._sha(
            "fresh_stationary_localization_evidence"
        )
        raw[STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD] = payload_sha256(raw)
        self.permit_path.write_text(
            json.dumps(raw, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.permit = load_startup_reseal_motion_permit(self.permit_path)

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

    def test_candidate_identity_is_persisted_in_one_use_receipt(self):
        candidate_master = self.root / "candidate_startup_master.json"
        self.authorization = replace(
            self.authorization,
            allowed_mission_leg_kinds=(
                MissionLegKind.COVERAGE,
                MissionLegKind.CANDIDATE_PREAPPROACH,
            ),
        )
        write_startup_reseal_motion_authorization(
            candidate_master,
            self.authorization,
        )
        rejected = json.loads(
            self.artifacts["rejected_semantic_log"].read_text(
                encoding="utf-8"
            )
        )
        rejected.update(
            {
                "mission_leg_kind": (
                    MissionLegKind.CANDIDATE_PREAPPROACH.value
                ),
                "mission_leg_index": 3,
                "target_id": "survey-vp-007",
                "coverage_leg_index": None,
                "target_viewpoint_id": "",
            }
        )
        self.artifacts["rejected_semantic_log"].write_text(
            json.dumps(rejected) + "\n",
            encoding="utf-8",
        )
        summary = json.loads(
            self.artifacts["startup_reseal_summary"].read_text(
                encoding="utf-8"
            )
        )
        summary["mission_leg_kind"] = (
            MissionLegKind.CANDIDATE_PREAPPROACH.value
        )
        self.artifacts["startup_reseal_summary"].write_text(
            json.dumps(summary) + "\n",
            encoding="utf-8",
        )
        localization_evidence = json.loads(
            self.artifacts["fresh_stationary_localization_evidence"].read_text(
                encoding="utf-8"
            )
        )
        localization_evidence["preflight_requirements"] = (
            ros_preflight_requirements_evidence(
                stationary_map_from_odom_pairing_requested=True,
                stationary_map_from_odom_pairing_required=True,
            )
        )
        self.artifacts["fresh_stationary_localization_evidence"].write_text(
            json.dumps(localization_evidence) + "\n",
            encoding="utf-8",
        )
        self.permit_path = self.root / "candidate_startup_permit.json"
        self.permit = replace(
            self.permit,
            master_authorization_path=str(candidate_master.absolute()),
            master_authorization_sha256=(
                startup_reseal_motion_authorization_sha256(
                    self.authorization
                )
            ),
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            rejected_semantic_log_sha256=self._sha(
                "rejected_semantic_log"
            ),
            startup_reseal_summary_sha256=self._sha(
                "startup_reseal_summary"
            ),
            fresh_stationary_localization_evidence_sha256=self._sha(
                "fresh_stationary_localization_evidence"
            ),
        )
        write_startup_reseal_motion_permit(self.permit_path, self.permit)

        receipt = self._consume(
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=3,
            target_id="survey-vp-007",
        )
        self.assertEqual(
            receipt.mission_leg_kind,
            MissionLegKind.CANDIDATE_PREAPPROACH,
        )
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume(
                mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                mission_leg_index=3,
                target_id="survey-vp-007",
            )

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
            "mission_leg_kind": MissionLegKind.CANDIDATE_PREAPPROACH,
            "mission_leg_index": 4,
            "target_id": "wrong-target",
        }
        for name, value in wrong.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"{name}.*mismatch"):
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

    def test_fresh_localization_accepts_current_requirement_schema(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        self._consume()
        self.assertTrue(receipt_path.exists())

    def test_fresh_localization_rejects_missing_requirement_schema(self):
        receipt_path = default_startup_reseal_motion_consumption_receipt_path(
            self.permit_path
        )
        payload = json.loads(
            self.artifacts["fresh_stationary_localization_evidence"].read_text(
                encoding="utf-8"
            )
        )
        payload.pop("preflight_requirements")
        self._rewrite_fresh_localization_and_permit(payload)

        with self.assertRaisesRegex(
            ValueError,
            "fresh stationary localization evidence fields mismatch",
        ):
            self._consume()
        self.assertFalse(receipt_path.exists())

    def test_rehashed_semantic_route_mismatch_rejects_before_claim(self):
        self._write_route_and_diagnostics(start_x_m=0.11)
        raw = json.loads(self.permit_path.read_text(encoding="utf-8"))
        raw.pop(STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD)
        raw["route_csv_sha256"] = self._sha("route_csv")
        raw["diagnostics_sha256"] = self._sha("diagnostics")
        raw[STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD] = payload_sha256(raw)
        self.permit_path.write_text(
            json.dumps(raw, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.permit = load_startup_reseal_motion_permit(self.permit_path)

        with self.assertRaisesRegex(ValueError, "replacement exact start differs"):
            self._consume()

        self.assertEqual(
            list(self.root.glob("startup_reseal_motion_consumption_*.json")),
            [],
        )

    def test_changed_validated_permit_and_tampered_permit_file_reject(self):
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            self._consume(permit=replace(self.permit, run_id="another-run"))
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            self._consume(
                permit=replace(
                    self.permit,
                    recovery_source_kind=(
                        STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
                    ),
                )
            )
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
