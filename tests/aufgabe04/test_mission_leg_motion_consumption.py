import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.aufgabe04.navigation.mission_leg_motion_consumption import (
    consume_mission_leg_motion_permit,
    default_mission_leg_motion_consumption_receipt_path,
    load_mission_leg_motion_consumption_receipt,
)
from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MISSION_LEG_RUN_CONFIRMATION,
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
    MissionLegMotionAuthorization,
    MissionLegMotionPermit,
    file_sha256,
    mission_leg_motion_authorization_sha256,
    write_mission_leg_motion_authorization,
    write_mission_leg_motion_permit,
)


class MissionLegMotionConsumptionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).absolute()
        self.master_path = self.root / "master.json"
        self.authorization = MissionLegMotionAuthorization(
            session_id="mission-001",
            robot_id="tb3_0",
            namespace="",
            cmd_vel_topic="/cmd_vel",
            semantic_map_id="arena",
            localization_branch_proof_id="branch-proof",
            allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS,
            scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            operator_confirmation=MISSION_LEG_RUN_CONFIRMATION,
        )
        write_mission_leg_motion_authorization(
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
            path.write_text(name + "\n", encoding="utf-8")
            self.artifacts[name] = path
        self.permit_path = self.root / "permit.json"
        self.permit = MissionLegMotionPermit(
            master_authorization_sha256=(
                mission_leg_motion_authorization_sha256(self.authorization)
            ),
            master_authorization_path=str(self.master_path),
            session_id="mission-001",
            robot_id="tb3_0",
            namespace="",
            cmd_vel_topic="/cmd_vel",
            semantic_map_id="arena",
            localization_branch_proof_id="branch-proof",
            run_id="mission-001-coverage-001",
            mission_leg_kind=MissionLegKind.COVERAGE,
            mission_leg_index=1,
            target_id="survey_vp_002",
            route_csv_path=str(self.artifacts["route_csv"]),
            route_csv_sha256=file_sha256(self.artifacts["route_csv"]),
            diagnostics_path=str(self.artifacts["diagnostics"]),
            diagnostics_sha256=file_sha256(self.artifacts["diagnostics"]),
            map_route_certificate_path=str(
                self.artifacts["map_route_certificate"]
            ),
            map_route_certificate_sha256=file_sha256(
                self.artifacts["map_route_certificate"]
            ),
            dry_preflight_path=str(self.artifacts["dry_preflight"]),
            dry_preflight_sha256=file_sha256(
                self.artifacts["dry_preflight"]
            ),
            dry_odom_certificate_path=str(
                self.artifacts["dry_odom_certificate"]
            ),
            dry_odom_certificate_sha256=file_sha256(
                self.artifacts["dry_odom_certificate"]
            ),
            dry_uncertainty_budget_path=str(
                self.artifacts["dry_uncertainty_budget"]
            ),
            dry_uncertainty_budget_sha256=file_sha256(
                self.artifacts["dry_uncertainty_budget"]
            ),
            dry_run_passed=True,
            additional_typed_run_required=False,
        )
        write_mission_leg_motion_permit(self.permit_path, self.permit)

    def tearDown(self):
        self.temporary.cleanup()

    def _consume(self, *, path=None, permit=None):
        return consume_mission_leg_motion_permit(
            permit_path=path or self.permit_path,
            permit=permit or self.permit,
            session_id=self.permit.session_id,
            run_id=self.permit.run_id,
            mission_leg_kind=self.permit.mission_leg_kind,
            mission_leg_index=self.permit.mission_leg_index,
            target_id=self.permit.target_id,
        )

    def test_claim_round_trip_and_replay_rejection(self):
        expected = default_mission_leg_motion_consumption_receipt_path(
            self.permit_path
        )
        receipt = self._consume()

        self.assertEqual(
            load_mission_leg_motion_consumption_receipt(expected), receipt
        )
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume()

    def test_identical_permit_copy_shares_one_claim_slot(self):
        copied = self.root / "copied_permit.json"
        shutil.copyfile(self.permit_path, copied)
        self._consume()

        self.assertEqual(
            default_mission_leg_motion_consumption_receipt_path(copied),
            default_mission_leg_motion_consumption_receipt_path(
                self.permit_path
            ),
        )
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume(path=copied)

    def test_identity_mismatch_does_not_claim(self):
        with self.assertRaisesRegex(ValueError, "target_id mismatch"):
            consume_mission_leg_motion_permit(
                permit_path=self.permit_path,
                permit=self.permit,
                session_id=self.permit.session_id,
                run_id=self.permit.run_id,
                mission_leg_kind=self.permit.mission_leg_kind,
                mission_leg_index=self.permit.mission_leg_index,
                target_id="wrong",
            )
        self.assertFalse(
            default_mission_leg_motion_consumption_receipt_path(
                self.permit_path
            ).exists()
        )

    def test_concurrent_claim_has_exactly_one_winner(self):
        def claim():
            try:
                return self._consume()
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = tuple(pool.map(lambda _index: claim(), range(2)))

        winners = [
            value
            for value in outcomes
            if not isinstance(value, ValueError)
        ]
        rejections = [
            value for value in outcomes if isinstance(value, ValueError)
        ]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(rejections), 1)
        self.assertIn("already consumed", str(rejections[0]))

    def test_permit_change_after_validation_fails_closed(self):
        self.artifacts["route_csv"].write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "route_csv hash mismatch"):
            self._consume()


if __name__ == "__main__":
    unittest.main()
