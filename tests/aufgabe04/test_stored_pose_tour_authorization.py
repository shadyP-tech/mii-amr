"""Separate stored-pose tour authority and per-visit single-use stages."""

from dataclasses import replace
import json
import os
from pathlib import Path
import unittest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import consume_mission_leg_motion_permit
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MissionLegKind, file_sha256, write_mission_leg_motion_authorization,
    write_mission_leg_motion_permit,
    validate_stored_pose_tour_target_evidence,
)
from tests.aufgabe04 import test_mission_leg_motion_permit as fixtures


def catalog_evidence(root, *, uid="candidate-work", qr="Werkbank", pose=None):
    pose = pose or {"x_m": 1., "y_m": 2., "yaw_rad": .5}
    catalog = root / f"{uid}-catalog.json"
    digest = write_content_hashed_json(catalog, {
        "catalog_kind": "real_autonomous_stand_facing_poses",
        "records": [{"candidate_uid": uid, "qr_id": qr, "facing_pose": pose}],
    }, hash_field="stand_facing_catalog_sha256")
    return {
        "candidate_uid": uid, "qr_id": qr, "pose_kind": "geometry_validated_facing_pose",
        "stored_pose": pose, "catalog_path": str(catalog), "catalog_sha256": digest,
        "source_artifacts": [{"path": str(catalog), "sha256": file_sha256(catalog)}],
    }


class StoredPoseTourAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MissionLegMotionPermitTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.master = replace(
            self.fixture.authorization, session_id="fresh-tour",
            allowed_leg_kinds=(MissionLegKind.STORED_POSE_TOUR,),
            scope_text=TOUR_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
        )
        self.master_path = self.root / "tour-master.json"
        self.master_hash = write_mission_leg_motion_authorization(self.master_path, self.master)
        self.serial = 0

    def _permit(self, *, visit=0, stage=0, final=True, uid="candidate-work", qr="Werkbank"):
        self.serial += 1
        run_id = f"tour-leg-{self.serial}"
        evidence = {
            **catalog_evidence(self.root, uid=uid, qr=qr),
            "tour_id": self.master.session_id, "visit_index": visit,
            "target_pose": {"x_m": 1., "y_m": 2., "yaw_rad": .5},
        }
        evidence_path = self.root / f"{run_id}-target.json"
        evidence_path.write_text(json.dumps(evidence))
        metadata = {
            "route_kind": "admitted_candidate_pose", "route_purpose": "stored_pose_tour",
            "selected_candidate_stand_id": uid, "tour_id": self.master.session_id,
            "visit_index": visit, "qr_id": qr,
            "route_csv_sha256": self.fixture.permit.route_csv_sha256,
            "selected_approach_pose": evidence["target_pose"] if final else {"x_m": .5, "y_m": 2., "yaw_rad": .5},
            "stored_start_target_pose": evidence["target_pose"],
            "target_evidence_json": str(evidence_path), "target_evidence_sha256": file_sha256(evidence_path),
            "return_to_start_stage": {"stage_index": stage, "final_stage": final, "start_candidate_uid": uid},
        }
        diagnostics = self.root / f"{run_id}-diagnostics.json"
        diagnostics.write_text(json.dumps({"metadata": metadata}))
        permit = replace(
            self.fixture.permit, session_id=self.master.session_id, run_id=run_id,
            master_authorization_path=str(self.master_path), master_authorization_sha256=self.master_hash,
            mission_leg_kind=MissionLegKind.STORED_POSE_TOUR, mission_leg_index=visit * 4 + stage,
            target_id=uid, diagnostics_path=str(diagnostics), diagnostics_sha256=file_sha256(diagnostics),
        )
        return permit, self.root / f"{run_id}-permit.json"

    def _consume(self, permit, path):
        write_mission_leg_motion_permit(path, permit)
        return consume_mission_leg_motion_permit(
            permit_path=path, permit=permit, session_id=permit.session_id, run_id=permit.run_id,
            mission_leg_kind=permit.mission_leg_kind, mission_leg_index=permit.mission_leg_index,
            target_id=permit.target_id,
        )

    def test_tour_scope_cannot_be_inherited_or_mixed_with_exploration(self):
        for scope in (MISSION_LEG_MOTION_AUTHORIZATION_SCOPE, LEGACY_SINGLE_RETURN_MISSION_LEG_MOTION_AUTHORIZATION_SCOPE):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "tour-only authorization"):
                replace(self.master, scope_text=scope)
        for kinds in ((MissionLegKind.COVERAGE,), (MissionLegKind.RETURN_TO_START, MissionLegKind.STORED_POSE_TOUR)):
            with self.subTest(kinds=kinds), self.assertRaisesRegex(ValueError, "tour-only authorization"):
                replace(self.master, allowed_leg_kinds=kinds)

    def test_relative_catalog_reference_matches_its_absolute_source_artifact(self):
        evidence = {
            **catalog_evidence(self.root), "tour_id": "fresh-tour", "visit_index": 0,
        }
        evidence["catalog_path"] = os.path.relpath(evidence["catalog_path"], Path.cwd())
        self.assertFalse(Path(evidence["catalog_path"]).is_absolute())
        self.assertTrue(Path(evidence["source_artifacts"][0]["path"]).is_absolute())
        validate_stored_pose_tour_target_evidence(evidence)

    def test_arbitrary_stored_qr_visits_have_independent_single_use_stage_slots(self):
        self._consume(*self._permit(final=False))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self._consume(*self._permit(final=False))
        with self.assertRaisesRegex(ValueError, "same master and Start target"):
            self._consume(*self._permit(stage=1, uid="other", qr="Other"))
        self._consume(*self._permit(stage=1))
        with self.assertRaisesRegex(ValueError, "after a final stage"):
            self._consume(*self._permit(stage=2))
        self.assertEqual(self._consume(*self._permit(visit=1, uid="other", qr="Other")).mission_leg_index, 4)

    def test_visit_index_and_stage_index_are_bound_to_the_encoded_permit_index(self):
        permit, path = self._permit(visit=2)
        write_mission_leg_motion_permit(path, permit)
        for index in (4, 9):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "visit_index mismatch|stage identity mismatch"):
                write_mission_leg_motion_permit(self.root / f"wrong-{index}.json", replace(permit, mission_leg_index=index))

    def test_qr_relabel_cannot_override_the_original_admitted_catalog(self):
        permit, path = self._permit()
        diagnostics_path = Path(permit.diagnostics_path)
        diagnostics = json.loads(diagnostics_path.read_text())
        metadata = diagnostics["metadata"]
        evidence_path = Path(metadata["target_evidence_json"])
        evidence = json.loads(evidence_path.read_text())
        evidence["qr_id"] = metadata["qr_id"] = "Start"
        evidence_path.write_text(json.dumps(evidence))
        metadata["target_evidence_sha256"] = file_sha256(evidence_path)
        diagnostics_path.write_text(json.dumps(diagnostics))
        permit = replace(permit, diagnostics_sha256=file_sha256(diagnostics_path))
        with self.assertRaisesRegex(ValueError, "differs from admitted catalog"):
            write_mission_leg_motion_permit(path, permit)

    def test_tour_id_substitution_and_last_nonfinal_stage_fail_closed(self):
        permit, path = self._permit(stage=3, final=False)
        with self.assertRaisesRegex(ValueError, "last return_to_start stage"):
            write_mission_leg_motion_permit(path, permit)
        permit, path = self._permit()
        diagnostics_path = Path(permit.diagnostics_path)
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["metadata"]["tour_id"] = "exploration-session"
        diagnostics_path.write_text(json.dumps(diagnostics))
        with self.assertRaisesRegex(ValueError, "tour_id mismatch"):
            write_mission_leg_motion_permit(path, replace(permit, diagnostics_sha256=file_sha256(diagnostics_path)))


if __name__ == "__main__":
    unittest.main()
