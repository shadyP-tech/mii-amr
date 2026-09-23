"""Start return preserves the exact-two registry's sealed rejected history."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import load_recommendation
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    REJECTION_BASIS_NEGATIVE_VISIBILITY, STATUS_REJECTED,
    coverage_survey_plan_sha256, stand_survey_registry_sha256,
)
from scripts.aufgabe04.real_robot.candidate.approach import CandidateApproachComplete
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import (
    CandidateQrGoalProgress, CandidateQrGoalProgressStore,
)
from scripts.aufgabe04.real_robot.candidate.qr_pose_discovery import write_qr_pose_catalog
from scripts.aufgabe04.real_robot.mission.stored_start_pose import load_stored_start_pose
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256, write_candidate_snapshot,
)
from scripts.aufgabe04.stations.server_identity_binding import write_observed_identities
from tests.aufgabe04.test_exact_two_camera_decision import (
    _fixture, _write_projected_exact_two_receipt,
)


class StartReturnExactTwoTest(unittest.TestCase):
    def test_loads_start_with_sealed_rejected_registry_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root, with_retained_negative_visibility_history=True,
                               with_frame_provenance=True)
            config = fixture.config
            start_uid = "survey_candidate_0003"
            rejected_uid = "survey_candidate_0006"
            full_sha = candidate_snapshot_sha256(config.snapshot)
            identities = {uid: "Start" if uid == start_uid else f"QR_{uid}"
                          for uid in config.snapshot.candidate_uids}
            goal = CandidateQrGoalProgress(
                config.snapshot.candidate_uids, expected_stand_count=5,
                candidate_snapshot_sha256=full_sha,
                perception_advisories_by_uid={
                    candidate.candidate_uid: [item.to_dict() for item in candidate.source.perception_advisories]
                    for candidate in config.snapshot.candidates
                },
            )
            records = []
            for uid in config.snapshot.candidate_uids:
                decision = _write_projected_exact_two_receipt(fixture, root / uid, uid)
                recommendation = load_recommendation(decision.recommendation_path)
                records.append({
                    **json.loads(decision.receipt_path.read_text()),
                    "candidate_uid": uid, "qr_id": identities[uid],
                    "recommendation_json": str(decision.recommendation_path),
                    "facing_pose": asdict(recommendation.material_target.pose),
                })
                goal.mark_inspection_started(uid)
                goal.record_validated_identity(uid, identities[uid],
                                               recommendation_path=decision.recommendation_path)
            goal.finalize_goal()
            goal_path, goal_sha = CandidateQrGoalProgressStore(config.session_root).write(goal)
            confirmed_path = config.session_root / "confirmed_candidate_snapshot.json"
            confirmed_sha = write_candidate_snapshot(confirmed_path, config.snapshot)
            observed_path = config.session_root / "observed_station_identities.json"
            observed_sha = write_observed_identities(
                observed_path, candidate_snapshot=config.snapshot,
                observed_qr_by_candidate=identities, session_id=config.session_id,
                observed_unix_sec=30.,
            )
            metadata = {
                "session_id": config.session_id, "planning_frame": config.planning_frame,
                "map_bundle_sha256": config.plan.map_bundle_sha256,
                "coverage_plan_sha256": coverage_survey_plan_sha256(config.plan),
                "candidate_snapshot_sha256": full_sha,
                "confirmed_candidate_snapshot_sha256": confirmed_sha,
                "candidate_goal_progress_sha256": goal_sha,
                "observed_station_identities_sha256": observed_sha,
                "robot_profile_sha256": config.robot_profile_sha256,
                "calibration_profile_sha256": config.calibration_profile_sha256,
                "source_registry_path": str(fixture.survey_root / "stand_registry.json"),
                "source_registry_sha256": stand_survey_registry_sha256(fixture.registry),
            }
            facing_path = config.session_root / "stand_facing_catalog.json"
            facing_sha = write_content_hashed_json(facing_path, {
                **metadata, "catalog_kind": "real_autonomous_stand_facing_poses",
                "records": records,
            }, hash_field="stand_facing_catalog_sha256")
            qr_path = config.session_root / "qr_observation_pose_catalog.json"
            qr_sha = write_qr_pose_catalog(qr_path, metadata=metadata, records=[])
            completed = CandidateApproachComplete(
                stand_count=5, visit_order=config.snapshot.candidate_uids,
                identity_registry_path=None, identity_registry_sha256=None,
                stand_facing_catalog_path=facing_path, stand_facing_catalog_sha256=facing_sha,
                facing_records=tuple(records), expected_stand_count=5,
                candidate_pool_count=5, confirmed_candidate_snapshot_path=confirmed_path,
                confirmed_candidate_snapshot_sha256=confirmed_sha,
                candidate_goal_progress_path=goal_path, candidate_goal_progress_sha256=goal_sha,
                observed_identities_path=observed_path, observed_identities_sha256=observed_sha,
                qr_observation_catalog_path=qr_path, qr_observation_catalog_sha256=qr_sha,
            )

            stored = load_stored_start_pose(completed, config)

            expected = next(record for record in records if record["candidate_uid"] == start_uid)
            self.assertEqual(stored.candidate_uid, start_uid)
            self.assertEqual(asdict(stored.pose), expected["facing_pose"])
            self.assertEqual(stored.evidence["pose_kind"], "geometry_validated_facing_pose")
            self.assertEqual(stored.registry, fixture.registry)
            self.assertNotIn(rejected_uid, config.snapshot.candidate_uids)
            rejected = stored.registry.candidate_for(rejected_uid)
            self.assertEqual(rejected.status, STATUS_REJECTED)
            self.assertEqual(rejected.rejection_basis, REJECTION_BASIS_NEGATIVE_VISIBILITY)
            self.assertIn(str(fixture.handoff_path.resolve()),
                          [source["path"] for source in stored.evidence["source_artifacts"]])


if __name__ == "__main__":
    unittest.main()
