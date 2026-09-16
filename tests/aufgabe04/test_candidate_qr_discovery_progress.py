"""QR-only discovery never fabricates the geometry needed for facing targets."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import (
    CandidateQrGoalProgress, GOAL_PROGRESS_HASH_FIELD,
    GEOMETRY_FACING_EVIDENCE, QR_OBSERVATION_EVIDENCE,
    validate_candidate_qr_goal_completion,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, new_candidate_snapshot
from tests.aufgabe04.test_candidate_frame_projection import _frozen_candidate


class CandidateQrDiscoveryProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.validation_count = 0
        def candidate(index):
            item = _frozen_candidate(0.2 + index * 0.1, 0.0, source_registry_sha256="b" * 64)
            return replace(item, candidate_uid=f"candidate_{index}",
                           source=replace(item.source, observation_ids=(f"observation_{index}",)))
        self.snapshot = new_candidate_snapshot(
            snapshot_id="qr_discovery", created_unix_sec=2.0, planning_frame="map",
            map_bundle_sha256="a" * 64, candidates=tuple(
                candidate(index)
                for index in range(4)
            ),
        )

    def progress(self):
        return CandidateQrGoalProgress(
            self.snapshot.candidate_uids, expected_stand_count=2,
            candidate_snapshot_sha256=candidate_snapshot_sha256(self.snapshot),
        )

    def validate(self, payload):
        path = self.root / f"goal_{self.validation_count}.json"
        self.validation_count += 1
        write_content_hashed_json(path, payload, hash_field=GOAL_PROGRESS_HASH_FIELD)
        uids = payload["confirmed_candidate_uids"]
        confirmed = replace(self.snapshot, candidates=tuple(
            item for item in self.snapshot.candidates if item.candidate_uid in uids
        ))
        identities = {item["candidate_uid"]: item["qr_id"]
                      for item in payload["candidate_dispositions"]
                      if item["candidate_uid"] in uids}
        return validate_candidate_qr_goal_completion(
            path, candidate_snapshot=self.snapshot,
            confirmed_candidate_snapshot=confirmed,
            observed_qr_by_candidate=identities, expected_stand_count=2,
        )

    def mixed_progress(self):
        progress = self.progress()
        progress.mark_inspection_started("candidate_0")
        progress.record_validated_identity(
            "candidate_0", "QR_0", recommendation_path=self.root / "facing.json",
        )
        progress.mark_inspection_started("candidate_1")
        progress.record_observed_identity(
            "candidate_1", "QR_1", observation_pose_path=self.root / "observation.json",
        )
        progress.finalize_goal()
        return progress

    def test_mixed_goal_completes_discovery_without_complete_facing_geometry(self):
        progress = self.mixed_progress()
        self.assertTrue(progress.complete)
        self.assertFalse(progress.facing_complete)
        self.assertEqual(progress.facing_ready_candidate_uids, ("candidate_0",))
        self.assertEqual(progress.qr_only_candidate_uids, ("candidate_1",))
        payload = self.validate(progress.to_dict())
        self.assertEqual(payload["confirmed_stand_count"], 2)
        self.assertEqual(payload["facing_ready_stand_count"], 1)
        self.assertEqual(payload["qr_only_stand_count"], 1)
        geometry, observation = payload["candidate_dispositions"][:2]
        self.assertEqual(geometry["evidence_kind"], GEOMETRY_FACING_EVIDENCE)
        self.assertEqual(observation["evidence_kind"], QR_OBSERVATION_EVIDENCE)
        self.assertFalse(observation["facing_ready"])
        self.assertEqual(observation["observation_pose_path"], str(self.root / "observation.json"))
        for field in ("stand_axis_rad", "stand_pose", "recommendation_path", "facing_pose"):
            self.assertNotIn(field, observation)

    def test_cross_tier_duplicate_quarantines_both_and_revokes_facing_readiness(self):
        for geometry_first in (True, False):
            with self.subTest(geometry_first=geometry_first):
                progress = self.progress()
                for index in range(4):
                    uid = f"candidate_{index}"
                    progress.mark_inspection_started(uid)
                    qr = "QR_DUPLICATE" if index < 2 else f"QR_{index}"
                    if (index == 0) == geometry_first and index < 2:
                        accepted = progress.record_validated_identity(
                            uid, qr, recommendation_path=self.root / f"{uid}_facing.json",
                        )
                    else:
                        accepted = progress.record_observed_identity(
                            uid, qr, observation_pose_path=self.root / f"{uid}_observation.json",
                        )
                    self.assertEqual(accepted, index != 1)
                progress.finalize_goal()
                payload = self.validate(progress.to_dict())
                self.assertEqual(progress.facing_ready_candidate_uids, ())
                self.assertEqual(progress.qr_only_candidate_uids, ("candidate_2", "candidate_3"))
                self.assertEqual(payload["confirmed_qr_ids"], ["QR_2", "QR_3"])
                for record in payload["candidate_dispositions"][:2]:
                    self.assertEqual(record["disposition"], "ambiguous_duplicate_qr")
                    self.assertEqual(record["conflicting_candidate_uids"], ["candidate_0", "candidate_1"])
                    self.assertFalse(record["facing_ready"])

    def test_rehashed_false_readiness_or_missing_evidence_is_rejected(self):
        original = self.mixed_progress().to_dict()
        mutations = (
            lambda value: value.update(facing_complete=True),
            lambda value: value.update(qr_only_stand_count=0),
            lambda value: value.update(facing_ready_candidate_uids=["candidate_0", "candidate_1"]),
            lambda value: value["candidate_dispositions"][1].update(facing_ready=True),
            lambda value: value["candidate_dispositions"][1].pop("observation_pose_path"),
            lambda value: value["candidate_dispositions"][0].pop("recommendation_path"),
            lambda value: value["candidate_dispositions"][1].update(recommendation_path="fake_facing.json"),
            lambda value: value["candidate_dispositions"][1].update(evidence_kind="unchecked_qr"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                payload = deepcopy(original)
                mutate(payload)
                with self.assertRaises(ValueError):
                    self.validate(payload)

    def test_legacy_geometry_only_ledger_is_still_valid(self):
        progress = self.progress()
        for index in range(2):
            uid = f"candidate_{index}"
            progress.mark_inspection_started(uid)
            progress.record_validated_identity(uid, f"QR_{index}", recommendation_path=self.root / f"{uid}.json")
        progress.finalize_goal()
        payload = progress.to_dict()
        for field in ("facing_complete", "facing_ready_stand_count", "facing_ready_candidate_uids",
                      "qr_only_stand_count", "qr_only_candidate_uids"):
            payload.pop(field)
        for record in payload["candidate_dispositions"]:
            record.pop("evidence_kind", None)
            record.pop("facing_ready", None)
        self.validate(payload)

    def test_same_candidate_cannot_be_counted_twice_across_evidence_types(self):
        progress = self.mixed_progress()
        with self.assertRaisesRegex(RuntimeError, "already"):
            progress.record_validated_identity(
                "candidate_1", "QR_1", recommendation_path=self.root / "new_facing.json",
            )


if __name__ == "__main__":
    unittest.main()
