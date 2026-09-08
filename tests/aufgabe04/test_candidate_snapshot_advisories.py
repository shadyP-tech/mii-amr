"""Unresolved LiDAR evidence remains bound across snapshot persistence."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.candidate_perception_advisory import (
    CandidatePerceptionAdvisory, VISIBILITY_GAP,
)
from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance, CandidatePoint2D,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshotError, candidate_snapshot_payload, candidate_snapshot_sha256,
    load_candidate_snapshot, write_candidate_snapshot,
)
from tests.aufgabe04.test_candidate_snapshot import _candidate, _snapshot


def advisory(**changes):
    value = CandidatePerceptionAdvisory(
        kind=VISIBILITY_GAP, candidate_uid="candidate_a", survey_id="survey_1",
        map_bundle_sha256="a" * 64, plan_sha256="c" * 64,
        viewpoint_id="vp_2", source_morphology_sha256="d" * 64,
        candidate_frame=CandidateFrameProvenance(
            "map", "odom", CandidatePoint2D(1.0, 2.0),
            source_evidence_id="e" * 64,
        ),
        candidate_source_viewpoint_ids=("vp_1",),
        source_observation_ids=("obs_001",), proposal_max_range_m=3.5,
        visibility_radius_m=1.35, eligible_other_viewpoint_ids=(),
    )
    return replace(value, **changes)


class CandidateSnapshotAdvisoryTests(unittest.TestCase):
    def _with_advisory(self, value=None):
        candidate = _candidate()
        return _snapshot(replace(candidate, source=replace(
            candidate.source, perception_advisories=(value or advisory(),)
        )))

    def test_immutable_round_trip_and_geometry_unchanged(self):
        snapshot = self._with_advisory()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            write_candidate_snapshot(path, snapshot)
            self.assertEqual(load_candidate_snapshot(path), snapshot)
        self.assertEqual(snapshot.candidates[0].geometry, _candidate().geometry)
        self.assertNotEqual(candidate_snapshot_sha256(snapshot), candidate_snapshot_sha256(_snapshot()))

    def test_empty_legacy_source_payload_is_unchanged(self):
        source = candidate_snapshot_payload(_snapshot())["candidates"][0]["source"]
        self.assertNotIn("perception_advisories", source)
        self.assertEqual(set(source), {
            "source_kind", "source_artifact_sha256", "detector_config_sha256",
            "observation_ids", "source_sha256",
        })

    def test_other_candidate_or_map_or_observations_rejected(self):
        for change in (
            {"candidate_uid": "candidate_b"},
            {"map_bundle_sha256": "f" * 64},
            {"source_observation_ids": ("unrelated_observation",)},
        ):
            with self.subTest(change=change), self.assertRaises(CandidateSnapshotError):
                self._with_advisory(advisory(**change))

    def test_rehashing_outer_snapshot_does_not_hide_advisory_tampering(self):
        payload = candidate_snapshot_payload(self._with_advisory())
        source = payload["candidates"][0]["source"]
        source["perception_advisories"][0]["visibility_radius_m"] = 2.0
        source["source_sha256"] = payload_sha256({
            k: v for k, v in source.items() if k != "source_sha256"
        })
        payload["candidate_snapshot_sha256"] = payload_sha256({
            k: v for k, v in payload.items() if k != "candidate_snapshot_sha256"
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(CandidateSnapshotError, "advisory hash mismatch"):
                load_candidate_snapshot(path)
