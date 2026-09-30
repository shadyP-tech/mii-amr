"""Replay the recorded wall target through current-frame target admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.aufgabe04.artifacts.candidate_perception_advisory import MORPHOLOGY_CONFLICT
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
    project_candidate_snapshot_to_planning_frame,
)
from scripts.aufgabe04.navigation.approach.candidate_target_admission import (
    evaluate_candidate_target_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    load_stand_survey_registry,
    stand_survey_registry_sha256,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / 'fixtures/wall_candidate_20260930'
WALL_UID = 'survey_candidate_0004'


class RecordedWallCandidateAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FIXTURE / 'manifest.json').read_text())
        cls.snapshot = load_candidate_snapshot(FIXTURE / 'candidate_snapshot.json')
        cls.registry = load_stand_survey_registry(FIXTURE / 'camera_source_stand_registry.json')
        grid = load_occupancy_grid(ROOT / 'maps/aufgabe03/arena_1p898x3p9_auto.yaml')
        cls.costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(
            ArenaBounds(**cls.manifest['arena_bounds'])
        )
        cls.projection_documents = {
            stage: json.loads((FIXTURE / f'{stage}_projection.json').read_text())
            for stage in ('selection', 'arrival')
        }
        cls.projections = {
            stage: project_candidate_snapshot_to_planning_frame(
                cls.snapshot, cls.registry,
                CandidatePlanningFrame.from_evidence(document['planning_frame_admission']),
            )
            for stage, document in cls.projection_documents.items()
        }

    def test_recorded_inputs_and_replayed_projections_match_original_hashes(self):
        for filename, record in self.manifest['files'].items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((FIXTURE / filename).read_bytes()).hexdigest(), record['sha256'])
        for filename, record in self.manifest['repository_map_files'].items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), record['sha256'])
        self.assertEqual(len(self.costmap.blocked_cells), 5109)
        self.assertEqual(len(self.snapshot.candidates), 6)
        self.assertEqual(self.snapshot.map_bundle_sha256, self.manifest['map_bundle_sha256'])
        for stage, projection in self.projections.items():
            with self.subTest(stage=stage):
                self.assertEqual(
                    candidate_snapshot_sha256(projection.projected_snapshot),
                    self.projection_documents[stage]['projected_candidate_snapshot_sha256'],
                )

    def test_frozen_wall_is_deferred_for_both_original_morphology_conflicts(self):
        wall = self.snapshot.candidate_for(WALL_UID)
        decision = evaluate_candidate_target_admission(wall, self.costmap)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reasons, ('unresolved_morphology_conflict',))
        self.assertAlmostEqual(decision.static_map_evidence.static_map_clearance_m, 0.06861894414464423)
        self.assertEqual(decision.static_map_evidence.disposition, 'boundary_provisional')
        self.assertTrue(decision.static_map_evidence.population_retained)
        conflicts = {
            advisory.sha256 for advisory in wall.source.perception_advisories
            if advisory.kind == MORPHOLOGY_CONFLICT
        }
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(set(decision.unresolved_morphology_conflict_sha256), conflicts)

    def test_selection_and_arrival_wall_targets_are_static_map_incompatible(self):
        for stage, projection in self.projections.items():
            with self.subTest(stage=stage):
                wall = projection.projected_snapshot.candidate_for(WALL_UID)
                decision = evaluate_candidate_target_admission(wall, self.costmap)
                self.assertFalse(decision.accepted)
                self.assertEqual(set(decision.reasons), {
                    'target_static_map_incompatible', 'unresolved_morphology_conflict',
                })
                self.assertEqual(decision.static_map_evidence.static_map_clearance_m, 0.0)
                self.assertEqual(decision.static_map_evidence.disposition, 'rejected')
                self.assertFalse(decision.static_map_evidence.population_retained)
                evidence = decision.to_evidence()
                self.assertEqual(evidence['candidate_uid'], WALL_UID)
                self.assertFalse(evidence['motion_authorized'])
                self.assertFalse(evidence['candidate_rejection_authorized'])

    def test_valid_recorded_controls_stay_available_with_all_keepouts_preserved(self):
        registry_hash_before = stand_survey_registry_sha256(self.registry)
        frozen_hash_before = candidate_snapshot_sha256(self.snapshot)
        stages = {'frozen': self.snapshot, **{
            stage: projection.projected_snapshot
            for stage, projection in self.projections.items()
        }}
        for stage, snapshot in stages.items():
            with self.subTest(stage=stage):
                snapshot_hash_before = candidate_snapshot_sha256(snapshot)
                decisions = {
                    candidate.candidate_uid: evaluate_candidate_target_admission(candidate, self.costmap)
                    for candidate in snapshot.candidates
                }
                self.assertEqual(
                    {uid for uid, decision in decisions.items() if decision.accepted},
                    set(snapshot.candidate_uids) - {WALL_UID},
                )
                # Three real controls carry visibility-gap advisories. Those
                # alone must not become a blanket single-view rejection.
                for uid in ('survey_candidate_0001', 'survey_candidate_0005', 'survey_candidate_0006'):
                    self.assertTrue(snapshot.candidate_for(uid).source.perception_advisories)
                    self.assertTrue(decisions[uid].accepted)
                self.assertEqual(snapshot.candidate_uids, self.snapshot.candidate_uids)
                self.assertEqual(len(snapshot.candidates), 6)
                for candidate in snapshot.candidates:
                    frozen = self.snapshot.candidate_for(candidate.candidate_uid)
                    self.assertEqual(candidate.geometry.keepout_radius_m, frozen.geometry.keepout_radius_m)
                    self.assertEqual(candidate.geometry.radius_m, frozen.geometry.radius_m)
                    self.assertEqual(candidate.geometry.uncertainty_m, frozen.geometry.uncertainty_m)
                    self.assertEqual(candidate.source, frozen.source)
                self.assertEqual(candidate_snapshot_sha256(snapshot), snapshot_hash_before)
        self.assertEqual(candidate_snapshot_sha256(self.snapshot), frozen_hash_before)
        self.assertEqual(stand_survey_registry_sha256(self.registry), registry_hash_before)

    def test_geometry_override_cannot_clear_original_morphology_conflicts(self):
        # An independently bound target override may change map geometry;
        # it must not silently erase the original conflicting source evidence.
        wall = self.projections['arrival'].projected_snapshot.candidate_for(WALL_UID)
        clear_geometry = self.snapshot.candidate_for('survey_candidate_0002').geometry
        decision = evaluate_candidate_target_admission(wall, self.costmap, target_geometry=clear_geometry)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reasons, ('unresolved_morphology_conflict',))
        self.assertTrue(decision.static_map_evidence.admitted)
        self.assertEqual(len(decision.unresolved_morphology_conflict_sha256), 2)


if __name__ == '__main__':
    unittest.main()
