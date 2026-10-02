from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
    project_candidate_snapshot_to_planning_frame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidatePoint2D,
    current_map_point_from_canonical_odom,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    CoverageSurveyConfig,
    StandSurveyRegistry,
    fuse_confirmed_stands,
    load_stand_survey_registry,
    stand_survey_registry_sha256,
    write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
)
from tests.aufgabe04.test_candidate_frame_registry import _provenance


_MAP_SHA = "a" * 64


def epoch_stand(x_m, y_m, serial, *, hits=3, transform=None):
    point = CandidatePoint2D(x_m, y_m)
    provenance = {}
    if transform is not None:
        point = current_map_point_from_canonical_odom(point, transform)
        provenance = _provenance(transform, "b" * 64)
    return ConfirmedStand(
        stand_id=f"stand_{serial}",
        x_m=point.x_m,
        y_m=point.y_m,
        confidence=0.8,
        hit_count=hits,
        first_seen_sec=float(serial),
        last_seen_sec=float(serial),
        first_confirmed_at_sec=float(serial),
        source_observation_ids=tuple(f"observation_{serial}_{i}" for i in range(hits)),
        provenance=provenance,
    )


def empty_registry():
    return StandSurveyRegistry(
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        survey_id="uncertainty_survey",
        planning_frame="map",
        map_bundle_sha256=_MAP_SHA,
    )


def fuse(registry, stand, serial, *, config=CoverageSurveyConfig()):
    return fuse_confirmed_stands(
        registry, (stand,), viewpoint_id=f"survey_vp_{serial:03}", config=config,
    )


class CandidateFusionUncertaintyTests(unittest.TestCase):
    def test_recorded_viewpoint_disagreements_expand_both_envelopes(self):
        for disagreement_m in (0.034, 0.060):
            with self.subTest(disagreement_m=disagreement_m):
                registry = fuse(empty_registry(), epoch_stand(1.0, 1.0, 1), 1)
                original = registry.candidates[0]
                registry = fuse(registry, epoch_stand(1.0 + disagreement_m, 1.0, 2), 2)
                candidate = registry.candidates[0]
                self.assertEqual(candidate.candidate_uid, original.candidate_uid)
                self.assertEqual(candidate.viewpoint_ids, ("survey_vp_001", "survey_vp_002"))
                self.assertEqual(candidate.hit_count, 6)
                self.assertAlmostEqual(candidate.x_m, 1.0 + disagreement_m / 2)
                self.assertAlmostEqual(candidate.uncertainty_m, 0.02 + disagreement_m / 2)
                self.assertAlmostEqual(candidate.keepout_radius_m, 0.31 + disagreement_m / 2)

    def test_large_historical_hit_count_cannot_hide_new_center_disagreement(self):
        registry = fuse(empty_registry(), epoch_stand(1.0, 1.0, 1, hits=27), 1)
        registry = fuse(registry, epoch_stand(1.060, 1.0, 2, hits=3), 2)
        candidate = registry.candidates[0]
        self.assertAlmostEqual(candidate.x_m, 1.006)
        self.assertAlmostEqual(candidate.uncertainty_m, 0.074)
        self.assertAlmostEqual(candidate.keepout_radius_m, 0.364)

    def test_frame_changes_do_not_count_as_physical_center_disagreement(self):
        t0 = PlanarTransform2D(3.0, 1.0, 0.4)
        t1 = PlanarTransform2D(-2.0, 5.0, -1.0)
        for separation_m in (0.0, 0.060):
            with self.subTest(separation_m=separation_m):
                registry = fuse(empty_registry(), epoch_stand(1.0, 1.0, 1, transform=t0), 1)
                registry = fuse(registry, epoch_stand(1.0 + separation_m, 1.0, 2, transform=t1), 2)
                candidate = registry.candidates[0]
                self.assertEqual(len(registry.candidates), 1)
                self.assertAlmostEqual(candidate.uncertainty_m, 0.02 + separation_m / 2)
                self.assertAlmostEqual(candidate.keepout_radius_m, 0.31 + separation_m / 2)
                self.assertAlmostEqual(candidate.frame_provenance.canonical_odom_point.x_m,
                                       1.0 + separation_m / 2)

    def test_later_fusion_keeps_every_previous_uncertainty_and_keepout_envelope(self):
        registry = empty_registry()
        previous_candidates = []
        for serial, center in enumerate(((1.0, 1.0), (1.06, 1.0), (1.03, 1.09)), start=1):
            registry = fuse(registry, epoch_stand(*center, serial), serial)
            candidate = registry.candidates[0]
            for previous in previous_candidates:
                shift = math.dist((candidate.x_m, candidate.y_m), (previous.x_m, previous.y_m))
                self.assertGreaterEqual(candidate.uncertainty_m + 1e-12, previous.uncertainty_m + shift)
                self.assertGreaterEqual(candidate.keepout_radius_m + 1e-12, previous.keepout_radius_m + shift)
            previous_candidates.append(candidate)

    def test_full_replay_does_not_expand_envelopes_or_change_identity(self):
        first = epoch_stand(1.0, 1.0, 1)
        second = epoch_stand(1.06, 1.0, 2)
        registry = fuse(fuse(empty_registry(), first, 1), second, 2)
        self.assertEqual(fuse(fuse(registry, first, 1), second, 2), registry)

    def test_association_limit_does_not_expand_with_uncertainty(self):
        registry = fuse(empty_registry(), epoch_stand(1.0, 1.0, 1), 1)
        registry = fuse(registry, epoch_stand(1.179, 1.0, 2), 2)
        center = registry.candidates[0].x_m
        registry = fuse(registry, epoch_stand(center + 0.181, 1.0, 3), 3)
        self.assertEqual(len(registry.candidates), 2)
        self.assertEqual(registry.candidates[0].viewpoint_ids, ("survey_vp_001", "survey_vp_002"))
        self.assertEqual(registry.candidates[1].viewpoint_ids, ("survey_vp_003",))

    def test_existing_larger_envelopes_are_never_replaced_by_config_defaults(self):
        registry = fuse(empty_registry(), epoch_stand(1.0, 1.0, 1), 1)
        registry = replace(registry, candidates=(replace(registry.candidates[0],
            uncertainty_m=0.12, keepout_radius_m=0.41),))
        registry = fuse(registry, epoch_stand(1.06, 1.0, 2), 2)
        self.assertAlmostEqual(registry.candidates[0].uncertainty_m, 0.15)
        self.assertAlmostEqual(registry.candidates[0].keepout_radius_m, 0.44)

    def test_persistence_and_fresh_frame_projection_preserve_expanded_envelopes(self):
        transform = PlanarTransform2D(0.0, 0.0, 0.0)
        registry = fuse(empty_registry(), epoch_stand(1.0, 1.0, 1, transform=transform), 1)
        registry = fuse(registry, epoch_stand(1.06, 1.0, 2, transform=transform), 2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            write_stand_survey_registry(path, registry)
            loaded = load_stand_survey_registry(path)
        self.assertEqual(loaded, registry)
        candidate = registry.candidates[0]
        snapshot = new_candidate_snapshot(
            snapshot_id="fused_uncertainty", created_unix_sec=2.0, planning_frame="map",
            map_bundle_sha256=_MAP_SHA,
            candidates=(FrozenCandidate(
                candidate_uid=candidate.candidate_uid,
                geometry=CandidateGeometry(candidate.x_m, candidate.y_m, candidate.radius_m,
                                           candidate.uncertainty_m, candidate.keepout_radius_m),
                source=CandidateSource("lidar/exact_two_multi_view", stand_survey_registry_sha256(registry),
                                       "c" * 64, candidate.source_observation_ids),
                confidence=candidate.confidence, hit_count=candidate.hit_count,
                first_seen_sec=candidate.first_seen_sec, last_seen_sec=candidate.last_seen_sec,
            ),),
        )
        projection = project_candidate_snapshot_to_planning_frame(snapshot, loaded,
            CandidatePlanningFrame(Pose2D(0.0, 0.0, 0.0), PlanarTransform2D(2.0, -1.0, 0.5)))
        geometry = projection.projected_snapshot.candidates[0].geometry
        self.assertAlmostEqual(geometry.uncertainty_m, 0.05)
        self.assertAlmostEqual(geometry.keepout_radius_m, 0.34)
        self.assertNotAlmostEqual(geometry.x_m, candidate.x_m)


if __name__ == "__main__":
    unittest.main()
