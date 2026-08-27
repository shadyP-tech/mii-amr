from __future__ import annotations

import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidateFrameProjectionError,
    CandidatePlanningFrame,
    project_candidate_snapshot_to_planning_frame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    CandidatePoint2D,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    STATUS_PENDING_CAMERA,
    StandSurveyRegistry,
    SurveyCandidate,
    stand_survey_registry_sha256,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
)


_MAP_SHA = "a" * 64
_CONFIG_SHA = "c" * 64


def _frozen_candidate(
    x_m: float,
    y_m: float,
    *,
    source_registry_sha256: str,
) -> FrozenCandidate:
    return FrozenCandidate(
        candidate_uid="survey_candidate_0003",
        geometry=CandidateGeometry(
            x_m=x_m,
            y_m=y_m,
            radius_m=0.06,
            uncertainty_m=0.02,
            keepout_radius_m=0.34,
        ),
        source=CandidateSource(
            source_kind="single_view_requires_camera_validation",
            source_artifact_sha256=source_registry_sha256,
            detector_config_sha256=_CONFIG_SHA,
            observation_ids=("stand_observation_0001",),
        ),
        confidence=0.8,
        hit_count=4,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
    )


def _snapshot(
    x_m: float,
    y_m: float,
    registry: StandSurveyRegistry,
):
    return new_candidate_snapshot(
        snapshot_id="projection_test",
        created_unix_sec=2.0,
        planning_frame="map",
        map_bundle_sha256=_MAP_SHA,
        candidates=(
            _frozen_candidate(
                x_m,
                y_m,
                source_registry_sha256=stand_survey_registry_sha256(registry),
            ),
        ),
    )


def _registry(
    x_m: float,
    y_m: float,
    provenance: CandidateFrameProvenance | None,
) -> StandSurveyRegistry:
    return StandSurveyRegistry(
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        survey_id="survey",
        planning_frame="map",
        map_bundle_sha256=_MAP_SHA,
        candidates=(
            SurveyCandidate(
                candidate_uid="survey_candidate_0003",
                x_m=x_m,
                y_m=y_m,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.34,
                confidence=0.8,
                hit_count=4,
                first_seen_sec=1.0,
                last_seen_sec=2.0,
                source_observation_ids=("stand_observation_0001",),
                viewpoint_ids=("survey_vp_001",),
                status=STATUS_PENDING_CAMERA,
                frame_provenance=provenance,
            ),
        ),
    )


class CandidateFrameProjectionTests(unittest.TestCase):
    def test_failed_run_candidate_is_projected_into_current_map_frame(self):
        frozen_point = CandidatePoint2D(1.731397076, 0.651743266)
        t0 = PlanarTransform2D(
            -1.602698535,
            -0.528703478,
            0.018426325,
        )
        t1 = PlanarTransform2D(
            -1.638358757,
            -0.437467937,
            -0.108542892,
        )
        provenance = CandidateFrameProvenance.from_frozen_map_observation(
            map_frame="map",
            odom_frame="odom",
            frozen_map_point=frozen_point,
            frozen_map_from_odom=t0,
            source_evidence_id="d" * 64,
        )
        registry = _registry(
            frozen_point.x_m,
            frozen_point.y_m,
            provenance,
        )
        source = _snapshot(frozen_point.x_m, frozen_point.y_m, registry)
        source_hash = candidate_snapshot_sha256(source)

        projection = project_candidate_snapshot_to_planning_frame(
            source,
            registry,
            CandidatePlanningFrame(Pose2D(1.2, 0.2, 0.7), t1),
        )

        projected = projection.projected_snapshot.candidates[0].geometry
        self.assertAlmostEqual(projected.x_m, 1.8184, places=4)
        self.assertAlmostEqual(projected.y_m, 0.3113, places=4)
        self.assertAlmostEqual(
            projection.candidate_results[0][1].diagnostics.candidate_map_displacement_m,
            0.3514,
            places=4,
        )
        self.assertEqual(
            candidate_snapshot_sha256(source),
            source_hash,
        )
        self.assertNotEqual(
            candidate_snapshot_sha256(projection.projected_snapshot),
            source_hash,
        )
        self.assertFalse(projection.motion_authorized)

    def test_missing_provenance_fails_closed(self):
        registry = _registry(1.0, 1.0, None)
        source = _snapshot(1.0, 1.0, registry)
        with self.assertRaises(CandidateFrameProjectionError) as caught:
            project_candidate_snapshot_to_planning_frame(
                source,
                registry,
                CandidatePlanningFrame(
                    Pose2D(0.0, 0.0, 0.0),
                    PlanarTransform2D(0.0, 0.0, 0.0),
                ),
            )
        self.assertEqual(caught.exception.code, "frame_provenance_missing")

    def test_fused_canonical_odom_geometry_needs_no_single_frozen_pair(self):
        provenance = CandidateFrameProvenance(
            map_frame="map",
            odom_frame="odom",
            canonical_odom_point=CandidatePoint2D(2.0, -1.0),
            source_evidence_id="fusion_evidence",
        )
        registry = _registry(9.0, 9.0, provenance)
        projection = project_candidate_snapshot_to_planning_frame(
            _snapshot(9.0, 9.0, registry),
            registry,
            CandidatePlanningFrame(
                Pose2D(0.0, 0.0, 0.0),
                PlanarTransform2D(0.5, 0.25, 0.0),
            ),
        )
        geometry = projection.projected_snapshot.candidates[0].geometry
        self.assertEqual((geometry.x_m, geometry.y_m), (2.5, -0.75))
        self.assertIsNone(
            projection.candidate_results[0][1].diagnostics.candidate_map_displacement_m
        )

    def test_snapshot_rejects_registry_with_mutated_frame_provenance(self):
        source_registry = _registry(
            1.0,
            0.0,
            CandidateFrameProvenance(
                map_frame="map",
                odom_frame="odom",
                canonical_odom_point=CandidatePoint2D(1.0, 0.0),
                source_evidence_id="original",
            ),
        )
        snapshot = _snapshot(1.0, 0.0, source_registry)
        mutated_registry = _registry(
            1.0,
            0.0,
            CandidateFrameProvenance(
                map_frame="map",
                odom_frame="odom",
                canonical_odom_point=CandidatePoint2D(1.0, 0.5),
                source_evidence_id="mutated",
            ),
        )

        with self.assertRaises(CandidateFrameProjectionError) as caught:
            project_candidate_snapshot_to_planning_frame(
                snapshot,
                mutated_registry,
                CandidatePlanningFrame(
                    Pose2D(0.0, 0.0, 0.0),
                    PlanarTransform2D(0.0, 0.0, 0.0),
                ),
            )

        self.assertEqual(caught.exception.code, "source_registry_mismatch")


if __name__ == "__main__":
    unittest.main()
