from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
    write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand


_MAP_SHA = "a" * 64


def _provenance(transform: PlanarTransform2D, certificate_sha256: str):
    return {
        "selected_observation": "selected",
        "provenance": {
            "observer_version": "observer-v6-frozen-odom",
            "map_frame": "map",
            "map_bundle_sha256": _MAP_SHA,
            "runtime_config": {
                "frozen_odom_observation_geometry": {
                    "schema_version": 1,
                    "mode": "frozen_map_from_odom",
                    "source_frames": {
                        "map_frame": "map",
                        "odom_frame": "odom",
                        "base_frame": "base_footprint",
                    },
                    "scan_tf_target_frame": "odom",
                    "map_from_odom": {
                        "x_m": transform.x_m,
                        "y_m": transform.y_m,
                        "yaw_rad": transform.yaw_rad,
                    },
                    "odom_execution_certificate_sha256": certificate_sha256,
                }
            },
        },
    }


def _stand(
    *,
    stand_id: str,
    observation_id: str,
    point: CandidatePoint2D,
    transform: PlanarTransform2D,
    timestamp: float,
    certificate_sha256: str,
) -> ConfirmedStand:
    return ConfirmedStand(
        stand_id=stand_id,
        x_m=point.x_m,
        y_m=point.y_m,
        confidence=0.8,
        hit_count=3,
        first_seen_sec=timestamp,
        last_seen_sec=timestamp,
        first_confirmed_at_sec=timestamp,
        source_observation_ids=(observation_id,),
        provenance=_provenance(transform, certificate_sha256),
    )


class CandidateFrameRegistryTests(unittest.TestCase):
    def test_cross_leg_map_shift_fuses_in_canonical_odom_and_round_trips(self):
        canonical = CandidatePoint2D(3.3553, 1.1188)
        t0 = PlanarTransform2D(-1.6027, -0.5287, 0.01843)
        t1 = PlanarTransform2D(-1.6384, -0.4375, -0.10854)
        p0 = current_map_point_from_canonical_odom(canonical, t0)
        p1 = current_map_point_from_canonical_odom(canonical, t1)
        self.assertGreater(
            ((p1.x_m - p0.x_m) ** 2 + (p1.y_m - p0.y_m) ** 2) ** 0.5,
            0.18,
        )
        registry = StandSurveyRegistry(
            schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
            survey_id="survey",
            planning_frame="map",
            map_bundle_sha256=_MAP_SHA,
        )
        config = CoverageSurveyConfig(
            minimum_candidate_hits=1,
            minimum_distinct_viewpoints=1,
        )
        registry = fuse_confirmed_stands(
            registry,
            (
                _stand(
                    stand_id="stand_1",
                    observation_id="observation_1",
                    point=p0,
                    transform=t0,
                    timestamp=1.0,
                    certificate_sha256="b" * 64,
                ),
            ),
            viewpoint_id="survey_vp_001",
            config=config,
        )
        registry = fuse_confirmed_stands(
            registry,
            (
                _stand(
                    stand_id="stand_2",
                    observation_id="observation_2",
                    point=p1,
                    transform=t1,
                    timestamp=2.0,
                    certificate_sha256="c" * 64,
                ),
            ),
            viewpoint_id="survey_vp_002",
            config=config,
        )

        self.assertEqual(len(registry.candidates), 1)
        candidate = registry.candidates[0]
        self.assertEqual(
            candidate.source_observation_ids,
            ("observation_1", "observation_2"),
        )
        self.assertIsNotNone(candidate.frame_provenance)
        assert candidate.frame_provenance is not None
        self.assertAlmostEqual(
            candidate.frame_provenance.canonical_odom_point.x_m,
            canonical.x_m,
            places=10,
        )
        self.assertAlmostEqual(
            candidate.frame_provenance.canonical_odom_point.y_m,
            canonical.y_m,
            places=10,
        )
        self.assertAlmostEqual(candidate.x_m, p1.x_m, places=10)
        self.assertAlmostEqual(candidate.y_m, p1.y_m, places=10)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stand_registry.json"
            write_stand_survey_registry(path, registry)
            loaded = load_stand_survey_registry(path)
        self.assertEqual(loaded, registry)

    def test_malformed_frozen_provenance_is_not_downgraded_to_legacy(self):
        registry = StandSurveyRegistry(
            schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
            survey_id="survey",
            planning_frame="map",
            map_bundle_sha256=_MAP_SHA,
        )
        malformed = _stand(
            stand_id="stand_1",
            observation_id="observation_1",
            point=CandidatePoint2D(1.0, 1.0),
            transform=PlanarTransform2D(0.0, 0.0, 0.0),
            timestamp=1.0,
            certificate_sha256="b" * 64,
        )
        malformed.provenance["provenance"]["runtime_config"].pop(
            "frozen_odom_observation_geometry"
        )
        with self.assertRaisesRegex(ValueError, "missing frame evidence"):
            fuse_confirmed_stands(
                registry,
                (malformed,),
                viewpoint_id="survey_vp_001",
                config=CoverageSurveyConfig(),
            )


if __name__ == "__main__":
    unittest.main()
