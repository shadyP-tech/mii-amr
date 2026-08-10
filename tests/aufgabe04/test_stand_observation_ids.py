import contextlib
import io
import unittest

from scripts.aufgabe04.perception.models import StandCandidate
from scripts.aufgabe04.perception.stand_explorer_node import build_parser
from scripts.aufgabe04.perception.stand_observation import (
    ObservationProvenance,
    PlanarTransform,
    observation_id_from_index,
    observations_from_candidates,
    validated_observation_id_scope,
)


def _candidate() -> StandCandidate:
    return StandCandidate(
        candidate_id="candidate_1",
        bearing_rad=0.0,
        distance_m=1.0,
        approximate_width_m=0.12,
        center_x_m=1.0,
        center_y_m=0.0,
        point_count=4,
        confidence=0.8,
    )


def _provenance() -> ObservationProvenance:
    return ObservationProvenance(
        schema_version=2,
        observer_version="test",
        resolved_scan_topic="/scan",
        scan_frame="base_scan",
        map_frame="map",
        base_frame="base_footprint",
        localization_source="amcl",
        scan_stamp_sec=10.0,
        tf_lookup_stamp_sec=10.0,
        tf_age_sec=0.0,
        runtime_config={},
    )


class StandObservationIdTest(unittest.TestCase):
    def test_legacy_id_is_unchanged_when_scope_is_omitted(self):
        self.assertEqual(
            observation_id_from_index(1),
            "stand_observation_000001",
        )

    def test_distinct_epoch_scopes_make_same_local_index_globally_unique(self):
        common = {
            "candidates": (_candidate(),),
            "transform_scan_to_map": PlanarTransform(0.0, 0.0, 0.0),
            "observed_at_sec": 10.0,
            "provenance": _provenance(),
            "start_index": 1,
        }

        first_epoch = observations_from_candidates(
            **common,
            observation_id_scope="survey_vp_001",
        )
        second_epoch = observations_from_candidates(
            **common,
            observation_id_scope="survey_vp_002",
        )

        self.assertEqual(
            first_epoch[0].observation_id,
            "stand_observation_survey_vp_001_000001",
        )
        self.assertEqual(
            second_epoch[0].observation_id,
            "stand_observation_survey_vp_002_000001",
        )
        self.assertNotEqual(
            first_epoch[0].observation_id,
            second_epoch[0].observation_id,
        )

    def test_scope_validation_rejects_ambiguous_or_unsafe_values(self):
        for value in ("", "_epoch", ".", "survey vp 001", "survey/vp/001", "a" * 65):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "observation ID scope"):
                    validated_observation_id_scope(value)

    def test_mapper_validates_scope_even_for_empty_candidate_epoch(self):
        with self.assertRaisesRegex(ValueError, "observation ID scope"):
            observations_from_candidates(
                (),
                transform_scan_to_map=PlanarTransform(0.0, 0.0, 0.0),
                observed_at_sec=10.0,
                provenance=_provenance(),
                observation_id_scope="unsafe/scope",
            )

    def test_cli_exposes_scope_and_prefix_alias_with_legacy_default(self):
        parser = build_parser()

        self.assertIsNone(parser.parse_args([]).observation_id_scope)
        self.assertEqual(
            parser.parse_args(
                ["--observation-id-scope", "survey_vp_001"]
            ).observation_id_scope,
            "survey_vp_001",
        )
        self.assertEqual(
            parser.parse_args(
                ["--observation-id-prefix", "survey_vp_002"]
            ).observation_id_scope,
            "survey_vp_002",
        )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--observation-id-scope", "survey/vp/003"])


if __name__ == "__main__":
    unittest.main()
