import dataclasses
import json
import math
import unittest

from scripts.aufgabe04.perception.lidar_stand_morphology import (
    StandWidthProfile,
    assess_stand_observation_track,
    assess_stand_width_samples,
    evaluate_stand_morphology_admission,
    lidar_detector_config_for_stand_width,
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.perception.stand_observation import (
    ObservationProvenance,
    StandObservation,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    CoverageSurveyConfig,
    StandSurveyRegistry,
    fuse_confirmed_stands,
)


def _observation(index: int, width_m: float) -> StandObservation:
    return StandObservation(
        observation_id=f"observation_{index:02d}",
        candidate_id="candidate_01",
        x_m=1.0,
        y_m=0.0,
        bearing_rad=0.0,
        distance_m=1.0,
        approximate_width_m=width_m,
        point_count=3,
        confidence=0.8,
        observed_at_sec=float(index),
        provenance=ObservationProvenance(
            schema_version=2,
            observer_version="test",
            resolved_scan_topic="/scan",
            scan_frame="base_scan",
            map_frame="map",
            base_frame="base_footprint",
            localization_source="amcl",
            scan_stamp_sec=float(index),
            tf_lookup_stamp_sec=float(index),
            tf_age_sec=0.0,
            runtime_config={},
        ),
    )


def _confirmed_stand(
    stand_id: str,
    observations: tuple[StandObservation, ...],
) -> ConfirmedStand:
    return ConfirmedStand(
        stand_id=stand_id,
        x_m=1.0,
        y_m=0.0,
        confidence=0.8,
        hit_count=len(observations),
        first_seen_sec=observations[0].observed_at_sec,
        last_seen_sec=observations[-1].observed_at_sec,
        first_confirmed_at_sec=observations[2].observed_at_sec,
        source_observation_ids=tuple(
            observation.observation_id for observation in observations
        ),
        provenance={},
    )


class StandWidthProfileTest(unittest.TestCase):
    def test_profile_is_derived_from_physical_stand_radius(self):
        profile = stand_width_profile_from_radius(0.06)

        self.assertAlmostEqual(profile.expected_diameter_m, 0.12)
        self.assertAlmostEqual(profile.detector_min_width_m, 0.03)
        self.assertAlmostEqual(profile.detector_max_width_m, 0.18)
        self.assertAlmostEqual(profile.track_median_max_width_m, 0.15)
        self.assertAlmostEqual(
            profile.track_max_median_absolute_deviation_m,
            0.036,
        )
        evidence = profile.to_evidence_dict()
        self.assertEqual(evidence["schema_version"], 1)
        self.assertEqual(
            evidence["track_width_gates"]["minimum_observation_count"],
            3,
        )
        json.dumps(evidence, allow_nan=False)

    def test_detector_config_replaces_only_profile_owned_width_bounds(self):
        base = LidarStandDetectorConfig(
            min_range_m=0.10,
            max_range_m=2.75,
            max_cluster_gap_m=0.07,
            min_cluster_points=4,
            min_width_m=0.01,
            max_width_m=0.90,
        )

        configured = lidar_detector_config_for_stand_width(
            stand_width_profile_from_radius(0.06),
            base_config=base,
        )

        self.assertAlmostEqual(configured.min_width_m, 0.03)
        self.assertAlmostEqual(configured.max_width_m, 0.18)
        self.assertEqual(configured.min_range_m, 0.10)
        self.assertEqual(configured.max_range_m, 2.75)
        self.assertEqual(configured.max_cluster_gap_m, 0.07)
        self.assertEqual(configured.min_cluster_points, 4)

    def test_invalid_or_internally_inconsistent_profiles_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "stand_radius_m"):
            stand_width_profile_from_radius(0.0)
        with self.assertRaisesRegex(ValueError, "detector_min_diameter_ratio"):
            stand_width_profile_from_radius(
                0.06,
                detector_min_diameter_ratio=1.0,
            )
        with self.assertRaisesRegex(ValueError, "track_median_max"):
            stand_width_profile_from_radius(
                0.06,
                detector_max_diameter_ratio=1.25,
                track_median_max_diameter_ratio=1.50,
            )
        invalid = StandWidthProfile(
            expected_diameter_m=0.12,
            detector_lower_tolerance_m=0.09,
            detector_upper_tolerance_m=0.03,
            track_median_upper_tolerance_m=0.04,
            track_max_median_absolute_deviation_m=0.03,
            minimum_track_inlier_fraction=0.75,
            minimum_track_observation_count=3,
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            lidar_detector_config_for_stand_width(invalid)


class StandWidthAssessmentTest(unittest.TestCase):
    def setUp(self):
        self.profile = stand_width_profile_from_radius(0.06)

    def test_stand_like_tracks_at_latest_run_medians_are_admitted_independently(self):
        width_tracks = (
            (0.041, 0.047, 0.052, 0.052, 0.056, 0.061, 0.083),
            (0.038, 0.049, 0.055, 0.057, 0.057, 0.064, 0.079),
            (0.044, 0.053, 0.059, 0.062, 0.062, 0.069, 0.081),
        )

        assessments = tuple(
            assess_stand_width_samples(widths, profile=self.profile)
            for widths in width_tracks
        )

        self.assertTrue(all(item.accepted for item in assessments))
        self.assertEqual(
            tuple(round(item.statistics.median_width_m, 3) for item in assessments),
            (0.052, 0.057, 0.062),
        )
        self.assertTrue(
            all(item.rejection_reasons == () for item in assessments)
        )

    def test_one_gross_width_outlier_does_not_override_robust_track_support(self):
        assessment = assess_stand_width_samples(
            (0.052, 0.054, 0.056, 0.058, 0.060, 0.062, 0.083, 0.4212),
            profile=self.profile,
        )

        self.assertTrue(assessment.accepted)
        self.assertAlmostEqual(assessment.statistics.median_width_m, 0.059)
        self.assertAlmostEqual(assessment.statistics.inlier_fraction, 0.875)
        self.assertLess(
            assessment.statistics.upper_quartile_width_m,
            self.profile.track_upper_quartile_max_width_m,
        )

    def test_latest_run_broad_false_candidate_shape_is_rejected(self):
        # Exact sorted widths from the latest broad false track. Fifteen of
        # sixteen exceed the 0.12 m survey-envelope reference diameter.
        false_widths = (
            0.0352,
            0.1251,
            0.1605,
            0.1622,
            0.1644,
            0.1969,
            0.2273,
            0.2289,
            0.2292,
            0.2341,
            0.2342,
            0.2613,
            0.2711,
            0.3319,
            0.3376,
            0.4212,
        )

        assessment = assess_stand_width_samples(
            false_widths,
            profile=self.profile,
        )

        self.assertFalse(assessment.accepted)
        self.assertAlmostEqual(assessment.statistics.median_width_m, 0.22905)
        self.assertEqual(
            assessment.rejection_reasons,
            (
                "median_width_above_maximum",
                "upper_quartile_width_above_maximum",
                "median_absolute_deviation_above_maximum",
                "width_inlier_fraction_below_minimum",
            ),
        )
        evidence = assessment.to_evidence_dict()
        self.assertEqual(
            evidence["rejection_reasons"],
            list(assessment.rejection_reasons),
        )
        self.assertFalse(evidence["gates"]["median_width_met"])
        json.dumps(evidence, allow_nan=False)

    def test_tight_producer_censoring_would_hide_the_false_track(self):
        # A broad fixture can produce a narrow-looking remainder if the
        # proposal producer censors its wider samples before track admission.
        synthetic_complete_widths = (
            0.0352,
            0.1251,
            0.1380,
            0.1520,
            0.1690,
            0.1880,
            0.2070,
            0.2210,
            0.2372,
            0.2510,
            0.2730,
            0.2980,
            0.3260,
            0.3540,
            0.3890,
            0.4212,
        )
        censored = tuple(
            width
            for width in synthetic_complete_widths
            if width <= self.profile.detector_max_width_m
        )

        self.assertFalse(
            assess_stand_width_samples(
                synthetic_complete_widths,
                profile=self.profile,
            ).accepted
        )
        self.assertTrue(
            assess_stand_width_samples(
                censored,
                profile=self.profile,
            ).accepted
        )

    def test_batch_admission_joins_source_ids_without_top_k_selection(self):
        true_observations = tuple(
            _observation(index, width)
            for index, width in enumerate((0.052, 0.057, 0.062), start=1)
        )
        false_observations = tuple(
            _observation(index, width)
            for index, width in enumerate(
                (0.188, 0.207, 0.221, 0.237, 0.251, 0.273),
                start=10,
            )
        )
        admission = evaluate_stand_morphology_admission(
            (
                _confirmed_stand("true_stand", true_observations),
                _confirmed_stand("broad_fixture", false_observations),
            ),
            (*true_observations, *false_observations),
            profile=self.profile,
        )

        self.assertEqual(
            tuple(stand.stand_id for stand in admission.admitted_stands),
            ("true_stand",),
        )
        self.assertEqual(
            tuple(stand.stand_id for stand in admission.rejected_stands),
            ("broad_fixture",),
        )
        self.assertEqual(
            admission.to_evidence_dict()["selection_policy"],
            "independent_per_track_no_expected_count_ranking",
        )

    def test_batch_admission_fails_closed_on_missing_or_cross_claimed_ids(self):
        observations = tuple(
            _observation(index, width)
            for index, width in enumerate((0.052, 0.057, 0.062), start=1)
        )
        stand = _confirmed_stand("stand_a", observations)

        with self.assertRaisesRegex(ValueError, "missing observations"):
            evaluate_stand_morphology_admission(
                (stand,),
                observations[:-1],
                profile=self.profile,
            )
        with self.assertRaisesRegex(ValueError, "multiple confirmed stands"):
            evaluate_stand_morphology_admission(
                (stand, _confirmed_stand("stand_b", observations)),
                observations,
                profile=self.profile,
            )

    def test_two_epoch_replay_keeps_five_true_tracks_without_top_k(self):
        config = CoverageSurveyConfig()
        registry = StandSurveyRegistry(
            schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
            survey_id="survey_01",
            planning_frame="map",
            map_bundle_sha256="a" * 64,
        )

        def track(
            stand_id: str,
            start_index: int,
            x_m: float,
            widths: tuple[float, ...],
        ):
            observations = tuple(
                _observation(start_index + offset, width)
                for offset, width in enumerate(widths)
            )
            return (
                dataclasses.replace(
                    _confirmed_stand(stand_id, observations),
                    x_m=x_m,
                ),
                observations,
            )

        first_tracks = (
            track("true_01", 100, -0.8, (0.048, 0.052, 0.058)),
            track("true_02", 110, -0.4, (0.051, 0.057, 0.063)),
            track("true_03", 120, 0.0, (0.055, 0.062, 0.069)),
        )
        second_tracks = (
            track("true_04", 200, 0.4, (0.049, 0.056, 0.064)),
            track("true_05", 210, 0.8, (0.052, 0.061, 0.070)),
            track(
                "broad_fixture",
                220,
                1.2,
                (0.160, 0.197, 0.229, 0.261, 0.332, 0.421),
            ),
        )
        for viewpoint_id, tracks in (
            ("viewpoint_01", first_tracks),
            ("viewpoint_02", second_tracks),
        ):
            admission = evaluate_stand_morphology_admission(
                tuple(item[0] for item in tracks),
                tuple(
                    observation
                    for item in tracks
                    for observation in item[1]
                ),
                profile=self.profile,
            )
            registry = fuse_confirmed_stands(
                registry,
                admission.admitted_stands,
                viewpoint_id=viewpoint_id,
                config=config,
            )

        self.assertEqual(len(registry.candidates), 5)
        self.assertEqual(
            tuple(candidate.candidate_uid for candidate in registry.candidates),
            tuple(f"survey_candidate_{index:04d}" for index in range(1, 6)),
        )

    def test_insufficient_track_is_rejected_without_candidate_ranking(self):
        assessment = assess_stand_width_samples(
            (0.052, 0.061),
            profile=self.profile,
        )

        self.assertFalse(assessment.accepted)
        self.assertEqual(
            assessment.rejection_reasons,
            ("insufficient_width_observations",),
        )
        self.assertTrue(assessment.median_width_met)

    def test_observation_api_reads_approximate_widths(self):
        observations = tuple(
            _observation(index, width)
            for index, width in enumerate((0.052, 0.057, 0.062), start=1)
        )

        assessment = assess_stand_observation_track(
            observations,
            profile=self.profile,
        )

        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.statistics.sample_count, 3)
        self.assertAlmostEqual(assessment.statistics.median_width_m, 0.057)

    def test_malformed_width_evidence_raises_instead_of_becoming_a_rejection(self):
        for widths in (
            (),
            (0.0,),
            (-0.1,),
            (math.nan,),
            (math.inf,),
            ("not-a-width",),
        ):
            with self.subTest(widths=widths):
                with self.assertRaisesRegex(ValueError, "width samples"):
                    assess_stand_width_samples(widths, profile=self.profile)

        with self.assertRaisesRegex(ValueError, "StandObservation"):
            assess_stand_observation_track(
                (_observation(1, 0.052), object()),
                profile=self.profile,
            )


if __name__ == "__main__":
    unittest.main()
