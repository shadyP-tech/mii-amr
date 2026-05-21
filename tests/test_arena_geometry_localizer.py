import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import analyze_arena_geometry_from_bag as bag_analyzer  # noqa: E402
import arena_geometry_localizer as arena  # noqa: E402


SYNTHETIC_ARENA_CONFIG = arena.ArenaGeometryConfig(arena_width_m=1.898)


def rotate_point(point, yaw_deg):
    yaw = math.radians(yaw_deg)
    x, y = point
    return (
        math.cos(yaw) * x - math.sin(yaw) * y,
        math.sin(yaw) * x + math.cos(yaw) * y,
    )


def transform_points(points, yaw_deg=0.0, lateral_offset_m=0.0):
    transformed = []
    for x, y in points:
        transformed.append(rotate_point((x, y - lateral_offset_m), yaw_deg))
    return transformed


def rectangular_points(
    include_clean=False,
    include_heater=False,
    yaw_deg=0.0,
    lateral_offset_m=0.0,
    width_m=1.898,
):
    half_length = 3.90 / 2.0
    half_width = width_m / 2.0
    points = []
    for index in range(61):
        x = -1.50 + index * 0.05
        points.append((x, -half_width))
        points.append((x, half_width))
    if include_clean:
        for index in range(39):
            y = -0.90 + index * 0.05
            points.append((-half_length, y))
    if include_heater:
        for index in range(39):
            y = -0.90 + index * 0.05
            points.append((half_length, y))
        for y_center in (-0.35, 0.35):
            for offset_index in range(10):
                y = y_center - 0.12 + offset_index * 0.025
                points.append((half_length - 0.16, y))
    return transform_points(points, yaw_deg=yaw_deg, lateral_offset_m=lateral_offset_m)


def profile_candidate(
    side,
    heater_score,
    clean_score,
    range_m,
    validity_failed_reason=None,
):
    return arena.ShortWallClassification(
        wall_type=arena.WALL_UNKNOWN,
        reason="profile_candidate",
        observed_axis_side=side,
        confidence=max(heater_score, clean_score),
        heater_feature_score=heater_score,
        clean_feature_score=clean_score,
        classification_margin=abs(heater_score - clean_score),
        short_wall_candidate_range_m=range_m,
        short_wall_visible_width_m=1.0,
        short_wall_rmse_m=0.0,
        point_count=30,
        profile_features={
            "point_count": 30,
            "visible_width_m": 1.0,
            "line_rmse_m": 0.0,
            "depth_p75_m": 0.0,
            "depth_p90_m": 0.0,
            "depth_p95_m": 0.0,
            "protrusion_fraction": 0.0,
            "protrusion_cluster_count": 0,
            "largest_protrusion_cluster_width_m": 0.0,
            "profile_roughness_m": 0.0,
            "outer_line_support_fraction": 1.0,
            "validity_failed_reason": validity_failed_reason,
        },
        heater_profile_score=heater_score,
        clean_profile_score=clean_score,
        validity_failed_reason=validity_failed_reason,
    )


class ArenaGeometryLocalizerTest(unittest.TestCase):
    def test_long_walls_only_are_rejected_as_non_unique(self):
        result = arena.analyze_points(rectangular_points(), SYNTHETIC_ARENA_CONFIG)

        self.assertFalse(result.success)
        self.assertTrue(result.long_wall_fit.ok)
        self.assertFalse(result.pose_unique)
        self.assertFalse(result.yaw_ambiguity_resolved)
        self.assertEqual(result.failure_reason, "pose_not_unique")
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_UNKNOWN)

    def test_pairwise_clean_heater_walls_resolve_pose_prior(self):
        result = arena.analyze_points(
            rectangular_points(include_clean=True, include_heater=True),
            SYNTHETIC_ARENA_CONFIG,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.pose_unique)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_HEATER)
        self.assertEqual(result.short_wall_classification.observed_axis_side, "axis_positive")
        self.assertEqual(
            result.short_wall_classification.reason,
            "pairwise_profile_heater_clean_valid",
        )
        self.assertEqual(
            result.short_wall_classification.selected_assignment,
            "positive_heater",
        )
        self.assertEqual(
            result.short_wall_candidates["axis_negative"].wall_type,
            arena.WALL_CLEAN,
        )
        self.assertIsNotNone(result.estimated_pose_prior)
        self.assertAlmostEqual(result.estimated_pose_prior.x, 0.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.y, 0.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.yaw_deg, 0.0, delta=2.0)
        self.assertIsNotNone(result.estimated_covariance)

    def test_single_short_wall_is_rejected_as_pairwise_incomplete(self):
        result = arena.analyze_points(
            rectangular_points(include_heater=True),
            SYNTHETIC_ARENA_CONFIG,
        )

        self.assertFalse(result.success)
        self.assertFalse(result.pose_unique)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_UNKNOWN)
        self.assertEqual(result.short_wall_classification.reason, "pairwise_profile_candidate_invalid")
        self.assertGreater(result.short_wall_candidates["axis_positive"].heater_profile_score, 0.75)

    def test_rotated_and_laterally_offset_scan_estimates_y_and_yaw(self):
        result = arena.analyze_points(
            rectangular_points(
                include_clean=True,
                include_heater=True,
                yaw_deg=12.0,
                lateral_offset_m=0.18,
            ),
            SYNTHETIC_ARENA_CONFIG,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.estimated_pose_prior.y, 0.18, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.yaw_deg, -12.0, delta=2.0)

    def test_map_frame_calibration_is_applied_to_pose_prior(self):
        config = arena.ArenaGeometryConfig(
            arena_width_m=1.898,
            map_center_x=1.0,
            map_center_y=2.0,
            map_yaw_deg=90.0,
        )
        result = arena.analyze_points(
            rectangular_points(include_clean=True, include_heater=True),
            config,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.estimated_pose_prior.x, 1.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.y, 2.0, delta=0.08)
        self.assertAlmostEqual(result.estimated_pose_prior.yaw_deg, 90.0, delta=2.0)

    def test_json_output_contains_required_commit_a_keys(self):
        result = arena.analyze_points(
            rectangular_points(include_clean=True),
            SYNTHETIC_ARENA_CONFIG,
        )
        data = result.to_dict()

        for key in [
            "success",
            "failure_reason",
            "pose_unique",
            "yaw_ambiguity_resolved",
            "estimated_pose_prior",
            "estimated_covariance",
            "long_wall_fit",
            "short_wall_candidates",
            "short_wall_classification",
            "diagnostics",
        ]:
            self.assertIn(key, data)

    def test_heater_side_width_profile_matches_lidar_width(self):
        config = arena.ArenaGeometryConfig(max_wall_separation_error_m=0.05)
        result = arena.analyze_points(
            rectangular_points(width_m=2.016),
            config,
        )

        self.assertTrue(result.long_wall_fit.ok)
        self.assertEqual(
            result.long_wall_fit.matched_width_profile_label,
            "heater_side_width",
        )
        self.assertEqual(result.long_wall_fit.width_match_mode, "dual")
        self.assertFalse(result.long_wall_fit.width_match_ambiguous)

    def test_clean_side_width_profile_matches_lidar_width(self):
        config = arena.ArenaGeometryConfig(max_wall_separation_error_m=0.05)
        result = arena.analyze_points(
            rectangular_points(width_m=1.967),
            config,
        )

        self.assertTrue(result.long_wall_fit.ok)
        self.assertEqual(
            result.long_wall_fit.matched_width_profile_label,
            "clean_side_width",
        )
        self.assertEqual(result.long_wall_fit.width_match_mode, "dual")
        self.assertFalse(result.long_wall_fit.width_match_ambiguous)

    def test_midpoint_width_profile_is_marked_ambiguous(self):
        config = arena.ArenaGeometryConfig(max_wall_separation_error_m=0.05)
        midpoint_width = (2.016 + 1.967) / 2.0
        width_match = arena.match_width_profile(midpoint_width, config)

        self.assertTrue(width_match.width_match_ambiguous)
        self.assertAlmostEqual(width_match.width_match_margin_m, 0.0)

    def test_out_of_tolerance_width_profile_fails_long_wall_fit(self):
        config = arena.ArenaGeometryConfig(max_wall_separation_error_m=0.05)
        result = arena.analyze_points(
            rectangular_points(width_m=1.80),
            config,
        )

        self.assertFalse(result.long_wall_fit.ok)
        self.assertEqual(result.failure_reason, "wall_separation_out_of_tolerance")

    def test_single_width_override_disables_dual_width_matching(self):
        config = arena.ArenaGeometryConfig(
            arena_width_m=1.90,
            max_wall_separation_error_m=0.05,
        )
        result = arena.analyze_points(
            rectangular_points(width_m=1.90),
            config,
        )

        self.assertTrue(result.long_wall_fit.ok)
        self.assertEqual(result.long_wall_fit.width_match_mode, "single")
        self.assertEqual(result.long_wall_fit.matched_width_profile_label, "arena_single")

    def test_short_wall_candidates_include_both_axis_sides(self):
        data = arena.analyze_points(
            rectangular_points(include_clean=True),
            SYNTHETIC_ARENA_CONFIG,
        ).to_dict()

        self.assertEqual(
            set(data["short_wall_candidates"].keys()),
            {"axis_negative", "axis_positive"},
        )
        self.assertEqual(
            data["short_wall_candidates"]["axis_negative"]["axis_side"],
            "axis_negative",
        )
        self.assertEqual(
            data["short_wall_candidates"]["axis_positive"]["axis_side"],
            "axis_positive",
        )

    def test_pairwise_profile_accepts_complementary_short_walls(self):
        result = arena.analyze_points(
            rectangular_points(include_clean=True, include_heater=True),
            SYNTHETIC_ARENA_CONFIG,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.pose_unique)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_HEATER)
        self.assertEqual(
            result.short_wall_classification.reason,
            "pairwise_profile_heater_clean_valid",
        )
        self.assertAlmostEqual(
            result.short_wall_classification.short_wall_range_sum_m,
            3.90,
            delta=0.01,
        )
        self.assertAlmostEqual(result.estimated_pose_prior.x, 0.0, delta=0.02)

    def test_pairwise_profile_rejects_both_heater_like_candidates(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": profile_candidate("axis_negative", 0.90, 0.10, 1.95),
            "axis_positive": profile_candidate("axis_positive", 0.85, 0.20, 1.95),
        }

        pairwise = arena.classify_short_wall_pairwise(candidates, config)
        classification = arena.select_short_wall_classification(candidates, config)

        self.assertFalse(pairwise.accepted)
        self.assertEqual(pairwise.reason, "pairwise_profile_both_heater_like")
        self.assertEqual(classification.wall_type, arena.WALL_UNKNOWN)

    def test_pairwise_profile_rejects_both_clean_like_candidates(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": profile_candidate("axis_negative", 0.10, 0.90, 1.95),
            "axis_positive": profile_candidate("axis_positive", 0.20, 0.85, 1.95),
        }

        pairwise = arena.classify_short_wall_pairwise(candidates, config)

        self.assertFalse(pairwise.accepted)
        self.assertEqual(pairwise.reason, "pairwise_profile_both_clean_like")

    def test_pairwise_profile_rejects_ambiguous_mid_confidence_scores(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": profile_candidate("axis_negative", 0.68, 0.66, 1.95),
            "axis_positive": profile_candidate("axis_positive", 0.64, 0.67, 1.95),
        }

        pairwise = arena.classify_short_wall_pairwise(candidates, config)

        self.assertFalse(pairwise.accepted)
        self.assertEqual(pairwise.reason, "pairwise_profile_ambiguous_scores")

    def test_pairwise_profile_rejects_winner_label_mismatch(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": profile_candidate("axis_negative", 0.10, 0.90, 1.95),
            "axis_positive": profile_candidate("axis_positive", 0.90, 0.10, 1.95),
        }

        def fake_heater_like(candidate, _config):
            return candidate.observed_axis_side == "axis_negative"

        def fake_clean_like(candidate, _config):
            return candidate.observed_axis_side == "axis_positive"

        with mock.patch.object(arena, "is_profile_heater_like", fake_heater_like), mock.patch.object(
            arena,
            "is_profile_clean_like",
            fake_clean_like,
        ):
            pairwise = arena.classify_short_wall_pairwise(candidates, config)

        self.assertFalse(pairwise.accepted)
        self.assertEqual(pairwise.reason, "pairwise_profile_assignment_label_mismatch")

    def test_pairwise_profile_rejects_broken_wall_before_scoring(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": profile_candidate(
                "axis_negative",
                0.90,
                0.10,
                1.95,
                validity_failed_reason="profile_line_rmse_too_high",
            ),
            "axis_positive": profile_candidate("axis_positive", 0.10, 0.90, 1.95),
        }

        pairwise = arena.classify_short_wall_pairwise(candidates, config)

        self.assertFalse(pairwise.accepted)
        self.assertEqual(pairwise.reason, "pairwise_profile_candidate_invalid")

    def test_pairwise_profile_rejects_range_sum_errors_directionally(self):
        config = arena.ArenaGeometryConfig(max_short_wall_range_sum_error_m=0.15)
        short_candidates = {
            "axis_negative": profile_candidate("axis_negative", 0.90, 0.10, 0.50),
            "axis_positive": profile_candidate("axis_positive", 0.10, 0.90, 2.00),
        }
        long_candidates = {
            "axis_negative": profile_candidate("axis_negative", 0.90, 0.10, 2.50),
            "axis_positive": profile_candidate("axis_positive", 0.10, 0.90, 2.00),
        }

        self.assertEqual(
            arena.classify_short_wall_pairwise(short_candidates, config).reason,
            "pairwise_profile_range_sum_too_short",
        )
        self.assertEqual(
            arena.classify_short_wall_pairwise(long_candidates, config).reason,
            "pairwise_profile_range_sum_too_long",
        )

    def test_pairwise_profile_candidate_order_is_key_based(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_positive": profile_candidate("axis_positive", 0.90, 0.10, 1.95),
            "axis_negative": profile_candidate("axis_negative", 0.10, 0.90, 1.95),
        }

        pairwise = arena.classify_short_wall_pairwise(candidates, config)

        self.assertTrue(pairwise.accepted)
        self.assertEqual(pairwise.assignment, "positive_heater")

    def test_profile_features_are_invariant_to_axis_sign_flip(self):
        config = arena.ArenaGeometryConfig()
        axis = (1.0, 0.0)
        flipped_axis = (-1.0, 0.0)
        normal = (0.0, 1.0)
        points = [(1.95, -0.5), (1.95, 0.0), (1.95, 0.5)]
        points += [(1.80, -0.2), (1.80, 0.2)]
        line = arena.fit_line(points[:3])

        original = arena.compute_short_wall_profile_features(
            points,
            axis,
            normal,
            "axis_positive",
            1.95,
            line,
            config,
        )
        flipped = arena.compute_short_wall_profile_features(
            points,
            flipped_axis,
            normal,
            "axis_negative",
            -1.95,
            line,
            config,
        )

        self.assertAlmostEqual(original["depth_p95_m"], flipped["depth_p95_m"])
        self.assertAlmostEqual(
            original["protrusion_fraction"],
            flipped["protrusion_fraction"],
        )

    def test_mirrored_arena_preserves_physical_heater_identification(self):
        mirrored = [(-x, y) for x, y in rectangular_points(include_clean=True, include_heater=True)]
        result = arena.analyze_points(mirrored, SYNTHETIC_ARENA_CONFIG)

        self.assertTrue(result.success)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_HEATER)
        self.assertEqual(result.short_wall_classification.observed_axis_side, "axis_negative")

    def test_noisy_clean_wall_does_not_become_heater_like(self):
        config = arena.ArenaGeometryConfig()
        points = [(-1.95 + (0.005 if index % 2 else -0.005), -0.9 + index * 0.06) for index in range(31)]
        line = arena.fit_line(points)
        features = arena.compute_short_wall_profile_features(
            points,
            (1.0, 0.0),
            (0.0, 1.0),
            "axis_negative",
            -1.95,
            line,
            config,
        )
        heater_score, clean_score = arena.score_short_wall_profile(features, config)

        self.assertLess(heater_score, config.profile_min_heater_like_score)
        self.assertGreater(clean_score, config.profile_min_clean_like_score)

    def test_complementary_short_wall_range_sum_must_be_consistent(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": arena.ShortWallClassification(
                wall_type=arena.WALL_HEATER,
                reason="heater_score_dominant",
                observed_axis_side="axis_negative",
                confidence=0.90,
                heater_feature_score=0.90,
                classification_margin=0.90,
                short_wall_candidate_range_m=0.50,
                short_wall_rmse_m=0.01,
                point_count=20,
            ),
            "axis_positive": arena.ShortWallClassification(
                wall_type=arena.WALL_CLEAN,
                reason="clean_score_dominant",
                observed_axis_side="axis_positive",
                confidence=0.90,
                clean_feature_score=0.90,
                classification_margin=0.90,
                short_wall_candidate_range_m=2.00,
                short_wall_rmse_m=0.01,
                point_count=20,
            ),
        }

        classification = arena.select_short_wall_classification(candidates, config)

        self.assertEqual(classification.wall_type, arena.WALL_AMBIGUOUS)
        self.assertEqual(classification.reason, "short_wall_range_inconsistent")
        self.assertAlmostEqual(classification.short_wall_range_sum_m, 2.50)
        self.assertAlmostEqual(classification.short_wall_range_sum_error_m, 1.40)

    def test_duplicate_valid_short_wall_types_are_rejected(self):
        config = arena.ArenaGeometryConfig()
        candidates = {
            "axis_negative": arena.ShortWallClassification(
                wall_type=arena.WALL_HEATER,
                reason="heater_score_dominant",
                observed_axis_side="axis_negative",
                confidence=0.90,
                heater_feature_score=0.90,
                classification_margin=0.90,
                short_wall_candidate_range_m=0.50,
                short_wall_rmse_m=0.01,
                point_count=20,
            ),
            "axis_positive": arena.ShortWallClassification(
                wall_type=arena.WALL_HEATER,
                reason="heater_score_dominant",
                observed_axis_side="axis_positive",
                confidence=0.85,
                heater_feature_score=0.85,
                classification_margin=0.85,
                short_wall_candidate_range_m=3.40,
                short_wall_rmse_m=0.01,
                point_count=20,
            ),
        }

        classification = arena.select_short_wall_classification(candidates, config)

        self.assertEqual(classification.wall_type, arena.WALL_AMBIGUOUS)
        self.assertEqual(classification.reason, "both_axis_candidates_valid")

    def test_forced_short_wall_classification_overrides_candidate_scores(self):
        config = arena.ArenaGeometryConfig(
            forced_short_wall_side="axis_positive",
            forced_short_wall_type=arena.WALL_CLEAN,
        )
        candidates = {
            "axis_negative": arena.ShortWallClassification(
                wall_type=arena.WALL_CLEAN,
                reason="clean_score_dominant",
                observed_axis_side="axis_negative",
                confidence=0.90,
                clean_feature_score=0.90,
                classification_margin=0.90,
                short_wall_candidate_range_m=3.00,
                short_wall_rmse_m=0.01,
                point_count=20,
            ),
            "axis_positive": arena.ShortWallClassification(
                wall_type=arena.WALL_UNKNOWN,
                reason="classification_confidence_too_low",
                observed_axis_side="axis_positive",
                confidence=0.45,
                clean_feature_score=0.45,
                heater_feature_score=0.30,
                classification_margin=0.15,
                short_wall_candidate_range_m=0.90,
                short_wall_rmse_m=0.02,
                point_count=20,
            ),
        }

        classification = arena.select_short_wall_classification(candidates, config)

        self.assertEqual(classification.wall_type, arena.WALL_CLEAN)
        self.assertEqual(classification.observed_axis_side, "axis_positive")
        self.assertEqual(classification.reason, "forced_short_wall_classification")

    def test_width_profile_does_not_override_short_wall_classification(self):
        config = arena.ArenaGeometryConfig(max_wall_separation_error_m=0.05)
        result = arena.analyze_points(
            rectangular_points(include_clean=True, include_heater=True, width_m=2.016),
            config,
        )

        self.assertTrue(result.success)
        self.assertEqual(
            result.long_wall_fit.matched_width_profile_label,
            "heater_side_width",
        )
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_HEATER)
        self.assertEqual(
            result.short_wall_classification.reason,
            "pairwise_profile_heater_clean_valid",
        )

    def test_unknown_selected_diagnostic_candidate_does_not_succeed(self):
        result = arena.analyze_points(rectangular_points(), SYNTHETIC_ARENA_CONFIG)

        self.assertFalse(result.success)
        self.assertFalse(result.pose_unique)
        self.assertEqual(result.short_wall_classification.wall_type, arena.WALL_UNKNOWN)

    def test_new_commit_a_files_do_not_contain_live_motion_or_initialpose_code(self):
        for relative in [
            "scripts/aufgabe03/arena_geometry_localizer.py",
            "scripts/aufgabe03/analyze_arena_geometry_from_bag.py",
        ]:
            text = (ROOT / relative).read_text()
            self.assertNotIn("cmd_vel", text)
            self.assertNotIn("initialpose", text)
            self.assertNotIn("PoseWithCovariance", text)

    def test_analyzer_accepts_json_samples_for_non_ros_debugging(self):
        sample_data = {
            "scan_samples": [
                {
                    "ranges": [1.0, float("inf"), 2.0],
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2.0,
                    "range_min": 0.05,
                    "range_max": 3.0,
                    "odom_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.json"
            path.write_text(json.dumps(sample_data))
            samples = arena.load_scan_samples_json(path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(len(arena.finite_scan_points(samples[0])), 2)
        self.assertEqual(len(arena.finite_scan_points(samples[0], range_stride=2)), 2)

    def test_analyzer_cli_writes_required_json_from_json_input(self):
        sample_data = {
            "scan_samples": [
                {
                    "ranges": [1.0, 2.0, 1.5],
                    "angle_min": 0.0,
                    "angle_increment": math.pi / 2.0,
                    "range_min": 0.05,
                    "range_max": 3.0,
                    "odom_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "samples.json"
            output_path = Path(tmpdir) / "diagnostics.json"
            input_path.write_text(json.dumps(sample_data))

            with contextlib.redirect_stdout(io.StringIO()):
                result = bag_analyzer.main(
                    [
                        "--input-json",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )
            written = json.loads(output_path.read_text())

        self.assertEqual(result, 0)
        self.assertIn("success", written)
        self.assertIn("long_wall_fit", written)
        self.assertEqual(written["source"]["type"], "json")
        self.assertEqual(written["source"]["range_stride"], 4)
        self.assertEqual(written["source"]["max_points"], 4000)
        self.assertIn("short_wall_candidates", written)

    def test_bag_cli_requires_exactly_one_input_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bag_analyzer.parse_args(["--output", "out.json"])
        args = bag_analyzer.parse_args(
            [
                "--input-json",
                "samples.json",
                "--output",
                "out.json",
                "--scan-stride",
                "2",
                "--range-stride",
                "3",
                "--max-points",
                "100",
            ]
        )
        self.assertEqual(args.input_json, Path("samples.json"))
        self.assertEqual(args.scan_stride, 2)
        self.assertEqual(args.range_stride, 3)
        self.assertEqual(args.max_points, 100)
        self.assertIsNone(args.arena_width_m)
        self.assertEqual(args.arena_heater_wall_width_m, 2.016)
        self.assertEqual(args.arena_clean_wall_width_m, 1.967)
        self.assertEqual(args.arena_width_match_min_margin_m, 0.015)
        self.assertEqual(args.arena_max_short_wall_range_sum_error_m, 0.15)
        self.assertIsNone(args.arena_force_short_wall_side)
        self.assertIsNone(args.arena_force_short_wall_type)

        single_width_args = bag_analyzer.parse_args(
            [
                "--input-json",
                "samples.json",
                "--output",
                "out.json",
                "--arena-width-m",
                "1.9",
            ]
        )
        config = bag_analyzer.config_from_args(single_width_args)
        self.assertEqual(config.arena_width_m, 1.9)

        forced_args = bag_analyzer.parse_args(
            [
                "--input-json",
                "samples.json",
                "--output",
                "out.json",
                "--arena-force-short-wall-side",
                "axis_positive",
                "--arena-force-short-wall-type",
                "clean",
            ]
        )
        forced_config = bag_analyzer.config_from_args(forced_args)
        self.assertEqual(forced_config.forced_short_wall_side, "axis_positive")
        self.assertEqual(forced_config.forced_short_wall_type, arena.WALL_CLEAN)


if __name__ == "__main__":
    unittest.main()
