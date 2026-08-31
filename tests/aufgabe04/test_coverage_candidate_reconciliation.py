import math
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation import (
    ACTION_REJECT_PROVISIONAL,
    ACTION_RETAIN,
    CoverageCandidateReconciliationConfig,
    reconcile_provisional_candidate_visibility,
)
from scripts.aufgabe04.navigation.planning.map_io import MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PROVISIONAL,
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyCandidate,
    SurveyViewpoint,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    lidar_visibility_receipt_from_scan,
)


MAP_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64
SURVEY_ID = "survey_01"
SOURCE_VIEWPOINT = "viewpoint_01"
CHECK_VIEWPOINT = "viewpoint_02"
SCAN_POSE = Pose2D(0.05, 0.05, 0.0)
CANDIDATE_POSE = Pose2D(0.85, 0.05, 0.0)
ANGLE_MIN = -math.pi
ANGLE_INCREMENT = math.radians(1.0)
SCAN_COUNT = 361
TARGET_INDEX = 180


def _grid(*, blocked=()) -> OccupancyGrid:
    rows = [[0 for _x in range(30)] for _y in range(30)]
    for cell in blocked:
        rows[cell.y][cell.x] = 1
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.10,
            origin=(-1.0, -1.0, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.196,
            mode="trinary",
        ),
        width=30,
        height=30,
        cells=tuple(tuple(row) for row in rows),
    )


def _candidate_cell() -> GridCell:
    return GridCell(18, 10)


def _plan(*, check_visible: bool = True, expected_count: int = 5):
    candidate_cell = _candidate_cell()
    config = CoverageSurveyConfig(expected_stand_count=expected_count)
    source = SurveyViewpoint(
        viewpoint_id=SOURCE_VIEWPOINT,
        pose=Pose2D(0.05, -0.45, math.pi / 2.0),
        cell=GridCell(10, 5),
        visible_cells=(candidate_cell,),
    )
    check = SurveyViewpoint(
        viewpoint_id=CHECK_VIEWPOINT,
        pose=SCAN_POSE,
        cell=GridCell(10, 10),
        visible_cells=(candidate_cell,) if check_visible else (),
    )
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        arena_bounds=ArenaBounds(),
        config=config,
        viewpoints=(source, check),
        surveyable_cells=(candidate_cell,),
        planned_covered_cells=(candidate_cell,),
        planned_coverage_ratio=1.0,
    )


def _candidate(**overrides) -> SurveyCandidate:
    values = {
        "candidate_uid": "survey_candidate_0006",
        "x_m": CANDIDATE_POSE.x_m,
        "y_m": CANDIDATE_POSE.y_m,
        "radius_m": 0.06,
        "uncertainty_m": 0.02,
        "keepout_radius_m": 0.31,
        "confidence": 0.91,
        "hit_count": 16,
        "first_seen_sec": 1.0,
        "last_seen_sec": 2.0,
        "source_observation_ids": ("observation_01",),
        "viewpoint_ids": (SOURCE_VIEWPOINT,),
        "status": STATUS_PROVISIONAL,
    }
    values.update(overrides)
    return SurveyCandidate(**values)


def _ranges(value):
    result = [3.0] * SCAN_COUNT
    result[TARGET_INDEX] = value
    return result


def _receipt(receipt_id: str, stamp: float, target_range, **overrides):
    values = {
        "receipt_id": receipt_id,
        "survey_id": SURVEY_ID,
        "viewpoint_id": CHECK_VIEWPOINT,
        "planning_frame": "map",
        "scan_frame": "base_scan",
        "scan_topic": "/scan",
        "map_bundle_sha256": MAP_SHA256,
        "observer_config_sha256": CONFIG_SHA256,
        "scan_stamp_sec": stamp,
        "pose_stamp_sec": stamp,
        "observer_clock_sec": stamp + 0.01,
        "scan_pose_map": SCAN_POSE,
        "angle_min_rad": ANGLE_MIN,
        "angle_increment_rad": ANGLE_INCREMENT,
        "range_min_m": 0.08,
        "range_max_m": 3.5,
        "ranges_m": _ranges(target_range),
    }
    values.update(overrides)
    return lidar_visibility_receipt_from_scan(**values)


def _config():
    return CoverageCandidateReconciliationConfig(
        observer_config_sha256=CONFIG_SHA256,
    )


def _decision(*, receipts, grid=None, plan=None, candidate=None):
    return reconcile_provisional_candidate_visibility(
        plan=plan or _plan(),
        candidate=candidate or _candidate(),
        occupancy_grid=grid or _grid(),
        receipts=receipts,
        config=_config(),
    )


class CoverageCandidateReconciliationTest(unittest.TestCase):
    def test_three_distinct_finite_clear_scans_reject_provisional_candidate(self):
        receipts = (
            _receipt("receipt_01", 1.00, 1.50),
            _receipt("receipt_02", 1.10, 1.55),
            _receipt("receipt_03", 1.20, 1.60),
        )

        decision = _decision(receipts=receipts)

        self.assertTrue(decision.reject_provisional)
        self.assertEqual(decision.action, ACTION_REJECT_PROVISIONAL)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(decision.distinct_clear_scan_stamps_sec, (1.0, 1.1, 1.2))
        self.assertEqual(
            {item.classification for item in decision.ray_evidence},
            {"clear"},
        )
        evidence = decision.to_evidence_dict()
        self.assertFalse(evidence["expected_stand_count_used"])
        self.assertEqual(len(decision.decision_sha256), 64)

    def test_limited_invalid_rays_do_not_veto_clear_majority(self):
        receipts = (
            _receipt("receipt_01", 1.00, 1.50),
            _receipt("receipt_02", 1.10, math.inf),
            _receipt("receipt_03", 1.20, 1.55),
            _receipt("receipt_04", 1.30, 1.60),
        )

        decision = _decision(receipts=receipts)

        self.assertTrue(decision.reject_provisional)
        self.assertEqual(decision.action, ACTION_REJECT_PROVISIONAL)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(
            [item.classification for item in decision.ray_evidence],
            ["clear", "invalid", "clear", "clear"],
        )
        self.assertEqual(
            decision.distinct_clear_scan_stamps_sec,
            (1.0, 1.2, 1.3),
        )

    def test_real_run_shape_sixty_four_clear_sixteen_invalid_rejects(self):
        receipts = tuple(
            _receipt(
                f"receipt_{index:03d}",
                float(index) * 0.10,
                1.50 if index <= 64 else math.inf,
            )
            for index in range(1, 81)
        )

        decision = _decision(receipts=receipts)

        self.assertTrue(decision.reject_provisional)
        policy = decision.ray_policy_decision
        self.assertEqual(policy.clear_ray_count, 64)
        self.assertEqual(policy.invalid_selected_ray_count, 16)
        self.assertAlmostEqual(policy.clear_ray_fraction, 0.8)
        self.assertAlmostEqual(policy.invalid_selected_ray_fraction, 0.2)
        self.assertTrue(policy.rejection_supported)

    def test_single_matching_return_vetoes_clear_majority(self):
        receipts = tuple(
            _receipt(
                f"receipt_{index:03d}",
                float(index) * 0.10,
                0.80 if index == 80 else 1.50,
            )
            for index in range(1, 81)
        )

        decision = _decision(receipts=receipts)

        self.assertFalse(decision.reject_provisional)
        self.assertIn("matching_return_supports_candidate", decision.reasons)

    def test_fraction_policy_rejects_bool_and_out_of_range_values(self):
        for overrides in (
            {"minimum_clear_ray_fraction": True},
            {"minimum_clear_ray_fraction": 1.01},
            {"maximum_invalid_selected_ray_fraction": -0.01},
        ):
            with self.subTest(overrides=overrides):
                config = CoverageCandidateReconciliationConfig(
                    observer_config_sha256=CONFIG_SHA256,
                    **overrides,
                )
                with self.assertRaisesRegex(ValueError, "finite number"):
                    config.validated()

    def test_invalid_ray_fraction_limit_retains_candidate(self):
        receipts = (
            _receipt("receipt_01", 1.00, 1.50),
            _receipt("receipt_02", 1.10, math.inf),
            _receipt("receipt_03", 1.20, 1.55),
            _receipt("receipt_04", 1.30, math.inf),
            _receipt("receipt_05", 1.40, 1.60),
        )

        decision = _decision(receipts=receipts)

        self.assertFalse(decision.reject_provisional)
        self.assertIn("insufficient_clear_ray_fraction", decision.reasons)
        self.assertIn(
            "selected_scan_ray_invalid_fraction_exceeds_limit",
            decision.reasons,
        )

    def test_repeated_scan_timestamps_are_insufficient(self):
        receipts = (
            _receipt("receipt_01", 1.0, 1.50),
            _receipt("receipt_02", 1.0, 1.55),
            _receipt("receipt_03", 1.0, 1.60),
        )

        decision = _decision(receipts=receipts)

        self.assertFalse(decision.reject_provisional)
        self.assertEqual(decision.action, ACTION_RETAIN)
        self.assertIn(
            "insufficient_distinct_clear_scan_times",
            decision.reasons,
        )
        self.assertEqual(decision.distinct_clear_scan_stamps_sec, (1.0,))

    def test_inf_or_nan_target_rays_retain_candidate(self):
        for invalid in (math.inf, math.nan):
            with self.subTest(invalid=invalid):
                receipts = tuple(
                    _receipt(f"receipt_{index}", float(index), invalid)
                    for index in range(1, 4)
                )

                decision = _decision(receipts=receipts)

                self.assertFalse(decision.reject_provisional)
                self.assertIn("selected_scan_ray_invalid", decision.reasons)
                self.assertTrue(
                    all(
                        item.selected_range_m is None
                        for item in decision.ray_evidence
                    )
                )

    def test_static_supercover_occlusion_retain_candidate(self):
        receipts = tuple(
            _receipt(f"receipt_{index}", float(index), 1.50)
            for index in range(1, 4)
        )

        decision = _decision(
            receipts=receipts,
            grid=_grid(blocked=(GridCell(14, 10),)),
        )

        self.assertFalse(decision.reject_provisional)
        self.assertIn(
            "actual_static_line_of_sight_blocked",
            decision.reasons,
        )
        self.assertTrue(
            all(item.classification == "blocked" for item in decision.ray_evidence)
        )

    def test_nearer_or_matching_return_retain_candidate(self):
        cases = (
            (0.50, "nearer_return_occludes_candidate", "nearer"),
            (0.80, "matching_return_supports_candidate", "matching"),
        )
        for target_range, reason, classification in cases:
            with self.subTest(target_range=target_range):
                receipts = tuple(
                    _receipt(
                        f"receipt_{index}",
                        float(index),
                        target_range,
                    )
                    for index in range(1, 4)
                )

                decision = _decision(receipts=receipts)

                self.assertFalse(decision.reject_provisional)
                self.assertIn(reason, decision.reasons)
                self.assertTrue(
                    all(
                        item.classification == classification
                        for item in decision.ray_evidence
                    )
                )

    def test_missing_or_identity_mismatched_receipts_retain_candidate(self):
        missing = _decision(receipts=())
        self.assertFalse(missing.reject_provisional)
        self.assertIn("visibility_receipts_missing", missing.reasons)
        self.assertIn(
            "planned_visible_viewpoint_receipts_missing",
            missing.reasons,
        )

        mismatched = tuple(
            _receipt(
                f"receipt_{index}",
                float(index),
                1.50,
                observer_config_sha256="c" * 64,
            )
            for index in range(1, 4)
        )
        decision = _decision(receipts=mismatched)
        self.assertFalse(decision.reject_provisional)
        self.assertIn("visibility_receipt_identity_mismatch", decision.reasons)

    def test_no_other_planned_visibility_retain_candidate(self):
        decision = _decision(
            receipts=tuple(
                _receipt(f"receipt_{index}", float(index), 1.50)
                for index in range(1, 4)
            ),
            plan=_plan(check_visible=False),
        )

        self.assertFalse(decision.reject_provisional)
        self.assertIn("no_other_planned_visible_viewpoint", decision.reasons)

    def test_full_envelope_must_fit_conservative_visibility_distance(self):
        base_plan = _plan()
        short_range_plan = replace(
            base_plan,
            config=replace(base_plan.config, visibility_radius_m=0.90),
        )
        receipts = tuple(
            _receipt(f"receipt_{index}", float(index), 1.50)
            for index in range(1, 4)
        )

        decision = _decision(receipts=receipts, plan=short_range_plan)

        self.assertFalse(decision.reject_provisional)
        self.assertIn(
            "candidate_envelope_outside_conservative_range",
            decision.reasons,
        )
        self.assertTrue(
            all(item.classification == "out_of_range" for item in decision.ray_evidence)
        )

    def test_decision_is_independent_of_expected_candidate_count(self):
        receipts = tuple(
            _receipt(f"receipt_{index}", float(index), 1.50)
            for index in range(1, 4)
        )

        expected_five = _decision(receipts=receipts, plan=_plan(expected_count=5))
        expected_ninety_nine = _decision(
            receipts=receipts,
            plan=_plan(expected_count=99),
        )

        self.assertTrue(expected_five.reject_provisional)
        self.assertTrue(expected_ninety_nine.reject_provisional)
        self.assertEqual(expected_five.reasons, expected_ninety_nine.reasons)


if __name__ == "__main__":
    unittest.main()
