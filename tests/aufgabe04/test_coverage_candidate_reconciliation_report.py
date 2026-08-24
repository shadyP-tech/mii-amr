import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation import (
    ACTION_REJECT_PROVISIONAL,
    ACTION_RETAIN,
    CoverageCandidateReconciliationConfig,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_report import (
    POLICY_MODE_EVIDENCE_ONLY,
    build_coverage_candidate_reconciliation_report,
)
from scripts.aufgabe04.navigation.planning.map_io import MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    STATUS_PROVISIONAL,
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
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


def _grid() -> OccupancyGrid:
    rows = tuple(tuple(0 for _x in range(30)) for _y in range(30))
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
        cells=rows,
    )


def _candidate_cell() -> GridCell:
    return GridCell(18, 10)


def _plan(*, expected_count: int = 5) -> CoverageSurveyPlan:
    candidate_cell = _candidate_cell()
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        arena_bounds=ArenaBounds(),
        config=CoverageSurveyConfig(expected_stand_count=expected_count),
        viewpoints=(
            SurveyViewpoint(
                viewpoint_id=SOURCE_VIEWPOINT,
                pose=Pose2D(0.05, -0.45, math.pi / 2.0),
                cell=GridCell(10, 5),
                visible_cells=(candidate_cell,),
            ),
            SurveyViewpoint(
                viewpoint_id=CHECK_VIEWPOINT,
                pose=SCAN_POSE,
                cell=GridCell(10, 10),
                visible_cells=(candidate_cell,),
            ),
        ),
        surveyable_cells=(candidate_cell,),
        planned_covered_cells=(candidate_cell,),
        planned_coverage_ratio=1.0,
    )


def _candidate(
    candidate_uid: str = "survey_candidate_0001",
    *,
    viewpoint_ids: tuple[str, ...] = (SOURCE_VIEWPOINT,),
) -> SurveyCandidate:
    suffix = candidate_uid.rsplit("_", 1)[-1]
    return SurveyCandidate(
        candidate_uid=candidate_uid,
        x_m=CANDIDATE_POSE.x_m,
        y_m=CANDIDATE_POSE.y_m,
        radius_m=0.06,
        uncertainty_m=0.02,
        keepout_radius_m=0.31,
        confidence=0.91,
        hit_count=16,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
        source_observation_ids=(f"observation_{suffix}",),
        viewpoint_ids=viewpoint_ids,
        status=STATUS_PROVISIONAL,
    )


def _registry(
    candidates: tuple[SurveyCandidate, ...] | None = None,
) -> StandSurveyRegistry:
    return StandSurveyRegistry(
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        candidates=candidates or (_candidate(),),
    )


def _ranges(target_range: float) -> list[float]:
    result = [3.0] * SCAN_COUNT
    result[TARGET_INDEX] = target_range
    return result


def _receipt(
    receipt_id: str,
    stamp: float,
    target_range: float,
    **overrides,
):
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


def _clear_receipts():
    return tuple(
        _receipt(f"receipt_{index}", float(index), 1.50)
        for index in range(1, 4)
    )


def _config() -> CoverageCandidateReconciliationConfig:
    return CoverageCandidateReconciliationConfig(
        observer_config_sha256=CONFIG_SHA256,
    )


def _report(*, plan=None, registry=None, receipts=None):
    return build_coverage_candidate_reconciliation_report(
        plan=plan or _plan(),
        registry=registry or _registry(),
        occupancy_grid=_grid(),
        receipts=_clear_receipts() if receipts is None else receipts,
        config=_config(),
    )


class CoverageCandidateReconciliationReportTest(unittest.TestCase):
    def test_clear_decision_is_only_recommended_and_registry_is_unchanged(self):
        registry = _registry()
        snapshot_before = registry

        report = _report(registry=registry)

        self.assertEqual(registry, snapshot_before)
        self.assertEqual(
            report.recommended_negative_visibility_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertEqual(report.decisions[0].action, ACTION_REJECT_PROVISIONAL)
        self.assertFalse(report.registry_mutation_applied)
        self.assertFalse(report.motion_authorized)
        self.assertEqual(report.policy_mode, POLICY_MODE_EVIDENCE_ONLY)
        evidence = report.to_evidence_dict()
        self.assertFalse(evidence["expected_stand_count_used"])
        self.assertFalse(evidence["registry_mutation_applied"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(len(report.report_sha256), 64)
        json.dumps(evidence, allow_nan=False)

    def test_missing_or_invalid_receipts_retain_provisional_candidate(self):
        missing = _report(receipts=())
        self.assertEqual(
            missing.retained_provisional_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertEqual(missing.decisions[0].action, ACTION_RETAIN)
        self.assertIn(
            "visibility_receipts_missing",
            missing.decisions[0].reasons,
        )

        invalid_for_config = tuple(
            _receipt(
                f"receipt_{index}",
                float(index),
                math.inf,
                observer_config_sha256="c" * 64,
            )
            for index in range(1, 4)
        )
        invalid = _report(receipts=invalid_for_config)
        self.assertEqual(
            invalid.recommended_negative_visibility_candidate_uids,
            (),
        )
        self.assertEqual(
            invalid.retained_provisional_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertIn(
            "visibility_receipt_identity_mismatch",
            invalid.decisions[0].reasons,
        )
        self.assertIn(
            "selected_scan_ray_invalid",
            invalid.decisions[0].reasons,
        )

    def test_recommendations_are_independent_of_expected_count(self):
        expected_five = _report(plan=_plan(expected_count=5))
        expected_ninety_nine = _report(plan=_plan(expected_count=99))

        self.assertEqual(
            expected_five.recommended_negative_visibility_candidate_uids,
            expected_ninety_nine.recommended_negative_visibility_candidate_uids,
        )
        self.assertEqual(
            tuple(decision.action for decision in expected_five.decisions),
            tuple(
                decision.action for decision in expected_ninety_nine.decisions
            ),
        )
        self.assertEqual(
            expected_five.registry_snapshot_sha256,
            expected_ninety_nine.registry_snapshot_sha256,
        )

    def test_candidates_and_receipts_have_stable_canonical_order(self):
        registry = _registry(
            (
                _candidate("survey_candidate_0001"),
                _candidate("survey_candidate_0002"),
                _candidate(
                    "survey_candidate_0003",
                    viewpoint_ids=(SOURCE_VIEWPOINT, CHECK_VIEWPOINT),
                ),
            )
        )
        receipts = _clear_receipts()

        forward = _report(registry=registry, receipts=receipts)
        reversed_receipts = _report(
            registry=registry,
            receipts=tuple(reversed(receipts)),
        )

        self.assertEqual(
            tuple(decision.candidate_uid for decision in forward.decisions),
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertEqual(
            forward.recommended_negative_visibility_candidate_uids,
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertEqual(
            forward.retained_provisional_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertEqual(
            forward.unevaluated_provisional_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertEqual(
            forward.receipt_set_sha256,
            reversed_receipts.receipt_set_sha256,
        )
        self.assertEqual(forward.report_sha256, reversed_receipts.report_sha256)


if __name__ == "__main__":
    unittest.main()
