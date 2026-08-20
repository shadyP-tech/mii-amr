from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.coverage_candidate_admission import (
    evaluate_coverage_candidate_admission,
)
from scripts.aufgabe04.navigation.coverage_candidate_lifecycle import (
    CoverageCandidatePopulation,
    ExactTwoLidarCheckpointDecision,
    classify_coverage_candidates,
    coverage_candidate_population_evidence,
    coverage_candidate_population_evidence_sha256,
    evaluate_exact_two_lidar_checkpoint,
    exact_two_lidar_checkpoint_evidence,
    exact_two_lidar_checkpoint_evidence_sha256,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    SurveyViewpoint,
    mark_viewpoint_visited,
    new_stand_survey_registry,
    new_survey_progress,
)


MAP_HASH = "a" * 64
SURVEYABLE_CELLS = tuple(GridCell(index, 0) for index in range(4))


def survey_plan(
    *,
    expected_stand_count: int | None = 5,
    exact_inspection_point_count: int | None = 2,
) -> CoverageSurveyPlan:
    config = CoverageSurveyConfig(
        lane_count=1,
        coverage_threshold=0.75,
        minimum_candidate_confidence=0.70,
        minimum_candidate_hits=3,
        minimum_distinct_viewpoints=2,
        expected_stand_count=expected_stand_count,
        exact_inspection_point_count=exact_inspection_point_count,
    )
    viewpoints = (
        SurveyViewpoint(
            viewpoint_id="survey_vp_001",
            pose=Pose2D(-1.0, 0.0, 0.0),
            cell=SURVEYABLE_CELLS[0],
            visible_cells=SURVEYABLE_CELLS[:3],
        ),
        SurveyViewpoint(
            viewpoint_id="survey_vp_002",
            pose=Pose2D(1.0, 0.0, 0.0),
            cell=SURVEYABLE_CELLS[-1],
            visible_cells=SURVEYABLE_CELLS[1:],
        ),
    )
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id="candidate_lifecycle_test",
        planning_frame="map",
        map_bundle_sha256=MAP_HASH,
        arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
        config=config,
        viewpoints=viewpoints,
        surveyable_cells=SURVEYABLE_CELLS,
        planned_covered_cells=SURVEYABLE_CELLS,
        planned_coverage_ratio=1.0,
    )


def complete_progress(plan: CoverageSurveyPlan):
    progress = new_survey_progress(plan)
    for viewpoint_id in plan.viewpoint_ids:
        progress = mark_viewpoint_visited(plan, progress, viewpoint_id)
    return progress


def candidate(
    index: int,
    *,
    status: str = STATUS_PROVISIONAL,
    confidence: float = 0.80,
    hit_count: int = 7,
    viewpoint_ids: tuple[str, ...] = ("survey_vp_001",),
) -> SurveyCandidate:
    return SurveyCandidate(
        candidate_uid=f"survey_candidate_{index:04d}",
        x_m=0.25 * index,
        y_m=0.10 * index,
        radius_m=0.06,
        uncertainty_m=0.02,
        keepout_radius_m=0.31,
        confidence=confidence,
        hit_count=hit_count,
        first_seen_sec=10.0 + index,
        last_seen_sec=11.0 + index,
        source_observation_ids=(f"candidate_{index}_observation",),
        viewpoint_ids=viewpoint_ids,
        status=status,
    )


def registry(
    plan: CoverageSurveyPlan,
    *candidates: SurveyCandidate,
) -> StandSurveyRegistry:
    return replace(
        new_stand_survey_registry(plan),
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_uid)),
    )


def latest_run_shape(plan: CoverageSurveyPlan) -> StandSurveyRegistry:
    """Two multi-view pending candidates plus three strong single-view ones."""

    return registry(
        plan,
        candidate(
            1,
            status=STATUS_PENDING_CAMERA,
            viewpoint_ids=("survey_vp_001", "survey_vp_002"),
        ),
        candidate(
            2,
            status=STATUS_PENDING_CAMERA,
            viewpoint_ids=("survey_vp_001", "survey_vp_002"),
        ),
        candidate(3),
        candidate(4, viewpoint_ids=("survey_vp_002",)),
        candidate(5),
    )


class CoverageCandidateLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = survey_plan()
        self.progress = complete_progress(self.plan)
        self.registry = latest_run_shape(self.plan)

    def test_latest_run_shape_completes_lidar_checkpoint_not_camera_queue(self):
        population = classify_coverage_candidates(self.plan, self.registry)
        decision = evaluate_exact_two_lidar_checkpoint(
            self.plan,
            self.progress,
            self.registry,
        )

        self.assertIsInstance(population, CoverageCandidatePopulation)
        self.assertIsInstance(decision, ExactTwoLidarCheckpointDecision)
        self.assertEqual(
            population.active_lidar_candidate_uids,
            tuple(f"survey_candidate_{index:04d}" for index in range(1, 6)),
        )
        self.assertEqual(
            population.multi_view_supported_candidate_uids,
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertEqual(
            population.camera_queue_candidate_uids,
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertTrue(decision.ready)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(
            decision.admitted_lidar_candidate_uids,
            population.active_lidar_candidate_uids,
        )
        self.assertFalse(decision.camera_approach_authorized)

        # The existing camera-ready gate remains stricter: provisional
        # candidates are not silently promoted into its approach snapshot.
        legacy = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            self.registry,
        )
        self.assertFalse(legacy.ready)

    def test_lifecycle_classes_preserve_status_boundaries(self):
        classified = classify_coverage_candidates(
            self.plan,
            registry(
                self.plan,
                candidate(1, status=STATUS_PROVISIONAL),
                candidate(
                    2,
                    status=STATUS_PENDING_CAMERA,
                    viewpoint_ids=("survey_vp_001", "survey_vp_002"),
                ),
                candidate(
                    3,
                    status=STATUS_CONFIRMED,
                    viewpoint_ids=("survey_vp_001", "survey_vp_002"),
                ),
                candidate(4, status=STATUS_REJECTED),
            ),
        )

        self.assertEqual(
            classified.lidar_static_map_admitted_candidate_uids,
            tuple(f"survey_candidate_{index:04d}" for index in range(1, 5)),
        )
        self.assertEqual(
            classified.active_lidar_candidate_uids,
            (
                "survey_candidate_0001",
                "survey_candidate_0002",
                "survey_candidate_0003",
            ),
        )
        self.assertEqual(
            classified.camera_queue_candidate_uids,
            ("survey_candidate_0002",),
        )
        self.assertEqual(
            classified.camera_confirmed_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertEqual(
            classified.rejected_candidate_uids,
            ("survey_candidate_0004",),
        )
        self.assertFalse(classified.candidates[0].multi_view_supported)
        self.assertFalse(classified.candidates[-1].active_lidar)

    def test_weak_active_candidate_fails_checkpoint_with_stable_uid(self):
        weak = replace(
            self.registry.candidates[-1],
            confidence=0.69,
            hit_count=2,
        )
        weak_registry = registry(
            self.plan,
            *self.registry.candidates[:-1],
            weak,
        )

        decision = evaluate_exact_two_lidar_checkpoint(
            self.plan,
            self.progress,
            weak_registry,
        )

        self.assertFalse(decision.ready)
        self.assertTrue(decision.active_lidar_candidate_count_met)
        self.assertFalse(decision.active_lidar_candidate_support_met)
        self.assertEqual(
            decision.unsupported_active_lidar_candidate_uids,
            ("survey_candidate_0005",),
        )
        self.assertIn(
            "active_lidar_candidate_support_not_met",
            decision.reasons,
        )
        weak_evidence = decision.population.candidates[-1]
        self.assertEqual(
            weak_evidence.support_reasons,
            ("confidence_below_minimum", "hit_count_below_minimum"),
        )
        self.assertEqual(decision.admitted_lidar_candidate_uids, ())

    def test_unknown_viewpoint_evidence_fails_basic_lidar_support(self):
        unknown = replace(
            self.registry.candidates[-1],
            viewpoint_ids=("survey_vp_unknown",),
        )
        unknown_registry = registry(
            self.plan,
            *self.registry.candidates[:-1],
            unknown,
        )

        decision = evaluate_exact_two_lidar_checkpoint(
            self.plan,
            self.progress,
            unknown_registry,
        )

        self.assertFalse(decision.ready)
        evidence = decision.population.candidates[-1]
        self.assertEqual(evidence.known_viewpoint_ids, ())
        self.assertEqual(
            evidence.unknown_viewpoint_ids,
            ("survey_vp_unknown",),
        )
        self.assertEqual(
            evidence.support_reasons,
            ("unknown_viewpoint_ids", "known_viewpoint_evidence_missing"),
        )

    def test_incomplete_coverage_and_count_mismatch_fail_independently(self):
        plan = replace(
            self.plan,
            config=replace(self.plan.config, coverage_threshold=0.80),
        )
        partial = mark_viewpoint_visited(
            plan,
            new_survey_progress(plan),
            "survey_vp_001",
        )
        short_registry = registry(
            plan,
            *latest_run_shape(plan).candidates[:-1],
        )

        decision = evaluate_exact_two_lidar_checkpoint(
            plan,
            partial,
            short_registry,
        )

        self.assertFalse(decision.ready)
        self.assertEqual(
            decision.reasons,
            (
                "planned_viewpoints_incomplete",
                "visited_coverage_below_threshold",
                "active_lidar_candidate_count_mismatch",
            ),
        )
        self.assertEqual(decision.unvisited_viewpoint_ids, ("survey_vp_002",))

    def test_non_exact_two_scope_and_unset_expected_count_are_not_ready(self):
        non_exact = survey_plan(
            expected_stand_count=None,
            exact_inspection_point_count=None,
        )
        decision = evaluate_exact_two_lidar_checkpoint(
            non_exact,
            complete_progress(non_exact),
            registry(non_exact),
        )

        self.assertFalse(decision.ready)
        self.assertEqual(
            decision.reasons,
            (
                "exact_two_inspection_scope_required",
                "expected_stand_count_unset",
            ),
        )

    def test_payloads_are_json_safe_frozen_and_deterministically_hashed(self):
        first_population = classify_coverage_candidates(
            self.plan,
            registry(self.plan, *reversed(self.registry.candidates)),
        )
        second_population = classify_coverage_candidates(
            self.plan,
            registry(self.plan, *self.registry.candidates),
        )
        first_decision = evaluate_exact_two_lidar_checkpoint(
            self.plan,
            self.progress,
            self.registry,
        )
        second_decision = evaluate_exact_two_lidar_checkpoint(
            self.plan,
            self.progress,
            self.registry,
        )

        population_payload = coverage_candidate_population_evidence(
            first_population
        )
        decision_payload = exact_two_lidar_checkpoint_evidence(first_decision)
        json.dumps(population_payload, sort_keys=True, allow_nan=False)
        json.dumps(decision_payload, sort_keys=True, allow_nan=False)
        self.assertEqual(first_population, second_population)
        self.assertEqual(
            coverage_candidate_population_evidence_sha256(first_population),
            coverage_candidate_population_evidence_sha256(population_payload),
        )
        self.assertEqual(
            exact_two_lidar_checkpoint_evidence_sha256(first_decision),
            exact_two_lidar_checkpoint_evidence_sha256(second_decision),
        )
        self.assertEqual(
            exact_two_lidar_checkpoint_evidence_sha256(first_decision),
            exact_two_lidar_checkpoint_evidence_sha256(decision_payload),
        )
        self.assertEqual(
            len(exact_two_lidar_checkpoint_evidence_sha256(first_decision)),
            64,
        )
        with self.assertRaises(FrozenInstanceError):
            first_decision.ready = False
        with self.assertRaises(FrozenInstanceError):
            first_population.candidates[0].active_lidar = False

        changed = evaluate_exact_two_lidar_checkpoint(
            self.plan,
            self.progress,
            registry(
                self.plan,
                *self.registry.candidates[:-1],
                replace(self.registry.candidates[-1], confidence=0.81),
            ),
        )
        self.assertNotEqual(
            first_decision.registry_snapshot_sha256,
            changed.registry_snapshot_sha256,
        )
        self.assertNotEqual(
            exact_two_lidar_checkpoint_evidence_sha256(first_decision),
            exact_two_lidar_checkpoint_evidence_sha256(changed),
        )

    def test_malformed_provenance_and_registry_order_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "another survey"):
            evaluate_exact_two_lidar_checkpoint(
                self.plan,
                replace(self.progress, survey_id="other"),
                self.registry,
            )
        reversed_registry = replace(
            self.registry,
            candidates=tuple(reversed(self.registry.candidates)),
        )
        with self.assertRaisesRegex(ValueError, "sorted with unique IDs"):
            classify_coverage_candidates(self.plan, reversed_registry)


if __name__ == "__main__":
    unittest.main()
