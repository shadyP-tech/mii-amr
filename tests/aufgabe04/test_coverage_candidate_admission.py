from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.coverage.coverage_candidate_admission import (
    CoverageCandidateAdmissionDecision,
    coverage_candidate_admission_evidence,
    coverage_candidate_admission_evidence_sha256,
    evaluate_coverage_candidate_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_population_retention import (
    STATIC_MAP_DISPOSITION_ADMITTED,
    STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
)
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
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
    *, expected_stand_count: int | None = 2
) -> CoverageSurveyPlan:
    config = CoverageSurveyConfig(
        lane_count=1,
        coverage_threshold=0.75,
        minimum_candidate_confidence=0.70,
        minimum_candidate_hits=3,
        minimum_distinct_viewpoints=2,
        expected_stand_count=expected_stand_count,
        exact_inspection_point_count=2,
    )
    viewpoints = (
        SurveyViewpoint(
            viewpoint_id="survey_vp_001",
            pose=Pose2D(-1.0, 0.0, 0.0),
            cell=SURVEYABLE_CELLS[0],
            visible_cells=SURVEYABLE_CELLS[:2],
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
        survey_id="candidate_admission_test",
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
    status: str = STATUS_PENDING_CAMERA,
    confidence: float = 0.80,
    hit_count: int = 3,
    viewpoint_ids: tuple[str, ...] = (
        "survey_vp_001",
        "survey_vp_002",
    ),
    static_map_disposition: str = STATIC_MAP_DISPOSITION_ADMITTED,
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
        static_map_disposition=static_map_disposition,
    )


def registry(
    plan: CoverageSurveyPlan,
    *candidates: SurveyCandidate,
) -> StandSurveyRegistry:
    return replace(
        new_stand_survey_registry(plan),
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_uid)),
    )


class CoverageCandidateAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = survey_plan()
        self.progress = complete_progress(self.plan)
        self.candidates = (candidate(1), candidate(2))
        self.registry = registry(self.plan, *self.candidates)

    def test_success_is_frozen_json_safe_and_deterministically_hashed(self):
        first = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            self.registry,
        )
        second = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            self.registry,
        )

        self.assertIsInstance(first, CoverageCandidateAdmissionDecision)
        self.assertTrue(first.ready)
        self.assertEqual(first.reasons, ())
        self.assertEqual(
            first.admitted_candidate_uids,
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertTrue(first.all_planned_viewpoints_visited)
        self.assertTrue(first.coverage_threshold_met)
        self.assertTrue(first.pending_candidate_count_met)
        self.assertEqual(len(first.progress_snapshot_sha256), 64)
        self.assertEqual(len(first.registry_snapshot_sha256), 64)
        self.assertTrue(
            all(evidence.admissible for evidence in first.candidate_evidence)
        )

        payload = coverage_candidate_admission_evidence(first)
        json.dumps(payload, sort_keys=True, allow_nan=False)
        self.assertEqual(
            coverage_candidate_admission_evidence_sha256(first),
            coverage_candidate_admission_evidence_sha256(second),
        )
        self.assertEqual(
            coverage_candidate_admission_evidence_sha256(first),
            coverage_candidate_admission_evidence_sha256(payload),
        )
        self.assertEqual(
            len(coverage_candidate_admission_evidence_sha256(first)),
            64,
        )
        with self.assertRaises(FrozenInstanceError):
            first.ready = False
        with self.assertRaises(FrozenInstanceError):
            first.candidate_evidence[0].confidence = 0.0

        changed_registry = registry(
            self.plan,
            replace(self.candidates[0], confidence=0.81),
            self.candidates[1],
        )
        changed = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            changed_registry,
        )
        self.assertNotEqual(
            coverage_candidate_admission_evidence_sha256(first),
            coverage_candidate_admission_evidence_sha256(changed),
        )

    def test_missing_or_excess_pending_candidates_fail_closed(self):
        cases = (
            registry(self.plan, candidate(1)),
            registry(self.plan, candidate(1), candidate(2), candidate(3)),
        )

        for candidate_registry in cases:
            with self.subTest(count=len(candidate_registry.candidates)):
                decision = evaluate_coverage_candidate_admission(
                    self.plan,
                    self.progress,
                    candidate_registry,
                )
                self.assertFalse(decision.ready)
                self.assertFalse(decision.pending_candidate_count_met)
                self.assertEqual(decision.admitted_candidate_uids, ())
                self.assertIn(
                    "pending_candidate_count_mismatch",
                    decision.reasons,
                )

    def test_low_confidence_or_hits_rejects_pending_candidate(self):
        weak = candidate(2, confidence=0.69, hit_count=2)
        decision = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(self.plan, candidate(1), weak),
        )

        self.assertFalse(decision.ready)
        self.assertTrue(decision.pending_candidate_count_met)
        self.assertIn(
            "pending_candidate_requirements_not_met",
            decision.reasons,
        )
        weak_evidence = decision.candidate_evidence[1]
        self.assertFalse(weak_evidence.admissible)
        self.assertEqual(
            weak_evidence.reasons,
            ("confidence_below_minimum", "hit_count_below_minimum"),
        )

    def test_legacy_full_coverage_gate_requires_strict_static_map_admission(self):
        boundary = candidate(
            2,
            static_map_disposition=(
                STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
            ),
        )
        decision = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(self.plan, candidate(1), boundary),
        )

        self.assertFalse(decision.ready)
        self.assertTrue(decision.pending_candidate_count_met)
        self.assertEqual(
            decision.candidate_evidence[1].static_map_disposition,
            STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
        )
        self.assertEqual(
            decision.candidate_evidence[1].reasons,
            ("strict_static_map_admission_required",),
        )
        self.assertIn(
            "pending_candidate_requirements_not_met",
            decision.reasons,
        )

    def test_unknown_viewpoint_cannot_supply_distinct_or_exact_two_evidence(self):
        unknown = candidate(
            2,
            viewpoint_ids=("survey_vp_001", "survey_vp_unknown"),
        )
        decision = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(self.plan, candidate(1), unknown),
        )

        self.assertFalse(decision.ready)
        evidence = decision.candidate_evidence[1]
        self.assertEqual(evidence.known_viewpoint_ids, ("survey_vp_001",))
        self.assertEqual(
            evidence.unknown_viewpoint_ids,
            ("survey_vp_unknown",),
        )
        self.assertEqual(
            evidence.reasons,
            (
                "unknown_viewpoint_ids",
                "distinct_known_viewpoint_count_below_minimum",
                "exact_two_planned_viewpoints_missing",
            ),
        )

    def test_replayed_viewpoint_ids_are_a_malformed_registry(self):
        replayed = candidate(
            2,
            viewpoint_ids=("survey_vp_001", "survey_vp_001"),
        )

        with self.assertRaisesRegex(ValueError, "viewpoint IDs must be unique"):
            evaluate_coverage_candidate_admission(
                self.plan,
                self.progress,
                registry(self.plan, candidate(1), replayed),
            )

    def test_extra_provisional_candidate_fails_even_when_pending_count_matches(self):
        decision = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(
                self.plan,
                candidate(1),
                candidate(2),
                candidate(
                    3,
                    status=STATUS_PROVISIONAL,
                    hit_count=1,
                    viewpoint_ids=("survey_vp_001",),
                ),
            ),
        )

        self.assertFalse(decision.ready)
        self.assertTrue(decision.pending_candidate_count_met)
        self.assertEqual(
            decision.provisional_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertIn("provisional_candidates_present", decision.reasons)

    def test_other_non_rejected_candidate_fails_but_rejected_is_allowed(self):
        confirmed = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(
                self.plan,
                candidate(1),
                candidate(2),
                candidate(3, status=STATUS_CONFIRMED),
            ),
        )
        rejected = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(
                self.plan,
                candidate(1),
                candidate(2),
                candidate(3, status=STATUS_REJECTED),
            ),
        )

        self.assertFalse(confirmed.ready)
        self.assertEqual(
            confirmed.other_non_rejected_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertIn(
            "other_non_rejected_candidates_present",
            confirmed.reasons,
        )
        self.assertTrue(rejected.ready)
        self.assertEqual(
            rejected.rejected_candidate_uids,
            ("survey_candidate_0003",),
        )

        changed_rejected = evaluate_coverage_candidate_admission(
            self.plan,
            self.progress,
            registry(
                self.plan,
                candidate(1),
                candidate(2),
                replace(candidate(3, status=STATUS_REJECTED), x_m=9.0),
            ),
        )
        self.assertTrue(changed_rejected.ready)
        self.assertNotEqual(
            rejected.registry_snapshot_sha256,
            changed_rejected.registry_snapshot_sha256,
        )
        self.assertNotEqual(
            coverage_candidate_admission_evidence_sha256(rejected),
            coverage_candidate_admission_evidence_sha256(changed_rejected),
        )

    def test_incomplete_coverage_and_unset_expected_count_return_decisions(self):
        incomplete_progress = mark_viewpoint_visited(
            self.plan,
            new_survey_progress(self.plan),
            self.plan.viewpoint_ids[0],
        )
        incomplete = evaluate_coverage_candidate_admission(
            self.plan,
            incomplete_progress,
            self.registry,
        )

        self.assertFalse(incomplete.ready)
        self.assertEqual(
            incomplete.reasons[:2],
            (
                "planned_viewpoints_incomplete",
                "visited_coverage_below_threshold",
            ),
        )
        self.assertEqual(
            incomplete.unvisited_viewpoint_ids,
            ("survey_vp_002",),
        )

        no_expected_plan = survey_plan(expected_stand_count=None)
        no_expected = evaluate_coverage_candidate_admission(
            no_expected_plan,
            complete_progress(no_expected_plan),
            registry(no_expected_plan, candidate(1), candidate(2)),
        )
        self.assertFalse(no_expected.ready)
        self.assertEqual(no_expected.reasons, ("expected_stand_count_unset",))

    def test_inconsistent_or_malformed_snapshots_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "another survey"):
            evaluate_coverage_candidate_admission(
                self.plan,
                replace(self.progress, survey_id="other"),
                self.registry,
            )
        with self.assertRaisesRegex(ValueError, "must be a CoverageSurveyPlan"):
            evaluate_coverage_candidate_admission(
                object(),
                self.progress,
                self.registry,
            )


if __name__ == "__main__":
    unittest.main()
