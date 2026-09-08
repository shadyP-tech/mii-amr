"""Fresh, bounded standoff proposals are separate from observed camera views."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachUnreachableError, compute_candidate_preapproach_plan,
    load_candidate_planning_context,
)
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, CandidateObservation, execute_candidate_approach_phase,
)
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, CandidateInspectionRouteUnavailableError,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.inspection_policy import candidate_view_options
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    MAX_STANDOFF_PROPOSALS_PER_DIRECTION, CandidateInspectionRouteSearch,
    bounded_inspection_standoffs,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.recovery_failure import CandidateStartupRecoveryError
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_detected_station_exploration import write_free_map


class CandidateInspectionRouteSearchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []

    def offsets(self, requested=0.70):
        return bounded_inspection_standoffs(
            requested, minimum_active_standoff_m=0.33,
            candidate_transit_radius_m=0.34, map_resolution_m=0.05,
        )

    def test_offsets_reach_smaller_radius_without_crossing_raster_keepout_or_work_cap(self):
        offsets = self.offsets()
        self.assertEqual(offsets[:4], (0.7, 0.65, 0.6, 0.55))
        self.assertTrue(all(offset > 0.34 + 0.05 / math.sqrt(2.0) for offset in offsets))
        self.assertLessEqual(len(offsets), MAX_STANDOFF_PROPOSALS_PER_DIRECTION)
        self.assertLess(offsets[-1], 0.40)
        huge = self.offsets(1000000.0)
        self.assertLessEqual(len(huge), MAX_STANDOFF_PROPOSALS_PER_DIRECTION)
        self.assertEqual(huge[-1], offsets[-1])

    def test_blocked_outer_pose_tries_fresh_smaller_standoffs_at_same_heading(self):
        search = CandidateInspectionRouteSearch("candidate", self.events.append)
        calls = []
        def move(offset, path):
            calls.append((offset, path))
            if offset > 0.55:
                raise CandidateInspectionRouteUnavailableError("target blocked", reason_code="static_proposal_unavailable")
            return "reached"
        result = search.move_direction(requested_normal_rad=0.8, standoffs=self.offsets(),
                                       output_root=self.root, move=move)
        self.assertEqual(result, "reached")
        self.assertEqual([offset for offset, _ in calls], [0.7, 0.65, 0.6, 0.55])
        self.assertEqual(len({path for _, path in calls}), 4)
        self.assertEqual([item["outcome"] for item in search.proposals],
                         ["proposal_rejected"] * 3 + ["route_completed"])
        self.assertEqual({item["requested_normal_rad"] for item in search.proposals}, {0.8})

    @staticmethod
    def unavailable(_frame, _path, index):
        raise CandidateObservationUnavailableError(
            candidate_uid="candidate", observation_attempt_index=index,
            reason="candidate-local missing observation", process_evidence={}, status_evidence={},
        )

    def blocked_controller(self, *, proposal_limit=64):
        search = CandidateInspectionRouteSearch("candidate", self.events.append, max_proposals=proposal_limit)
        def blocked(offset, path):
            raise CandidateInspectionRouteUnavailableError("target blocked")
        def move(frame, normal, root, index, source):
            return search.move_direction(requested_normal_rad=normal, standoffs=self.offsets(),
                                         output_root=root, move=blocked)
        with self.assertRaises(CandidateObservationUnavailableError) as caught:
            execute_candidate_inspection(
                candidate_uid="candidate", candidate_root=self.root, initial_frame=0.0, max_views=8,
                effects=CandidateInspectionEffects(
                    capture=self.unavailable, canonical_normal=lambda value: value,
                    move_view=move, move_opposite=lambda *args: self.fail("unexpected opposite move"),
                    progress_evidence=lambda *args: {}, route_search_evidence=search.to_dict,
                ),
            )
        return search, caught.exception, json.loads((self.root / "inspection_progress.json").read_text())

    def test_all_blocked_directions_exhaust_proposals_not_eight_camera_views(self):
        search, error, progress = self.blocked_controller()
        self.assertEqual(len(search.proposals), len(self.offsets()) * 7)
        self.assertEqual(progress["local_view_count"], 1)
        self.assertEqual(progress["max_views"], 8)
        self.assertEqual(progress["termination_reason"], "view_proposals_exhausted")
        self.assertTrue(progress["proposal_search_exhausted"])
        self.assertFalse(progress["view_budget_exhausted"])
        self.assertEqual(len(progress["exhausted_view_normals_rad"]), 7)
        self.assertEqual(progress["achieved_view_normals_rad"], [0.0])
        self.assertEqual(error.process_evidence["inspection_exhaustion_reason"], "view_proposals_exhausted")

    def test_global_proposal_ceiling_stops_inside_heading_without_marking_it_observed_or_exhausted(self):
        search, error, progress = self.blocked_controller(proposal_limit=2)
        self.assertEqual(len(search.proposals), 2)
        self.assertEqual(progress["termination_reason"], "route_proposal_budget_exhausted")
        self.assertEqual(progress["exhausted_view_normals_rad"], [])
        self.assertEqual(progress["achieved_view_normals_rad"], [0.0])
        self.assertEqual(error.process_evidence["local_view_count"], 1)

    def test_view_budget_counts_observations_and_keeps_achieved_heading_diversity(self):
        moves = []
        def move(frame, normal, root, index, source):
            moves.append(normal)
            return normal
        with self.assertRaises(CandidateObservationUnavailableError) as caught:
            execute_candidate_inspection(
                candidate_uid="candidate", candidate_root=self.root, initial_frame=0.0, max_views=8,
                effects=CandidateInspectionEffects(
                    capture=self.unavailable, canonical_normal=lambda value: value, move_view=move,
                    move_opposite=lambda *args: None, progress_evidence=lambda *args: {},
                ),
            )
        progress = caught.exception.status_evidence
        self.assertEqual(progress["termination_reason"], "view_budget_exhausted")
        self.assertEqual(progress["local_view_count"], 8)
        self.assertEqual(len(moves), 7)
        normals = progress["achieved_view_normals_rad"]
        for index, normal in enumerate(normals):
            self.assertTrue(all(abs(math.remainder(normal - old, 2 * math.pi)) >= math.radians(20)
                                for old in normals[:index]))

    def test_attempted_pose_does_not_blacklist_heading_until_all_its_standoffs_exhaust(self):
        options = candidate_view_options(
            0.0, classification="unobservable", achieved_normals=[0.0],
            attempted_normals=[math.pi / 2], exhausted_normals=[],
        )
        self.assertIn(math.pi / 2, options)
        exhausted = candidate_view_options(
            0.0, classification="unobservable", achieved_normals=[0.0],
            attempted_normals=[math.pi / 2], exhausted_normals=[math.pi / 2],
        )
        self.assertNotIn(math.pi / 2, exhausted)

    def adapter_setup(self):
        fixture = fixtures.AutonomousCandidateApproachTest()
        config = fixture._config(self.root, (fixture._candidate("candidate", 0.0, 0.0),))
        write_free_map(self.root, resolution=0.05)
        return fixture, replace(config, max_candidate_inspection_views=2)

    def adapter_effects(self, fixture, config, *, reject_kind):
        proposals, motions, observations = [], [], []
        def plan(request):
            if request.inspection_view_path is not None:
                proposals.append(request)
                if reject_kind == "static" and request.approach_offset_m > 0.55:
                    raise CandidatePreapproachUnreachableError("candidate", "blocked requested target")
            return {"route_csv": str(request.output_dir / "route.csv")}
        def run(request):
            motions.append(request)
            if "_inspection_" in request.run_id:
                if reject_kind == "uncertainty" and sum("_inspection_" in item.run_id for item in motions) == 1:
                    return fixture._route_uncertainty_rejection(request)
                if reject_kind == "published_motion":
                    return fixture._route_uncertainty_rejection(request, motion_published=True)
                if reject_kind == "issued_permit":
                    return fixture._route_uncertainty_rejection(request, report_mission_leg_permit=True)
                if reject_kind == "tf":
                    raise RuntimeError("TF connectivity failure")
            return fixture._completed(request)
        def capture(request):
            observations.append(request)
            if request.attempt_index == 0:
                raise CandidateObservationUnavailableError(
                    candidate_uid="candidate", observation_attempt_index=0,
                    reason="no candidate-local observation", process_evidence={}, status_evidence={},
                )
            return CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
        effects = CandidateApproachEffects(
            select_initial_preapproach=fixture._nearest_selection,
            read_current_pose=lambda: Pose2D(0.7, 0.0, math.pi),
            plan_preapproach=plan, run_motion_leg=run, capture_observation=capture,
            validate_facing=lambda request: {"candidate_uid": "candidate"}, commit_decision=lambda request: None,
        )
        return effects, proposals, motions, observations

    def test_adapter_smaller_feasible_radius_keeps_same_heading_and_full_snapshot(self):
        fixture, config = self.adapter_setup()
        effects, proposals, motions, observations = self.adapter_effects(fixture, config, reject_kind="static")
        result = execute_candidate_approach_phase(config, effects)
        self.assertEqual(result.stand_count, 1)
        self.assertEqual([request.approach_offset_m for request in proposals], [0.7, 0.65, 0.6, 0.55])
        self.assertTrue(all(request.snapshot == config.snapshot for request in proposals))
        self.assertEqual(len({request.output_dir for request in proposals}), 4)
        self.assertEqual(len(motions), 2)
        self.assertEqual(len(observations), 2)

    def test_adapter_strict_no_motion_uncertainty_rejection_gets_new_route_and_child(self):
        fixture, config = self.adapter_setup()
        effects, proposals, motions, observations = self.adapter_effects(fixture, config, reject_kind="uncertainty")
        execute_candidate_approach_phase(config, effects)
        self.assertEqual([request.approach_offset_m for request in proposals], [0.7, 0.65])
        self.assertEqual(len({request.run_id for request in motions}), 3)
        self.assertEqual(len({request.sealed["route_csv"] for request in motions}), 3)
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(request.snapshot == config.snapshot for request in proposals))

    def test_adapter_never_retries_after_published_motion_permit_or_terminal_tf_failure(self):
        for kind in ("published_motion", "issued_permit", "tf"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                previous = self.root
                self.root = Path(directory)
                fixture, config = self.adapter_setup()
                effects, proposals, motions, observations = self.adapter_effects(fixture, config, reject_kind=kind)
                with self.assertRaises((CandidateStartupRecoveryError, RuntimeError)):
                    execute_candidate_approach_phase(config, effects)
                self.assertEqual(len(proposals), 1)
                self.assertEqual(len(motions), 2)
                self.assertEqual(len(observations), 1)
                events = [json.loads(line) for line in next(config.session_root.glob(
                    "candidates/*/inspection_route_proposals.jsonl")).read_text().splitlines()]
                self.assertEqual(events[-1]["outcome"], "terminal_failure")
                self.root = previous

    def test_real_planner_finds_smaller_radius_while_preserving_all_stand_keepouts(self):
        fixture = fixtures.AutonomousCandidateApproachTest()
        map_yaml = write_free_map(self.root, width=80, height=60, resolution=0.05)
        config = fixture._config(self.root, (
            fixture._candidate("candidate", 0.0, -0.1),
            fixture._candidate("other_stand", 0.8, 0.3),
        ))
        grid, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
        config = replace(config, map_yaml=map_yaml,
                         plan=replace(config.plan, map_bundle_sha256=bundle.bundle_sha256),
                         snapshot=replace(config.snapshot, map_bundle_sha256=bundle.bundle_sha256))
        context = load_candidate_planning_context(
            map_yaml, semantic_map_id="arena", plan=config.plan, snapshot=config.snapshot,
            inflation_radius_m=config.inflation_radius_m,
            candidate_transit_radius_m=config.candidate_transit_radius_m,
            physical_clearance=config.physical_clearance,
        )
        search = CandidateInspectionRouteSearch("candidate", self.events.append)
        def plan(offset, path):
            try:
                return compute_candidate_preapproach_plan(
                    map_yaml=map_yaml, semantic_map_id="arena", plan=config.plan,
                    snapshot=config.snapshot, candidate_uid="candidate",
                    start=Pose2D(0.0, 0.6, -math.pi / 2), approach_offset_m=offset,
                    inflation_radius_m=config.inflation_radius_m,
                    candidate_transit_radius_m=config.candidate_transit_radius_m,
                    physical_clearance=config.physical_clearance,
                    inspection_view_normal_rad=-math.pi / 2, planning_context=context,
                )
            except CandidatePreapproachUnreachableError as exc:
                raise CandidateInspectionRouteUnavailableError(str(exc)) from exc
        result = search.move_direction(
            requested_normal_rad=-math.pi / 2,
            standoffs=bounded_inspection_standoffs(
                .70, minimum_active_standoff_m=config.physical_clearance["minimum_active_standoff_m"],
                candidate_transit_radius_m=config.candidate_transit_radius_m,
                map_resolution_m=grid.metadata.resolution,
            ),
            output_root=self.root / "proposals", move=plan,
        )
        self.assertEqual(search.proposals[0]["approach_offset_m"], .7)
        self.assertEqual(search.proposals[0]["outcome"], "proposal_rejected")
        self.assertLess(result.approach_offset_m, .7)
        self.assertGreaterEqual(result.endpoint_standoff_m, config.physical_clearance["minimum_active_standoff_m"])
        self.assertEqual(len(config.snapshot.candidates), 2)
        self.assertFalse((self.root / "proposals").exists(), "pure planner must not materialize routes")
