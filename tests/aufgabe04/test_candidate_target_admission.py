from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.candidate_perception_advisory import (
    CandidatePerceptionAdvisory, MORPHOLOGY_CONFLICT, VISIBILITY_GAP,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance, CandidatePoint2D,
)
from scripts.aufgabe04.navigation.approach.candidate_target_admission import (
    NoEligibleCameraTargetError, TARGET_STATIC_MAP_INCOMPATIBLE,
    UNRESOLVED_MORPHOLOGY_CONFLICT, evaluate_candidate_target_admission,
    load_candidate_target_costmap,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    compute_candidate_preapproach_plan, load_candidate_planning_context,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_materialization import (
    materialize_candidate_preapproach_plan,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import CandidatePreapproachUnreachableError
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import plan_and_select_camera_candidate
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import CameraCandidateSelectionConfig, NoFeasibleCameraCandidateError
from scripts.aufgabe04.navigation.planning.map_io import CELL_OCCUPIED
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, write_candidate_snapshot
from tests.aufgabe04 import test_candidate_preapproach_planning as planning_fixtures
from tests.aufgabe04 import test_lidar_inspection_planning as alignment_fixtures
from tests.aufgabe04.test_candidate_preapproach_planning import PHYSICAL_CLEARANCE
from tests.aufgabe04.test_stand_candidate_static_map_admission import costmap_from_rows


def _advisory(candidate, kind=MORPHOLOGY_CONFLICT, map_hash="c" * 64):
    frame = CandidateFrameProvenance("map", "odom", CandidatePoint2D(.5, .45), source_evidence_id="d" * 64)
    track = replace(frame, canonical_odom_point=CandidatePoint2D(.55, .45))
    extra = {} if kind == VISIBILITY_GAP else {
        "track_id": "conflicting_track", "track_frame": track,
        "track_source_observation_ids": ("rejected_observation",),
        "rejection_reasons": ("median_width_above_maximum",),
        "association_distance_m": .05, "association_limit_m": .12,
        "possible_candidate_uids": (candidate.candidate_uid,),
    }
    return CandidatePerceptionAdvisory(
        kind=kind, candidate_uid=candidate.candidate_uid, survey_id="survey",
        map_bundle_sha256=map_hash, plan_sha256="e" * 64, viewpoint_id="view2",
        source_morphology_sha256="f" * 64, candidate_frame=frame,
        candidate_source_viewpoint_ids=("view1",),
        source_observation_ids=candidate.source.observation_ids,
        proposal_max_range_m=3.5, visibility_radius_m=1.35,
        eligible_other_viewpoint_ids=(), **extra,
    )


def _with_advisory(candidate, kind=MORPHOLOGY_CONFLICT, map_hash="c" * 64):
    return replace(candidate, source=replace(candidate.source,
        perception_advisories=(_advisory(candidate, kind, map_hash),)))


class CandidateTargetAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.helper = planning_fixtures.CandidatePreapproachPlanningTest()
        rows = [[0] * 20 for _ in range(20)]
        rows[4][4] = CELL_OCCUPIED
        self.costmap = costmap_from_rows(rows)
        self.candidate = self.helper._candidate("candidate1", .8, .45)

    def test_clear_single_view_and_visibility_gap_remain_eligible(self):
        for candidate in (self.candidate, _with_advisory(self.candidate, VISIBILITY_GAP)):
            with self.subTest(candidate=candidate):
                decision = evaluate_candidate_target_admission(candidate, self.costmap)
                self.assertTrue(decision.accepted)
                self.assertEqual(decision.reasons, ())

    def test_nominal_fit_near_wall_preserves_boundary_camera_eligibility(self):
        candidate = replace(self.candidate, geometry=replace(self.candidate.geometry, x_m=.57))
        decision = evaluate_candidate_target_admission(candidate, self.costmap)
        self.assertTrue(decision.accepted)
        self.assertTrue(decision.static_map_evidence.boundary_provisional)
        self.assertFalse(decision.static_map_evidence.admitted)

    def test_moved_blocked_and_outside_targets_are_rejected(self):
        for x in (.45, -.1):
            with self.subTest(x=x):
                decision = evaluate_candidate_target_admission(self.candidate, self.costmap,
                    target_geometry=replace(self.candidate.geometry, x_m=x))
                self.assertEqual(decision.reasons, (TARGET_STATIC_MAP_INCOMPATIBLE,))
                self.assertFalse(decision.accepted)
                self.assertNotEqual(decision.candidate_geometry_sha256, decision.target_geometry_sha256)

    def test_conflict_cannot_be_cleared_by_replacing_target_geometry(self):
        candidate = _with_advisory(self.candidate)
        decision = evaluate_candidate_target_admission(candidate, self.costmap,
            target_geometry=replace(candidate.geometry, x_m=1.2))
        self.assertEqual(decision.reasons, (UNRESOLVED_MORPHOLOGY_CONFLICT,))
        self.assertEqual(decision.unresolved_morphology_conflict_sha256,
            (candidate.source.perception_advisories[0].sha256,))
        with self.assertRaises(FrozenInstanceError):
            decision.accepted = True
        evidence = decision.to_evidence()
        evidence["reasons"].clear()
        self.assertFalse(decision.accepted)
        self.assertFalse(evidence["candidate_rejection_authorized"])

    def test_malformed_geometry_advisory_and_route_costmap_fail_closed(self):
        candidate = _with_advisory(self.candidate)
        invalid = replace(candidate, candidate_uid="different_candidate")
        with self.assertRaises(ValueError):
            evaluate_candidate_target_admission(invalid, self.costmap)
        with self.assertRaises(ValueError):
            evaluate_candidate_target_admission(candidate, self.costmap,
                target_geometry=replace(candidate.geometry, x_m=float("nan")))
        with self.assertRaises(ValueError):
            evaluate_candidate_target_admission(candidate, self.costmap.with_inflation(.2))

    def _fixture(self, root):
        _, snapshot, path, prepared = self.helper._materialization_fixture(root)
        plan = self.helper._plan(snapshot.map_bundle_sha256)
        return snapshot, path, prepared, {
            "map_yaml": prepared.dry_run.grid.metadata.yaml_path,
            "semantic_map_id": "arena", "plan": plan, "snapshot": snapshot,
            "approach_offset_m": .7, "inflation_radius_m": .25,
            "candidate_transit_radius_m": .31, "physical_clearance": PHYSICAL_CLEARANCE,
        }

    def test_loader_validates_map_frame_and_applies_arena_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, _, prepared, kwargs = self._fixture(Path(tmp))
            args = {name: kwargs[name] for name in ("semantic_map_id", "plan", "snapshot")}
            static = load_candidate_target_costmap(kwargs["map_yaml"], **args)
            self.assertIn("arena_boundary", static.cell_sources.values())
            for invalid in (replace(snapshot, planning_frame="other_map"),
                            replace(snapshot, map_bundle_sha256="e" * 64)):
                with self.assertRaises(ValueError):
                    load_candidate_target_costmap(kwargs["map_yaml"], **{**args, "snapshot": invalid})

    def test_selector_excludes_conflict_before_previews_and_preserves_keepouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, _, prepared, kwargs = self._fixture(Path(tmp))
            conflict = _with_advisory(self.helper._candidate("candidate2", .5, .45),
                map_hash=snapshot.map_bundle_sha256)
            snapshot = replace(snapshot, candidates=tuple(sorted((*snapshot.candidates, conflict), key=lambda c: c.candidate_uid)))
            kwargs["snapshot"] = snapshot
            context = load_candidate_planning_context(**{
                key: value for key, value in kwargs.items() if key != "approach_offset_m"
            })
            before = context.costmaps.planning_costmap.blocked_cells
            module = "scripts.aufgabe04.navigation.approach.candidate_preapproach_selection"
            with patch(f"{module}.load_candidate_planning_context", return_value=context), patch(
                f"{module}.compute_candidate_preapproach_plan", wraps=compute_candidate_preapproach_plan,
            ) as compute:
                result = plan_and_select_camera_candidate(**kwargs,
                    current_pose=prepared.start, unresolved=set(snapshot.candidate_uids),
                    selection_config=CameraCandidateSelectionConfig(.055, .18))
            self.assertEqual([call.kwargs["candidate_uid"] for call in compute.call_args_list], ["candidate_1"])
            self.assertEqual(result.to_evidence()["candidate_target_admission"]["excluded_candidate_uids"], ["candidate2"])
            self.assertEqual(context.costmaps.planning_costmap.blocked_cells, before)
            self.assertIn(context.costmaps.planning_costmap.world_to_grid(.5, .45), before)
            self.assertEqual(len(snapshot.candidates), 2)

    def test_compute_and_materializer_refuse_conflict_before_routes_or_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, path, prepared, kwargs = self._fixture(root)
            conflict = _with_advisory(snapshot.candidates[0], map_hash=snapshot.map_bundle_sha256)
            snapshot = replace(snapshot, candidates=(conflict,))
            kwargs["snapshot"] = snapshot
            with patch("scripts.aufgabe04.navigation.approach.candidate_preapproach_compute.plan_safety_ranked_quantized_goal") as route:
                with self.assertRaisesRegex(CandidatePreapproachUnreachableError, UNRESOLVED_MORPHOLOGY_CONFLICT):
                    compute_candidate_preapproach_plan(**kwargs, candidate_uid=conflict.candidate_uid, start=prepared.start)
            route.assert_not_called()
            path = root / "conflict_snapshot.json"
            write_candidate_snapshot(path, snapshot)
            prepared = replace(prepared, candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot))
            output = root / "rejected"
            with self.assertRaisesRegex(CandidatePreapproachUnreachableError, UNRESOLVED_MORPHOLOGY_CONFLICT):
                materialize_candidate_preapproach_plan(prepared, snapshot=snapshot,
                    snapshot_path=path, output_dir=output, physical_clearance=PHYSICAL_CLEARANCE)
            self.assertFalse(output.exists())

    def test_route_failure_keeps_target_deferral_evidence_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, _, prepared, kwargs = self._fixture(Path(tmp))
            conflict = _with_advisory(self.helper._candidate("candidate2", .5, .45),
                map_hash=snapshot.map_bundle_sha256)
            snapshot = replace(snapshot, candidates=tuple(sorted((*snapshot.candidates, conflict), key=lambda c: c.candidate_uid)))
            kwargs["snapshot"] = snapshot
            module = "scripts.aufgabe04.navigation.approach.candidate_preapproach_selection"
            with patch(f"{module}.compute_candidate_preapproach_plan",
                side_effect=CandidatePreapproachUnreachableError("candidate_1", "blocked route")):
                with self.assertRaises(NoFeasibleCameraCandidateError) as raised:
                    plan_and_select_camera_candidate(**kwargs, current_pose=prepared.start,
                        unresolved=set(snapshot.candidate_uids),
                        selection_config=CameraCandidateSelectionConfig(.055, .18))
            self.assertEqual(raised.exception.target_admission_evidence["excluded_candidate_uids"], ["candidate2"])
            self.assertEqual([c.candidate_uid for c in raised.exception.rejected_candidates], ["candidate_1"])

    def test_materializer_rechecks_grid_and_arena_instead_of_trusting_precomputed_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, path, prepared, _ = self._fixture(root)
            dry_run = prepared.dry_run
            cell = dry_run.base_costmap.world_to_grid(.5, 0.)
            rows = [list(row) for row in dry_run.grid.cells]
            rows[cell.y][cell.x] = CELL_OCCUPIED
            grid = replace(dry_run.grid, cells=tuple(tuple(row) for row in rows))
            changed = replace(prepared, dry_run=replace(dry_run, grid=grid))
            output = root / "rejected"
            with self.assertRaisesRegex(CandidatePreapproachUnreachableError, TARGET_STATIC_MAP_INCOMPATIBLE):
                materialize_candidate_preapproach_plan(changed, snapshot=snapshot,
                    snapshot_path=path, output_dir=output, physical_clearance=PHYSICAL_CLEARANCE)
            self.assertFalse(output.exists())
            changed = replace(prepared, dry_run=replace(dry_run,
                arena_bounds=replace(dry_run.arena_bounds, length_m=1.)))
            with self.assertRaisesRegex(ValueError, "arena differs"):
                materialize_candidate_preapproach_plan(changed, snapshot=snapshot,
                    snapshot_path=path, output_dir=output, physical_clearance=PHYSICAL_CLEARANCE)
            self.assertFalse(output.exists())

    def test_validated_fitted_center_can_admit_raw_boundary_target(self):
        for case in ("calibrated", "uncalibrated", "conflicted"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                kwargs, _ = alignment_fixtures.LidarInspectionPlanningTest().fixture(Path(tmp))
                candidate = kwargs["snapshot"].candidates[0]
                candidate = replace(candidate, geometry=replace(candidate.geometry, y_m=.92))
                if case == "conflicted":
                    candidate = _with_advisory(candidate, map_hash=kwargs["snapshot"].map_bundle_sha256)
                snapshot = replace(kwargs["snapshot"], candidates=(candidate,))
                hint = kwargs["lidar_inspection_hints"][candidate.candidate_uid]
                hint = replace(hint, snapshot_sha256=candidate_snapshot_sha256(snapshot),
                    center_x_m=.5, center_y_m=.86)
                kwargs.update(snapshot=snapshot, lidar_inspection_hints={candidate.candidate_uid: hint})
                if case == "calibrated":
                    selection = plan_and_select_camera_candidate(**kwargs)
                    decision = selection.to_evidence()["candidate_target_admission"]["candidate_decisions"][candidate.candidate_uid]
                    self.assertTrue(decision["accepted"])
                    self.assertAlmostEqual(decision["static_map_evidence"]["pose"]["y_m"], .86)
                    self.assertIsNotNone(selection.selected_plan.camera_alignment)
                else:
                    if case == "uncalibrated":
                        kwargs["camera_calibration"] = None
                    module = "scripts.aufgabe04.navigation.approach.candidate_preapproach_selection"
                    with patch(f"{module}.compute_candidate_preapproach_plan") as compute:
                        with self.assertRaises(NoEligibleCameraTargetError) as raised:
                            plan_and_select_camera_candidate(**kwargs)
                    compute.assert_not_called()
                    if case == "conflicted":
                        self.assertIn(UNRESOLVED_MORPHOLOGY_CONFLICT,
                            raised.exception.target_admission_evidence["candidate_decisions"][candidate.candidate_uid]["reasons"])

    def test_blocked_fitted_center_is_gated_before_valid_raw_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = alignment_fixtures.LidarInspectionPlanningTest().fixture(Path(tmp))
            candidate = kwargs["snapshot"].candidates[0]
            candidate = replace(candidate, geometry=replace(candidate.geometry, y_m=.86))
            snapshot = replace(kwargs["snapshot"], candidates=(candidate,))
            hint = kwargs["lidar_inspection_hints"][candidate.candidate_uid]
            hint = replace(hint, snapshot_sha256=candidate_snapshot_sha256(snapshot),
                center_x_m=.5, center_y_m=.92)
            kwargs.update(snapshot=snapshot, lidar_inspection_hints={candidate.candidate_uid: hint})
            selection = plan_and_select_camera_candidate(**kwargs)
            self.assertIsNone(selection.selected_plan.camera_alignment)
            views = selection.to_evidence()["lidar_inspection_hints"]["candidate_views"][candidate.candidate_uid]["views"]
            self.assertTrue(all(TARGET_STATIC_MAP_INCOMPATIBLE in view["reason"] for view in views))
