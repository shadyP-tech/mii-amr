import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    write_backside_axis_frame_projection,
)
from scripts.aufgabe04.navigation.approach.camera_axis_binding import (
    load_opposite_face_normal,
)

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateSelectionConfig,
    NoFeasibleCameraCandidateError,
)
from scripts.aufgabe04.navigation.approach.candidate_goal_cell_selection import (
    GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M,
    GOAL_CELL_RANKING_RATIONALE,
    GOAL_CELL_SELECTION_POLICY,
    GoalCellRouteOptionEvidence,
    select_deterministic_goal_cell_option,
    validate_goal_cell_selection_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachUnreachableError,
    compute_candidate_preapproach_plan,
    materialize_candidate_preapproach_plan,
    route_turn_metrics,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import (
    plan_and_select_camera_candidate,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyViewpoint,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import (
    GridCell,
    Pose2D,
    Route,
    RoutePoint,
)
from scripts.aufgabe04.navigation.planning.map_io import (
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.planning.certified_exact_start_route import (
    certify_and_smooth_exact_start_route,
)
from scripts.aufgabe04.navigation.planning.global_planner import plan_route
from scripts.aufgabe04.navigation.planning.route_costmaps import (
    build_station_route_costmaps,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from tests.aufgabe04.test_detected_station_exploration import write_free_map
from tests.aufgabe04.backside_axis_fixture import (
    backside_axis_payload,
    write_candidate_frame_projection_fixture,
)


PHYSICAL_CLEARANCE = {
    "minimum_active_standoff_m": 0.32,
    "minimum_candidate_transit_radius_m": 0.31,
    "minimum_static_inflation_m": 0.25,
}


class CandidatePreapproachPlanningTest(unittest.TestCase):
    @staticmethod
    def _candidate(uid: str, x_m: float, y_m: float) -> FrozenCandidate:
        return FrozenCandidate(
            candidate_uid=uid,
            geometry=CandidateGeometry(
                x_m=x_m,
                y_m=y_m,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.31,
            ),
            source=CandidateSource(
                source_kind="lidar/stand_coverage_survey",
                source_artifact_sha256="a" * 64,
                detector_config_sha256="b" * 64,
                observation_ids=(f"observation_{uid}",),
            ),
            confidence=0.9,
            hit_count=4,
            first_seen_sec=1.0,
            last_seen_sec=2.0,
        )

    @staticmethod
    def _plan(map_bundle_sha256: str) -> CoverageSurveyPlan:
        cell = GridCell(0, 0)
        return CoverageSurveyPlan(
            schema_version=1,
            survey_id="survey",
            planning_frame="map",
            map_bundle_sha256=map_bundle_sha256,
            arena_bounds=ArenaBounds(),
            config=CoverageSurveyConfig(snap_radius_m=0.30),
            viewpoints=(
                SurveyViewpoint(
                    viewpoint_id="survey_vp_001",
                    pose=Pose2D(0.0, 0.0, 0.0),
                    cell=cell,
                    visible_cells=(cell,),
                ),
            ),
            surveyable_cells=(cell,),
            planned_covered_cells=(cell,),
            planned_coverage_ratio=1.0,
        )

    def _materialization_fixture(
        self,
        root: Path,
    ) -> tuple[FrozenCandidate, object, Path, object]:
        """Build one valid no-write plan for materialization guard tests."""

        map_yaml = write_free_map(root)
        _, bundle = load_occupancy_grid_with_bundle(
            map_yaml,
            semantic_map_id="arena",
            planning_frame="map",
        )
        candidate = self._candidate("candidate_1", 0.50, 0.0)
        snapshot = new_candidate_snapshot(
            snapshot_id="snapshot",
            created_unix_sec=3.0,
            planning_frame="map",
            map_bundle_sha256=bundle.bundle_sha256,
            candidates=(candidate,),
        )
        snapshot_path = root / "candidate_snapshot.json"
        write_candidate_snapshot(snapshot_path, snapshot)
        prepared = compute_candidate_preapproach_plan(
            map_yaml=map_yaml,
            semantic_map_id="arena",
            plan=self._plan(bundle.bundle_sha256),
            snapshot=snapshot,
            candidate_uid=candidate.candidate_uid,
            start=Pose2D(-0.40, 0.0, 0.0),
            approach_offset_m=0.70,
            inflation_radius_m=0.25,
            candidate_transit_radius_m=0.31,
            physical_clearance=PHYSICAL_CLEARANCE,
        )
        return candidate, snapshot, snapshot_path, prepared

    def test_compute_writes_nothing_and_materializes_the_same_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.50, 0.0)
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(candidate,),
            )
            snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            output_dir = root / "selected_route"

            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=self._plan(bundle.bundle_sha256),
                snapshot=snapshot,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(-0.40, 0.0, 0.0),
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=PHYSICAL_CLEARANCE,
            )

            self.assertIsNone(prepared.goal_cell_selection)
            self.assertFalse(output_dir.exists())
            materialize_candidate_preapproach_plan(
                prepared,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                output_dir=output_dir,
                physical_clearance=PHYSICAL_CLEARANCE,
                selection_evidence={
                    "selected_candidate_uid": candidate.candidate_uid,
                    "motion_authorized": False,
                },
            )

            metadata = json.loads(
                (output_dir / "route_diagnostics.json").read_text()
            )["metadata"]
            self.assertEqual(
                metadata["selected_candidate_stand_id"], candidate.candidate_uid
            )
            self.assertEqual(
                metadata["order"], "route-aware-camera-selection"
            )
            self.assertEqual(
                metadata["candidate_route_metrics"]["route_length_m"],
                prepared.route_length_m,
            )
            self.assertTrue(
                (
                    output_dir
                    / "preapproach_execution"
                    / "route_certificate.json"
                ).is_file()
            )

    def test_materialization_seals_shifted_derived_axis_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(
                root, width=60, height=60, resolution=0.05
            )
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.45, 0.15)
            snapshot = new_candidate_snapshot(
                snapshot_id="shifted_snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(candidate,),
            )
            snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            source_axis_path = root / "source_axis.json"
            source_axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_x_m=0.45,
                        stand_y_m=-0.05,
                        robot_x_m=0.45,
                        robot_y_m=0.65,
                    )
                ),
                encoding="utf-8",
            )
            source_projection_path = root / "source_projection.json"
            source_projection_sha256, _, _ = (
                write_candidate_frame_projection_fixture(
                    source_projection_path,
                    candidate_uid=candidate.candidate_uid,
                    canonical_x_m=0.45,
                    canonical_y_m=-0.05,
                    transform_x_m=0.0,
                    transform_y_m=0.0,
                    transform_yaw_rad=0.0,
                )
            )
            target_projection_path = root / "target_projection.json"
            target_projection_sha256, target_x_m, target_y_m = (
                write_candidate_frame_projection_fixture(
                    target_projection_path,
                    candidate_uid=candidate.candidate_uid,
                    canonical_x_m=0.45,
                    canonical_y_m=-0.05,
                    transform_x_m=0.0,
                    transform_y_m=0.20,
                    transform_yaw_rad=0.0,
                )
            )
            self.assertAlmostEqual(target_x_m, candidate.geometry.x_m)
            self.assertAlmostEqual(target_y_m, candidate.geometry.y_m)
            projected_axis_path = root / "projected_axis.json"
            write_backside_axis_frame_projection(
                projected_axis_path,
                axis_evidence_path=source_axis_path,
                source_candidate_projection_path=source_projection_path,
                source_candidate_projection_sha256=(
                    source_projection_sha256
                ),
                target_candidate_projection_path=target_projection_path,
                target_candidate_projection_sha256=(
                    target_projection_sha256
                ),
                target_candidate_x_m=target_x_m,
                target_candidate_y_m=target_y_m,
            )
            selected_normal_rad = -math.pi / 2.0
            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=self._plan(bundle.bundle_sha256),
                snapshot=snapshot,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(-0.40, 0.15, 0.0),
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=PHYSICAL_CLEARANCE,
                approach_normal_rad=selected_normal_rad,
            )
            output_dir = root / "selected_opposite_route"

            outputs = materialize_candidate_preapproach_plan(
                prepared,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                output_dir=output_dir,
                physical_clearance=PHYSICAL_CLEARANCE,
                axis_observation_path=projected_axis_path,
                approach_normal_rad=selected_normal_rad,
            )

            self.assertTrue(Path(outputs["route_certificate_json"]).is_file())
            metadata = json.loads(
                Path(outputs["diagnostics_json"]).read_text(encoding="utf-8")
            )["metadata"]
            self.assertEqual(
                metadata["axis_evidence_kind"],
                "backside_axis_frame_projection",
            )
            self.assertEqual(
                metadata["source_axis_observation_json"],
                str(source_axis_path.absolute()),
            )
            self.assertEqual(
                len(metadata["axis_frame_projection_sha256"]), 64
            )

    def test_axis_goal_already_satisfied_keeps_one_point_no_motion_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            target = Pose2D(0.05, 0.05, math.pi)
            candidate = self._candidate("candidate_1", -0.65, 0.05)
            snapshot = new_candidate_snapshot(
                snapshot_id="already_at_axis_target",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(candidate,),
            )

            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=self._plan(bundle.bundle_sha256),
                snapshot=snapshot,
                candidate_uid=candidate.candidate_uid,
                start=target,
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=PHYSICAL_CLEARANCE,
                approach_normal_rad=0.0,
            )

            self.assertIsNotNone(prepared.goal_cell_selection)
            evidence = prepared.goal_cell_selection
            assert evidence is not None
            assert prepared.result.route is not None
            self.assertEqual(len(prepared.result.route.points), 1)
            self.assertAlmostEqual(prepared.result.route.length_m, 0.0)
            self.assertEqual(
                evidence.selected_goal_cell,
                prepared.dry_run.planning_costmap.world_to_grid(target),
            )
            self.assertAlmostEqual(evidence.selected_route_length_m, 0.0)
            self.assertGreater(
                evidence.selected_route_raw_clearance_lower_bound_m,
                0.25,
            )
            self.assertEqual(
                evidence.to_metadata()["selected_route_clearance_score_kind"],
                "static_point_clearance_m",
            )
            self.assertTrue(
                evidence.to_metadata()[
                    "final_route_uncertainty_preflight_authoritative"
                ]
            )
            validate_goal_cell_selection_binding(
                evidence,
                base_costmap=prepared.dry_run.base_costmap,
                planning_costmap=prepared.dry_run.planning_costmap,
                result=prepared.result,
                expected_requested_goal=evidence.requested_goal,
                stand=Pose2D(
                    candidate.geometry.x_m,
                    candidate.geometry.y_m,
                ),
                minimum_standoff_m=prepared.minimum_active_standoff_m,
            )

    def test_clearance_tie_band_is_symmetric_for_micrometre_jitter(self):
        def option(cell_x: int, clearance_m: float):
            cell = GridCell(cell_x, 2)
            return GoalCellRouteOptionEvidence(
                cell=cell,
                goal=Pose2D(float(cell_x), 2.0),
                continuous_target_error_m=0.02,
                endpoint_standoff_m=0.45,
                accepted=True,
                rejection_reason=None,
                route_raw_clearance_lower_bound_m=clearance_m,
                route_length_m=1.0,
            )

        jitter_m = GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M / 1000.0
        lower_cell = GridCell(1, 2)
        left_lower = select_deterministic_goal_cell_option(
            (option(1, 0.50), option(2, 0.50 + jitter_m))
        )
        right_lower = select_deterministic_goal_cell_option(
            (option(1, 0.50 + jitter_m), option(2, 0.50))
        )

        self.assertEqual(left_lower.cell, lower_cell)
        self.assertEqual(right_lower.cell, lower_cell)
        self.assertEqual(
            select_deterministic_goal_cell_option(
                (option(1, 0.50), option(2, 0.55))
            ).cell,
            GridCell(2, 2),
        )
        no_motion = replace(
            option(1, 0.30),
            route_length_m=0.0,
        )
        self.assertEqual(
            select_deterministic_goal_cell_option(
                (no_motion, option(2, 0.80))
            ).cell,
            no_motion.cell,
        )

    def test_projected_axis_target_crossing_cell_boundary_prefers_wall_clearance(
        self,
    ):
        """Reproduce the 2026-09-04 opposite-face runtime reseal geometry."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = (
                Path(__file__).resolve().parents[2]
                / "maps"
                / "aufgabe03"
                / "arena_1p898x3p9_auto.yaml"
            )
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena_1p898x3p9_auto",
                planning_frame="map",
            )
            stand_x_m = -1.0735299548049737
            stand_y_m = -0.45335329118568435
            candidate = self._candidate(
                "survey_candidate_0001",
                stand_x_m,
                stand_y_m,
            )
            candidate = replace(
                candidate,
                geometry=replace(candidate.geometry, keepout_radius_m=0.34),
            )
            snapshot = new_candidate_snapshot(
                snapshot_id="runtime_projected_snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(candidate,),
            )
            snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            desired_normal_rad = 3.121007182714598
            axis_path = root / "axis_observation.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_id=candidate.candidate_uid,
                        stand_x_m=stand_x_m,
                        stand_y_m=stand_y_m,
                        robot_x_m=stand_x_m + 0.70,
                        robot_y_m=stand_y_m,
                        stand_axis_rad=(
                            desired_normal_rad - math.pi / 2.0
                        ),
                    )
                ),
                encoding="utf-8",
            )
            selected_normal_rad = load_opposite_face_normal(axis_path)
            runtime_clearance = {
                "minimum_active_standoff_m": 0.33,
                "minimum_candidate_transit_radius_m": 0.34,
                "minimum_static_inflation_m": 0.25,
            }

            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id="arena_1p898x3p9_auto",
                plan=self._plan(bundle.bundle_sha256),
                snapshot=snapshot,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(
                    -1.0275464637212612,
                    -0.040347242622465695,
                    2.8109302178920768,
                ),
                approach_offset_m=0.45,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.34,
                physical_clearance=runtime_clearance,
                approach_normal_rad=selected_normal_rad,
            )

            evidence = prepared.goal_cell_selection
            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence.requested_goal_cell, GridCell(25, 24))
            self.assertEqual(evidence.selected_goal_cell, GridCell(26, 24))
            self.assertTrue(
                all(
                    option.continuous_target_error_m
                    <= evidence.quantization_envelope_m + 1.0e-9
                    for option in evidence.options
                )
            )
            options = {option.cell: option for option in evidence.options}
            wallward = options[GridCell(25, 24)]
            inward = options[GridCell(26, 24)]
            self.assertTrue(wallward.accepted)
            self.assertTrue(inward.accepted)
            self.assertGreater(
                inward.route_raw_clearance_lower_bound_m,
                wallward.route_raw_clearance_lower_bound_m + 0.049,
            )
            self.assertGreaterEqual(prepared.endpoint_standoff_m, 0.33)
            self.assertAlmostEqual(
                prepared.selected_approach_pose.x_m,
                -1.495,
            )

            # Replay the failed run's final dry-preflight envelope.  The
            # wallward containing cell exhausts the unchanged budget, while
            # the selected rasterization-equivalent cell is admitted.
            wallward_result = plan_route(
                prepared.dry_run.planning_costmap,
                prepared.start,
                wallward.goal,
                snap_radius_m=0.30,
            )
            wallward_result, _, _ = certify_and_smooth_exact_start_route(
                wallward_result,
                base_costmap=prepared.dry_run.base_costmap,
                planning_costmap=prepared.dry_run.planning_costmap,
                exact_start=prepared.start,
                required_clearance_m=0.25,
            )
            assert wallward_result.route is not None
            covariance = PlanarCovariance(
                xx_m2=0.004079206381264974,
                xy_m2=0.0,
                yy_m2=0.004079206381264974,
            )
            admission_config = RouteUncertaintyAdmissionConfig(
                robot_radius_m=0.105,
                collision_margin_m=0.02,
                fixed_odom_tracking_bound_m=0.03,
                empirical_odom_drift_bound_m=0.02,
                braking_latency_distance_m=0.015,
                localization_sigma_multiplier=2.0,
                heading_sigma_rad=0.08475754253341364,
                heading_lever_arm_m=0.105,
                sampling_spacing_m=0.005,
                heading_reference_x_m=prepared.start.x_m,
                heading_reference_y_m=prepared.start.y_m,
            )
            wallward_admission = evaluate_route_uncertainty_admission(
                prepared.dry_run.base_costmap,
                tuple(point.pose for point in wallward_result.route.points),
                covariance,
                admission_config,
            )
            selected_admission = evaluate_route_uncertainty_admission(
                prepared.dry_run.base_costmap,
                tuple(point.pose for point in prepared.result.route.points),
                covariance,
                admission_config,
            )
            self.assertFalse(wallward_admission.decision.accepted)
            self.assertAlmostEqual(
                wallward_admission.decision.remaining_margin_m,
                -0.02647932523744201,
            )
            self.assertTrue(selected_admission.decision.accepted)
            self.assertAlmostEqual(
                selected_admission.decision.remaining_margin_m,
                0.029936499450787635,
            )

            tampered = replace(
                evidence,
                selected_route_raw_clearance_lower_bound_m=(
                    evidence.selected_route_raw_clearance_lower_bound_m + 0.01
                ),
            )
            tampered_output = root / "tampered_route"
            with self.assertRaisesRegex(
                ValueError,
                "route-clearance evidence mismatch",
            ):
                materialize_candidate_preapproach_plan(
                    replace(prepared, goal_cell_selection=tampered),
                    snapshot=snapshot,
                    snapshot_path=snapshot_path,
                    output_dir=tampered_output,
                    physical_clearance=runtime_clearance,
                    axis_observation_path=axis_path,
                    approach_normal_rad=selected_normal_rad,
                )
            self.assertFalse(tampered_output.exists())

            inflated_wallward = replace(
                wallward,
                route_raw_clearance_lower_bound_m=999.0,
            )
            tampered_options = tuple(
                inflated_wallward if option is wallward else option
                for option in evidence.options
            )
            unselected_tampered = replace(
                evidence,
                options=tampered_options,
            )
            unselected_tampered_output = root / "unselected_tampered_route"
            with self.assertRaisesRegex(
                ValueError,
                "not deterministic winner",
            ):
                materialize_candidate_preapproach_plan(
                    replace(
                        prepared,
                        goal_cell_selection=unselected_tampered,
                    ),
                    snapshot=snapshot,
                    snapshot_path=snapshot_path,
                    output_dir=unselected_tampered_output,
                    physical_clearance=runtime_clearance,
                    axis_observation_path=axis_path,
                    approach_normal_rad=selected_normal_rad,
                )
            self.assertFalse(unselected_tampered_output.exists())

            outputs = materialize_candidate_preapproach_plan(
                prepared,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                output_dir=root / "selected_opposite_route",
                physical_clearance=runtime_clearance,
                axis_observation_path=axis_path,
                approach_normal_rad=selected_normal_rad,
            )
            self.assertTrue(Path(outputs["route_certificate_json"]).is_file())
            metadata = json.loads(
                Path(outputs["diagnostics_json"]).read_text(encoding="utf-8")
            )["metadata"]
            selection = metadata["goal_cell_selection"]
            self.assertEqual(
                selection["requested_goal_cell"],
                {"x": 25, "y": 24},
            )
            self.assertEqual(
                selection["selected_goal_cell"],
                {"x": 26, "y": 24},
            )
            self.assertTrue(
                selection["final_route_uncertainty_preflight_authoritative"]
            )
            self.assertEqual(selection["policy"], GOAL_CELL_SELECTION_POLICY)
            self.assertEqual(
                selection["ranking_rationale"],
                list(GOAL_CELL_RANKING_RATIONALE),
            )
            self.assertAlmostEqual(
                selection["selected_route_raw_clearance_lower_bound_m"],
                0.47252847057799335,
            )
            self.assertEqual(
                selection["clearance_ranking_tolerance_m"],
                GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M,
            )
            self.assertFalse(selection["unselected_option_evidence_persisted"])
            self.assertNotIn("options", selection)

    def test_map_binding_mismatch_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            candidate = self._candidate("candidate_1", 0.50, 0.0)
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256="c" * 64,
                candidates=(candidate,),
            )

            with self.assertRaisesRegex(ValueError, "map differs"):
                compute_candidate_preapproach_plan(
                    map_yaml=map_yaml,
                    semantic_map_id="arena",
                    plan=self._plan("d" * 64),
                    snapshot=snapshot,
                    candidate_uid=candidate.candidate_uid,
                    start=Pose2D(0.0, 0.0, 0.0),
                    approach_offset_m=0.70,
                    inflation_radius_m=0.25,
                    candidate_transit_radius_m=0.31,
                    physical_clearance=PHYSICAL_CLEARANCE,
                )

    def test_approach_inside_rasterized_transit_keepout_is_config_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.50, 0.0)
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(candidate,),
            )

            with self.assertRaisesRegex(
                ValueError,
                "outside its transit keepout after map rasterization",
            ):
                compute_candidate_preapproach_plan(
                    map_yaml=map_yaml,
                    semantic_map_id="arena",
                    plan=self._plan(bundle.bundle_sha256),
                    snapshot=snapshot,
                    candidate_uid=candidate.candidate_uid,
                    start=Pose2D(-0.40, 0.0, 0.0),
                    approach_offset_m=0.33,
                    inflation_radius_m=0.25,
                    candidate_transit_radius_m=0.31,
                    physical_clearance=PHYSICAL_CLEARANCE,
                )

    def test_materialization_rejects_motion_authority_in_selection_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, snapshot, snapshot_path, prepared = (
                self._materialization_fixture(root)
            )
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(ValueError, "motion_authorized"):
                materialize_candidate_preapproach_plan(
                    prepared,
                    snapshot=snapshot,
                    snapshot_path=snapshot_path,
                    output_dir=output_dir,
                    physical_clearance=PHYSICAL_CLEARANCE,
                    selection_evidence={
                        "selected_candidate_uid": candidate.candidate_uid,
                        "motion_authorized": True,
                    },
                )

            self.assertFalse(output_dir.exists())

    def test_materialization_rechecks_rasterized_self_keepout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, snapshot, snapshot_path, prepared = self._materialization_fixture(
                root
            )
            unsafe = replace(prepared, approach_offset_m=0.33)
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(
                ValueError,
                "outside its transit keepout after map rasterization",
            ):
                materialize_candidate_preapproach_plan(
                    unsafe,
                    snapshot=snapshot,
                    snapshot_path=snapshot_path,
                    output_dir=output_dir,
                    physical_clearance=PHYSICAL_CLEARANCE,
                )

            self.assertFalse(output_dir.exists())

    def test_materialization_rejects_selection_candidate_uid_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, snapshot, snapshot_path, prepared = self._materialization_fixture(
                root
            )
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(ValueError, "selected_candidate_uid"):
                materialize_candidate_preapproach_plan(
                    prepared,
                    snapshot=snapshot,
                    snapshot_path=snapshot_path,
                    output_dir=output_dir,
                    physical_clearance=PHYSICAL_CLEARANCE,
                    selection_evidence={
                        "selected_candidate_uid": "different_candidate",
                        "motion_authorized": False,
                    },
                )

            self.assertFalse(output_dir.exists())

    def test_materialization_rejects_mismatched_source_snapshot_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, snapshot, _, prepared = self._materialization_fixture(
                root
            )
            wrong_snapshot = new_candidate_snapshot(
                snapshot_id="wrong_snapshot",
                created_unix_sec=4.0,
                planning_frame=snapshot.planning_frame,
                map_bundle_sha256=snapshot.map_bundle_sha256,
                candidates=(
                    self._candidate(
                        candidate.candidate_uid,
                        candidate.geometry.x_m + 0.05,
                        candidate.geometry.y_m,
                    ),
                ),
            )
            wrong_path = root / "wrong_candidate_snapshot.json"
            write_candidate_snapshot(wrong_path, wrong_snapshot)
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(ValueError, "source artifact.*binding"):
                materialize_candidate_preapproach_plan(
                    prepared,
                    snapshot=snapshot,
                    snapshot_path=wrong_path,
                    output_dir=output_dir,
                    physical_clearance=PHYSICAL_CLEARANCE,
                    selection_evidence={
                        "selected_candidate_uid": candidate.candidate_uid,
                        "motion_authorized": False,
                    },
                )

            self.assertFalse(output_dir.exists())

    def test_materialization_rejects_changed_axis_face_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.50, 0.0)
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(candidate,),
            )
            snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            selected_normal_rad = math.pi / 2.0
            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=self._plan(bundle.bundle_sha256),
                snapshot=snapshot,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(-0.40, 0.0, 0.0),
                approach_offset_m=0.40,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=PHYSICAL_CLEARANCE,
                approach_normal_rad=selected_normal_rad,
            )
            axis_path = root / "axis_observation.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_id=candidate.candidate_uid,
                        stand_x_m=0.50,
                        # The original observation was from -y. A changed
                        # artifact from +y resolves to the opposite normal.
                        robot_x_m=0.50,
                        robot_y_m=0.70,
                    )
                )
            )
            output_dir = root / "must_not_exist"

            with self.assertRaisesRegex(
                ValueError,
                "no longer resolves to the prepared face normal",
            ):
                materialize_candidate_preapproach_plan(
                    prepared,
                    snapshot=snapshot,
                    snapshot_path=snapshot_path,
                    output_dir=output_dir,
                    physical_clearance=PHYSICAL_CLEARANCE,
                    axis_observation_path=axis_path,
                    approach_normal_rad=selected_normal_rad,
                )

            self.assertFalse(output_dir.exists())

    def test_materialization_rejects_stale_same_uid_candidate_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            original = self._candidate("candidate_1", 0.50, 0.0)
            original_snapshot = new_candidate_snapshot(
                snapshot_id="snapshot_original",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(original,),
            )
            prepared = compute_candidate_preapproach_plan(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=self._plan(bundle.bundle_sha256),
                snapshot=original_snapshot,
                candidate_uid=original.candidate_uid,
                start=Pose2D(-0.40, 0.0, 0.0),
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=PHYSICAL_CLEARANCE,
            )
            moved_snapshot = new_candidate_snapshot(
                snapshot_id="snapshot_moved",
                created_unix_sec=4.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(self._candidate("candidate_1", 0.55, 0.0),),
            )
            moved_path = root / "moved_snapshot.json"
            write_candidate_snapshot(moved_path, moved_snapshot)
            output_dir = root / "stale_route"

            with self.assertRaisesRegex(ValueError, "stale candidate snapshot"):
                materialize_candidate_preapproach_plan(
                    prepared,
                    snapshot=moved_snapshot,
                    snapshot_path=moved_path,
                    output_dir=output_dir,
                    physical_clearance=PHYSICAL_CLEARANCE,
                )

            self.assertFalse(output_dir.exists())

    def test_all_candidate_specific_route_failures_are_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            outside = self._candidate("candidate_outside", 20.0, 20.0)
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=(outside,),
            )

            with self.assertRaises(NoFeasibleCameraCandidateError) as raised:
                plan_and_select_camera_candidate(
                    map_yaml=map_yaml,
                    semantic_map_id="arena",
                    plan=self._plan(bundle.bundle_sha256),
                    snapshot=snapshot,
                    current_pose=Pose2D(0.0, 0.0, 0.0),
                    unresolved={outside.candidate_uid},
                    approach_offset_m=0.70,
                    inflation_radius_m=0.25,
                    candidate_transit_radius_m=0.31,
                    physical_clearance=PHYSICAL_CLEARANCE,
                    selection_config=CameraCandidateSelectionConfig(
                        linear_speed_mps=0.055,
                        angular_speed_radps=0.18,
                    ),
                )

            evidence = raised.exception.to_evidence()
            self.assertFalse(evidence["motion_authorized"])
            self.assertEqual(
                evidence["rejected_candidates"][0]["candidate_uid"],
                outside.candidate_uid,
            )

    def test_selection_builds_shared_costmaps_once_for_all_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidates = (
                self._candidate("candidate_a", 0.50, 0.45),
                self._candidate("candidate_b", 1.10, -0.40),
            )
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=bundle.bundle_sha256,
                candidates=candidates,
            )

            with patch(
                "scripts.aufgabe04.navigation.approach."
                "candidate_preapproach_compute."
                "build_station_route_costmaps",
                wraps=build_station_route_costmaps,
            ) as build_costmaps:
                selection = plan_and_select_camera_candidate(
                    map_yaml=map_yaml,
                    semantic_map_id="arena",
                    plan=self._plan(bundle.bundle_sha256),
                    snapshot=snapshot,
                    current_pose=Pose2D(0.0, 0.0, 0.0),
                    unresolved=set(snapshot.candidate_uids),
                    approach_offset_m=0.70,
                    inflation_radius_m=0.25,
                    candidate_transit_radius_m=0.31,
                    physical_clearance=PHYSICAL_CLEARANCE,
                    selection_config=CameraCandidateSelectionConfig(
                        linear_speed_mps=0.055,
                        angular_speed_radps=0.18,
                    ),
                )

            self.assertIn(selection.selected_candidate_uid, snapshot.candidate_uids)
            build_costmaps.assert_called_once()

    def test_candidate_preview_failure_is_reconsidered_at_the_next_live_pose(self):
        candidates = (
            self._candidate("candidate_a", 0.50, 0.45),
            self._candidate("candidate_b", 1.10, -0.40),
        )
        snapshot = new_candidate_snapshot(
            snapshot_id="snapshot",
            created_unix_sec=3.0,
            planning_frame="map",
            map_bundle_sha256="c" * 64,
            candidates=candidates,
        )
        preview_calls: list[tuple[str, float]] = []

        def preview(**kwargs):
            candidate_uid = kwargs["candidate_uid"]
            start_x_m = kwargs["start"].x_m
            preview_calls.append((candidate_uid, start_x_m))
            if candidate_uid == "candidate_a" and start_x_m == 0.0:
                raise CandidatePreapproachUnreachableError(
                    candidate_uid,
                    "blocked only from the first live pose",
                )
            return SimpleNamespace(
                candidate_uid=candidate_uid,
                route_length_m=(
                    0.25 if candidate_uid == "candidate_a" else 1.50
                ),
                turn_burden_rad=0.0,
                initial_turn_rad=0.0,
                inside_requested_standoff=False,
            )

        call_kwargs = {
            "map_yaml": Path("unused.yaml"),
            "semantic_map_id": "arena",
            "plan": self._plan(snapshot.map_bundle_sha256),
            "snapshot": snapshot,
            "unresolved": set(snapshot.candidate_uids),
            "approach_offset_m": 0.70,
            "inflation_radius_m": 0.25,
            "candidate_transit_radius_m": 0.31,
            "physical_clearance": PHYSICAL_CLEARANCE,
            "selection_config": CameraCandidateSelectionConfig(
                linear_speed_mps=0.055,
                angular_speed_radps=0.18,
            ),
        }
        module = (
            "scripts.aufgabe04.navigation.approach."
            "candidate_preapproach_selection"
        )
        with (
            patch(f"{module}.load_candidate_planning_context", return_value=object()),
            patch(f"{module}.compute_candidate_preapproach_plan", side_effect=preview),
        ):
            first = plan_and_select_camera_candidate(
                current_pose=Pose2D(0.0, 0.0, 0.0),
                **call_kwargs,
            )
            second = plan_and_select_camera_candidate(
                current_pose=Pose2D(1.0, 0.0, 0.0),
                **call_kwargs,
            )

        self.assertEqual(first.selected_candidate_uid, "candidate_b")
        self.assertEqual(second.selected_candidate_uid, "candidate_a")
        self.assertEqual(
            [call for call in preview_calls if call[0] == "candidate_a"],
            [("candidate_a", 0.0), ("candidate_a", 1.0)],
        )

    def test_turn_metrics_use_the_last_motion_segment_for_terminal_heading(self):
        route = Route(
            points=(
                RoutePoint(0, GridCell(0, 0), Pose2D(0.0, 0.0, 0.0)),
                RoutePoint(1, GridCell(1, 0), Pose2D(1.0, 0.0, 0.0)),
            ),
            requested_start=Pose2D(0.0, 0.0, 0.0),
            requested_goal=Pose2D(1.0, 0.0, 0.0),
            snapped_start=Pose2D(0.0, 0.0, 0.0),
            snapped_goal=Pose2D(1.0, 0.0, 0.0),
            length_m=1.0,
        )

        initial, total = route_turn_metrics(
            route,
            start_yaw_rad=0.0,
            terminal_yaw_rad=math.pi / 2.0,
        )

        self.assertAlmostEqual(initial, 0.0)
        self.assertAlmostEqual(total, math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
