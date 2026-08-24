import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateSelectionConfig,
    NoFeasibleCameraCandidateError,
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
                    {
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_axis_rad": 0.0,
                        "stand_center": {"x_m": 0.50, "y_m": 0.0},
                        # The original observation was from -y.  A changed
                        # artifact from +y resolves to the opposite normal.
                        "robot_pose": {"x_m": 0.50, "y_m": 0.70},
                    }
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
