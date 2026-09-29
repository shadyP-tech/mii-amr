"""One frozen occupancy artifact must constrain A*, smoothing and child budgets."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.approach.admitted_pose_route import plan_admitted_pose_route, validate_admitted_pose_route_binding
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import evaluate_route_uncertainty_admission
from scripts.aufgabe04.navigation.execution.route_uncertainty_evidence import RouteUncertaintyAdmissionRejected
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.route_smoothing import segment_is_collision_free
from scripts.aufgabe04.navigation.planning.temporary_obstacle_overlay import TemporaryObstacleMap, apply_bound_temporary_obstacles
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.navigation.station_segment import localization_admission
from tests.aufgabe04 import test_admitted_pose_route as fixtures
from tests.aufgabe04 import test_odom_execution_anchor_admission as odom_fixtures
from tests.aufgabe04.test_admitted_return_uncertainty import _context
from tests.aufgabe04.test_stored_pose_tour_authorization import catalog_evidence
from tests.aufgabe04.test_temporary_obstacle_overlay import write_capture


class TemporaryObstacleRouteTest(unittest.TestCase):
    def fixture(self, root, *, obstacle=(.01, .01), dynamic=False):
        args = fixtures.AdmittedPoseRouteTest()._fixture(root, fine_grid=True)
        args["purpose"] = "stored_pose_tour"
        args["target_evidence"].update({
            **catalog_evidence(root, uid=args["candidate_uid"], pose=asdict(args["target"])),
            "tour_id": "tour", "visit_index": 0,
        })
        if dynamic:
            args["target_evidence"]["tour_navigation"] = {
                "execution_index": 0, "stage_index": 0, "replan_count": 0,
                "previous_terminal_json": "", "previous_terminal_sha256": "",
            }
        args["route_uncertainty_context"] = _context(args["start"], heading_sigma_rad=.001)
        # The scan sensor is a metre behind the observed occupied cell; its
        # transform is unrelated to the later stopped route planning frame.
        source = write_capture(root, base=(obstacle[0]-1., obstacle[1], 0.))
        occupancy = TemporaryObstacleMap("tour", "odom", args["snapshot"].map_bundle_sha256)
        occupancy.update_from_capture(source, now_sec=100.23)
        frame = CandidatePlanningFrame.from_evidence(args["target_evidence"]["planning_frame_admission"])
        args["temporary_obstacle_overlay_path"] = occupancy.write_projection(root / "overlay.json", frame, now_sec=100.24)
        return args, source

    def test_frozen_overlay_forces_detour_preserving_target_pool_and_stage_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp), dynamic=True)
            original_snapshot = args["snapshot_path"].read_bytes()
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
            self.assertTrue(result["is_final_stage"])
            self.assertEqual(leg.raw_waypoints[-1].pose, args["target"])
            self.assertGreater(len(leg.raw_waypoints), 2)
            self.assertEqual(args["snapshot_path"].read_bytes(), original_snapshot)
            metadata = json.loads(Path(result["diagnostics_json"]).read_text())["metadata"]
            full = json.loads(Path(metadata["return_to_start_stage"]["full_return_route_json"]).read_text())
            self.assertEqual(full["temporary_obstacle_overlay_sha256"], metadata["temporary_obstacle_overlay_sha256"])
            self.assertEqual(full["tour_navigation"], args["target_evidence"]["tour_navigation"])
            grid, _ = load_occupancy_grid_with_bundle(args["map_yaml"], semantic_map_id="arena", planning_frame="map")
            static = Costmap.from_occupancy_grid(grid).with_arena_bounds(args["plan"].arena_bounds)
            base = apply_bound_temporary_obstacles(static, metadata)
            context = args["route_uncertainty_context"]
            direct = evaluate_route_uncertainty_admission(base, (args["start"], args["target"]),
                context.covariance, context.admission_config)
            self.assertFalse(direct.decision.accepted)
            self.assertTrue(evaluate_route_uncertainty_admission(static, (args["start"], args["target"]),
                context.covariance, context.admission_config).decision.accepted)
            for first, second in zip(leg.raw_waypoints, leg.raw_waypoints[1:]):
                self.assertTrue(segment_is_collision_free(base.with_inflation(args["inflation_radius_m"]), first.pose, second.pose))
            self.assertTrue(validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), leg,
                candidate_snapshot_path=Path(result["candidate_snapshot"])).ok)

    def test_route_child_rejects_mutated_occupancy_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            args, source = self.fixture(Path(temp))
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0)
            source.write_text("changed")
            status = validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), leg,
                candidate_snapshot_path=Path(result["candidate_snapshot"]))
            self.assertFalse(status.ok)
            self.assertIn("source capture hash", status.failures[0])

    def test_occupied_exact_goal_is_rejected_without_snapping(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp), obstacle=(.431, .213))
            with self.assertRaisesRegex(ValueError, "exact stored target is blocked"):
                plan_admitted_pose_route(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_dynamic_tour_cannot_drop_overlay_or_change_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp), dynamic=True)
            changed = dict(args)
            changed.pop("temporary_obstacle_overlay_path")
            with self.assertRaisesRegex(ValueError, "requires a temporary"):
                plan_admitted_pose_route(**changed)
            changed = deepcopy(args)
            changed["target_evidence"]["planning_frame_admission"]["map_from_odom"]["x_m"] = .1
            with self.assertRaises(ValueError):
                plan_admitted_pose_route(**changed)

    def test_real_child_admission_uses_frozen_raw_occupancy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Existing real child fixture maps odom(5,0) to map(1,1).
            source = write_capture(root, base=(4.3, 0., 0.))
            occupancy = TemporaryObstacleMap("tour", "odom", "d"*64)
            occupancy.update_from_capture(source, now_sec=100.23)
            frame = CandidatePlanningFrame(Pose2D(1., 1., 0.), PlanarTransform2D(-4., 1., 0.))
            path = occupancy.write_projection(root / "overlay.json", frame, now_sec=100.24)
            binding = {"route_purpose": "stored_pose_tour", "tour_id": "tour", "planning_frame": "map",
                "map_bundle_sha256": "d"*64, "planning_frame_admission": frame.to_evidence(),
                "temporary_obstacle_overlay_json": str(path), "temporary_obstacle_overlay_sha256": file_sha256(path)}
            original = localization_admission.apply_bound_temporary_obstacles
            applied = []
            def bind_costmap(base, metadata, **kwargs):
                # Replace only fixture diagnostics with real overlay fields;
                # production reconstruction, rasterization and admission run.
                result = original(base, {**metadata, **binding}, **kwargs)
                applied.append(result)
                return result
            helper = odom_fixtures.OdomExecutionAnchorAdmissionTest()
            with patch.object(localization_admission, "apply_bound_temporary_obstacles", side_effect=bind_costmap):
                with self.assertRaises(RouteUncertaintyAdmissionRejected):
                    helper._assert_real_admission(stationary_turn=False, return_stage=True)
            self.assertEqual(len(applied), 1)
            self.assertIn("temporary_obstacle", applied[0].cell_sources.values())

    def test_child_reprojects_frozen_odom_cells_through_its_new_certificate_transform(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_capture(root, base=(8., 0., 0.))
            occupancy = TemporaryObstacleMap("tour", "odom", "d"*64)
            occupancy.update_from_capture(source, now_sec=100.23)
            angle = .16
            # Both frames map stationary odom(5,0) to map(1,1), so startup
            # position admission passes despite the newer yaw correction.
            parent_transform = PlanarTransform2D(1.-5.*math.cos(angle), 1.-5.*math.sin(angle), angle)
            parent_frame = CandidatePlanningFrame(Pose2D(1., 1., angle), parent_transform)
            overlay = occupancy.write_projection(root / "overlay.json", parent_frame, now_sec=100.24)
            overlay_bytes = overlay.read_bytes()
            binding = {"route_purpose": "stored_pose_tour", "tour_id": "tour", "planning_frame": "map",
                "map_bundle_sha256": "d"*64, "planning_frame_admission": parent_frame.to_evidence(),
                "temporary_obstacle_overlay_json": str(overlay), "temporary_obstacle_overlay_sha256": file_sha256(overlay)}
            route = (Pose2D(1.,1.,0.), Pose2D(5.,1.,0.))
            static = Costmap.from_occupancy_grid(odom_fixtures.grid())
            stale = apply_bound_temporary_obstacles(static, binding)
            fresh = apply_bound_temporary_obstacles(static, binding,
                execution_map_from_odom=PlanarTransform2D(-4.,1.,0.))
            context = _context(route[0], heading_sigma_rad=.02)
            self.assertTrue(evaluate_route_uncertainty_admission(stale, route,
                context.covariance, context.admission_config).decision.accepted)
            self.assertFalse(evaluate_route_uncertainty_admission(fresh, route,
                context.covariance, context.admission_config).decision.accepted)
            original_build = localization_admission._build_odom_execution_admission
            def invoke(**kwargs):
                kwargs["diagnostics_snapshot"].metadata.update(binding)
                with patch.object(localization_admission, "poses_from_waypoints", return_value=route):
                    return original_build(**kwargs)
            with patch.object(localization_admission, "_build_odom_execution_admission", side_effect=invoke):
                with self.assertRaises(RouteUncertaintyAdmissionRejected):
                    odom_fixtures.OdomExecutionAnchorAdmissionTest()._assert_real_admission(
                        stationary_turn=False, return_stage=True, heading_lever_arm_m=.1)
            self.assertEqual(overlay.read_bytes(), overlay_bytes)


if __name__ == "__main__":
    unittest.main()
