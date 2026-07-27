import math
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    circular_keepout_cells,
    face_normal_candidates,
    greedy_line_of_sight_shortcut,
    minimum_static_obstacle_inflation_m,
    plan_axis_acquisition,
    plan_dynamic_approach,
    plan_fixed_approach,
    segment_is_collision_free,
    supercover_segment_cells,
    with_dynamic_stand_keepout,
)
from scripts.aufgabe04.navigation.map_io import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.waypoint_controller import (
    ControllerConfig,
    compute_join_anchor_command,
    compute_waypoint_command,
)


def costmap_from_blocked(
    *,
    width=25,
    height=25,
    resolution=0.1,
    blocked=(),
    unknown=(),
):
    blocked = set(blocked)
    unknown = set(unknown)
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            cell = GridCell(x, y)
            if cell in blocked:
                row.append(CELL_OCCUPIED)
            elif cell in unknown:
                row.append(CELL_UNKNOWN)
            else:
                row.append(CELL_FREE)
        rows.append(tuple(row))
    grid = OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=resolution,
            origin=(0.0, 0.0, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.2,
            mode="trinary",
        ),
        width=width,
        height=height,
        cells=tuple(rows),
    )
    return Costmap.from_occupancy_grid(grid)


class KeepoutRasterizationTest(unittest.TestCase):
    def test_static_inflation_covers_live_lidar_stop_and_tracking_tube(self):
        self.assertAlmostEqual(
            minimum_static_obstacle_inflation_m(
                robot_radius_m=0.105,
                tracking_margin_m=0.03,
                lidar_stop_distance_m=0.18,
                scan_origin_to_base_offset_m=0.0,
                lidar_clearance_margin_m=0.02,
            ),
            0.23,
        )
        self.assertAlmostEqual(
            minimum_static_obstacle_inflation_m(
                robot_radius_m=0.30,
                tracking_margin_m=0.03,
                lidar_stop_distance_m=0.18,
                scan_origin_to_base_offset_m=-0.04,
                lidar_clearance_margin_m=0.02,
            ),
            0.33,
        )

    def test_radius_is_physical_sum_and_static_costmap_is_not_reinflated(self):
        config = DynamicApproachConfig(
            stand_radius_m=0.06,
            stand_position_uncertainty_m=0.03,
            robot_radius_m=0.10,
            collision_margin_m=0.02,
        )
        self.assertAlmostEqual(config.stand_keepout_radius_m, 0.21)

        static = costmap_from_blocked(blocked={GridCell(0, 0)})
        overlaid, cells = with_dynamic_stand_keepout(static, Pose2D(1.05, 1.05), config)
        self.assertIn(GridCell(0, 0), overlaid.blocked_cells)
        self.assertNotIn(GridCell(1, 0), overlaid.blocked_cells)
        self.assertGreater(len(cells), 0)

    def test_tracking_tube_expands_each_robot_center_envelope_once(self):
        config = DynamicApproachConfig(
            stand_radius_m=0.06,
            stand_position_uncertainty_m=0.03,
            robot_radius_m=0.10,
            collision_margin_m=0.02,
            tracking_margin_m=0.03,
            minimum_non_target_keepout_radius_m=0.40,
        )

        self.assertAlmostEqual(config.stand_keepout_radius_m, 0.24)
        self.assertAlmostEqual(config.minimum_lidar_standoff_m, 0.29)
        self.assertAlmostEqual(config.non_target_stand_keepout_radius_m, 0.43)
        with self.assertRaisesRegex(ValueError, "tracking_margin_m must be non-negative"):
            DynamicApproachConfig(tracking_margin_m=-0.001)

    def test_closed_disk_includes_boundary_touching_cell_but_not_next_cell(self):
        costmap = costmap_from_blocked()
        cells = circular_keepout_cells(costmap, Pose2D(1.05, 1.05), 0.15)
        # Cell 12 begins at x=1.20, exactly radius 0.15 from the center.
        self.assertIn(GridCell(12, 10), cells)
        self.assertNotIn(GridCell(13, 10), cells)


class FixedArrivalPlanningTest(unittest.TestCase):
    def test_plans_exact_stored_face_without_substitution(self):
        costmap = costmap_from_blocked(width=50, height=50)
        stand = Pose2D(2.5, 2.5)
        config = DynamicApproachConfig(standoff_distance_m=0.32)
        face = face_normal_candidates(stand, 0.3, config)[1]

        result = plan_fixed_approach(
            costmap,
            Pose2D(0.25, 0.25),
            stand,
            face,
            config=config,
        )

        self.assertIsNotNone(result.plan)
        assert result.plan is not None
        self.assertEqual(result.plan.selected_face_id, 1)
        self.assertEqual(result.plan.target, face.target)
        self.assertEqual(result.plan.entry, face.entry)

    def test_rejects_mutated_target_instead_of_recomputing_it(self):
        costmap = costmap_from_blocked(width=50, height=50)
        stand = Pose2D(2.5, 2.5)
        config = DynamicApproachConfig(standoff_distance_m=0.32)
        face = face_normal_candidates(stand, 0.3, config)[0]
        malformed = face.__class__(
            face_id=face.face_id,
            normal_rad=face.normal_rad,
            target=Pose2D(face.target.x_m + 0.02, face.target.y_m, face.target.yaw_rad),
            entry=face.entry,
        )

        result = plan_fixed_approach(
            costmap,
            Pose2D(0.25, 0.25),
            stand,
            malformed,
            config=config,
        )

        self.assertIsNone(result.plan)
        self.assertIn(
            "fixed_target_standoff_mismatch",
            result.diagnostics.candidates[0].rejection_reasons,
        )


class SupercoverTest(unittest.TestCase):
    def test_corner_grazing_includes_all_four_cells_and_is_blocked(self):
        costmap = costmap_from_blocked(
            width=3,
            height=3,
            resolution=1.0,
            blocked={GridCell(1, 0)},
        )
        start = Pose2D(0.5, 0.5)
        end = Pose2D(2.5, 2.5)
        cells = set(supercover_segment_cells(costmap, start, end))
        self.assertTrue(
            {GridCell(0, 0), GridCell(1, 0), GridCell(0, 1), GridCell(1, 1)}
            <= cells
        )
        self.assertFalse(segment_is_collision_free(costmap, start, end))

    def test_boundary_following_segment_checks_both_sides(self):
        costmap = costmap_from_blocked(
            width=3,
            height=3,
            resolution=1.0,
            blocked={GridCell(1, 0)},
        )
        self.assertFalse(
            segment_is_collision_free(costmap, Pose2D(1.0, 0.2), Pose2D(1.0, 0.8))
        )


class ShortcutTest(unittest.TestCase):
    def test_unsafe_diagonal_is_not_used(self):
        costmap = costmap_from_blocked(
            width=3,
            height=3,
            resolution=1.0,
            blocked={GridCell(1, 1)},
        )
        poses = (Pose2D(0.5, 0.5), Pose2D(2.5, 0.5), Pose2D(2.5, 2.5))
        shortened = greedy_line_of_sight_shortcut(costmap, poses)
        self.assertEqual(shortened, poses)

    def test_safe_open_path_shortcuts_to_endpoints(self):
        costmap = costmap_from_blocked(width=4, height=4, resolution=1.0)
        poses = (Pose2D(0.5, 0.5), Pose2D(0.5, 1.5), Pose2D(2.5, 2.5))
        self.assertEqual(
            greedy_line_of_sight_shortcut(costmap, poses),
            (poses[0], poses[-1]),
        )

    def test_colliding_input_polyline_is_rejected_instead_of_preserved(self):
        costmap = costmap_from_blocked(
            width=3,
            height=3,
            resolution=1.0,
            blocked={GridCell(1, 1)},
        )
        with self.assertRaisesRegex(ValueError, "colliding segment"):
            greedy_line_of_sight_shortcut(
                costmap,
                (Pose2D(0.5, 0.5), Pose2D(2.5, 2.5)),
            )


class CandidateSelectionTest(unittest.TestCase):
    def setUp(self):
        self.start = Pose2D(0.25, 0.25)
        self.stand = Pose2D(1.05, 1.05)

    def test_axial_pi_flip_preserves_stable_face_ids(self):
        first = face_normal_candidates(self.stand, 0.2)
        flipped = face_normal_candidates(self.stand, 0.2 + math.pi)
        for a, b in zip(first, flipped):
            self.assertEqual(a.face_id, b.face_id)
            self.assertAlmostEqual(a.target.x_m, b.target.x_m)
            self.assertAlmostEqual(a.target.y_m, b.target.y_m)

    def test_blocked_preferred_face_selects_valid_alternate(self):
        # Axis zero: face 0 is north and face 1 is south.
        costmap = costmap_from_blocked(blocked={GridCell(10, 15)})
        result = plan_dynamic_approach(costmap, self.start, self.stand, 0.0)
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.selected_face_id, 1)
        by_face = {item.face_id: item for item in result.diagnostics.candidates}
        self.assertFalse(by_face[0].valid)
        self.assertIn("terminal_corridor_blocked", by_face[0].rejection_reasons)
        self.assertTrue(by_face[1].valid)

    def test_both_faces_blocked_withdraws_target(self):
        costmap = costmap_from_blocked(
            blocked={GridCell(10, 15), GridCell(10, 5)}
        )
        result = plan_dynamic_approach(costmap, self.start, self.stand, 0.0)
        self.assertIsNone(result.plan)
        self.assertEqual(result.diagnostics.failure_reason, "no_valid_face_candidate")
        self.assertTrue(all(not item.valid for item in result.diagnostics.candidates))

    def test_hard_face_evidence_never_falls_back_to_other_valid_face(self):
        costmap = costmap_from_blocked(blocked={GridCell(10, 15)})
        result = plan_dynamic_approach(
            costmap,
            self.start,
            self.stand,
            0.0,
            hard_face_id=0,
        )
        self.assertIsNone(result.plan)
        self.assertEqual(result.diagnostics.failure_reason, "hard_face_0_invalid")
        by_face = {item.face_id: item for item in result.diagnostics.candidates}
        self.assertFalse(by_face[0].valid)
        self.assertTrue(by_face[1].valid)

    def test_ambiguous_equal_cost_uses_face_id_tie_break(self):
        result = plan_dynamic_approach(
            costmap_from_blocked(),
            Pose2D(0.25, 1.05),
            self.stand,
            0.0,
        )
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.selected_face_id, 0)


class AxisAcquisitionPlanningTest(unittest.TestCase):
    def test_open_map_uses_fixed_direct_observation_target_without_corridor(self):
        costmap = costmap_from_blocked(width=40, height=30)
        start = Pose2D(3.15, 0.65, math.pi)
        stand = Pose2D(1.05, 1.05)
        bearing = math.atan2(start.y_m - stand.y_m, start.x_m - stand.x_m)
        target = Pose2D(
            stand.x_m + 0.70 * math.cos(bearing),
            stand.y_m + 0.70 * math.sin(bearing),
            math.atan2(stand.y_m - start.y_m, stand.x_m - start.x_m),
        )

        result = plan_axis_acquisition(costmap, start, stand, target)

        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.target, target)
        self.assertEqual(len(result.plan.waypoints), 2)
        self.assertTrue(all(not waypoint.corridor for waypoint in result.plan.waypoints))
        self.assertTrue(all(not waypoint.protected for waypoint in result.plan.waypoints))

    def test_obstacle_causes_astar_detour_but_never_moves_acquisition_target(self):
        blocked = {GridCell(20, y) for y in range(3, 12)}
        costmap = costmap_from_blocked(width=40, height=30, blocked=blocked)
        start = Pose2D(3.15, 0.65, math.pi)
        stand = Pose2D(1.05, 1.05)
        target = Pose2D(1.75, 1.05, math.pi)

        result = plan_axis_acquisition(costmap, start, stand, target)

        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.target, target)
        self.assertGreater(len(result.plan.waypoints), 2)


class CorridorSafetyTest(unittest.TestCase):
    def setUp(self):
        self.start = Pose2D(0.25, 0.25)
        self.stand = Pose2D(1.05, 1.05)

    def test_unknown_cell_in_terminal_corridor_is_blocked(self):
        costmap = costmap_from_blocked(unknown={GridCell(10, 15)})
        result = plan_dynamic_approach(
            costmap,
            self.start,
            self.stand,
            0.0,
            hard_face_id=0,
        )
        self.assertIsNone(result.plan)
        self.assertIn(
            "terminal_corridor_blocked",
            result.diagnostics.candidates[0].rejection_reasons,
        )

    def test_exact_entry_boundary_and_connector_are_conservatively_rejected(self):
        config = DynamicApproachConfig(
            standoff_distance_m=0.30,
            terminal_corridor_length_m=0.45,
        )
        # North entry is exactly y=1.80. The closed supercover must see row 18.
        costmap = costmap_from_blocked(blocked={GridCell(10, 18)})
        result = plan_dynamic_approach(
            costmap,
            self.start,
            self.stand,
            0.0,
            hard_face_id=0,
            config=config,
        )
        self.assertIsNone(result.plan)
        reasons = result.diagnostics.candidates[0].rejection_reasons
        self.assertIn("entry_not_traversable", reasons)
        self.assertIn("terminal_corridor_blocked", reasons)

    def test_standoff_must_clear_configuration_space_keepout(self):
        config = DynamicApproachConfig(
            stand_radius_m=0.08,
            stand_position_uncertainty_m=0.03,
            robot_radius_m=0.13,
            collision_margin_m=0.02,
            standoff_distance_m=0.25,
        )
        result = plan_dynamic_approach(
            costmap_from_blocked(), self.start, self.stand, 0.0, config=config
        )
        self.assertIsNone(result.plan)
        self.assertTrue(
            all(
                "standoff_inside_stand_keepout" in item.rejection_reasons
                for item in result.diagnostics.candidates
            )
        )

    def test_standoff_outside_body_envelope_but_inside_execution_tube_is_rejected(self):
        config = DynamicApproachConfig(
            stand_radius_m=0.06,
            stand_position_uncertainty_m=0.02,
            robot_radius_m=0.105,
            collision_margin_m=0.02,
            tracking_margin_m=0.03,
            standoff_distance_m=0.22,
            lidar_stop_distance_m=0.10,
            lidar_clearance_margin_m=0.0,
        )
        self.assertAlmostEqual(config.stand_keepout_radius_m, 0.235)

        result = plan_dynamic_approach(
            costmap_from_blocked(), self.start, self.stand, 0.0, config=config
        )

        self.assertIsNone(result.plan)
        self.assertTrue(
            all(
                "standoff_inside_stand_keepout" in item.rejection_reasons
                for item in result.diagnostics.candidates
            )
        )

    def test_standoff_must_be_compatible_with_lidar_stop_geometry(self):
        config = DynamicApproachConfig(
            stand_radius_m=0.06,
            robot_radius_m=0.05,
            stand_position_uncertainty_m=0.0,
            collision_margin_m=0.0,
            standoff_distance_m=0.30,
            lidar_stop_distance_m=0.18,
            scan_origin_to_base_offset_m=0.05,
            lidar_clearance_margin_m=0.02,
        )
        self.assertAlmostEqual(config.minimum_lidar_standoff_m, 0.31)
        result = plan_dynamic_approach(
            costmap_from_blocked(), self.start, self.stand, 0.0, config=config
        )
        self.assertIsNone(result.plan)
        self.assertIn(
            "standoff_incompatible_with_lidar_stop",
            result.diagnostics.candidates[0].rejection_reasons,
        )

    def test_terminal_waypoints_are_protected_collinear_and_final_yaw_faces_stand(self):
        result = plan_dynamic_approach(
            costmap_from_blocked(),
            self.start,
            self.stand,
            0.0,
            hard_face_id=0,
        )
        self.assertIsNotNone(result.plan)
        plan = result.plan
        corridor = [waypoint for waypoint in plan.waypoints if waypoint.corridor]
        self.assertGreaterEqual(len(corridor), 7)
        self.assertTrue(all(waypoint.protected for waypoint in corridor))
        expected_yaw = math.atan2(
            self.stand.y_m - plan.target.y_m,
            self.stand.x_m - plan.target.x_m,
        )
        self.assertTrue(
            all(
                math.isnan(item.pose.yaw_rad)
                for item in plan.waypoints
                if not item.corridor
            )
        )
        self.assertTrue(
            all(
                abs(item.pose.yaw_rad - expected_yaw) < 1e-9
                for item in corridor
            )
        )
        self.assertAlmostEqual(
            math.hypot(
                corridor[0].pose.x_m - corridor[-1].pose.x_m,
                corridor[0].pose.y_m - corridor[-1].pose.y_m,
            ),
            0.40,
        )
        cross_products = []
        for waypoint in corridor[1:-1]:
            cross_products.append(
                (waypoint.pose.x_m - corridor[0].pose.x_m)
                * (corridor[-1].pose.y_m - corridor[0].pose.y_m)
                - (waypoint.pose.y_m - corridor[0].pose.y_m)
                * (corridor[-1].pose.x_m - corridor[0].pose.x_m)
            )
        self.assertTrue(all(abs(value) < 1e-9 for value in cross_products))

    def test_every_route_segment_avoids_dynamic_stand_keepout(self):
        base = costmap_from_blocked()
        config = DynamicApproachConfig(tracking_margin_m=0.03)
        result = plan_dynamic_approach(base, self.start, self.stand, 0.0, config=config)
        self.assertIsNotNone(result.plan)
        self.assertAlmostEqual(result.diagnostics.keepout_radius_m, 0.235)
        augmented, keepout = with_dynamic_stand_keepout(base, self.stand, config)
        poses = [item.pose for item in result.plan.waypoints]
        for first, second in zip(poses, poses[1:]):
            cells = set(supercover_segment_cells(augmented, first, second))
            self.assertTrue(cells.isdisjoint(keepout))
            self.assertTrue(segment_is_collision_free(augmented, first, second))

    def test_path_turns_into_sampled_straight_corridor_for_controller_lookahead(self):
        result = plan_dynamic_approach(
            costmap_from_blocked(),
            Pose2D(0.25, 0.25),
            self.stand,
            0.0,
            hard_face_id=0,
        )
        self.assertIsNotNone(result.plan)
        waypoints = result.plan.waypoints
        first_corridor = next(i for i, item in enumerate(waypoints) if item.corridor)
        self.assertGreaterEqual(first_corridor, 2)
        # The exact entry is retained twice as a zero-length semantic boundary:
        # unconstrained staging first, then protected forward corridor motion.
        incoming_start = waypoints[first_corridor - 2].pose
        incoming = waypoints[first_corridor - 1].pose
        entry = waypoints[first_corridor].pose
        target = waypoints[-1].pose
        self.assertAlmostEqual(
            math.hypot(entry.x_m - incoming.x_m, entry.y_m - incoming.y_m),
            0.0,
        )
        incoming_vector = (
            incoming.x_m - incoming_start.x_m,
            incoming.y_m - incoming_start.y_m,
        )
        corridor_vector = (target.x_m - entry.x_m, target.y_m - entry.y_m)
        cross = incoming_vector[0] * corridor_vector[1] - incoming_vector[1] * corridor_vector[0]
        self.assertGreater(abs(cross), 0.01)
        self.assertGreaterEqual(len(waypoints) - first_corridor, 7)

    def test_controller_pursuit_cannot_bypass_collision_checked_terminal_bend(self):
        blocked = {
            GridCell(14, 18),
            GridCell(14, 19),
            GridCell(14, 20),
            GridCell(15, 18),
            GridCell(16, 18),
            GridCell(16, 19),
            GridCell(16, 20),
            GridCell(17, 18),
        }
        costmap = costmap_from_blocked(width=50, height=50, blocked=blocked)
        result = plan_dynamic_approach(
            costmap,
            Pose2D(0.25, 0.25),
            Pose2D(2.5, 2.5),
            math.pi / 3.0,
            hard_face_id=0,
        )
        self.assertIsNotNone(result.plan)
        poses = tuple(item.pose for item in result.plan.waypoints)
        step = compute_waypoint_command(
            poses[0],
            poses,
            0,
            ControllerConfig(enforce_heading_corridor=True),
        )
        self.assertLess(step.pursuit_index, len(poses))
        self.assertTrue(
            segment_is_collision_free(costmap, poses[0], poses[step.pursuit_index])
        )

    def test_handoff_join_pursues_only_certified_start_before_lookahead(self):
        blocked = {
            GridCell(7, 6),
            GridCell(7, 7),
            GridCell(7, 8),
            GridCell(8, 6),
            GridCell(8, 7),
            GridCell(9, 6),
            GridCell(10, 6),
            GridCell(10, 7),
        }
        base = costmap_from_blocked(width=50, height=50, blocked=blocked)
        config = DynamicApproachConfig()
        result = plan_dynamic_approach(
            base,
            Pose2D(0.55, 0.55),
            Pose2D(2.5, 2.5),
            0.0,
            hard_face_id=0,
            config=config,
        )
        self.assertIsNotNone(result.plan)
        poses = tuple(item.pose for item in result.plan.waypoints)
        current = Pose2D(0.6814695, 0.6044565, 0.0)
        augmented, _ = with_dynamic_stand_keepout(
            base, Pose2D(2.5, 2.5), config
        )
        self.assertLess(
            math.hypot(current.x_m - poses[0].x_m, current.y_m - poses[0].y_m),
            result.diagnostics.start_join_clearance_m,
        )
        ordinary = compute_waypoint_command(
            current,
            poses,
            0,
            ControllerConfig(enforce_heading_corridor=True),
        )
        self.assertFalse(
            segment_is_collision_free(
                augmented, current, poses[ordinary.pursuit_index]
            )
        )
        join = compute_join_anchor_command(
            current,
            poses[0],
            ControllerConfig(enforce_heading_corridor=True),
        )
        self.assertEqual(join.pursuit_index, 0)
        self.assertTrue(segment_is_collision_free(augmented, current, poses[0]))


if __name__ == "__main__":
    unittest.main()
