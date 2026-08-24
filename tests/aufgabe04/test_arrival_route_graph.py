from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.missions.arrival_route_graph import (
    ArrivalRouteNode,
    NonTargetStandKeepout,
    _continuous_non_target_clearances,
    _with_non_target_stand_keepouts,
    build_arrival_route_graph,
    build_required_arrival_route_graph,
    resolve_station_arrival_order,
    selected_edges,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    DynamicApproachConfig,
    FaceNormalCandidate,
    circular_keepout_cells,
)
from scripts.aufgabe04.navigation.planning.map_io import CELL_FREE, MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.stations.arrival_pose_geometry import (
    ArrivalGeometryConfig,
    arrival_face_candidates,
)


def free_costmap() -> Costmap:
    width = height = 80
    rows = tuple(tuple(CELL_FREE for _ in range(width)) for _ in range(height))
    return Costmap.from_occupancy_grid(
        OccupancyGrid(
            metadata=MapMetadata(
                yaml_path=Path("map.yaml"),
                image_path=Path("map.pgm"),
                resolution=0.05,
                origin=(0.0, 0.0, 0.0),
                negate=0,
                occupied_thresh=0.65,
                free_thresh=0.2,
                mode="trinary",
            ),
            width=width,
            height=height,
            cells=rows,
        )
    )


def arena_frame_free_costmap() -> Costmap:
    """Free raster with the resolution/origin of the Aufgabe 04 arena map."""

    width = 120
    height = 80
    rows = tuple(tuple(CELL_FREE for _ in range(width)) for _ in range(height))
    return Costmap.from_occupancy_grid(
        OccupancyGrid(
            metadata=MapMetadata(
                yaml_path=Path("arena.yaml"),
                image_path=Path("arena.pgm"),
                resolution=0.05,
                origin=(-2.82, -1.69, 0.0),
                negate=0,
                occupied_thresh=0.65,
                free_thresh=0.2,
                mode="trinary",
            ),
            width=width,
            height=height,
            cells=rows,
        )
    )


def node(station_id: str, stand: Pose2D, axis: float, face_id: int):
    config = DynamicApproachConfig(
        standoff_distance_m=0.32,
        terminal_corridor_length_m=0.40,
    )
    geometry = arrival_face_candidates(
        stand,
        axis,
        ArrivalGeometryConfig(
            standoff_distance_m=config.standoff_distance_m,
            terminal_corridor_length_m=config.terminal_corridor_length_m,
        ),
    )[face_id]
    return ArrivalRouteNode(
        station_id=station_id,
        arrival_id=f"{station_id}/face_{face_id}",
        stand=stand,
        face=FaceNormalCandidate(
            face_id=face_id,
            normal_rad=geometry.outward_normal_rad,
            target=geometry.target_pose,
            entry=geometry.corridor_entry_pose,
        ),
        config=config,
    )


def frozen_catalog_node(
    station_id: str,
    stand: tuple[float, float],
    normal_rad: float,
    target: tuple[float, float],
    entry: tuple[float, float],
) -> ArrivalRouteNode:
    config = DynamicApproachConfig(
        standoff_distance_m=0.30,
        terminal_corridor_length_m=0.40,
    )
    yaw_rad = math.atan2(
        math.sin(normal_rad + math.pi),
        math.cos(normal_rad + math.pi),
    )
    return ArrivalRouteNode(
        station_id=station_id,
        arrival_id=f"{station_id}::face",
        stand=Pose2D(*stand),
        face=FaceNormalCandidate(
            face_id=0,
            normal_rad=normal_rad,
            target=Pose2D(*target, yaw_rad),
            entry=Pose2D(*entry, yaw_rad),
        ),
        config=config,
    )


class ArrivalRouteGraphTest(unittest.TestCase):
    def test_safe_exact_source_with_unsafe_cell_center_uses_egress_anchor(self):
        costmap = free_costmap()
        source = frozen_catalog_node(
            "source",
            (1.024, 1.025),
            0.0,
            (1.285, 1.025),
            (1.685, 1.025),
        )
        target = node("target", Pose2D(3.0, 1.025), 0.0, 1)

        graph = build_arrival_route_graph(
            costmap,
            Pose2D(0.2, 0.2),
            (source, target),
        )
        edge = graph.edges[(source.arrival_id, target.arrival_id)]
        overlay = edge.non_target_overlay

        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertTrue(overlay.start_cell_was_rasterized)
        self.assertGreater(overlay.exact_start_minimum_margin_m, 0.0)
        self.assertLessEqual(overlay.cell_center_minimum_margin_m, 0.0)
        self.assertFalse(overlay.start_cell_exempted)
        self.assertEqual(
            overlay.blocked_cell_count,
            overlay.rasterized_cell_count,
        )
        self.assertIsNone(overlay.egress_failure_reason)
        self.assertIsNotNone(overlay.egress_anchor)
        self.assertIsNotNone(overlay.egress_anchor_cell)
        self.assertTrue(overlay.egress_cells)
        self.assertGreater(overlay.egress_connector_minimum_margin_m, 0.0)
        self.assertTrue(overlay.egress_continuous_clearance_validated)

        self.assertIsNotNone(
            edge.result.plan,
            edge.result.diagnostics.failure_reason,
        )
        assert edge.result.plan is not None
        anchor = overlay.egress_anchor
        anchor_cell = overlay.egress_anchor_cell
        assert anchor is not None
        assert anchor_cell is not None
        exact_start = source.face.target
        self.assertAlmostEqual(
            edge.result.plan.waypoints[0].pose.x_m,
            exact_start.x_m,
        )
        self.assertAlmostEqual(
            edge.result.plan.waypoints[0].pose.y_m,
            exact_start.y_m,
        )
        self.assertAlmostEqual(edge.result.plan.waypoints[1].pose.x_m, anchor.x_m)
        self.assertAlmostEqual(edge.result.plan.waypoints[1].pose.y_m, anchor.y_m)
        self.assertEqual(costmap.world_to_grid(anchor), anchor_cell)

        radius_m = source.config.non_target_stand_keepout_radius_m
        rasterized = circular_keepout_cells(costmap, source.stand, radius_m)
        self.assertNotIn(anchor_cell, rasterized)
        outward_dot = (
            (anchor.x_m - exact_start.x_m)
            * (exact_start.x_m - source.stand.x_m)
            + (anchor.y_m - exact_start.y_m)
            * (exact_start.y_m - source.stand.y_m)
        )
        self.assertGreater(outward_dot, 0.0)
        first_segment = SimpleNamespace(
            waypoints=edge.result.plan.waypoints[:2],
        )
        (clearance,) = _continuous_non_target_clearances(
            first_segment,
            (
                NonTargetStandKeepout(
                    station_id=source.station_id,
                    stand=source.stand,
                    radius_m=radius_m,
                ),
            ),
        )
        self.assertGreater(clearance.minimum_route_clearance_m, radius_m)

    def test_no_safe_egress_anchor_keeps_source_edge_unreachable(self):
        base = free_costmap()
        source = frozen_catalog_node(
            "source",
            (1.024, 1.025),
            0.0,
            (1.285, 1.025),
            (1.685, 1.025),
        )
        target = node("target", Pose2D(3.0, 1.025), 0.0, 1)
        start_cell = base.world_to_grid(source.face.target)
        boxed_in = base.with_blocked_cells(
            GridCell(x, y)
            for y in range(base.height)
            for x in range(base.width)
            if GridCell(x, y) != start_cell
        )
        self.assertTrue(boxed_in.is_traversable(start_cell))

        graph = build_arrival_route_graph(
            boxed_in,
            Pose2D(0.2, 0.2),
            (source, target),
        )
        edge = graph.edges[(source.arrival_id, target.arrival_id)]
        overlay = edge.non_target_overlay

        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertTrue(overlay.start_cell_was_rasterized)
        self.assertGreater(overlay.exact_start_minimum_margin_m, 0.0)
        self.assertLessEqual(overlay.cell_center_minimum_margin_m, 0.0)
        self.assertFalse(overlay.start_cell_exempted)
        self.assertIsNone(overlay.egress_anchor)
        self.assertIsNone(overlay.egress_anchor_cell)
        self.assertFalse(overlay.egress_cells)
        self.assertFalse(overlay.egress_continuous_clearance_validated)
        self.assertEqual(
            overlay.egress_failure_reason,
            "start_egress_no_safe_anchor",
        )
        self.assertIsNone(edge.result.plan)
        self.assertIn(
            "start_egress_no_safe_anchor",
            edge.result.diagnostics.failure_reason,
        )

        keepout = NonTargetStandKeepout(
            station_id=source.station_id,
            stand=source.stand,
            radius_m=source.config.non_target_stand_keepout_radius_m,
        )
        planning_costmap, direct_diagnostics = _with_non_target_stand_keepouts(
            boxed_in,
            (keepout,),
            start=source.face.target,
        )
        self.assertTrue(planning_costmap.is_blocked(start_cell))
        self.assertEqual(
            direct_diagnostics.egress_failure_reason,
            "start_egress_no_safe_anchor",
        )

    def test_safe_source_cell_raster_artifact_does_not_close_catalog_graph(self):
        # Exact values from the frozen gazebo_arrival_e2e_006 catalog.  A and
        # C are 0.300 m from their source stands, but their containing 5 cm
        # cells overlap the conservative 0.260 m source-stand raster disk.
        first = frozen_catalog_node(
            "station_A",
            (-0.395, -0.415),
            0.794987466134322,
            (-0.18491188596302915, -0.200843551718869),
            (0.09520559941959861, 0.08469837932263896),
        )
        second = frozen_catalog_node(
            "station_B",
            (-1.695, -0.615),
            0.32815771314900855,
            (-1.4110086846126604, -0.5183101205680319),
            (-1.0323535974295406, -0.38939028132540787),
        )
        third = frozen_catalog_node(
            "station_C",
            (0.405, 0.685),
            -2.0774372388223883,
            (0.25942713820416263, 0.42268617667273045),
            (0.06532998914304616, 0.07293441223637098),
        )

        graph = build_arrival_route_graph(
            arena_frame_free_costmap(),
            Pose2D(1.55, -0.60, 2.996),
            (first, second, third),
        )

        for source in (first, third):
            for target in (first, second, third):
                if source is target:
                    continue
                edge = graph.edges[(source.arrival_id, target.arrival_id)]
                self.assertIsNotNone(edge.result.plan)
                first_pose = edge.result.plan.waypoints[0].pose
                self.assertAlmostEqual(first_pose.x_m, source.face.target.x_m)
                self.assertAlmostEqual(first_pose.y_m, source.face.target.y_m)
                self.assertTrue(edge.non_target_overlay.start_cell_was_rasterized)
                self.assertTrue(edge.non_target_overlay.start_cell_exempted)
                self.assertGreater(
                    edge.non_target_overlay.exact_start_minimum_margin_m,
                    0.0,
                )
                self.assertGreater(
                    edge.non_target_overlay.cell_center_minimum_margin_m,
                    0.0,
                )
                self.assertGreater(
                    edge.non_target_overlay.start_connector_minimum_margin_m,
                    0.0,
                )
                self.assertTrue(
                    all(
                        clearance.minimum_route_clearance_m
                        > clearance.radius_m
                        for clearance in edge.non_target_clearances
                    )
                )

    def test_source_cell_exemption_rejects_unsafe_center_or_connector(self):
        costmap = free_costmap()

        center_inside = NonTargetStandKeepout(
            station_id="center_inside",
            stand=Pose2D(1.024, 1.025),
            radius_m=0.26,
        )
        overlay, diagnostics = _with_non_target_stand_keepouts(
            costmap,
            (center_inside,),
            start=Pose2D(1.285, 1.025),
        )
        self.assertTrue(diagnostics.start_cell_was_rasterized)
        self.assertGreater(diagnostics.exact_start_minimum_margin_m, 0.0)
        self.assertLessEqual(diagnostics.cell_center_minimum_margin_m, 0.0)
        self.assertFalse(diagnostics.start_cell_exempted)
        self.assertTrue(overlay.is_blocked(diagnostics.start_cell))

        connector_crosses = NonTargetStandKeepout(
            station_id="connector_crosses",
            stand=Pose2D(1.013, 1.013),
            radius_m=0.005,
        )
        overlay, diagnostics = _with_non_target_stand_keepouts(
            costmap,
            (connector_crosses,),
            start=Pose2D(1.001, 1.001),
        )
        self.assertGreater(diagnostics.exact_start_minimum_margin_m, 0.0)
        self.assertGreater(diagnostics.cell_center_minimum_margin_m, 0.0)
        self.assertLessEqual(diagnostics.start_connector_minimum_margin_m, 0.0)
        self.assertFalse(diagnostics.start_cell_exempted)
        self.assertTrue(overlay.is_blocked(diagnostics.start_cell))

    def test_source_cell_exemption_never_removes_static_occupancy(self):
        costmap = arena_frame_free_costmap()
        start = Pose2D(-0.18491188596302915, -0.200843551718869)
        start_cell = costmap.world_to_grid(start)
        statically_blocked = costmap.with_blocked_cells((start_cell,))
        keepout = NonTargetStandKeepout(
            station_id="station_A",
            stand=Pose2D(-0.395, -0.415),
            radius_m=0.26,
        )

        overlay, diagnostics = _with_non_target_stand_keepouts(
            statically_blocked,
            (keepout,),
            start=start,
        )

        self.assertTrue(diagnostics.start_cell_was_rasterized)
        self.assertFalse(diagnostics.start_cell_exempted)
        self.assertTrue(overlay.is_blocked(start_cell))

    def test_non_target_transit_uses_lidar_radius_and_preserves_target_corridor(self):
        non_target = node("A", Pose2D(1.5, 1.49), 0.0, 0)
        target = node("B", Pose2D(3.0, 1.0), 0.0, 0)
        start = Pose2D(0.4, 1.72)

        self.assertAlmostEqual(non_target.config.stand_keepout_radius_m, 0.205)
        self.assertAlmostEqual(non_target.config.minimum_lidar_standoff_m, 0.26)
        self.assertAlmostEqual(
            non_target.config.non_target_stand_keepout_radius_m,
            0.26,
        )
        graph = build_arrival_route_graph(
            free_costmap(),
            start,
            (non_target, target),
        )
        edge = graph.edges[("mission_start", target.arrival_id)]

        self.assertIsNotNone(edge.result.plan)
        self.assertEqual(edge.result.plan.target, target.face.target)
        self.assertEqual(edge.result.plan.entry, target.face.entry)
        self.assertEqual(len(edge.non_target_clearances), 1)
        clearance = edge.non_target_clearances[0]
        self.assertEqual(clearance.station_id, "A")
        self.assertAlmostEqual(clearance.radius_m, 0.26)
        self.assertGreater(clearance.minimum_route_clearance_m, 0.26)

    def test_continuous_non_target_validation_rejects_smoothed_chord(self):
        crossing_plan = SimpleNamespace(
            waypoints=(
                SimpleNamespace(pose=Pose2D(0.0, 0.0)),
                SimpleNamespace(pose=Pose2D(1.0, 0.0)),
            )
        )
        keepout = NonTargetStandKeepout(
            station_id="A",
            stand=Pose2D(0.5, 0.20),
            radius_m=0.26,
        )

        with self.assertRaisesRegex(
            ValueError,
            "non_target_stand_route_clearance_failed",
        ):
            _continuous_non_target_clearances(crossing_plan, (keepout,))

    def test_graph_marks_edge_unreachable_when_continuous_validation_fails(self):
        first = node("A", Pose2D(1.5, 1.5), 0.0, 0)
        second = node("B", Pose2D(2.8, 2.6), math.pi / 4.0, 1)
        with patch(
            "scripts.aufgabe04.navigation.missions.arrival_route_graph."
            "_continuous_non_target_clearances",
            side_effect=ValueError("non_target_stand_route_clearance_failed:test"),
        ):
            graph = build_arrival_route_graph(
                free_costmap(),
                Pose2D(0.4, 0.4),
                (first, second),
            )

        edge = graph.edges[("mission_start", first.arrival_id)]
        self.assertIsNone(edge.result.plan)
        self.assertIn(
            "non_target_stand_route_clearance_failed",
            edge.result.diagnostics.failure_reason,
        )

    def test_normalizes_geometry_pose_shape_at_public_node_boundary(self):
        route_node = node("A", Pose2D(1.5, 1.5), 0.0, 0)

        self.assertIs(type(route_node.stand), Pose2D)
        self.assertIs(type(route_node.face.target), Pose2D)
        self.assertIs(type(route_node.face.entry), Pose2D)

    def test_builds_directed_exact_target_routes(self):
        first = node("A", Pose2D(1.5, 1.5), 0.0, 0)
        second = node("B", Pose2D(2.8, 2.6), math.pi / 4.0, 1)

        graph = build_arrival_route_graph(
            free_costmap(),
            Pose2D(0.4, 0.4),
            (first, second),
        )

        self.assertEqual(len(graph.edges), 4)
        self.assertIsNotNone(graph.directed_costs[("mission_start", first.arrival_id)])
        self.assertIsNotNone(graph.directed_costs[(first.arrival_id, second.arrival_id)])
        edges = selected_edges(graph, (first.arrival_id, second.arrival_id))
        self.assertEqual(edges[-1].result.plan.target, second.face.target)
        self.assertEqual(edges[-1].result.plan.entry, second.face.entry)

    def test_task_order_plans_only_required_directed_edges(self):
        first = node("A", Pose2D(1.5, 1.5), 0.0, 0)
        second = node("B", Pose2D(2.8, 2.6), math.pi / 4.0, 1)
        arrival_order = resolve_station_arrival_order(
            (first, second), ("B", "A")
        )

        graph = build_required_arrival_route_graph(
            free_costmap(),
            Pose2D(0.4, 0.4),
            (first, second),
            arrival_order,
        )

        self.assertEqual(
            tuple(graph.edges),
            (
                ("mission_start", second.arrival_id),
                (second.arrival_id, first.arrival_id),
            ),
        )
        self.assertEqual(
            tuple(edge.target_id for edge in selected_edges(graph, arrival_order)),
            arrival_order,
        )

    def test_station_order_rejects_ambiguous_candidate_identity(self):
        first = node("A", Pose2D(1.5, 1.5), 0.0, 0)
        alternate = node("A", Pose2D(1.6, 1.5), 0.0, 1)

        with self.assertRaisesRegex(ValueError, "ambiguous frozen arrivals"):
            resolve_station_arrival_order((first, alternate), ("A",))

    def test_task_order_preserves_nonconsecutive_repeated_station(self):
        first = node("A", Pose2D(1.5, 1.5), 0.0, 0)
        second = node("B", Pose2D(2.8, 2.6), math.pi / 4.0, 1)
        arrivals = resolve_station_arrival_order(
            (first, second), ("B", "A", "B")
        )

        graph = build_required_arrival_route_graph(
            free_costmap(),
            Pose2D(0.4, 0.4),
            (first, second),
            arrivals,
        )

        self.assertEqual(arrivals, (second.arrival_id, first.arrival_id, second.arrival_id))
        self.assertEqual(len(selected_edges(graph, arrivals)), 3)

    def test_selected_edges_rejects_uncomputed_order(self):
        first = node("A", Pose2D(1.5, 1.5), 0.0, 0)
        graph = build_arrival_route_graph(
            free_costmap(), Pose2D(0.4, 0.4), (first,)
        )

        with self.assertRaisesRegex(ValueError, "missing selected edge"):
            selected_edges(graph, ("unknown",))


if __name__ == "__main__":
    unittest.main()
