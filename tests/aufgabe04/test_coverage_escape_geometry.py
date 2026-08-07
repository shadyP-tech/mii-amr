import math
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.coverage_escape_geometry import (
    EGRESS_MODE_FORWARD,
    EGRESS_MODE_STRAIGHT_REVERSE,
    CircularEscapeKeepout,
    EgressConnectorChoice,
    choose_egress_connectors,
    find_reverse_transition_anchors,
    validate_executable_escape_route,
)
from scripts.aufgabe04.navigation.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.models import Pose2D


def _open_costmap() -> Costmap:
    width = 80
    height = 80
    grid = OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.05,
            origin=(-2.0, -2.0, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.2,
            mode="trinary",
        ),
        width=width,
        height=height,
        cells=tuple(tuple(CELL_FREE for _x in range(width)) for _y in range(height)),
    )
    return Costmap.from_occupancy_grid(grid)


class CoverageEscapeGeometryTest(unittest.TestCase):
    def setUp(self):
        self.costmap = _open_costmap()
        self.recorded_start = Pose2D(
            -0.8636705568,
            -0.4674798031,
            3.06736,
        )
        self.keepout = CircularEscapeKeepout(
            candidate_uid="recorded_blocker",
            center=Pose2D(-1.13, -0.467, 0.0),
            hard_exclusion_radius_m=0.235,
            route_keepout_radius_m=0.34,
        )

    def _recorded_connector(self) -> EgressConnectorChoice:
        anchor = Pose2D(-0.745, -0.465, 0.0)
        heading = math.atan2(
            anchor.y_m - self.recorded_start.y_m,
            anchor.x_m - self.recorded_start.x_m,
        )
        reverse_error = math.atan2(
            math.sin(heading + math.pi - self.recorded_start.yaw_rad),
            math.cos(heading + math.pi - self.recorded_start.yaw_rad),
        )
        return EgressConnectorChoice(
            anchor=anchor,
            mode=EGRESS_MODE_STRAIGHT_REVERSE,
            connector_distance_m=math.hypot(
                anchor.x_m - self.recorded_start.x_m,
                anchor.y_m - self.recorded_start.y_m,
            ),
            connector_heading_error_rad=reverse_error,
            minimum_hard_clearance_m=0.03,
        )

    def test_forward_connectors_are_ordered_before_reverse_connectors(self):
        start = Pose2D(0.025, 0.025, 0.0)

        choices = choose_egress_connectors(
            self.costmap,
            self.costmap,
            start,
            (),
            blocker_candidate_uids=(),
            search_radius_m=0.20,
        )

        self.assertTrue(choices)
        self.assertEqual(choices[0].mode, EGRESS_MODE_FORWARD)
        self.assertIn(
            EGRESS_MODE_STRAIGHT_REVERSE,
            {choice.mode for choice in choices},
        )

    def test_recorded_early_corner_is_not_an_executable_reverse_escape(self):
        old_anchor = Pose2D(-0.745, -0.465, 0.0)
        old_next = Pose2D(-0.745, -0.415, 0.0)
        route = (
            self.recorded_start,
            old_anchor,
            old_next,
            Pose2D(-0.695, -0.415, 0.0),
        )

        with self.assertRaisesRegex(
            ValueError,
            "material multi-corner reverse chain",
        ):
            validate_executable_escape_route(
                self.costmap,
                self.costmap,
                self.recorded_start,
                self._recorded_connector(),
                route,
                (self.keepout,),
                transition_waypoint_index=2,
                tracking_tube_radius_m=0.03,
            )

    def test_recorded_sideways_chord_is_not_misclassified_as_forward(self):
        choices = choose_egress_connectors(
            self.costmap,
            self.costmap,
            self.recorded_start,
            (self.keepout,),
            blocker_candidate_uids=(self.keepout.candidate_uid,),
            search_radius_m=0.40,
            forward_translation_heading_limit_rad=1.25,
        )

        self.assertTrue(choices)
        self.assertEqual(choices[0].mode, EGRESS_MODE_STRAIGHT_REVERSE)
        self.assertNotIn(
            EGRESS_MODE_FORWARD,
            {choice.mode for choice in choices},
        )
        sideways = Pose2D(-0.845, -0.165, 0.0)
        sideways_heading = math.atan2(
            sideways.y_m - self.recorded_start.y_m,
            sideways.x_m - self.recorded_start.x_m,
        )
        sideways_error = math.atan2(
            math.sin(sideways_heading - self.recorded_start.yaw_rad),
            math.cos(sideways_heading - self.recorded_start.yaw_rad),
        )
        self.assertGreater(abs(sideways_error), 1.25)
        self.assertLess(abs(sideways_error), math.pi / 2.0)

    def test_straight_reverse_prefix_has_explicit_forward_transition(self):
        transition = Pose2D(-0.695, -0.465, 0.0)
        route = (
            self.recorded_start,
            self._recorded_connector().anchor,
            transition,
            Pose2D(-0.695, -0.315, 0.0),
        )

        geometry = validate_executable_escape_route(
            self.costmap,
            self.costmap,
            self.recorded_start,
            self._recorded_connector(),
            route,
            (self.keepout,),
            transition_waypoint_index=2,
            tracking_tube_radius_m=0.03,
        )

        self.assertEqual(geometry.mode, EGRESS_MODE_STRAIGHT_REVERSE)
        self.assertEqual(geometry.transition_anchor, transition)
        self.assertEqual(geometry.transition_waypoint_index, 2)
        self.assertEqual(geometry.forward_waypoint_index, 3)
        self.assertLessEqual(abs(geometry.connector_heading_error_rad), 0.10)
        self.assertGreater(
            geometry.minimum_transition_keepout_tube_clearance_m,
            0.0,
        )

    def test_transition_search_seeks_a_farther_collinear_anchor(self):
        connector = self._recorded_connector()

        transitions = find_reverse_transition_anchors(
            self.costmap,
            self.recorded_start,
            connector,
            (self.keepout,),
            tracking_tube_radius_m=0.03,
            search_radius_m=0.40,
        )

        self.assertTrue(transitions)
        first = transitions[0]
        self.assertGreater(
            first.distance_from_start_m,
            connector.connector_distance_m,
        )
        self.assertGreater(first.anchor.x_m, connector.anchor.x_m)
        self.assertLessEqual(abs(first.reverse_heading_error_rad), 0.10)
        self.assertGreater(first.minimum_keepout_tube_clearance_m, 0.0)

    def test_reverse_transition_fails_without_full_tube_clearance(self):
        enlarged = CircularEscapeKeepout(
            candidate_uid=self.keepout.candidate_uid,
            center=self.keepout.center,
            hard_exclusion_radius_m=self.keepout.hard_exclusion_radius_m,
            route_keepout_radius_m=0.38,
        )
        route = (
            self.recorded_start,
            self._recorded_connector().anchor,
            Pose2D(-0.695, -0.465, 0.0),
            Pose2D(-0.695, -0.315, 0.0),
        )

        with self.assertRaisesRegex(
            ValueError,
            "keepout plus execution-tube clearance",
        ):
            validate_executable_escape_route(
                self.costmap,
                self.costmap,
                self.recorded_start,
                self._recorded_connector(),
                route,
                (enlarged,),
                transition_waypoint_index=2,
                tracking_tube_radius_m=0.03,
            )


if __name__ == "__main__":
    unittest.main()
