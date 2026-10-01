"""The metric suffix preserves obstacles while resolving candidate grid rounding."""

import copy
from dataclasses import asdict, replace
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.approach.exact_stored_goal_connector import (
    EXACT_STORED_GOAL_CONNECTOR_POLICY, candidate_goal_anchors, certify_stored_goal_connector,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import point_to_segment_distance_m
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.planning.route_smoothing import segment_is_collision_free
from scripts.aufgabe04.stations.models import Station, StationPose


def free_map(resolution=.05):
    metadata = MapMetadata(Path("map.yaml"), Path("map.pgm"), resolution,
                           (-1., -1., 0.), 0, .65, .196, "trinary")
    return Costmap.from_occupancy_grid(OccupancyGrid(
        metadata, 40, 40, tuple((0,) * 40 for _ in range(40))))


class ExactStoredGoalConnectorTest(unittest.TestCase):
    def setUp(self):
        self.base = free_map()
        self.static = self.base.with_inflation(.25)
        self.center = Pose2D(.004, .024, 0.)
        self.station = Station("start", StationPose(.004, .024, 0.), 0., .34)
        self.planning = self.static.with_station_keepouts((self.station,))
        self.target = Pose2D(-.337, .024, -.71)
        self.anchor = Pose2D(-.375, .025, 0.)
        self.calls = []

    def _validate_candidates(self, poses):
        self.calls.append(poses)
        if point_to_segment_distance_m(self.center, *poses) < .34:
            raise ValueError("original candidate keepout violated")

    def _certify(self, **changes):
        args = dict(base_costmap=self.base, static_costmap=self.static,
                    planning_costmap=self.planning, anchor=self.anchor, target=self.target,
                    inflation_radius_m=.25, validate_candidate_clearance=self._validate_candidates)
        return certify_stored_goal_connector(**{**args, **changes})

    def test_candidate_rounding_gets_full_metric_proof_without_changing_maps_or_goal(self):
        before = copy.deepcopy((self.base, self.static, self.planning))
        self.assertFalse(segment_is_collision_free(self.planning, self.target, self.target))
        self.assertTrue(segment_is_collision_free(self.static, self.anchor, self.target))
        proof = self._certify(anchor=replace(self.anchor, yaw_rad=math.nan))
        self.assertEqual(proof["policy"], EXACT_STORED_GOAL_CONNECTOR_POLICY)
        self.assertEqual(proof["target"], asdict(self.target))
        self.assertEqual(proof["anchor"], asdict(self.anchor))
        self.assertEqual(self.calls, [(self.anchor, self.target)])
        static = proof["static_clearance"]
        self.assertTrue(static["validated"])
        self.assertEqual(static["exact_start"], asdict(self.anchor))
        self.assertEqual(static["anchor"], asdict(self.target))
        self.assertAlmostEqual(static["minimum_continuous_clearance_m"],
                               static["minimum_sampled_clearance_m"] - static["sample_spacing_m"] / 2)
        self.assertGreater(static["minimum_continuous_clearance_m"], .25)
        self.assertGreaterEqual(static["sample_count"], 2)
        self.assertTrue(proof["candidate_keepouts_continuously_validated"])
        self.assertFalse(proof["motion_authorized"])
        self.assertEqual((self.base, self.static, self.planning), before)

    def test_anchor_search_is_deterministic_free_and_metric_bounded(self):
        anchors = candidate_goal_anchors(self.planning, self.target)
        self.assertTrue(anchors)
        self.assertEqual(anchors, candidate_goal_anchors(self.planning, self.target))
        self.assertAlmostEqual(anchors[0].x_m, self.anchor.x_m)
        self.assertAlmostEqual(anchors[0].y_m, self.anchor.y_m)
        keys = []
        for anchor in anchors:
            length = math.hypot(anchor.x_m - self.target.x_m, anchor.y_m - self.target.y_m)
            self.assertGreater(length, 0)
            self.assertLessEqual(length, .15 + 1e-9)
            self.assertTrue(self.planning.is_traversable(self.planning.world_to_grid(anchor)))
            cell = self.planning.world_to_grid(anchor)
            keys.append((length, cell.x, cell.y))
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(candidate_goal_anchors(self.planning, Pose2D(-2., 0., 0.)), ())
        coarse = free_map(.1)
        for anchor in candidate_goal_anchors(coarse, self.target):
            self.assertLessEqual(math.hypot(anchor.x_m - self.target.x_m,
                                           anchor.y_m - self.target.y_m), .15 + 1e-9)

    def test_callback_can_reject_measured_or_original_candidate_clearance(self):
        for message in ("original candidate keepout", "measured target standoff"):
            def rejected(poses):
                self.assertEqual(poses, (self.anchor, self.target))
                raise ValueError(message)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self._certify(validate_candidate_clearance=rejected)
        with self.assertRaisesRegex(ValueError, "must raise on failure"):
            self._certify(validate_candidate_clearance=lambda poses: False)
        with self.assertRaisesRegex(ValueError, "requires a candidate-clearance"):
            self._certify(validate_candidate_clearance=None)

    def test_static_obstacle_cannot_be_laundered_by_candidate_source_label(self):
        base = self.base.with_blocked_cells((self.base.world_to_grid(self.target),))
        static = base.with_inflation(0.)
        planning = static.with_station_keepouts((self.station,))
        # Candidate rasterization overwrites cell_sources; the separate static
        # layer must still reject the same physically blocked target.
        with self.assertRaisesRegex(ValueError, "static inflation or temporary"):
            self._certify(base_costmap=base, static_costmap=static,
                          planning_costmap=planning, inflation_radius_m=0.)
        self.assertEqual(self.calls, [])

    def test_entire_connector_checks_static_and_unsupported_overlay_cells(self):
        anchor = Pose2D(-.475, .025, 0.)
        middle = GridCell(12, 20)
        base = self.base.with_blocked_cells((middle,))
        planning = base.with_station_keepouts((self.station,))
        self.assertTrue(segment_is_collision_free(base, anchor, anchor))
        self.assertTrue(segment_is_collision_free(base, self.target, self.target))
        with self.assertRaisesRegex(ValueError, "static inflation or temporary"):
            self._certify(base_costmap=base, static_costmap=base, planning_costmap=planning,
                          inflation_radius_m=0., anchor=anchor)
        with self.assertRaisesRegex(ValueError, "unsupported planning overlay"):
            self._certify(planning_costmap=self.planning.with_blocked_cells((middle,)), anchor=anchor)

    def test_invalid_geometry_missing_raster_block_and_map_substitution_reject(self):
        cases = (
            {"anchor": self.target}, {"anchor": Pose2D(-.335, .025, 0.)},
            {"anchor": Pose2D(-.525, .025, 0.)}, {"anchor": Pose2D(-.374, .025, 0.)},
            {"target": replace(self.target, yaw_rad=math.nan)},
            {"target": replace(self.target, x_m=math.inf)},
            {"planning_costmap": self.static}, {"inflation_radius_m": -.1},
            {"inflation_radius_m": True},
            {"static_costmap": replace(self.static, metadata=replace(self.static.metadata, origin=(0., 0., 0.)))},
            {"static_costmap": self.static.with_blocked_cells((GridCell(0, 0),))},
        )
        for changed in cases:
            with self.subTest(changed=tuple(changed)), self.assertRaises(ValueError):
                self._certify(**changed)
        blocked_base = self.base.with_blocked_cells((GridCell(0, 0),))
        with self.assertRaisesRegex(ValueError, "removed static obstacles"):
            self._certify(base_costmap=blocked_base, static_costmap=blocked_base.with_inflation(.25))
        with self.assertRaises(ValueError):
            candidate_goal_anchors(self.planning, replace(self.target, y_m=math.nan))


if __name__ == "__main__":
    unittest.main()
