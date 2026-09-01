import unittest

from scripts.aufgabe04.navigation.execution import route_uncertainty_defaults
from scripts.aufgabe04.navigation.station_segment.cli import build_parser
from scripts.aufgabe04.real_robot.execution import child_runner


class RouteUncertaintyDefaultsTest(unittest.TestCase):
    def test_child_exports_and_station_cli_share_one_execution_budget(self):
        args = build_parser().parse_args(["--leg-index", "0"])

        self.assertEqual(
            child_runner.DEFAULT_TRACKING_TUBE_RADIUS_M,
            route_uncertainty_defaults.DEFAULT_TRACKING_TUBE_RADIUS_M,
        )
        self.assertEqual(
            child_runner.DEFAULT_COLLISION_MARGIN_M,
            route_uncertainty_defaults.DEFAULT_COLLISION_MARGIN_M,
        )
        self.assertEqual(
            child_runner.DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
            route_uncertainty_defaults.DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
        )
        self.assertEqual(
            args.certified_route_tube_radius_m,
            route_uncertainty_defaults.DEFAULT_TRACKING_TUBE_RADIUS_M,
        )
        self.assertEqual(
            args.uncertainty_collision_margin_m,
            route_uncertainty_defaults.DEFAULT_COLLISION_MARGIN_M,
        )
        self.assertEqual(
            args.uncertainty_odom_drift_bound_m,
            route_uncertainty_defaults.DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
        )
        self.assertEqual(
            args.uncertainty_braking_latency_distance_m,
            route_uncertainty_defaults.DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
        )
        self.assertEqual(
            args.uncertainty_clearance_sample_spacing_m,
            route_uncertainty_defaults.DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
        )


if __name__ == "__main__":
    unittest.main()
