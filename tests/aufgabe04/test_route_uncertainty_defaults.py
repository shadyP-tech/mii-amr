from dataclasses import replace
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.navigation.execution import route_uncertainty_defaults
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance, RouteClearanceSegment, evaluate_segment_uncertainty_budget,
)
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import DynamicApproachConfig
from scripts.aufgabe04.navigation.missions import (
    plan_arrival_catalog_route, plan_detected_stand_exploration,
    run_detected_stand_observe_plan,
)
from scripts.aufgabe04.navigation.station_segment.cli import build_parser
from scripts.aufgabe04.real_robot.autonomous_runner.mission_config import _physical_clearance
from scripts.aufgabe04.real_robot.execution import child_runner


class RouteUncertaintyDefaultsTest(unittest.TestCase):
    def test_planning_and_execution_use_ten_mm_extra_collision_padding(self):
        margin = route_uncertainty_defaults.DEFAULT_COLLISION_MARGIN_M
        self.assertEqual(margin, 0.01)
        self.assertEqual(DynamicApproachConfig().collision_margin_m, margin)
        for module in (plan_arrival_catalog_route, plan_detected_stand_exploration,
                       run_detected_stand_observe_plan):
            with self.subTest(parser=module.__name__):
                self.assertEqual(module.build_parser().get_default("collision_margin_m"), margin)
        physical = _physical_clearance(
            SimpleNamespace(robot_radius_m=.105, scan_origin_to_base_offset_m=0.),
            approach_offset_m=.5,
        )
        self.assertAlmostEqual(physical["minimum_static_inflation_m"], .25)
        self.assertAlmostEqual(physical["minimum_candidate_transit_radius_m"], .34)

    def test_recorded_clearance_budget_changes_only_collision_reserve(self):
        # Rounded aggregate values from the Oct 2 audit; this is not a route replay.
        position_sigma = .129376 / 2.
        segment = RouteClearanceSegment(
            segment_id="audited_limiting_subsegment",
            raw_centerline_clearance_m=.372508, robot_radius_m=.105,
            collision_margin_m=route_uncertainty_defaults.DEFAULT_COLLISION_MARGIN_M,
            fixed_odom_tracking_bound_m=route_uncertainty_defaults.DEFAULT_TRACKING_TUBE_RADIUS_M,
            empirical_odom_drift_bound_m=route_uncertainty_defaults.DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
            braking_latency_distance_m=route_uncertainty_defaults.DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
            localization_sigma_multiplier=route_uncertainty_defaults.DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
            heading_contribution_m=.056892,
            covariance=PlanarCovariance(position_sigma ** 2, 0., position_sigma ** 2),
            segment_normal_x=1., segment_normal_y=0., is_corner=False,
        )
        previous = evaluate_segment_uncertainty_budget(replace(segment, collision_margin_m=.02))
        current = evaluate_segment_uncertainty_budget(segment)
        self.assertFalse(previous.accepted)
        self.assertTrue(current.accepted)
        self.assertAlmostEqual(previous.remaining_margin_m, -.003760)
        self.assertAlmostEqual(current.remaining_margin_m, .006240)
        old_budget, new_budget = previous.evidence["budget_m"], current.evidence["budget_m"]
        for term in ("robot_radius_m", "fixed_odom_tracking_bound_m", "empirical_odom_drift_bound_m",
                     "braking_latency_distance_m", "projected_localization_term_m", "heading_contribution_m"):
            with self.subTest(term=term):
                self.assertEqual(old_budget[term], new_budget[term])
        required = new_budget["required_clearance_m"]
        for available in (required, required - .001):
            with self.subTest(available=available):
                self.assertFalse(evaluate_segment_uncertainty_budget(
                    replace(segment, raw_centerline_clearance_m=available)).accepted)

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
