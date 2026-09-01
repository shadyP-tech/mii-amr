import dataclasses
import math
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_selection import (
    NO_ACCEPTED_ROUTE_OPTIONS,
    NO_ROUTE_OPTIONS,
    ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION,
    RouteUncertaintySelectionOption,
    evaluate_route_uncertainty_selection,
    route_uncertainty_selection_evidence_sha256,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)


def open_costmap(*, width=40, height=40, resolution=1.0):
    metadata = MapMetadata(
        yaml_path=Path("map.yaml"),
        image_path=Path("map.pgm"),
        resolution=resolution,
        origin=(0.0, 0.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.20,
        mode="trinary",
    )
    grid = OccupancyGrid(
        metadata=metadata,
        width=width,
        height=height,
        cells=tuple(tuple([CELL_FREE] * width) for _ in range(height)),
    )
    return Costmap.from_occupancy_grid(grid)


def config(**overrides):
    values = {
        "robot_radius_m": 0.10,
        "collision_margin_m": 0.0,
        "fixed_odom_tracking_bound_m": 0.0,
        "empirical_odom_drift_bound_m": 0.0,
        "braking_latency_distance_m": 0.0,
        "localization_sigma_multiplier": 2.0,
        "heading_sigma_rad": 0.0,
        "heading_lever_arm_m": 0.10,
        "sampling_spacing_m": 0.25,
    }
    values.update(overrides)
    return RouteUncertaintyAdmissionConfig(**values)


def option(option_id, plan_order, route):
    return RouteUncertaintySelectionOption(
        option_id=option_id,
        plan_order=plan_order,
        map_route=route,
    )


class RouteUncertaintySelectionTest(unittest.TestCase):
    def setUp(self):
        self.costmap = open_costmap()
        self.covariance = PlanarCovariance(0.0, 0.0, 0.0)
        self.config = config()

    def evaluate(self, options):
        return evaluate_route_uncertainty_selection(
            self.costmap,
            options,
            self.covariance,
            self.config,
        )

    def test_maximum_admitted_margin_precedes_shorter_route(self):
        short_lower_margin = option(
            "short-near-edge",
            0,
            (Pose2D(1.0, 1.0), Pose2D(1.1, 1.0)),
        )
        long_higher_margin = option(
            "long-centre",
            1,
            (Pose2D(10.0, 15.0), Pose2D(13.0, 15.0)),
        )

        decision = self.evaluate((short_lower_margin, long_higher_margin))

        self.assertTrue(decision.ready)
        self.assertEqual(decision.selected_option_id, "long-centre")
        self.assertEqual(
            tuple(item.option.option_id for item in decision.ranked_options),
            ("long-centre", "short-near-edge"),
        )
        self.assertGreater(
            decision.ranked_options[0].minimum_remaining_margin_m,
            decision.ranked_options[1].minimum_remaining_margin_m,
        )
        self.assertGreater(
            decision.ranked_options[0].route_length_m,
            decision.ranked_options[1].route_length_m,
        )

    def test_shortest_route_breaks_equal_margin(self):
        longer = option(
            "longer",
            0,
            (Pose2D(10.0, 15.0), Pose2D(12.0, 15.0)),
        )
        shorter = option(
            "shorter",
            1,
            (Pose2D(10.0, 15.0), Pose2D(11.0, 15.0)),
        )

        decision = self.evaluate((longer, shorter))

        self.assertEqual(decision.selected_option_id, "shorter")
        self.assertAlmostEqual(
            decision.ranked_options[0].minimum_remaining_margin_m,
            decision.ranked_options[1].minimum_remaining_margin_m,
        )
        self.assertEqual(decision.ranked_options[0].route_length_m, 1.0)
        self.assertEqual(decision.ranked_options[1].route_length_m, 2.0)

    def test_plan_order_then_stable_id_break_exact_ties(self):
        route = (Pose2D(10.0, 15.0), Pose2D(11.0, 15.0))
        plan_first = option("z-plan-first", 0, route)
        plan_second = option("a-plan-second", 1, route)

        by_plan_order = self.evaluate((plan_second, plan_first))

        self.assertEqual(by_plan_order.selected_option_id, "z-plan-first")

        alpha = option("alpha", 5, route)
        zulu = option("zulu", 5, route)
        by_stable_id = self.evaluate((zulu, alpha))

        self.assertEqual(by_stable_id.selected_option_id, "alpha")
        self.assertEqual(
            tuple(item.option.option_id for item in by_stable_id.ranked_options),
            ("alpha", "zulu"),
        )

    def test_accepted_options_rank_before_rejections_and_every_admission_is_bound(self):
        accepted = option(
            "accepted",
            1,
            (Pose2D(10.0, 15.0), Pose2D(11.0, 15.0)),
        )
        rejected = option(
            "rejected",
            0,
            (Pose2D(0.05, 0.05), Pose2D(0.30, 0.05)),
        )

        decision = self.evaluate((rejected, accepted))

        self.assertEqual(decision.selected_option_id, "accepted")
        self.assertEqual(
            tuple(item.accepted for item in decision.ranked_options),
            (True, False),
        )
        evidence_options = decision.evidence["options"]
        self.assertEqual(len(evidence_options), 2)
        for ranked, evidence in zip(decision.ranked_options, evidence_options):
            self.assertEqual(evidence["accepted"], ranked.accepted)
            self.assertEqual(
                evidence["minimum_remaining_margin_m"],
                ranked.minimum_remaining_margin_m,
            )
            self.assertEqual(
                evidence["admission_evidence"],
                ranked.admission.to_evidence_dict(),
            )
            self.assertEqual(
                evidence["admission_evidence_sha256"],
                route_uncertainty_admission_evidence_sha256(
                    ranked.admission
                ),
            )

    def test_all_rejected_returns_fail_closed_decision_without_route(self):
        first = option(
            "edge-a",
            0,
            (Pose2D(0.05, 0.05), Pose2D(0.30, 0.05)),
        )
        second = option(
            "edge-b",
            1,
            (Pose2D(0.05, 0.50), Pose2D(0.30, 0.50)),
        )

        decision = self.evaluate((first, second))

        self.assertFalse(decision.ready)
        self.assertEqual(decision.reason, NO_ACCEPTED_ROUTE_OPTIONS)
        self.assertIsNone(decision.selected_option_id)
        self.assertIsNone(decision.selected_option)
        self.assertIsNone(decision.selected_route)
        self.assertFalse(decision.motion_authorized)
        self.assertTrue(decision.evidence["decision"]["fail_closed"])
        self.assertTrue(
            all(not item.accepted for item in decision.ranked_options)
        )

    def test_empty_options_fail_closed_without_inventing_identity(self):
        decision = self.evaluate(())

        self.assertFalse(decision.ready)
        self.assertEqual(decision.reason, NO_ROUTE_OPTIONS)
        self.assertIsNone(decision.selected_option_id)
        self.assertEqual(decision.ranked_options, ())
        self.assertEqual(decision.evidence["options"], [])
        self.assertEqual(
            decision.evidence["schema_version"],
            ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION,
        )

    def test_selected_route_is_the_original_immutable_route(self):
        route = (Pose2D(10.0, 15.0), Pose2D(11.0, 15.0))
        selected = option("unchanged", 0, route)

        decision = self.evaluate((selected,))

        self.assertIs(decision.selected_option, selected)
        self.assertIs(decision.selected_route, route)
        self.assertIs(decision.selected_route[0], route[0])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            selected.plan_order = 2

    def test_selection_evidence_is_deterministic_across_input_order(self):
        first = option(
            "first",
            0,
            (Pose2D(10.0, 15.0), Pose2D(11.0, 15.0)),
        )
        second = option(
            "second",
            1,
            (Pose2D(10.0, 15.0), Pose2D(12.0, 15.0)),
        )

        forward = self.evaluate((first, second))
        reverse = self.evaluate((second, first))

        self.assertEqual(forward.to_evidence_dict(), reverse.to_evidence_dict())
        self.assertEqual(
            route_uncertainty_selection_evidence_sha256(forward),
            route_uncertainty_selection_evidence_sha256(reverse.evidence),
        )
        self.assertEqual(
            len(route_uncertainty_selection_evidence_sha256(forward)),
            64,
        )

    def test_invalid_source_route_is_rejected_by_exact_admission(self):
        malformed = option(
            "malformed",
            0,
            (Pose2D(math.nan, 1.0), Pose2D(2.0, 1.0)),
        )

        decision = self.evaluate((malformed,))

        self.assertFalse(decision.ready)
        self.assertIsNone(decision.ranked_options[0].route_length_m)
        self.assertIn(
            "map_route_pose_0_nonfinite",
            decision.ranked_options[0]
            .admission.evidence["validation"]["errors"],
        )

    def test_duplicate_identity_and_mutable_route_contracts_are_rejected(self):
        route = (Pose2D(10.0, 15.0), Pose2D(11.0, 15.0))
        duplicate_a = option("duplicate", 0, route)
        duplicate_b = option("duplicate", 1, route)

        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            self.evaluate((duplicate_a, duplicate_b))
        with self.assertRaisesRegex(TypeError, "immutable tuple"):
            RouteUncertaintySelectionOption(
                option_id="mutable",
                plan_order=0,
                map_route=list(route),
            )


if __name__ == "__main__":
    unittest.main()
