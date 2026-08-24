import dataclasses
import json
import math
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    CELL_OCCUPIED,
    MapMetadata,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
    UNCERTAINTY_BUDGET_EXHAUSTED,
)


def costmap_from_rows(rows, *, resolution=1.0):
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
        width=len(rows[0]),
        height=len(rows),
        cells=tuple(tuple(row) for row in rows),
    )
    return Costmap.from_occupancy_grid(grid)


def open_costmap(*, width=20, height=20, resolution=1.0):
    return costmap_from_rows(
        [[CELL_FREE] * width for _ in range(height)],
        resolution=resolution,
    )


def config(**overrides):
    values = {
        "robot_radius_m": 0.10,
        "collision_margin_m": 0.05,
        "fixed_odom_tracking_bound_m": 0.05,
        "empirical_odom_drift_bound_m": 0.02,
        "braking_latency_distance_m": 0.03,
        "localization_sigma_multiplier": 2.0,
        "heading_sigma_rad": 0.01,
        "heading_lever_arm_m": 0.10,
        "sampling_spacing_m": 1.0,
    }
    values.update(overrides)
    return RouteUncertaintyAdmissionConfig(**values)


class RouteUncertaintyAdmissionTest(unittest.TestCase):
    def test_config_is_frozen_validated_and_binds_heading_contribution(self):
        admission_config = config()

        self.assertAlmostEqual(admission_config.heading_contribution_m, 0.002)
        self.assertAlmostEqual(
            admission_config.to_evidence_dict()["heading_contribution_m"],
            0.002,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            admission_config.robot_radius_m = 0.2

        for broken in (
            {"robot_radius_m": 0.0},
            {"collision_margin_m": -0.01},
            {"localization_sigma_multiplier": 0.0},
            {"heading_sigma_rad": math.nan},
            {"sampling_spacing_m": math.inf},
        ):
            with self.subTest(broken=broken):
                with self.assertRaises(ValueError):
                    config(**broken)

    def test_samples_endpoints_at_bounded_spacing_and_applies_lipschitz_loss(self):
        result = evaluate_route_uncertainty_admission(
            open_costmap(),
            (Pose2D(5.0, 10.0), Pose2D(7.5, 10.0)),
            PlanarCovariance(0.0001, 0.0, 0.0001),
            config(sampling_spacing_m=1.0),
        )

        sampled = result.evidence["sampling"]["segments"][0]
        self.assertEqual(sampled["interval_count"], 3)
        self.assertEqual(sampled["sample_count"], 4)
        self.assertLessEqual(sampled["actual_spacing_m"], 1.0)
        self.assertEqual(sampled["samples"][0]["x_m"], 5.0)
        self.assertEqual(sampled["samples"][-1]["x_m"], 7.5)
        self.assertAlmostEqual(
            sampled["lipschitz_deduction_m"],
            sampled["actual_spacing_m"] / 2.0,
        )
        self.assertAlmostEqual(
            sampled["clearance_lower_bound_m"],
            sampled["minimum_sampled_clearance_m"]
            - sampled["actual_spacing_m"] / 2.0,
        )
        self.assertTrue(result.decision.accepted)

    def test_profile_uses_raw_obstacle_clearance_and_budget_terms_once(self):
        rows = [[CELL_FREE] * 10 for _ in range(10)]
        rows[5][5] = CELL_OCCUPIED
        result = evaluate_route_uncertainty_admission(
            costmap_from_rows(rows),
            (Pose2D(1.0, 4.0), Pose2D(9.0, 4.0)),
            PlanarCovariance(0.0001, 0.0, 0.0001),
            config(),
        )

        segment = result.segments[0]
        sampled = result.evidence["sampling"]["segments"][0]
        # The nearest sampled point is 1 m from the occupied cell square.
        # point_clearance_to_blocked_m contributes its strict 1e-6 interior
        # epsilon, followed by this module's 0.5 m sample-gap deduction.
        self.assertAlmostEqual(sampled["minimum_sampled_clearance_m"], 0.999999)
        self.assertAlmostEqual(segment.raw_centerline_clearance_m, 0.499999)
        decision = result.decision.segment_decisions[0]
        required = decision.evidence["budget_m"]["required_clearance_m"]
        # radius + margin + tracking + drift + braking + 2*position sigma
        # + 2*heading sigma*lever arm.
        self.assertAlmostEqual(required, 0.272)
        self.assertAlmostEqual(decision.remaining_margin_m, 0.227999)
        self.assertTrue(result.decision.accepted)

    def test_interior_vertex_gets_worst_axis_corner_entry(self):
        covariance = PlanarCovariance(0.04, 0.0, 0.01)
        result = evaluate_route_uncertainty_admission(
            open_costmap(),
            (
                Pose2D(5.0, 5.0),
                Pose2D(10.0, 5.0),
                Pose2D(10.0, 10.0),
            ),
            covariance,
            config(
                robot_radius_m=0.01,
                collision_margin_m=0.0,
                fixed_odom_tracking_bound_m=0.0,
                empirical_odom_drift_bound_m=0.0,
                braking_latency_distance_m=0.0,
                heading_sigma_rad=0.0,
            ),
        )

        self.assertEqual(
            [segment.segment_id for segment in result.segments],
            ["segment:0000", "corner:0001", "segment:0001"],
        )
        corner_segment = result.segments[1]
        corner_decision = result.decision.segment_decisions[1]
        self.assertTrue(corner_segment.is_corner)
        self.assertEqual(
            corner_decision.evidence["geometry"]["projection_mode"],
            "corner_worst_axis",
        )
        self.assertAlmostEqual(
            corner_decision.evidence["localization"]["sigma_m"], 0.2
        )
        self.assertEqual(
            corner_segment.raw_centerline_clearance_m,
            min(
                result.segments[0].raw_centerline_clearance_m,
                result.segments[2].raw_centerline_clearance_m,
            ),
        )

    def test_heading_uncertainty_uses_segment_local_reference_lever(self):
        result = evaluate_route_uncertainty_admission(
            open_costmap(width=40, height=40),
            (
                Pose2D(5.0, 5.0),
                Pose2D(5.1, 5.0),
                Pose2D(15.0, 5.0),
            ),
            PlanarCovariance(0.0, 0.0, 0.0),
            config(
                robot_radius_m=0.1,
                heading_sigma_rad=0.1,
                heading_lever_arm_m=0.1,
                heading_reference_x_m=5.0,
                heading_reference_y_m=5.0,
            ),
        )

        first = result.segments[0]
        second = result.segments[2]
        self.assertAlmostEqual(first.heading_contribution_m, 0.04)
        self.assertAlmostEqual(second.heading_contribution_m, 2.02)
        self.assertLess(
            first.heading_contribution_m,
            second.heading_contribution_m,
        )

    def test_unconstrained_nan_yaw_is_admitted_and_canonicalized(self):
        result = evaluate_route_uncertainty_admission(
            open_costmap(),
            (
                Pose2D(5.0, 5.0, 0.0),
                Pose2D(8.0, 5.0, math.nan),
                Pose2D(8.0, 8.0, 1.0),
            ),
            PlanarCovariance(0.0001, 0.0, 0.0001),
            config(),
        )

        self.assertTrue(result.evidence["validation"]["ok"])
        self.assertTrue(result.decision.accepted)
        intermediate = result.evidence["route"]["poses"][1]
        self.assertIsNone(intermediate["yaw_rad"])
        self.assertEqual(intermediate["yaw_mode"], "unconstrained_nan")
        json.dumps(result.evidence, sort_keys=True, allow_nan=False)

    def test_short_nonfinite_and_zero_length_routes_fail_closed(self):
        routes = (
            (),
            (Pose2D(1.0, 1.0),),
            (Pose2D(1.0, 1.0), Pose2D(math.nan, 2.0)),
            (Pose2D(1.0, 1.0), Pose2D(2.0, 2.0, math.inf)),
            (Pose2D(1.0, 1.0), Pose2D(1.0, 1.0, 1.0)),
        )
        for route in routes:
            with self.subTest(route=route):
                result = evaluate_route_uncertainty_admission(
                    open_costmap(),
                    route,
                    PlanarCovariance(0.0, 0.0, 0.0),
                    config(),
                )
                self.assertFalse(result.decision.accepted)
                self.assertEqual(result.decision.reason, UNCERTAINTY_BUDGET_EXHAUSTED)
                self.assertEqual(result.segments, ())
                self.assertFalse(result.evidence["validation"]["ok"])
                json.dumps(result.evidence, sort_keys=True, allow_nan=False)

        zero_length = evaluate_route_uncertainty_admission(
            open_costmap(),
            (Pose2D(1.0, 1.0), Pose2D(1.0, 1.0)),
            PlanarCovariance(0.0, 0.0, 0.0),
            config(),
        )
        self.assertIn(
            "map_route_segment_0_zero_length_ambiguous",
            zero_length.evidence["validation"]["errors"],
        )

    def test_exhausted_clearance_rejects(self):
        result = evaluate_route_uncertainty_admission(
            open_costmap(width=4, height=4),
            (Pose2D(0.2, 2.0), Pose2D(3.8, 2.0)),
            PlanarCovariance(0.0, 0.0, 0.0),
            config(sampling_spacing_m=0.2),
        )

        self.assertFalse(result.decision.accepted)
        self.assertEqual(result.decision.reason, UNCERTAINTY_BUDGET_EXHAUSTED)
        self.assertLessEqual(result.decision.remaining_margin_m, 0.0)

    def test_complete_evidence_and_hash_are_deterministic_and_map_bound(self):
        arguments = (
            open_costmap(),
            (Pose2D(5.0, 5.0), Pose2D(8.0, 5.0)),
            PlanarCovariance(0.0001, 0.0, 0.0001),
            config(),
        )
        first = evaluate_route_uncertainty_admission(*arguments)
        second = evaluate_route_uncertainty_admission(*arguments)

        encoded = json.dumps(
            first.evidence,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertIn('"probability_guarantee":false', encoded)
        self.assertIn('"distance_property":"1_lipschitz"', encoded)
        self.assertEqual(
            route_uncertainty_admission_evidence_sha256(first),
            route_uncertainty_admission_evidence_sha256(second.evidence),
        )

        changed_rows = [[CELL_FREE] * 20 for _ in range(20)]
        changed_rows[0][0] = CELL_OCCUPIED
        changed = evaluate_route_uncertainty_admission(
            costmap_from_rows(changed_rows), *arguments[1:]
        )
        self.assertNotEqual(
            first.evidence["costmap"]["blocked_geometry_sha256"],
            changed.evidence["costmap"]["blocked_geometry_sha256"],
        )
        self.assertNotEqual(
            route_uncertainty_admission_evidence_sha256(first),
            route_uncertainty_admission_evidence_sha256(changed),
        )


if __name__ == "__main__":
    unittest.main()
