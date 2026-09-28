"""Stored-Start return admission using a self-contained recorded regression."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import (
    evaluate_admitted_return_stage_uncertainty,
    select_admitted_return_prefix,
)
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
    evaluate_stationary_turn_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)


FIXTURE = Path(__file__).with_name("fixtures") / "start_return_uncertainty_20260923.json"
TARGET_SHA256 = "a" * 64


def _recorded_case():
    fixture = json.loads(FIXTURE.read_text())
    source_map = fixture["map"]
    cells = tuple(
        value
        for value, count in source_map["cells_rle"]
        for _ in range(count)
    )
    width, height = source_map["width"], source_map["height"]
    if len(cells) != width * height:
        raise AssertionError("recorded fixture occupancy has the wrong dimensions")
    metadata = MapMetadata(
        yaml_path=Path("recorded-fixture.yaml"),
        image_path=Path("recorded-fixture.pgm"),
        resolution=source_map["resolution"],
        origin=tuple(source_map["origin"]),
        negate=source_map["negate"],
        occupied_thresh=source_map["occupied_thresh"],
        free_thresh=source_map["free_thresh"],
        mode=source_map["mode"],
    )
    grid = OccupancyGrid(
        metadata=metadata,
        width=width,
        height=height,
        cells=tuple(cells[y * width:(y + 1) * width] for y in range(height)),
    )
    costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(
        ArenaBounds(**source_map["arena"])
    )
    poses = tuple(Pose2D(**pose) for pose in fixture["full_poses"])
    uncertainty = CandidateRouteUncertaintyContext(
        covariance=PlanarCovariance(**fixture["covariance_m2"]),
        admission_config=RouteUncertaintyAdmissionConfig(**fixture["admission_config"]),
        source_evidence={
            "recorded_budget_file_sha256": fixture["provenance"]["budget_file_sha256"],
            "selection_only": True,
            "motion_authorized": False,
        },
    )
    return fixture, costmap, poses, uncertainty


def _open_corridor() -> Costmap:
    metadata = MapMetadata(
        yaml_path=Path("synthetic.yaml"),
        image_path=Path("synthetic.pgm"),
        resolution=0.05,
        origin=(-3.0, -1.0, 0.0),
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.20,
        mode="trinary",
    )
    grid = OccupancyGrid(
        metadata=metadata,
        width=120,
        height=40,
        cells=tuple((CELL_FREE,) * 120 for _ in range(40)),
    )
    return Costmap.from_occupancy_grid(grid).with_arena_bounds(
        ArenaBounds(length_m=5.0, width_m=2.0)
    )


def _context(start: Pose2D, *, heading_sigma_rad=0.20):
    return CandidateRouteUncertaintyContext(
        covariance=PlanarCovariance(0.0, 0.0, 0.0),
        admission_config=RouteUncertaintyAdmissionConfig(
            robot_radius_m=0.105,
            collision_margin_m=0.020,
            fixed_odom_tracking_bound_m=0.030,
            empirical_odom_drift_bound_m=0.020,
            braking_latency_distance_m=0.075,
            localization_sigma_multiplier=2.0,
            heading_sigma_rad=heading_sigma_rad,
            heading_lever_arm_m=0.105,
            sampling_spacing_m=0.005,
            heading_reference_x_m=start.x_m,
            heading_reference_y_m=start.y_m,
        ),
        source_evidence={"selection_only": True, "motion_authorized": False},
    )


def _select(costmap, poses, uncertainty, *, target_sha256=TARGET_SHA256):
    return select_admitted_return_prefix(
        full_poses=poses,
        base_costmap=costmap,
        uncertainty=uncertainty,
        target_evidence_sha256=target_sha256,
    )


class RecordedReturnUncertaintyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.costmap, cls.poses, cls.uncertainty = _recorded_case()

    def test_recorded_complete_return_reproduces_geometry_and_rejection(self):
        admission = evaluate_route_uncertainty_admission(
            self.costmap, self.poses, self.uncertainty.covariance,
            self.uncertainty.admission_config,
        )
        expected = self.fixture["expected"]
        self.assertEqual(
            admission.evidence["costmap"], self.fixture["map"]["expected_costmap"]
        )
        self.assertFalse(admission.decision.accepted)
        self.assertAlmostEqual(
            admission.decision.remaining_margin_m,
            expected["whole_minimum_margin_m"], places=12,
        )
        self.assertEqual(
            admission.decision.limiting_segment_id,
            expected["whole_limiting_segment_id"],
        )
        self.assertEqual(len(admission.segments), expected["whole_budget_entry_count"])

    def test_recorded_return_selects_admitted_first_vertex_then_fresh_suffix(self):
        original_config = self.uncertainty.admission_config
        stage = _select(
            self.costmap, self.poses, self.uncertainty,
            target_sha256=self.fixture["provenance"]["target_evidence_sha256"],
        )
        self.assertFalse(stage.is_final_stage)
        self.assertEqual(stage.end_segment_index, 0)
        self.assertEqual(stage.end_fraction, 1.0)
        self.assertEqual(len(stage.poses), 2)
        self.assertEqual(stage.poses[0], self.poses[0])
        self.assertEqual(stage.stage_target_pose.x_m, self.poses[1].x_m)
        self.assertEqual(stage.stage_target_pose.y_m, self.poses[1].y_m)
        self.assertEqual(stage.poses[-1], stage.stage_target_pose)
        incoming_yaw = math.atan2(
            self.poses[1].y_m - self.poses[0].y_m,
            self.poses[1].x_m - self.poses[0].x_m,
        )
        outgoing_yaw = math.atan2(
            self.poses[2].y_m - self.poses[1].y_m,
            self.poses[2].x_m - self.poses[1].x_m,
        )
        # Stop along the admitted incoming leg. Turning toward the next leg
        # belongs to its new stopped preflight and covariance envelope.
        self.assertAlmostEqual(stage.stage_target_pose.yaw_rad, incoming_yaw)
        self.assertNotAlmostEqual(stage.stage_target_pose.yaw_rad, outgoing_yaw)
        self.assertIs(stage.evidence["motion_authorized"], False)
        self.assertEqual(self.uncertainty.admission_config, original_config)
        first_admission = evaluate_route_uncertainty_admission(
            self.costmap, stage.poses, self.uncertainty.covariance, original_config,
        )
        self.assertTrue(first_admission.decision.accepted)
        self.assertAlmostEqual(
            first_admission.decision.remaining_margin_m,
            self.fixture["expected"]["prefix_minimum_margin_m"], places=12,
        )

        # This models a separate stopped preflight. It deliberately keeps the
        # recorded covariance; an actual future run must supply fresh evidence.
        fresh_context = replace(
            self.uncertainty,
            admission_config=replace(
                original_config,
                heading_reference_x_m=stage.stage_target_pose.x_m,
                heading_reference_y_m=stage.stage_target_pose.y_m,
            ),
            source_evidence={"fresh_stopped_stage": True, "motion_authorized": False},
        )
        suffix = (stage.stage_target_pose, *self.poses[2:])
        final_stage = _select(self.costmap, suffix, fresh_context)
        self.assertTrue(final_stage.is_final_stage)
        self.assertEqual(final_stage.poses[-1], self.poses[-1])
        self.assertEqual(final_stage.stage_target_pose, self.poses[-1])
        self.assertIs(final_stage.evidence["motion_authorized"], False)
        final_admission = evaluate_route_uncertainty_admission(
            self.costmap, final_stage.poses, fresh_context.covariance,
            fresh_context.admission_config,
        )
        self.assertTrue(final_admission.decision.accepted)
        self.assertAlmostEqual(
            final_admission.decision.remaining_margin_m,
            self.fixture["expected"]["fresh_anchor_suffix_minimum_margin_m"],
            places=12,
        )


class ReturnUncertaintySelectionTest(unittest.TestCase):
    def setUp(self):
        self.costmap = _open_corridor()

    def test_admitted_whole_route_preserves_exact_target_and_yaw(self):
        poses = (Pose2D(-0.5, 0.0, 0.4), Pose2D(0.5, 0.0, -0.71))
        stage = _select(self.costmap, poses, _context(poses[0], heading_sigma_rad=0.0))
        self.assertTrue(stage.is_final_stage)
        self.assertEqual(stage.stage_target_pose, poses[-1])
        self.assertEqual(stage.poses[-1], poses[-1])
        self.assertEqual(stage.poses[0], poses[0])
        self.assertEqual(stage.end_fraction, 1.0)
        self.assertIs(stage.evidence["motion_authorized"], False)

    def test_long_straight_return_splits_inside_segment_with_same_reserves(self):
        poses = (Pose2D(-2.0, 0.0, 0.2), Pose2D(2.0, 0.0, -0.71))
        uncertainty = _context(poses[0])
        rejected = evaluate_route_uncertainty_admission(
            self.costmap, poses, uncertainty.covariance, uncertainty.admission_config,
        )
        self.assertFalse(rejected.decision.accepted)
        stage = _select(self.costmap, poses, uncertainty)
        self.assertFalse(stage.is_final_stage)
        self.assertEqual(stage.end_segment_index, 0)
        self.assertGreater(stage.end_fraction, 0.0)
        self.assertLess(stage.end_fraction, 1.0)
        self.assertEqual(stage.stage_target_pose.y_m, 0.0)
        self.assertAlmostEqual(
            stage.stage_target_pose.x_m,
            poses[0].x_m + stage.end_fraction * (poses[-1].x_m - poses[0].x_m),
        )
        self.assertGreater(stage.stage_target_pose.x_m - poses[0].x_m, 0.10)
        self.assertAlmostEqual(stage.stage_target_pose.yaw_rad, 0.0)
        admission = evaluate_route_uncertainty_admission(
            self.costmap, stage.poses, uncertainty.covariance,
            uncertainty.admission_config,
        )
        self.assertTrue(admission.decision.accepted)
        self.assertGreater(admission.decision.remaining_margin_m, 0.0)
        self.assertEqual(uncertainty.admission_config.braking_latency_distance_m, 0.075)

    def test_unsafe_initial_clearance_cannot_be_hidden_by_shorter_prefix(self):
        poses = (Pose2D(0.0, 0.80, 0.0), Pose2D(1.0, 0.80, 0.0))
        with self.assertRaises(ValueError):
            _select(self.costmap, poses, _context(poses[0]))

    def test_stale_heading_anchor_is_not_reset_by_selection(self):
        poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.5, 0.0, 0.0))
        context = _context(Pose2D(-1.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            _select(self.costmap, poses, context)

    def test_return_selection_rejects_reduced_braking_reserve(self):
        poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.5, 0.0, 0.0))
        context = _context(poses[0])
        context = replace(
            context,
            admission_config=replace(context.admission_config, braking_latency_distance_m=0.020),
        )
        with self.assertRaises(ValueError):
            _select(self.costmap, poses, context)

    def test_stationary_return_admits_heading_without_translation(self):
        poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, math.pi / 2.0))
        context = _context(poses[0])
        stage = _select(self.costmap, poses, context)
        self.assertTrue(stage.is_final_stage)
        self.assertEqual(stage.poses, poses)
        self.assertEqual(stage.stage_target_pose, poses[-1])
        self.assertIs(stage.evidence["motion_authorized"], False)
        admission = evaluate_stationary_turn_uncertainty_admission(
            self.costmap, stage.poses, context.covariance, context.admission_config,
            start_pose=poses[0], target_evidence_sha256=TARGET_SHA256,
        )
        self.assertTrue(admission.decision.accepted)
        self.assertIs(admission.evidence["sampling"]["translation_permitted"], False)
        self.assertFalse(evaluate_route_uncertainty_admission(
            self.costmap, poses, context.covariance, context.admission_config,
        ).decision.accepted)

    def test_stationary_return_rejects_largest_covariance_axis(self):
        poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, math.pi / 2.0))
        context = replace(_context(poses[0]), covariance=PlanarCovariance(0.0, 0.0, 0.36))
        with self.assertRaises(ValueError):
            _select(self.costmap, poses, context)

    def test_stage_boundary_retains_worst_axis_for_turns(self):
        covariance = PlanarCovariance(.02, .0199, .02)
        for poses, limiting in (
            ((Pose2D(-.5, .4, 0.), Pose2D(.2, .4, 0.)), "initial_orientation"),
            ((Pose2D(0., 0., math.pi/2), Pose2D(0., .4, math.pi/2)), "stopped_endpoint_orientation"),
        ):
            with self.subTest(limiting=limiting):
                context = _context(poses[0], heading_sigma_rad=0.)
                ordinary = evaluate_route_uncertainty_admission(
                    self.costmap, poses, covariance, context.admission_config,
                )
                self.assertTrue(ordinary.decision.accepted)
                staged = evaluate_admitted_return_stage_uncertainty(
                    self.costmap, poses, covariance, context.admission_config,
                    start_pose=poses[0], target_evidence_sha256=TARGET_SHA256,
                    is_final_stage=False,
                )
                self.assertFalse(staged.decision.accepted)
                self.assertEqual(staged.decision.limiting_segment_id, limiting)


if __name__ == "__main__":
    unittest.main()
