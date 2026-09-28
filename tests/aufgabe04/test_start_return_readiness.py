"""The return preview must charge exactly the faster child admission reserves."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.control.return_to_start_speed_policy import return_to_start_speed_policy_arguments
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.station_segment.cli import build_parser
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import CandidateRouteUncertaintyReadinessRequest
from scripts.aufgabe04.real_robot.mission.start_return_readiness import load_start_return_readiness
from tests.aufgabe04.test_candidate_route_uncertainty_readiness import _preflight_payload


class StartReturnReadinessTest(unittest.TestCase):
    def test_preview_uses_fast_child_budget_and_conservative_covariance(self):
        start = Pose2D(1., .2, .3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stopped.json"
            path.write_text(json.dumps(_preflight_payload(start)))
            context = load_start_return_readiness(CandidateRouteUncertaintyReadinessRequest(
                path, start, "map", "odom", .105, 2.,
            ))
        args = build_parser().parse_args(["--leg-index", "0", *return_to_start_speed_policy_arguments()])
        config = context.admission_config
        for field, arg in (
            ("braking_latency_distance_m", "uncertainty_braking_latency_distance_m"),
            ("collision_margin_m", "uncertainty_collision_margin_m"),
            ("fixed_odom_tracking_bound_m", "certified_route_tube_radius_m"),
            ("empirical_odom_drift_bound_m", "uncertainty_odom_drift_bound_m"),
            ("sampling_spacing_m", "uncertainty_clearance_sample_spacing_m"),
        ):
            self.assertEqual(getattr(config, field), getattr(args, arg))
        self.assertEqual(config.braking_latency_distance_m, .075)
        self.assertEqual(config.heading_reference_x_m, 1.)
        self.assertEqual(config.heading_reference_y_m, .2)
        self.assertEqual(config.heading_sigma_rad, .1)
        self.assertEqual(context.source_evidence["motion_speed_policy"], "unloaded_return_to_start")
        self.assertFalse(context.source_evidence["motion_authorized"])

    def test_new_stage_cannot_reuse_preflight_from_another_start(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stopped.json"
            path.write_text(json.dumps(_preflight_payload(Pose2D(1., .2, .3))))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_start_return_readiness(CandidateRouteUncertaintyReadinessRequest(
                    path, Pose2D(.5, .2, .3), "map", "odom", .105, 2.,
                ))
