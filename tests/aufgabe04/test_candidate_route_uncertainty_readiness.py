from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.execution.route_uncertainty_defaults import (
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
    DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
    DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.station_segment.cli import build_parser
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import (
    CandidateRouteUncertaintyReadinessRequest,
    load_candidate_route_uncertainty_readiness,
)


def _preflight_payload(start: Pose2D) -> dict[str, object]:
    covariance = [0.0] * 36
    covariance[0] = 0.004
    covariance[7] = 0.009
    covariance[35] = 0.01
    return {
        "ok": True,
        "route_pose": {
            "frame_id": "map",
            "child_frame_id": "base_footprint",
            "x_m": start.x_m,
            "y_m": start.y_m,
            "yaw_rad": start.yaw_rad,
        },
        "stationary_amcl_samples": [
            {"covariance": list(covariance)} for _ in range(3)
        ],
    }


class CandidateRouteUncertaintyReadinessTest(unittest.TestCase):
    def test_uses_exact_station_child_budget_defaults(self):
        start = Pose2D(-0.75, 0.20, 0.40)
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(
                json.dumps(_preflight_payload(start), sort_keys=True) + "\n"
            )
            context = load_candidate_route_uncertainty_readiness(
                CandidateRouteUncertaintyReadinessRequest(
                    preflight_json=preflight,
                    expected_start=start,
                    planning_frame="map",
                    robot_radius_m=0.105,
                    sigma_multiplier=2.0,
                )
            )

        child_defaults = build_parser().parse_args(["--leg-index", "0"])
        config = context.admission_config
        self.assertEqual(config.robot_radius_m, 0.105)
        self.assertEqual(config.localization_sigma_multiplier, 2.0)
        self.assertEqual(
            config.collision_margin_m,
            DEFAULT_COLLISION_MARGIN_M,
        )
        self.assertEqual(
            config.collision_margin_m,
            child_defaults.uncertainty_collision_margin_m,
        )
        self.assertEqual(
            config.fixed_odom_tracking_bound_m,
            DEFAULT_TRACKING_TUBE_RADIUS_M,
        )
        self.assertEqual(
            config.fixed_odom_tracking_bound_m,
            child_defaults.certified_route_tube_radius_m,
        )
        self.assertEqual(
            config.empirical_odom_drift_bound_m,
            DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
        )
        self.assertEqual(
            config.empirical_odom_drift_bound_m,
            child_defaults.uncertainty_odom_drift_bound_m,
        )
        self.assertEqual(
            config.braking_latency_distance_m,
            DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
        )
        self.assertEqual(
            config.braking_latency_distance_m,
            child_defaults.uncertainty_braking_latency_distance_m,
        )
        self.assertEqual(
            config.sampling_spacing_m,
            DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
        )
        self.assertEqual(
            config.sampling_spacing_m,
            child_defaults.uncertainty_clearance_sample_spacing_m,
        )
        self.assertEqual(config.heading_reference_x_m, start.x_m)
        self.assertEqual(config.heading_reference_y_m, start.y_m)
        self.assertEqual(config.heading_lever_arm_m, 0.105)
        self.assertAlmostEqual(config.heading_sigma_rad, 0.10)

    def test_source_evidence_is_content_pose_and_frame_bound(self):
        start = Pose2D(-0.75, 0.20, 0.40)
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            raw = (
                json.dumps(_preflight_payload(start), sort_keys=True) + "\n"
            ).encode("utf-8")
            preflight.write_bytes(raw)
            context = load_candidate_route_uncertainty_readiness(
                CandidateRouteUncertaintyReadinessRequest(
                    preflight_json=preflight,
                    expected_start=start,
                    planning_frame="map",
                    robot_radius_m=0.105,
                    sigma_multiplier=2.0,
                )
            )

            source = context.source_evidence
            self.assertEqual(
                source["source_preplanning_localization_json"],
                str(preflight),
            )
            self.assertEqual(
                source["source_preplanning_localization_sha256"],
                hashlib.sha256(raw).hexdigest(),
            )
            self.assertEqual(source["planning_frame"], "map")
            self.assertEqual(
                source["admitted_start_pose"],
                {"x_m": -0.75, "y_m": 0.20, "yaw_rad": 0.40},
            )
            self.assertEqual(source["covariance_envelope"]["sample_count"], 3)
            self.assertTrue(source["child_budget_defaults_shared"])
            self.assertTrue(source["selection_only"])
            self.assertFalse(source["motion_authorized"])

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_candidate_route_uncertainty_readiness(
                    CandidateRouteUncertaintyReadinessRequest(
                        preflight_json=preflight,
                        expected_start=Pose2D(-0.70, 0.20, 0.40),
                        planning_frame="map",
                        robot_radius_m=0.105,
                        sigma_multiplier=2.0,
                    )
                )
            with self.assertRaisesRegex(ValueError, "frame mismatch"):
                load_candidate_route_uncertainty_readiness(
                    CandidateRouteUncertaintyReadinessRequest(
                        preflight_json=preflight,
                        expected_start=start,
                        planning_frame="odom",
                        robot_radius_m=0.105,
                        sigma_multiplier=2.0,
                    )
                )


if __name__ == "__main__":
    unittest.main()
