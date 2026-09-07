from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.preflight_route_uncertainty_context import (
    load_preflight_route_uncertainty_context,
)


def preflight_payload(start: Pose2D) -> dict[str, object]:
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


def load_context(preflight: Path, start: Pose2D):
    return load_preflight_route_uncertainty_context(
        preflight_json=preflight,
        expected_start=start,
        planning_frame="map",
        robot_radius_m=0.105,
        collision_margin_m=0.02,
        tracking_tube_radius_m=0.03,
        odom_drift_bound_m=0.02,
        braking_latency_distance_m=0.015,
        sigma_multiplier=2.0,
        clearance_sample_spacing_m=0.005,
    )


class PreflightRouteUncertaintyContextTest(unittest.TestCase):
    def test_binds_content_pose_covariance_and_admission_config(self):
        start = Pose2D(-0.75, 0.20, 0.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            raw = (
                json.dumps(preflight_payload(start), sort_keys=True) + "\n"
            ).encode("utf-8")
            preflight.write_bytes(raw)

            context = load_context(preflight, start)

        self.assertEqual(context.preflight_json, preflight)
        self.assertEqual(
            context.preflight_sha256,
            hashlib.sha256(raw).hexdigest(),
        )
        self.assertEqual(context.expected_start, start)
        self.assertEqual(context.planning_frame, "map")
        self.assertAlmostEqual(context.covariance.xx_m2, 0.009)
        self.assertAlmostEqual(context.covariance.xy_m2, 0.0)
        self.assertAlmostEqual(context.covariance.yy_m2, 0.009)
        self.assertEqual(context.covariance_evidence["sample_count"], 3)
        self.assertAlmostEqual(
            context.admission_config.heading_sigma_rad,
            0.1,
        )
        self.assertAlmostEqual(
            context.admission_config.heading_lever_arm_m,
            0.105,
        )
        self.assertAlmostEqual(
            context.admission_config.heading_reference_x_m,
            start.x_m,
        )
        self.assertAlmostEqual(
            context.admission_config.heading_reference_y_m,
            start.y_m,
        )

    def test_rejects_duplicate_json_keys_before_context_is_created(self):
        start = Pose2D(-0.75, 0.20, 0.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(
                '{"ok": true, "ok": true, "route_pose": {}}\n'
            )

            with self.assertRaisesRegex(ValueError, "malformed"):
                load_context(preflight, start)

    def test_rejects_a_preflight_bound_to_another_start(self):
        expected_start = Pose2D(-0.75, 0.20, 0.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(
                json.dumps(
                    preflight_payload(Pose2D(-0.70, 0.20, 0.4)),
                    sort_keys=True,
                )
                + "\n"
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_context(preflight, expected_start)


if __name__ == "__main__":
    unittest.main()
