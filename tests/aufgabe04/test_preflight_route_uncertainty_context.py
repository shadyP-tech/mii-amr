from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.preflight_route_uncertainty_context import (
    COMPOSED_CANDIDATE_POSE_BASIS,
    PREFLIGHT_ROUTE_POSE_BASIS,
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


def load_context(preflight: Path, start: Pose2D, **kwargs):
    options = {"planning_frame": "map", **kwargs}
    return load_preflight_route_uncertainty_context(
        preflight_json=preflight,
        expected_start=start,
        robot_radius_m=0.105,
        collision_margin_m=0.02,
        tracking_tube_radius_m=0.03,
        odom_drift_bound_m=0.02,
        braking_latency_distance_m=0.015,
        sigma_multiplier=2.0,
        clearance_sample_spacing_m=0.005,
        **options,
    )


COMPOSED_START = Pose2D(1.25, -0.625, 0.0)
CHAINED_START = Pose2D(1.246, -0.625, -0.001)


def candidate_preflight_payload() -> dict[str, object]:
    def capture(target, source, pose):
        return {
            "available": True,
            "target_frame": target,
            "source_frame": source,
            "observed_target_frame": target,
            "observed_source_frame": source,
            "stamp_sec": 100.0,
            "capture_time_sec": 100.01,
            "age_sec": 0.01,
            "x_m": pose.x_m,
            "y_m": pose.y_m,
            "yaw_rad": pose.yaw_rad,
        }

    direct = capture("map", "odom", Pose2D(1.0, -0.5, 0.0))
    odom = capture("odom", "base_footprint", Pose2D(0.25, -0.125, 0.0))
    return {
        **preflight_payload(CHAINED_START),
        "failures": [],
        "runtime_config": {
            "map_frame": "map",
            "odom_frame": "odom",
            "base_frame": "base_footprint",
        },
        "map_from_odom": deepcopy(direct),
        "odom_pose": {
            "frame_id": "odom",
            "child_frame_id": "base_footprint",
            "x_m": 0.25,
            "y_m": -0.125,
            "yaw_rad": 0.0,
        },
        "observations": [
            {"name": "tf map->odom", "ok": True, "data": direct},
            {"name": "tf odom->base_footprint", "ok": True, "data": odom},
            {"name": "odom freshness", "ok": True, "data": {}},
        ],
    }


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
        self.assertEqual(context.pose_basis, PREFLIGHT_ROUTE_POSE_BASIS)
        self.assertEqual(context.pose_provenance, {
            "pose_basis": PREFLIGHT_ROUTE_POSE_BASIS,
            "route_pose": preflight_payload(start)["route_pose"],
        })
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

    def test_candidate_basis_uses_original_captures_and_preserves_source_bytes(self):
        payload = candidate_preflight_payload()
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
            preflight.write_bytes(raw)

            context = load_context(
                preflight, COMPOSED_START,
                pose_basis=COMPOSED_CANDIDATE_POSE_BASIS, odom_frame="odom",
            )
            chained_context = load_context(preflight, CHAINED_START)

            self.assertEqual(preflight.read_bytes(), raw)
        self.assertEqual(context.preflight_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(context.expected_start, COMPOSED_START)
        self.assertEqual(context.pose_basis, COMPOSED_CANDIDATE_POSE_BASIS)
        self.assertEqual(context.pose_provenance, {
            "pose_basis": COMPOSED_CANDIDATE_POSE_BASIS,
            "map_from_odom_capture": payload["map_from_odom"],
            "odom_pose_capture": payload["observations"][1]["data"],
        })
        self.assertEqual(context.covariance, chained_context.covariance)
        self.assertEqual(context.covariance_evidence, chained_context.covariance_evidence)
        self.assertEqual(context.admission_config.heading_reference_x_m, COMPOSED_START.x_m)
        self.assertEqual(context.admission_config.heading_reference_y_m, COMPOSED_START.y_m)
        self.assertEqual(context.admission_config.fixed_odom_tracking_bound_m, 0.03)

    def test_basis_selection_is_explicit_and_has_no_chained_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(json.dumps(candidate_preflight_payload()))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_context(preflight, COMPOSED_START)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_context(
                    preflight, CHAINED_START,
                    pose_basis=COMPOSED_CANDIDATE_POSE_BASIS, odom_frame="odom",
                )
            for basis in ("latest_tf", "", None):
                with self.subTest(basis=basis), self.assertRaisesRegex(ValueError, "unsupported"):
                    load_context(preflight, CHAINED_START, pose_basis=basis)

    def test_candidate_basis_requires_explicit_frames_matching_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(json.dumps(candidate_preflight_payload()))
            for options in (
                {},
                {"odom_frame": ""},
                {"odom_frame": "/"},
                {"odom_frame": "another_odom"},
                {"odom_frame": "odom", "planning_frame": "another_map"},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    load_context(
                        preflight, COMPOSED_START,
                        pose_basis=COMPOSED_CANDIDATE_POSE_BASIS, **options,
                    )
            for field in ("map_frame", "odom_frame", "base_frame"):
                for value in (None, "another_frame"):
                    payload = candidate_preflight_payload()
                    payload["runtime_config"][field] = value
                    preflight.write_text(json.dumps(payload))
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        load_context(
                            preflight, COMPOSED_START,
                            pose_basis=COMPOSED_CANDIDATE_POSE_BASIS, odom_frame="odom",
                        )

    def test_candidate_basis_rejects_missing_captures_without_using_route_pose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            for field in ("runtime_config", "map_from_odom", "odom_pose", "observations"):
                payload = candidate_preflight_payload()
                payload["route_pose"] = preflight_payload(COMPOSED_START)["route_pose"]
                payload.pop(field)
                preflight.write_text(json.dumps(payload))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_context(
                        preflight, COMPOSED_START,
                        pose_basis=COMPOSED_CANDIDATE_POSE_BASIS, odom_frame="odom",
                    )

    def test_candidate_basis_rejects_failed_or_inconsistent_capture_evidence(self):
        mutations = (
            (("failures",), ["localization failed"]),
            (("map_from_odom", "x_m"), 1.01),
            (("odom_pose", "x_m"), 0.26),
            (("odom_pose", "frame_id"), "map"),
            (("odom_pose", "child_frame_id"), "camera"),
            (("observations", 0, "ok"), False),
            (("observations", 1, "ok"), False),
            (("observations", 2, "ok"), False),
            (("observations", 0, "data", "observed_target_frame"), "other_map"),
            (("observations", 1, "data", "available"), False),
            (("observations", 1, "data", "stamp_sec"), -1.0),
            (("observations", 1, "data", "capture_time_sec"), float("inf")),
            (("odom_pose", "x_m"), float("nan")),
            (("map_from_odom", "yaw_rad"), float("nan")),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            for path, value in mutations:
                payload = candidate_preflight_payload()
                target = payload
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                preflight.write_text(json.dumps(payload))
                with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                    load_context(
                        preflight, COMPOSED_START,
                        pose_basis=COMPOSED_CANDIDATE_POSE_BASIS, odom_frame="odom",
                    )

    def test_both_bases_reject_any_changed_or_nonfinite_expected_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(json.dumps(candidate_preflight_payload()))
            for basis, start in (
                (PREFLIGHT_ROUTE_POSE_BASIS, CHAINED_START),
                (COMPOSED_CANDIDATE_POSE_BASIS, COMPOSED_START),
            ):
                for index in range(3):
                    values = [start.x_m, start.y_m, start.yaw_rad]
                    for changed in (math.nextafter(values[index], math.inf), math.nan, math.inf):
                        changed_values = list(values)
                        changed_values[index] = changed
                        with self.subTest(basis=basis, index=index, changed=changed):
                            with self.assertRaisesRegex(ValueError, "does not match|non-finite"):
                                load_context(
                                    preflight, Pose2D(*changed_values),
                                    pose_basis=basis, odom_frame="odom",
                                )

    def test_both_bases_reject_symlinks_and_duplicate_source_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preflight = Path(tmpdir) / "preflight.json"
            preflight.write_text(json.dumps(candidate_preflight_payload()))
            symlink = Path(tmpdir) / "alias.json"
            symlink.symlink_to(preflight)
            for basis, start in (
                (PREFLIGHT_ROUTE_POSE_BASIS, CHAINED_START),
                (COMPOSED_CANDIDATE_POSE_BASIS, COMPOSED_START),
            ):
                with self.subTest(basis=basis), self.assertRaisesRegex(ValueError, "symlink"):
                    load_context(symlink, start, pose_basis=basis, odom_frame="odom")
            raw = json.dumps(candidate_preflight_payload()).replace(
                '"x_m": 1.0', '"x_m": 1.0, "x_m": 1.0', 1,
            )
            preflight.write_text(raw)
            for basis, start in (
                (PREFLIGHT_ROUTE_POSE_BASIS, CHAINED_START),
                (COMPOSED_CANDIDATE_POSE_BASIS, COMPOSED_START),
            ):
                with self.subTest(basis=basis), self.assertRaisesRegex(ValueError, "malformed"):
                    load_context(preflight, start, pose_basis=basis, odom_frame="odom")


if __name__ == "__main__":
    unittest.main()
