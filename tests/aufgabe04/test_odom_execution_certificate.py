import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    MAP_FROM_ODOM_CONVENTION,
    OdomExecutionCertificate,
    PlanarTransform2D,
    load_odom_execution_certificate,
    map_pose_to_odom,
    normalize_yaw,
    odom_execution_certificate_sha256,
    odom_pose_to_map,
    pose_route_sha256,
    transform_map_route_to_odom,
    transform_odom_route_to_map,
    validate_odom_execution_identity,
    write_odom_execution_certificate,
)


MAP_ROUTE = (
    Pose2D(1.0, 2.0, 0.2),
    Pose2D(1.5, 2.25, -2.8),
    Pose2D(2.0, 2.5, 3.4),
)


def certificate_for(
    map_route=MAP_ROUTE,
    transform=PlanarTransform2D(0.4, -0.7, 0.3),
):
    odom_route = transform_map_route_to_odom(map_route, transform)
    certificate = OdomExecutionCertificate(
        source_map_route_sha256=pose_route_sha256(map_route),
        source_map_execution_certificate_sha256="b" * 64,
        transformed_odom_route_sha256=pose_route_sha256(odom_route),
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        map_from_odom=transform,
        transform_stamp_sec=100.0,
        transform_capture_time_sec=100.05,
        waypoint_count=len(map_route),
        tracking_tube_radius_m=0.03,
        command_owner="/tb3/aufgabe04_simple_waypoint_follower",
        uncertainty_budget_sha256="c" * 64,
        ambiguity_evidence_sha256="d" * 64,
    )
    return certificate, odom_route


class PlanarTransform2DTest(unittest.TestCase):
    def assertPoseAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual.x_m, expected.x_m, places=12)
        self.assertAlmostEqual(actual.y_m, expected.y_m, places=12)
        self.assertAlmostEqual(
            normalize_yaw(actual.yaw_rad - expected.yaw_rad),
            0.0,
            places=12,
        )

    def test_identity(self):
        transform = PlanarTransform2D(0.0, 0.0, 0.0)
        pose = Pose2D(1.25, -3.5, math.pi / 3.0)

        self.assertPoseAlmostEqual(odom_pose_to_map(pose, transform), pose)
        self.assertPoseAlmostEqual(map_pose_to_odom(pose, transform), pose)
        self.assertIn("p_map = R", MAP_FROM_ODOM_CONVENTION)

    def test_translation_uses_inverse_for_map_to_odom(self):
        transform = PlanarTransform2D(10.0, -2.0, 0.0)
        pose_map = Pose2D(11.0, 1.0, 0.4)

        pose_odom = map_pose_to_odom(pose_map, transform)

        self.assertPoseAlmostEqual(pose_odom, Pose2D(1.0, 3.0, 0.4))
        self.assertNotAlmostEqual(pose_odom.x_m, 21.0)
        self.assertPoseAlmostEqual(odom_pose_to_map(pose_odom, transform), pose_map)

    def test_quarter_turn_rotation_and_wrong_sign_guard(self):
        transform = PlanarTransform2D(0.0, 0.0, math.pi / 2.0)

        pose_map = odom_pose_to_map(Pose2D(1.0, 0.0, 0.0), transform)
        pose_odom = map_pose_to_odom(Pose2D(0.0, 1.0, math.pi / 2.0), transform)

        self.assertPoseAlmostEqual(pose_map, Pose2D(0.0, 1.0, math.pi / 2.0))
        self.assertPoseAlmostEqual(pose_odom, Pose2D(1.0, 0.0, 0.0))
        self.assertNotAlmostEqual(pose_odom.x_m, -1.0)

    def test_pose_and_route_round_trip(self):
        transform = PlanarTransform2D(-0.8, 1.7, -1.1)
        odom_route = transform_map_route_to_odom(MAP_ROUTE, transform)
        reconstructed = transform_odom_route_to_map(odom_route, transform)

        for actual, expected in zip(reconstructed, MAP_ROUTE):
            self.assertPoseAlmostEqual(actual, expected)
        inverse = transform.inverse()
        self.assertPoseAlmostEqual(
            odom_pose_to_map(MAP_ROUTE[0], inverse),
            map_pose_to_odom(MAP_ROUTE[0], transform),
        )

    def test_yaw_is_normalized_and_hash_is_container_independent(self):
        self.assertAlmostEqual(normalize_yaw(3.0 * math.pi), -math.pi)
        route_with_wrapped_yaw = list(MAP_ROUTE)
        route_with_wrapped_yaw[0] = replace(
            route_with_wrapped_yaw[0],
            yaw_rad=route_with_wrapped_yaw[0].yaw_rad + math.tau,
        )
        self.assertEqual(
            pose_route_sha256(list(MAP_ROUTE)),
            pose_route_sha256(tuple(route_with_wrapped_yaw)),
        )

    def test_unconstrained_nan_yaw_survives_transform_round_trip_and_hashes(self):
        map_route = (
            Pose2D(0.0, 0.0, 0.1),
            Pose2D(0.4, 0.2, math.nan),
            Pose2D(0.8, 0.5, -0.2),
        )
        transform = PlanarTransform2D(0.5, -0.25, math.pi / 3.0)

        odom_route = transform_map_route_to_odom(map_route, transform)
        reconstructed = transform_odom_route_to_map(odom_route, transform)

        self.assertTrue(math.isnan(odom_route[1].yaw_rad))
        self.assertTrue(math.isnan(reconstructed[1].yaw_rad))
        self.assertAlmostEqual(reconstructed[1].x_m, map_route[1].x_m)
        self.assertAlmostEqual(reconstructed[1].y_m, map_route[1].y_m)
        equivalent = list(map_route)
        equivalent[1] = replace(
            equivalent[1],
            yaw_rad=math.copysign(math.nan, -1.0),
        )
        self.assertEqual(
            pose_route_sha256(map_route),
            pose_route_sha256(equivalent),
        )
        constrained = list(map_route)
        constrained[1] = replace(constrained[1], yaw_rad=0.0)
        self.assertNotEqual(
            pose_route_sha256(map_route),
            pose_route_sha256(constrained),
        )

        certificate, certified_odom_route = certificate_for(map_route, transform)
        validate_odom_execution_identity(
            certificate,
            source_map_route=map_route,
            transformed_odom_route=certified_odom_route,
            source_map_execution_certificate_sha256="b" * 64,
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            tracking_tube_radius_m=0.03,
            command_owner="/tb3/aufgabe04_simple_waypoint_follower",
        )

    def test_infinite_pose_yaw_remains_invalid(self):
        transform = PlanarTransform2D(0.0, 0.0, 0.0)
        for yaw_rad in (math.inf, -math.inf):
            route = (Pose2D(0.0, 0.0), Pose2D(1.0, 0.0, yaw_rad))
            with self.subTest(yaw_rad=yaw_rad):
                with self.assertRaisesRegex(ValueError, "finite or NaN"):
                    transform_map_route_to_odom(route, transform)
                with self.assertRaisesRegex(ValueError, "finite or NaN"):
                    pose_route_sha256(route)

    def test_rejects_nonfinite_or_malformed_inputs(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    PlanarTransform2D(value, 0.0, 0.0)
                with self.assertRaisesRegex(ValueError, "finite"):
                    normalize_yaw(value)
        with self.assertRaisesRegex(ValueError, "Pose2D"):
            map_pose_to_odom((0.0, 0.0, 0.0), PlanarTransform2D(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "at least two"):
            transform_map_route_to_odom(
                (Pose2D(0.0, 0.0),),
                PlanarTransform2D(0, 0, 0),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            pose_route_sha256(
                (Pose2D(0.0, 0.0), Pose2D(math.nan, 1.0))
            )


class OdomExecutionCertificateTest(unittest.TestCase):
    def test_certificate_is_frozen_and_validates_all_runtime_identity(self):
        certificate, odom_route = certificate_for()

        with self.assertRaises(FrozenInstanceError):
            certificate.map_frame = "other"
        validate_odom_execution_identity(
            certificate,
            source_map_route=MAP_ROUTE,
            transformed_odom_route=odom_route,
            source_map_execution_certificate_sha256="b" * 64,
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            tracking_tube_radius_m=0.03,
            command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            waypoint_count=3,
            map_from_odom=certificate.map_from_odom,
            transform_stamp_sec=100.0,
            transform_capture_time_sec=100.05,
            uncertainty_budget_sha256="c" * 64,
            ambiguity_evidence_sha256="d" * 64,
        )

    def test_future_dated_transform_stamp_is_bound_without_order_rejection(self):
        certificate, odom_route = certificate_for()
        future_dated = replace(
            certificate,
            transform_stamp_sec=101.1,
            transform_capture_time_sec=100.0,
        )

        validate_odom_execution_identity(
            future_dated,
            source_map_route=MAP_ROUTE,
            transformed_odom_route=odom_route,
            source_map_execution_certificate_sha256="b" * 64,
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            tracking_tube_radius_m=0.03,
            command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            transform_stamp_sec=101.1,
            transform_capture_time_sec=100.0,
        )
        with self.assertRaisesRegex(ValueError, "transform_stamp_sec mismatch"):
            validate_odom_execution_identity(
                future_dated,
                source_map_route=MAP_ROUTE,
                transformed_odom_route=odom_route,
                source_map_execution_certificate_sha256="b" * 64,
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
                tracking_tube_radius_m=0.03,
                command_owner="/tb3/aufgabe04_simple_waypoint_follower",
                transform_stamp_sec=101.0,
                transform_capture_time_sec=100.0,
            )

    def test_persistence_is_content_hashed_idempotent_and_immutable(self):
        certificate, _ = certificate_for()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "odom_certificate.json"

            digest = write_odom_execution_certificate(path, certificate)
            retry_digest = write_odom_execution_certificate(path, certificate)

            self.assertEqual(digest, odom_execution_certificate_sha256(certificate))
            self.assertEqual(retry_digest, digest)
            self.assertEqual(load_odom_execution_certificate(path), certificate)
            with self.assertRaisesRegex(ValueError, "refusing to replace immutable"):
                write_odom_execution_certificate(
                    path,
                    replace(certificate, tracking_tube_radius_m=0.02),
                )

    def test_hash_tampering_is_rejected(self):
        certificate, _ = certificate_for()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "odom_certificate.json"
            write_odom_execution_certificate(path, certificate)
            payload = json.loads(path.read_text())
            payload["waypoint_count"] = 99
            path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_odom_execution_certificate(path)

    def test_malformed_transform_with_valid_content_hash_is_rejected(self):
        certificate, _ = certificate_for()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "odom_certificate.json"
            write_odom_execution_certificate(path, certificate)
            stored = json.loads(path.read_text())
            stored.pop("odom_execution_certificate_sha256")
            stored["map_from_odom"]["yaw_rad"] = 2.0 * math.pi
            stored["odom_execution_certificate_sha256"] = payload_sha256(stored)
            path.write_text(json.dumps(stored))

            with self.assertRaisesRegex(ValueError, "yaw_rad must be normalized"):
                load_odom_execution_certificate(path)

    def test_frame_mismatch_is_rejected(self):
        certificate, odom_route = certificate_for()
        with self.assertRaisesRegex(ValueError, "map_frame mismatch"):
            validate_odom_execution_identity(
                certificate,
                source_map_route=MAP_ROUTE,
                transformed_odom_route=odom_route,
                source_map_execution_certificate_sha256="b" * 64,
                map_frame="world",
                odom_frame="odom",
                base_frame="base_footprint",
                tracking_tube_radius_m=0.03,
                command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            )

    def test_source_map_execution_certificate_mismatch_is_rejected(self):
        certificate, odom_route = certificate_for()
        with self.assertRaisesRegex(
            ValueError,
            "source_map_execution_certificate_sha256 mismatch",
        ):
            validate_odom_execution_identity(
                certificate,
                source_map_route=MAP_ROUTE,
                transformed_odom_route=odom_route,
                source_map_execution_certificate_sha256="e" * 64,
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
                tracking_tube_radius_m=0.03,
                command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            )

    def test_source_and_transformed_route_hash_mismatches_are_rejected(self):
        certificate, odom_route = certificate_for()
        changed_map_route = list(MAP_ROUTE)
        changed_map_route[1] = replace(changed_map_route[1], x_m=1.51)
        with self.assertRaisesRegex(ValueError, "source_map_route_sha256 mismatch"):
            validate_odom_execution_identity(
                certificate,
                source_map_route=changed_map_route,
                transformed_odom_route=odom_route,
                source_map_execution_certificate_sha256="b" * 64,
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
                tracking_tube_radius_m=0.03,
                command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            )
        changed_odom_route = list(odom_route)
        changed_odom_route[1] = replace(changed_odom_route[1], y_m=0.01)
        with self.assertRaisesRegex(
            ValueError, "transformed_odom_route_sha256 mismatch"
        ):
            validate_odom_execution_identity(
                certificate,
                source_map_route=MAP_ROUTE,
                transformed_odom_route=changed_odom_route,
                source_map_execution_certificate_sha256="b" * 64,
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
                tracking_tube_radius_m=0.03,
                command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            )

    def test_wrong_sign_route_cannot_be_certified_as_correct_geometry(self):
        transform = PlanarTransform2D(0.5, -0.25, math.pi / 2.0)
        wrong = tuple(odom_pose_to_map(pose, transform) for pose in MAP_ROUTE)
        certificate, _ = certificate_for(MAP_ROUTE, transform)
        forged = replace(
            certificate,
            transformed_odom_route_sha256=pose_route_sha256(wrong),
        )

        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_odom_execution_identity(
                forged,
                source_map_route=MAP_ROUTE,
                transformed_odom_route=wrong,
                source_map_execution_certificate_sha256="b" * 64,
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
                tracking_tube_radius_m=0.03,
                command_owner="/tb3/aufgabe04_simple_waypoint_follower",
            )

    def test_constructor_rejects_invalid_fields(self):
        certificate, _ = certificate_for()
        invalid_cases = (
            ({"source_map_route_sha256": "A" * 64}, "lowercase SHA-256"),
            (
                {"source_map_execution_certificate_sha256": "bad"},
                "lowercase SHA-256",
            ),
            ({"map_frame": ""}, "frame id"),
            ({"odom_frame": "map"}, "must be distinct"),
            ({"waypoint_count": 1}, "integer >= 2"),
            ({"tracking_tube_radius_m": 0.0}, "finite and positive"),
            ({"command_owner": "relative"}, "absolute node identity"),
            ({"transform_stamp_sec": math.inf}, "finite"),
            ({"transform_stamp_sec": -1.0}, "non-negative"),
            ({"transform_capture_time_sec": math.nan}, "finite"),
            ({"transform_capture_time_sec": -1.0}, "non-negative"),
            ({"uncertainty_budget_sha256": "bad"}, "lowercase SHA-256"),
        )
        for changes, message in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    replace(certificate, **changes)


if __name__ == "__main__":
    unittest.main()
