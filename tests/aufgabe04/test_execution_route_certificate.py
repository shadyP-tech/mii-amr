import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCertificate,
    check_execution_route_tube,
    execution_route_certificate_sha256,
    load_execution_route_certificate,
    validate_execution_route_identity,
    write_execution_route_certificate,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


class ExecutionRouteCertificateTest(unittest.TestCase):
    def test_rejects_shortcuts_and_pose_deviation(self):
        route = (
            Pose2D(0.0, 0.0),
            Pose2D(0.10, 0.0),
            Pose2D(0.10, 0.10),
        )

        valid = check_execution_route_tube(
            Pose2D(0.05, 0.01),
            route,
            target_index=1,
            pursuit_index=1,
            tracking_tube_radius_m=0.03,
        )
        shortcut = check_execution_route_tube(
            Pose2D(0.05, 0.01),
            route,
            target_index=1,
            pursuit_index=2,
            tracking_tube_radius_m=0.03,
        )
        outside = check_execution_route_tube(
            Pose2D(0.05, 0.04),
            route,
            target_index=1,
            pursuit_index=1,
            tracking_tube_radius_m=0.03,
        )

        self.assertTrue(valid.ok)
        self.assertEqual(shortcut.reason, "uncertified pursuit shortcut")
        self.assertEqual(outside.reason, "pose left certified route tube")

    def test_certificate_binds_route_bytes_and_runtime_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.csv"
            route_path.write_text("x,y\n0,0\n1,0\n")
            digest = hashlib.sha256(route_path.read_bytes()).hexdigest()
            certificate = ExecutionRouteCertificate(
                route_sha256=digest,
                planning_frame="odom",
                route_kind="catalog_face_approach",
                waypoint_count=2,
                tracking_tube_radius_m=0.03,
                exact_vertex_pursuit=True,
                command_owner="/aufgabe04_simple_waypoint_follower",
            )

            validate_execution_route_identity(
                certificate,
                route_path=route_path,
                planning_frame="odom",
                route_kind="catalog_face_approach",
                waypoint_count=2,
                command_owner="/aufgabe04_simple_waypoint_follower",
            )
            route_path.write_text(route_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "route_sha256 mismatch"):
                validate_execution_route_identity(
                    certificate,
                    route_path=route_path,
                    planning_frame="odom",
                    route_kind="catalog_face_approach",
                    waypoint_count=2,
                    command_owner="/aufgabe04_simple_waypoint_follower",
                )

    def test_certificate_persistence_is_content_hashed_and_immutable(self):
        certificate = ExecutionRouteCertificate(
            route_sha256="a" * 64,
            planning_frame="odom",
            route_kind="catalog_face_approach",
            waypoint_count=3,
            tracking_tube_radius_m=0.03,
            exact_vertex_pursuit=True,
            command_owner="/aufgabe04_simple_waypoint_follower",
            map_bundle_sha256="b" * 64,
            candidate_snapshot_sha256="c" * 64,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "certificate.json"
            digest = write_execution_route_certificate(path, certificate)
            loaded = load_execution_route_certificate(path)

            self.assertEqual(digest, execution_route_certificate_sha256(certificate))
            self.assertEqual(loaded, certificate)
            with self.assertRaisesRegex(ValueError, "refusing to replace immutable"):
                write_execution_route_certificate(
                    path,
                    ExecutionRouteCertificate(
                        **{
                            **certificate.__dict__,
                            "tracking_tube_radius_m": 0.02,
                        }
                    ),
                )


if __name__ == "__main__":
    unittest.main()
