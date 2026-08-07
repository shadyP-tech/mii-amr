from collections import deque
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


from scripts.aufgabe04.navigation.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    odom_execution_certificate_sha256,
    write_odom_execution_certificate,
)
from scripts.aufgabe04.perception.stand_observation import PlanarTransform
from scripts.aufgabe04.perception import stand_explorer_node


def _certificate(
    *,
    map_frame: str = "map",
    odom_frame: str = "odom",
    base_frame: str = "base_footprint",
) -> OdomExecutionCertificate:
    return OdomExecutionCertificate(
        source_map_route_sha256="a" * 64,
        source_map_execution_certificate_sha256="b" * 64,
        transformed_odom_route_sha256="c" * 64,
        map_frame=map_frame,
        odom_frame=odom_frame,
        base_frame=base_frame,
        map_from_odom=PlanarTransform2D(1.0, 2.0, math.pi / 2.0),
        transform_stamp_sec=10.0,
        transform_capture_time_sec=10.1,
        waypoint_count=2,
        tracking_tube_radius_m=0.03,
        command_owner="/aufgabe04_simple_waypoint_follower",
        uncertainty_budget_sha256="d" * 64,
        ambiguity_evidence_sha256="e" * 64,
    )


class StandExplorerFrozenFrameTest(unittest.TestCase):
    def test_nonzero_yaw_composition_maps_odom_scan_pose_into_map(self):
        composed = stand_explorer_node.compose_frozen_scan_pose_in_map(
            odom_from_scan=PlanarTransform(3.0, 4.0, -math.pi / 4.0),
            map_from_odom=PlanarTransform2D(1.0, 2.0, math.pi / 2.0),
        )

        self.assertAlmostEqual(composed.x_m, -3.0)
        self.assertAlmostEqual(composed.y_m, 5.0)
        self.assertAlmostEqual(composed.yaw_rad, math.pi / 4.0)

    def test_certificate_loader_binds_hash_source_path_and_frames(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "odom_execution_certificate.json"
            certificate = _certificate()
            write_odom_execution_certificate(path, certificate)

            frozen = stand_explorer_node.load_frozen_observer_frame(
                path,
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
            )

        evidence = frozen.runtime_evidence()
        self.assertEqual(
            frozen.certificate_sha256,
            odom_execution_certificate_sha256(certificate),
        )
        self.assertEqual(frozen.certificate_path, path.resolve())
        self.assertEqual(
            evidence["odom_execution_certificate_sha256"],
            frozen.certificate_sha256,
        )
        self.assertEqual(
            evidence["source_frames"],
            {
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_footprint",
            },
        )
        self.assertEqual(evidence["scan_tf_target_frame"], "odom")

    def test_certificate_loader_fails_closed_for_frame_mismatch(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "odom_execution_certificate.json"
            write_odom_execution_certificate(path, _certificate())

            with self.assertRaisesRegex(ValueError, "odom_frame mismatch"):
                stand_explorer_node.load_frozen_observer_frame(
                    path,
                    map_frame="map",
                    odom_frame="wheel_odom",
                    base_frame="base_footprint",
                )

    def test_certificate_loader_fails_closed_for_missing_or_malformed_file(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing = root / "missing.json"
            malformed = root / "malformed.json"
            malformed.write_text("not-json\n")

            for path in (missing, malformed):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(
                        ValueError,
                        "invalid odom execution certificate",
                    ):
                        stand_explorer_node.load_frozen_observer_frame(
                            path,
                            map_frame="map",
                            odom_frame="odom",
                            base_frame="base_footprint",
                        )

    def test_certificate_mode_queries_only_exact_time_odom_from_scan(self):
        calls = []
        processed = []
        marker_transform = object()

        class FakeDuration:
            def __init__(self, *, seconds):
                self.seconds = seconds

        class FakeBuffer:
            def can_transform(self, target, source, query_time, *, timeout):
                calls.append(("can", target, source, query_time, timeout.seconds))
                return True

            def lookup_transform(self, target, source, query_time, *, timeout):
                calls.append(
                    ("lookup", target, source, query_time, timeout.seconds)
                )
                return marker_transform

        query_time = object()
        pending = stand_explorer_node._PendingScan(
            message=object(),
            scan_frame="base_scan",
            scan_stamp_sec=10.0,
            query_time=query_time,
            deadline_monotonic_sec=stand_explorer_node.time.monotonic() + 10.0,
        )
        certificate = _certificate()
        frozen = stand_explorer_node.FrozenObserverFrame(
            certificate_path=Path("certificate.json"),
            certificate=certificate,
            certificate_sha256=odom_execution_certificate_sha256(certificate),
        )
        fake_node = SimpleNamespace(
            pending_scans=deque((pending,)),
            tf_buffer=FakeBuffer(),
            runtime=SimpleNamespace(map_frame="map", odom_frame="odom"),
            frozen_observer_frame=frozen,
            get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
            _process_scan_with_transform=lambda item, transform: processed.append(
                (item, transform)
            ),
        )
        original_duration = stand_explorer_node.Duration
        try:
            stand_explorer_node.Duration = FakeDuration
            stand_explorer_node.StandExplorerNode._drain_pending_scans(fake_node)
        finally:
            stand_explorer_node.Duration = original_duration

        self.assertEqual(
            calls,
            [
                ("can", "odom", "base_scan", query_time, 0.0),
                ("lookup", "odom", "base_scan", query_time, 0.0),
            ],
        )
        self.assertNotIn("map", [call[1] for call in calls])
        self.assertEqual(processed, [(pending, marker_transform)])

    def test_frozen_mode_rejects_tf_frame_label_mismatch(self):
        transform = SimpleNamespace(
            header=SimpleNamespace(frame_id="map"),
            child_frame_id="base_scan",
        )

        with self.assertRaisesRegex(ValueError, "parent frame mismatch"):
            stand_explorer_node._validated_frozen_tf_frames(
                transform,
                expected_parent_frame="odom",
                expected_child_frame="base_scan",
            )

    def test_frozen_mode_rejects_malformed_tf_quaternion(self):
        transform = SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=1.0, y=2.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            )
        )

        with self.assertRaisesRegex(ValueError, "not normalized"):
            stand_explorer_node._validated_planar_pose_from_tf(transform)

    def test_frozen_certificate_evidence_is_persisted_in_observer_summary(self):
        certificate = _certificate()
        frozen = stand_explorer_node.FrozenObserverFrame(
            certificate_path=Path("certificate.json"),
            certificate=certificate,
            certificate_sha256=odom_execution_certificate_sha256(certificate),
        )
        node = SimpleNamespace(
            frozen_observer_frame=frozen,
            started_unix_sec=1.0,
            output_jsonl=Path("observations.jsonl"),
            map_bundle=None,
            runtime=SimpleNamespace(
                map_frame="map",
                as_log_dict=lambda: {"map_frame": "map"},
            ),
            last_scan_pose_map=None,
            last_processed_scan_stamp_sec=None,
            processed_scan_count=0,
            detected_candidate_count=0,
            accepted_observation_count=0,
            last_confirmed_stand_count=0,
            timing_limits=SimpleNamespace(as_dict=lambda: {}),
        )

        payload = stand_explorer_node.observer_summary_payload(node)
        evidence = payload[stand_explorer_node.FROZEN_FRAME_EVIDENCE_KEY]

        self.assertEqual(
            evidence["odom_execution_certificate_sha256"],
            frozen.certificate_sha256,
        )
        self.assertEqual(evidence["source_frames"]["odom_frame"], "odom")
        self.assertEqual(
            payload["observer_version"],
            stand_explorer_node.FROZEN_ODOM_OBSERVER_VERSION,
        )

    def test_legacy_mode_keeps_map_target_and_existing_observer_version(self):
        args = stand_explorer_node.build_parser().parse_args([])
        runtime = SimpleNamespace(map_frame="map", odom_frame="odom")

        self.assertIsNone(args.odom_execution_certificate_json)
        self.assertEqual(
            stand_explorer_node._observation_tf_target_frame(runtime, None),
            "map",
        )
        self.assertEqual(
            stand_explorer_node._observer_version(None),
            stand_explorer_node.OBSERVER_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
