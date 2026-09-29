"""A rejected head cannot consume the decoded symbol's exact-time scan evidence."""

from collections import deque
from contextlib import ExitStack
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint, StandAxisEdgeDebugArtifacts, StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.ingestion_runtime import BoundedSensorIngress
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    registered_target_is_unique, validated_witnessed_fragmentation,
)
from tests.aufgabe04 import test_camera_observer_processing as fixtures


class _QrResolutionReached(Exception):
    """Stop after the runtime passes its resolver to independent QR binding."""


class ObserverQrScanPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        self.adapter.images = deque(maxlen=8)
        self.adapter.camera_infos = deque(maxlen=8)
        self.adapter.scans = deque(maxlen=20)
        self.adapter._sensor_ingress = BoundedSensorIngress()
        self.adapter._camera_pipeline_counters = {}
        self.lookups = []

        def lookup(target, source, stamp=None):
            self.lookups.append((target, source, stamp))
            if source == "camera":
                return self.adapter._lookup_static_transform(target, source)
            return self.adapter._lookup(target, source, stamp)

        self.adapter._lookup_scan_witness = lookup
        self.patch = patch(
            "scripts.aufgabe04.real_robot.observer.scan_witness_collection.transform_mismatches",
            return_value=())
        self.patch.start()
        self.addCleanup(self.patch.stop)

    @staticmethod
    def sample(stamp=100.):
        whole = int(stamp)
        return SimpleNamespace(
            header=SimpleNamespace(frame_id="scan", stamp=SimpleNamespace(
                sec=whole, nanosec=round((stamp - whole) * 1e9))),
            ranges=(.6,) * 5, angle_min=-.02, angle_increment=.01,
            range_min=.01, range_max=10., angle_max=.02)

    def mock_owners(self):
        owners = tuple(Mock() for _ in range(3))
        (self.adapter._scan_target_persistence, self.adapter._qr_scan_target_persistence,
         self.adapter._registration_persistence) = owners
        return owners

    def test_source_scan_and_exact_time_context_are_fanned_out_to_all_owners(self):
        owners = self.mock_owners()
        message = self.sample()
        self.adapter._on_scan(message)
        self.adapter._drain_received_sensors()
        self.adapter._collect_scan_witnesses()
        first = owners[0].ingest_scan.call_args
        for owner in owners:
            owner.ingest_scan.assert_called_once()
            call = owner.ingest_scan.call_args
            self.assertIs(call.args[0], first.args[0])
            self.assertIs(call.kwargs["context"], first.kwargs["context"])
        self.assertIs(self.lookups[0][2], message.header.stamp)
        self.assertIs(self.lookups[1][2], message.header.stamp)
        self.assertEqual(first.kwargs["context"].image_stamp_sec, 100.)

    def test_ingress_gap_resets_every_owner_and_drops_pending_older_sources(self):
        owners = self.mock_owners()
        self.adapter._pending_scan_witnesses = deque(["older-source"], maxlen=20)
        for index in range(21):
            self.adapter._sensor_ingress.offer("scans", index)
        self.adapter._drain_received_sensors()
        for owner in owners:
            owner.reset.assert_called_once()
        self.assertEqual(tuple(self.adapter._pending_scan_witnesses), tuple(range(1, 21)))

    def test_contract_and_motion_epoch_resets_include_qr_owner(self):
        for reset in ("_reset_observation_evidence", "_reset_scan_witnesses"):
            with self.subTest(reset=reset):
                owners = self.mock_owners()
                self.adapter._pending_scan_witnesses = deque(["pending-source"], maxlen=20)
                getattr(self.adapter, reset)()
                for owner in owners:
                    owner.reset.assert_called_once()
                self.assertFalse(self.adapter._pending_scan_witnesses)

    def test_expired_exact_tf_source_resets_every_owner(self):
        owners = self.mock_owners()
        self.adapter._on_scan(self.sample())
        self.adapter._drain_received_sensors()
        self.fixture.clock_sec = 100.6
        self.adapter._collect_scan_witnesses()
        for owner in owners:
            owner.reset.assert_called_once()
            owner.ingest_scan.assert_not_called()

    def test_selected_qr_resolver_keeps_witnesses_after_rejected_head_commit(self):
        adapter = self.adapter
        for stamp in (99.4, 99.6, 99.8):
            self.fixture.clock_sec = stamp + .1
            adapter._on_scan(self.sample(stamp))
            adapter._drain_received_sensors()
            adapter._collect_scan_witnesses()
        self.fixture.clock_sec = 100.1
        adapter._next_sensor_tuple.return_value.scan.value.ranges = (.6, .6, math.nan, .6, .6)
        frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
        rejected_heads = []
        accepted_qr = []

        def decode(crop, _cv2, **_kwargs):
            u, v = crop.shape[1] / 2., crop.shape[0] / 2.
            corners = tuple((u + x, v + y) for x, y in
                            ((-20, -20), (20, -20), (20, 20), (-20, 20)))
            return (DecodedQrObservation("QR_1", corners, "test_decoder", 1.),)

        def metric(_cv2, _crop, **options):
            u, v = options["expected_head_center_u_px"], options["expected_head_center_v_px"]
            corners = tuple(ImagePoint(u + x, v + y) for x, y in
                            ((-26, -26), (26, -26), (26, 26), (-26, 26)))
            return StandAxisImageEstimate(
                usable=True, reason="axis_estimated_model_current_frame_refined",
                mode="face_visible", corners=corners, axis_line=None,
                left_height_px=52., right_height_px=52., height_ratio=1.,
                yaw_proxy=0., yaw_deg=0., closer_side="equal", contour_area_px=2704.,
                source="model_current_frame_refined", evidence_state="fresh_refined",
                model_profile_sha256=adapter.stand_model_profile.sha256), StandAxisEdgeDebugArtifacts(
                    edges=None, qr_detected=True, evidence_state="fresh_refined",
                    model_profile_sha256=adapter.stand_model_profile.sha256,
                    model_pose=PlanarPoseHypothesis((0., 0., 0.), (0., 0., .6),
                                                   (0., 0., 1.), 0., .2, True))

        def association(options, bearing):
            return associate_camera_registered_candidate_lidar_target(
                options["scan"], observed_camera_bearing_rad=bearing,
                **{key: options[key] for key in (
                    "map_bearing_rad", "cone_half_angle_rad", "accepted_range_m",
                    "now_sec", "max_scan_age_sec", "min_cluster_sample_count",
                    "max_camera_map_bearing_delta_rad")})

        def reject_head(**options):
            # A different image proposal consumes its own history before QR
            # binding. Its off-target ray has no eligible historical returns.
            rejected_heads.append(options["resolve_lidar_association"](
                association(options, .12), options["scan"]))
            self.assertFalse(adapter._scan_target_persistence._pending_scans._entries)
            return None

        def resolve_qr(_observations, **options):
            self.assertTrue(rejected_heads)
            self.assertEqual(len(adapter._qr_scan_target_persistence._pending_scans._entries), 3)
            raw = association(options, 0.)
            self.assertFalse(raw.associated)
            accepted_qr.append(options["resolve_lidar_association"](raw, options["scan"]))
            raise _QrResolutionReached

        module = "scripts.aufgabe04.real_robot.observer.node."
        with ExitStack() as stack:
            for name, options in {
                "camera_info_mismatches": dict(return_value=()),
                "transform_mismatches": dict(return_value=()),
                "compressed_msg_to_bgr_frame": dict(return_value=frame),
                "_rectify_bgr_frame": dict(side_effect=lambda value, *_, **_kwargs: value),
                "detect_qr_observations_bgr": dict(side_effect=decode),
                "detect_native_qr_observations_bgr": dict(side_effect=decode),
                "estimate_stand_axis_from_metric_model": dict(side_effect=metric),
                "requires_measured_head_admission": dict(return_value=True),
                "associate_current_measured_head": dict(side_effect=reject_head),
                "bind_qr_observations_to_target": dict(side_effect=resolve_qr),
            }.items():
                stack.enter_context(patch(module + name, **options))
            with self.assertRaises(_QrResolutionReached):
                adapter._process_latest()

        self.assertFalse(rejected_heads[0].associated)
        self.assertTrue(registered_target_is_unique(accepted_qr[0]))
        self.assertTrue(validated_witnessed_fragmentation(
            accepted_qr[0].witnessed_fragmentation).associated)
        self.assertEqual(accepted_qr[0].search_association.eligible_cluster_count, 2)
        self.assertEqual(accepted_qr[0].search_association.selected_cluster_source_indices, (0, 1, 3, 4))
        self.assertEqual(adapter._qr_scan_target_persistence._last_stamp, 100.)
        self.assertFalse(adapter._qr_scan_target_persistence._pending_scans._entries)


if __name__ == "__main__":
    unittest.main()
