"""Source-age admission/publication regressions with deterministic delayed work."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.observer.camera_publication import camera_source_freshness
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode


class CameraPublicationFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = 100.1
        self.adapter = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        self.adapter.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self.clock * 1e9))))
        self.adapter.args = SimpleNamespace(max_sensor_age_sec=.5, max_future_timestamp_sec=.05)
        self.adapter.completed = False
        self.adapter.axis_observation_committed = False

    def commit(self, kind="recommendation", *, image=100., scan=100., payload=None):
        path = Path(self.directory.name) / f"{kind}.json"
        accepted = self.adapter._commit_sensor_artifact(
            path, {"created_at": self.clock} if payload is None else payload,
            image_stamp_sec=image, scan_stamp_sec=scan, artifact_kind=kind,
        )
        return accepted, path

    def test_image_and_scan_ages_are_required_not_artifact_creation_age(self):
        self.clock = 100.6
        for kind in ("recommendation", "backside_axis_observation", "inspection_observation"):
            for image, scan in ((100., 100.5), (100.5, 100.)):
                with self.subTest(kind=kind, image=image):
                    accepted, path = self.commit(kind, image=image, scan=scan)
                    self.assertFalse(accepted)
                    self.assertFalse(path.exists())
                    self.assertFalse(self.adapter._last_camera_publication_freshness["accepted"])
        self.assertFalse(self.adapter.completed)
        self.assertFalse(self.adapter.axis_observation_committed)

    def test_delayed_real_debug_encoding_cannot_publish_a_late_artifact(self):
        def imwrite(path, image):
            self.clock += .08
            Path(path).write_bytes(b"diagnostic")
            return True
        self.adapter.cv2 = SimpleNamespace(imwrite=imwrite)
        self.adapter.args.debug_dir = Path(self.directory.name) / "debug"
        debug = SimpleNamespace(**{key: object() for key in (
            "edges", "raw_edges", "face_mask", "rectangle_mask", "rectangle_overlay")})
        self.assertTrue(self.adapter._source_freshness(100., 100.).accepted)
        self.adapter._write_debug(object(), object(), debug, metadata={})
        for kind in ("recommendation", "backside_axis_observation", "inspection_observation"):
            accepted, path = self.commit(kind)
            self.assertFalse(accepted)
            self.assertFalse(path.exists())

    def test_slow_serialization_fsync_is_rechecked_before_atomic_publication(self):
        for kind in ("recommendation", "backside_axis_observation", "inspection_observation"):
            with self.subTest(kind=kind):
                self.clock = 100.1
                with patch("scripts.aufgabe04.real_robot.observer.node.os.fsync",
                           side_effect=lambda _: setattr(self, "clock", 100.6)):
                    accepted, path = self.commit(kind)
                self.assertFalse(accepted)
                self.assertFalse(path.exists())
                self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_late_write_leaves_existing_artifact_unchanged(self):
        accepted, path = self.commit(payload={"original": True})
        self.assertTrue(accepted)
        self.clock = 100.6
        accepted, _ = self.commit(payload={"original": False})
        self.assertFalse(accepted)
        self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_exact_age_boundary_passes_without_rewriting_source_time(self):
        self.clock = 100.5
        payload = {"sensor_stamp_sec": 100., "observation_unix_sec": 100.}
        accepted, path = self.commit(payload=payload)
        self.assertTrue(accepted)
        self.assertEqual(json.loads(path.read_text()), payload)

    def test_invalid_or_future_sources_cannot_publish(self):
        for stamp in (None, float("nan"), float("inf"), True, "100", 100.2):
            with self.subTest(stamp=stamp):
                self.assertFalse(self.commit(image=stamp)[0])
                self.assertFalse(self.commit(scan=stamp)[0])
        with self.assertRaises(ValueError):
            camera_source_freshness(image_stamp_sec=100, scan_stamp_sec=100,
                                    now_sec=100, max_age_sec=0, max_future_sec=.05)

    def test_delayed_association_cannot_seed_later_axis_or_qr_consensus(self):
        adapter = self.adapter
        adapter.args = SimpleNamespace(
            max_sensor_age_sec=.5, max_future_timestamp_sec=.05,
            stream_id="run", stand_id="stand", stand_x=1., stand_y=0., expected_qr_id="auto",
            stationary_translation_m=.02, stationary_rotation_deg=2.,
            consensus_frames=7, consensus_max_deviation_deg=8.,
            consensus_axis_ttl_sec=5., consensus_qr_ttl_sec=5., sync_tolerance_sec=.1,
        )
        adapter.observation_evidence = None
        self.clock = 100.6  # Detector/association began at the supplied old time.
        update = adapter._record_observation_frame(
            robot_pose=Pose2D(0., 0., 0.), image_stamp_sec=100., scan_stamp_sec=100.,
            observed_at_sec=100.1, lidar_associated=True, axis_yaw_rad=0.,
            axis_source="model_current_frame_refined", qr_texts=("QR_1",),
        )
        self.assertFalse(update.frame_accepted)
        self.assertFalse(update.axis_sample_accepted)
        self.assertFalse(update.qr_sample_accepted)
        self.assertEqual(update.snapshot.current_axis_sample_count, 0)
        self.assertEqual(update.snapshot.current_qr_sample_count, 0)
        self.assertFalse(adapter._last_evidence_source_freshness["accepted"])


if __name__ == "__main__":
    unittest.main()
