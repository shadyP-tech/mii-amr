import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from scripts.aufgabe04.real_robot.observer.capture_history import (
    BoundedObserverCapture, sensor_capture_metadata,
)


class ObserverCaptureHistoryTests(unittest.TestCase):
    def test_original_bytes_detached_metadata_and_disk_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BoundedObserverCapture(Path(tmp), max_frames=1, max_bytes=2048)
            raw = b"original compressed bytes\x00\xff"
            metadata = {"source_stamp": 10.25, "roi": [1, 2, 3, 4]}
            self.assertTrue(store.submit(raw, metadata=metadata, compressed_format="jpeg"))
            metadata["roi"][0] = 999
            self.assertFalse(store.submit(raw, metadata={}))
            result = store.close()
            self.assertTrue(result["flushed"])
            self.assertEqual(result["drop_reasons"], {"frame_limit": 1})
            payload = json.loads((Path(tmp) / "frame_000001.json").read_text())
            self.assertEqual((Path(tmp) / payload["image_file"]).read_bytes(), raw)
            self.assertEqual(payload["image_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(payload["metadata"]["roi"], [1, 2, 3, 4])
            self.assertFalse(payload["authoritative"])
            frame_bytes = sum(p.stat().st_size for p in Path(tmp).glob("frame_*"))
            self.assertEqual(frame_bytes, result["written_bytes"])
            self.assertLessEqual(frame_bytes, result["reserved_bytes"])
            self.assertLessEqual(result["reserved_bytes"], 2048)
            self.assertEqual(len(list(Path(tmp).iterdir())), 3)
            self.assertTrue(json.loads((Path(tmp) / "summary.json").read_text())["drain_complete"])

    def test_blocked_disk_never_blocks_submit_or_shutdown(self):
        entered, release = threading.Event(), threading.Event()
        original = BoundedObserverCapture._write_frame
        def blocked(store, *args):
            entered.set()
            release.wait(5)
            return original(store, *args)
        with tempfile.TemporaryDirectory() as tmp, patch.object(BoundedObserverCapture, "_write_frame", blocked):
            store = BoundedObserverCapture(Path(tmp), max_queue_frames=1)
            try:
                self.assertTrue(store.submit(b"image", metadata={}))
                self.assertTrue(entered.wait(2))
                self.assertFalse(store.submit(b"next", metadata={}))
                result = store.close(timeout_sec=0)
                self.assertFalse(result["flushed"])
                self.assertEqual(result["drop_reasons"], {"queue_frame_limit": 1})
                self.assertFalse(store.submit(b"next", metadata={}))
            finally:
                release.set()
                self.assertTrue(store.close()["flushed"])

    def test_invalid_and_oversized_payloads_never_enter_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BoundedObserverCapture(Path(tmp), max_bytes=1024, max_queue_bytes=700)
            self.assertFalse(store.submit(b"image", metadata={"age": float("nan")}))
            self.assertFalse(store.submit(b"x" * 1200, metadata={}))
            self.assertFalse(store.submit(b"x" * 200, metadata={}))
            result = store.close()
            self.assertEqual(result["accepted_frames"], 0)
            self.assertEqual(result["drop_reasons"], {"invalid_payload": 1, "byte_limit": 1, "queue_byte_limit": 1})

    def test_close_cannot_end_drain_while_submission_is_being_accepted(self):
        enqueue_entered, release_enqueue = threading.Event(), threading.Event()
        close_lock_attempted = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            store = BoundedObserverCapture(Path(tmp))
            original_lock, original_put = store._lock, store._queue.put_nowait

            class ObservedLock:
                def __enter__(self):
                    if threading.current_thread().name == "capture-close-test":
                        close_lock_attempted.set()
                    original_lock.acquire()

                def __exit__(self, *args):
                    original_lock.release()

            def blocked_enqueue(item):
                enqueue_entered.set()
                release_enqueue.wait(2)
                return original_put(item)

            store._lock = ObservedLock()
            accepted = []
            submitter = threading.Thread(
                target=lambda: accepted.append(store.submit(b"image", metadata={})),
            )
            closer = threading.Thread(
                target=lambda: store.close(timeout_sec=0), name="capture-close-test",
            )
            with patch.object(store._queue, "put_nowait", blocked_enqueue):
                try:
                    submitter.start()
                    self.assertTrue(enqueue_entered.wait(2))
                    closer.start()
                    self.assertTrue(close_lock_attempted.wait(2))
                    # The accept operation already passed its closed check.
                    # Closing cannot let the worker finish before its enqueue.
                    self.assertFalse(store._closing.is_set())
                finally:
                    release_enqueue.set()
                    submitter.join(timeout=2)
                    if closer.ident is not None:
                        closer.join(timeout=2)
                    result = store.close()
            self.assertEqual(accepted, [True])
            self.assertTrue(result["flushed"])
            self.assertEqual(result["written_frames"], 1)
            self.assertEqual(result["pending_frames"], 0)

    def test_write_failure_keeps_reservation_and_reports_fault(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(BoundedObserverCapture, "_write_frame", side_effect=OSError("disk full")):
            store = BoundedObserverCapture(Path(tmp), max_frames=1)
            self.assertTrue(store.submit(b"image", metadata={}))
            result = store.close()
            self.assertEqual(result["write_failures"], 1)
            self.assertEqual(result["written_frames"], 0)
            self.assertGreater(result["reserved_bytes"], 0)
            self.assertIn("disk full", result["last_error"])
            self.assertFalse(result["authoritative"])

    def test_existing_capture_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original"
            original.write_bytes(b"evidence")
            with self.assertRaisesRegex(ValueError, "empty"):
                BoundedObserverCapture(Path(tmp))
            self.assertEqual(original.read_bytes(), b"evidence")

    def test_sensor_metadata_keeps_source_headers_and_invalid_returns(self):
        header = NS(frame_id="laser", stamp=NS(sec=10, nanosec=250000000))
        camera = NS(header=header, width=640, height=480, distortion_model="plumb_bob",
                    k=[1.0] * 9, d=[0.0] * 5, r=[1.0] * 9, p=[1.0] * 12)
        scan = NS(header=header, angle_min=-3.14, angle_max=3.12, angle_increment=.028,
                  range_min=.1, range_max=10., time_increment=.001, scan_time=.2,
                  ranges=[1., float("nan"), float("inf"), -float("inf")], intensities=[])
        metadata = sensor_capture_metadata(NS(header=header), camera, scan)
        self.assertEqual(metadata["scan"]["ranges"], [1., "nan", "+inf", "-inf"])
        self.assertEqual(metadata["image_header"]["stamp_nanosec"], 250000000)
        self.assertEqual(metadata["camera_info"]["k"], camera.k)
        camera.k[0] = 7.
        self.assertEqual(metadata["camera_info"]["k"][0], 1.)
        scan.ranges = [1.] * 8193
        with self.assertRaisesRegex(ValueError, "8192"):
            sensor_capture_metadata(NS(header=header), camera, scan)


if __name__ == "__main__":
    unittest.main()
