from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None

from scripts.aufgabe04.perception.debug.stand_axis_recording import DebugWindowRecorder


FRAME_WINDOW = "aufgabe04/stand-axis"
EDGES_WINDOW = "aufgabe04/stand-axis-edges"


class FakeWriter:
    def __init__(self, opened=True):
        self.frames = []
        self.opened = opened
        self.released = False
        self.fail_write = False

    def isOpened(self):
        return self.opened and not self.released

    def write(self, frame):
        if self.fail_write:
            return False
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeCv2:
    if cv2 is not None:
        COLOR_GRAY2BGR = cv2.COLOR_GRAY2BGR
        INTER_NEAREST = cv2.INTER_NEAREST
        cvtColor = staticmethod(cv2.cvtColor)
        resize = staticmethod(cv2.resize)
        imencode = staticmethod(cv2.imencode)
        VideoWriter_fourcc = staticmethod(cv2.VideoWriter_fourcc)

    def __init__(self):
        self.writers = []
        self.calls = []
        self.fail_open_index = None

    def VideoWriter(self, path, codec, fps, size):
        writer = FakeWriter(opened=len(self.writers) != self.fail_open_index)
        self.writers.append(writer)
        self.calls.append((Path(path).name, fps, size))
        return writer


@unittest.skipIf(cv2 is None or np is None, "OpenCV/numpy unavailable")
class TestStandAxisRecording(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "recordings"
        self.fake_cv2 = FakeCv2()
        self.recorder = DebugWindowRecorder(self.fake_cv2, self.directory, 15.0)
        self.addCleanup(self.recorder.stop)
        self.original = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        self.annotated = np.full((10, 12, 3), 255, dtype=np.uint8)
        self.images = {
            FRAME_WINDOW: self.annotated,
            EDGES_WINDOW: np.zeros((5, 6), dtype=np.uint8),
        }

    def sessions(self):
        return sorted(self.directory.glob("recording_*"))

    @staticmethod
    def records(session):
        return [json.loads(line) for line in (session / "metadata.jsonl").read_text().splitlines()]

    def test_preserves_unannotated_pixels_and_metadata_for_rejected_estimate(self):
        self.recorder.start(
            self.images,
            source_frame=self.original,
            metadata={"estimate": {"usable": False, "reason": "corner_evidence_insufficient"}},
        )
        self.recorder.write(
            self.images,
            source_frame=self.original + np.uint8(1),
            metadata={"estimate": {"usable": True, "yaw_deg": 7.1}},
        )
        self.recorder.stop()
        session = self.sessions()[0]
        records = self.records(session)
        self.assertEqual([record["frame_index"] for record in records], [0, 1])
        self.assertEqual(records[0]["window_frames"], {"annotated.avi": 0, "edges.avi": 0})
        self.assertEqual(records[1]["window_frames"], {"annotated.avi": 1, "edges.avi": 1})
        self.assertFalse(records[0]["estimate"]["usable"])
        self.assertEqual(records[0]["source_frame"], "source_000000.png")
        self.assertEqual(records[1]["recording_status"], "complete")
        np.testing.assert_array_equal(cv2.imread(str(session / records[0]["source_frame"])), self.original)
        np.testing.assert_array_equal(cv2.imread(str(session / records[1]["source_frame"])), self.original + np.uint8(1))
        np.testing.assert_array_equal(self.fake_cv2.writers[0].frames[0], self.annotated)
        self.assertTrue(all(writer.released for writer in self.fake_cv2.writers))

    def test_missing_window_has_explicit_independent_video_index(self):
        self.recorder.start(self.images, source_frame=self.original)
        self.recorder.write({FRAME_WINDOW: self.annotated}, source_frame=self.original)
        self.recorder.write(self.images, source_frame=self.original)
        records = self.records(self.sessions()[0])
        self.assertEqual(records[1]["window_frames"], {"annotated.avi": 1, "edges.avi": None})
        self.assertEqual(records[2]["window_frames"], {"annotated.avi": 2, "edges.avi": 1})
        self.assertEqual([len(writer.frames) for writer in self.fake_cv2.writers], [3, 2])

    def test_inactive_or_no_matching_window_does_not_write_evidence(self):
        self.recorder.write(self.images, source_frame=self.original, metadata={"bad": object()})
        self.assertFalse(self.directory.exists())
        self.recorder.start(self.images)
        self.recorder.write({"unknown": self.original}, source_frame=self.original)
        session = self.sessions()[0]
        self.assertEqual(len(self.records(session)), 1)
        self.assertEqual(list(session.glob("source_*.png")), [])
        self.recorder.stop()
        self.recorder.write(self.images, source_frame=self.original)
        self.assertEqual(len(self.records(session)), 1)

    def test_legacy_call_resizes_diagnostic_pixels_and_keeps_filenames(self):
        self.recorder.start(self.images)
        self.recorder.write({FRAME_WINDOW: np.zeros((3, 4, 3), dtype=np.uint8)})
        self.assertEqual(self.fake_cv2.calls, [("annotated.avi", 15.0, (12, 10)), ("edges.avi", 15.0, (6, 5))])
        self.assertEqual(self.fake_cv2.writers[0].frames[1].shape, (10, 12, 3))
        self.assertEqual(self.fake_cv2.writers[1].frames[0].shape, (5, 6, 3))
        self.assertIsNone(self.records(self.sessions()[0])[0]["source_frame"])

    def test_stop_restart_resets_indices_and_closes_metadata_file(self):
        with patch("scripts.aufgabe04.perception.debug.stand_axis_recording.time.time_ns", side_effect=[1, 2]):
            self.recorder.start(self.images, source_frame=self.original)
            first_stream = self.recorder._metadata_stream
            self.recorder.stop()
            self.assertTrue(first_stream.closed)
            self.recorder.stop()
            self.recorder.start(self.images, source_frame=self.original)
        self.assertEqual(len(self.sessions()), 2)
        self.assertEqual([self.records(session)[0]["frame_index"] for session in self.sessions()], [0, 0])
        self.assertTrue(all(writer.released for writer in self.fake_cv2.writers[:2]))

    def test_failed_png_encoding_records_no_video_or_success_metadata(self):
        with patch.object(self.fake_cv2, "imencode", return_value=(False, np.array([], dtype=np.uint8))):
            with self.assertRaisesRegex(RuntimeError, "encode original"):
                self.recorder.start(self.images, source_frame=self.original)
        self.assertFalse(self.recorder.active)
        self.assertEqual(self.records(self.sessions()[0]), [])
        self.assertEqual(list(self.sessions()[0].glob("source*")), [])
        self.assertTrue(all(writer.released and not writer.frames for writer in self.fake_cv2.writers))

    def test_invalid_metadata_is_rejected_before_video_writes(self):
        for metadata in ({"raw_image": self.original}, {"residual": float("nan")}):
            with self.subTest(metadata=list(metadata)):
                with self.assertRaises((RuntimeError, ValueError)):
                    self.recorder.start(self.images, metadata=metadata, source_frame=self.original)
                self.assertFalse(self.recorder.active)
        self.assertTrue(all(writer.released and not writer.frames for writer in self.fake_cv2.writers))
        self.assertTrue(all(not self.records(session) for session in self.sessions()))
        self.assertTrue(all(not list(session.glob("source*")) for session in self.sessions()))

    def test_writer_open_failure_releases_every_handle_and_can_restart(self):
        self.fake_cv2.fail_open_index = 1
        with self.assertRaisesRegex(RuntimeError, "could not open video"):
            self.recorder.start(self.images)
        self.assertFalse(self.recorder.active)
        self.assertTrue(all(writer.released for writer in self.fake_cv2.writers))
        self.fake_cv2.fail_open_index = None
        self.recorder.start(self.images)
        self.assertTrue(self.recorder.active)

    def test_partial_video_failure_has_truthful_indices_and_stops_recording(self):
        self.recorder.start(self.images, source_frame=self.original)
        stream = self.recorder._metadata_stream
        self.fake_cv2.writers[1].fail_write = True
        with self.assertRaisesRegex(RuntimeError, "could not write recording frame"):
            self.recorder.write(self.images, source_frame=self.original)
        self.assertFalse(self.recorder.active)
        self.assertTrue(stream.closed)
        self.assertTrue(all(writer.released for writer in self.fake_cv2.writers))
        record = self.records(self.sessions()[0])[-1]
        self.assertEqual(record["recording_status"], "write_failed")
        self.assertEqual(record["window_frames"], {"annotated.avi": 1, "edges.avi": None})
        self.assertTrue((self.sessions()[0] / record["source_frame"]).exists())

    def test_reserved_metadata_fields_cannot_claim_false_alignment(self):
        self.recorder.start(self.images, metadata={"frame_index": 99, "source_frame": "wrong.png", "window_frames": {}, "recording_status": "invented"})
        record = self.records(self.sessions()[0])[0]
        self.assertEqual(record["frame_index"], 0)
        self.assertIsNone(record["source_frame"])
        self.assertEqual(record["window_frames"], {"annotated.avi": 0, "edges.avi": 0})
        self.assertEqual(record["recording_status"], "complete")


if __name__ == "__main__":
    unittest.main()
