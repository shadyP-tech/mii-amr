"""Incoming timing must include images rejected before model processing."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.perception.debug.color_mask_viewer import RosCompressedImageTopicFrameSource
from scripts.aufgabe04.perception.debug.frame_receipt_diagnostics import FrameReceiptDiagnostics
from scripts.aufgabe04.perception.debug.viewer_frame_timing import ViewerFrameTiming


def test_callback_counts_rejected_and_future_header_ages_without_relabeling():
    # Exercise the real callback without initializing a ROS node.
    source = RosCompressedImageTopicFrameSource.__new__(RosCompressedImageTopicFrameSource)
    source.max_frame_age_sec = .25
    source._lock = threading.Lock()
    source._receipt_diagnostics = FrameReceiptDiagnostics()
    source.latest_sequence = source.received_count = source._last_fps_count = 0
    source._last_fps_sec = 10.
    source.receive_fps = None
    for sec, nanosec in ((9, 700000000), (9, 900000000), (10, 100000000)):
        message = SimpleNamespace(data=b"jpeg", format="jpeg",
            header=SimpleNamespace(frame_id="camera", stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))
        with patch("scripts.aufgabe04.perception.debug.color_mask_viewer.time.time", return_value=10.), \
             patch("scripts.aufgabe04.perception.debug.color_mask_viewer.time.monotonic", return_value=20.):
            source._on_image(message)
    summary = source.receipt_diagnostics()
    assert summary["arrivals"] == 3
    assert summary["age_rejections"] == 1
    assert summary["future_stamp"] == 1
    assert summary["all_header_to_receipt_age"]["count"] == 3
    assert summary["rejected_header_to_receipt_age"]["minimum_sec"] > .25
    assert summary["accepted_header_to_receipt_age"]["minimum_sec"] < 0.
    assert summary["header_receipt_clock_offset_measured"] is False
    assert source.latest_sequence == source.received_count == 2
    assert source.latest_stamp_sec == 10.1


def test_work_deadline_preserves_source_and_receipt_limits():
    timing = ViewerFrameTiming(10., 9.8)
    assert timing.work_deadline(max_result_age_sec=.18, max_frame_age_sec=.25) == 10.05
    assert timing.work_deadline(max_result_age_sec=.18, max_frame_age_sec=0.) == 10.18
    assert timing.work_deadline(max_result_age_sec=0., max_frame_age_sec=0.) is None
    assert not timing.assess(now_sec=10.06, max_result_age_sec=.18, max_frame_age_sec=.25).accepted
