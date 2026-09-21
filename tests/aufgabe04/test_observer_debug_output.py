"""Disk stalls cannot queue camera history or mutate a submitted snapshot."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

np = pytest.importorskip("numpy")

from scripts.aufgabe04.real_robot.observer.debug_output import (
    ObserverDebugWriter, write_debug_snapshot,
)


def test_blocked_writer_retains_only_latest_detached_pending_frame(tmp_path):
    entered, release = Event(), Event()
    recorded = []
    def write(_cv, _directory, images, metadata):
        if not recorded:
            entered.set()
            assert release.wait(2.)
        recorded.append((int(images["frame"][0, 0]), metadata["nested"][0]))
    with patch("scripts.aufgabe04.real_robot.observer.debug_output.write_debug_snapshot", write):
        writer = ObserverDebugWriter(object(), tmp_path)
        try:
            image = np.zeros((2, 2), dtype=np.uint8)
            metadata = {"nested": [0]}
            assert writer.submit({"frame": image}, metadata)
            assert entered.wait(1.)
            for value in range(1, 10):
                image[:] = value
                metadata["nested"][0] = value
                assert writer.submit({"frame": image}, metadata)
            image[:] = 99
            metadata["nested"][0] = 99
            assert writer.snapshot()["pending"] == 1
            assert writer.snapshot()["replaced"] == 8
            assert writer.close(timeout_sec=0.)["worker_alive"]
        finally:
            release.set()
            writer.close()
    assert recorded == [(0, 0), (9, 9)]


def test_write_failure_is_visible_and_does_not_prevent_shutdown(tmp_path):
    with patch("scripts.aufgabe04.real_robot.observer.debug_output.write_debug_snapshot",
               side_effect=OSError("disk unavailable")):
        writer = ObserverDebugWriter(object(), tmp_path)
        assert writer.submit({"frame": np.zeros((2, 2), dtype=np.uint8)}, {})
        result = writer.close()
    assert result["failures"] == 1
    assert "disk unavailable" in result["last_error"]
    assert not result["worker_alive"]


def test_repeated_image_objects_encode_once_and_missing_images_are_removed(tmp_path):
    backend = SimpleNamespace(imencode=Mock(return_value=(True, np.array([1, 2], np.uint8))))
    image = np.zeros((2, 2), dtype=np.uint8)
    write_debug_snapshot(backend, tmp_path, {"frame.png": image, "roi.png": image}, {"frame": 1})
    assert backend.imencode.call_count == 1
    assert (tmp_path / "frame.png").read_bytes() == (tmp_path / "roi.png").read_bytes()
    write_debug_snapshot(backend, tmp_path, {"frame.png": image, "roi.png": None}, {"frame": 2})
    assert not (tmp_path / "roi.png").exists()
    metadata = json.loads((tmp_path / "latest_metadata.json").read_text())
    assert metadata["artifacts"] == ["frame.png"] and metadata["stand_axis"]["frame"] == 2


def test_oversized_snapshot_never_enters_queue(tmp_path):
    writer = ObserverDebugWriter(object(), tmp_path, max_image_bytes=1)
    try:
        assert not writer.submit({"frame": np.zeros((2, 2), np.uint8)}, {})
        assert writer.snapshot()["oversized"] == 1
        assert writer.snapshot()["pending"] == 0
    finally:
        writer.close()
