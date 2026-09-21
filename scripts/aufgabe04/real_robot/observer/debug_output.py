"""Latest-only asynchronous camera diagnostics; never authoritative receipts."""

from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time


def debug_images(frame, roi, debug):
    return dict(zip(("latest_frame.png", "latest_head_roi.png", "latest_edges.png",
                     "latest_raw_edges.png", "latest_side_evidence.png",
                     "latest_rectangle_mask.png", "latest_rectangle_overlay.png"),
                    (frame, roi, debug.edges, debug.raw_edges, debug.face_mask,
                     debug.rectangle_mask, debug.rectangle_overlay)))


def write_debug_snapshot(cv2, directory, images, metadata):
    """Write metadata last; encode identical image objects only once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written, encoded = [], {}
    for name, image in images.items():
        path = directory / name
        if image is None:
            path.unlink(missing_ok=True)
            continue
        if hasattr(cv2, "imencode"):
            if id(image) not in encoded:
                ok, data = cv2.imencode(".png", image)
                encoded[id(image)] = data.tobytes() if ok else None
            data = encoded[id(image)]
            if data is not None:
                temporary = path.with_suffix(".tmp")
                try:
                    temporary.write_bytes(data)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                written.append(name)
            else:
                path.unlink(missing_ok=True)
        elif cv2.imwrite(str(path), image):  # Simple injected backends.
            written.append(name)
        else:
            path.unlink(missing_ok=True)
    path = directory / "latest_metadata.json"
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps({"schema_version": 1,
            "observed_unix_sec": time.time(), "artifacts": written,
            "stand_axis": metadata}, allow_nan=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ObserverDebugWriter:
    """At most one writing and one pending snapshot; disk never blocks submit.

    Arrays and metadata are detached on submission. Replaced pending frames
    are counted. Worker failures are diagnostics, not observation authority.
    """

    def __init__(self, cv2, directory, *, max_image_bytes=16 * 1024 * 1024):
        self.cv2, self.directory = cv2, directory
        self.max_image_bytes = max_image_bytes
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._submitted = self._replaced = self._written = self._failed = self._oversized = 0
        self._last_error = None
        self._last_write_sec = None
        self._thread = threading.Thread(target=self._run, name="observer-debug-writer", daemon=True)
        self._thread.start()

    def submit(self, images, metadata):
        unique = {id(image): image for image in images.values() if image is not None}
        if sum(image.nbytes for image in unique.values()) > self.max_image_bytes:
            with self._condition:
                self._oversized += 1
            return False
        copies = {key: image.copy() for key, image in unique.items()}
        snapshot = ({name: None if image is None else copies[id(image)]
                     for name, image in images.items()}, deepcopy(metadata))
        with self._condition:
            if self._closed:
                return False
            self._submitted += 1
            self._replaced += int(self._pending is not None)
            self._pending = snapshot
            self._condition.notify()
        return True

    def snapshot(self):
        with self._condition:
            return dict(diagnostic_only=True, submitted=self._submitted,
                replaced=self._replaced, written=self._written, failures=self._failed,
                oversized=self._oversized, pending=int(self._pending is not None),
                last_error=self._last_error, last_write_sec=self._last_write_sec,
                worker_alive=self._thread.is_alive())

    def close(self, timeout_sec=2.):
        if not math.isfinite(timeout_sec) or not 0 <= timeout_sec <= 60:
            raise ValueError("debug shutdown timeout must be within 0..60 seconds")
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout_sec)
        return self.snapshot()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._pending is None:
                    return
                images, metadata = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                write_debug_snapshot(self.cv2, self.directory, images, metadata)
                with self._condition:
                    self._written += 1
            except Exception as exc:
                with self._condition:
                    self._failed += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            finally:
                with self._condition:
                    self._last_write_sec = time.monotonic() - started
