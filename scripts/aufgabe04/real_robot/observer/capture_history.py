"""Bounded asynchronous raw-camera evidence; never motion or pose authority."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import queue
import threading
from typing import Mapping


def sensor_capture_metadata(image_message, camera_info, scan_message) -> dict[str, object]:
    """Detach ROS sensor primitives without importing ROS or encoding images.

    Invalid scan returns remain explicit tagged values, not fabricated zero
    ranges. Other numeric metadata must be finite for reliable replay.
    """

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("capture metadata contains nonfinite calibration/geometry")
        return number

    def header(message):
        value = message.header
        return {"frame_id": str(value.frame_id), "stamp_sec": int(value.stamp.sec),
                "stamp_nanosec": int(value.stamp.nanosec)}

    def vector(values, *, ranges=False):
        if len(values) > 8192:
            raise ValueError("capture sensor vector exceeds 8192 elements")
        result = []
        for value in values:
            number = float(value)
            if math.isfinite(number):
                result.append(number)
            elif ranges:
                result.append("nan" if math.isnan(number) else ("+inf" if number > 0 else "-inf"))
            else:
                raise ValueError("nonfinite camera calibration")
        return result

    return {
        "image_header": header(image_message),
        "camera_info": {
            "header": header(camera_info), "width": int(camera_info.width),
            "height": int(camera_info.height), "distortion_model": str(camera_info.distortion_model),
            **{name: vector(getattr(camera_info, name)) for name in ("k", "d", "r", "p")},
            "binning_x": int(getattr(camera_info, "binning_x", 0)),
            "binning_y": int(getattr(camera_info, "binning_y", 0)),
            "roi": {name: getattr(camera_info.roi, name) for name in
                    ("x_offset", "y_offset", "width", "height", "do_rectify")}
                    if hasattr(camera_info, "roi") else None,
        },
        "scan": {
            "header": header(scan_message),
            **{name: finite(getattr(scan_message, name)) for name in
               ("angle_min", "angle_max", "angle_increment", "range_min", "range_max",
                "time_increment", "scan_time")},
            "ranges": vector(scan_message.ranges, ranges=True),
            "intensities": vector(getattr(scan_message, "intensities", ()), ranges=True),
            "nonfinite_encoding": "tagged strings nan/+inf/-inf preserve invalid sensor returns",
        },
    }


class BoundedObserverCapture:
    """Preserve original bytes and detached metadata without image encoding.

    Quotas include a conservative metadata-envelope allowance. A failed write
    does not refund its reservation, so repeated disk faults cannot evade the
    run budget. Only the worker performs filesystem writes after construction.
    """

    def __init__(self, output_dir: Path, *, max_frames: int = 64,
                 max_bytes: int = 32 * 1024 * 1024, max_queue_frames: int = 4,
                 max_queue_bytes: int = 8 * 1024 * 1024) -> None:
        limits = dict(max_frames=max_frames, max_bytes=max_bytes,
                      max_queue_frames=max_queue_frames, max_queue_bytes=max_queue_bytes)
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("capture limits must be positive integers")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise ValueError("capture history requires an empty output directory")
        self._limits = limits
        self._queue = queue.Queue(maxsize=max_queue_frames)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._submitted = self._accepted = self._written = self._write_failures = 0
        self._reserved_bytes = self._written_bytes = self._pending_bytes = 0
        self._pending_frames = 0
        self._drops: dict[str, int] = {}
        self._last_error: str | None = None
        self._worker = threading.Thread(target=self._run,
                                        name="aufgabe04-camera-capture", daemon=True)
        self._worker.start()

    def submit(self, original_compressed: bytes, *, metadata: Mapping[str, object],
               compressed_format: str = "") -> bool:
        """Try once; never wait for disk or queue capacity."""

        try:
            if not isinstance(original_compressed, bytes) or not original_compressed:
                raise ValueError("capture requires nonempty immutable compressed bytes")
            # Freeze nested data before the observer reuses its dictionaries.
            metadata_bytes = json.dumps(
                {"compressed_format": str(compressed_format), "metadata": dict(metadata)},
                allow_nan=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            with self._lock:
                self._submitted += 1
                self._drop("invalid_payload")
            return False
        reserved = len(original_compressed) + len(metadata_bytes) + 512
        with self._lock:
            self._submitted += 1
            if self._closing.is_set():
                return self._drop("closed")
            if self._accepted >= self._limits["max_frames"]:
                return self._drop("frame_limit")
            if self._reserved_bytes + reserved > self._limits["max_bytes"]:
                return self._drop("byte_limit")
            if self._pending_bytes + reserved > self._limits["max_queue_bytes"]:
                return self._drop("queue_byte_limit")
            if self._pending_frames >= self._limits["max_queue_frames"]:
                return self._drop("queue_frame_limit")
            index = self._accepted + 1
            try:
                self._queue.put_nowait((index, original_compressed, metadata_bytes, reserved))
            except queue.Full:
                return self._drop("queue_frame_limit")
            self._accepted += 1
            self._reserved_bytes += reserved
            self._pending_bytes += reserved
            self._pending_frames += 1
            return True

    def _drop(self, reason: str) -> bool:
        self._drops[reason] = self._drops.get(reason, 0) + 1
        return False

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 1, "diagnostic_only": True, "authoritative": False,
                "limits": dict(self._limits), "submitted_frames": self._submitted,
                "accepted_frames": self._accepted, "written_frames": self._written,
                "write_failures": self._write_failures,
                "reserved_bytes": self._reserved_bytes, "written_bytes": self._written_bytes,
                "pending_frames": self._pending_frames, "pending_bytes": self._pending_bytes,
                "dropped_frames": sum(self._drops.values()), "drop_reasons": dict(self._drops),
                "last_error": self._last_error, "closed": self._closing.is_set(),
                "worker_alive": self._worker.is_alive(),
                "byte_limit_scope": "compressed_bytes_and_frame_metadata; summary is bounded separately",
            }

    def close(self, timeout_sec: float = 2.0) -> dict[str, object]:
        """Bound the shutdown wait; a blocked diagnostic writer cannot hang ROS."""

        if not 0 <= timeout_sec <= 60:
            raise ValueError("capture close timeout must be between zero and 60 seconds")
        # Serialize closing with submit's admission/enqueue reservation. A
        # submission accepted before this point must be visible to the drain.
        with self._lock:
            self._closing.set()
        self._worker.join(timeout=timeout_sec)
        result = self.snapshot()
        result["flushed"] = not result["worker_alive"] and result["pending_frames"] == 0
        return result

    def _run(self) -> None:
        while not self._closing.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            index, compressed, metadata, reserved = item
            try:
                written = self._write_frame(index, compressed, metadata)
                with self._lock:
                    self._written += 1
                    self._written_bytes += written
            except Exception as exc:
                with self._lock:
                    self._write_failures += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            finally:
                with self._lock:
                    self._pending_frames -= 1
                    self._pending_bytes -= reserved
                self._queue.task_done()
            self._write_summary()
        self._write_summary(drain_complete=True)

    def _write_frame(self, index: int, compressed: bytes, metadata: bytes) -> int:
        image_name = f"frame_{index:06d}.compressed"
        payload = {
            **json.loads(metadata), "schema_version": 1, "authoritative": False,
            "capture_index": index, "image_file": image_name,
            "image_sha256": hashlib.sha256(compressed).hexdigest(),
        }
        encoded = (json.dumps(payload, allow_nan=False, sort_keys=True,
                              separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > len(metadata) + 512:
            raise ValueError("capture metadata exceeded reserved envelope")
        self._atomic_bytes(self.output_dir / image_name, compressed)
        self._atomic_bytes(self.output_dir / f"frame_{index:06d}.json", encoded)
        return len(compressed) + len(encoded)

    def _write_summary(self, *, drain_complete: bool = False) -> None:
        try:
            payload = (json.dumps({**self.snapshot(), "drain_complete": drain_complete},
                                  allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
            self._atomic_bytes(self.output_dir / "summary.json", payload)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:256]

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
