"""Keep ROS ingestion responsive while one owner performs camera work.

The ingestion thread only appends immutable sensor references to bounded
mailboxes and services the thread-safe TF buffer. Evidence, scan persistence,
tracking, capture metadata and detector work remain on the main owner thread.
No detector jobs are queued: a due iteration uses the latest sensor snapshot.
"""

from collections import Counter, deque
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class SensorIngressBatch:
    images: tuple
    scans: tuple
    camera_infos: tuple
    counts: dict


class BoundedSensorIngress:
    """Transfer callback receipts without sharing observer state across threads."""

    LIMITS = {"images": 8, "scans": 20, "camera_infos": 8}
    INVALID = {"images": "invalid_image_headers", "scans": "invalid_scan_headers",
               "camera_infos": "invalid_camera_info_headers"}

    def __init__(self):
        self._queues = {key: deque(maxlen=limit) for key, limit in self.LIMITS.items()}
        self._counts = Counter()
        self._lock = threading.Lock()

    def offer(self, channel, sample):
        """A None sample records a malformed receipt without retaining it."""
        with self._lock:
            queue = self._queues[channel]
            self._counts[f"received_{channel}"] += 1
            if sample is None:
                self._counts[self.INVALID[channel]] += 1
                return
            if len(queue) == queue.maxlen:
                self._counts[f"ingress_overwritten_{channel}"] += 1
            queue.append(sample)

    def drain(self):
        with self._lock:
            batch = SensorIngressBatch(
                **{key: tuple(queue) for key, queue in self._queues.items()},
                counts=dict(self._counts))
            for queue in self._queues.values():
                queue.clear()
            self._counts.clear()
            return batch


class ObserverWorkSchedule:
    """Single-owner periodic work without timer backlog or detector reentrancy."""

    def __init__(self, *, process_rate_hz, tf_retry_rate_hz, monotonic=time.monotonic):
        if any(not math.isfinite(rate) or rate <= 0
               for rate in (process_rate_hz, tf_retry_rate_hz)):
            raise ValueError("observer work rates must be finite and positive")
        self._process_period = 1. / process_rate_hz
        self._retry_period = 1. / tf_retry_rate_hz
        self._next_process = self._next_retry = -math.inf
        self._clock = monotonic
        self._owner = threading.get_ident()
        self._running = False

    def run_due(self, *, drain, collect_witnesses, process, retry):
        if threading.get_ident() != self._owner or self._running:
            raise RuntimeError("observer work requires its non-reentrant owner thread")
        self._running = True
        try:
            drain()
            collect_witnesses()
            now = self._clock()
            if now >= self._next_process:
                self._next_process = now + self._process_period
                self._next_retry = now + self._retry_period
                process()
            elif now >= self._next_retry:
                self._next_retry = now + self._retry_period
                retry()
        finally:
            self._running = False


class ObserverIngestionLoop:
    """Own one background executor, surfacing its failure to the camera owner."""

    def __init__(self, *, spin_once, wake, ok):
        self._spin_once, self._wake, self._ok = spin_once, wake, ok
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._failure = None
        self._thread = threading.Thread(target=self._run,
                                        name="observer-ros-ingestion", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set() and self._ok():
                self._spin_once()
        except BaseException as exc:
            self._failure = exc
        finally:
            self._finished.set()

    def raise_if_failed(self):
        if self._failure is not None:
            raise RuntimeError("passive observer ROS ingestion failed") from self._failure

    def wait(self, timeout_sec=.01):
        """Short owner pause; failures wake it immediately."""
        self._finished.wait(timeout_sec)
        self.raise_if_failed()

    def close(self, *, timeout_sec=2.):
        self._stop.set()
        self._wake()
        if self._thread.ident is not None:
            self._thread.join(timeout_sec)
            if self._thread.is_alive():
                raise RuntimeError("passive observer ROS ingestion did not stop")
