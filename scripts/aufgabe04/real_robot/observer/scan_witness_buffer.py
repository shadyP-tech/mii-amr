"""Bounded scan-only evidence waiting for a current verified head bearing.

No camera success or target selection is needed to retain a scan. The owner
validates each source-time context before ingestion; this buffer enforces one
candidate and stopped epoch, distinct stamps, ordering and bounded storage.
"""

from collections import deque


MAX_PENDING_SCANS = 32


class StoppedScanWitnessBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._entries = deque(maxlen=MAX_PENDING_SCANS)
        self._anchor = None
        self._last_stamp = None

    def ingest(self, entry):
        """Return whether a discontinuity invalidates the owner's witnesses."""
        # Late import keeps the persistence schema and its geometric limits
        # authoritative without duplicating them in the bounded input queue.
        from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
            _CANDIDATE_GEOMETRY_FIELDS, _pose, _stationary, _same_extrinsic,
            MAX_HISTORY_SEC, MAX_SCAN_GAP_SEC,
        )
        context, scan = entry["context"], entry["scan"]
        stamp = scan["scan_stamp_sec"]
        key = (context["target_key"], context["epoch_key"], scan["scan_frame_id"],
               *(context[field] for field in _CANDIDATE_GEOMETRY_FIELDS))
        robot, pose = _pose(context["robot_pose"]), _pose(context["scan_pose_map"])
        extrinsic = _pose(context["scan_pose_robot"])
        discontinuity = bool(
            self._anchor is not None and (
                key != self._anchor[0] or not _stationary(self._anchor[1], robot)
                or not _stationary(self._anchor[2], pose)
                or not _same_extrinsic(self._anchor[3], extrinsic))
            or self._last_stamp is not None and (
                stamp < self._last_stamp or stamp - self._last_stamp > MAX_SCAN_GAP_SEC))
        if discontinuity:
            self.reset()
        if self._anchor is None:
            self._anchor = (key, robot, pose, extrinsic)
        if stamp == self._last_stamp:
            return discontinuity
        while self._entries and stamp - self._entries[0]["scan"]["scan_stamp_sec"] > MAX_HISTORY_SEC:
            self._entries.popleft()
        self._entries.append(entry)
        self._last_stamp = stamp
        return discontinuity

    def take_before(self, stamp, *, after_stamp=None):
        """Consume earlier scans once; the current scan remains current evidence."""
        entries = []
        while self._entries and self._entries[0]["scan"]["scan_stamp_sec"] <= stamp:
            entry = self._entries.popleft()
            old_stamp = entry["scan"]["scan_stamp_sec"]
            if old_stamp < stamp and (after_stamp is None or old_stamp > after_stamp):
                entries.append(entry)
        return entries
