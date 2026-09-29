"""Pure validation and ray tracing for stopped, stamped tour LaserScans."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json


CAPTURE_HASH_FIELD = "tour_scan_capture_sha256"
RESOLUTION_M = .05
RAY_CAP_M = 3.
TTL_SEC = 30.


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"temporary occupancy {name} must be finite numeric")
    return float(value)


def pose(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping) or set(value) != {"x_m", "y_m", "yaw_rad"}:
        raise ValueError(f"temporary occupancy {name} is not a planar pose")
    result = tuple(number(value[k], name) for k in ("x_m", "y_m", "yaw_rad"))
    if any(abs(v) > 1e6 for v in result):
        raise ValueError(f"temporary occupancy {name} exceeds bounded local coordinates")
    return result


def validate_capture(path: Path, *, tour_id: str, odom_frame: str, now_sec: float) -> dict:
    data = load_content_hashed_json(path, hash_field=CAPTURE_HASH_FIELD)
    for key, expected in (("schema_version", 1), ("artifact_kind", "stored_pose_tour_scan_capture"),
                          ("tour_id", tour_id), ("odom_frame", odom_frame)):
        if type(data.get(key)) is not type(expected) or data.get(key) != expected:
            raise ValueError(f"temporary scan capture {key} mismatch")
    for key in ("base_frame", "scan_frame"):
        if not isinstance(data.get(key), str) or not data[key] or data[key].startswith("/"):
            raise ValueError(f"temporary scan capture {key} is invalid")
    captured = number(data.get("captured_at_unix_sec"), "capture time")
    now = number(now_sec, "admission time")
    if captured <= 0. or not -.02 <= now - captured <= .5:
        raise ValueError("temporary scan capture is stale or from the future")
    scans = data.get("scans")
    if not isinstance(scans, list) or len(scans) != 3:
        raise ValueError("temporary scan capture requires exactly three distinct scans")
    stamps, receipts, poses = [], [], []
    for scan in scans:
        if not isinstance(scan, Mapping):
            raise ValueError("temporary scan must be an object")
        stamp = number(scan.get("stamp_sec"), "scan stamp")
        receipt = number(scan.get("received_at_unix_sec"), "scan receipt")
        if stamp <= 0. or not -.02 <= receipt - stamp <= .25:
            raise ValueError("temporary scan was stale or future stamped at receipt")
        if captured - receipt < -.02:
            raise ValueError("temporary scan receipt is from the future")
        for key in ("scan_pose_stamp_sec", "base_pose_stamp_sec"):
            if abs(number(scan.get(key), key) - stamp) > 1e-9:
                raise ValueError("temporary scan and transform timestamps differ")
        pose(scan.get("scan_pose_odom"), "scan pose")
        poses.append(pose(scan.get("base_pose_odom"), "base pose"))
        minimum, maximum = number(scan.get("range_min"), "range_min"), number(scan.get("range_max"), "range_max")
        increment = number(scan.get("angle_increment"), "angle_increment")
        number(scan.get("angle_min"), "angle_min")
        ranges = scan.get("ranges")
        if not 0. <= minimum < maximum or increment == 0.:
            raise ValueError("temporary scan range or angle bounds are invalid")
        if not isinstance(ranges, list) or not 2 <= len(ranges) <= 4096:
            raise ValueError("temporary scan beam count is invalid")
        if abs(increment) * (len(ranges) - 1) > math.tau + .05:
            raise ValueError("temporary scan angular extent exceeds one revolution")
        for value in ranges:
            if value is not None:
                number(value, "beam range")
        stamps.append(stamp)
        receipts.append(receipt)
    if any(b-a < .08-1e-6 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("temporary scan stamps are repeated, reordered or too close")
    if stamps[-1]-stamps[0] > .5+1e-6 or max(receipts)-min(receipts) > .5+1e-6:
        raise ValueError("temporary scan cohort exceeds the stopped window")
    if not -.02 <= captured-stamps[-1] <= .25:
        raise ValueError("temporary scan latest sample is stale at capture")
    for index, a in enumerate(poses):
        for b in poses[index+1:]:
            if math.hypot(a[0]-b[0], a[1]-b[1]) > .015+1e-9:
                raise ValueError("temporary scan base moved during capture")
            if abs(math.remainder(a[2]-b[2], math.tau)) > math.radians(2.)+1e-9:
                raise ValueError("temporary scan base rotated during capture")
    # A fixed sensor must retain its mounting transform throughout the cohort.
    mounts = []
    for scan, base in zip(scans, poses):
        sensor = pose(scan["scan_pose_odom"], "scan pose")
        dx, dy = sensor[0]-base[0], sensor[1]-base[1]
        mounts.append((math.cos(base[2])*dx+math.sin(base[2])*dy,
                       -math.sin(base[2])*dx+math.cos(base[2])*dy,
                       math.remainder(sensor[2]-base[2], math.tau)))
    if any(math.hypot(a[0]-mounts[0][0], a[1]-mounts[0][1]) > .005
           or abs(math.remainder(a[2]-mounts[0][2], math.tau)) > .01 for a in mounts[1:]):
        raise ValueError("temporary scan mounting transform changed")
    return data


def _ray_cells(x: float, y: float, ex: float, ey: float) -> set[tuple[int, int]]:
    """Grid traversal; corner crossings visit both incident cells conservatively."""
    dx, dy = ex-x, ey-y
    length = math.hypot(dx, dy)
    if length == 0.:
        return set()
    # Oversampling alone can skip short corner crossings; merge all exact grid
    # boundary parameters and sample each resulting open interval instead.
    times = [0., 1.]
    for a, b, delta in ((x, ex, dx), (y, ey, dy)):
        if delta:
            low, high = sorted((a, b))
            for gridline in range(math.floor(low/RESOLUTION_M)+1, math.ceil(high/RESOLUTION_M)):
                t = (gridline*RESOLUTION_M-a)/delta
                if 0. < t < 1.:
                    times.append(t)
    times = sorted(set(times))
    return {(math.floor((x+dx*((a+b)/2))/RESOLUTION_M),
             math.floor((y+dy*((a+b)/2))/RESOLUTION_M))
            for a, b in zip(times, times[1:])}


def scan_cells(scan: Mapping) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return raw endpoints and finite-ray freespace, without robot inflation."""
    x, y, yaw = pose(scan["scan_pose_odom"], "scan pose")
    occupied, free = set(), set()
    for index, raw in enumerate(scan["ranges"]):
        if raw is None or raw <= 0. or not scan["range_min"] <= raw <= scan["range_max"]:
            continue
        distance = min(raw, RAY_CAP_M)
        angle = yaw+scan["angle_min"]+index*scan["angle_increment"]
        ux, uy = math.cos(angle), math.sin(angle)
        if raw <= RAY_CAP_M:
            occupied.add((math.floor((x+distance*ux)/RESOLUTION_M),
                          math.floor((y+distance*uy)/RESOLUTION_M)))
        # Never clear the last cell-sized surface band from a finite return.
        clear_distance = max(0., distance-RESOLUTION_M)
        free.update(_ray_cells(x, y, x+clear_distance*ux, y+clear_distance*uy))
    return occupied, free-occupied
