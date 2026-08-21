"""Target-agnostic, content-hashed LiDAR visibility scan receipts.

Receipts retain the complete angular scan needed for later negative-visibility
reasoning without naming a candidate.  Non-finite and sensor-invalid samples
are represented by ``None`` so JSON artifacts never turn missing evidence into
clearance.  The module is pure Python and never interacts with ROS or motion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.models import Pose2D


LIDAR_VISIBILITY_RECEIPT_SCHEMA_VERSION = 1
VISIBILITY_EVIDENCE_ENABLED_KEY = "lidar_visibility_evidence_enabled"
VISIBILITY_RECEIPTS_JSONL_KEY = "lidar_visibility_receipts_jsonl"
VISIBILITY_RECEIPT_COUNT_KEY = "lidar_visibility_receipt_count"
VISIBILITY_RECEIPTS_FILE_SHA256_KEY = (
    "lidar_visibility_receipts_file_sha256"
)
VISIBILITY_RECEIPT_SET_SHA256_KEY = "lidar_visibility_receipt_set_sha256"
VISIBILITY_OBSERVER_CONFIG_KEY = "lidar_visibility_observer_config"
VISIBILITY_OBSERVER_CONFIG_SHA256_KEY = (
    "lidar_visibility_observer_config_sha256"
)
_EXACT_TIME_EPSILON_SEC = 1.0e-9
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME_OR_TOPIC = re.compile(r"^/?[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELD = "receipt_sha256"
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "survey_id",
        "viewpoint_id",
        "planning_frame",
        "scan_frame",
        "scan_topic",
        "map_bundle_sha256",
        "observer_config_sha256",
        "scan_stamp_sec",
        "pose_stamp_sec",
        "observer_clock_sec",
        "scan_pose_map",
        "angle_min_rad",
        "angle_increment_rad",
        "range_min_m",
        "range_max_m",
        "ranges_m",
    }
)


@dataclass(frozen=True)
class LidarVisibilityReceipt:
    """One exact-time scan and map-frame sensor pose."""

    schema_version: int
    receipt_id: str
    survey_id: str
    viewpoint_id: str
    planning_frame: str
    scan_frame: str
    scan_topic: str
    map_bundle_sha256: str
    observer_config_sha256: str
    scan_stamp_sec: float
    pose_stamp_sec: float
    observer_clock_sec: float
    scan_pose_map: Pose2D
    angle_min_rad: float
    angle_increment_rad: float
    range_min_m: float
    range_max_m: float
    ranges_m: tuple[float | None, ...]

    @property
    def finite_range_count(self) -> int:
        return sum(value is not None for value in self.ranges_m)

    @property
    def receipt_sha256(self) -> str:
        return payload_sha256(visibility_receipt_payload(self))

    def to_evidence_dict(self) -> dict[str, object]:
        payload = visibility_receipt_payload(self)
        payload[_HASH_FIELD] = payload_sha256(payload)
        return payload


def sanitized_scan_ranges(
    ranges_m: Iterable[float],
    *,
    range_min_m: float,
    range_max_m: float,
) -> tuple[float | None, ...]:
    """Map invalid returns to ``None`` while preserving scan indexing."""

    minimum = _finite_nonnegative(range_min_m, "range_min_m")
    maximum = _finite_positive(range_max_m, "range_max_m")
    if maximum <= minimum:
        raise ValueError("range_max_m must be greater than range_min_m")
    sanitized: list[float | None] = []
    try:
        iterator = iter(ranges_m)
    except TypeError as exc:
        raise ValueError("ranges_m must be iterable") from exc
    for raw_value in iterator:
        if isinstance(raw_value, bool):
            sanitized.append(None)
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            sanitized.append(None)
            continue
        sanitized.append(
            value
            if math.isfinite(value) and minimum <= value <= maximum
            else None
        )
    if not sanitized:
        raise ValueError("ranges_m must contain at least one sample")
    return tuple(sanitized)


def lidar_visibility_receipt_from_scan(
    *,
    receipt_id: str,
    survey_id: str,
    viewpoint_id: str,
    planning_frame: str,
    scan_frame: str,
    scan_topic: str,
    map_bundle_sha256: str,
    observer_config_sha256: str,
    scan_stamp_sec: float,
    pose_stamp_sec: float,
    observer_clock_sec: float,
    scan_pose_map: Pose2D,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    ranges_m: Iterable[float],
) -> LidarVisibilityReceipt:
    """Build and validate one JSON-safe exact-time visibility receipt."""

    receipt = LidarVisibilityReceipt(
        schema_version=LIDAR_VISIBILITY_RECEIPT_SCHEMA_VERSION,
        receipt_id=receipt_id,
        survey_id=survey_id,
        viewpoint_id=viewpoint_id,
        planning_frame=planning_frame,
        scan_frame=scan_frame,
        scan_topic=scan_topic,
        map_bundle_sha256=map_bundle_sha256,
        observer_config_sha256=observer_config_sha256,
        scan_stamp_sec=float(scan_stamp_sec),
        pose_stamp_sec=float(pose_stamp_sec),
        observer_clock_sec=float(observer_clock_sec),
        scan_pose_map=scan_pose_map,
        angle_min_rad=float(angle_min_rad),
        angle_increment_rad=float(angle_increment_rad),
        range_min_m=float(range_min_m),
        range_max_m=float(range_max_m),
        ranges_m=sanitized_scan_ranges(
            ranges_m,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
        ),
    )
    return validate_lidar_visibility_receipt(receipt)


def validate_lidar_visibility_receipt(
    receipt: LidarVisibilityReceipt,
) -> LidarVisibilityReceipt:
    if not isinstance(receipt, LidarVisibilityReceipt):
        raise ValueError("receipt must be a LidarVisibilityReceipt")
    if receipt.schema_version != LIDAR_VISIBILITY_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported LiDAR visibility receipt schema_version")
    for value, name in (
        (receipt.receipt_id, "receipt_id"),
        (receipt.survey_id, "survey_id"),
        (receipt.viewpoint_id, "viewpoint_id"),
    ):
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"{name} must be a safe non-empty identifier")
    for value, name in (
        (receipt.planning_frame, "planning_frame"),
        (receipt.scan_frame, "scan_frame"),
        (receipt.scan_topic, "scan_topic"),
    ):
        if (
            not isinstance(value, str)
            or _SAFE_FRAME_OR_TOPIC.fullmatch(value) is None
        ):
            raise ValueError(f"{name} is invalid")
    for value, name in (
        (receipt.map_bundle_sha256, "map_bundle_sha256"),
        (receipt.observer_config_sha256, "observer_config_sha256"),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    scan_stamp = _finite_nonnegative(receipt.scan_stamp_sec, "scan_stamp_sec")
    pose_stamp = _finite_nonnegative(receipt.pose_stamp_sec, "pose_stamp_sec")
    _finite_nonnegative(receipt.observer_clock_sec, "observer_clock_sec")
    if abs(scan_stamp - pose_stamp) > _EXACT_TIME_EPSILON_SEC:
        raise ValueError("scan pose must be resolved at the exact scan timestamp")
    if not isinstance(receipt.scan_pose_map, Pose2D) or not all(
        math.isfinite(value)
        for value in (
            receipt.scan_pose_map.x_m,
            receipt.scan_pose_map.y_m,
            receipt.scan_pose_map.yaw_rad,
        )
    ):
        raise ValueError("scan_pose_map must be a finite Pose2D")
    _finite(receipt.angle_min_rad, "angle_min_rad")
    increment = _finite(receipt.angle_increment_rad, "angle_increment_rad")
    if abs(increment) <= 1.0e-15:
        raise ValueError("angle_increment_rad must be non-zero")
    minimum = _finite_nonnegative(receipt.range_min_m, "range_min_m")
    maximum = _finite_positive(receipt.range_max_m, "range_max_m")
    if maximum <= minimum:
        raise ValueError("range_max_m must be greater than range_min_m")
    if not isinstance(receipt.ranges_m, tuple) or not receipt.ranges_m:
        raise ValueError("ranges_m must be a non-empty tuple")
    for value in receipt.ranges_m:
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            raise ValueError("ranges_m contains an unsanitized range")
    return receipt


def visibility_receipt_payload(
    receipt: LidarVisibilityReceipt,
) -> dict[str, object]:
    validate_lidar_visibility_receipt(receipt)
    return {
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "survey_id": receipt.survey_id,
        "viewpoint_id": receipt.viewpoint_id,
        "planning_frame": receipt.planning_frame,
        "scan_frame": receipt.scan_frame,
        "scan_topic": receipt.scan_topic,
        "map_bundle_sha256": receipt.map_bundle_sha256,
        "observer_config_sha256": receipt.observer_config_sha256,
        "scan_stamp_sec": receipt.scan_stamp_sec,
        "pose_stamp_sec": receipt.pose_stamp_sec,
        "observer_clock_sec": receipt.observer_clock_sec,
        "scan_pose_map": {
            "x_m": receipt.scan_pose_map.x_m,
            "y_m": receipt.scan_pose_map.y_m,
            "yaw_rad": receipt.scan_pose_map.yaw_rad,
        },
        "angle_min_rad": receipt.angle_min_rad,
        "angle_increment_rad": receipt.angle_increment_rad,
        "range_min_m": receipt.range_min_m,
        "range_max_m": receipt.range_max_m,
        "ranges_m": list(receipt.ranges_m),
    }


def visibility_receipts_sha256(
    receipts: Iterable[LidarVisibilityReceipt],
) -> str:
    """Canonical digest independent of JSONL whitespace."""

    payloads = [visibility_receipt_payload(receipt) for receipt in receipts]
    return payload_sha256({"receipts": payloads})


def append_lidar_visibility_receipts(
    path: Path,
    receipts: Iterable[LidarVisibilityReceipt],
) -> None:
    """Append validated compact records, refusing duplicate receipt IDs."""

    target = Path(path)
    if target.is_symlink():
        raise ValueError("visibility receipt JSONL path must not be a symlink")
    items = tuple(receipts)
    for item in items:
        validate_lidar_visibility_receipt(item)
    supplied_ids = tuple(item.receipt_id for item in items)
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("duplicate visibility receipt_id in append batch")
    existing_ids: set[str] = set()
    if target.exists():
        existing_ids = {
            item.receipt_id for item in load_lidar_visibility_receipts(target)
        }
    duplicate_ids = existing_ids.intersection(supplied_ids)
    if duplicate_ids:
        raise ValueError(
            "visibility receipt_id already exists: "
            + ", ".join(sorted(duplicate_ids))
        )
    if not items:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    item.to_evidence_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def load_lidar_visibility_receipts(
    path: Path,
) -> tuple[LidarVisibilityReceipt, ...]:
    receipts, _raw_sha256 = load_lidar_visibility_receipt_snapshot(path)
    return receipts


def load_lidar_visibility_receipt_snapshot(
    path: Path,
) -> tuple[tuple[LidarVisibilityReceipt, ...], str]:
    """Load, strictly validate each hash, and hash the exact parsed bytes."""

    target = Path(path)
    if target.is_symlink():
        raise ValueError("visibility receipt JSONL path must not be a symlink")
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise ValueError("visibility receipt JSONL is unavailable") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("visibility receipt JSONL must be UTF-8") from exc
    receipts: list[LidarVisibilityReceipt] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(value, Mapping):
                raise ValueError("line payload must be an object")
            receipt = _receipt_from_hashed_payload(value)
            if receipt.receipt_id in seen_ids:
                raise ValueError("duplicate receipt_id")
            seen_ids.add(receipt.receipt_id)
            receipts.append(receipt)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid visibility receipt JSONL line {line_number}: {exc}"
            ) from exc
    return tuple(receipts), hashlib.sha256(raw).hexdigest()


def _receipt_from_hashed_payload(
    payload: Mapping[str, object],
) -> LidarVisibilityReceipt:
    if frozenset(payload) != _PAYLOAD_FIELDS | {_HASH_FIELD}:
        raise ValueError("visibility receipt fields do not match schema")
    stored_hash = payload.get(_HASH_FIELD)
    if not isinstance(stored_hash, str) or _SHA256.fullmatch(stored_hash) is None:
        raise ValueError("receipt_sha256 must be a lowercase SHA-256")
    unhashed = dict(payload)
    del unhashed[_HASH_FIELD]
    actual_hash = payload_sha256(unhashed)
    if stored_hash != actual_hash:
        raise ValueError("visibility receipt hash mismatch")
    pose = _required_mapping(unhashed, "scan_pose_map")
    if frozenset(pose) != {"x_m", "y_m", "yaw_rad"}:
        raise ValueError("scan_pose_map fields do not match schema")
    ranges = unhashed.get("ranges_m")
    if not isinstance(ranges, list):
        raise ValueError("ranges_m must be an array")
    receipt = LidarVisibilityReceipt(
        schema_version=_required_integer(unhashed, "schema_version"),
        receipt_id=_required_string(unhashed, "receipt_id"),
        survey_id=_required_string(unhashed, "survey_id"),
        viewpoint_id=_required_string(unhashed, "viewpoint_id"),
        planning_frame=_required_string(unhashed, "planning_frame"),
        scan_frame=_required_string(unhashed, "scan_frame"),
        scan_topic=_required_string(unhashed, "scan_topic"),
        map_bundle_sha256=_required_string(unhashed, "map_bundle_sha256"),
        observer_config_sha256=_required_string(
            unhashed, "observer_config_sha256"
        ),
        scan_stamp_sec=_required_number(unhashed, "scan_stamp_sec"),
        pose_stamp_sec=_required_number(unhashed, "pose_stamp_sec"),
        observer_clock_sec=_required_number(unhashed, "observer_clock_sec"),
        scan_pose_map=Pose2D(
            _required_number(pose, "x_m"),
            _required_number(pose, "y_m"),
            _required_number(pose, "yaw_rad"),
        ),
        angle_min_rad=_required_number(unhashed, "angle_min_rad"),
        angle_increment_rad=_required_number(unhashed, "angle_increment_rad"),
        range_min_m=_required_number(unhashed, "range_min_m"),
        range_max_m=_required_number(unhashed, "range_max_m"),
        ranges_m=tuple(
            None if value is None else _finite_json_number(value, "ranges_m")
            for value in ranges
        ),
    )
    return validate_lidar_visibility_receipt(receipt)


def _required_mapping(
    payload: Mapping[str, object], name: str
) -> Mapping[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_integer(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _required_number(payload: Mapping[str, object], name: str) -> float:
    return _finite_json_number(payload.get(name), name)


def _finite_json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_positive(value: float, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_nonnegative(value: float, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON value {value!r}")


__all__ = [
    "LIDAR_VISIBILITY_RECEIPT_SCHEMA_VERSION",
    "VISIBILITY_EVIDENCE_ENABLED_KEY",
    "VISIBILITY_OBSERVER_CONFIG_KEY",
    "VISIBILITY_OBSERVER_CONFIG_SHA256_KEY",
    "VISIBILITY_RECEIPT_COUNT_KEY",
    "VISIBILITY_RECEIPTS_FILE_SHA256_KEY",
    "VISIBILITY_RECEIPTS_JSONL_KEY",
    "VISIBILITY_RECEIPT_SET_SHA256_KEY",
    "LidarVisibilityReceipt",
    "append_lidar_visibility_receipts",
    "lidar_visibility_receipt_from_scan",
    "load_lidar_visibility_receipt_snapshot",
    "load_lidar_visibility_receipts",
    "sanitized_scan_ranges",
    "validate_lidar_visibility_receipt",
    "visibility_receipt_payload",
    "visibility_receipts_sha256",
]
