"""Immutable, ROS-free controller traces with append-only JSONL storage.

The trace contract deliberately accepts unavailable runtime measurements as
``None``.  A fail-closed stop must remain recordable when, for example, a pose
lookup or scan summary failed; NaN and infinity are never used as sentinels in
the persisted artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_controller import VelocityCommand


CONTROLLER_TRACE_SCHEMA_VERSION = 2
_LEGACY_CONTROLLER_TRACE_SCHEMA_VERSION = 1

_CONTROLLER_TRACE_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "timestamp_sec",
        "event",
        "reason",
        "fail_closed",
        "route_revision",
        "route_kind",
        "target_index",
        "pursuit_index",
        "progress_mode",
        "egress_phase",
        "map_pose",
        "odom_pose",
        "active_segment_start_index",
        "active_segment_end_index",
        "distance_to_target_m",
        "pose_distance_to_segment_m",
        "maximum_chord_distance_to_segment_m",
        "tracking_tube_radius_m",
        "nominal_command",
        "effective_command",
        "front_clearance",
        "front_cluster_summary",
    }
)
_CONTROLLER_TRACE_FIELDS_V2 = _CONTROLLER_TRACE_FIELDS_V1 | {"diagnostics"}
_CONTROLLER_TRACE_FIELDS_BY_VERSION = {
    _LEGACY_CONTROLLER_TRACE_SCHEMA_VERSION: _CONTROLLER_TRACE_FIELDS_V1,
    CONTROLLER_TRACE_SCHEMA_VERSION: _CONTROLLER_TRACE_FIELDS_V2,
}


@dataclass(frozen=True)
class ControllerTraceRecord:
    """One control-cycle or fail-closed event suitable for JSONL evidence.

    All measurements other than the record timestamp are optional so a stop
    can be captured even when the failing sensor or transform supplied no
    usable numeric value.  Optional JSON summaries are defensively copied and
    recursively validated at construction time.
    """

    timestamp_sec: float
    event: str
    fail_closed: bool
    reason: str = ""
    route_revision: int | None = None
    route_kind: str = ""
    target_index: int | None = None
    pursuit_index: int | None = None
    progress_mode: str = ""
    egress_phase: str = ""
    map_pose: Pose2D | None = None
    odom_pose: Pose2D | None = None
    active_segment_start_index: int | None = None
    active_segment_end_index: int | None = None
    distance_to_target_m: float | None = None
    pose_distance_to_segment_m: float | None = None
    maximum_chord_distance_to_segment_m: float | None = None
    tracking_tube_radius_m: float | None = None
    nominal_command: VelocityCommand | None = None
    effective_command: VelocityCommand | None = None
    front_clearance: Mapping[str, object] | None = field(
        default=None,
        hash=False,
    )
    front_cluster_summary: Mapping[str, object] | None = field(
        default=None,
        hash=False,
    )
    schema_version: int = CONTROLLER_TRACE_SCHEMA_VERSION
    diagnostics: Mapping[str, object] | None = field(
        default=None,
        hash=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version not in _CONTROLLER_TRACE_FIELDS_BY_VERSION
        ):
            raise ValueError("unsupported controller trace schema_version")

        object.__setattr__(
            self,
            "timestamp_sec",
            _finite_number(self.timestamp_sec, "timestamp_sec"),
        )
        _require_string(self.event, "event", non_empty=True)
        _require_string(self.reason, "reason")
        if type(self.fail_closed) is not bool:
            raise ValueError("fail_closed must be boolean")

        object.__setattr__(
            self,
            "route_revision",
            _optional_non_negative_integer(self.route_revision, "route_revision"),
        )
        _require_string(self.route_kind, "route_kind")
        _require_string(self.progress_mode, "progress_mode")
        _require_string(self.egress_phase, "egress_phase")
        object.__setattr__(
            self,
            "target_index",
            _optional_non_negative_integer(self.target_index, "target_index"),
        )
        object.__setattr__(
            self,
            "pursuit_index",
            _optional_non_negative_integer(self.pursuit_index, "pursuit_index"),
        )

        object.__setattr__(self, "map_pose", _validated_pose(self.map_pose, "map_pose"))
        object.__setattr__(
            self,
            "odom_pose",
            _validated_pose(self.odom_pose, "odom_pose"),
        )

        segment_start = _optional_non_negative_integer(
            self.active_segment_start_index,
            "active_segment_start_index",
        )
        segment_end = _optional_non_negative_integer(
            self.active_segment_end_index,
            "active_segment_end_index",
        )
        if (segment_start is None) != (segment_end is None):
            raise ValueError("active segment indices must both be present or absent")
        if (
            segment_start is not None
            and segment_end is not None
            and segment_end < segment_start
        ):
            raise ValueError(
                "active_segment_end_index must not precede the start index"
            )
        object.__setattr__(self, "active_segment_start_index", segment_start)
        object.__setattr__(self, "active_segment_end_index", segment_end)

        for name in (
            "distance_to_target_m",
            "pose_distance_to_segment_m",
            "maximum_chord_distance_to_segment_m",
        ):
            object.__setattr__(
                self,
                name,
                _optional_non_negative_number(getattr(self, name), name),
            )
        tracking_tube_radius_m = _optional_non_negative_number(
            self.tracking_tube_radius_m,
            "tracking_tube_radius_m",
        )
        if tracking_tube_radius_m == 0.0:
            raise ValueError("tracking_tube_radius_m must be positive when present")
        object.__setattr__(
            self,
            "tracking_tube_radius_m",
            tracking_tube_radius_m,
        )

        object.__setattr__(
            self,
            "nominal_command",
            _validated_command(self.nominal_command, "nominal_command"),
        )
        object.__setattr__(
            self,
            "effective_command",
            _validated_command(self.effective_command, "effective_command"),
        )
        object.__setattr__(
            self,
            "front_clearance",
            _validated_json_object(self.front_clearance, "front_clearance"),
        )
        object.__setattr__(
            self,
            "front_cluster_summary",
            _validated_json_object(
                self.front_cluster_summary,
                "front_cluster_summary",
            ),
        )
        diagnostics = _validated_json_object(self.diagnostics, "diagnostics")
        if (
            self.schema_version == _LEGACY_CONTROLLER_TRACE_SCHEMA_VERSION
            and diagnostics is not None
        ):
            raise ValueError("schema-v1 controller traces cannot contain diagnostics")
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_payload(self) -> dict[str, object]:
        """Return a detached, JSON-safe representation of the record."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "timestamp_sec": self.timestamp_sec,
            "event": self.event,
            "reason": self.reason,
            "fail_closed": self.fail_closed,
            "route_revision": self.route_revision,
            "route_kind": self.route_kind,
            "target_index": self.target_index,
            "pursuit_index": self.pursuit_index,
            "progress_mode": self.progress_mode,
            "egress_phase": self.egress_phase,
            "map_pose": _pose_payload(self.map_pose),
            "odom_pose": _pose_payload(self.odom_pose),
            "active_segment_start_index": self.active_segment_start_index,
            "active_segment_end_index": self.active_segment_end_index,
            "distance_to_target_m": self.distance_to_target_m,
            "pose_distance_to_segment_m": self.pose_distance_to_segment_m,
            "maximum_chord_distance_to_segment_m": (
                self.maximum_chord_distance_to_segment_m
            ),
            "tracking_tube_radius_m": self.tracking_tube_radius_m,
            "nominal_command": _command_payload(self.nominal_command),
            "effective_command": _command_payload(self.effective_command),
            "front_clearance": _mutable_json_copy(self.front_clearance),
            "front_cluster_summary": _mutable_json_copy(
                self.front_cluster_summary
            ),
        }
        if self.schema_version == CONTROLLER_TRACE_SCHEMA_VERSION:
            payload["diagnostics"] = _mutable_json_copy(self.diagnostics)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ControllerTraceRecord":
        """Validate and construct one record from a decoded JSON object."""

        if not isinstance(payload, Mapping):
            raise ValueError("controller trace payload must be an object")
        if "schema_version" not in payload:
            raise ValueError(
                "controller trace fields mismatch: missing=['schema_version'], extra=[]"
            )
        schema_version = _integer(payload["schema_version"], "schema_version")
        expected_fields = _CONTROLLER_TRACE_FIELDS_BY_VERSION.get(schema_version)
        if expected_fields is None:
            raise ValueError("unsupported controller trace schema_version")
        if frozenset(payload) != expected_fields:
            missing = sorted(expected_fields - frozenset(payload))
            extra = sorted(frozenset(payload) - expected_fields)
            raise ValueError(
                "controller trace fields mismatch: "
                f"missing={missing!r}, extra={extra!r}"
            )
        return cls(
            schema_version=schema_version,
            timestamp_sec=_number(payload["timestamp_sec"], "timestamp_sec"),
            event=_string(payload["event"], "event"),
            reason=_string(payload["reason"], "reason"),
            fail_closed=_boolean(payload["fail_closed"], "fail_closed"),
            route_revision=_optional_integer(
                payload["route_revision"],
                "route_revision",
            ),
            route_kind=_string(payload["route_kind"], "route_kind"),
            target_index=_optional_integer(payload["target_index"], "target_index"),
            pursuit_index=_optional_integer(
                payload["pursuit_index"],
                "pursuit_index",
            ),
            progress_mode=_string(payload["progress_mode"], "progress_mode"),
            egress_phase=_string(payload["egress_phase"], "egress_phase"),
            map_pose=_pose_from_payload(payload["map_pose"], "map_pose"),
            odom_pose=_pose_from_payload(payload["odom_pose"], "odom_pose"),
            active_segment_start_index=_optional_integer(
                payload["active_segment_start_index"],
                "active_segment_start_index",
            ),
            active_segment_end_index=_optional_integer(
                payload["active_segment_end_index"],
                "active_segment_end_index",
            ),
            distance_to_target_m=_optional_number(
                payload["distance_to_target_m"],
                "distance_to_target_m",
            ),
            pose_distance_to_segment_m=_optional_number(
                payload["pose_distance_to_segment_m"],
                "pose_distance_to_segment_m",
            ),
            maximum_chord_distance_to_segment_m=_optional_number(
                payload["maximum_chord_distance_to_segment_m"],
                "maximum_chord_distance_to_segment_m",
            ),
            tracking_tube_radius_m=_optional_number(
                payload["tracking_tube_radius_m"],
                "tracking_tube_radius_m",
            ),
            nominal_command=_command_from_payload(
                payload["nominal_command"],
                "nominal_command",
            ),
            effective_command=_command_from_payload(
                payload["effective_command"],
                "effective_command",
            ),
            front_clearance=_json_object_from_payload(
                payload["front_clearance"],
                "front_clearance",
            ),
            front_cluster_summary=_json_object_from_payload(
                payload["front_cluster_summary"],
                "front_cluster_summary",
            ),
            diagnostics=(
                _json_object_from_payload(payload["diagnostics"], "diagnostics")
                if schema_version == CONTROLLER_TRACE_SCHEMA_VERSION
                else None
            ),
        )


@dataclass(frozen=True)
class ControllerTraceWriter:
    """Small path-bound facade over the append-only writer function."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def append(self, record: ControllerTraceRecord) -> None:
        append_controller_trace(self.path, record)


def append_controller_trace(path: Path, record: ControllerTraceRecord) -> None:
    """Append exactly one validated record without replacing existing bytes."""

    if not isinstance(record, ControllerTraceRecord):
        raise TypeError("record must be a ControllerTraceRecord")
    # Serialize before touching the filesystem so invalid evidence never
    # creates a partial artifact or directory tree.
    line = json.dumps(
        record.to_payload(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)


def load_controller_traces(path: Path) -> tuple[ControllerTraceRecord, ...]:
    """Load and validate all non-empty lines in a controller trace JSONL."""

    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("controller trace JSONL must be UTF-8") from exc

    records: list[ControllerTraceRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_strict_object_pairs,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(payload, Mapping):
                raise ValueError("line payload must be an object")
            records.append(ControllerTraceRecord.from_payload(payload))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid controller trace JSONL line {line_number}: {exc}"
            ) from exc
    return tuple(records)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_non_negative_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_non_negative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_string(value: object, name: str, *, non_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if non_empty and not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validated_pose(value: object, name: str) -> Pose2D | None:
    if value is None:
        return None
    if not isinstance(value, Pose2D):
        raise ValueError(f"{name} must be a Pose2D or None")
    return Pose2D(
        _finite_number(value.x_m, f"{name}.x_m"),
        _finite_number(value.y_m, f"{name}.y_m"),
        _finite_number(value.yaw_rad, f"{name}.yaw_rad"),
    )


def _validated_command(value: object, name: str) -> VelocityCommand | None:
    if value is None:
        return None
    if not isinstance(value, VelocityCommand):
        raise ValueError(f"{name} must be a VelocityCommand or None")
    return VelocityCommand(
        linear_x_mps=_finite_number(
            value.linear_x_mps,
            f"{name}.linear_x_mps",
        ),
        angular_z_radps=_finite_number(
            value.angular_z_radps,
            f"{name}.angular_z_radps",
        ),
    )


def _validated_json_object(
    value: object,
    name: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object or None")
    frozen = _freeze_json_value(value, name, set())
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json_value(value: object, name: str, active_ids: set[int]) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_ids:
            raise ValueError(f"{name} must not contain a reference cycle")
        active_ids.add(identity)
        try:
            copied: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{name} JSON object keys must be strings")
                copied[key] = _freeze_json_value(
                    item,
                    f"{name}.{key}",
                    active_ids,
                )
        finally:
            active_ids.remove(identity)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise ValueError(f"{name} must not contain a reference cycle")
        active_ids.add(identity)
        try:
            return tuple(
                _freeze_json_value(item, f"{name}[{index}]", active_ids)
                for index, item in enumerate(value)
            )
        finally:
            active_ids.remove(identity)
    raise ValueError(
        f"{name} contains non-JSON value of type {type(value).__name__}"
    )


def _mutable_json_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json_copy(item) for item in value]
    return value


def _pose_payload(pose: Pose2D | None) -> dict[str, float] | None:
    if pose is None:
        return None
    return {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad}


def _pose_from_payload(value: object, name: str) -> Pose2D | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object or null")
    expected = frozenset({"x_m", "y_m", "yaw_rad"})
    if frozenset(value) != expected:
        raise ValueError(f"{name} fields mismatch")
    return Pose2D(
        _number(value["x_m"], f"{name}.x_m"),
        _number(value["y_m"], f"{name}.y_m"),
        _number(value["yaw_rad"], f"{name}.yaw_rad"),
    )


def _command_payload(command: VelocityCommand | None) -> dict[str, float] | None:
    if command is None:
        return None
    return {
        "linear_x_mps": command.linear_x_mps,
        "angular_z_radps": command.angular_z_radps,
    }


def _command_from_payload(value: object, name: str) -> VelocityCommand | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object or null")
    expected = frozenset({"linear_x_mps", "angular_z_radps"})
    if frozenset(value) != expected:
        raise ValueError(f"{name} fields mismatch")
    return VelocityCommand(
        linear_x_mps=_number(value["linear_x_mps"], f"{name}.linear_x_mps"),
        angular_z_radps=_number(
            value["angular_z_radps"],
            f"{name}.angular_z_radps",
        ),
    )


def _json_object_from_payload(
    value: object,
    name: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object or null")
    return value


def _strict_object_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _number(value: object, name: str) -> float:
    return _finite_number(value, name)


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value
