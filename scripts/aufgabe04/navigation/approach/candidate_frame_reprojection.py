"""Pure SE(2) reprojection for map-frame stand candidates.

A stand observation is frozen against the ``map <- odom`` transform that was
live when the observation was made.  If localization later changes that
transform, the old map coordinates no longer identify the same physical point.
This module preserves the observation-frame provenance and applies the only
safe conversion order::

    p_odom = inverse(T_map_from_odom_frozen) * p_map_frozen
    p_map_current = T_map_from_odom_current * p_odom

The implementation is deliberately ROS-free.  It owns neither transform
lookup nor motion; callers must supply both immutable transform values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    normalize_yaw,
)


CANDIDATE_FRAME_PROVENANCE_SCHEMA_VERSION = 1
CANDIDATE_FRAME_REPROJECTION_SCHEMA_VERSION = 1

_POINT_FIELDS = frozenset({"x_m", "y_m"})
_TRANSFORM_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "map_frame",
        "odom_frame",
        "canonical_odom_point",
        "frozen_map_point",
        "frozen_map_from_odom",
        "source_evidence_id",
    }
)
_DIAGNOSTIC_FIELDS = frozenset(
    {
        "candidate_map_displacement_x_m",
        "candidate_map_displacement_y_m",
        "candidate_map_displacement_m",
        "map_from_odom_translation_drift_m",
        "map_from_odom_absolute_yaw_drift_rad",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "provenance",
        "current_map_from_odom",
        "canonical_odom_point",
        "current_map_point",
        "diagnostics",
    }
)


class CandidateFrameReprojectionError(ValueError):
    """Candidate-frame contract failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CandidatePoint2D:
    """A finite planar point without an implied heading."""

    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _canonical_finite(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _canonical_finite(self.y_m, "y_m"))

    def to_mapping(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CandidatePoint2D":
        return _point_from_mapping(value, "point")


@dataclass(frozen=True)
class CandidateFrameProvenance:
    """Bind canonical odom geometry to its observation evidence.

    ``canonical_odom_point`` is authoritative and remains valid across AMCL
    ``map <- odom`` corrections.  A single-view observation can retain the
    complete frozen point/transform pair.  A fused multi-view candidate can
    instead bind the immutable artifact or record that produced its odom
    geometry through ``source_evidence_id``.
    """

    map_frame: str
    odom_frame: str
    canonical_odom_point: CandidatePoint2D
    frozen_map_point: CandidatePoint2D | None = None
    frozen_map_from_odom: PlanarTransform2D | None = None
    source_evidence_id: str | None = None
    schema_version: int = CANDIDATE_FRAME_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_provenance(self)

    def to_mapping(self) -> dict[str, object]:
        return candidate_frame_provenance_to_mapping(self)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "CandidateFrameProvenance":
        return candidate_frame_provenance_from_mapping(value)

    @classmethod
    def from_frozen_map_observation(
        cls,
        *,
        map_frame: str,
        odom_frame: str,
        frozen_map_point: CandidatePoint2D,
        frozen_map_from_odom: PlanarTransform2D,
        source_evidence_id: str | None = None,
    ) -> "CandidateFrameProvenance":
        """Construct canonical odom geometry from one frozen observation."""

        checked_point = _validated_point(frozen_map_point, "frozen_map_point")
        checked_transform = _validated_transform(
            frozen_map_from_odom, "frozen_map_from_odom"
        )
        return cls(
            map_frame=map_frame,
            odom_frame=odom_frame,
            canonical_odom_point=canonical_odom_point_from_frozen_map(
                checked_point, checked_transform
            ),
            frozen_map_point=checked_point,
            frozen_map_from_odom=checked_transform,
            source_evidence_id=source_evidence_id,
        )


@dataclass(frozen=True)
class CandidateFrameDriftDiagnostics:
    """Explain how localization drift moved the candidate in map coordinates."""

    candidate_map_displacement_x_m: float | None
    candidate_map_displacement_y_m: float | None
    candidate_map_displacement_m: float | None
    map_from_odom_translation_drift_m: float | None
    map_from_odom_absolute_yaw_drift_rad: float | None

    def __post_init__(self) -> None:
        available = tuple(
            getattr(self, name) is not None for name in _DIAGNOSTIC_FIELDS
        )
        if any(available) and not all(available):
            raise CandidateFrameReprojectionError(
                "invalid_diagnostics",
                "drift diagnostics must be entirely available or entirely unavailable",
            )
        for name in _DIAGNOSTIC_FIELDS:
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = _canonical_finite(raw_value, name)
            if name.endswith("_m") or name.endswith("_rad"):
                if name not in {
                    "candidate_map_displacement_x_m",
                    "candidate_map_displacement_y_m",
                } and value < 0.0:
                    raise CandidateFrameReprojectionError(
                        "invalid_diagnostics", f"{name} must be nonnegative"
                    )
            object.__setattr__(self, name, value)

    @property
    def frozen_reference_available(self) -> bool:
        return self.candidate_map_displacement_m is not None

    def to_mapping(self) -> dict[str, float | None]:
        return {
            "candidate_map_displacement_x_m": (
                self.candidate_map_displacement_x_m
            ),
            "candidate_map_displacement_y_m": (
                self.candidate_map_displacement_y_m
            ),
            "candidate_map_displacement_m": self.candidate_map_displacement_m,
            "map_from_odom_translation_drift_m": (
                self.map_from_odom_translation_drift_m
            ),
            "map_from_odom_absolute_yaw_drift_rad": (
                self.map_from_odom_absolute_yaw_drift_rad
            ),
        }


@dataclass(frozen=True)
class CandidateFrameReprojectionResult:
    """Canonical odom geometry and its equivalent in the current map frame."""

    provenance: CandidateFrameProvenance
    current_map_from_odom: PlanarTransform2D
    canonical_odom_point: CandidatePoint2D
    current_map_point: CandidatePoint2D
    diagnostics: CandidateFrameDriftDiagnostics
    schema_version: int = CANDIDATE_FRAME_REPROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_result(self)

    def to_mapping(self) -> dict[str, object]:
        return candidate_frame_reprojection_result_to_mapping(self)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "CandidateFrameReprojectionResult":
        return candidate_frame_reprojection_result_from_mapping(value)


def reproject_candidate_point(
    provenance: CandidateFrameProvenance,
    current_map_from_odom: PlanarTransform2D,
) -> CandidateFrameReprojectionResult:
    """Project authoritative odom geometry into the current map frame."""

    checked_provenance = _validated_provenance_copy(provenance)
    current_transform = _validated_transform(
        current_map_from_odom, "current_map_from_odom"
    )
    canonical_odom_point = checked_provenance.canonical_odom_point
    current_map_point = current_map_point_from_canonical_odom(
        canonical_odom_point, current_transform
    )
    diagnostics = _drift_diagnostics(
        checked_provenance, current_transform, current_map_point
    )
    return CandidateFrameReprojectionResult(
        provenance=checked_provenance,
        current_map_from_odom=current_transform,
        canonical_odom_point=canonical_odom_point,
        current_map_point=current_map_point,
        diagnostics=diagnostics,
    )


def canonical_odom_point_from_frozen_map(
    frozen_map_point: CandidatePoint2D,
    frozen_map_from_odom: PlanarTransform2D,
) -> CandidatePoint2D:
    """Return ``inverse(T0) * p_map_frozen`` with strict finite validation."""

    point = _validated_point(frozen_map_point, "frozen_map_point")
    transform = _validated_transform(
        frozen_map_from_odom, "frozen_map_from_odom"
    )
    return _map_point_to_odom(point, transform)


def current_map_point_from_canonical_odom(
    canonical_odom_point: CandidatePoint2D,
    current_map_from_odom: PlanarTransform2D,
) -> CandidatePoint2D:
    """Return ``T1 * p_odom`` for fused or single-view candidate geometry."""

    point = _validated_point(canonical_odom_point, "canonical_odom_point")
    transform = _validated_transform(
        current_map_from_odom, "current_map_from_odom"
    )
    return _odom_point_to_map(point, transform)


def candidate_frame_provenance_to_mapping(
    provenance: CandidateFrameProvenance,
) -> dict[str, object]:
    checked = _validated_provenance_copy(provenance)
    return {
        "schema_version": checked.schema_version,
        "map_frame": checked.map_frame,
        "odom_frame": checked.odom_frame,
        "canonical_odom_point": checked.canonical_odom_point.to_mapping(),
        "frozen_map_point": (
            None
            if checked.frozen_map_point is None
            else checked.frozen_map_point.to_mapping()
        ),
        "frozen_map_from_odom": (
            None
            if checked.frozen_map_from_odom is None
            else _transform_to_mapping(checked.frozen_map_from_odom)
        ),
        "source_evidence_id": checked.source_evidence_id,
    }


def candidate_frame_provenance_from_mapping(
    value: Mapping[str, object],
) -> CandidateFrameProvenance:
    payload = _strict_mapping(value, _PROVENANCE_FIELDS, "provenance")
    return CandidateFrameProvenance(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        map_frame=_frame_id(payload["map_frame"], "map_frame"),
        odom_frame=_frame_id(payload["odom_frame"], "odom_frame"),
        canonical_odom_point=_point_from_mapping(
            payload["canonical_odom_point"], "canonical_odom_point"
        ),
        frozen_map_point=_optional_point_from_mapping(
            payload["frozen_map_point"], "frozen_map_point"
        ),
        frozen_map_from_odom=_optional_transform_from_mapping(
            payload["frozen_map_from_odom"], "frozen_map_from_odom"
        ),
        source_evidence_id=_optional_evidence_id(
            payload["source_evidence_id"], "source_evidence_id"
        ),
    )


def candidate_frame_reprojection_result_to_mapping(
    result: CandidateFrameReprojectionResult,
) -> dict[str, object]:
    checked = _validated_result_copy(result)
    return {
        "schema_version": checked.schema_version,
        "provenance": checked.provenance.to_mapping(),
        "current_map_from_odom": _transform_to_mapping(
            checked.current_map_from_odom
        ),
        "canonical_odom_point": checked.canonical_odom_point.to_mapping(),
        "current_map_point": checked.current_map_point.to_mapping(),
        "diagnostics": checked.diagnostics.to_mapping(),
    }


def candidate_frame_reprojection_result_from_mapping(
    value: Mapping[str, object],
) -> CandidateFrameReprojectionResult:
    """Strictly load evidence and reject geometry inconsistent with provenance."""

    payload = _strict_mapping(value, _RESULT_FIELDS, "reprojection result")
    schema_version = _integer(payload["schema_version"], "schema_version")
    if schema_version != CANDIDATE_FRAME_REPROJECTION_SCHEMA_VERSION:
        raise CandidateFrameReprojectionError(
            "unsupported_schema", "unsupported candidate reprojection schema"
        )
    provenance_payload = payload["provenance"]
    if not isinstance(provenance_payload, Mapping):
        raise CandidateFrameReprojectionError(
            "invalid_mapping", "provenance must be an object"
        )
    provenance = candidate_frame_provenance_from_mapping(provenance_payload)
    current_transform = _transform_from_mapping(
        payload["current_map_from_odom"], "current_map_from_odom"
    )
    expected = reproject_candidate_point(provenance, current_transform)

    supplied_odom = _point_from_mapping(
        payload["canonical_odom_point"], "canonical_odom_point"
    )
    supplied_current = _point_from_mapping(
        payload["current_map_point"], "current_map_point"
    )
    supplied_diagnostics = _diagnostics_from_mapping(payload["diagnostics"])
    if (
        supplied_odom != expected.canonical_odom_point
        or supplied_current != expected.current_map_point
        or supplied_diagnostics != expected.diagnostics
    ):
        raise CandidateFrameReprojectionError(
            "inconsistent_reprojection",
            "reprojection geometry or diagnostics do not match the bound transforms",
        )
    return expected


def _map_point_to_odom(
    point_map: CandidatePoint2D,
    map_from_odom: PlanarTransform2D,
) -> CandidatePoint2D:
    delta_x = point_map.x_m - map_from_odom.x_m
    delta_y = point_map.y_m - map_from_odom.y_m
    cosine = math.cos(map_from_odom.yaw_rad)
    sine = math.sin(map_from_odom.yaw_rad)
    return CandidatePoint2D(
        x_m=cosine * delta_x + sine * delta_y,
        y_m=-sine * delta_x + cosine * delta_y,
    )


def _odom_point_to_map(
    point_odom: CandidatePoint2D,
    map_from_odom: PlanarTransform2D,
) -> CandidatePoint2D:
    cosine = math.cos(map_from_odom.yaw_rad)
    sine = math.sin(map_from_odom.yaw_rad)
    return CandidatePoint2D(
        x_m=(
            cosine * point_odom.x_m
            - sine * point_odom.y_m
            + map_from_odom.x_m
        ),
        y_m=(
            sine * point_odom.x_m
            + cosine * point_odom.y_m
            + map_from_odom.y_m
        ),
    )


def _validate_provenance(provenance: CandidateFrameProvenance) -> None:
    if (
        type(provenance.schema_version) is not int
        or provenance.schema_version
        != CANDIDATE_FRAME_PROVENANCE_SCHEMA_VERSION
    ):
        raise CandidateFrameReprojectionError(
            "unsupported_schema", "unsupported candidate frame provenance schema"
        )
    map_frame = _frame_id(provenance.map_frame, "map_frame")
    odom_frame = _frame_id(provenance.odom_frame, "odom_frame")
    if map_frame == odom_frame:
        raise CandidateFrameReprojectionError(
            "invalid_provenance", "map_frame and odom_frame must be distinct"
        )
    if not isinstance(provenance.frozen_map_point, CandidatePoint2D):
        if provenance.frozen_map_point is not None:
            raise CandidateFrameReprojectionError(
                "invalid_provenance",
                "frozen_map_point must be a CandidatePoint2D or None",
            )
    _validated_point(provenance.canonical_odom_point, "canonical_odom_point")
    frozen_pair = (
        provenance.frozen_map_point is not None,
        provenance.frozen_map_from_odom is not None,
    )
    if frozen_pair[0] != frozen_pair[1]:
        raise CandidateFrameReprojectionError(
            "incomplete_frozen_reference",
            "frozen_map_point and frozen_map_from_odom must appear together",
        )
    if provenance.source_evidence_id is not None:
        _evidence_id(provenance.source_evidence_id, "source_evidence_id")
    if not frozen_pair[0] and provenance.source_evidence_id is None:
        raise CandidateFrameReprojectionError(
            "missing_source_evidence",
            "canonical odom geometry requires a frozen reference or source_evidence_id",
        )
    if frozen_pair[0]:
        checked_transform = _validated_transform(
            provenance.frozen_map_from_odom, "frozen_map_from_odom"
        )
        derived = _map_point_to_odom(
            _validated_point(provenance.frozen_map_point, "frozen_map_point"),
            checked_transform,
        )
        if not _points_close(derived, provenance.canonical_odom_point):
            raise CandidateFrameReprojectionError(
                "inconsistent_canonical_odom_point",
                "canonical_odom_point does not match inverse(T0) * frozen_map_point",
            )


def _validated_provenance_copy(
    provenance: CandidateFrameProvenance,
) -> CandidateFrameProvenance:
    if not isinstance(provenance, CandidateFrameProvenance):
        raise CandidateFrameReprojectionError(
            "invalid_provenance",
            "provenance must be a CandidateFrameProvenance",
        )
    return CandidateFrameProvenance(
        schema_version=provenance.schema_version,
        map_frame=provenance.map_frame,
        odom_frame=provenance.odom_frame,
        canonical_odom_point=_validated_point(
            provenance.canonical_odom_point, "canonical_odom_point"
        ),
        frozen_map_point=(
            None
            if provenance.frozen_map_point is None
            else _validated_point(provenance.frozen_map_point, "frozen_map_point")
        ),
        frozen_map_from_odom=(
            None
            if provenance.frozen_map_from_odom is None
            else _validated_transform(
                provenance.frozen_map_from_odom, "frozen_map_from_odom"
            )
        ),
        source_evidence_id=provenance.source_evidence_id,
    )


def _validate_result(result: CandidateFrameReprojectionResult) -> None:
    if (
        type(result.schema_version) is not int
        or result.schema_version != CANDIDATE_FRAME_REPROJECTION_SCHEMA_VERSION
    ):
        raise CandidateFrameReprojectionError(
            "unsupported_schema", "unsupported candidate reprojection schema"
        )
    if not isinstance(result.provenance, CandidateFrameProvenance):
        raise CandidateFrameReprojectionError(
            "invalid_result", "provenance must be CandidateFrameProvenance"
        )
    _validate_provenance(result.provenance)
    _validated_transform(result.current_map_from_odom, "current_map_from_odom")
    for name in ("canonical_odom_point", "current_map_point"):
        if not isinstance(getattr(result, name), CandidatePoint2D):
            raise CandidateFrameReprojectionError(
                "invalid_result", f"{name} must be a CandidatePoint2D"
            )
    if not isinstance(result.diagnostics, CandidateFrameDriftDiagnostics):
        raise CandidateFrameReprojectionError(
            "invalid_result",
            "diagnostics must be CandidateFrameDriftDiagnostics",
        )
    canonical = _validated_point(
        result.provenance.canonical_odom_point, "provenance.canonical_odom_point"
    )
    if result.canonical_odom_point != canonical:
        raise CandidateFrameReprojectionError(
            "inconsistent_reprojection",
            "result canonical_odom_point differs from provenance",
        )
    current_transform = _validated_transform(
        result.current_map_from_odom, "current_map_from_odom"
    )
    expected_current = current_map_point_from_canonical_odom(
        canonical, current_transform
    )
    expected_diagnostics = _drift_diagnostics(
        result.provenance, current_transform, expected_current
    )
    if (
        result.current_map_point != expected_current
        or result.diagnostics != expected_diagnostics
    ):
        raise CandidateFrameReprojectionError(
            "inconsistent_reprojection",
            "result geometry or diagnostics do not match the bound transforms",
        )


def _validated_result_copy(
    result: CandidateFrameReprojectionResult,
) -> CandidateFrameReprojectionResult:
    if not isinstance(result, CandidateFrameReprojectionResult):
        raise CandidateFrameReprojectionError(
            "invalid_result",
            "result must be a CandidateFrameReprojectionResult",
        )
    expected = reproject_candidate_point(
        _validated_provenance_copy(result.provenance),
        _validated_transform(result.current_map_from_odom, "current_map_from_odom"),
    )
    if result != expected:
        raise CandidateFrameReprojectionError(
            "inconsistent_reprojection",
            "result geometry or diagnostics do not match the bound transforms",
        )
    return expected


def _point_from_mapping(value: object, name: str) -> CandidatePoint2D:
    payload = _strict_mapping(value, _POINT_FIELDS, name)
    return CandidatePoint2D(
        x_m=_finite(payload["x_m"], f"{name}.x_m"),
        y_m=_finite(payload["y_m"], f"{name}.y_m"),
    )


def _optional_point_from_mapping(
    value: object, name: str
) -> CandidatePoint2D | None:
    return None if value is None else _point_from_mapping(value, name)


def _transform_from_mapping(value: object, name: str) -> PlanarTransform2D:
    payload = _strict_mapping(value, _TRANSFORM_FIELDS, name)
    raw_yaw = _finite(payload["yaw_rad"], f"{name}.yaw_rad")
    if raw_yaw != normalize_yaw(raw_yaw):
        raise CandidateFrameReprojectionError(
            "invalid_transform", f"{name}.yaw_rad must be normalized"
        )
    try:
        return PlanarTransform2D(
            x_m=_finite(payload["x_m"], f"{name}.x_m"),
            y_m=_finite(payload["y_m"], f"{name}.y_m"),
            yaw_rad=raw_yaw,
        )
    except ValueError as exc:
        raise CandidateFrameReprojectionError(
            "invalid_transform", f"invalid {name}: {exc}"
        ) from exc


def _optional_transform_from_mapping(
    value: object, name: str
) -> PlanarTransform2D | None:
    return None if value is None else _transform_from_mapping(value, name)


def _transform_to_mapping(transform: PlanarTransform2D) -> dict[str, float]:
    checked = _validated_transform(transform, "map_from_odom")
    return {
        "x_m": checked.x_m,
        "y_m": checked.y_m,
        "yaw_rad": checked.yaw_rad,
    }


def _diagnostics_from_mapping(value: object) -> CandidateFrameDriftDiagnostics:
    payload = _strict_mapping(value, _DIAGNOSTIC_FIELDS, "diagnostics")
    return CandidateFrameDriftDiagnostics(
        **{
            name: (
                None
                if payload[name] is None
                else _finite(payload[name], f"diagnostics.{name}")
            )
            for name in _DIAGNOSTIC_FIELDS
        }
    )


def _drift_diagnostics(
    provenance: CandidateFrameProvenance,
    current_transform: PlanarTransform2D,
    current_map_point: CandidatePoint2D,
) -> CandidateFrameDriftDiagnostics:
    frozen_point = provenance.frozen_map_point
    frozen_transform = provenance.frozen_map_from_odom
    if frozen_point is None or frozen_transform is None:
        return CandidateFrameDriftDiagnostics(
            candidate_map_displacement_x_m=None,
            candidate_map_displacement_y_m=None,
            candidate_map_displacement_m=None,
            map_from_odom_translation_drift_m=None,
            map_from_odom_absolute_yaw_drift_rad=None,
        )
    displacement_x = _canonical_zero(current_map_point.x_m - frozen_point.x_m)
    displacement_y = _canonical_zero(current_map_point.y_m - frozen_point.y_m)
    translation_drift_x = current_transform.x_m - frozen_transform.x_m
    translation_drift_y = current_transform.y_m - frozen_transform.y_m
    yaw_drift = normalize_yaw(
        current_transform.yaw_rad - frozen_transform.yaw_rad
    )
    return CandidateFrameDriftDiagnostics(
        candidate_map_displacement_x_m=displacement_x,
        candidate_map_displacement_y_m=displacement_y,
        candidate_map_displacement_m=math.hypot(
            displacement_x, displacement_y
        ),
        map_from_odom_translation_drift_m=math.hypot(
            translation_drift_x, translation_drift_y
        ),
        map_from_odom_absolute_yaw_drift_rad=abs(yaw_drift),
    )


def _validated_point(point: object, name: str) -> CandidatePoint2D:
    if not isinstance(point, CandidatePoint2D):
        raise CandidateFrameReprojectionError(
            "invalid_point", f"{name} must be a CandidatePoint2D"
        )
    return CandidatePoint2D(point.x_m, point.y_m)


def _points_close(left: CandidatePoint2D, right: CandidatePoint2D) -> bool:
    return math.isclose(
        left.x_m, right.x_m, rel_tol=0.0, abs_tol=1e-12
    ) and math.isclose(
        left.y_m,
        right.y_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _validated_transform(
    transform: object, name: str
) -> PlanarTransform2D:
    if not isinstance(transform, PlanarTransform2D):
        raise CandidateFrameReprojectionError(
            "invalid_transform", f"{name} must be a PlanarTransform2D"
        )
    try:
        return PlanarTransform2D(transform.x_m, transform.y_m, transform.yaw_rad)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CandidateFrameReprojectionError(
            "invalid_transform", f"invalid {name}: {exc}"
        ) from exc


def _strict_mapping(
    value: object, expected_fields: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CandidateFrameReprojectionError(
            "invalid_mapping", f"{name} must be an object"
        )
    if frozenset(value) != expected_fields:
        raise CandidateFrameReprojectionError(
            "mapping_fields_mismatch", f"{name} fields mismatch"
        )
    return value


def _frame_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value.startswith("/")
        or any(character.isspace() for character in value)
    ):
        raise CandidateFrameReprojectionError(
            "invalid_frame", f"{name} must be an unprefixed frame id"
        )
    return value


def _evidence_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise CandidateFrameReprojectionError(
            "invalid_evidence_id", f"{name} must be a non-empty token"
        )
    return value


def _optional_evidence_id(value: object, name: str) -> str | None:
    return None if value is None else _evidence_id(value, name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateFrameReprojectionError(
            "invalid_mapping", f"{name} must be an integer"
        )
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateFrameReprojectionError(
            "invalid_number", f"{name} must be numeric"
        )
    number = float(value)
    if not math.isfinite(number):
        raise CandidateFrameReprojectionError(
            "invalid_number", f"{name} must be finite"
        )
    return number


def _canonical_finite(value: object, name: str) -> float:
    return _canonical_zero(_finite(value, name))


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value
