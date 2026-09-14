"""Strict reconstruction of persisted continuity decisions for recovery.

Schema one retains the odom-origin metric. Schema two requires its fixed route
anchor and diagnostic origin displacement. Recomputing a record is not proof
of its origin: callers with authenticated authority also supply that context.
"""

from __future__ import annotations

import math
from typing import Mapping

from scripts.aufgabe04.navigation.localization.map_odom_drift_reference import (
    RouteDriftAnchor,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    normalize_yaw,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    MapOdomContinuityResult,
    OdomExecutionContext,
    evaluate_map_odom_continuity,
)


def _transform(value: object) -> PlanarTransform2D:
    if not isinstance(value, Mapping) or set(value) != {"x_m", "y_m", "yaw_rad"}:
        raise ValueError("continuity transform fields mismatch")
    for component in value.values():
        if (
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
        ):
            raise ValueError("continuity transform must be finite")
    if value["yaw_rad"] != normalize_yaw(value["yaw_rad"]):
        raise ValueError("continuity transform yaw must be normalized")
    return PlanarTransform2D(**value)


def _require_equal(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{name} fields mismatch")
        for key, item in expected.items():
            _require_equal(actual[key], item, f"{name}.{key}")
    elif isinstance(expected, float):
        if (
            isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise ValueError(f"{name} recomputation mismatch")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} recomputation mismatch")


def validate_continuity_evidence(
    value: object, *, context: OdomExecutionContext | None = None,
) -> MapOdomContinuityResult:
    if not isinstance(value, Mapping):
        raise ValueError("continuity evidence must be an object")
    version = value.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported continuity evidence schema")
    if version == 1 and (
        "drift_reference" in value or "origin_translation_drift_m" in value
    ):
        raise ValueError("legacy continuity cannot contain anchored evidence")
    try:
        reference = (
            RouteDriftAnchor.from_evidence(value["drift_reference"])
            if version == 2 else None
        )
        observed_context = OdomExecutionContext(
            map_frame=value["map_frame"], odom_frame=value["odom_frame"],
            base_frame=value["base_frame"],
            frozen_map_from_odom=_transform(value["frozen_map_from_odom"]),
            certificate_sha256=value["certificate_sha256"],
            max_map_from_odom_translation_drift_m=value["max_translation_drift_m"],
            max_map_from_odom_yaw_drift_rad=value["max_yaw_drift_rad"],
            drift_reference=reference,
        )
        if context is not None:
            if (
                not isinstance(context, OdomExecutionContext)
                or context != observed_context
            ):
                raise ValueError("continuity execution context mismatch")
        live = value["live_map_from_odom"]
        result = evaluate_map_odom_continuity(
            observed_context, None if live is None else _transform(live),
        )
        _require_equal(value, result.to_evidence(), "continuity")
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed continuity evidence: {exc}") from exc


__all__ = ["validate_continuity_evidence"]
