"""ROS-free clearance admission for route-localization uncertainty.

This module only accounts for whether a certified route segment has positive
clearance left after explicit geometric and uncertainty deductions.  It does
not generate commands, select steering, set speeds, or attach a probability
guarantee to a ``k``-sigma multiplier.

The clearance supplied by callers is the *raw centreline-to-obstacle*
clearance.  Robot radius and every other allowance therefore remain separate
evidence fields and are deducted exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import payload_sha256


UNCERTAINTY_BUDGET_EVIDENCE_SCHEMA_VERSION = 1
UNCERTAINTY_BUDGET_EXHAUSTED = "uncertainty_budget_exhausted"

_PSD_RELATIVE_TOLERANCE = 1.0e-12


class CovarianceValidationError(ValueError):
    """A planar covariance is unavailable, ambiguous, or not PSD."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PlanarCovariance:
    """Symmetric 2-D position covariance in square metres.

    Only the unique entries are accepted.  A caller with a full matrix must
    resolve its symmetry before constructing this value; silently choosing one
    of two cross-covariance entries would make the admission ambiguous.
    """

    xx_m2: float
    xy_m2: float
    yy_m2: float

    def __post_init__(self) -> None:
        xx = _strict_finite_number(self.xx_m2, "xx_m2")
        xy = _strict_finite_number(self.xy_m2, "xy_m2")
        yy = _strict_finite_number(self.yy_m2, "yy_m2")
        if xx < 0.0 or yy < 0.0:
            raise CovarianceValidationError(
                "covariance_not_psd",
                "planar covariance diagonal variances must be non-negative",
            )
        scale = max(xx, abs(xy), yy)
        if scale == 0.0:
            determinant = 0.0
            tolerance = 0.0
        else:
            scaled_xx = xx / scale
            scaled_xy = xy / scale
            scaled_yy = yy / scale
            determinant = scaled_xx * scaled_yy - scaled_xy * scaled_xy
            tolerance = _determinant_tolerance(
                scaled_xx, scaled_xy, scaled_yy
            )
        if determinant < -tolerance:
            raise CovarianceValidationError(
                "covariance_not_psd",
                "planar covariance must be positive semidefinite",
            )

        # Normalize numeric subclasses to ordinary floats so evidence encoding
        # is stable across callers.
        object.__setattr__(self, "xx_m2", xx)
        object.__setattr__(self, "xy_m2", xy)
        object.__setattr__(self, "yy_m2", yy)

    def to_evidence_dict(self) -> dict[str, float]:
        return {
            "xx_m2": self.xx_m2,
            "xy_m2": self.xy_m2,
            "yy_m2": self.yy_m2,
        }


@dataclass(frozen=True)
class RouteClearanceSegment:
    """All inputs for one route-segment clearance admission.

    ``segment_normal_x`` and ``segment_normal_y`` bind the projection geometry
    to the evidence.  The vector need not already be unit length; it is
    normalized before use.  At a corner, ``is_corner=True`` selects the largest
    covariance eigen-direction instead of treating either adjacent segment
    normal as sufficient.

    Numeric fields intentionally remain runtime-validated by the evaluator.
    This lets missing or non-finite data return a fail-closed decision rather
    than escape the admission path as an uncaught constructor exception.
    """

    segment_id: object
    raw_centerline_clearance_m: object
    robot_radius_m: object
    collision_margin_m: object
    fixed_odom_tracking_bound_m: object
    empirical_odom_drift_bound_m: object
    braking_latency_distance_m: object
    localization_sigma_multiplier: object
    heading_contribution_m: object
    covariance: object
    segment_normal_x: object
    segment_normal_y: object
    is_corner: object = False


@dataclass(frozen=True)
class UncertaintyBudgetDecision:
    """Fail-closed admission decision for one route segment."""

    segment_id: str | None
    accepted: bool
    reason: str
    remaining_margin_m: float | None
    evidence: dict[str, object]

    def to_evidence_dict(self) -> dict[str, object]:
        return dict(self.evidence)


@dataclass(frozen=True)
class RouteUncertaintyBudgetDecision:
    """Aggregate admission requiring every ordered segment to be accepted."""

    accepted: bool
    reason: str
    remaining_margin_m: float | None
    limiting_segment_id: str | None
    segment_decisions: tuple[UncertaintyBudgetDecision, ...]
    evidence: dict[str, object]

    def to_evidence_dict(self) -> dict[str, object]:
        return dict(self.evidence)


def validate_planar_covariance(value: object) -> PlanarCovariance:
    """Return a strict planar covariance from a value or canonical mapping.

    Mappings may use exactly ``xx_m2, xy_m2, yy_m2`` or exactly
    ``xx, xy, yy``.  Extra/full-matrix keys are rejected because they leave the
    chosen cross-covariance convention ambiguous.
    """

    if isinstance(value, PlanarCovariance):
        return value
    if value is None:
        raise CovarianceValidationError(
            "covariance_missing", "planar covariance is required"
        )
    if not isinstance(value, Mapping):
        raise CovarianceValidationError(
            "covariance_ambiguous",
            "planar covariance must be a PlanarCovariance or canonical mapping",
        )

    keys = set(value.keys())
    metric_keys = {"xx_m2", "xy_m2", "yy_m2"}
    short_keys = {"xx", "xy", "yy"}
    if keys == metric_keys:
        entries = (value["xx_m2"], value["xy_m2"], value["yy_m2"])
    elif keys == short_keys:
        entries = (value["xx"], value["xy"], value["yy"])
    else:
        raise CovarianceValidationError(
            "covariance_ambiguous",
            "covariance mapping must contain exactly one supported xx/xy/yy key set",
        )
    try:
        return PlanarCovariance(*entries)
    except CovarianceValidationError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise CovarianceValidationError(
            "covariance_nonfinite", "covariance entries must be finite numbers"
        ) from exc


def projected_sigma_m(
    covariance: PlanarCovariance | Mapping[str, object],
    normal_x: object,
    normal_y: object,
) -> float:
    """Project one-sigma position spread onto an arbitrary segment normal."""

    covariance = validate_planar_covariance(covariance)
    unit_x, unit_y = _unit_normal(normal_x, normal_y)
    scale = max(
        abs(covariance.xx_m2),
        abs(covariance.xy_m2),
        abs(covariance.yy_m2),
    )
    if scale == 0.0:
        return 0.0
    variance = (
        unit_x * unit_x * (covariance.xx_m2 / scale)
        + 2.0 * unit_x * unit_y * (covariance.xy_m2 / scale)
        + unit_y * unit_y * (covariance.yy_m2 / scale)
    )
    normalized_scale = max(
        abs(covariance.xx_m2 / scale),
        abs(covariance.xy_m2 / scale),
        abs(covariance.yy_m2 / scale),
    )
    tolerance = max(
        _PSD_RELATIVE_TOLERANCE * normalized_scale,
        16.0 * math.ulp(normalized_scale),
    )
    if variance < -tolerance:
        # This should only be reachable for a covariance accepted within the
        # determinant round-off tolerance.  It still fails instead of taking
        # sqrt(abs(variance)).
        raise CovarianceValidationError(
            "covariance_not_psd", "projected covariance variance is negative"
        )
    return math.sqrt(scale) * math.sqrt(max(0.0, variance))


def radial_sigma_m(
    covariance: PlanarCovariance | Mapping[str, object],
) -> float:
    """Return the largest-axis one-sigma spread for a corner/junction."""

    covariance = validate_planar_covariance(covariance)
    scale = max(
        abs(covariance.xx_m2),
        abs(covariance.xy_m2),
        abs(covariance.yy_m2),
    )
    if scale == 0.0:
        return 0.0
    scaled_xx = covariance.xx_m2 / scale
    scaled_xy = covariance.xy_m2 / scale
    scaled_yy = covariance.yy_m2 / scale
    largest_scaled_eigenvalue = 0.5 * (
        scaled_xx
        + scaled_yy
        + math.hypot(
            scaled_xx - scaled_yy,
            2.0 * scaled_xy,
        )
    )
    return math.sqrt(scale) * math.sqrt(max(0.0, largest_scaled_eigenvalue))


def evaluate_segment_uncertainty_budget(
    segment: RouteClearanceSegment | object,
) -> UncertaintyBudgetDecision:
    """Evaluate one segment; malformed evidence returns a rejected decision."""

    if not isinstance(segment, RouteClearanceSegment):
        return _invalid_segment_decision(
            validation_errors=("segment_input_missing_or_ambiguous",)
        )

    errors: list[str] = []
    segment_id = _segment_id_or_none(segment.segment_id)
    if segment_id is None:
        errors.append("segment_id_missing_or_ambiguous")

    numeric_names = (
        "raw_centerline_clearance_m",
        "robot_radius_m",
        "collision_margin_m",
        "fixed_odom_tracking_bound_m",
        "empirical_odom_drift_bound_m",
        "braking_latency_distance_m",
        "localization_sigma_multiplier",
        "heading_contribution_m",
    )
    numeric: dict[str, float | None] = {}
    for name in numeric_names:
        raw_value = getattr(segment, name)
        value = _finite_number_or_none(raw_value)
        numeric[name] = value
        if value is None:
            errors.append(f"{name}_missing_or_nonfinite")
        elif value < 0.0:
            errors.append(f"{name}_negative")

    normal_input = {
        "x": _finite_number_or_none(segment.segment_normal_x),
        "y": _finite_number_or_none(segment.segment_normal_y),
    }
    unit_normal: tuple[float, float] | None = None
    try:
        unit_normal = _unit_normal(
            segment.segment_normal_x, segment.segment_normal_y
        )
    except ValueError as exc:
        errors.append(str(exc))

    covariance: PlanarCovariance | None = None
    covariance_error: str | None = None
    try:
        covariance = validate_planar_covariance(segment.covariance)
    except CovarianceValidationError as exc:
        covariance_error = exc.code
        errors.append(exc.code)

    is_corner = segment.is_corner if isinstance(segment.is_corner, bool) else None
    if is_corner is None:
        errors.append("is_corner_missing_or_ambiguous")

    sigma_m: float | None = None
    projection_mode: str | None = None
    if covariance is not None and unit_normal is not None and is_corner is not None:
        if is_corner:
            projection_mode = "corner_worst_axis"
            sigma_m = radial_sigma_m(covariance)
        else:
            projection_mode = "segment_normal"
            sigma_m = projected_sigma_m(
                covariance, unit_normal[0], unit_normal[1]
            )

    errors = _deduplicated(errors)
    required_clearance_m: float | None = None
    localization_term_m: float | None = None
    remaining_margin_m: float | None = None
    if not errors and sigma_m is not None:
        multiplier = numeric["localization_sigma_multiplier"]
        assert multiplier is not None
        localization_term_m = multiplier * sigma_m
        if not math.isfinite(localization_term_m):
            errors.append("projected_localization_term_m_nonfinite")
            localization_term_m = None
        else:
            try:
                required_clearance_m = math.fsum(
                    (
                        _required_number(numeric, "robot_radius_m"),
                        _required_number(numeric, "collision_margin_m"),
                        _required_number(numeric, "fixed_odom_tracking_bound_m"),
                        _required_number(
                            numeric, "empirical_odom_drift_bound_m"
                        ),
                        _required_number(numeric, "braking_latency_distance_m"),
                        localization_term_m,
                        _required_number(numeric, "heading_contribution_m"),
                    )
                )
            except OverflowError:
                required_clearance_m = None
            if required_clearance_m is None or not math.isfinite(
                required_clearance_m
            ):
                errors.append("required_clearance_m_nonfinite")
                required_clearance_m = None
            else:
                remaining_margin_m = (
                    _required_number(numeric, "raw_centerline_clearance_m")
                    - required_clearance_m
                )
                if not math.isfinite(remaining_margin_m):
                    errors.append("remaining_margin_m_nonfinite")
                    remaining_margin_m = None

    errors = _deduplicated(errors)

    accepted = (
        not errors
        and remaining_margin_m is not None
        and remaining_margin_m > 0.0
    )
    reason = "" if accepted else UNCERTAINTY_BUDGET_EXHAUSTED
    evidence = _segment_evidence(
        segment_id=segment_id,
        accepted=accepted,
        reason=reason,
        remaining_margin_m=remaining_margin_m,
        numeric=numeric,
        covariance=covariance,
        covariance_error=covariance_error,
        normal_input=normal_input,
        unit_normal=unit_normal,
        is_corner=is_corner,
        projection_mode=projection_mode,
        sigma_m=sigma_m,
        localization_term_m=localization_term_m,
        required_clearance_m=required_clearance_m,
        validation_errors=errors,
    )
    return UncertaintyBudgetDecision(
        segment_id=segment_id,
        accepted=accepted,
        reason=reason,
        remaining_margin_m=remaining_margin_m,
        evidence=evidence,
    )


def evaluate_route_uncertainty_budget(
    segments: Sequence[RouteClearanceSegment] | object,
) -> RouteUncertaintyBudgetDecision:
    """Require positive remaining clearance on every ordered route segment."""

    route_errors: list[str] = []
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
        ordered_segments: tuple[object, ...] = ()
        route_errors.append("route_segments_missing_or_ambiguous")
    else:
        ordered_segments = tuple(segments)
        if not ordered_segments:
            route_errors.append("route_segments_missing_or_ambiguous")

    decisions = tuple(
        evaluate_segment_uncertainty_budget(segment) for segment in ordered_segments
    )
    valid_ids = [
        decision.segment_id
        for decision in decisions
        if decision.segment_id is not None
    ]
    if len(valid_ids) != len(set(valid_ids)):
        route_errors.append("duplicate_segment_id")

    unknown_margin = next(
        (decision for decision in decisions if decision.remaining_margin_m is None),
        None,
    )
    if unknown_margin is not None:
        remaining_margin_m = None
        limiting_segment_id = unknown_margin.segment_id
    elif decisions:
        limiting = min(
            enumerate(decisions),
            key=lambda item: (item[1].remaining_margin_m, item[0]),
        )[1]
        remaining_margin_m = limiting.remaining_margin_m
        limiting_segment_id = limiting.segment_id
    else:
        remaining_margin_m = None
        limiting_segment_id = None

    accepted = not route_errors and bool(decisions) and all(
        decision.accepted for decision in decisions
    )
    reason = "" if accepted else UNCERTAINTY_BUDGET_EXHAUSTED
    evidence: dict[str, object] = {
        "schema_version": UNCERTAINTY_BUDGET_EVIDENCE_SCHEMA_VERSION,
        "scope": _scope_evidence(),
        "acceptance_convention": "remaining_margin_m > 0",
        "decision": {
            "accepted": accepted,
            "reason": reason,
            "minimum_remaining_margin_m": remaining_margin_m,
            "limiting_segment_id": limiting_segment_id,
        },
        "validation": {
            "ok": not route_errors,
            "errors": _deduplicated(route_errors),
        },
        "segments": [
            {
                "route_index": index,
                **decision.to_evidence_dict(),
            }
            for index, decision in enumerate(decisions)
        ],
    }
    return RouteUncertaintyBudgetDecision(
        accepted=accepted,
        reason=reason,
        remaining_margin_m=remaining_margin_m,
        limiting_segment_id=limiting_segment_id,
        segment_decisions=decisions,
        evidence=evidence,
    )


def uncertainty_budget_evidence_sha256(
    value: (
        UncertaintyBudgetDecision
        | RouteUncertaintyBudgetDecision
        | Mapping[str, object]
    ),
) -> str:
    """Hash canonical finite-JSON evidence without adding a self-hash field."""

    if isinstance(value, (UncertaintyBudgetDecision, RouteUncertaintyBudgetDecision)):
        evidence = value.evidence
    elif isinstance(value, Mapping):
        evidence = value
    else:
        raise ValueError("uncertainty budget evidence must be a decision or mapping")
    return payload_sha256(evidence)


def _invalid_segment_decision(
    *, validation_errors: Sequence[str]
) -> UncertaintyBudgetDecision:
    reason = UNCERTAINTY_BUDGET_EXHAUSTED
    numeric = {
        "raw_centerline_clearance_m": None,
        "robot_radius_m": None,
        "collision_margin_m": None,
        "fixed_odom_tracking_bound_m": None,
        "empirical_odom_drift_bound_m": None,
        "braking_latency_distance_m": None,
        "localization_sigma_multiplier": None,
        "heading_contribution_m": None,
    }
    evidence = _segment_evidence(
        segment_id=None,
        accepted=False,
        reason=reason,
        remaining_margin_m=None,
        numeric=numeric,
        covariance=None,
        covariance_error="covariance_missing",
        normal_input={"x": None, "y": None},
        unit_normal=None,
        is_corner=None,
        projection_mode=None,
        sigma_m=None,
        localization_term_m=None,
        required_clearance_m=None,
        validation_errors=list(validation_errors),
    )
    return UncertaintyBudgetDecision(
        segment_id=None,
        accepted=False,
        reason=reason,
        remaining_margin_m=None,
        evidence=evidence,
    )


def _segment_evidence(
    *,
    segment_id: str | None,
    accepted: bool,
    reason: str,
    remaining_margin_m: float | None,
    numeric: Mapping[str, float | None],
    covariance: PlanarCovariance | None,
    covariance_error: str | None,
    normal_input: Mapping[str, float | None],
    unit_normal: tuple[float, float] | None,
    is_corner: bool | None,
    projection_mode: str | None,
    sigma_m: float | None,
    localization_term_m: float | None,
    required_clearance_m: float | None,
    validation_errors: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": UNCERTAINTY_BUDGET_EVIDENCE_SCHEMA_VERSION,
        "scope": _scope_evidence(),
        "segment_id": segment_id,
        "acceptance_convention": "remaining_margin_m > 0",
        "decision": {
            "accepted": accepted,
            "reason": reason,
            "remaining_margin_m": remaining_margin_m,
        },
        "geometry": {
            "is_corner": is_corner,
            "projection_mode": projection_mode,
            "segment_normal_input": dict(normal_input),
            "segment_normal_unit": (
                {"x": unit_normal[0], "y": unit_normal[1]}
                if unit_normal is not None
                else None
            ),
        },
        "covariance_m2": (
            covariance.to_evidence_dict() if covariance is not None else None
        ),
        "covariance_error": covariance_error,
        "localization": {
            "sigma_m": sigma_m,
            "sigma_mode": projection_mode,
            "sigma_multiplier": numeric["localization_sigma_multiplier"],
            "k_sigma_term_m": localization_term_m,
        },
        "budget_m": {
            "raw_centerline_clearance_m": numeric[
                "raw_centerline_clearance_m"
            ],
            "robot_radius_m": numeric["robot_radius_m"],
            "collision_margin_m": numeric["collision_margin_m"],
            "fixed_odom_tracking_bound_m": numeric[
                "fixed_odom_tracking_bound_m"
            ],
            "empirical_odom_drift_bound_m": numeric[
                "empirical_odom_drift_bound_m"
            ],
            "braking_latency_distance_m": numeric[
                "braking_latency_distance_m"
            ],
            "projected_localization_term_m": localization_term_m,
            "heading_contribution_m": numeric["heading_contribution_m"],
            "required_clearance_m": required_clearance_m,
            "remaining_margin_m": remaining_margin_m,
        },
        "validation": {
            "ok": not validation_errors,
            "errors": list(validation_errors),
        },
    }


def _scope_evidence() -> dict[str, bool]:
    return {
        "admission_only": True,
        "generates_commands": False,
        "probability_guarantee": False,
    }


def _strict_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise CovarianceValidationError(
            "covariance_nonfinite", f"{name} must be a finite number"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CovarianceValidationError(
            "covariance_nonfinite", f"{name} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise CovarianceValidationError(
            "covariance_nonfinite", f"{name} must be a finite number"
        )
    return result


def _finite_number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _unit_normal(normal_x: object, normal_y: object) -> tuple[float, float]:
    x = _finite_number_or_none(normal_x)
    y = _finite_number_or_none(normal_y)
    if x is None or y is None:
        raise ValueError("segment_normal_missing_or_nonfinite")
    magnitude = math.hypot(x, y)
    if magnitude <= 0.0:
        raise ValueError("segment_normal_zero")
    return x / magnitude, y / magnitude


def _determinant_tolerance(xx: float, xy: float, yy: float) -> float:
    scale = max(abs(xx * yy), xy * xy)
    return max(
        _PSD_RELATIVE_TOLERANCE * scale,
        16.0 * math.ulp(scale),
    )


def _segment_id_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _required_number(
    values: Mapping[str, float | None], name: str
) -> float:
    value = values[name]
    assert value is not None
    return value


def _deduplicated(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "CovarianceValidationError",
    "PlanarCovariance",
    "RouteClearanceSegment",
    "RouteUncertaintyBudgetDecision",
    "UNCERTAINTY_BUDGET_EVIDENCE_SCHEMA_VERSION",
    "UNCERTAINTY_BUDGET_EXHAUSTED",
    "UncertaintyBudgetDecision",
    "evaluate_route_uncertainty_budget",
    "evaluate_segment_uncertainty_budget",
    "projected_sigma_m",
    "radial_sigma_m",
    "uncertainty_budget_evidence_sha256",
    "validate_planar_covariance",
]
