"""Pure policy for mixed LiDAR clear/invalid negative visibility rays.

Ray geometry belongs to ``coverage_candidate_reconciliation``.  This module
owns only the fail-closed interpretation of those classifications so the
thresholds, hard vetoes, and evidence summary remain independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


NEGATIVE_VISIBILITY_RAY_POLICY_SCHEMA_VERSION = 1
_EPSILON = 1.0e-12
_HARD_VETO_REASON_BY_CLASSIFICATION = {
    "blocked": "actual_static_line_of_sight_blocked",
    "out_of_range": "candidate_envelope_outside_conservative_range",
    "no_intersection": "no_scan_ray_intersects_candidate_envelope",
    "nearer": "nearer_return_occludes_candidate",
    "matching": "matching_return_supports_candidate",
}
_KNOWN_CLASSIFICATIONS = frozenset(
    {*_HARD_VETO_REASON_BY_CLASSIFICATION, "clear", "invalid"}
)


@dataclass(frozen=True)
class NegativeVisibilityRayPolicy:
    """Bounded dropout policy; it is deterministic, not sensor calibration."""

    minimum_distinct_clear_scan_count: int = 3
    minimum_clear_ray_fraction: float = 0.75
    maximum_invalid_selected_ray_fraction: float = 0.25

    def validated(self) -> "NegativeVisibilityRayPolicy":
        if (
            type(self.minimum_distinct_clear_scan_count) is not int
            or self.minimum_distinct_clear_scan_count < 2
        ):
            raise ValueError(
                "minimum_distinct_clear_scan_count must be an integer >= 2"
            )
        _fraction(self.minimum_clear_ray_fraction, "minimum_clear_ray_fraction")
        _fraction(
            self.maximum_invalid_selected_ray_fraction,
            "maximum_invalid_selected_ray_fraction",
        )
        return self

    def to_evidence_dict(self) -> dict[str, object]:
        self.validated()
        return {
            "schema_version": NEGATIVE_VISIBILITY_RAY_POLICY_SCHEMA_VERSION,
            "minimum_distinct_clear_scan_count": (
                self.minimum_distinct_clear_scan_count
            ),
            "minimum_clear_ray_fraction": self.minimum_clear_ray_fraction,
            "maximum_invalid_selected_ray_fraction": (
                self.maximum_invalid_selected_ray_fraction
            ),
            "invalid_rays_are_neutral_only_within_bounded_dropout": True,
            "matching_or_nearer_return_is_hard_veto": True,
        }


@dataclass(frozen=True)
class NegativeVisibilityRayPolicyDecision:
    """Finite evidence summary returned by the mixed-ray policy."""

    schema_version: int
    clear_ray_count: int
    invalid_selected_ray_count: int
    selected_ray_count: int
    clear_ray_fraction: float | None
    invalid_selected_ray_fraction: float | None
    distinct_clear_scan_count: int
    rejection_supported: bool
    reasons: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "clear_ray_count": self.clear_ray_count,
            "invalid_selected_ray_count": self.invalid_selected_ray_count,
            "selected_ray_count": self.selected_ray_count,
            "clear_ray_fraction": self.clear_ray_fraction,
            "invalid_selected_ray_fraction": (
                self.invalid_selected_ray_fraction
            ),
            "distinct_clear_scan_count": self.distinct_clear_scan_count,
            "rejection_supported": self.rejection_supported,
            "reasons": list(self.reasons),
        }


def evaluate_negative_visibility_ray_policy(
    classifications: Iterable[str],
    *,
    distinct_clear_scan_count: int,
    policy: NegativeVisibilityRayPolicy,
) -> NegativeVisibilityRayPolicyDecision:
    """Interpret ray classes without using candidate count or motion state."""

    policy.validated()
    if type(distinct_clear_scan_count) is not int or distinct_clear_scan_count < 0:
        raise ValueError("distinct_clear_scan_count must be a non-negative integer")
    items = tuple(str(value) for value in classifications)
    unknown = tuple(sorted(set(items) - _KNOWN_CLASSIFICATIONS))
    if unknown:
        raise ValueError(f"unknown visibility ray classifications: {unknown}")

    clear_count = items.count("clear")
    invalid_count = items.count("invalid")
    selected_count = clear_count + invalid_count
    clear_fraction = (
        None if selected_count == 0 else clear_count / selected_count
    )
    invalid_fraction = (
        None if selected_count == 0 else invalid_count / selected_count
    )
    reasons: list[str] = []
    for classification, reason in _HARD_VETO_REASON_BY_CLASSIFICATION.items():
        if classification in items:
            reasons.append(reason)
    if invalid_count and clear_count == 0:
        reasons.append("selected_scan_ray_invalid")
    if (
        clear_fraction is not None
        and clear_fraction + _EPSILON < policy.minimum_clear_ray_fraction
    ):
        reasons.append("insufficient_clear_ray_fraction")
    if (
        invalid_fraction is not None
        and invalid_fraction
        > policy.maximum_invalid_selected_ray_fraction + _EPSILON
    ):
        reasons.append("selected_scan_ray_invalid_fraction_exceeds_limit")
    if distinct_clear_scan_count < policy.minimum_distinct_clear_scan_count:
        reasons.append("insufficient_distinct_clear_scan_times")

    return NegativeVisibilityRayPolicyDecision(
        schema_version=NEGATIVE_VISIBILITY_RAY_POLICY_SCHEMA_VERSION,
        clear_ray_count=clear_count,
        invalid_selected_ray_count=invalid_count,
        selected_ray_count=selected_count,
        clear_ray_fraction=clear_fraction,
        invalid_selected_ray_fraction=invalid_fraction,
        distinct_clear_scan_count=distinct_clear_scan_count,
        rejection_supported=not reasons,
        reasons=tuple(reasons),
    )


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return parsed


__all__ = [
    "NEGATIVE_VISIBILITY_RAY_POLICY_SCHEMA_VERSION",
    "NegativeVisibilityRayPolicy",
    "NegativeVisibilityRayPolicyDecision",
    "evaluate_negative_visibility_ray_policy",
]
