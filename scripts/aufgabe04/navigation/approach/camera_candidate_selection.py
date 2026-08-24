"""Pure route-aware selection for post-LiDAR camera exploration.

The LiDAR admission phase decides *which* stand candidates camera exploration
may visit.  This module decides *which admitted candidate to visit next* from
already-computed route previews.  It intentionally has no ROS, filesystem, or
motion dependencies: planning and execution remain separate safety gates.

Selection first defers a route whose initial heading change crosses the
configured large-turn threshold.  Large in-place turns are a distinct risk for
localization continuity on the real robot, so this is a lexicographic tier and
not an arbitrary conversion to metres or seconds.  Within a tier, candidates
are ordered by estimated execution duration.  LiDAR evidence is only a stable
tie-breaker; in particular, single-view candidates remain selectable and are
not filtered out before camera validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from scripts.aufgabe04.navigation.approach.exact_two_camera_contract import (
    SUPPORT_CLASS_MULTI_VIEW,
    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
)


CAMERA_CANDIDATE_SELECTION_SCHEMA_VERSION = 1


class CameraCandidateSelectionError(ValueError):
    """Selection-contract failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class NoFeasibleCameraCandidateError(CameraCandidateSelectionError):
    """Raised before motion when no candidate has a feasible route preview."""

    def __init__(self, rejected_candidates: tuple["CameraCandidateRouteOption", ...]):
        super().__init__(
            "no_feasible_camera_candidate",
            "no admitted camera candidate has a feasible pre-approach route",
        )
        self.rejected_candidates = rejected_candidates

    def to_evidence(self) -> dict[str, object]:
        return {
            "schema_version": CAMERA_CANDIDATE_SELECTION_SCHEMA_VERSION,
            "error_code": self.code,
            "reason": str(self),
            "rejected_candidates": [
                option.to_dict() for option in self.rejected_candidates
            ],
            "motion_authorized": False,
        }


@dataclass(frozen=True)
class CameraCandidateSelectionConfig:
    """Robot-speed model and initial-turn risk boundary used for ranking."""

    linear_speed_mps: float
    angular_speed_radps: float
    large_initial_turn_threshold_rad: float = 3.0 * math.pi / 4.0

    def __post_init__(self) -> None:
        _positive_finite(self.linear_speed_mps, "linear_speed_mps")
        _positive_finite(self.angular_speed_radps, "angular_speed_radps")
        threshold = _positive_finite(
            self.large_initial_turn_threshold_rad,
            "large_initial_turn_threshold_rad",
        )
        if threshold > math.pi:
            raise CameraCandidateSelectionError(
                "invalid_config",
                "large_initial_turn_threshold_rad must not exceed pi",
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "linear_speed_mps": self.linear_speed_mps,
            "angular_speed_radps": self.angular_speed_radps,
            "large_initial_turn_threshold_rad": (
                self.large_initial_turn_threshold_rad
            ),
        }


@dataclass(frozen=True)
class CameraCandidateRouteOption:
    """One admitted candidate plus the result of its no-write route preview.

    Route metrics may be ``None`` only when the preview is infeasible.  A
    failed preview must carry a reason so the orchestrator can persist useful
    evidence without inventing route geometry.
    """

    candidate_uid: str
    feasible: bool
    failure_reason: str | None
    route_length_m: float | None
    turn_burden_rad: float | None
    initial_turn_rad: float | None
    inside_requested_standoff: bool
    support_class: str
    confidence: float
    hit_count: int

    def __post_init__(self) -> None:
        _nonempty_string(self.candidate_uid, "candidate_uid")
        _boolean(self.feasible, "feasible")
        _boolean(self.inside_requested_standoff, "inside_requested_standoff")
        _nonempty_string(self.support_class, "support_class")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise CameraCandidateSelectionError(
                "invalid_option", "confidence must be between zero and one"
            )
        if (
            isinstance(self.hit_count, bool)
            or not isinstance(self.hit_count, int)
            or self.hit_count < 0
        ):
            raise CameraCandidateSelectionError(
                "invalid_option", "hit_count must be a nonnegative integer"
            )

        if self.feasible:
            if self.failure_reason is not None:
                raise CameraCandidateSelectionError(
                    "invalid_option",
                    "a feasible route option cannot carry a failure_reason",
                )
            _nonnegative_finite(self.route_length_m, "route_length_m")
            _nonnegative_finite(self.turn_burden_rad, "turn_burden_rad")
            initial_turn = _finite(self.initial_turn_rad, "initial_turn_rad")
            if abs(initial_turn) > math.pi:
                raise CameraCandidateSelectionError(
                    "invalid_option", "initial_turn_rad must be normalized to [-pi, pi]"
                )
        else:
            _nonempty_string(self.failure_reason, "failure_reason")
            _optional_nonnegative_finite(self.route_length_m, "route_length_m")
            _optional_nonnegative_finite(self.turn_burden_rad, "turn_burden_rad")
            if self.initial_turn_rad is not None:
                initial_turn = _finite(self.initial_turn_rad, "initial_turn_rad")
                if abs(initial_turn) > math.pi:
                    raise CameraCandidateSelectionError(
                        "invalid_option",
                        "initial_turn_rad must be normalized to [-pi, pi]",
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "feasible": self.feasible,
            "failure_reason": self.failure_reason,
            "route_length_m": self.route_length_m,
            "turn_burden_rad": self.turn_burden_rad,
            "initial_turn_rad": self.initial_turn_rad,
            "inside_requested_standoff": self.inside_requested_standoff,
            "support_class": self.support_class,
            "confidence": self.confidence,
            "hit_count": self.hit_count,
        }


@dataclass(frozen=True)
class RankedCameraCandidate:
    """A feasible route option decorated with its deterministic rank evidence."""

    rank: int
    option: CameraCandidateRouteOption
    estimated_duration_sec: float
    large_initial_turn: bool
    risk_tier: int
    support_priority: int

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise CameraCandidateSelectionError(
                "invalid_selection", "rank must be a positive integer"
            )
        if not isinstance(self.option, CameraCandidateRouteOption) or not self.option.feasible:
            raise CameraCandidateSelectionError(
                "invalid_selection", "ranked candidates must contain feasible options"
            )
        _nonnegative_finite(self.estimated_duration_sec, "estimated_duration_sec")
        _boolean(self.large_initial_turn, "large_initial_turn")
        if self.risk_tier not in (0, 1):
            raise CameraCandidateSelectionError(
                "invalid_selection", "risk_tier must be zero or one"
            )
        if (
            isinstance(self.support_priority, bool)
            or not isinstance(self.support_priority, int)
            or self.support_priority < 0
        ):
            raise CameraCandidateSelectionError(
                "invalid_selection", "support_priority must be nonnegative"
            )

    @property
    def candidate_uid(self) -> str:
        return self.option.candidate_uid

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            **self.option.to_dict(),
            "estimated_duration_sec": self.estimated_duration_sec,
            "large_initial_turn": self.large_initial_turn,
            "risk_tier": self.risk_tier,
            "support_priority": self.support_priority,
        }


@dataclass(frozen=True)
class CameraCandidateSelection:
    """Motion-neutral result of ranking all route-previewed candidates."""

    selected_candidate_uid: str
    ranked_candidates: tuple[RankedCameraCandidate, ...]
    rejected_candidates: tuple[CameraCandidateRouteOption, ...]
    config: CameraCandidateSelectionConfig
    motion_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _nonempty_string(self.selected_candidate_uid, "selected_candidate_uid")
        if not isinstance(self.ranked_candidates, tuple) or not self.ranked_candidates:
            raise CameraCandidateSelectionError(
                "invalid_selection", "ranked_candidates must be a non-empty tuple"
            )
        if not isinstance(self.rejected_candidates, tuple):
            raise CameraCandidateSelectionError(
                "invalid_selection", "rejected_candidates must be a tuple"
            )
        if not isinstance(self.config, CameraCandidateSelectionConfig):
            raise CameraCandidateSelectionError(
                "invalid_selection", "config has the wrong type"
            )
        if any(
            not isinstance(candidate, RankedCameraCandidate)
            for candidate in self.ranked_candidates
        ):
            raise CameraCandidateSelectionError(
                "invalid_selection",
                "ranked_candidates contains an invalid row",
            )
        if any(
            not isinstance(candidate, CameraCandidateRouteOption)
            or candidate.feasible
            for candidate in self.rejected_candidates
        ):
            raise CameraCandidateSelectionError(
                "invalid_selection",
                "rejected_candidates must contain infeasible route options",
            )
        ranks = tuple(candidate.rank for candidate in self.ranked_candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise CameraCandidateSelectionError(
                "invalid_selection", "ranked candidate ranks must be contiguous"
            )
        ranked_uids = tuple(
            candidate.candidate_uid for candidate in self.ranked_candidates
        )
        rejected_uids = tuple(
            candidate.candidate_uid for candidate in self.rejected_candidates
        )
        if len(set(ranked_uids + rejected_uids)) != len(
            ranked_uids + rejected_uids
        ):
            raise CameraCandidateSelectionError(
                "invalid_selection", "candidate UIDs must be unique across the result"
            )
        if rejected_uids != tuple(sorted(rejected_uids)):
            raise CameraCandidateSelectionError(
                "invalid_selection", "rejected candidates must be sorted by UID"
            )
        if self.ranked_candidates[0].candidate_uid != self.selected_candidate_uid:
            raise CameraCandidateSelectionError(
                "invalid_selection", "selected_candidate_uid must match rank one"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAMERA_CANDIDATE_SELECTION_SCHEMA_VERSION,
            "selected_candidate_uid": self.selected_candidate_uid,
            "selection_policy": self.config.to_dict(),
            "ranked_candidates": [
                candidate.to_dict() for candidate in self.ranked_candidates
            ],
            "rejected_candidates": [
                candidate.to_dict() for candidate in self.rejected_candidates
            ],
            "motion_authorized": self.motion_authorized,
        }

    def to_evidence(self) -> dict[str, object]:
        """Return the JSON-ready, explicitly motion-neutral evidence payload."""

        return self.to_dict()


def select_camera_candidate(
    options: Iterable[CameraCandidateRouteOption],
    config: CameraCandidateSelectionConfig,
) -> CameraCandidateSelection:
    """Rank feasible route previews and select the safest efficient next visit.

    A route preview is evidence, not motion authority.  The returned result is
    therefore always ``motion_authorized=False``; route certification and the
    live execution gates remain the orchestrator's responsibility.
    """

    if not isinstance(config, CameraCandidateSelectionConfig):
        raise CameraCandidateSelectionError(
            "invalid_config", "config must be a CameraCandidateSelectionConfig"
        )
    try:
        frozen_options = tuple(options)
    except TypeError as exc:
        raise CameraCandidateSelectionError(
            "invalid_options", "options must be iterable"
        ) from exc
    if not frozen_options:
        raise CameraCandidateSelectionError(
            "invalid_options", "at least one camera candidate route option is required"
        )
    if any(not isinstance(option, CameraCandidateRouteOption) for option in frozen_options):
        raise CameraCandidateSelectionError(
            "invalid_options", "every option must be a CameraCandidateRouteOption"
        )
    candidate_uids = tuple(option.candidate_uid for option in frozen_options)
    if len(candidate_uids) != len(set(candidate_uids)):
        raise CameraCandidateSelectionError(
            "duplicate_candidate_uid", "camera candidate UIDs must be unique"
        )

    feasible = tuple(option for option in frozen_options if option.feasible)
    rejected = tuple(
        sorted(
            (option for option in frozen_options if not option.feasible),
            key=lambda option: option.candidate_uid,
        )
    )
    if not feasible:
        raise NoFeasibleCameraCandidateError(rejected)

    scored = tuple(_score(option, config) for option in feasible)
    ordered = tuple(sorted(scored, key=_rank_key))
    ranked = tuple(
        RankedCameraCandidate(
            rank=index,
            option=row.option,
            estimated_duration_sec=row.estimated_duration_sec,
            large_initial_turn=row.large_initial_turn,
            risk_tier=row.risk_tier,
            support_priority=row.support_priority,
        )
        for index, row in enumerate(ordered, start=1)
    )
    return CameraCandidateSelection(
        selected_candidate_uid=ranked[0].candidate_uid,
        ranked_candidates=ranked,
        rejected_candidates=rejected,
        config=config,
    )


@dataclass(frozen=True)
class _ScoredOption:
    option: CameraCandidateRouteOption
    estimated_duration_sec: float
    large_initial_turn: bool
    risk_tier: int
    support_priority: int


def _score(
    option: CameraCandidateRouteOption,
    config: CameraCandidateSelectionConfig,
) -> _ScoredOption:
    # Feasible-option validation guarantees these values are floats.
    assert option.route_length_m is not None
    assert option.turn_burden_rad is not None
    assert option.initial_turn_rad is not None
    estimated_duration_sec = (
        option.route_length_m / config.linear_speed_mps
        + option.turn_burden_rad / config.angular_speed_radps
    )
    large_initial_turn = (
        abs(option.initial_turn_rad) >= config.large_initial_turn_threshold_rad
    )
    return _ScoredOption(
        option=option,
        estimated_duration_sec=estimated_duration_sec,
        large_initial_turn=large_initial_turn,
        risk_tier=1 if large_initial_turn else 0,
        support_priority=_support_priority(option.support_class),
    )


def _rank_key(row: _ScoredOption) -> tuple[object, ...]:
    return (
        row.risk_tier,
        row.estimated_duration_sec,
        -row.support_priority,
        -row.option.confidence,
        -row.option.hit_count,
        row.option.candidate_uid,
    )


def _support_priority(support_class: str) -> int:
    if support_class == SUPPORT_CLASS_MULTI_VIEW:
        return 2
    if support_class == SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION:
        return 1
    return 0


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraCandidateSelectionError(
            "invalid_option", f"{field_name} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise CameraCandidateSelectionError(
            "invalid_option", f"{field_name} must be a finite number"
        )
    return number


def _positive_finite(value: object, field_name: str) -> float:
    try:
        number = _finite(value, field_name)
    except CameraCandidateSelectionError as exc:
        raise CameraCandidateSelectionError("invalid_config", str(exc)) from exc
    if number <= 0.0:
        raise CameraCandidateSelectionError(
            "invalid_config", f"{field_name} must be positive"
        )
    return number


def _nonnegative_finite(value: object, field_name: str) -> float:
    number = _finite(value, field_name)
    if number < 0.0:
        raise CameraCandidateSelectionError(
            "invalid_option", f"{field_name} must be nonnegative"
        )
    return number


def _optional_nonnegative_finite(value: object, field_name: str) -> None:
    if value is not None:
        _nonnegative_finite(value, field_name)


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraCandidateSelectionError(
            "invalid_option", f"{field_name} must be a non-empty string"
        )
    return value


def _boolean(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise CameraCandidateSelectionError(
            "invalid_option", f"{field_name} must be a boolean"
        )


__all__ = [
    "CAMERA_CANDIDATE_SELECTION_SCHEMA_VERSION",
    "CameraCandidateRouteOption",
    "CameraCandidateSelection",
    "CameraCandidateSelectionConfig",
    "CameraCandidateSelectionError",
    "NoFeasibleCameraCandidateError",
    "RankedCameraCandidate",
    "select_camera_candidate",
]
