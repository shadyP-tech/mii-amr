"""Pure uncertainty-aware selection across immutable route options.

Every option is evaluated by :func:`evaluate_route_uncertainty_admission`
before ranking.  Selection therefore cannot turn a rejected route into an
executable one, and it never rewrites a caller's route.  The result is
motion-neutral evidence only; the existing execution certificates and live
runtime gates remain responsible for motion authority.

Ranking is deterministic and safety preserving: admitted routes always
precede rejected routes.  Admitted routes are partitioned into descending,
non-overlapping clearance-margin bands; margins within the configured band
tolerance are treated as tied, then shorter route length, explicit planner
order, and stable option identity decide.  When no route is admitted the
returned decision is explicitly fail closed and carries no selected route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    RouteUncertaintyAdmissionResult,
    evaluate_route_uncertainty_admission,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap


ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION = 2

# One tenth of a millimetre is deliberately much larger than the observed
# micrometre-scale numerical variation while preserving millimetre-scale
# residual-clearance distinctions.  The tie policy never turns a rejected
# route into an accepted one.
DEFAULT_ACCEPTED_MARGIN_TIE_TOLERANCE_M = 0.0001

NO_ROUTE_OPTIONS = "no_route_options"
NO_ACCEPTED_ROUTE_OPTIONS = "no_route_option_passed_uncertainty_admission"

ROUTE_UNCERTAINTY_SELECTION_RANKING_ORDER = (
    "accepted_first",
    "accepted_margin_tie_band_descending",
    "route_length_m_ascending_within_accepted_margin_tie_band",
    "plan_order_ascending",
    "option_id_ascending",
)


@dataclass(frozen=True)
class RouteUncertaintySelectionPolicy:
    """Validated deterministic policy for ranking independently safe routes.

    The tolerance applies only after exact route uncertainty admission.  It
    cannot admit a rejected option or rank a rejected option above an accepted
    option.  Accepted options are grouped from the largest remaining margin;
    every margin no more than the inclusive tolerance below that group maximum
    is tied and resolved by the documented non-safety fallback keys.
    """

    accepted_margin_tie_tolerance_m: float = (
        DEFAULT_ACCEPTED_MARGIN_TIE_TOLERANCE_M
    )

    def __post_init__(self) -> None:
        value = self.accepted_margin_tie_tolerance_m
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                "accepted_margin_tie_tolerance_m must be finite and "
                "non-negative"
            )
        object.__setattr__(
            self,
            "accepted_margin_tie_tolerance_m",
            float(value),
        )

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "accepted_precedes_rejected": True,
            "accepted_margin_tie_tolerance_m": (
                self.accepted_margin_tie_tolerance_m
            ),
            "accepted_margin_tie_boundary": "inclusive",
            "accepted_margin_tie_grouping": (
                "descending_nonoverlapping_bands_anchored_at_group_maximum"
            ),
            "accepted_margin_tie_fallback_order": [
                "route_length_m_ascending",
                "plan_order_ascending",
                "option_id_ascending",
            ],
            "rejected_option_order_unchanged": True,
        }


DEFAULT_ROUTE_UNCERTAINTY_SELECTION_POLICY = (
    RouteUncertaintySelectionPolicy()
)


@dataclass(frozen=True)
class RouteUncertaintySelectionOption:
    """One stable planning option and its unchanged map-frame route.

    ``plan_order`` is explicit so replay does not depend on mapping or set
    iteration order.  Route geometry is intentionally left to the exact
    admission evaluator: malformed, non-finite, or zero-length source routes
    become rejected evidence instead of being repaired here.
    """

    option_id: str
    plan_order: int
    map_route: tuple[Pose2D, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.option_id, str) or not self.option_id.strip():
            raise ValueError("option_id must be a non-empty string")
        if type(self.plan_order) is not int or self.plan_order < 0:
            raise ValueError("plan_order must be a non-negative integer")
        if not isinstance(self.map_route, tuple):
            raise TypeError("map_route must be an immutable tuple")


@dataclass(frozen=True)
class RankedRouteUncertaintyOption:
    """One exactly evaluated option decorated with deterministic rank."""

    rank: int
    option: RouteUncertaintySelectionOption
    route_length_m: float | None
    admission: RouteUncertaintyAdmissionResult
    admission_evidence_sha256: str

    @property
    def accepted(self) -> bool:
        return self.admission.decision.accepted

    @property
    def minimum_remaining_margin_m(self) -> float | None:
        return self.admission.decision.remaining_margin_m

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "option_id": self.option.option_id,
            "plan_order": self.option.plan_order,
            "accepted": self.accepted,
            "reason": self.admission.decision.reason,
            "minimum_remaining_margin_m": self.minimum_remaining_margin_m,
            "limiting_segment_id": (
                self.admission.decision.limiting_segment_id
            ),
            "route_length_m": self.route_length_m,
            "admission_evidence_sha256": self.admission_evidence_sha256,
            "admission_evidence": self.admission.to_evidence_dict(),
        }


@dataclass(frozen=True)
class RouteUncertaintySelectionDecision:
    """Motion-neutral route ranking or an explicit fail-closed no-selection."""

    ready: bool
    reason: str
    selected_option_id: str | None
    ranked_options: tuple[RankedRouteUncertaintyOption, ...]
    evidence: dict[str, object]
    motion_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.ready) is not bool:
            raise TypeError("ready must be boolean")
        if not isinstance(self.ranked_options, tuple):
            raise TypeError("ranked_options must be a tuple")
        if any(
            not isinstance(item, RankedRouteUncertaintyOption)
            for item in self.ranked_options
        ):
            raise TypeError("ranked_options contains an invalid item")
        ranks = tuple(item.rank for item in self.ranked_options)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("ranked option ranks must be contiguous")
        option_ids = tuple(item.option.option_id for item in self.ranked_options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("ranked option IDs must be unique")

        if self.ready:
            if self.reason:
                raise ValueError("ready selection cannot carry a failure reason")
            if not self.ranked_options or not self.ranked_options[0].accepted:
                raise ValueError("ready selection must rank an admitted route first")
            if self.selected_option_id != self.ranked_options[0].option.option_id:
                raise ValueError("selected option must match rank one")
        elif (
            not self.reason
            or self.selected_option_id is not None
            or any(item.accepted for item in self.ranked_options)
        ):
            raise ValueError(
                "not-ready selection must fail closed without an admitted route"
            )

    @property
    def selected_option(self) -> RouteUncertaintySelectionOption | None:
        if not self.ready:
            return None
        return self.ranked_options[0].option

    @property
    def selected_route(self) -> tuple[Pose2D, ...] | None:
        option = self.selected_option
        return None if option is None else option.map_route

    def to_evidence_dict(self) -> dict[str, object]:
        return dict(self.evidence)


@dataclass(frozen=True)
class _EvaluatedOption:
    option: RouteUncertaintySelectionOption
    route_length_m: float | None
    admission: RouteUncertaintyAdmissionResult
    admission_evidence_sha256: str


@dataclass(frozen=True)
class _AcceptedMarginTieBand:
    band: int
    maximum_margin_m: float
    minimum_margin_m: float
    ordered_option_ids: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "band": self.band,
            "maximum_margin_m": self.maximum_margin_m,
            "minimum_margin_m": self.minimum_margin_m,
            "ordered_option_ids": list(self.ordered_option_ids),
        }


def evaluate_route_uncertainty_selection(
    costmap: Costmap,
    options: Iterable[RouteUncertaintySelectionOption],
    covariance: PlanarCovariance,
    config: RouteUncertaintyAdmissionConfig,
    *,
    selection_policy: RouteUncertaintySelectionPolicy = (
        DEFAULT_ROUTE_UNCERTAINTY_SELECTION_POLICY
    ),
) -> RouteUncertaintySelectionDecision:
    """Evaluate every route exactly and select only from admitted options.

    An empty option set and an all-rejected set both return durable,
    motion-neutral fail-closed evidence.  Ambiguous option identity is a
    caller contract error because no deterministic evidence partition can be
    constructed for duplicate IDs.
    """

    if not isinstance(selection_policy, RouteUncertaintySelectionPolicy):
        raise TypeError(
            "selection_policy must be a RouteUncertaintySelectionPolicy"
        )

    try:
        frozen_options = tuple(options)
    except TypeError as exc:
        raise ValueError("route uncertainty options must be iterable") from exc
    if any(
        not isinstance(option, RouteUncertaintySelectionOption)
        for option in frozen_options
    ):
        raise TypeError(
            "every route uncertainty option must be a "
            "RouteUncertaintySelectionOption"
        )
    option_ids = tuple(option.option_id for option in frozen_options)
    if len(option_ids) != len(set(option_ids)):
        raise ValueError("route uncertainty option IDs must be unique")

    evaluated = tuple(
        _evaluate_option(costmap, option, covariance, config)
        for option in frozen_options
    )
    ordered, accepted_margin_tie_bands = _order_evaluated_options(
        evaluated,
        selection_policy,
    )
    ranked = tuple(
        RankedRouteUncertaintyOption(
            rank=rank,
            option=item.option,
            route_length_m=item.route_length_m,
            admission=item.admission,
            admission_evidence_sha256=item.admission_evidence_sha256,
        )
        for rank, item in enumerate(ordered, start=1)
    )

    accepted = tuple(item for item in ranked if item.accepted)
    if accepted:
        ready = True
        reason = ""
        selected_option_id = accepted[0].option.option_id
    else:
        ready = False
        reason = NO_ROUTE_OPTIONS if not ranked else NO_ACCEPTED_ROUTE_OPTIONS
        selected_option_id = None

    evidence: dict[str, object] = {
        "schema_version": ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION,
        "scope": {
            "selection_only": True,
            "exact_route_uncertainty_admission_required": True,
            "generates_commands": False,
            "mutates_route": False,
        },
        "ranking_order": list(ROUTE_UNCERTAINTY_SELECTION_RANKING_ORDER),
        "ranking_policy": selection_policy.to_evidence_dict(),
        "accepted_margin_tie_bands": [
            band.to_evidence_dict() for band in accepted_margin_tie_bands
        ],
        "decision": {
            "ready": ready,
            "reason": reason,
            "selected_option_id": selected_option_id,
            "fail_closed": not ready,
        },
        "options": [item.to_evidence_dict() for item in ranked],
        "motion_authorized": False,
    }
    return RouteUncertaintySelectionDecision(
        ready=ready,
        reason=reason,
        selected_option_id=selected_option_id,
        ranked_options=ranked,
        evidence=evidence,
    )


def route_uncertainty_selection_evidence_sha256(
    value: RouteUncertaintySelectionDecision | Mapping[str, object],
) -> str:
    """Hash complete finite selection evidence canonically."""

    if isinstance(value, RouteUncertaintySelectionDecision):
        evidence = value.evidence
    elif isinstance(value, Mapping):
        evidence = value
    else:
        raise ValueError("selection evidence must be a decision or mapping")
    return payload_sha256(evidence)


def _evaluate_option(
    costmap: Costmap,
    option: RouteUncertaintySelectionOption,
    covariance: PlanarCovariance,
    config: RouteUncertaintyAdmissionConfig,
) -> _EvaluatedOption:
    admission = evaluate_route_uncertainty_admission(
        costmap,
        option.map_route,
        covariance,
        config,
    )
    return _EvaluatedOption(
        option=option,
        route_length_m=_route_length_or_none(option.map_route),
        admission=admission,
        admission_evidence_sha256=(
            route_uncertainty_admission_evidence_sha256(admission)
        ),
    )


def _ranking_key(item: _EvaluatedOption) -> tuple[object, ...]:
    margin = item.admission.decision.remaining_margin_m
    margin_key = -margin if margin is not None and math.isfinite(margin) else math.inf
    length_key = (
        item.route_length_m
        if item.route_length_m is not None and math.isfinite(item.route_length_m)
        else math.inf
    )
    return (
        0 if item.admission.decision.accepted else 1,
        margin_key,
        length_key,
        item.option.plan_order,
        item.option.option_id,
    )


def _order_evaluated_options(
    evaluated: tuple[_EvaluatedOption, ...],
    policy: RouteUncertaintySelectionPolicy,
) -> tuple[tuple[_EvaluatedOption, ...], tuple[_AcceptedMarginTieBand, ...]]:
    """Return a total order without a non-transitive pairwise comparator.

    A naive ``abs(left_margin - right_margin) <= tolerance`` comparator is not
    transitive for chains of three nearby margins.  Anchoring each descending
    band at its maximum produces deterministic evidence independent of input
    iteration order while retaining the intended inclusive tie boundary.
    """

    accepted = tuple(
        item for item in evaluated if item.admission.decision.accepted
    )
    rejected = tuple(
        sorted(
            (
                item
                for item in evaluated
                if not item.admission.decision.accepted
            ),
            key=_ranking_key,
        )
    )

    finite_margin = sorted(
        (
            item
            for item in accepted
            if _finite_remaining_margin(item) is not None
        ),
        key=_accepted_exact_margin_key,
    )
    nonfinite_margin = tuple(
        sorted(
            (
                item
                for item in accepted
                if _finite_remaining_margin(item) is None
            ),
            key=_accepted_fallback_key,
        )
    )

    ordered_accepted: list[_EvaluatedOption] = []
    bands: list[_AcceptedMarginTieBand] = []
    remaining = finite_margin
    tolerance_m = policy.accepted_margin_tie_tolerance_m
    while remaining:
        group_maximum_m = _finite_remaining_margin(remaining[0])
        assert group_maximum_m is not None
        tied: list[_EvaluatedOption] = []
        outside: list[_EvaluatedOption] = []
        for item in remaining:
            margin_m = _finite_remaining_margin(item)
            assert margin_m is not None
            if group_maximum_m - margin_m <= tolerance_m:
                tied.append(item)
            else:
                outside.append(item)
        tied.sort(key=_accepted_fallback_key)
        band_number = len(bands) + 1
        band_minimum_m = min(
            margin
            for item in tied
            if (margin := _finite_remaining_margin(item)) is not None
        )
        bands.append(
            _AcceptedMarginTieBand(
                band=band_number,
                maximum_margin_m=group_maximum_m,
                minimum_margin_m=band_minimum_m,
                ordered_option_ids=tuple(
                    item.option.option_id for item in tied
                ),
            )
        )
        ordered_accepted.extend(tied)
        remaining = outside

    ordered_accepted.extend(nonfinite_margin)
    return tuple(ordered_accepted) + rejected, tuple(bands)


def _finite_remaining_margin(item: _EvaluatedOption) -> float | None:
    margin = item.admission.decision.remaining_margin_m
    return margin if margin is not None and math.isfinite(margin) else None


def _accepted_exact_margin_key(item: _EvaluatedOption) -> tuple[object, ...]:
    margin = _finite_remaining_margin(item)
    assert margin is not None
    return (-margin, *_accepted_fallback_key(item))


def _accepted_fallback_key(item: _EvaluatedOption) -> tuple[object, ...]:
    length = item.route_length_m
    length_key = (
        length
        if length is not None and math.isfinite(length)
        else math.inf
    )
    return (
        length_key,
        item.option.plan_order,
        item.option.option_id,
    )


def _route_length_or_none(route: tuple[Pose2D, ...]) -> float | None:
    if len(route) < 2:
        return None
    lengths: list[float] = []
    for start, end in zip(route, route[1:]):
        if not isinstance(start, Pose2D) or not isinstance(end, Pose2D):
            return None
        length_m = math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
        if not math.isfinite(length_m):
            return None
        lengths.append(length_m)
    try:
        result = math.fsum(lengths)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


__all__ = [
    "DEFAULT_ACCEPTED_MARGIN_TIE_TOLERANCE_M",
    "DEFAULT_ROUTE_UNCERTAINTY_SELECTION_POLICY",
    "NO_ACCEPTED_ROUTE_OPTIONS",
    "NO_ROUTE_OPTIONS",
    "ROUTE_UNCERTAINTY_SELECTION_RANKING_ORDER",
    "ROUTE_UNCERTAINTY_SELECTION_SCHEMA_VERSION",
    "RankedRouteUncertaintyOption",
    "RouteUncertaintySelectionPolicy",
    "RouteUncertaintySelectionDecision",
    "RouteUncertaintySelectionOption",
    "evaluate_route_uncertainty_selection",
    "route_uncertainty_selection_evidence_sha256",
]
