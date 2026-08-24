"""Pure global route-order optimization for surveyed stand arrival poses.

The optimizer deliberately knows nothing about ROS, occupancy grids, or A*.
Its input is a set of eligible arrival-node IDs for every stand and a directed
pairwise cost table produced by a separate path-planning layer.

A cost-table entry has one of three meanings:

* a finite, non-negative number: the directed transition is traversable;
* ``None`` or positive infinity: the transition was computed and is
  unreachable;
* a missing key: the transition was not computed, which is rejected as an
  incomplete input rather than silently treated as unreachable.

For an unconstrained station order, :func:`optimize_full_route` uses exact
Held--Karp dynamic programming and jointly selects one arrival node per stand.
The result is an open path starting at ``start_id``; it does not implicitly
return to the start.  A fixed station order can also be supplied, in which case
dynamic programming still chooses the least-cost arrival node for each stand.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple


DirectedEdge = Tuple[str, str]
DirectedCosts = Mapping[DirectedEdge, Optional[float]]


class RouteOptimizationError(ValueError):
    """Base class for invalid or unsolvable route-optimization requests."""


class IncompleteRouteInputError(RouteOptimizationError):
    """Raised when stands, arrival choices, or a fixed order are incomplete."""


class IncompleteCostMatrixError(RouteOptimizationError):
    """Raised when a required directed transition has not been computed."""


class UnreachableRouteError(RouteOptimizationError):
    """Raised when no path can visit every requested stand exactly once."""


class ExactOptimizationLimitError(RouteOptimizationError):
    """Raised rather than silently substituting a non-optimal heuristic."""


@dataclass(frozen=True)
class OptimizedVisit:
    """One selected stand arrival and the cost paid to reach it."""

    station_id: str
    arrival_id: str
    inbound_cost: float


@dataclass(frozen=True)
class FullRoutePlan:
    """An exact, open route through one arrival node for every stand."""

    start_id: str
    visits: Tuple[OptimizedVisit, ...]
    total_cost: float
    algorithm: str
    optimal: bool
    fixed_station_order: bool

    @property
    def station_order(self) -> Tuple[str, ...]:
        return tuple(visit.station_id for visit in self.visits)

    @property
    def arrival_order(self) -> Tuple[str, ...]:
        return tuple(visit.arrival_id for visit in self.visits)


@dataclass(frozen=True)
class _PartialRoute:
    cost: float
    # Pairs are retained so equal-cost solutions have a stable, documented
    # station-then-arrival lexicographic tie-break independent of dict order.
    path: Tuple[Tuple[str, str], ...]
    edge_costs: Tuple[float, ...]


def optimize_full_route(
    *,
    start_id: str,
    arrivals_by_station: Mapping[str, Sequence[str]],
    directed_costs: DirectedCosts,
    fixed_station_order: Optional[Sequence[str]] = None,
    exact_station_limit: int = 12,
) -> FullRoutePlan:
    """Return the minimum-cost route through all requested stands.

    ``arrivals_by_station`` may contain multiple eligible arrival poses for a
    stand (for example, its two face normals).  Exactly one is selected.

    All transitions that the requested optimization may inspect must be
    present in ``directed_costs``.  Use ``None`` or ``math.inf`` to explicitly
    mark an A* transition as unreachable.  Missing keys raise
    :class:`IncompleteCostMatrixError`.

    The unconstrained algorithm is exponential in the number of stands.  If
    ``exact_station_limit`` is exceeded, this function fails explicitly; it
    never labels a heuristic route as optimal.
    """

    start, arrivals = _normalize_request(start_id, arrivals_by_station)
    costs = _normalize_costs(directed_costs)

    if fixed_station_order is not None:
        station_order = _validate_fixed_order(fixed_station_order, arrivals)
        _validate_cost_coverage(start, arrivals, costs, station_order)
        route = _optimize_fixed_order(start, station_order, arrivals, costs)
        algorithm = "exact_fixed_order_dynamic_programming"
        fixed = True
    else:
        if exact_station_limit < 1:
            raise IncompleteRouteInputError("exact_station_limit must be at least 1")
        if len(arrivals) > exact_station_limit:
            raise ExactOptimizationLimitError(
                "exact Held-Karp optimization requested for "
                f"{len(arrivals)} stands, exceeding exact_station_limit="
                f"{exact_station_limit}; no heuristic fallback was applied"
            )
        _validate_cost_coverage(start, arrivals, costs, None)
        route = _optimize_held_karp(start, arrivals, costs)
        algorithm = "exact_held_karp"
        fixed = False

    visits = tuple(
        OptimizedVisit(
            station_id=station_id,
            arrival_id=arrival_id,
            inbound_cost=edge_cost,
        )
        for (station_id, arrival_id), edge_cost in zip(
            route.path, route.edge_costs
        )
    )
    return FullRoutePlan(
        start_id=start,
        visits=visits,
        total_cost=route.cost,
        algorithm=algorithm,
        optimal=True,
        fixed_station_order=fixed,
    )


def _normalize_request(
    start_id: str,
    arrivals_by_station: Mapping[str, Sequence[str]],
) -> Tuple[str, Dict[str, Tuple[str, ...]]]:
    start = str(start_id).strip()
    if not start:
        raise IncompleteRouteInputError("start_id must be non-empty")
    if not arrivals_by_station:
        raise IncompleteRouteInputError("at least one stand is required")

    normalized: Dict[str, Tuple[str, ...]] = {}
    owners: Dict[str, str] = {}
    for raw_station_id, raw_arrivals in arrivals_by_station.items():
        station_id = str(raw_station_id).strip()
        if not station_id:
            raise IncompleteRouteInputError("station IDs must be non-empty")
        if station_id in normalized:
            # This is possible for custom Mapping implementations even though
            # ordinary dict keys are unique.
            raise IncompleteRouteInputError(
                f"duplicate station ID after normalization: {station_id!r}"
            )

        arrival_ids = tuple(
            sorted({str(arrival_id).strip() for arrival_id in raw_arrivals})
        )
        if not arrival_ids or any(not arrival_id for arrival_id in arrival_ids):
            raise IncompleteRouteInputError(
                f"station {station_id!r} has no complete eligible arrival IDs"
            )

        for arrival_id in arrival_ids:
            if arrival_id == start:
                raise IncompleteRouteInputError(
                    f"arrival ID {arrival_id!r} conflicts with start_id"
                )
            previous_owner = owners.get(arrival_id)
            if previous_owner is not None and previous_owner != station_id:
                raise IncompleteRouteInputError(
                    f"arrival ID {arrival_id!r} is shared by stations "
                    f"{previous_owner!r} and {station_id!r}"
                )
            owners[arrival_id] = station_id
        normalized[station_id] = arrival_ids

    return start, dict(sorted(normalized.items()))


def _normalize_costs(
    directed_costs: DirectedCosts,
) -> Dict[DirectedEdge, Optional[float]]:
    normalized: Dict[DirectedEdge, Optional[float]] = {}
    for raw_edge, raw_cost in directed_costs.items():
        if not isinstance(raw_edge, tuple) or len(raw_edge) != 2:
            raise IncompleteCostMatrixError(
                f"cost key must be a (source_id, target_id) pair: {raw_edge!r}"
            )
        source = str(raw_edge[0]).strip()
        target = str(raw_edge[1]).strip()
        if not source or not target:
            raise IncompleteCostMatrixError(
                f"cost edge IDs must be non-empty: {raw_edge!r}"
            )
        edge = (source, target)
        if edge in normalized:
            raise IncompleteCostMatrixError(
                f"duplicate cost edge after normalization: {edge!r}"
            )
        if raw_cost is None:
            normalized[edge] = None
            continue

        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise IncompleteCostMatrixError(
                f"cost for edge {edge!r} is not numeric: {raw_cost!r}"
            ) from exc
        if math.isnan(cost) or cost == -math.inf or cost < 0.0:
            raise IncompleteCostMatrixError(
                f"cost for edge {edge!r} must be non-negative or unreachable"
            )
        normalized[edge] = None if cost == math.inf else cost
    return normalized


def _validate_fixed_order(
    fixed_station_order: Sequence[str],
    arrivals: Mapping[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    order = tuple(str(station_id).strip() for station_id in fixed_station_order)
    if len(order) != len(arrivals):
        raise IncompleteRouteInputError(
            "fixed_station_order must contain every requested station exactly once"
        )
    if any(not station_id for station_id in order):
        raise IncompleteRouteInputError("fixed station IDs must be non-empty")
    if len(set(order)) != len(order) or set(order) != set(arrivals):
        raise IncompleteRouteInputError(
            "fixed_station_order must contain every requested station exactly once"
        )
    return order


def _validate_cost_coverage(
    start_id: str,
    arrivals: Mapping[str, Tuple[str, ...]],
    costs: Mapping[DirectedEdge, Optional[float]],
    fixed_order: Optional[Tuple[str, ...]],
) -> None:
    required = set()
    if fixed_order is None:
        for station_id, arrival_ids in arrivals.items():
            for arrival_id in arrival_ids:
                required.add((start_id, arrival_id))
            for other_station_id, other_arrival_ids in arrivals.items():
                if station_id == other_station_id:
                    continue
                for source_id in arrival_ids:
                    for target_id in other_arrival_ids:
                        required.add((source_id, target_id))
    else:
        first_station = fixed_order[0]
        for arrival_id in arrivals[first_station]:
            required.add((start_id, arrival_id))
        for source_station, target_station in zip(fixed_order, fixed_order[1:]):
            for source_id in arrivals[source_station]:
                for target_id in arrivals[target_station]:
                    required.add((source_id, target_id))

    missing = sorted(edge for edge in required if edge not in costs)
    if missing:
        preview = ", ".join(f"{source}->{target}" for source, target in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise IncompleteCostMatrixError(
            f"directed cost matrix is missing {len(missing)} required edge(s): "
            f"{preview}{suffix}; use None or infinity for computed-unreachable edges"
        )


def _edge_cost(
    costs: Mapping[DirectedEdge, Optional[float]],
    source_id: str,
    target_id: str,
) -> Optional[float]:
    # Coverage has already been validated.  Keeping this helper total makes the
    # dynamic-programming loops easier to audit.
    return costs[(source_id, target_id)]


def _is_better(candidate: _PartialRoute, incumbent: Optional[_PartialRoute]) -> bool:
    if incumbent is None:
        return True
    if candidate.cost != incumbent.cost:
        return candidate.cost < incumbent.cost
    return candidate.path < incumbent.path


def _optimize_fixed_order(
    start_id: str,
    station_order: Tuple[str, ...],
    arrivals: Mapping[str, Tuple[str, ...]],
    costs: Mapping[DirectedEdge, Optional[float]],
) -> _PartialRoute:
    routes: Dict[str, _PartialRoute] = {}
    first_station = station_order[0]
    for arrival_id in arrivals[first_station]:
        edge_cost = _edge_cost(costs, start_id, arrival_id)
        if edge_cost is None:
            continue
        routes[arrival_id] = _PartialRoute(
            cost=edge_cost,
            path=((first_station, arrival_id),),
            edge_costs=(edge_cost,),
        )

    for station_id in station_order[1:]:
        next_routes: Dict[str, _PartialRoute] = {}
        for arrival_id in arrivals[station_id]:
            best: Optional[_PartialRoute] = None
            for previous_arrival_id, previous in routes.items():
                edge_cost = _edge_cost(costs, previous_arrival_id, arrival_id)
                if edge_cost is None:
                    continue
                candidate = _PartialRoute(
                    cost=previous.cost + edge_cost,
                    path=previous.path + ((station_id, arrival_id),),
                    edge_costs=previous.edge_costs + (edge_cost,),
                )
                if _is_better(candidate, best):
                    best = candidate
            if best is not None:
                next_routes[arrival_id] = best
        routes = next_routes
        if not routes:
            break

    best_route: Optional[_PartialRoute] = None
    for route in routes.values():
        if _is_better(route, best_route):
            best_route = route
    if best_route is None:
        raise UnreachableRouteError(
            "no route can visit all stands in the requested fixed order"
        )
    return best_route


def _optimize_held_karp(
    start_id: str,
    arrivals: Mapping[str, Tuple[str, ...]],
    costs: Mapping[DirectedEdge, Optional[float]],
) -> _PartialRoute:
    station_ids = tuple(sorted(arrivals))
    station_index = {station_id: index for index, station_id in enumerate(station_ids)}
    # State: (visited-station mask, final arrival ID).  The station owning an
    # arrival is globally unique and can therefore be recovered without adding
    # it to the state key.
    routes: Dict[Tuple[int, str], _PartialRoute] = {}

    for station_id in station_ids:
        bit = 1 << station_index[station_id]
        for arrival_id in arrivals[station_id]:
            edge_cost = _edge_cost(costs, start_id, arrival_id)
            if edge_cost is None:
                continue
            routes[(bit, arrival_id)] = _PartialRoute(
                cost=edge_cost,
                path=((station_id, arrival_id),),
                edge_costs=(edge_cost,),
            )

    full_mask = (1 << len(station_ids)) - 1
    for mask in range(1, full_mask + 1):
        states_at_mask = sorted(
            (
                (last_arrival_id, route)
                for (state_mask, last_arrival_id), route in routes.items()
                if state_mask == mask
            ),
            key=lambda item: item[0],
        )
        if not states_at_mask:
            continue
        for last_arrival_id, route in states_at_mask:
            for next_station_id in station_ids:
                bit = 1 << station_index[next_station_id]
                if mask & bit:
                    continue
                next_mask = mask | bit
                for next_arrival_id in arrivals[next_station_id]:
                    edge_cost = _edge_cost(costs, last_arrival_id, next_arrival_id)
                    if edge_cost is None:
                        continue
                    candidate = _PartialRoute(
                        cost=route.cost + edge_cost,
                        path=route.path + ((next_station_id, next_arrival_id),),
                        edge_costs=route.edge_costs + (edge_cost,),
                    )
                    key = (next_mask, next_arrival_id)
                    if _is_better(candidate, routes.get(key)):
                        routes[key] = candidate

    best_route: Optional[_PartialRoute] = None
    for (mask, _arrival_id), route in routes.items():
        if mask == full_mask and _is_better(route, best_route):
            best_route = route
    if best_route is None:
        raise UnreachableRouteError(
            "no directed route can visit every requested stand exactly once"
        )
    return best_route


__all__ = [
    "DirectedCosts",
    "ExactOptimizationLimitError",
    "FullRoutePlan",
    "IncompleteCostMatrixError",
    "IncompleteRouteInputError",
    "OptimizedVisit",
    "RouteOptimizationError",
    "UnreachableRouteError",
    "optimize_full_route",
]
