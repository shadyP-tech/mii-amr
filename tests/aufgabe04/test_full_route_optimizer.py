import itertools
import math
import unittest

from scripts.aufgabe04.navigation.planning.full_route_optimizer import (
    ExactOptimizationLimitError,
    IncompleteCostMatrixError,
    IncompleteRouteInputError,
    UnreachableRouteError,
    optimize_full_route,
)


def complete_costs(start_id, arrivals_by_station, default=20.0):
    costs = {}
    all_arrivals = [
        arrival_id
        for arrival_ids in arrivals_by_station.values()
        for arrival_id in arrival_ids
    ]
    owners = {
        arrival_id: station_id
        for station_id, arrival_ids in arrivals_by_station.items()
        for arrival_id in arrival_ids
    }
    for arrival_id in all_arrivals:
        costs[(start_id, arrival_id)] = default
    for source_id in all_arrivals:
        for target_id in all_arrivals:
            if owners[source_id] != owners[target_id]:
                costs[(source_id, target_id)] = default
    return costs


def brute_force(start_id, arrivals_by_station, costs):
    best = None
    station_ids = sorted(arrivals_by_station)
    for station_order in itertools.permutations(station_ids):
        choices = [arrivals_by_station[station_id] for station_id in station_order]
        for arrival_order in itertools.product(*choices):
            path = tuple(zip(station_order, arrival_order))
            edges = tuple(zip((start_id,) + arrival_order[:-1], arrival_order))
            edge_costs = [costs[edge] for edge in edges]
            if any(cost is None or math.isinf(cost) for cost in edge_costs):
                continue
            candidate = (sum(edge_costs), path)
            if best is None or candidate < best:
                best = candidate
    return best


class FullRouteOptimizerTest(unittest.TestCase):
    def test_held_karp_matches_brute_force_and_jointly_selects_arrivals(self):
        arrivals = {
            "stand_a": ("a_front", "a_back"),
            "stand_b": ("b_front", "b_back"),
            "stand_c": ("c_front",),
        }
        costs = complete_costs("start", arrivals, default=50.0)
        costs.update(
            {
                ("start", "b_back"): 1.0,
                ("b_back", "a_front"): 2.0,
                ("a_front", "c_front"): 3.0,
                # Tempting first leg, but its onward route is deliberately bad.
                ("start", "a_back"): 0.5,
                ("a_back", "b_front"): 40.0,
                ("a_back", "b_back"): 40.0,
            }
        )

        expected = brute_force("start", arrivals, costs)
        plan = optimize_full_route(
            start_id="start",
            arrivals_by_station=arrivals,
            directed_costs=costs,
        )

        self.assertEqual(plan.total_cost, expected[0])
        self.assertEqual(
            tuple((visit.station_id, visit.arrival_id) for visit in plan.visits),
            expected[1],
        )
        self.assertEqual(plan.station_order, ("stand_b", "stand_a", "stand_c"))
        self.assertEqual(plan.arrival_order, ("b_back", "a_front", "c_front"))
        self.assertEqual(tuple(v.inbound_cost for v in plan.visits), (1.0, 2.0, 3.0))
        self.assertEqual(plan.algorithm, "exact_held_karp")
        self.assertTrue(plan.optimal)
        self.assertFalse(plan.fixed_station_order)

    def test_costs_are_directed(self):
        arrivals = {"a": ("a1",), "b": ("b1",)}
        costs = complete_costs("s", arrivals, default=99.0)
        costs.update(
            {
                ("s", "a1"): 1.0,
                ("s", "b1"): 1.0,
                ("a1", "b1"): 10.0,
                ("b1", "a1"): 2.0,
            }
        )

        plan = optimize_full_route(
            start_id="s",
            arrivals_by_station=arrivals,
            directed_costs=costs,
        )

        self.assertEqual(plan.station_order, ("b", "a"))
        self.assertEqual(plan.total_cost, 3.0)

    def test_fixed_order_is_preserved_while_arrival_choices_are_optimized(self):
        arrivals = {
            "a": ("a1", "a2"),
            "b": ("b1", "b2"),
        }
        # Only edges relevant to the fixed order are required.
        costs = {
            ("s", "a1"): 1.0,
            ("s", "a2"): 2.0,
            ("a1", "b1"): 20.0,
            ("a1", "b2"): 9.0,
            ("a2", "b1"): 1.0,
            ("a2", "b2"): 8.0,
        }

        plan = optimize_full_route(
            start_id="s",
            arrivals_by_station=arrivals,
            directed_costs=costs,
            fixed_station_order=("a", "b"),
        )

        self.assertEqual(plan.station_order, ("a", "b"))
        self.assertEqual(plan.arrival_order, ("a2", "b1"))
        self.assertEqual(plan.total_cost, 3.0)
        self.assertEqual(plan.algorithm, "exact_fixed_order_dynamic_programming")
        self.assertTrue(plan.fixed_station_order)

    def test_missing_cost_is_rejected_instead_of_assumed_unreachable(self):
        arrivals = {"a": ("a1",), "b": ("b1",)}
        costs = {
            ("s", "a1"): 1.0,
            ("s", "b1"): 1.0,
            ("a1", "b1"): 1.0,
            # b1 -> a1 was never computed.
        }

        with self.assertRaisesRegex(IncompleteCostMatrixError, "b1->a1"):
            optimize_full_route(
                start_id="s",
                arrivals_by_station=arrivals,
                directed_costs=costs,
            )

    def test_explicitly_unreachable_graph_is_rejected(self):
        arrivals = {"a": ("a1",), "b": ("b1",)}
        costs = {
            ("s", "a1"): 1.0,
            ("s", "b1"): 1.0,
            ("a1", "b1"): None,
            ("b1", "a1"): math.inf,
        }

        with self.assertRaises(UnreachableRouteError):
            optimize_full_route(
                start_id="s",
                arrivals_by_station=arrivals,
                directed_costs=costs,
            )

    def test_equal_cost_ties_are_deterministic(self):
        arrivals_one = {"b": ("b2", "b1"), "a": ("a2", "a1")}
        arrivals_two = {"a": ("a1", "a2"), "b": ("b1", "b2")}
        costs = complete_costs("s", arrivals_one, default=1.0)

        first = optimize_full_route(
            start_id="s",
            arrivals_by_station=arrivals_one,
            directed_costs=dict(reversed(tuple(costs.items()))),
        )
        second = optimize_full_route(
            start_id="s",
            arrivals_by_station=arrivals_two,
            directed_costs=costs,
        )

        self.assertEqual(first.station_order, ("a", "b"))
        self.assertEqual(first.arrival_order, ("a1", "b1"))
        self.assertEqual(first, second)

    def test_incomplete_candidate_and_fixed_order_inputs_are_rejected(self):
        with self.assertRaises(IncompleteRouteInputError):
            optimize_full_route(
                start_id="s",
                arrivals_by_station={"a": ()},
                directed_costs={},
            )

        arrivals = {"a": ("a1",), "b": ("b1",)}
        with self.assertRaises(IncompleteRouteInputError):
            optimize_full_route(
                start_id="s",
                arrivals_by_station=arrivals,
                directed_costs={},
                fixed_station_order=("a", "a"),
            )

    def test_exact_limit_fails_without_claiming_heuristic_optimality(self):
        arrivals = {"a": ("a1",), "b": ("b1",)}
        with self.assertRaisesRegex(ExactOptimizationLimitError, "no heuristic"):
            optimize_full_route(
                start_id="s",
                arrivals_by_station=arrivals,
                directed_costs={},
                exact_station_limit=1,
            )


if __name__ == "__main__":
    unittest.main()
