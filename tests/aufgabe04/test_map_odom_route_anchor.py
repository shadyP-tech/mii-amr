"""Route-anchor regression evidence independent of ignored run recordings.

The three recorded cases are counterfactual metric comparisons, not newly
issued certificates or permission to resume the original robot run.
"""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig, _heading_contribution_for_points,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.map_odom_drift_reference import RouteDriftAnchor
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D, normalize_yaw,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext, evaluate_map_odom_continuity,
    evaluate_map_odom_stationary_stability, validate_map_odom_continuity_evidence,
)


_FIXTURE = Path(__file__).with_name("fixtures") / "map_odom_route_anchor_20260911T153146Z.json"
_CASES = json.loads(_FIXTURE.read_text())["cases"]


def _apply(transform, point):
    cosine, sine = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    return (
        cosine * point[0] - sine * point[1] + transform.x_m,
        sine * point[0] + cosine * point[1] + transform.y_m,
    )


def _rebase(transform, *, origin, angle):
    """q_old = R(angle) q_new + origin; compose the same physical map action."""
    x_m, y_m = _apply(transform, origin)
    return PlanarTransform2D(x_m, y_m, normalize_yaw(transform.yaw_rad + angle))


def _point_pose(point):
    # Route-start yaw is unspecified in the recorded CSV. These checks concern
    # position only, so choose zero solely for the pose-conversion API.
    return Pose2D(point["x_m"], point["y_m"], point["yaw_rad"] or 0.0)


def _context(case, *, anchored=True):
    evidence = case["recorded_continuity"]
    frozen = PlanarTransform2D(**evidence["frozen_map_from_odom"])
    return OdomExecutionContext(
        map_frame=evidence["map_frame"], odom_frame=evidence["odom_frame"],
        base_frame=evidence["base_frame"], certificate_sha256=evidence["certificate_sha256"],
        frozen_map_from_odom=frozen,
        max_map_from_odom_translation_drift_m=evidence["max_translation_drift_m"],
        max_map_from_odom_yaw_drift_rad=evidence["max_yaw_drift_rad"],
        drift_reference=(RouteDriftAnchor.from_route_start(
            Pose2D(**case["map_route"][0]), frozen,
        ) if anchored else None),
    )


def _simple_context():
    return OdomExecutionContext(
        map_frame="map", odom_frame="odom", base_frame="base_footprint",
        certificate_sha256="a" * 64,
        frozen_map_from_odom=PlanarTransform2D(-4.0, 0.0, 0.0),
        max_map_from_odom_translation_drift_m=0.125,
        max_map_from_odom_yaw_drift_rad=0.10,
        drift_reference=RouteDriftAnchor(0.0, 0.0, 4.0, 0.0),
    )


class MapOdomRouteAnchorTest(unittest.TestCase):
    def test_all_three_recorded_false_stops_change_only_the_metric(self):
        self.assertEqual(len(_CASES), 3)
        for case in _CASES:
            with self.subTest(run=case["run_suffix"]):
                live = PlanarTransform2D(**case["recorded_continuity"]["live_map_from_odom"])
                legacy = evaluate_map_odom_continuity(_context(case, anchored=False), live)
                self.assertEqual(legacy.to_evidence(), case["recorded_continuity"])
                self.assertFalse(legacy.accepted)
                context = _context(case)
                result = evaluate_map_odom_continuity(context, live)
                self.assertTrue(result.accepted, result.reason)
                self.assertAlmostEqual(result.translation_drift_m, case["expected_anchor_displacement_m"], places=12)
                self.assertEqual(result.max_translation_drift_m, legacy.max_translation_drift_m)
                self.assertEqual(result.max_yaw_drift_rad, legacy.max_yaw_drift_rad)
                self.assertEqual(result.to_evidence()["schema_version"], 2)
                self.assertAlmostEqual(result.to_evidence()["origin_translation_drift_m"], legacy.translation_drift_m)
                self.assertEqual(validate_map_odom_continuity_evidence(result.to_evidence(), context=context), result)

    def test_arbitrary_common_odom_translation_and_rotation_preserve_decisions(self):
        for case in _CASES:
            context = _context(case)
            live = PlanarTransform2D(**case["recorded_continuity"]["live_map_from_odom"])
            baseline = evaluate_map_odom_continuity(context, live)
            for origin, angle in (((100.0, -70.0), 2.4), ((0.7, -3.0), -2.8), ((4.5, 2.3), 0.9)):
                with self.subTest(run=case["run_suffix"], origin=origin, angle=angle):
                    frozen_new = _rebase(context.frozen_map_from_odom, origin=origin, angle=angle)
                    live_new = _rebase(live, origin=origin, angle=angle)
                    rebased = replace(context, frozen_map_from_odom=frozen_new,
                                      drift_reference=RouteDriftAnchor.from_route_start(Pose2D(**case["map_route"][0]), frozen_new))
                    result = evaluate_map_odom_continuity(rebased, live_new)
                    self.assertEqual(result.accepted, baseline.accepted)
                    self.assertEqual(result.reason, baseline.reason)
                    self.assertAlmostEqual(result.translation_drift_m, baseline.translation_drift_m, places=10)
                    self.assertAlmostEqual(result.absolute_yaw_drift_rad, baseline.absolute_yaw_drift_rad, places=12)
                    for pose in case["map_route"]:
                        old_point = context.map_pose_to_odom(_point_pose(pose))
                        new_point = rebased.map_pose_to_odom(_point_pose(pose))
                        world_old = _apply(live, (old_point.x_m, old_point.y_m))
                        world_new = _apply(live_new, (new_point.x_m, new_point.y_m))
                        self.assertLess(math.dist(world_old, world_new), 1e-10)

    def test_pure_translation_accepts_equality_and_rejects_next_representable_value(self):
        context = _simple_context()
        at_limit = PlanarTransform2D(-3.875, 0.0, 0.0)
        above = PlanarTransform2D(math.nextafter(-3.875, math.inf), 0.0, 0.0)
        self.assertTrue(evaluate_map_odom_continuity(context, at_limit).accepted)
        result = evaluate_map_odom_continuity(context, above)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "map_from_odom_translation_drift")

    def test_yaw_about_the_fixed_anchor_still_requires_reseal(self):
        context = _simple_context()
        yaw = 0.20
        live = PlanarTransform2D(-4.0 * math.cos(yaw), -4.0 * math.sin(yaw), yaw)
        result = evaluate_map_odom_continuity(context, live)
        self.assertAlmostEqual(result.translation_drift_m, 0.0, places=12)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "map_from_odom_yaw_drift")
        self.assertTrue(result.requires_zero_cycle)

    def test_recorded_remaining_route_and_footprint_fit_the_fixed_anchor_bound(self):
        for case in _CASES:
            context = _context(case)
            live = PlanarTransform2D(**case["recorded_continuity"]["live_map_from_odom"])
            result = evaluate_map_odom_continuity(context, live)
            anchor = context.drift_reference
            radius = case["robot_radius_m"]
            poses = tuple(_point_pose(point) for point in case["map_route"])
            config = RouteUncertaintyAdmissionConfig(
                robot_radius_m=radius, collision_margin_m=0.02,
                fixed_odom_tracking_bound_m=0.03, empirical_odom_drift_bound_m=0.02,
                braking_latency_distance_m=0.015, localization_sigma_multiplier=2.0,
                heading_sigma_rad=result.max_yaw_drift_rad / 2.0,
                heading_lever_arm_m=radius, sampling_spacing_m=0.005,
                heading_reference_x_m=anchor.map_x_m, heading_reference_y_m=anchor.map_y_m,
            )
            lever = max(math.hypot(p.x_m - anchor.map_x_m, p.y_m - anchor.map_y_m) for p in poses) + radius
            yaw_budget = _heading_contribution_for_points(poses, config)
            self.assertAlmostEqual(yaw_budget, result.max_yaw_drift_rad * lever)
            observed_bound = result.translation_drift_m + result.absolute_yaw_drift_rad * lever
            self.assertLessEqual(observed_bound, result.max_translation_drift_m + yaw_budget)
            vertex_shifts = []
            # Endpoints bound each straight centerline segment; sample the full
            # footprint circle to independently exercise the added radius arm.
            for pose in poses:
                point = context.map_pose_to_odom(pose)
                center = (point.x_m, point.y_m)
                vertex_shifts.append(math.dist(_apply(live, center), _apply(context.frozen_map_from_odom, center)))
                for index in range(64):
                    angle = index * 2.0 * math.pi / 64
                    support = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
                    displacement = math.dist(_apply(live, support), _apply(context.frozen_map_from_odom, support))
                    self.assertLessEqual(displacement, observed_bound + 1e-12)
            self.assertAlmostEqual(max(vertex_shifts), case["expected_maximum_vertex_displacement_m"], places=12)

    def test_stationary_window_uses_the_same_fixed_anchor_and_final_transform(self):
        for case in _CASES:
            context = _context(case)
            live = PlanarTransform2D(**case["recorded_continuity"]["live_map_from_odom"])
            samples = (live, context.frozen_map_from_odom)
            limits = dict(max_translation_drift_m=context.max_map_from_odom_translation_drift_m,
                          max_yaw_drift_rad=context.max_map_from_odom_yaw_drift_rad)
            legacy = evaluate_map_odom_stationary_stability(samples, **limits)
            anchored = evaluate_map_odom_stationary_stability(samples, **limits, drift_reference=context.drift_reference)
            self.assertFalse(legacy.accepted)
            self.assertTrue(anchored.accepted)
            self.assertEqual(anchored.frozen_map_from_odom, context.frozen_map_from_odom)
            self.assertAlmostEqual(anchored.max_observed_translation_drift_m, case["expected_anchor_displacement_m"], places=12)
            self.assertEqual(anchored.to_evidence()["schema_version"], 2)
            self.assertEqual(anchored.to_evidence()["drift_reference"], context.drift_reference.to_evidence())

    def test_anchor_rejects_malformed_nonfinite_boolean_and_wrong_transform_inputs(self):
        valid = _simple_context().drift_reference.to_evidence()
        invalid = (None, {}, {**valid, "metric": "origin"}, {**valid, "extra": 1},
                   {**valid, "map_anchor": {"x_m": 0.0}}, {**valid, "odom_anchor": []})
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RouteDriftAnchor.from_evidence(value)
        for value in (math.nan, math.inf, -math.inf, True, "4.0", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RouteDriftAnchor(0.0, 0.0, value, 0.0)
        with self.assertRaises(ValueError):
            replace(_simple_context(), drift_reference=RouteDriftAnchor(0.0, 0.0, 5.0, 0.0))
        with self.assertRaises(ValueError):
            _simple_context().drift_reference.validate_route_start(Pose2D(0.01, 0.0))

    def test_continuity_rejects_missing_live_transform_without_losing_anchor(self):
        context = _simple_context()
        result = evaluate_map_odom_continuity(context, None)
        self.assertFalse(result.accepted)
        self.assertTrue(result.requires_zero_cycle)
        self.assertEqual(result.reason, "map_from_odom_missing")
        self.assertEqual(result.to_evidence()["drift_reference"], context.drift_reference.to_evidence())
        self.assertEqual(validate_map_odom_continuity_evidence(result.to_evidence(), context=context), result)

    def test_schema_two_cannot_drop_the_anchor_or_change_recomputed_measurements(self):
        context = _simple_context()
        evidence = evaluate_map_odom_continuity(context, PlanarTransform2D(-3.8, 0.0, 0.0)).to_evidence()
        mutations = [("schema_version", 1), ("schema_version", True), ("drift_reference", None),
                     ("translation_drift_m", 0.3), ("relative_translation_x_m", 0.3),
                     ("absolute_yaw_drift_rad", 0.05), ("origin_translation_drift_m", 0.3),
                     ("accepted", 0), ("requires_zero_cycle", 1),
                     ("live_map_from_odom", {"x_m": math.nan, "y_m": 0.0, "yaw_rad": 0.0})]
        for field, value in mutations:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_map_odom_continuity_evidence({**evidence, field: value}, context=context)
        changed = deepcopy(evidence)
        del changed["drift_reference"]
        with self.assertRaises(ValueError):
            validate_map_odom_continuity_evidence(changed)
        legacy = evaluate_map_odom_continuity(replace(context, drift_reference=None), PlanarTransform2D(-3.8, 0.0, 0.0)).to_evidence()
        with self.assertRaises(ValueError):
            validate_map_odom_continuity_evidence(legacy, context=context)
        with self.assertRaises(ValueError):
            validate_map_odom_continuity_evidence(evidence, context=replace(context, certificate_sha256="b" * 64))


if __name__ == "__main__":
    unittest.main()
