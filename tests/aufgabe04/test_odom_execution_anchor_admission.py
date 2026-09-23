"""Real admission and route revision use one certificate-bound drift anchor."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import RouteUpdate, RouteUpdateKind
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig, evaluate_route_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.map_odom_drift_reference import RouteDriftAnchor
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D, load_odom_execution_certificate,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import OdomExecutionContext
from scripts.aufgabe04.navigation.localization.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import CELL_FREE, MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.station_segment import localization_admission as module


def grid():
    metadata = MapMetadata(yaml_path=Path("map.yaml"), image_path=Path("map.pgm"),
                           resolution=.25, origin=(0., 0., 0.), negate=0,
                           occupied_thresh=.65, free_thresh=.2, mode="trinary")
    return OccupancyGrid(metadata=metadata, width=40, height=40,
                         cells=tuple((CELL_FREE,) * 40 for _ in range(40)))


def config(anchor):
    return RouteUncertaintyAdmissionConfig(
        robot_radius_m=.1, collision_margin_m=.02, fixed_odom_tracking_bound_m=.03,
        empirical_odom_drift_bound_m=.01, braking_latency_distance_m=.02,
        localization_sigma_multiplier=2, heading_sigma_rad=.02,
        heading_lever_arm_m=.1, sampling_spacing_m=.05,
        heading_reference_x_m=anchor.map_x_m, heading_reference_y_m=anchor.map_y_m,
    )


def context(anchor, transform):
    return OdomExecutionContext(
        map_frame="map", odom_frame="odom", base_frame="base_footprint",
        frozen_map_from_odom=transform, certificate_sha256="a" * 64,
        max_map_from_odom_translation_drift_m=.05, max_map_from_odom_yaw_drift_rad=.04,
        drift_reference=anchor,
    )


class OdomExecutionAnchorAdmissionTest(unittest.TestCase):
    def test_real_admission_publishes_v2_and_shares_stationary_budget_context_anchor(self):
        self._assert_real_admission(stationary_turn=False)

    def test_stationary_admitted_turn_keeps_odom_certificate_and_uncertainty_gates(self):
        self._assert_real_admission(stationary_turn=True)

    def _assert_real_admission(self, *, stationary_turn):
        transform = PlanarTransform2D(-4, 1, 0)
        route = (Pose2D(1, 1, float("nan")), Pose2D(1, 1, 1)) if stationary_turn else (Pose2D(1, 1, 0), Pose2D(1.3, 1, 0))
        samples = []
        for index in range(2):
            stamp = 10 + index
            receipt = 20 + index
            samples.append(dict(
                amcl_sample_index=index, source="direct_dynamic_tf", target_frame="map",
                source_frame="odom", observed_target_frame="map", observed_source_frame="odom",
                stamp_sec=stamp, stamp_nanoseconds=stamp * 10**9,
                receipt_time_sec=receipt, receipt_time_nanoseconds=receipt * 10**9,
                capture_time_sec=receipt + .01, capture_time_nanoseconds=receipt * 10**9 + 10**7,
                x_m=transform.x_m, y_m=transform.y_m, yaw_rad=transform.yaw_rad,
            ))
        covariance = [0.] * 36
        covariance[0] = covariance[7] = .000625
        covariance[35] = .0004
        preflight = RosPreflightResult(
            ok=True, failures=[], observations=[], runtime_config={},
            route_pose=dict(frame_id="map", child_frame_id="base_footprint", x_m=1, y_m=1, yaw_rad=0),
            odom_pose=dict(frame_id="odom", child_frame_id="base_footprint", x_m=5, y_m=0, yaw_rad=0),
            map_from_odom=dict(target_frame="map", source_frame="odom", x_m=-4, y_m=1,
                               yaw_rad=0, stamp_sec=12., capture_time_sec=22.),
            stationary_map_from_odom_samples=samples,
            stationary_amcl_samples=[dict(covariance=covariance)],
        )
        diagnostics = SimpleNamespace(metadata=dict(
            arena_boundary_overlay=True, arena_bounds=dict(length_m=10., width_m=10.,
                center_x_m=5., center_y_m=5., yaw_deg=0., margin_m=0.), map_bundle_sha256="d" * 64,
        ))
        if stationary_turn:
            diagnostics.metadata.update(
                stationary_turn=True,
                exact_start_connector={"exact_start": {"x_m": 1., "y_m": 1., "yaw_rad": 0.}},
                target_evidence_sha256="e" * 64,
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                certified_route_tube_radius_m=.03, max_stationary_amcl_yaw_spread_rad=.1,
                uncertainty_robot_radius_m=.1, uncertainty_sigma_multiplier=2.,
                max_map_odom_translation_drift_m=.15, max_map_odom_yaw_drift_rad=.1,
                uncertainty_map_yaml=root / "map.yaml", uncertainty_collision_margin_m=.02,
                uncertainty_odom_drift_bound_m=.01, uncertainty_braking_latency_distance_m=.02,
                uncertainty_heading_lever_arm_m=.7, uncertainty_clearance_sample_spacing_m=.05,
                localization_branch_proof_id="test-branch", uncertainty_budget_json=root / "budget.json",
                odom_execution_certificate_json=root / "certificate.json", coverage_transient_replan_session_root=None,
            )
            resolved = SimpleNamespace(map_frame="map", odom_frame="odom", base_frame="base_footprint", namespace="/")
            with patch.object(module, "poses_from_waypoints", return_value=route), \
                 patch.object(module, "load_occupancy_grid", return_value=grid()), \
                 patch.object(module, "_resolved_map_execution_certificate", return_value=(
                     SimpleNamespace(route_kind="detected_stand_preapproach"), "b" * 64)):
                _, admitted_context, evidence, gate = module._build_odom_execution_admission(
                    args=args, resolved=resolved, leg=SimpleNamespace(
                        executable_waypoints=[],
                        route_kind="admitted_candidate_pose" if stationary_turn else "detected_stand_preapproach",
                        stationary_turn=stationary_turn,
                    ),
                    preflight=preflight, diagnostics_snapshot=diagnostics,
                )
            certificate = load_odom_execution_certificate(args.odom_execution_certificate_json)
            budget = json.loads(args.uncertainty_budget_json.read_text())
            expected = RouteDriftAnchor.from_route_start(route[0], transform)
            self.assertEqual(certificate.schema_version, 2)
            self.assertEqual(certificate.drift_reference, expected)
            self.assertEqual(admitted_context.drift_reference, expected)
            self.assertEqual(budget["schema_version"], 2)
            self.assertEqual(budget["runtime_map_odom_continuity_allocation"]["drift_reference"], expected.to_evidence())
            self.assertEqual(budget["runtime_map_odom_continuity_allocation"]["route_yaw_lever_arm_m"], .7)
            self.assertEqual(budget["stationary_map_from_odom_stability"]["drift_reference"], expected.to_evidence())
            self.assertEqual(evidence["drift_reference"], expected.to_evidence())
            self.assertEqual(gate._config.heading_reference_x_m, expected.map_x_m)
            self.assertEqual(gate._config.heading_reference_y_m, expected.map_y_m)
            if stationary_turn:
                self.assertEqual(
                    budget["admission"]["sampling"]["method"],
                    "stationary_circular_footprint_largest_covariance_eigenvalue",
                )
                self.assertEqual(budget["admission"]["sampling"]["target_evidence_sha256"], "e" * 64)

    def test_revision_keeps_original_anchor_and_consumes_its_full_lever_arm(self):
        transform = PlanarTransform2D(-4, 1, 0)
        anchor = RouteDriftAnchor.from_route_start(Pose2D(1, 1), transform)
        costmap = Costmap.from_occupancy_grid(grid())
        covariance = PlanarCovariance(.000625, 0, .000625)
        cfg = config(anchor)
        route = (Pose2D(4, 4), Pose2D(4.3, 4))
        update = RouteUpdate(kind=RouteUpdateKind.ADOPT, waypoints=route, target_index=1,
                             route_revision=2, route_hash="b" * 64)
        gate = module._OdomRouteUncertaintyGate(costmap=costmap, covariance=covariance,
            config=cfg, evidence_root=None, drift_reference=anchor)
        result = gate.adapt(update, context(anchor, transform))
        self.assertEqual(result.kind, RouteUpdateKind.ADOPT)
        self.assertEqual(result.event_fields["replacement_route_drift_reference"], anchor.to_evidence())
        fixed = evaluate_route_uncertainty_admission(costmap, route, covariance, cfg)
        recentered = evaluate_route_uncertainty_admission(costmap, route, covariance,
            replace(cfg, heading_reference_x_m=4, heading_reference_y_m=4))
        self.assertAlmostEqual(result.event_fields["replacement_route_uncertainty_remaining_margin_m"],
                               fixed.decision.remaining_margin_m)
        self.assertLess(fixed.decision.remaining_margin_m, recentered.decision.remaining_margin_m - .1)
        self.assertEqual(gate._config, cfg)
        for invalid in (None, RouteDriftAnchor.from_route_start(route[0], transform)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "drift_reference mismatch"):
                gate.adapt(update, context(invalid, transform))
        for changes in (dict(max_map_from_odom_translation_drift_m=.051),
                        dict(max_map_from_odom_yaw_drift_rad=.041)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "reserved uncertainty"):
                gate.adapt(update, replace(context(anchor, transform), **changes))
        with self.assertRaisesRegex(ValueError, "heading reference"):
            module._OdomRouteUncertaintyGate(costmap=costmap, covariance=covariance,
                config=replace(cfg, heading_reference_x_m=4), evidence_root=None, drift_reference=anchor)


if __name__ == "__main__":
    unittest.main()
