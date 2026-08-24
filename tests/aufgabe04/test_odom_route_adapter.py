import math
import unittest

from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
    pose_route_sha256,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    CONTINUITY_ACCEPTED,
    CONTINUITY_RESEAL,
    OdomExecutionContext,
    adapt_map_route_update_to_odom,
    evaluate_map_odom_continuity,
)


CERTIFICATE_SHA256 = "a" * 64
SOURCE_MAP_ROUTE_SHA256 = "b" * 64


def _context(
    *,
    transform: PlanarTransform2D | None = None,
    translation_limit_m: float = 0.03,
    yaw_limit_rad: float = 0.1,
) -> OdomExecutionContext:
    return OdomExecutionContext(
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        frozen_map_from_odom=(
            PlanarTransform2D(1.0, 2.0, math.pi / 2.0)
            if transform is None
            else transform
        ),
        certificate_sha256=CERTIFICATE_SHA256,
        max_map_from_odom_translation_drift_m=translation_limit_m,
        max_map_from_odom_yaw_drift_rad=yaw_limit_rad,
    )


class OdomExecutionContextTests(unittest.TestCase):
    def test_pose_helpers_use_declared_map_from_odom_direction(self) -> None:
        context = _context()

        pose_odom = context.map_pose_to_odom(Pose2D(1.0, 3.0, math.pi / 2.0))

        self.assertAlmostEqual(pose_odom.x_m, 1.0)
        self.assertAlmostEqual(pose_odom.y_m, 0.0)
        self.assertAlmostEqual(pose_odom.yaw_rad, 0.0)
        recovered = context.odom_pose_to_map(pose_odom)
        self.assertAlmostEqual(recovered.x_m, 1.0)
        self.assertAlmostEqual(recovered.y_m, 3.0)
        self.assertAlmostEqual(recovered.yaw_rad, math.pi / 2.0)

    def test_context_rejects_ambiguous_frames_hashes_and_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            OdomExecutionContext(
                map_frame="map",
                odom_frame="map",
                base_frame="base_footprint",
                frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                certificate_sha256=CERTIFICATE_SHA256,
                max_map_from_odom_translation_drift_m=0.03,
                max_map_from_odom_yaw_drift_rad=0.1,
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            _context().__class__(
                map_frame="map",
                odom_frame="odom",
                base_frame="base_footprint",
                frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                certificate_sha256="not-a-hash",
                max_map_from_odom_translation_drift_m=0.03,
                max_map_from_odom_yaw_drift_rad=0.1,
            )
        with self.assertRaisesRegex(ValueError, "<= pi"):
            _context(yaw_limit_rad=math.pi + 0.01)


class MapOdomContinuityTests(unittest.TestCase):
    def test_exact_thresholds_are_accepted(self) -> None:
        frozen = PlanarTransform2D(0.0, 0.0, 0.0)
        context = _context(
            transform=frozen,
            translation_limit_m=0.05,
            yaw_limit_rad=0.04,
        )
        live = PlanarTransform2D(0.03, 0.04, 0.04)

        result = evaluate_map_odom_continuity(context, live)

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, CONTINUITY_ACCEPTED)
        self.assertAlmostEqual(result.translation_drift_m, 0.05)
        self.assertAlmostEqual(result.relative_yaw_rad, 0.04)
        self.assertAlmostEqual(result.absolute_yaw_drift_rad, 0.04)
        self.assertFalse(result.to_evidence()["fail_closed"])

    def test_relative_yaw_is_normalized_across_pi_boundary(self) -> None:
        context = _context(
            transform=PlanarTransform2D(0.0, 0.0, math.pi - 0.02),
            translation_limit_m=0.01,
            yaw_limit_rad=0.041,
        )

        result = evaluate_map_odom_continuity(
            context,
            PlanarTransform2D(0.0, 0.0, -math.pi + 0.02),
        )

        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.relative_yaw_rad, 0.04)

    def test_map_jump_is_refused_with_zero_reseal_evidence(self) -> None:
        context = _context(
            transform=PlanarTransform2D(0.0, 0.0, 0.0),
            translation_limit_m=0.03,
            yaw_limit_rad=0.1,
        )

        result = evaluate_map_odom_continuity(
            context,
            PlanarTransform2D(0.031, 0.0, 0.11),
        )
        evidence = result.to_evidence()

        self.assertFalse(result.accepted)
        self.assertTrue(result.requires_zero_reseal)
        self.assertTrue(result.requires_zero_cycle)
        self.assertEqual(result.decision, CONTINUITY_RESEAL)
        self.assertEqual(
            result.reason,
            "map_from_odom_translation_and_yaw_drift",
        )
        self.assertTrue(evidence["fail_closed"])
        self.assertTrue(evidence["requires_zero_cycle"])
        self.assertTrue(evidence["requires_reseal"])
        self.assertEqual(evidence["certificate_sha256"], CERTIFICATE_SHA256)
        self.assertEqual(evidence["base_frame"], "base_footprint")
        self.assertEqual(
            evidence["threshold_semantics"],
            "accept_if_observed_less_than_or_equal_to_limit",
        )

    def test_missing_and_malformed_live_transform_fail_closed(self) -> None:
        context = _context()

        missing = evaluate_map_odom_continuity(context, None)
        malformed = evaluate_map_odom_continuity(
            context,
            {"x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.0},
        )

        self.assertEqual(missing.reason, "map_from_odom_missing")
        self.assertEqual(malformed.reason, "map_from_odom_malformed")
        for result in (missing, malformed):
            self.assertFalse(result.accepted)
            self.assertEqual(result.decision, CONTINUITY_RESEAL)
            self.assertIsNone(result.translation_drift_m)
            self.assertTrue(result.to_evidence()["fail_closed"])


class OdomRouteUpdateAdapterTests(unittest.TestCase):
    def test_adopt_is_transformed_and_preserves_identity_and_events(self) -> None:
        context = _context()
        map_waypoints = (
            Pose2D(1.0, 2.0, math.pi / 2.0),
            Pose2D(1.0, 3.0, math.pi / 2.0),
        )
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=map_waypoints,
            target_index=0,
            reason="sealed dynamic repair",
            route_revision=4,
            target_revision=9,
            route_hash=SOURCE_MAP_ROUTE_SHA256,
            requires_zero_cycle=False,
            event_name="transient_navigation_blockage_replanned",
            event_fields={"planner_fact": "preserved", "join_index": 0},
        )

        adapted = adapt_map_route_update_to_odom(update, context)

        self.assertIs(adapted.kind, RouteUpdateKind.ADOPT)
        self.assertEqual(adapted.target_index, update.target_index)
        self.assertEqual(adapted.reason, update.reason)
        self.assertEqual(adapted.route_revision, update.route_revision)
        self.assertEqual(adapted.target_revision, update.target_revision)
        self.assertEqual(adapted.event_name, update.event_name)
        self.assertTrue(adapted.requires_zero_cycle)
        self.assertEqual(adapted.event_fields["planner_fact"], "preserved")
        self.assertEqual(adapted.event_fields["join_index"], 0)
        self.assertEqual(
            adapted.event_fields["source_map_route_sha256"],
            SOURCE_MAP_ROUTE_SHA256,
        )
        self.assertEqual(
            adapted.event_fields["source_map_pose_route_sha256"],
            pose_route_sha256(map_waypoints),
        )
        self.assertEqual(adapted.event_fields["source_route_frame"], "map")
        self.assertEqual(adapted.event_fields["execution_route_frame"], "odom")
        self.assertEqual(adapted.route_hash, pose_route_sha256(adapted.waypoints))
        self.assertEqual(
            adapted.event_fields["transformed_odom_route_sha256"],
            adapted.route_hash,
        )
        self.assertEqual(
            adapted.event_fields["odom_execution_certificate_sha256"],
            CERTIFICATE_SHA256,
        )
        self.assertAlmostEqual(adapted.waypoints[0].x_m, 0.0)
        self.assertAlmostEqual(adapted.waypoints[0].y_m, 0.0)
        self.assertAlmostEqual(adapted.waypoints[1].x_m, 1.0)
        self.assertAlmostEqual(adapted.waypoints[1].y_m, 0.0)

    def test_non_adopt_update_is_returned_unchanged(self) -> None:
        update = RouteUpdate(
            kind=RouteUpdateKind.STOP,
            reason="planner unavailable",
            requires_zero_cycle=True,
            event_fields={"fail_closed": True},
        )

        adapted = adapt_map_route_update_to_odom(update, _context())

        self.assertIs(adapted, update)

    def test_conflicting_source_identity_is_rejected(self) -> None:
        update = RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=(Pose2D(0.0, 0.0), Pose2D(1.0, 0.0)),
            route_hash=SOURCE_MAP_ROUTE_SHA256,
            requires_zero_cycle=True,
            event_fields={"source_map_route_sha256": "c" * 64},
        )

        with self.assertRaisesRegex(ValueError, "conflicts"):
            adapt_map_route_update_to_odom(update, _context())


if __name__ == "__main__":
    unittest.main()
