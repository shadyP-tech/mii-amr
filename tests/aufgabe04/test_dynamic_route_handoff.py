from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    DynamicRouteSource,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.route_revision_store import RouteRevisionStore


def _route_csv(points: list[tuple[float, float]], *, route_kind: str = "") -> str:
    rows = [
        "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,yaw_rad,"
        "segment_length_m,cumulative_length_m,route_kind"
    ]
    cumulative = 0.0
    previous: tuple[float, float] | None = None
    for index, (x_m, y_m) in enumerate(points):
        segment = 0.0
        if previous is not None:
            segment = ((x_m - previous[0]) ** 2 + (y_m - previous[1]) ** 2) ** 0.5
            cumulative += segment
        rows.append(
            f"0,{index},{index},0,{x_m},{y_m},0.0,{segment},{cumulative},{route_kind}"
        )
        previous = (x_m, y_m)
    return "\n".join(rows) + "\n"


def _publish(
    store: RouteRevisionStore,
    points: list[tuple[float, float]],
    *,
    target_revision: int = 1,
    observation_unix_sec: float = 100.0,
    start_join_clearance_m: float = 1.0,
    takeover: bool = False,
    route_kind: str = "",
    safety_diagnostics: dict | None = None,
):
    safety = {
        "keepout_clear": True,
        "corridor_clear": True,
        "start_join_clearance_m": start_join_clearance_m,
        "arena_bounds": {
            "length_m": 3.9,
            "width_m": 1.898,
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "yaw_deg": 0.0,
            "margin_m": 0.0,
        },
        "arena_boundary_overlay": True,
    }
    if safety_diagnostics is not None:
        safety.update(safety_diagnostics)
    return store.publish_active(
        _route_csv(points, route_kind=route_kind),
        {"status": "planned"},
        target_revision=target_revision,
        observation_unix_sec=observation_unix_sec,
        source_robot_pose={"x_m": points[0][0], "y_m": points[0][1], "yaw_rad": 0.0},
        target={"x_m": points[-1][0], "y_m": points[-1][1], "yaw_rad": 0.0},
        evidence={"axis_rad": 0.2, "confidence": 0.91},
        previous_route_length_m=0.0,
        new_route_length_m=max(0.0, 0.1 * (len(points) - 1)),
        safety_diagnostics=safety,
        takeover=takeover,
    )


class TestDynamicRouteHandoff(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.tmp_path = Path(self._temporary_directory.name)

    @staticmethod
    def _egress_safety(*, minimum_clearance_m: float = 0.30) -> dict:
        return {
            "known_stand_start_cell_exempted": True,
            "known_stand_start_cell": {"x": 53, "y": 28},
            "known_stand_keepout_rasterized_cell_count": 100,
            "known_stand_keepout_cell_count": 99,
            "known_stand_keepouts": [
                {"x_m": -0.395, "y_m": -0.415, "radius_m": 0.26}
            ],
            "known_stand_keepout_clearances": [
                {
                    "x_m": -0.395,
                    "y_m": -0.415,
                    "radius_m": 0.26,
                    "minimum_route_clearance_m": minimum_clearance_m,
                }
            ],
        }

    def test_start_cell_exemption_adopts_unspliced_egress_vertex_lock(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        points = [
            (-0.131011, -0.270103),
            (-0.195, -0.115),
            (-0.595, -0.115),
        ]
        _publish(
            store,
            points,
            route_kind="axis_acquisition",
            safety_diagnostics=self._egress_safety(),
        )
        source = DynamicRouteSource(
            manifest,
            stream_id="sim",
            forward_splice_min_offset_m=0.01,
        )

        update = source.poll(Pose2D(-0.125, -0.270103, -2.702), 100.0)

        self.assertIs(update.kind, RouteUpdateKind.ADOPT)
        self.assertEqual(
            [(pose.x_m, pose.y_m) for pose in update.waypoints],
            points,
        )
        self.assertFalse(update.event_fields["forward_splice"])
        self.assertTrue(update.event_fields["start_egress_vertex_lock"])
        self.assertEqual(update.event_fields["start_egress_waypoint_index"], 1)
        self.assertTrue(
            update.event_fields["start_egress_continuous_clearance_validated"]
        )

    def test_missing_arena_boundary_evidence_stops_before_adoption(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(
            store,
            [(0.0, 0.0), (0.2, 0.0)],
            route_kind="axis_acquisition",
            safety_diagnostics={"arena_bounds": None},
        )

        source = DynamicRouteSource(manifest, stream_id="sim")
        update = source.poll(
            Pose2D(0.0, 0.0, 0.0),
            100.0,
        )

        self.assertIs(update.kind, RouteUpdateKind.STOP)
        self.assertEqual(
            update.event_fields["fault_code"],
            "invalid_arena_boundary_evidence",
        )
        repeated = source.poll(Pose2D(0.0, 0.0, 0.0), 100.1)
        self.assertIs(repeated.kind, RouteUpdateKind.STOP)
        self.assertTrue(repeated.requires_zero_cycle)
        self.assertEqual(
            repeated.event_fields["fault_code"],
            "invalid_arena_boundary_evidence",
        )
        self.assertIsNone(repeated.event_name)

    def test_start_cell_exemption_with_malformed_clearance_fails_closed(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(
            store,
            [
                (-0.131011, -0.270103),
                (-0.195, -0.115),
                (-0.595, -0.115),
            ],
            route_kind="axis_acquisition",
            safety_diagnostics=self._egress_safety(minimum_clearance_m=0.25),
        )

        update = DynamicRouteSource(manifest, stream_id="sim").poll(
            Pose2D(-0.131011, -0.270103, -2.702),
            100.0,
        )

        self.assertIs(update.kind, RouteUpdateKind.STOP)
        self.assertEqual(
            update.event_fields["fault_code"],
            "invalid_egress_certificate",
        )

    def test_adopt_splices_forward_inside_certified_start_disk(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim-run", writer_id="planner", now_fn=lambda: 100.0
        )
        committed = _publish(
            store,
            [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)],
            route_kind="synchronized_face_approach",
        )
        source = DynamicRouteSource(
            manifest,
            stream_id="sim-run",
            expected_writer_id="planner",
            max_join_distance_m=0.08,
            max_forward_window=3,
        )

        update = source.poll(Pose2D(0.05, 0.0), now_unix_sec=100.0)

        self.assertIs(update.kind, RouteUpdateKind.ADOPT)
        xs = [pose.x_m for pose in update.waypoints]
        self.assertEqual(xs[0], 0.05)
        self.assertGreater(xs[1], xs[0])
        self.assertLessEqual(xs[1], 0.08)
        self.assertEqual(xs[2:], [0.1, 0.2, 0.3])
        self.assertEqual(update.target_index, 0)
        self.assertTrue(update.requires_zero_cycle)
        self.assertEqual(update.route_revision, 1)
        self.assertEqual(update.route_hash, committed.route_hash)
        self.assertEqual(update.event_name, "dynamic_route_adopted")
        self.assertEqual(update.event_fields["join_index"], 0)
        self.assertEqual(update.event_fields["adopted_waypoint_count"], 5)
        self.assertTrue(update.event_fields["forward_splice"])
        self.assertTrue(update.event_fields["requires_zero_cycle"])
        self.assertEqual(update.event_fields["source_robot_pose"]["x_m"], 0.0)
        self.assertEqual(update.event_fields["adoption_robot_pose"]["x_m"], 0.05)
        self.assertEqual(update.event_fields["target_revision"], 1)
        self.assertEqual(update.event_fields["route_sha256"], committed.route_hash)
        self.assertEqual(
            update.event_fields["route_kind"], "synchronized_face_approach"
        )

    def test_duplicate_revision_is_unchanged(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(store, [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)])
        source = DynamicRouteSource(manifest, stream_id="sim")
        self.assertIs(
            source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )

        duplicate = source.poll(Pose2D(0.05, 0.0), 100.1)

        self.assertIs(duplicate.kind, RouteUpdateKind.UNCHANGED)
        self.assertEqual(duplicate.waypoints, ())
        self.assertFalse(duplicate.requires_zero_cycle)

    def test_survey_completion_is_success_not_withdrawal(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        active = _publish(
            store,
            [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
            route_kind="axis_acquisition",
        )
        source = DynamicRouteSource(manifest, stream_id="sim")
        self.assertIs(
            source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )
        store.complete_survey(
            "arrival pose recorded",
            completion={"candidate_uid": "A", "catalog_revision": 1},
        )

        update = source.poll(Pose2D(0.05, 0.0), 100.0)

        self.assertIs(update.kind, RouteUpdateKind.COMPLETE)
        self.assertTrue(update.requires_zero_cycle)
        self.assertEqual(update.route_hash, active.route_hash)
        self.assertEqual(update.event_name, "dynamic_survey_completed")
        self.assertFalse(update.event_fields["fail_closed"])

    def test_fresh_same_geometry_revision_is_heartbeat_not_rejoin(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        clock = [100.0]
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: clock[0]
        )
        points = [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0)]
        first = _publish(store, points, target_revision=1)
        source = DynamicRouteSource(
            manifest, stream_id="sim", max_join_distance_m=0.08
        )
        self.assertIs(source.poll(Pose2D(0.0, 0.0), 100.0).kind, RouteUpdateKind.ADOPT)

        clock[0] = 101.0
        heartbeat = _publish(
            store, points, target_revision=1, observation_unix_sec=101.0
        )
        update = source.poll(Pose2D(0.30, 0.0), 101.0)

        self.assertEqual(heartbeat.route_revision, first.route_revision + 1)
        self.assertEqual(heartbeat.route_hash, first.route_hash)
        self.assertIs(update.kind, RouteUpdateKind.UNCHANGED)
        self.assertTrue(update.event_fields["heartbeat"])
        self.assertTrue(update.event_fields["installed_route_unchanged"])
        self.assertFalse(update.requires_zero_cycle)

    def test_unsafe_join_stops_without_replacing_current_route(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(store, [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)])
        source = DynamicRouteSource(
            manifest, stream_id="sim", max_join_distance_m=0.1, max_forward_window=2
        )

        rejected = source.poll(Pose2D(2.0, 2.0), 100.0)

        self.assertIs(rejected.kind, RouteUpdateKind.STOP)
        self.assertTrue(rejected.requires_zero_cycle)
        self.assertEqual(rejected.event_name, "dynamic_route_stopped")
        self.assertEqual(rejected.event_fields["fault_code"], "unsafe_route_join")
        # The same immutable rejection remains fail-closed, while the semantic
        # event itself is de-duplicated on subsequent controller polls.
        repeated = source.poll(Pose2D(2.0, 2.0), 100.1)
        self.assertIs(repeated.kind, RouteUpdateKind.STOP)
        self.assertTrue(repeated.requires_zero_cycle)
        self.assertIsNone(repeated.event_name)
        self.assertIsNone(repeated.event_name)

    def test_non_finite_live_pose_fails_closed_before_adoption(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(store, [(0.0, 0.0), (0.1, 0.0)])
        update = DynamicRouteSource(manifest, stream_id="sim").poll(
            Pose2D(float("nan"), 0.0, 0.0), 100.0
        )
        self.assertIs(update.kind, RouteUpdateKind.STOP)
        self.assertEqual(update.event_fields["fault_code"], "invalid_current_pose")

    def test_forward_window_prevents_joining_arbitrarily_far_along_new_route(
        self,
    ) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(store, [(index * 0.1, 0.0) for index in range(8)])
        source = DynamicRouteSource(
            manifest,
            stream_id="sim",
            max_join_distance_m=0.05,
            max_forward_window=2,
        )

        update = source.poll(Pose2D(0.6, 0.0), 100.0)

        self.assertIs(update.kind, RouteUpdateKind.STOP)
        self.assertEqual(update.event_fields["fault_code"], "unsafe_route_join")

    def test_new_revision_adopts_forward_suffix_after_progress(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        clock = [100.0]
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: clock[0]
        )
        first_points = [(index * 0.1, 0.0) for index in range(6)]
        _publish(store, first_points, target_revision=1)
        source = DynamicRouteSource(
            manifest, stream_id="sim", max_join_distance_m=0.08, max_forward_window=4
        )
        self.assertIs(
            source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )

        clock[0] = 101.0
        second_points = [(0.30, 0.02), (0.4, 0.04), (0.5, 0.05)]
        second = _publish(
            store, second_points, target_revision=2, observation_unix_sec=101.0
        )
        update = source.poll(Pose2D(0.31, 0.02), 101.0)

        self.assertIs(update.kind, RouteUpdateKind.ADOPT)
        self.assertEqual(update.route_revision, second.route_revision)
        self.assertEqual(second.route_revision, 2)
        xs = [pose.x_m for pose in update.waypoints]
        self.assertEqual(xs[0], 0.31)
        self.assertGreater(xs[1], xs[0])
        self.assertEqual(xs[-2:], [0.4, 0.5])
        self.assertEqual(update.event_fields["join_index"], 0)
        self.assertTrue(update.event_fields["forward_splice"])

    def test_join_must_fit_planner_certified_free_disk(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(
            store,
            [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
            start_join_clearance_m=0.05,
        )
        source = DynamicRouteSource(
            manifest, stream_id="sim", max_join_distance_m=0.10
        )

        update = source.poll(Pose2D(0.07, 0.0), 100.0)

        self.assertIs(update.kind, RouteUpdateKind.STOP)
        self.assertEqual(update.event_fields["fault_code"], "unsafe_route_join")
        self.assertAlmostEqual(update.event_fields["effective_join_limit_m"], 0.05)

    def test_installed_terminal_route_does_not_expire_but_still_polls(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        clock = [100.0]
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: clock[0]
        )
        _publish(store, [(0.0, 0.0), (0.2, 0.0)], observation_unix_sec=100.0)
        source = DynamicRouteSource(
            manifest,
            stream_id="sim",
            max_manifest_age_sec=1.0,
            max_observation_age_sec=1.0,
            terminal_route_lock_distance_m=0.42,
            now_fn=lambda: clock[0],
        )
        self.assertIs(source.poll(Pose2D(0.0, 0.0), 100.0).kind, RouteUpdateKind.ADOPT)

        clock[0] = 105.0
        unchanged = source.poll(Pose2D(0.10, 0.0), 105.0)

        self.assertIs(unchanged.kind, RouteUpdateKind.UNCHANGED)

    def test_route_still_expires_outside_terminal_lock(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(store, [(0.0, 0.0), (1.0, 0.0)], observation_unix_sec=100.0)
        source = DynamicRouteSource(
            manifest,
            stream_id="sim",
            max_manifest_age_sec=1.0,
            max_observation_age_sec=1.0,
            terminal_route_lock_distance_m=0.42,
        )
        self.assertIs(source.poll(Pose2D(0.0, 0.0), 100.0).kind, RouteUpdateKind.ADOPT)

        stale = source.poll(Pose2D(0.20, 0.0), 105.0)

        self.assertIs(stale.kind, RouteUpdateKind.STOP)
        self.assertIn(stale.event_fields["fault_code"], {"manifest_stale", "observation_stale"})

    def test_withdrawal_always_returns_stop_and_deduplicates_event(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        clock = [100.0]
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: clock[0]
        )
        _publish(store, [(0.0, 0.0), (0.1, 0.0)])
        source = DynamicRouteSource(manifest, stream_id="sim")
        self.assertIs(
            source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )
        clock[0] = 101.0
        store.withdraw("target evidence withdrawn", observation_unix_sec=101.0)

        stopped = source.poll(Pose2D(0.05, 0.0), 101.0)
        self.assertIs(stopped.kind, RouteUpdateKind.STOP)
        self.assertEqual(stopped.reason, "target evidence withdrawn")
        self.assertTrue(stopped.requires_zero_cycle)
        self.assertEqual(stopped.event_name, "dynamic_route_withdrawn")

        repeated = source.poll(Pose2D(0.05, 0.0), 101.1)
        self.assertIs(repeated.kind, RouteUpdateKind.STOP)
        self.assertIsNone(repeated.event_name)

    def test_planner_and_observation_age_faults_fail_closed(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: 100.0
        )
        _publish(
            store,
            [(0.0, 0.0), (0.1, 0.0)],
            observation_unix_sec=90.0,
        )

        stale_observation = DynamicRouteSource(
            manifest,
            stream_id="sim",
            max_manifest_age_sec=5.0,
            max_observation_age_sec=5.0,
        ).poll(Pose2D(0.0, 0.0), 100.0)
        self.assertIs(stale_observation.kind, RouteUpdateKind.STOP)
        self.assertEqual(
            stale_observation.event_fields["fault_code"], "observation_stale"
        )

        dead_planner = DynamicRouteSource(
            manifest,
            stream_id="sim",
            max_manifest_age_sec=1.0,
            max_observation_age_sec=99.0,
        ).poll(Pose2D(0.0, 0.0), 102.0)
        self.assertIs(dead_planner.kind, RouteUpdateKind.STOP)
        self.assertEqual(dead_planner.event_fields["fault_code"], "manifest_stale")

    def test_corruption_wrong_stream_rollback_and_path_faults_stop(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        clock = [100.0]
        store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner", now_fn=lambda: clock[0]
        )
        first = _publish(store, [(0.0, 0.0), (0.1, 0.0)])
        first_manifest = manifest.read_bytes()

        wrong_stream = DynamicRouteSource(manifest, stream_id="other").poll(
            Pose2D(0.0, 0.0), 100.0
        )
        self.assertIs(wrong_stream.kind, RouteUpdateKind.STOP)
        self.assertEqual(wrong_stream.event_fields["fault_code"], "wrong_stream")

        corrupt_source = DynamicRouteSource(manifest, stream_id="sim")
        self.assertIsNotNone(first.route_path)
        first.route_path.write_text(first.route_path.read_text() + "corrupt")
        corrupt = corrupt_source.poll(Pose2D(0.0, 0.0), 100.0)
        self.assertIs(corrupt.kind, RouteUpdateKind.STOP)
        self.assertEqual(
            corrupt.event_fields["fault_code"], "artifact_hash_mismatch"
        )

        # Re-publish from clean manifest/artifacts and then demonstrate rollback.
        first.route_path.write_text(_route_csv([(0.0, 0.0), (0.1, 0.0)]))
        manifest.write_bytes(first_manifest)
        rollback_source = DynamicRouteSource(manifest, stream_id="sim")
        self.assertIs(
            rollback_source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )
        clock[0] = 101.0
        _publish(
            store,
            [(0.05, 0.0), (0.15, 0.0)],
            target_revision=2,
            observation_unix_sec=101.0,
        )
        self.assertIs(
            rollback_source.poll(Pose2D(0.05, 0.0), 101.0).kind,
            RouteUpdateKind.ADOPT,
        )
        manifest.write_bytes(first_manifest)
        rollback = rollback_source.poll(Pose2D(0.05, 0.0), 101.1)
        self.assertIs(rollback.kind, RouteUpdateKind.STOP)
        self.assertEqual(rollback.event_fields["fault_code"], "revision_rollback")

        traversal_payload = json.loads(first_manifest)
        traversal_payload["route"]["relative_path"] = "../route.csv"
        manifest.write_text(json.dumps(traversal_payload))
        unsafe_path = DynamicRouteSource(manifest, stream_id="sim").poll(
            Pose2D(0.0, 0.0), 100.0
        )
        self.assertIs(unsafe_path.kind, RouteUpdateKind.STOP)
        self.assertEqual(unsafe_path.event_fields["fault_code"], "unsafe_path")

    def test_writer_takeover_requires_explicit_consumer_permission(self) -> None:
        manifest = self.tmp_path / "dynamic_manifest.json"
        clock = [100.0]
        first_store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner-a", now_fn=lambda: clock[0]
        )
        _publish(first_store, [(0.0, 0.0), (0.1, 0.0)])
        strict_source = DynamicRouteSource(manifest, stream_id="sim")
        permissive_source = DynamicRouteSource(
            manifest, stream_id="sim", allow_writer_takeover=True
        )
        self.assertIs(
            strict_source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )
        self.assertIs(
            permissive_source.poll(Pose2D(0.0, 0.0), 100.0).kind,
            RouteUpdateKind.ADOPT,
        )

        clock[0] = 101.0
        second_store = RouteRevisionStore(
            manifest, stream_id="sim", writer_id="planner-b", now_fn=lambda: clock[0]
        )
        _publish(
            second_store,
            [(0.02, 0.0), (0.12, 0.0)],
            target_revision=2,
            observation_unix_sec=101.0,
            takeover=True,
        )

        strict = strict_source.poll(Pose2D(0.02, 0.0), 101.0)
        self.assertIs(strict.kind, RouteUpdateKind.STOP)
        self.assertEqual(strict.event_fields["fault_code"], "wrong_writer")
        permissive = permissive_source.poll(Pose2D(0.02, 0.0), 101.0)
        self.assertIs(permissive.kind, RouteUpdateKind.ADOPT)
        self.assertEqual(permissive.event_fields["writer_generation"], 2)


if __name__ == "__main__":
    unittest.main()
