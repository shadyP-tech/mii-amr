import json
import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    circular_keepout_cells,
)
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_synchronized_viewpoint import (
    _known_stand_keepout_costmap,
    _validate_known_stand_route_clearance,
    load_recommended_pose,
    main,
)
from scripts.aufgabe04.navigation.route_revision_store import read_committed_revision
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    FaceCandidate,
    MaterialTarget,
    SideEvidence,
    StandGeometry,
    SynchronizedViewpointRecommendation,
    recommendation_to_dict,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import load_arrival_pose_catalog


class PlanSynchronizedViewpointTest(unittest.TestCase):
    @staticmethod
    def _point_to_segment_distance(point, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        denominator = dx * dx + dy * dy
        if denominator <= 1.0e-12:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        fraction = max(
            0.0,
            min(
                1.0,
                ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
                / denominator,
            ),
        )
        closest = (start[0] + fraction * dx, start[1] + fraction * dy)
        return math.hypot(point[0] - closest[0], point[1] - closest[1])

    def test_loads_finite_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recommended.json"
            path.write_text(json.dumps({
                "source": "synchronized_lidar_camera_viewpoint",
                "pose": {"x_m": 0.2, "y_m": -0.1, "yaw_rad": math.pi},
            }))
            pose = load_recommended_pose(path)
        self.assertAlmostEqual(pose.x_m, 0.2)
        self.assertAlmostEqual(pose.yaw_rad, math.pi)

    def test_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recommended.json"
            path.write_text(json.dumps({"source": "hidden_truth", "pose": {}}))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_recommended_pose(path)

    def test_survey_only_records_pose_without_publishing_physical_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text(
                "P2\n50 40\n255\n" + " ".join(["254"] * 2000) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            stand = Pose2D(1.5, 1.5)
            acquisition = Pose2D(0.75, 1.5, 0.0)
            acquisition_rec = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(stand, 0.06, 0.02, "lidar_cluster"),
                robot_pose=Pose2D(0.4, 1.5),
                axis_confidence=0.0,
                axis_state="axis_acquisition",
                face_candidates=(
                    FaceCandidate("acquisition_near", math.pi, acquisition, False),
                    FaceCandidate(
                        "acquisition_far", 0.0, Pose2D(2.25, 1.5, math.pi), False
                    ),
                ),
                side_evidence=SideEvidence(
                    "none", 0.0, False, False, None, "axis_unknown"
                ),
                material_target=MaterialTarget(
                    "acquisition_near", acquisition, "axis_acquisition"
                ),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(
                json.dumps(recommendation_to_dict(acquisition_rec))
            )
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            catalog_path = root / "arrival_pose_catalog.json"
            common = [
                "--map", str(map_yaml),
                "--start-x", "0.4",
                "--start-y", "1.5",
                "--recommended-pose-json", str(recommendation_path),
                "--route-csv", str(route),
                "--diagnostics-json", str(diagnostics),
            ]
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                self.assertEqual(main(common), 0)
            active = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.0
            )
            route_before = route.read_bytes()

            face_a = Pose2D(1.15, 1.5, 0.0)
            committed_rec = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.5,
                sensor_stamp_sec=10.5,
                stand=StandGeometry(stand, 0.06, 0.02, "lidar_cluster"),
                robot_pose=acquisition,
                axis_confidence=0.93,
                axis_state="target_committed",
                face_candidates=(
                    FaceCandidate("face_a", math.pi, face_a, True),
                    FaceCandidate(
                        "face_b", 0.0, Pose2D(1.85, 1.5, math.pi), True
                    ),
                ),
                side_evidence=SideEvidence(
                    "none", 0.0, False, False, None, "axis_only"
                ),
                material_target=MaterialTarget(
                    "face_a", face_a, "robot_facing_axis"
                ),
            )
            recommendation_path.write_text(
                json.dumps(recommendation_to_dict(committed_rec))
            )
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.5,
            ):
                status = main(
                    common
                    + [
                        "--workflow-mode", "survey-only",
                        "--arrival-pose-catalog", str(catalog_path),
                        "--candidate-uid", "A",
                        "--expected-candidate-uid", "A",
                        "--world-id", "test_world",
                        "--world-sha256", "a" * 64,
                        "--session-id", "test_session",
                    ]
                )
            terminal = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.5
            )
            catalog = load_arrival_pose_catalog(catalog_path)
            route_after = route.read_bytes()

        self.assertEqual(status, 0)
        self.assertEqual(terminal.status, "survey_complete")
        self.assertEqual(terminal.route_hash, active.route_hash)
        self.assertEqual(route_after, route_before)
        self.assertEqual(len(catalog.records), 1)
        self.assertEqual(catalog.records[0].candidate_uid, "A")
        self.assertAlmostEqual(catalog.records[0].arrival_pose.x_m, face_a.x_m)

    def test_one_shot_planner_commits_manifest_and_safe_terminal_corridor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pgm = root / "map.pgm"
            pgm.write_text("P2\n40 30\n255\n" + " ".join(["254"] * 1200) + "\n")
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            target = Pose2D(1.85, 1.5, math.pi)
            recommendation = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(Pose2D(1.5, 1.5), 0.06, 0.02, "lidar_cluster"),
                robot_pose=Pose2D(0.4, 1.5),
                axis_confidence=0.9,
                axis_state="resolved",
                face_candidates=(
                    FaceCandidate("face_a", 0.0, target, True),
                    FaceCandidate("face_b", math.pi, Pose2D(1.15, 1.5, 0.0), True),
                ),
                side_evidence=SideEvidence(
                    "qr_registry", 0.98, True, True, "face_a", "sim_qr_consensus"
                ),
                material_target=MaterialTarget("face_a", target, "hard_qr"),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(json.dumps(recommendation_to_dict(recommendation)))
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"

            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                status = main(
                    [
                        "--map",
                        str(map_yaml),
                        "--start-x",
                        "0.4",
                        "--start-y",
                        "1.5",
                        "--recommended-pose-json",
                        str(recommendation_path),
                        "--route-csv",
                        str(route),
                        "--diagnostics-json",
                        str(diagnostics),
                        "--standoff-distance-m",
                        "0.35",
                    ]
                )

            manifest = route.with_suffix(".manifest.json")
            first_committed = read_committed_revision(
                manifest, expected_stream_id="sim-stand-viewpoint", now_unix_sec=100.0
            )
            # Restarting the planner against a positive target revision must
            # restore the effective material target instead of withdrawing a
            # still-fresh active route.
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.5,
            ):
                restart_status = main(
                    [
                        "--map",
                        str(map_yaml),
                        "--start-x",
                        "0.4",
                        "--start-y",
                        "1.5",
                        "--recommended-pose-json",
                        str(recommendation_path),
                        "--route-csv",
                        str(route),
                        "--diagnostics-json",
                        str(diagnostics),
                        "--standoff-distance-m",
                        "0.35",
                    ]
                )
            committed = read_committed_revision(
                manifest, expected_stream_id="sim-stand-viewpoint", now_unix_sec=100.5
            )
            with committed.route_path.open() as committed_route_file:
                committed_geometry = [
                    (
                        row["world_x_m"],
                        row["world_y_m"],
                        row["yaw_rad"],
                        row["protected"],
                        row["corridor"],
                    )
                    for row in csv.DictReader(committed_route_file)
                ]
            refreshed_payload = recommendation_to_dict(recommendation)
            refreshed_payload["observation_unix_sec"] = 105.0
            recommendation_path.write_text(json.dumps(refreshed_payload))
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=105.0,
            ):
                with patch(
                    "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.plan_axis_acquisition",
                    side_effect=AssertionError(
                        "refresh heartbeat must not run hypothetical live-start A*"
                    ),
                ):
                    heartbeat_status = main(
                        [
                            "--map",
                            str(map_yaml),
                            "--start-x",
                            "0.4",
                            "--start-y",
                            "1.5",
                            "--recommended-pose-json",
                            str(recommendation_path),
                            "--route-csv",
                            str(route),
                            "--diagnostics-json",
                            str(diagnostics),
                            "--standoff-distance-m",
                            "0.35",
                        ]
                    )
            heartbeat = read_committed_revision(
                manifest,
                expected_stream_id="sim-stand-viewpoint",
                now_unix_sec=105.0,
            )
            with heartbeat.route_path.open() as heartbeat_route_file:
                heartbeat_geometry = [
                    (
                        row["world_x_m"],
                        row["world_y_m"],
                        row["yaw_rad"],
                        row["protected"],
                        row["corridor"],
                    )
                    for row in csv.DictReader(heartbeat_route_file)
                ]
            aliases_exist = route.exists() and diagnostics.exists()

        self.assertEqual(status, 0)
        self.assertEqual(restart_status, 0)
        self.assertEqual(heartbeat_status, 0)
        self.assertEqual(committed.status, "active")
        self.assertEqual(committed.route_revision, first_committed.route_revision)
        self.assertGreater(heartbeat.route_revision, committed.route_revision)
        self.assertEqual(heartbeat_geometry, committed_geometry)
        self.assertEqual(
            heartbeat.manifest["new_route_length_m"],
            committed.manifest["new_route_length_m"],
        )
        self.assertGreater(committed.target_revision, 0)
        self.assertTrue(committed.manifest["simulation_only"])
        self.assertTrue(aliases_exist)

    def test_axis_acquisition_plans_to_fixed_initial_ray_without_face_corridor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text(
                "P2\n80 60\n255\n" + " ".join(["254"] * 4800) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            stand = Pose2D(1.5, 1.5)
            near = Pose2D(2.2, 1.5, math.pi)
            far = Pose2D(0.8, 1.5, 0.0)
            recommendation = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(stand, 0.06, 0.02, "lidar_cluster"),
                robot_pose=Pose2D(3.4, 1.5, math.pi),
                axis_confidence=0.0,
                axis_state="axis_acquisition",
                face_candidates=(
                    FaceCandidate("acquisition_near", 0.0, near, False),
                    FaceCandidate("acquisition_far", math.pi, far, False),
                ),
                side_evidence=SideEvidence("none", 0.0, False, False, None, "axis_unknown"),
                material_target=MaterialTarget("acquisition_near", near, "axis_acquisition"),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(json.dumps(recommendation_to_dict(recommendation)))
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                status = main(
                    [
                        "--map", str(map_yaml),
                        "--start-x", "3.4",
                        "--start-y", "1.5",
                        "--recommended-pose-json", str(recommendation_path),
                        "--route-csv", str(route),
                        "--diagnostics-json", str(diagnostics),
                    ]
                )
            committed = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.0
            )
            self.assertEqual(status, 0, committed.reason)
            with route.open() as route_file:
                rows = list(csv.DictReader(route_file))

            # gazebo_arrival_e2e_006 reached a point target inside the terminal
            # route-lock radius while the runner performed preflight.  The
            # geometry must stay locked, but the planner must still heartbeat
            # it before the observation-age gate expires.
            refreshed_payload = recommendation_to_dict(recommendation)
            refreshed_payload["observation_unix_sec"] = 105.0
            refreshed_payload["robot_pose"] = {
                "x_m": 2.15,
                "y_m": 1.5,
                "yaw_rad": math.pi,
            }
            recommendation_path.write_text(json.dumps(refreshed_payload))
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=105.0,
            ):
                with patch(
                    "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.plan_axis_acquisition",
                    side_effect=AssertionError(
                        "terminal-lock heartbeat must not run live-start A*"
                    ),
                ):
                    heartbeat_status = main(
                        [
                            "--map", str(map_yaml),
                            "--start-x", "2.15",
                            "--start-y", "1.5",
                            "--recommended-pose-json", str(recommendation_path),
                            "--route-csv", str(route),
                            "--diagnostics-json", str(diagnostics),
                        ]
                    )
            heartbeat = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=105.0
            )
            with heartbeat.route_path.open() as heartbeat_route_file:
                heartbeat_rows = list(csv.DictReader(heartbeat_route_file))

        self.assertEqual(status, 0)
        self.assertEqual(heartbeat_status, 0)
        self.assertGreater(heartbeat.route_revision, committed.route_revision)
        self.assertEqual(heartbeat_rows, rows)
        self.assertEqual(
            heartbeat.manifest["target_revision"],
            committed.manifest["target_revision"],
        )
        self.assertAlmostEqual(float(rows[-1]["world_x_m"]), near.x_m)
        self.assertAlmostEqual(float(rows[-1]["world_y_m"]), near.y_m)
        self.assertTrue(all(row["corridor"] == "false" for row in rows))
        self.assertTrue(all(row["route_kind"] == "axis_acquisition" for row in rows))
        self.assertEqual(committed.manifest["target"]["face_id"], "acquisition_near")

    def test_station_b_regression_avoids_station_a_from_sampling_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text(
                "P2\n100 60\n255\n" + " ".join(["254"] * 6000) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [-2.82, -1.69, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            # Recorded gazebo_arrival_e2e_005 A->B geometry at the exact pose
            # where watch-mode replanning withdrew with astar_failed:no_path.
            # The robot is outside station A's disk, but its containing cell's
            # center is inside the disk and every neighboring cell is touched
            # by the conservative rasterization.
            station_a = (-0.395, -0.415)
            start = Pose2D(
                -0.17361488092693012,
                -0.24796208570165812,
                2.4001406695606864,
            )
            base = Costmap.from_occupancy_grid(load_occupancy_grid(map_yaml))
            unsafe_grid_center = base.grid_to_world(base.world_to_grid(start))
            self.assertGreater(
                math.hypot(start.x_m - station_a[0], start.y_m - station_a[1]),
                0.26,
            )
            self.assertLess(
                math.hypot(
                    unsafe_grid_center.x_m - station_a[0],
                    unsafe_grid_center.y_m - station_a[1],
                ),
                0.26,
            )
            target_stand = Pose2D(-1.695, -0.615)
            acquisition = Pose2D(
                -1.157935327461692,
                -0.49641864602172964,
                -2.9242838935157778,
            )
            recommendation = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="B",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(
                    target_stand, 0.06, 0.02, "lidar_cluster"
                ),
                robot_pose=start,
                axis_confidence=0.0,
                axis_state="axis_acquisition",
                face_candidates=(
                    FaceCandidate(
                        "acquisition_near",
                        0.21730876007401578,
                        acquisition,
                        False,
                    ),
                    FaceCandidate(
                        "acquisition_far",
                        -2.9242838935157778,
                        Pose2D(
                            -2.2320646725383084,
                            -0.7335813539782702,
                            0.21730876007401534,
                        ),
                        False,
                    ),
                ),
                side_evidence=SideEvidence(
                    "none", 0.0, False, False, None, "axis_unknown"
                ),
                material_target=MaterialTarget(
                    "acquisition_near", acquisition, "axis_acquisition"
                ),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(
                json.dumps(recommendation_to_dict(recommendation))
            )
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                status = main(
                    [
                        "--map", str(map_yaml),
                        "--start-x", str(start.x_m),
                        "--start-y", str(start.y_m),
                        "--start-yaw", str(start.yaw_rad),
                        "--recommended-pose-json", str(recommendation_path),
                        "--route-csv", str(route),
                        "--diagnostics-json", str(diagnostics),
                        "--known-stand-keepout",
                        str(station_a[0]),
                        str(station_a[1]),
                        "0.26",
                    ]
                )
            with route.open() as route_file:
                rows = list(csv.DictReader(route_file))
            points = [
                (float(row["world_x_m"]), float(row["world_y_m"]))
                for row in rows
            ]
            committed = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.0
            )

        self.assertEqual(status, 0, committed.reason)
        self.assertGreater(len(points), 2)
        clearances = [
            self._point_to_segment_distance(station_a, segment_start, segment_end)
            for segment_start, segment_end in zip(points, points[1:])
        ]
        self.assertGreater(min(clearances), 0.26)
        self.assertGreater(
            committed.manifest["safety_diagnostics"][
                "known_stand_keepout_cell_count"
            ],
            0,
        )
        self.assertEqual(
            committed.manifest["safety_diagnostics"]["known_stand_keepouts"],
            [{"x_m": station_a[0], "y_m": station_a[1], "radius_m": 0.26}],
        )
        self.assertEqual(
            committed.manifest["safety_diagnostics"]["known_stand_start_cell"],
            {"x": 52, "y": 28},
        )
        self.assertTrue(
            committed.manifest["safety_diagnostics"][
                "known_stand_start_cell_exempted"
            ]
        )
        egress_anchor = committed.manifest["safety_diagnostics"][
            "known_stand_egress_anchor"
        ]
        self.assertIsNotNone(egress_anchor)
        self.assertTrue(
            committed.manifest["safety_diagnostics"]
            ["known_stand_egress_continuous_clearance_validated"]
        )
        self.assertAlmostEqual(points[0][0], start.x_m)
        self.assertAlmostEqual(points[0][1], start.y_m)
        self.assertAlmostEqual(points[1][0], egress_anchor["x_m"])
        self.assertAlmostEqual(points[1][1], egress_anchor["y_m"])
        self.assertGreater(
            committed.manifest["safety_diagnostics"]
            ["known_stand_keepout_clearances"][0]
            ["minimum_route_clearance_m"],
            0.26,
        )

    def test_known_stand_overlay_exempts_only_safe_exact_start_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text(
                "P2\n100 60\n255\n" + " ".join(["254"] * 6000) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [-2.82, -1.69, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            base = Costmap.from_occupancy_grid(load_occupancy_grid(map_yaml))
            center = Pose2D(-0.395, -0.415)
            radius_m = 0.26
            start = Pose2D(-0.131011, -0.270103)
            start_cell = base.world_to_grid(start)

            self.assertEqual((start_cell.x, start_cell.y), (53, 28))
            self.assertGreater(
                math.hypot(start.x_m - center.x_m, start.y_m - center.y_m),
                radius_m,
            )
            self.assertIn(
                start_cell,
                circular_keepout_cells(base, center, radius_m),
            )

            safe = _known_stand_keepout_costmap(
                base,
                [(center.x_m, center.y_m, radius_m)],
                start=start,
            )
            self.assertTrue(safe.start_cell_exempted)
            self.assertTrue(safe.costmap.is_traversable(start_cell))
            self.assertIsNotNone(safe.egress_anchor)
            self.assertNotIn(
                base.world_to_grid(safe.egress_anchor),
                circular_keepout_cells(base, center, radius_m),
            )
            self.assertGreater(
                self._point_to_segment_distance(
                    (center.x_m, center.y_m),
                    (start.x_m, start.y_m),
                    (safe.egress_anchor.x_m, safe.egress_anchor.y_m),
                ),
                radius_m,
            )
            self.assertEqual(
                safe.rasterized_cell_count,
                safe.blocked_cell_count + 1,
            )

            boundary_start = Pose2D(center.x_m + radius_m, center.y_m)
            boundary_cell = base.world_to_grid(boundary_start)
            boundary = _known_stand_keepout_costmap(
                base,
                [(center.x_m, center.y_m, radius_m)],
                start=boundary_start,
            )
            self.assertFalse(boundary.start_cell_exempted)
            self.assertTrue(boundary.costmap.is_blocked(boundary_cell))

            statically_blocked_base = base.with_blocked_cells([start_cell])
            static = _known_stand_keepout_costmap(
                statically_blocked_base,
                [(center.x_m, center.y_m, radius_m)],
                start=start,
            )
            self.assertFalse(static.start_cell_exempted)
            self.assertTrue(static.costmap.is_blocked(start_cell))

    def test_known_stand_continuous_clearance_rejects_crossing_segment(self):
        crossing = SimpleNamespace(
            waypoints=(
                SimpleNamespace(pose=Pose2D(-0.30, 0.0)),
                SimpleNamespace(pose=Pose2D(0.30, 0.0)),
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "known_stand_keepout_route_clearance_failed",
        ):
            _validate_known_stand_route_clearance(
                crossing,
                ({"x_m": 0.0, "y_m": 0.0, "radius_m": 0.26},),
            )

        safe = SimpleNamespace(
            waypoints=(
                SimpleNamespace(pose=Pose2D(-0.30, 0.27)),
                SimpleNamespace(pose=Pose2D(0.30, 0.27)),
            )
        )
        clearances = _validate_known_stand_route_clearance(
            safe,
            ({"x_m": 0.0, "y_m": 0.0, "radius_m": 0.26},),
        )
        self.assertAlmostEqual(clearances[0]["minimum_route_clearance_m"], 0.27)

    def test_viewpoint_sampling_plans_exact_latched_point_without_face_corridor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text(
                "P2\n80 60\n255\n" + " ".join(["254"] * 4800) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            stand = Pose2D(1.5, 1.5)
            sample = Pose2D(1.5, 1.8, -math.pi / 2.0)
            opposite = Pose2D(1.5, 1.2, math.pi / 2.0)
            recommendation = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(stand, 0.06, 0.02, "lidar_cluster"),
                robot_pose=Pose2D(2.1, 1.5, math.pi),
                axis_confidence=0.0,
                axis_state="viewpoint_sampling",
                face_candidates=(
                    FaceCandidate("sampling_near", math.pi / 2.0, sample, False),
                    FaceCandidate("sampling_far", -math.pi / 2.0, opposite, False),
                ),
                side_evidence=SideEvidence(
                    "none", 0.0, False, False, None, "axis_uncommitted"
                ),
                material_target=MaterialTarget(
                    "sampling_near", sample, "viewpoint_sampling"
                ),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(
                json.dumps(recommendation_to_dict(recommendation))
            )
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                status = main(
                    [
                        "--map", str(map_yaml),
                        "--start-x", "2.1",
                        "--start-y", "1.5",
                        "--recommended-pose-json", str(recommendation_path),
                        "--route-csv", str(route),
                        "--diagnostics-json", str(diagnostics),
                    ]
                )
            committed = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.0
            )
            self.assertEqual(status, 0, committed.reason)
            with route.open() as route_file:
                rows = list(csv.DictReader(route_file))
            diagnostics_payload = json.loads(diagnostics.read_text())

        self.assertEqual(status, 0)
        self.assertAlmostEqual(float(rows[-1]["world_x_m"]), sample.x_m)
        self.assertAlmostEqual(float(rows[-1]["world_y_m"]), sample.y_m)
        self.assertTrue(all(row["corridor"] == "false" for row in rows))
        self.assertTrue(all(row["protected"] == "false" for row in rows))
        self.assertTrue(all(row["route_kind"] == "viewpoint_sampling" for row in rows))
        self.assertEqual(committed.manifest["target"]["face_id"], "sampling_near")
        self.assertEqual(
            committed.manifest["target"]["evidence_state"], "viewpoint_sampling"
        )
        self.assertEqual(
            diagnostics_payload["metadata"]["approach_phase"],
            "viewpoint_sampling",
        )

    def test_sampling_at_point_three_meter_keeps_current_target_free_and_avoids_other_stand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text(
                "P2\n100 60\n255\n" + " ".join(["254"] * 6000) + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            current_stand = Pose2D(1.5, 1.5)
            sampling_target = Pose2D(1.8, 1.5, math.pi)
            other_stand = (2.5, 1.5)
            start = Pose2D(3.4, 1.5, math.pi)
            recommendation = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(
                    current_stand, 0.06, 0.02, "lidar_cluster"
                ),
                robot_pose=start,
                axis_confidence=0.0,
                axis_state="viewpoint_sampling",
                face_candidates=(
                    FaceCandidate(
                        "sampling_near", 0.0, sampling_target, False
                    ),
                    FaceCandidate(
                        "sampling_far",
                        math.pi,
                        Pose2D(1.2, 1.5, 0.0),
                        False,
                    ),
                ),
                side_evidence=SideEvidence(
                    "none", 0.0, False, False, None, "axis_uncommitted"
                ),
                material_target=MaterialTarget(
                    "sampling_near", sampling_target, "viewpoint_sampling"
                ),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(
                json.dumps(recommendation_to_dict(recommendation))
            )
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                status = main(
                    [
                        "--map", str(map_yaml),
                        "--start-x", str(start.x_m),
                        "--start-y", str(start.y_m),
                        "--start-yaw", str(start.yaw_rad),
                        "--recommended-pose-json", str(recommendation_path),
                        "--route-csv", str(route),
                        "--diagnostics-json", str(diagnostics),
                        # Deliberately omit the current stand. This is exactly
                        # the non-target-only contract of the survey runner.
                        "--known-stand-keepout",
                        str(other_stand[0]),
                        str(other_stand[1]),
                        "0.26",
                    ]
                )
            with route.open() as route_file:
                rows = list(csv.DictReader(route_file))
            points = [
                (float(row["world_x_m"]), float(row["world_y_m"]))
                for row in rows
            ]
            committed = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.0
            )

        self.assertEqual(status, 0, committed.reason)
        self.assertAlmostEqual(
            math.hypot(
                sampling_target.x_m - current_stand.x_m,
                sampling_target.y_m - current_stand.y_m,
            ),
            0.30,
        )
        self.assertAlmostEqual(points[-1][0], sampling_target.x_m)
        self.assertAlmostEqual(points[-1][1], sampling_target.y_m)
        self.assertGreater(len(points), 2)
        clearances = [
            self._point_to_segment_distance(
                other_stand, segment_start, segment_end
            )
            for segment_start, segment_end in zip(points, points[1:])
        ]
        self.assertGreater(min(clearances), 0.26)
        self.assertEqual(
            committed.manifest["safety_diagnostics"]["known_stand_keepouts"],
            [{"x_m": 2.5, "y_m": 1.5, "radius_m": 0.26}],
        )

    def test_committed_robot_facing_side_never_falls_back_through_stand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            width, height = 40, 30
            pixels = [254] * (width * height)
            # Occupy grid cell (20, 15), on face_a's east corridor.  PGM rows
            # are vertically flipped into map cells.
            pixels[(height - 1 - 15) * width + 20] = 0
            pgm = root / "map.pgm"
            pgm.write_text(
                f"P2\n{width} {height}\n255\n"
                + " ".join(str(value) for value in pixels)
                + "\n"
            )
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n"
            )
            face_a = Pose2D(1.85, 1.5, math.pi)
            face_b = Pose2D(1.15, 1.5, 0.0)
            rec = SynchronizedViewpointRecommendation(
                schema_version=1,
                simulation_only=True,
                stream_id="sim-stand-viewpoint",
                stand_id="A",
                planning_frame="odom",
                source="synchronized_lidar_camera_viewpoint",
                observation_unix_sec=100.0,
                sensor_stamp_sec=10.0,
                stand=StandGeometry(Pose2D(1.5, 1.5), 0.06, 0.02, "lidar_cluster"),
                robot_pose=Pose2D(0.4, 1.5),
                axis_confidence=0.8,
                axis_state="resolved",
                face_candidates=(
                    FaceCandidate("face_a", 0.0, face_a, True),
                    FaceCandidate("face_b", math.pi, face_b, True),
                ),
                side_evidence=SideEvidence(
                    "none", 0.0, False, False, None, "no_qr_evidence"
                ),
                material_target=MaterialTarget(
                    "face_a", face_a, "ambiguous_axis"
                ),
            )
            recommendation_path = root / "recommendation.json"
            recommendation_path.write_text(json.dumps(recommendation_to_dict(rec)))
            route = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            argv = [
                "--map",
                str(map_yaml),
                "--start-x",
                "0.4",
                "--start-y",
                "1.5",
                "--recommended-pose-json",
                str(recommendation_path),
                "--route-csv",
                str(route),
                "--diagnostics-json",
                str(diagnostics),
                "--standoff-distance-m",
                "0.35",
            ]
            with patch(
                "scripts.aufgabe04.navigation.plan_synchronized_viewpoint.time.time",
                return_value=100.0,
            ):
                self.assertEqual(main(argv), 1)
            withdrawn = read_committed_revision(
                route.with_suffix(".manifest.json"), now_unix_sec=100.0
            )

        self.assertEqual(withdrawn.status, "withdrawn")
        self.assertIn("hard_face", withdrawn.reason)


if __name__ == "__main__":
    unittest.main()
