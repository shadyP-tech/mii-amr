import json
import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_synchronized_viewpoint import (
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


class PlanSynchronizedViewpointTest(unittest.TestCase):
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
            aliases_exist = route.exists() and diagnostics.exists()

        self.assertEqual(status, 0)
        self.assertEqual(restart_status, 0)
        self.assertEqual(committed.status, "active")
        self.assertEqual(committed.route_revision, first_committed.route_revision)
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

        self.assertEqual(status, 0)
        self.assertAlmostEqual(float(rows[-1]["world_x_m"]), near.x_m)
        self.assertAlmostEqual(float(rows[-1]["world_y_m"]), near.y_m)
        self.assertTrue(all(row["corridor"] == "false" for row in rows))
        self.assertTrue(all(row["route_kind"] == "axis_acquisition" for row in rows))
        self.assertEqual(committed.manifest["target"]["face_id"], "acquisition_near")

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
