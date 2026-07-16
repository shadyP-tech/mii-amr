import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.ros_runtime_config import (  # noqa: E402
    RuntimeConfig,
    resolve_runtime_config,
)
from scripts.aufgabe04.navigation.run_single_station_segment import build_parser  # noqa: E402
from scripts.aufgabe04.navigation.safety_checks import (  # noqa: E402
    validate_route_diagnostics_json,
    validate_speed_limits,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg  # noqa: E402


ROUTE_HEADER = (
    "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
    "segment_length_m,cumulative_length_m\n"
)


def write_route(path, rows):
    path.write_text(ROUTE_HEADER + "\n".join(rows) + "\n")


def write_diagnostics(path, *, status="ok", failure=None, count=2, length=0.5):
    path.write_text(
        json.dumps(
            {
                "legs": [
                    {
                        "diagnostics": {"status": status, "route_length_m": length},
                        "failure": failure,
                        "route_length_m": length,
                        "route_point_count": count,
                    }
                ]
            }
        )
    )


class WaypointCsvTest(unittest.TestCase):
    def test_loads_selected_leg_and_thins_deterministically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,1,1,0,0.05,0.0,0.05,0.05",
                    "0,2,2,0,0.10,0.0,0.05,0.10",
                    "0,3,3,0,0.20,0.0,0.10,0.20",
                ],
            )

            leg = load_route_leg(route_csv, 0, thinning_min_spacing_m=0.11)

        self.assertEqual(len(leg.raw_waypoints), 4)
        self.assertEqual([wp.point_index for wp in leg.executable_waypoints], [0, 3])

    def test_rejects_missing_header_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            route_csv.write_text("leg_index,point_index\n0,0\n")

            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_route_leg(route_csv, 0)

    def test_rejects_non_contiguous_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,2,2,0,0.10,0.0,0.10,0.10",
                ],
            )

            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_route_leg(route_csv, 0)

    def test_rejects_nan_coordinate_and_missing_leg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(route_csv, ["0,0,0,0,nan,0.0,0.0,0.0"])

            with self.assertRaisesRegex(ValueError, "finite"):
                load_route_leg(route_csv, 0, require_motion=False)
            with self.assertRaisesRegex(ValueError, "not found"):
                load_route_leg(route_csv, 5, require_motion=False)

    def test_rejects_zero_length_motion_but_allows_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(route_csv, ["0,0,0,0,0.0,0.0,0.0,0.0"])

            with self.assertRaisesRegex(ValueError, "fewer than two"):
                load_route_leg(route_csv, 0)
            leg = load_route_leg(route_csv, 0, require_motion=False)

        self.assertEqual(leg.route_length_m, 0.0)

    def test_protected_corridor_points_survive_thinning_and_provenance_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            route_csv.write_text(
                "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,yaw_rad,"
                "segment_length_m,cumulative_length_m,protected,simulation_only,"
                "route_kind,stream_id,route_revision,target_revision,manifest_path\n"
                "0,0,0,0,0.0,0.0,,0.0,0.0,false,true,synchronized_viewpoint,s1,4,2,route.manifest.json\n"
                "0,1,1,0,0.05,0.0,,0.05,0.05,true,true,synchronized_viewpoint,s1,4,2,route.manifest.json\n"
                "0,2,2,0,0.10,0.0,0.0,0.05,0.10,true,true,synchronized_viewpoint,s1,4,2,route.manifest.json\n"
            )

            leg = load_route_leg(route_csv, 0, thinning_min_spacing_m=0.20)

        self.assertEqual([waypoint.point_index for waypoint in leg.executable_waypoints], [0, 1, 2])
        self.assertTrue(leg.simulation_only)
        self.assertEqual(leg.route_kind, "synchronized_viewpoint")
        self.assertEqual(leg.stream_id, "s1")
        self.assertEqual(leg.route_revision, 4)
        self.assertEqual(leg.target_revision, 2)
        self.assertEqual(leg.manifest_path, Path("route.manifest.json"))


class DiagnosticsGateTest(unittest.TestCase):
    def test_accepts_matching_ok_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics)

            status = validate_route_diagnostics_json(diagnostics, 0, csv_point_count=2)

        self.assertTrue(status.ok)

    def test_rejects_failed_or_mismatched_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics, status="failed", failure={"reason": "blocked"}, count=3)

            status = validate_route_diagnostics_json(diagnostics, 0, csv_point_count=2)

        self.assertFalse(status.ok)
        self.assertGreaterEqual(len(status.failures), 3)

    def test_rejects_zero_length_motion_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics, count=1, length=0.0)

            status = validate_route_diagnostics_json(diagnostics, 0, csv_point_count=1)
            noop_status = validate_route_diagnostics_json(
                diagnostics,
                0,
                csv_point_count=1,
                require_motion=False,
            )

        self.assertFalse(status.ok)
        self.assertTrue(noop_status.ok)

    def test_rejects_unsafe_speed_caps(self):
        self.assertTrue(validate_speed_limits(0.05, 0.10).ok)
        self.assertFalse(validate_speed_limits(0.20, 0.10).ok)
        self.assertFalse(validate_speed_limits(0.05, math.inf).ok)


class SegmentRunnerCliGateTest(unittest.TestCase):
    def test_physical_face_route_has_tighter_default_goal_tolerance(self):
        args = build_parser().parse_args(["--leg-index", "0"])

        self.assertEqual(args.goal_tolerance_m, 0.08)
        self.assertEqual(args.viewpoint_sampling_goal_tolerance_m, 0.01)
        self.assertEqual(args.physical_goal_tolerance_m, 0.03)

    def test_yes_bypass_argument_is_rejected(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--leg-index", "0", "--yes"])


class RuntimeConfigTest(unittest.TestCase):
    def test_namespaces_relative_topics_only(self):
        resolved = resolve_runtime_config(
            RuntimeConfig(
                namespace="robot1",
                scan_topic="scan",
                odom_topic="/odom",
                cmd_vel_topic="cmd_vel",
                amcl_topic="/amcl_pose",
            )
        )

        self.assertEqual(resolved.scan_topic, "/robot1/scan")
        self.assertEqual(resolved.odom_topic, "/odom")
        self.assertEqual(resolved.cmd_vel_topic, "/robot1/cmd_vel")
        self.assertEqual(resolved.amcl_topic, "/amcl_pose")
        self.assertEqual(resolved.base_frame, "base_footprint")


if __name__ == "__main__":
    unittest.main()
