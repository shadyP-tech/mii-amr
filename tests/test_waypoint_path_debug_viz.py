import argparse
import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import waypoint_path_debug_viz as path_viz  # noqa: E402


def write_waypoints(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "world_x_m", "world_y_m"])
        writer.writerows(rows)


class FakeStamp:
    sec = 0
    nanosec = 0


class FakeHeader:
    def __init__(self):
        self.frame_id = ""
        self.stamp = None


class FakeVector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class FakeQuaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0


class FakePose:
    def __init__(self):
        self.position = FakeVector3()
        self.orientation = FakeQuaternion()


class FakeColor:
    def __init__(self):
        self.r = 0.0
        self.g = 0.0
        self.b = 0.0
        self.a = 0.0


class FakePoint(FakeVector3):
    pass


class FakePoseStamped:
    def __init__(self):
        self.header = FakeHeader()
        self.pose = FakePose()


class FakeNavPath:
    def __init__(self):
        self.header = FakeHeader()
        self.poses = []


class FakeMarker:
    ADD = 0
    SPHERE = 2
    DELETEALL = 3
    CUBE_LIST = 6
    SPHERE_LIST = 7
    TEXT_VIEW_FACING = 9

    def __init__(self):
        self.header = FakeHeader()
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = self.ADD
        self.pose = FakePose()
        self.scale = FakeVector3()
        self.color = FakeColor()
        self.points = []
        self.text = ""


class FakeMarkerArray:
    def __init__(self, markers=None):
        self.markers = list(markers or [])


def install_fake_rviz_messages(testcase):
    follower = path_viz.follower
    originals = {
        "Point": follower.Point,
        "PoseStamped": follower.PoseStamped,
        "NavPath": follower.NavPath,
        "Marker": follower.Marker,
        "MarkerArray": follower.MarkerArray,
    }
    follower.Point = FakePoint
    follower.PoseStamped = FakePoseStamped
    follower.NavPath = FakeNavPath
    follower.Marker = FakeMarker
    follower.MarkerArray = FakeMarkerArray

    def restore():
        for name, value in originals.items():
            setattr(follower, name, value)

    testcase.addCleanup(restore)


class WaypointPathDebugVizTest(unittest.TestCase):
    def test_cli_defaults_match_follower_topics(self):
        parser = path_viz.build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.waypoints, Path("results/aufgabe03/aufgabe03_waypoints.csv"))
        self.assertEqual(args.path_topic, "/mii_amr/planned_path")
        self.assertEqual(args.waypoint_marker_topic, "/mii_amr/planned_waypoints")
        self.assertEqual(args.map_frame, "map")
        self.assertFalse(args.skip_first_waypoint)
        self.assertTrue(args.watch_file)

    def test_load_display_waypoints_can_skip_first_waypoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.csv"
            write_waypoints(
                path,
                [
                    [0, 0.0, 0.0],
                    [1, 0.2, 0.0],
                    [2, 0.4, 0.0],
                ],
            )
            args = argparse.Namespace(
                waypoints=path,
                skip_first_waypoint=True,
                min_waypoint_spacing_m=0.0,
            )

            waypoints = path_viz.load_display_waypoints(args)

        self.assertEqual([waypoint.index for waypoint in waypoints], [1, 2])

    def test_build_route_messages_uses_path_and_marker_topics_payloads(self):
        install_fake_rviz_messages(self)
        args = argparse.Namespace(
            map_frame="map",
            current_waypoint_index=1,
        )
        waypoints = [
            path_viz.follower.Waypoint(0, 0.0, 0.0),
            path_viz.follower.Waypoint(1, 0.3, 0.0),
            path_viz.follower.Waypoint(2, 0.3, 0.3),
        ]

        path_msg, marker_msg = path_viz.build_route_messages(
            args,
            waypoints,
            FakeStamp(),
        )

        self.assertEqual(path_msg.header.frame_id, "map")
        self.assertEqual(len(path_msg.poses), 3)
        self.assertEqual(path_msg.poses[-1].pose.position.y, 0.3)
        marker_namespaces = [marker.ns for marker in marker_msg.markers]
        self.assertIn("planned_waypoints", marker_namespaces)
        self.assertIn("current_waypoint", marker_namespaces)
        self.assertIn("goal_waypoint", marker_namespaces)
        labels = [
            marker.text
            for marker in marker_msg.markers
            if marker.ns == "planned_waypoint_labels"
        ]
        self.assertEqual(labels, ["0", "1", "2"])

    def test_validation_rejects_invalid_values(self):
        parser = path_viz.build_arg_parser()
        invalid = [
            ["--min-waypoint-spacing-m", "-0.1"],
            ["--current-waypoint-index", "-1"],
            ["--watch-period-sec", "0"],
        ]

        for argv in invalid:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        path_viz.validate_args(parser, args)


if __name__ == "__main__":
    unittest.main()
