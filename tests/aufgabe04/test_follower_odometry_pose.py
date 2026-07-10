import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.simple_waypoint_follower import (  # noqa: E402
    pose_from_odometry,
    remaining_route_distance,
)
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402


def odometry(*, header_frame="odom", child_frame="base_footprint", yaw=0.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=header_frame),
        child_frame_id=child_frame,
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.25, y=-0.4),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw / 2.0),
                    w=math.cos(yaw / 2.0),
                ),
            )
        ),
    )


class FollowerOdometryPoseTest(unittest.TestCase):
    def test_remaining_distance_uses_current_target_and_future_segments(self):
        remaining = remaining_route_distance(
            Pose2D(0.5, 0.0),
            (Pose2D(0.0, 0.0), Pose2D(1.0, 0.0), Pose2D(1.0, 1.0)),
            1,
        )

        self.assertAlmostEqual(remaining, 1.5)
        self.assertAlmostEqual(
            remaining_route_distance(
                Pose2D(1.0, 1.0),
                (Pose2D(0.0, 0.0), Pose2D(1.0, 1.0)),
                1,
            ),
            0.0,
        )

    def test_sim_time_control_loop_does_not_block_on_ros_rate_sleep(self):
        source = (
            ROOT
            / "scripts"
            / "aufgabe04"
            / "navigation"
            / "simple_waypoint_follower.py"
        ).read_text()

        self.assertNotIn("create_rate(", source)
        self.assertNotIn("rate.sleep()", source)
        self.assertIn("def _spin_control_period", source)

    def test_uses_odometry_pose_when_configured_frames_match(self):
        pose = pose_from_odometry(
            odometry(yaw=0.7),
            odom_frame="/odom",
            base_frame="/base_footprint",
        )

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.x_m, 1.25)
        self.assertAlmostEqual(pose.y_m, -0.4)
        self.assertAlmostEqual(pose.yaw_rad, 0.7)

    def test_rejects_mismatched_header_or_child_frame(self):
        self.assertIsNone(
            pose_from_odometry(
                odometry(header_frame="map"),
                odom_frame="odom",
                base_frame="base_footprint",
            )
        )
        self.assertIsNone(
            pose_from_odometry(
                odometry(child_frame="base_link"),
                odom_frame="odom",
                base_frame="base_footprint",
            )
        )


if __name__ == "__main__":
    unittest.main()
