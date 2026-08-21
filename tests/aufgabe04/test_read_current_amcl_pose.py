import math
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.read_current_amcl_pose import (  # noqa: E402
    CurrentAmclPose,
    CurrentAmclPoseReader,
    planner_args_from_pose,
    pose2d_from_current_amcl_pose,
    read_current_pose2d_from_amcl,
    validate_current_amcl_pose,
    yaw_from_quaternion,
)
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation import read_current_amcl_pose as pose_reader  # noqa: E402


class ReadCurrentAmclPoseTest(unittest.TestCase):
    @staticmethod
    def _pose(**overrides) -> CurrentAmclPose:
        values = {
            "x_m": 0.125,
            "y_m": -0.25,
            "yaw_rad": 0.75,
            "frame_id": "map",
            "topic": "/amcl_pose",
            "header_stamp_sec": 10.0,
            "receipt_age_sec": 0.1,
            "header_age_sec": 0.2,
        }
        values.update(overrides)
        return CurrentAmclPose(**values)

    def test_converts_current_amcl_evidence_to_exact_navigation_pose(self):
        result = pose2d_from_current_amcl_pose(self._pose())

        self.assertEqual(result, Pose2D(0.125, -0.25, 0.75))
        self.assertIs(type(result), Pose2D)

    def test_navigation_pose_adapter_rejects_wrong_type_and_nonfinite_pose(self):
        with self.assertRaisesRegex(TypeError, "requires CurrentAmclPose"):
            pose2d_from_current_amcl_pose(Pose2D(0.0, 0.0, 0.0))
        for field in ("x_m", "y_m", "yaw_rad"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    pose2d_from_current_amcl_pose(
                        self._pose(**{field: math.nan})
                    )

    def test_reads_current_pose2d_through_fresh_amcl_reader(self):
        with patch.object(
            pose_reader,
            "read_current_amcl_pose",
            return_value=self._pose(x_m=1.25, y_m=-0.5, yaw_rad=0.125),
        ) as reader:
            result = read_current_pose2d_from_amcl(
                namespace="/tb3",
                amcl_topic="amcl_pose",
                map_frame="map",
                timeout_sec=4.0,
                max_age_sec=1.5,
            )

        self.assertEqual(result, Pose2D(1.25, -0.5, 0.125))
        reader.assert_called_once_with(
            namespace="/tb3",
            amcl_topic="amcl_pose",
            map_frame="map",
            timeout_sec=4.0,
            max_age_sec=1.5,
            nomotion_update_service="/request_nomotion_update",
        )

    def test_formats_planner_start_args(self):
        pose = CurrentAmclPose(
            x_m=0.1234567,
            y_m=-0.25,
            yaw_rad=1.5707963,
            frame_id="map",
            topic="/amcl_pose",
            header_stamp_sec=10.0,
            receipt_age_sec=0.1,
            header_age_sec=0.2,
        )

        self.assertEqual(
            planner_args_from_pose(pose, precision=3),
            "--start-x 0.123 --start-y -0.250 --start-yaw 1.571",
        )

    def test_validates_frame_and_freshness(self):
        pose = CurrentAmclPose(
            x_m=0.0,
            y_m=0.0,
            yaw_rad=0.0,
            frame_id="/map",
            topic="/amcl_pose",
            header_stamp_sec=10.0,
            receipt_age_sec=0.1,
            header_age_sec=0.2,
        )

        validate_current_amcl_pose(pose, expected_frame="map", max_age_sec=1.0)

        stale = CurrentAmclPose(**{**pose.__dict__, "header_age_sec": 2.0})
        with self.assertRaisesRegex(ValueError, "header age"):
            validate_current_amcl_pose(stale, expected_frame="map", max_age_sec=1.0)

        wrong_frame = CurrentAmclPose(**{**pose.__dict__, "frame_id": "odom"})
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            validate_current_amcl_pose(wrong_frame, expected_frame="map", max_age_sec=1.0)

    def test_yaw_from_quaternion(self):
        yaw = math.pi / 2.0
        q = SimpleNamespace(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )

        self.assertAlmostEqual(yaw_from_quaternion(q), yaw)

    def test_stationary_update_is_requested_once_after_client_is_ready(self):
        future = object()
        client = SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=lambda _request: future,
        )
        messages = []
        reader = CurrentAmclPoseReader.__new__(CurrentAmclPoseReader)
        reader.nomotion_client = client
        reader.nomotion_future = None
        reader.get_logger = lambda: SimpleNamespace(info=messages.append)

        class FakeEmpty:
            class Request:
                pass

        with patch.object(pose_reader, "Empty", FakeEmpty):
            self.assertTrue(reader.maybe_request_nomotion_update())
            self.assertFalse(reader.maybe_request_nomotion_update())

        self.assertIs(reader.nomotion_future, future)
        self.assertEqual(messages, ["requested stationary AMCL update"])

    def test_stationary_update_waits_for_service_readiness(self):
        client = SimpleNamespace(
            service_is_ready=lambda: False,
            call_async=lambda _request: self.fail("service must not be called"),
        )
        reader = CurrentAmclPoseReader.__new__(CurrentAmclPoseReader)
        reader.nomotion_client = client
        reader.nomotion_future = None

        self.assertFalse(reader.maybe_request_nomotion_update())
        self.assertIsNone(reader.nomotion_future)


if __name__ == "__main__":
    unittest.main()
