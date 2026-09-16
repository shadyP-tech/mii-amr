"""Live ROS-shaped calibration must validate before JSON serialization."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy

from scripts.aufgabe04.artifacts.current_head_front_observation import _camera_value
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import _camera_context
from scripts.aufgabe04.real_robot.observer.camera_context import camera_context_signature
from tests.aufgabe04 import test_immediate_front_observation as front_fixtures
from tests.aufgabe04 import test_qr_observation_pose as qr_fixtures


def camera_info():
    return SimpleNamespace(
        k=numpy.array([640., 0., 400., 0., 640., 300., 0., 0., 1.]),
        d=numpy.array([-.01, .02, 0., 0., 0.]),
        r=numpy.eye(3).ravel(),
        p=numpy.array([640., 0., 400., 0., 0., 640., 300., 0., 0., 0., 1., 0.]),
        distortion_model="plumb_bob",
    )


def signature(info=None, **changes):
    return camera_context_signature(**{
        "camera_frame": "camera", "intrinsics": numpy.array([640., 640., 400., 300.]),
        "camera_info": camera_info() if info is None else info,
        "scan_translation": numpy.array([0., 0., .1]),
        "scan_rotation": numpy.array([0., 0., 0., 1.]), **changes,
    })


class CameraContextTests(unittest.TestCase):
    def test_live_numpy_arrays_produce_strict_immutable_context_before_serialization(self):
        info = camera_info()
        current = signature(info)
        self.assertIs(type(info.k[0]), numpy.float64)
        self.assertTrue(_camera_context(current))
        self.assertTrue(_camera_value(current))
        self.assertIs(type(current[1]), float)
        for values in (*current[5:9], *current[10:]):
            self.assertIs(type(values), tuple)
            self.assertTrue(all(type(value) is float for value in values))
        info.k[0] = 641.
        self.assertEqual(current[5][0], 640.)
        self.assertNotEqual(current, signature(info))

    def test_calibration_and_extrinsic_changes_remain_distinct(self):
        original = camera_info()
        current = signature(original)
        for field in ("k", "d", "r", "p"):
            with self.subTest(field=field):
                changed = deepcopy(original)
                getattr(changed, field)[0] += .001
                self.assertNotEqual(current, signature(changed))
        for changes in ({"camera_frame": "other_camera"},
                        {"intrinsics": (641., 640., 400., 300.)},
                        {"scan_translation": (.001, 0., .1)},
                        {"scan_rotation": (.001, 0., 0., 1.)}):
            with self.subTest(changes=changes):
                self.assertNotEqual(current, signature(original, **changes))
        original.distortion_model = ""
        self.assertNotEqual(current, signature(original))
        self.assertTrue(_camera_context(signature(original)))

    def test_nonfinite_and_nonreal_values_are_not_normalized_into_calibration(self):
        for bad in (float("nan"), numpy.inf, -numpy.inf, True, numpy.bool_(False),
                    "640", 640 + 0j):
            for field in ("k", "d", "r", "p"):
                with self.subTest(field=field, bad=bad):
                    info = camera_info()
                    setattr(info, field, (bad,))
                    with self.assertRaises(ValueError):
                        signature(info)
            for field in ("intrinsics", "scan_translation", "scan_rotation"):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        signature(**{field: (bad,)})

    def test_live_context_publishes_immediate_front_receipt_without_json_roundtrip(self):
        fixture = front_fixtures.ImmediateFrontObservationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        current = signature()
        update, metadata = fixture.frame(camera_signature=current)
        self.assertTrue(update.frame_accepted)
        self.assertIsNotNone(fixture.recommendation(), metadata)
        self.assertTrue(fixture.adapter.completed)
        receipt = fixture.recommendation().axis_measurement
        self.assertEqual(receipt["policy"], "current_head_and_bound_qr")
        self.assertEqual(receipt["camera_signature"][5], list(current[5]))

    def test_live_context_publishes_qr_fallback_without_json_roundtrip(self):
        fixture = qr_fixtures.QrObservationPoseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.adapter.args.qr_pose_fallback_delay_sec = 0.
        current = signature()
        update, metadata = fixture.frame(camera_signature=current)
        self.assertTrue(update.frame_accepted)
        self.assertTrue(update.qr_sample_accepted)
        result = fixture.result()
        self.assertIsNotNone(result, metadata)
        self.assertIsNone(result["stand_axis_rad"])
        self.assertFalse(result["facing_ready"])
        self.assertFalse(result["motion_authorized"])
        self.assertEqual(result["camera_signature"][5], list(current[5]))

    def test_invalid_live_calibration_clears_receipts_and_hints_without_image_processing(self):
        cases = (("k", 0, numpy.nan), ("d", 0, numpy.inf),
                 ("r", 0, numpy.nan), ("p", 1, numpy.nan), ("p", 0, numpy.nan))
        for field, index, bad in cases:
            with self.subTest(field=field, index=index):
                fixture = front_fixtures.ImmediateFrontObservationTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture.frame(usable=False, publish=False)
                adapter = fixture.adapter
                self.assertIsNotNone(adapter._immediate_front_admission.identity)
                tracking = adapter._candidate_search()
                tracking._hint = object()
                adapter.model_pose_tracker.accept(object(), now_sec=100.,
                    profile_sha256=adapter.stand_model_profile.sha256,
                    camera_signature=(400., 400., 400., 300.))
                adapter._qr_observation_pose_fallback = object()
                info = adapter._next_sensor_tuple.return_value.camera_info.value
                live = camera_info()
                for name in ("k", "d", "r", "p", "distortion_model"):
                    setattr(info, name, getattr(live, name))
                getattr(info, field)[index] = bad
                module = "scripts.aufgabe04.real_robot.observer.node."
                with patch(module + "camera_info_mismatches", return_value=()), \
                     patch(module + "transform_mismatches", return_value=()), \
                     patch(module + "compressed_msg_to_bgr_frame") as decode:
                    adapter._process_latest()
                decode.assert_not_called()
                self.assertEqual(adapter._write_status.call_args.args, ("camera_context_invalid",))
                self.assertIsNone(adapter.observation_evidence)
                self.assertIsNone(adapter._immediate_front_admission)
                self.assertIsNone(adapter._qr_observation_pose_fallback)
                self.assertIsNone(adapter.model_pose_tracker._pose)
                self.assertIsNone(tracking._hint)
                self.assertIsNone(adapter.tf_retry_scheduler.pending_frame)
                self.assertFalse(adapter.completed)
                self.assertFalse(adapter.args.recommended_pose_json.exists())

    def test_unparseable_camera_info_is_discarded_without_a_crash(self):
        fixture = front_fixtures.ImmediateFrontObservationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        adapter = fixture.adapter
        with patch("scripts.aufgabe04.real_robot.observer.node.camera_info_mismatches",
                   side_effect=ValueError("malformed calibration coefficient")):
            adapter._process_latest()
        self.assertEqual(adapter._write_status.call_args.args, ("camera_context_invalid",))
        self.assertIsNone(adapter.tf_retry_scheduler.pending_frame)
        self.assertFalse(adapter.completed)


if __name__ == "__main__":
    unittest.main()
