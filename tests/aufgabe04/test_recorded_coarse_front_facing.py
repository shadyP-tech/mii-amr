"""Recorded geometry policy replay, not an authenticated mission replay.

The viewer had QR decoding disabled and supplied no mission TF timeline here.
Explicit coarse permission below is a test input, not inferred from its pixels.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.artifacts.bounded_orientation import (
    BOUNDED_ORIENTATION_POLICY,
    COARSE_FRONT_ORIENTATION_POLICY,
    MAXIMUM_QR_VIEW_OBLIQUITY_RAD,
    validate_bounded_endpoint,
    validated_bounded_orientation,
)
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import (
    CurrentHeadOrientationBounds,
    HeadOrientationHypothesis,
    validated_current_head_orientation_bounds,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.bounded_head_window import enclose_intervals


FIXTURE = Path(__file__).parent / "fixtures/coarse_front_facing_20261001/inputs.json"


def measured_bounds(payload):
    values = dict(payload)
    values["hypotheses"] = tuple(HeadOrientationHypothesis(**{
        key: tuple(value) if key in {"rotation_vector", "translation_xyz_m", "face_normal_xyz"} else value
        for key, value in hypothesis.items()
    }) for hypothesis in payload["hypotheses"])
    values["corners"] = tuple(ImagePoint(**point) for point in payload["corners"])
    for key in ("frame_shape", "camera_matrix", "head_size_m"):
        values[key] = tuple(values[key])
    return CurrentHeadOrientationBounds(**values)


class RecordedCoarseFrontFacingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def payload(self):
        intervals = [(frame["head_orientation_bounds"]["center_rad"],
                      frame["head_orientation_bounds"]["half_width_rad"])
                     for frame in self.fixture["frames"]]
        center, half = enclose_intervals(intervals)
        return {"policy": COARSE_FRONT_ORIENTATION_POLICY, "center_rad": center,
                "half_width_rad": half, "sample_count": len(intervals)}

    def endpoint(self, *, radial_offset_deg=0.0, stand_uncertainty_m=None):
        normal = self.payload()["center_rad"] + math.pi / 2
        scenario = self.fixture["endpoint_scenario"]
        radius = scenario["center_distance_m"]
        radial = normal + math.radians(radial_offset_deg)
        return {"selected_normal_rad": normal, "stand_x_m": 0., "stand_y_m": 0.,
                "stand_uncertainty_m": (scenario["stand_uncertainty_m"]
                                         if stand_uncertainty_m is None else stand_uncertainty_m),
                "target_x_m": radius * math.cos(radial),
                "target_y_m": radius * math.sin(radial), "expected_sample_count": 7}

    def test_seven_original_records_preserve_signed_bounds_and_every_alternative(self):
        frames = self.fixture["frames"]
        self.assertEqual([frame["frame_index"] for frame in frames], list(range(7)))
        stamps = [frame["source_stamp_sec"] for frame in frames]
        self.assertEqual(len(set(stamps)), 7)
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len({frame["source_frame"] for frame in frames}), 7)
        self.assertAlmostEqual(stamps[-1] - stamps[0], .7998490333557129)
        self.assertTrue(self.fixture["provenance"]["no_qr_decode"])
        self.assertFalse(self.fixture["provenance"]["tf_timeline_in_fixture"])
        payload = self.payload()
        for frame in frames:
            with self.subTest(frame_index=frame["frame_index"]):
                recorded = frame["head_orientation_bounds"]
                signed = measured_bounds(recorded)
                self.assertTrue(validated_current_head_orientation_bounds(signed))
                self.assertEqual(len(signed.hypotheses), 2)
                self.assertEqual(signed.noise_allowance_multiplier, 3.)
                # This independently checks the original production binding;
                # rewriting a measured uncertainty cannot retain the source SHA.
                contents = {key: value for key, value in recorded.items() if key != "binding_sha256"}
                self.assertEqual(hashlib.sha256(json.dumps(contents, sort_keys=True, allow_nan=False,
                    separators=(",", ":")).encode()).hexdigest(), signed.binding_sha256)
                for hypothesis in signed.hypotheses:
                    offset = (hypothesis.yaw_rad - payload["center_rad"] + math.pi / 2) % math.pi - math.pi / 2
                    self.assertLessEqual(abs(offset) + 3 * hypothesis.yaw_std_rad,
                                         payload["half_width_rad"] + 1e-12)
        expected = self.fixture["expected_full_union"]
        self.assertAlmostEqual(payload["center_rad"], expected["center_rad"], places=14)
        self.assertAlmostEqual(payload["half_width_rad"], expected["half_width_rad"], places=14)
        self.assertAlmostEqual(math.degrees(payload["half_width_rad"]), 19.78878162939927)

    def test_complete_interval_plus_position_reserve_fits_thirty_but_not_twenty(self):
        payload = self.payload()
        original = copy.deepcopy(payload)
        evidence = validate_bounded_endpoint(payload, allow_coarse_front=True, **self.endpoint())
        position = math.asin((.03 + .02) / .35)
        self.assertAlmostEqual(evidence["worst_case_view_obliquity_rad"], payload["half_width_rad"] + position)
        self.assertAlmostEqual(math.degrees(evidence["worst_case_view_obliquity_rad"]), 28.00199233113746)
        self.assertGreater(evidence["worst_case_view_obliquity_rad"], MAXIMUM_QR_VIEW_OBLIQUITY_RAD)
        self.assertAlmostEqual(evidence["maximum_view_obliquity_rad"], math.radians(30))
        self.assertAlmostEqual(evidence["total_position_reserve_m"], .05)
        self.assertEqual(evidence["bounded_orientation"], payload)
        self.assertEqual(payload, original)
        self.assertFalse(evidence["motion_authorized"])
        # The old policy rejects this unchanged measurement at its 15 degree
        # representation cap, before its stricter 20 degree endpoint limit.
        legacy = {**payload, "policy": BOUNDED_ORIENTATION_POLICY}
        with self.assertRaisesRegex(ValueError, "half width exceeds 15"):
            validate_bounded_endpoint(legacy, **self.endpoint())

    def test_real_endpoint_offset_is_part_of_the_whole_interval_check(self):
        payload = self.payload()
        near = validate_bounded_endpoint(payload, allow_coarse_front=True,
                                         **self.endpoint(radial_offset_deg=1))
        self.assertLess(near["worst_case_view_obliquity_rad"], math.radians(30))
        # Moving only this hypothetical endpoint three degrees off the minimax
        # direction exceeds 30 degrees even though the midpoint alone looks good.
        with self.assertRaisesRegex(ValueError, "viewing obliquity"):
            validate_bounded_endpoint(payload, allow_coarse_front=True,
                                      **self.endpoint(radial_offset_deg=3))
        with self.assertRaisesRegex(ValueError, "viewing obliquity"):
            validate_bounded_endpoint(payload, allow_coarse_front=True,
                                      **self.endpoint(stand_uncertainty_m=.04))

    def test_coarse_permission_and_seven_samples_remain_required(self):
        payload = self.payload()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validated_bounded_orientation(payload)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_bounded_endpoint(payload, **self.endpoint())
        with self.assertRaisesRegex(ValueError, "7 to 32"):
            validated_bounded_orientation({**payload, "sample_count": 6}, allow_coarse_front=True)
        parsed = validated_bounded_orientation(payload, allow_coarse_front=True)
        self.assertEqual(parsed.half_width_rad, payload["half_width_rad"])
        self.assertEqual(parsed.sample_count, 7)


if __name__ == "__main__":
    unittest.main()
