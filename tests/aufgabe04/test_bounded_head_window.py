"""Intervals preserve ambiguity without promoting stale or unstable heads."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.real_robot.observer.bounded_head_window import (
    BoundedHeadSample, BoundedHeadWindow, enclose_intervals,
)


def update(*, accepted=True, epoch=0, poisoned=False, target="stand", qr=True):
    return SimpleNamespace(frame_accepted=accepted, qr_sample_accepted=qr,
        snapshot=SimpleNamespace(target_key=target, motion_epoch=epoch, poisoned=poisoned,
                                 tentative_qr_id="QR_002" if qr else None,
                                 latched_qr_id="QR_002" if qr else None))


def sample(index, **changes):
    return replace(BoundedHeadSample(
        10. + index * .2, math.radians(7), math.radians(9), "a" * 64, (640., 640.),
        "front", "QR_002", ((350., 250.), (450., 250.), (450., 350.), (350., 350.)),
        (400., 300.), 100.), **changes)


class BoundedHeadWindowTests(unittest.TestCase):
    def feed(self, window, indices, **changes):
        result = None
        for i in indices:
            current = sample(i, **changes)
            result = window.observe(current, update=update(), observed_at_sec=current.stamp_sec + .1)
        return result

    def test_seven_distinct_frames_keep_noise_allowance(self):
        window = BoundedHeadWindow()
        self.assertIsNone(self.feed(window, range(6)))
        receipt = self.feed(window, (6,))
        self.assertEqual(receipt["sample_count"], 7)
        self.assertAlmostEqual(receipt["half_width_rad"], math.radians(9))
        self.assertFalse(window.metadata["motion_authorized"])

    def test_all_intervals_are_retained_not_averaged_or_outlier_filtered(self):
        window = BoundedHeadWindow()
        self.feed(window, range(6))
        self.assertIsNone(self.feed(window, (6,), axis_center_rad=math.radians(30)))
        self.assertGreater(window.metadata["half_width_rad"], math.radians(15))
        self.assertIsNone(self.feed(window, range(7, 13)))
        self.assertIsNotNone(self.feed(window, (13,)))

    def test_border_switch_is_retained_until_eviction_independently_of_angles(self):
        window = BoundedHeadWindow()
        self.feed(window, range(6))
        shifted = tuple((x + 5, y) for x, y in sample(6).corners)
        self.assertIsNone(self.feed(window, (6,), corners=shifted))
        self.assertEqual(window.metadata["reason"], "head_border_choice_unstable")
        self.assertIsNone(self.feed(window, range(7, 13)))
        self.assertIsNotNone(self.feed(window, (13,)))

    def test_common_image_projection_shift_is_not_a_border_switch(self):
        window = BoundedHeadWindow()
        self.feed(window, range(6))
        shifted = tuple((x + 5, y + 3) for x, y in sample(6).corners)
        self.assertIsNotNone(self.feed(window, (6,), corners=shifted, projected_center_px=(405., 303.)))

    def test_duplicates_soft_misses_and_stale_frames_do_not_supply_samples(self):
        window = BoundedHeadWindow()
        self.feed(window, range(6))
        self.assertIsNone(self.feed(window, (5,)))
        self.assertIsNone(window.observe(sample(6), update=update(accepted=False), observed_at_sec=11.3))
        self.assertIsNone(window.observe(None, update=update(), observed_at_sec=11.4))
        self.assertIsNone(window.observe(sample(6), update=update(), observed_at_sec=20.))
        self.assertIsNone(self.feed(window, (7,)))

    def test_motion_target_calibration_face_and_identity_do_not_mix(self):
        for field, change in (("epoch", 1), ("target", "new"),
                              ("camera_signature", (600., 600.)), ("model_sha256", "b" * 64),
                              ("qr_id", "QR_003")):
            with self.subTest(field=field):
                window = BoundedHeadWindow()
                self.feed(window, range(6))
                current = sample(6, **({field: change} if field not in {"epoch", "target"} else {}))
                gate = update(**({field: change} if field in {"epoch", "target"} else {}))
                self.assertIsNone(window.observe(current, update=gate, observed_at_sec=11.3))
                self.assertEqual(window.metadata["sample_count"], 1)

    def test_poison_and_qr_veto_backside(self):
        window = BoundedHeadWindow()
        self.feed(window, range(6))
        self.assertIsNone(window.observe(sample(6), update=update(poisoned=True), observed_at_sec=11.3))
        self.assertIsNone(self.feed(window, (7,)))
        back = sample(8, face="backside", qr_id=None)
        self.assertIsNone(window.observe(back, update=update(), observed_at_sec=11.7))
        self.assertIsNone(window.observe(back, update=update(qr=False), observed_at_sec=11.7))
        self.assertEqual(window.metadata["sample_count"], 1)

    def test_axial_wrap_keeps_whole_arcs_and_rejects_unbounded_union(self):
        center, half = enclose_intervals(((math.radians(89), math.radians(4)),
                                          (math.radians(-89), math.radians(4))))
        self.assertAlmostEqual(abs(center), math.pi / 2)
        self.assertAlmostEqual(half, math.radians(5))
        self.assertIsNone(enclose_intervals(((0., math.radians(80)),
                                             (math.pi / 2, math.radians(80)))))


if __name__ == "__main__":
    unittest.main()
