"""The current measured head has independent quality and face admission."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    HeadModelQuality, MEASURED_HEAD_AXIS_SOURCE,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.real_robot.observer.axis_sample_policy import admit_axis_sample
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose, PassiveObserverEvidence
from scripts.aufgabe04.real_robot.observer.front_observation import front_observation_decision
from scripts.aufgabe04.real_robot.observer.head_model_admission import (
    measured_head_front_is_current, measured_head_needs_full_qr_decode,
)
from scripts.aufgabe04.real_robot.observer.qr_target_binding import QrTargetBinding
from tests.aufgabe04 import test_observer_axis_sample_policy as fixtures


def quality(**changes):
    """Synthetic admitted contract; this does not certify any recorded frame."""
    values = dict(
        accepted=True, reason="current_measured_head_observable",
        raw_border_support_mean=.98, raw_corner_support_accepted=True,
        centered_neck_supported=True, minimum_edge_length_px=35.,
        reprojection_rmse_px=.457, ambiguity_gap_px=2.094,
        axis_ambiguous=False, all_corners_positive_depth=True,
        yaw_std_deg=1.463, max_yaw_std_deg=3., corner_sigma_px=.75,
        jacobian_condition_number=120., head_size_m=(.078, .078),
        profile_sha256="a" * 64,
        neck_junction_verified=True,
    )
    return HeadModelQuality(**{**values, **changes})


def head_estimate(yaw_deg=37.815, **changes):
    return fixtures.estimate(**{
        "source": MEASURED_HEAD_AXIS_SOURCE, "yaw_deg": yaw_deg,
        "reason": "axis_estimated_current_measured_head",
        "pose_reprojection_rmse_px": .457, "pose_ambiguity_gap_px": 2.094,
        **changes,
    })


def head_debug(**changes):
    # Keep the admission fixture independent of image-producing constructors.
    values = dict(vars(fixtures.debug()))
    values.update(model_pose_fit_source=MEASURED_HEAD_AXIS_SOURCE,
                  head_model_quality=quality(), qr_detected=False)
    return SimpleNamespace(**{**values, **changes})


class HeadModelAdmissionTests(unittest.TestCase):
    def admit(self, yaw_deg=37.815, **changes):
        fields = dict(estimate=head_estimate(yaw_deg), debug=head_debug(),
                      yaw_rad=math.radians(yaw_deg), qr_texts=(),
                      lidar_target_associated=True,
                      conditioning=axis_conditioning(math.radians(yaw_deg)))
        return admit_axis_sample(**{**fields, **changes})

    def test_measured_quality_has_no_legacy_35_degree_cap(self):
        for yaw_deg in (18., 35.01, 37.815, 60., 74.):
            with self.subTest(yaw_deg=yaw_deg):
                admission = self.admit(yaw_deg)
                self.assertTrue(admission.accepted)
                self.assertEqual(admission.source, MEASURED_HEAD_AXIS_SOURCE)
                self.assertEqual(admission.reason, "measured_head_geometry_quality_accepted")
                self.assertFalse(admission.qr_bound_model_fallback)
                self.assertFalse(admission.metadata()["measured_head_admission"]["front_back_authorized"])

    def test_every_head_quality_gate_applies_even_below_generic_limit(self):
        for changes in (
            {"accepted": False}, {"raw_border_support_mean": .59},
            {"raw_corner_support_accepted": False}, {"centered_neck_supported": False},
            {"neck_junction_verified": False},
            {"minimum_edge_length_px": 23.9}, {"reprojection_rmse_px": 2.01},
            {"axis_ambiguous": True}, {"all_corners_positive_depth": False},
            {"yaw_std_deg": 3.01}, {"yaw_std_deg": math.nan},
            {"max_yaw_std_deg": 10.}, {"profile_sha256": "b" * 64},
            {"face_semantics": "front"},
        ):
            with self.subTest(changes=changes):
                result = self.admit(18., debug=head_debug(head_model_quality=quality(**changes)))
                self.assertFalse(result.accepted)
                self.assertIsNone(result.source)
                self.assertIsNone(result.yaw_rad)

    def test_source_quality_cannot_be_replaced_by_projection_or_joint_fit(self):
        cases = (
            {"debug": head_debug(head_model_quality=None)},
            {"debug": head_debug(model_pose_fit_source="joint_qr_head")},
            {"debug": head_debug(model_profile_sha256="b" * 64)},
            {"estimate": head_estimate(evidence_state="predicted_only")},
            {"estimate": head_estimate(model_measurement_status="synthetic")},
            {"estimate": head_estimate(usable=False)},
            {"estimate": head_estimate(corners=None)},
            {"lidar_target_associated": False},
            {"yaw_rad": math.radians(35.)},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(self.admit(**case).accepted)

    def test_seven_head_samples_keep_identity_separate_and_do_not_mix_joint_source(self):
        pose = EvidencePose(0., 0., 0.)
        evidence = PassiveObserverEvidence(target_key="candidate", anchor_pose=pose,
                                           required_axis_samples=7, max_axis_deviation_rad=math.radians(5.))
        for index in range(7):
            stamp = 10. + .1 * index
            admission = self.admit()
            update = evidence.record_frame(
                target_key="candidate", pose=pose, frame_stamp_sec=stamp,
                lidar_stamp_sec=stamp, observed_at_sec=stamp + .01,
                lidar_associated=True, axis_yaw_rad=admission.yaw_rad,
                axis_source=admission.source, qr_texts=() if index < 5 else ("QR_003",),
            )
            if index < 6:
                self.assertIsNone(update.axis_consensus)
        self.assertEqual(update.axis_consensus.source, MEASURED_HEAD_AXIS_SOURCE)
        self.assertEqual(update.axis_consensus.sample_count, 7)
        self.assertEqual(update.resolved_qr_id, "QR_003")
        legacy = evidence.record_frame(
            target_key="candidate", pose=pose, frame_stamp_sec=10.8, lidar_stamp_sec=10.8,
            observed_at_sec=10.81, lidar_associated=True, axis_yaw_rad=.6,
            axis_source="model_current_frame_qr_pose_refined", qr_texts=("QR_003",),
        )
        self.assertEqual(legacy.snapshot.current_axis_sample_count_by_source[
            "model_current_frame_qr_pose_refined"], 1)

    def test_current_front_proof_needs_marker_own_binding_and_live_identity(self):
        binding = QrTargetBinding(True, "decoded_qr_target_associated", ("QR_003",), 1)
        self.assertTrue(measured_head_front_is_current(qr_binding=binding, marker_verified=True,
                                                       resolved_qr_id="QR_003"))
        for current, verified, identity in (
            (binding, False, "QR_003"), (binding, True, None),
            (binding, True, "QR_002"),
            (replace(binding, accepted=False), True, "QR_003"),
            (replace(binding, symbol_count=2), True, "QR_003"),
            (replace(binding, qr_texts_for_evidence=()), True, "QR_003"),
        ):
            self.assertFalse(measured_head_front_is_current(
                qr_binding=current, marker_verified=verified, resolved_qr_id=identity))

    def test_marker_veto_does_not_discard_a_measured_head_plane(self):
        for texts, verified in (((), False), ((), True), (("QR_003",), True)):
            decision = front_observation_decision(
                qr_texts=texts, qr_marker_detected=verified, qr_marker_verified=verified,
                estimate_source=MEASURED_HEAD_AXIS_SOURCE, marker_seen_in_stationary_epoch=True,
            )
            self.assertFalse(decision.withhold_backside_axis)

    def test_head_tracking_requires_recent_independent_qr_refresh_to_use_native_decode(self):
        fields = dict(previous_axis_source=MEASURED_HEAD_AXIS_SOURCE,
                      previous_bound_qr_stamp_sec=10., image_stamp_sec=10.2, max_age_sec=.5)
        self.assertFalse(measured_head_needs_full_qr_decode(**fields))
        for changes in ({"previous_bound_qr_stamp_sec": None}, {"image_stamp_sec": 10.6},
                        {"image_stamp_sec": 10.}, {"previous_bound_qr_stamp_sec": math.nan}):
            self.assertTrue(measured_head_needs_full_qr_decode(**{**fields, **changes}))
        self.assertFalse(measured_head_needs_full_qr_decode(**{
            **fields, "previous_axis_source": "model_current_frame_refined",
            "previous_bound_qr_stamp_sec": None,
        }))


if __name__ == "__main__":
    unittest.main()
