import math
import unittest
from dataclasses import replace

from scripts.aufgabe04.perception.stand_axis.models import (
    ImagePoint,
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.real_robot.observer.axis_sample_policy import (
    MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD,
    QR_BOUND_MODEL_AXIS_SAMPLE_SOURCE,
    admit_axis_sample,
    normalize_qr_bound_model_obliqueness_limit,
)
from scripts.aufgabe04.real_robot.observer.evidence import (
    EvidencePose,
    PassiveObserverEvidence,
)


def estimate(**overrides):
    kwargs = {
        "usable": True,
        "reason": "axis_estimated_model_current_frame_refined",
        "mode": "metric_model_only",
        "corners": (
            ImagePoint(10.0, 10.0),
            ImagePoint(40.0, 10.0),
            ImagePoint(40.0, 80.0),
            ImagePoint(10.0, 80.0),
        ),
        "axis_line": None,
        "left_height_px": 70.0,
        "right_height_px": 70.0,
        "height_ratio": 1.0,
        "yaw_proxy": None,
        "yaw_deg": 32.25,
        "closer_side": None,
        "contour_area_px": 2100.0,
        "source": "model_current_frame_refined",
        "evidence_state": "fresh_refined",
        "model_profile_sha256": "a" * 64,
        "model_measurement_status": "measured",
        "pose_reprojection_rmse_px": 0.42,
        "pose_ambiguity_gap_px": 1.0,
    }
    kwargs.update(overrides)
    return StandAxisImageEstimate(**kwargs)


def debug(**overrides):
    artifacts = StandAxisEdgeDebugArtifacts(
        edges=None,
        raw_edges=None,
        evidence_state="fresh_refined",
        model_pose_fit_source="joint_qr_head",
        model_profile_sha256="a" * 64,
        model_measurement_status="measured",
        pose_reprojection_rmse_px=0.42,
        pose_ambiguity_gap_px=1.0,
        qr_detected=True,
    )
    return replace(artifacts, **overrides)


class ObserverAxisSamplePolicyTest(unittest.TestCase):
    def test_normal_conditioned_sample_keeps_estimator_source(self):
        conditioning = axis_conditioning(
            math.radians(18.0),
            max_obliqueness_rad=math.radians(30.0),
        )

        admitted = admit_axis_sample(
            estimate=estimate(yaw_deg=18.0),
            debug=debug(model_pose_fit_source="head_only_qr_unavailable"),
            conditioning=conditioning,
            yaw_rad=math.radians(18.0),
            qr_texts=(),
            lidar_target_associated=True,
        )

        self.assertTrue(admitted.accepted)
        self.assertEqual(admitted.source, "model_current_frame_refined")
        self.assertFalse(admitted.qr_bound_model_fallback)

    def test_qr_joint_model_pose_admits_bounded_oblique_axis_samples(self):
        for yaw_deg in (30.01, 32.27, 35.0):
            with self.subTest(yaw_deg=yaw_deg):
                conditioning = axis_conditioning(
                    math.radians(yaw_deg),
                    max_obliqueness_rad=math.radians(30.0),
                )

                admitted = admit_axis_sample(
                    estimate=estimate(yaw_deg=yaw_deg),
                    debug=debug(),
                    conditioning=conditioning,
                    yaw_rad=math.radians(yaw_deg),
                    qr_texts=("QR_003",),
                    lidar_target_associated=True,
                )

                self.assertTrue(admitted.accepted)
                self.assertEqual(
                    admitted.reason,
                    "qr_bound_model_axis_oblique_recovery",
                )
                self.assertEqual(
                    admitted.source,
                    QR_BOUND_MODEL_AXIS_SAMPLE_SOURCE,
                )
                self.assertTrue(admitted.qr_bound_model_fallback)

    def test_oblique_recovery_requires_current_qr_and_joint_fit(self):
        conditioning = axis_conditioning(
            math.radians(32.27),
            max_obliqueness_rad=math.radians(30.0),
        )
        cases = (
            {"qr_texts": ()},
            {"qr_texts": ("",)},
            {"debug": debug(qr_detected=False)},
            {"debug": debug(evidence_state="predicted_only")},
            {"debug": debug(model_pose_fit_source="head_only_qr_unavailable")},
            {"debug": debug(model_profile_sha256="b" * 64)},
            {"estimate": estimate(source="model_projection")},
            {"estimate": estimate(evidence_state="predicted_only")},
            {"estimate": estimate(model_measurement_status="synthetic")},
            {"estimate": estimate(pose_reprojection_rmse_px=None)},
            {"estimate": estimate(pose_ambiguity_gap_px=math.nan)},
        )

        for case in cases:
            with self.subTest(case=case):
                admitted = admit_axis_sample(
                    estimate=case.get("estimate", estimate()),
                    debug=case.get("debug", debug()),
                    conditioning=conditioning,
                    yaw_rad=math.radians(32.27),
                    qr_texts=case.get("qr_texts", ("QR_003",)),
                    lidar_target_associated=True,
                )
                self.assertFalse(admitted.accepted)
                self.assertEqual(admitted.source, None)

    def test_oblique_recovery_keeps_hard_upper_bound(self):
        conditioning = axis_conditioning(
            math.radians(36.0),
            max_obliqueness_rad=math.radians(30.0),
        )

        admitted = admit_axis_sample(
            estimate=estimate(yaw_deg=36.0),
            debug=debug(),
            conditioning=conditioning,
            yaw_rad=math.radians(36.0),
            qr_texts=("QR_003",),
            lidar_target_associated=True,
        )

        self.assertFalse(admitted.accepted)
        self.assertEqual(admitted.reason, "oblique_silhouette")

    def test_latest_run_oblique_sequence_reaches_seven_frame_consensus(self):
        run_obliqueness_deg = (
            32.001,
            31.919,
            31.911,
            32.205,
            31.916,
            32.156,
            32.284,
        )
        target_key = "survey_candidate_0003"
        robot_pose = EvidencePose(0.39, -0.12, -2.04)
        evidence = PassiveObserverEvidence(
            target_key=target_key,
            anchor_pose=robot_pose,
            required_axis_samples=7,
            max_axis_deviation_rad=math.radians(8.0),
        )
        update = None

        for index, yaw_deg in enumerate(run_obliqueness_deg):
            yaw_rad = math.radians(yaw_deg)
            admission = admit_axis_sample(
                estimate=estimate(yaw_deg=yaw_deg),
                debug=debug(),
                conditioning=axis_conditioning(
                    yaw_rad,
                    max_obliqueness_rad=math.radians(30.0),
                ),
                yaw_rad=yaw_rad,
                qr_texts=("QR_003",),
                lidar_target_associated=True,
            )
            self.assertTrue(admission.accepted)
            self.assertTrue(admission.qr_bound_model_fallback)
            update = evidence.record_frame(
                target_key=target_key,
                pose=robot_pose,
                frame_stamp_sec=10.0 + index * 0.1,
                lidar_stamp_sec=9.99 + index * 0.1,
                observed_at_sec=10.01 + index * 0.1,
                lidar_associated=True,
                axis_yaw_rad=admission.yaw_rad,
                axis_source=admission.source,
                qr_texts=("QR_003",),
            )

        self.assertIsNotNone(update)
        self.assertIsNotNone(update.axis_consensus)
        self.assertEqual(update.axis_consensus.sample_count, 7)
        self.assertEqual(
            update.axis_consensus.source,
            QR_BOUND_MODEL_AXIS_SAMPLE_SOURCE,
        )
        self.assertEqual(update.resolved_qr_id, "QR_003")

    def test_every_axis_sample_still_requires_lidar_association(self):
        conditioning = axis_conditioning(
            math.radians(18.0),
            max_obliqueness_rad=math.radians(30.0),
        )

        admission = admit_axis_sample(
            estimate=estimate(yaw_deg=18.0),
            debug=debug(),
            conditioning=conditioning,
            yaw_rad=math.radians(18.0),
            qr_texts=("QR_003",),
            lidar_target_associated=False,
        )

        self.assertFalse(admission.accepted)
        self.assertEqual(admission.reason, "lidar_target_unassociated")

    def test_qr_bound_limit_clamps_roundoff_but_rejects_real_expansion(self):
        normalized = normalize_qr_bound_model_obliqueness_limit(
            MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD + 5.0e-13,
            generic_max_obliqueness_rad=math.radians(30.0),
        )
        self.assertEqual(normalized, MAX_QR_BOUND_MODEL_OBLIQUENESS_RAD)

        with self.assertRaisesRegex(ValueError, "cannot exceed 35 degrees"):
            normalize_qr_bound_model_obliqueness_limit(
                math.radians(35.01),
                generic_max_obliqueness_rad=math.radians(30.0),
            )
        with self.assertRaisesRegex(ValueError, "cannot be below"):
            normalize_qr_bound_model_obliqueness_limit(
                math.radians(29.0),
                generic_max_obliqueness_rad=math.radians(30.0),
            )


if __name__ == "__main__":
    unittest.main()
