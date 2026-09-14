"""Separate backside semantics preserve independent head quality and binding."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.artifacts.backside_axis_observation import BACKSIDE_AXIS_SAMPLE_SOURCE
from scripts.aufgabe04.perception.stand_axis.head_backside_classification import (
    classify_current_head_backside, is_classified_measured_head_backside,
)
from scripts.aufgabe04.perception.stand_axis.model_diagnostics import metric_fit_diagnostics_payload
from scripts.aufgabe04.perception.stand_axis.head_model_neck import HeadNeckJunction
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint, StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import RectifiedCameraMatrix
from scripts.aufgabe04.perception.stand_axis_consensus import axis_conditioning
from scripts.aufgabe04.real_robot.observer.axis_sample_policy import admit_axis_sample
from tests.aufgabe04.test_head_model_admission import head_estimate, quality


def classified_head(*, yaw_deg=37.815, u=80., v=80., height=90., profile_sha256="a" * 64,
                    qr_detected=False, qr_marker_verified=False):
    corners = tuple(ImagePoint(u + x * height, v + y * height)
                    for x, y in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)))
    estimate = head_estimate(yaw_deg, corners=corners, left_height_px=height,
                             right_height_px=height, model_profile_sha256=profile_sha256)
    debug = StandAxisEdgeDebugArtifacts(
        edges=None, model_pose_fit_source=estimate.source, evidence_state="fresh_refined",
        model_measurement_status="measured", model_profile_sha256=profile_sha256,
        head_model_quality=quality(profile_sha256=profile_sha256),
        head_neck_junction=HeadNeckJunction(True, "head_neck_junction_verified"),
        qr_detected=qr_detected, qr_marker_verified=qr_marker_verified,
    )
    options = dict(
        model_profile=SimpleNamespace(committable=True, environment="physical",
                                     sha256=profile_sha256, head_width_m=.078, head_height_m=.078),
        camera=RectifiedCameraMatrix(640., 640., 80., 80.),
        expected_center_u_px=u, expected_center_v_px=v, expected_height_px=height,
    )
    return estimate, debug, options


class CurrentHeadBacksideClassificationTests(unittest.TestCase):
    def admit(self, estimate, debug):
        yaw = math.radians(estimate.yaw_deg)
        return admit_axis_sample(estimate=estimate, debug=debug, yaw_rad=yaw,
                                 conditioning=axis_conditioning(yaw), qr_texts=(),
                                 lidar_target_associated=True)

    def test_classification_preserves_head_angle_and_independent_quality_above_35_degrees(self):
        for yaw in (10., 37.815, 60.):
            estimate, debug, options = classified_head(yaw_deg=yaw)
            side, proof = classify_current_head_backside(estimate, debug, **options)
            self.assertTrue(is_classified_measured_head_backside(side, proof))
            self.assertIs(proof.head_model_quality, debug.head_model_quality)
            self.assertEqual(side.corners, estimate.corners)
            self.assertEqual(side.yaw_deg, estimate.yaw_deg)
            self.assertEqual(proof.model_pose_fit_source, estimate.source)
            self.assertIsNone(side.camera_face_normal_xyz)
            self.assertIsNone(side.camera_face_center_xyz_m)
            self.assertTrue(self.admit(side, proof).accepted)
            self.assertEqual(self.admit(side, proof).source, BACKSIDE_AXIS_SAMPLE_SOURCE)

    def test_marker_missing_structure_projection_and_quality_cannot_classify(self):
        estimate, debug, options = classified_head()
        cases = (
            (estimate, replace(debug, qr_detected=True), {}),
            (estimate, replace(debug, qr_marker_verified=True), {}),
            (estimate, replace(debug, qr_marker_verified=None), {}),
            (estimate, replace(debug, head_neck_junction=None), {}),
            (estimate, replace(debug, head_model_quality=quality(neck_junction_verified=False)), {}),
            (estimate, replace(debug, head_model_quality=quality(axis_ambiguous=True)), {}),
            (replace(estimate, usable=False), debug, {}),
            (replace(estimate, evidence_state="predicted_only"), debug, {}),
            (estimate, debug, {"expected_height_px": None}),
            (estimate, debug, {"expected_height_px": 200.}),
            (estimate, debug, {"expected_center_u_px": 180.}),
        )
        for est, dbg, changes in cases:
            with self.subTest(changes=changes, debug=dbg):
                result, diagnostic = classify_current_head_backside(est, dbg, **{**options, **changes})
                self.assertEqual(result.source, estimate.source)
                self.assertFalse(diagnostic.head_backside_classification.accepted)

    def test_relabeling_or_mixing_current_proofs_cannot_escape_quality_even_at_10_degrees(self):
        estimate, debug, options = classified_head(yaw_deg=10.)
        side, proof = classify_current_head_backside(estimate, debug, **options)
        for est, dbg in (
            (replace(estimate, source=BACKSIDE_AXIS_SAMPLE_SOURCE), debug),
            (side, replace(proof, head_backside_classification=None)),
            (side, replace(proof, head_model_quality=quality(yaw_std_deg=3.1))),
            (side, replace(proof, qr_detected=True)),
            (replace(side, yaw_deg=11.), proof),
            (replace(side, corners=estimate.corners[1:] + estimate.corners[:1]), proof),
            (replace(side, model_profile_sha256="b" * 64), proof),
            (replace(side, evidence_state="predicted_only"), proof),
        ):
            self.assertFalse(self.admit(est, dbg).accepted)

    def test_classifier_decision_is_in_saved_metric_diagnostics(self):
        estimate, debug, options = classified_head()
        _, proof = classify_current_head_backside(estimate, debug, **options)
        # The actual junction is a dataclass. This test's stub need not be serialized.
        payload = metric_fit_diagnostics_payload(replace(proof, head_neck_junction=None))
        self.assertTrue(payload["head_backside_classification"]["accepted"])
        self.assertEqual(payload["head_backside_classification"]["corners"][0],
                         {"u_px": 35., "v_px": 35.})


if __name__ == "__main__":
    unittest.main()
