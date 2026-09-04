from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.configuration.geometry import (
    CameraIntrinsics,
    ImageRoi,
    OpticalProjection,
)
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    HeadRoiEvaluation,
    QR_MODEL_REACQUISITION_MODE,
    select_camera_target_measurement,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_SAMPLE_SOURCE,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO,
    HeadRoiAttempt,
    MAX_BACKSIDE_REACQUISITION_PADDING_SCALE,
    MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
    registered_head_roi_attempt,
    target_centered_head_roi_attempts,
)


def _estimate(
    *,
    usable: bool,
    reason: str,
    corners=None,
    source: str = BACKSIDE_AXIS_SAMPLE_SOURCE,
):
    return SimpleNamespace(
        usable=usable,
        reason=reason,
        source=source,
        corners=corners,
    )


def _debug(
    *,
    qr_detected: bool = False,
    center_error=None,
    model_pose=None,
):
    return SimpleNamespace(
        qr_detected=qr_detected,
        head_center_error_ratio=center_error,
        model_pose=model_pose,
    )


class CameraTargetRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.nominal = HeadRoiAttempt(
            roi=ImageRoi(320, 250, 480, 390, 80.0),
            source="nominal_projection",
            padding_scale=1.8,
            expected_center_u_px=400.0,
            expected_center_v_px=310.0,
            expected_head_height_px=80.0,
        )
        self.proposal = HeadRoiAttempt(
            roi=ImageRoi(220, 130, 580, 490, 80.0),
            source="target_centered_backside_reacquisition",
            padding_scale=4.5,
            expected_center_u_px=400.0,
            expected_center_v_px=310.0,
            expected_head_height_px=80.0,
            backside_target_crop_half_width_ratio=2.25,
        )

    @staticmethod
    def _corners(center_u_local: float, center_v_local: float):
        return (
            ImagePoint(center_u_local - 40.0, center_v_local - 40.0),
            ImagePoint(center_u_local + 40.0, center_v_local - 40.0),
            ImagePoint(center_u_local + 40.0, center_v_local + 40.0),
            ImagePoint(center_u_local - 40.0, center_v_local + 40.0),
        )

    def test_offset_proposal_requires_and_selects_strict_second_pass(self):
        calls = []

        def evaluate(attempt, pose_hint):
            calls.append((attempt.source, pose_hint))
            if attempt.source == "nominal_projection":
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=False,
                        reason="model_backside_head_and_neck_unavailable",
                    ),
                    _debug(),
                )
            if attempt.source == "target_centered_backside_reacquisition":
                # Full-frame centre is (310, 312): 1.125 head heights from
                # the projected centre, matching the real failure class.
                corners = self._corners(90.0, 182.0)
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=False,
                        reason="model_backside_target_center_mismatch",
                        corners=corners,
                    ),
                    _debug(center_error=1.125),
                )
            self.assertEqual(
                attempt.source,
                "camera_registered_backside_reacquisition",
            )
            self.assertAlmostEqual(attempt.expected_center_u_px, 310.0)
            self.assertAlmostEqual(attempt.expected_center_v_px, 312.0)
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(
                    usable=True,
                    reason="axis_estimated_model_backside_current_frame",
                    corners=self._corners(90.0, 182.0),
                ),
                _debug(center_error=0.0),
            )

        selection = select_camera_target_measurement(
            (self.nominal, self.proposal),
            tracked_pose=None,
            evaluate=evaluate,
            enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )

        self.assertTrue(selection.registered)
        self.assertTrue(selection.selected.estimate.usable)
        self.assertEqual(len(calls), 3)
        metadata = selection.metadata(enabled=True)
        self.assertTrue(metadata["strict_retry_applied"])
        self.assertTrue(metadata["measurement_accepted"])
        self.assertAlmostEqual(
            metadata["decision"]["center_offset_ratio"],
            1.125,
            delta=0.001,
        )
        self.assertEqual(metadata["final_strict_head_center_error_ratio"], 0.0)

    def test_proposal_beyond_bound_cannot_bypass_strict_pass(self):
        def evaluate(attempt, _pose_hint):
            if attempt.source == "nominal_projection":
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=False,
                        reason="model_backside_head_and_neck_unavailable",
                    ),
                    _debug(),
                )
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(
                    usable=True,
                    reason="axis_estimated_model_backside_current_frame",
                    corners=self._corners(40.0, 180.0),
                ),
                _debug(center_error=0.0),
            )

        selection = select_camera_target_measurement(
            (self.nominal, self.proposal),
            tracked_pose=None,
            evaluate=evaluate,
            enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )

        self.assertFalse(selection.registered)
        self.assertFalse(selection.selected.estimate.usable)
        self.assertIsNotNone(selection.decision)
        self.assertEqual(
            selection.decision.reason,
            "detected_head_outside_registration_window",
        )

    def test_tracking_context_keeps_reacquisition_inactive(self):
        calls = []

        def evaluate(attempt, pose_hint):
            calls.append((attempt.source, pose_hint))
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(
                    usable=False,
                    reason="model_backside_head_and_neck_unavailable",
                ),
                _debug(),
            )

        tracked = object()
        selection = select_camera_target_measurement(
            (self.nominal, self.proposal),
            tracked_pose=tracked,
            evaluate=evaluate,
            enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )

        self.assertFalse(selection.registered)
        self.assertEqual(calls, [("nominal_projection", tracked)])

    def test_qr_pose_seed_failure_uses_bounded_strict_reacquisition(self):
        calls = []
        proposal_pose = object()

        def evaluate(attempt, pose_hint):
            calls.append((attempt.source, pose_hint))
            if attempt.source == "nominal_projection":
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=False,
                        reason="model_pose_seed_unavailable",
                        source="model_seed",
                    ),
                    _debug(qr_detected=True),
                )
            if attempt.source == "target_centered_backside_reacquisition":
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=True,
                        reason="axis_estimated_model_current_frame_refined",
                        corners=self._corners(160.0, 180.0),
                        source="model_current_frame_refined",
                    ),
                    _debug(qr_detected=True, model_pose=proposal_pose),
                )
            self.assertEqual(
                attempt.source,
                REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
            )
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(
                    usable=True,
                    reason="axis_estimated_model_current_frame_refined",
                    corners=self._corners(160.0, 180.0),
                    source="model_current_frame_refined",
                ),
                _debug(qr_detected=True, model_pose=proposal_pose),
            )

        selection = select_camera_target_measurement(
            (self.nominal, self.proposal),
            tracked_pose=None,
            evaluate=evaluate,
            enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )

        self.assertTrue(selection.registered)
        self.assertTrue(selection.selected.estimate.usable)
        self.assertEqual(
            selection.reacquisition_mode,
            QR_MODEL_REACQUISITION_MODE,
        )
        self.assertEqual(
            selection.selected.attempt.source,
            REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
        )
        self.assertEqual(len(calls), 3)
        self.assertIsNone(calls[1][1])
        self.assertIsNone(calls[2][1])
        metadata = selection.metadata(enabled=True)
        self.assertEqual(
            metadata["reacquisition_mode"],
            QR_MODEL_REACQUISITION_MODE,
        )
        self.assertTrue(metadata["strict_retry_applied"])
        self.assertTrue(metadata["measurement_accepted"])

    def test_bad_tracked_projection_does_not_suppress_qr_reacquisition(self):
        calls = []
        tracked = object()
        proposal_pose = object()

        def evaluate(attempt, pose_hint):
            calls.append((attempt.source, pose_hint))
            if attempt.source == "nominal_projection":
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=False,
                        reason="projected_head_outside_image",
                        corners=self._corners(10.0, 10.0),
                        source="model_projection",
                    ),
                    _debug(model_pose=tracked),
                )
            if attempt.source == "target_centered_backside_reacquisition":
                return HeadRoiEvaluation(
                    attempt,
                    object(),
                    _estimate(
                        usable=True,
                        reason="axis_estimated_model_current_frame_refined",
                        corners=self._corners(160.0, 180.0),
                        source="model_current_frame_refined",
                    ),
                    _debug(qr_detected=True, model_pose=proposal_pose),
                )
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(
                    usable=True,
                    reason="axis_estimated_model_current_frame_refined",
                    corners=self._corners(160.0, 180.0),
                    source="model_current_frame_refined",
                ),
                _debug(qr_detected=True, model_pose=proposal_pose),
            )

        selection = select_camera_target_measurement(
            (self.nominal, self.proposal),
            tracked_pose=tracked,
            evaluate=evaluate,
            enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )

        self.assertTrue(selection.registered)
        self.assertEqual(calls[0], ("nominal_projection", tracked))
        self.assertIsNone(calls[1][1])
        self.assertEqual(
            selection.selected.attempt.source,
            REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
        )

    def test_invalid_center_limit_is_rejected_even_when_reacquisition_is_inactive(self):
        def evaluate(attempt, _pose_hint):
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(usable=True, reason="nominal_success"),
                _debug(),
            )

        for invalid in (
            0.0,
            -0.1,
            float("nan"),
            float("inf"),
            MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO + 0.001,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "max_center_offset_ratio"):
                    select_camera_target_measurement(
                        (self.nominal,),
                        tracked_pose=None,
                        evaluate=evaluate,
                        enable_reacquisition=False,
                        max_center_offset_ratio=invalid,
                    )

    def test_qr_or_non_backside_wide_evidence_cannot_enter_strict_retry(self):
        cases = (
            (True, BACKSIDE_AXIS_SAMPLE_SOURCE),
            (False, "adaptive_edge_current_frame"),
        )
        for qr_detected, proposal_source in cases:
            with self.subTest(
                qr_detected=qr_detected,
                proposal_source=proposal_source,
            ):
                calls = []

                def evaluate(attempt, _pose_hint):
                    calls.append(attempt.source)
                    if attempt.source == "nominal_projection":
                        return HeadRoiEvaluation(
                            attempt,
                            object(),
                            _estimate(
                                usable=False,
                                reason="model_backside_head_and_neck_unavailable",
                            ),
                            _debug(),
                        )
                    return HeadRoiEvaluation(
                        attempt,
                        object(),
                        _estimate(
                            usable=True,
                            reason="wide_result",
                            corners=self._corners(90.0, 182.0),
                            source=proposal_source,
                        ),
                        _debug(qr_detected=qr_detected),
                    )

                selection = select_camera_target_measurement(
                    (self.nominal, self.proposal),
                    tracked_pose=None,
                    evaluate=evaluate,
                    enable_reacquisition=True,
                    max_center_offset_ratio=1.5,
                )

                self.assertFalse(selection.registered)
                self.assertIs(selection.selected.attempt, self.nominal)
                self.assertEqual(
                    calls,
                    [
                        "nominal_projection",
                        "target_centered_backside_reacquisition",
                    ],
                )

    def test_failed_wide_proposal_never_becomes_selected_measurement(self):
        def evaluate(attempt, _pose_hint):
            if attempt.source == "nominal_projection":
                reason = "model_backside_head_and_neck_unavailable"
            else:
                reason = "model_backside_target_crop_unavailable"
            return HeadRoiEvaluation(
                attempt,
                object(),
                _estimate(usable=False, reason=reason),
                _debug(),
            )

        selection = select_camera_target_measurement(
            (self.nominal, self.proposal),
            tracked_pose=None,
            evaluate=evaluate,
            enable_reacquisition=True,
            max_center_offset_ratio=1.5,
        )

        self.assertIs(selection.selected.attempt, self.nominal)
        self.assertIs(selection.proposal.attempt, self.proposal)


class HeadRoiReacquisitionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(
            width_px=640,
            height_px=480,
            fx_px=400.0,
            fy_px=400.0,
            cx_px=320.0,
            cy_px=240.0,
        )
        self.projection = OpticalProjection(
            u_px=320.0,
            v_px=240.0,
            depth_m=0.5,
            expected_size_px=64.0,
            inside_image=True,
        )

    def test_reacquisition_padding_cap_is_independent_of_legacy_nominal_scale(self):
        attempts = target_centered_head_roi_attempts(
            self.projection,
            self.intrinsics,
            expected_head_height_px=64.0,
            nominal_padding_scale=8.0,
            backside_reacquisition_padding_scale=(
                MAX_BACKSIDE_REACQUISITION_PADDING_SCALE
            ),
        )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].padding_scale, 8.0)
        self.assertEqual(
            attempts[1].padding_scale,
            MAX_BACKSIDE_REACQUISITION_PADDING_SCALE,
        )
        self.assertLessEqual(
            attempts[1].roi.x1 - attempts[1].roi.x0,
            int(64.0 * MAX_BACKSIDE_REACQUISITION_PADDING_SCALE),
        )

    def test_reacquisition_padding_cap_rejects_invalid_values(self):
        for invalid in (
            0.99,
            float("nan"),
            float("inf"),
            MAX_BACKSIDE_REACQUISITION_PADDING_SCALE + 0.001,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    target_centered_head_roi_attempts(
                        self.projection,
                        self.intrinsics,
                        expected_head_height_px=64.0,
                        nominal_padding_scale=1.8,
                        backside_reacquisition_padding_scale=invalid,
                    )

    def test_hand_built_proposal_cannot_bypass_padding_cap_or_policy(self):
        base = HeadRoiAttempt(
            roi=ImageRoi(100, 100, 540, 460, 64.0),
            source="target_centered_backside_reacquisition",
            padding_scale=MAX_BACKSIDE_REACQUISITION_PADDING_SCALE,
            expected_center_u_px=320.0,
            expected_center_v_px=240.0,
            expected_head_height_px=64.0,
            backside_target_crop_half_width_ratio=(
                BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO
            ),
        )
        corners = CameraTargetRegistrationTest._corners(220.0, 140.0)

        overwide = replace(
            base,
            padding_scale=MAX_BACKSIDE_REACQUISITION_PADDING_SCALE + 0.1,
        )
        with self.assertRaisesRegex(ValueError, "certified"):
            registered_head_roi_attempt(overwide, corners)

        wrong_policy = replace(
            base,
            backside_target_crop_half_width_ratio=3.0,
        )
        with self.assertRaisesRegex(ValueError, "target-crop policy"):
            registered_head_roi_attempt(wrong_policy, corners)


if __name__ == "__main__":
    unittest.main()
