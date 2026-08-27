from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from scripts.aufgabe04.navigation.approach.candidate_arrival_admission import (
    ARRIVAL_GEOMETRY_ADMITTED,
    ARRIVAL_GEOMETRY_REJECTED,
    DEFAULT_MAX_BEARING_ERROR_RAD,
    ERROR_INVALID_CONFIGURATION,
    ERROR_INVALID_ROBOT_POSE,
    ERROR_INVALID_TARGET,
    REASON_BEARING_ERROR_ABOVE_MAXIMUM,
    REASON_RANGE_ABOVE_MAXIMUM,
    REASON_RANGE_BELOW_MINIMUM,
    CandidateArrivalAdmissionConfig,
    CandidateArrivalAdmissionError,
    evaluate_candidate_arrival_admission,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


class CandidateArrivalAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CandidateArrivalAdmissionConfig(
            min_range_m=0.60,
            max_range_m=0.80,
        )

    def evaluate(self, pose: Pose2D, target_x_m: float, target_y_m: float):
        return evaluate_candidate_arrival_admission(
            pose,
            target_x_m=target_x_m,
            target_y_m=target_y_m,
            config=self.config,
        )

    def test_aligned_target_is_admitted_without_motion_authority(self):
        decision = self.evaluate(Pose2D(1.0, 2.0, 0.0), 1.70, 2.0)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.decision, ARRIVAL_GEOMETRY_ADMITTED)
        self.assertEqual(decision.reasons, ())
        self.assertAlmostEqual(decision.range_m, 0.70)
        self.assertAlmostEqual(decision.target_bearing_rad, 0.0)
        self.assertAlmostEqual(decision.signed_bearing_error_rad, 0.0)
        self.assertAlmostEqual(decision.absolute_bearing_error_rad, 0.0)
        self.assertFalse(decision.motion_authorized)
        self.assertAlmostEqual(
            decision.config.max_bearing_error_rad,
            math.radians(3.0),
        )

        evidence = decision.to_evidence_dict()
        self.assertFalse(evidence["motion_authorized"])
        self.assertFalse(evidence["scope"]["proves_stationarity"])
        self.assertFalse(evidence["scope"]["authorizes_corrective_motion"])
        self.assertEqual(evidence["threshold_semantics"], "inclusive")

    def test_signed_bearing_error_wraps_across_pi(self):
        target_bearing = math.radians(-179.0)
        decision = self.evaluate(
            Pose2D(0.0, 0.0, math.radians(179.0)),
            0.70 * math.cos(target_bearing),
            0.70 * math.sin(target_bearing),
        )

        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(
            decision.signed_bearing_error_rad,
            math.radians(2.0),
        )
        self.assertAlmostEqual(
            decision.absolute_bearing_error_rad,
            math.radians(2.0),
        )

    def test_opposite_signed_errors_have_matching_absolute_error(self):
        positive = self.evaluate(
            Pose2D(0.0, 0.0, 0.0),
            0.70 * math.cos(math.radians(2.5)),
            0.70 * math.sin(math.radians(2.5)),
        )
        negative = self.evaluate(
            Pose2D(0.0, 0.0, 0.0),
            0.70 * math.cos(math.radians(-2.5)),
            0.70 * math.sin(math.radians(-2.5)),
        )

        self.assertGreater(positive.signed_bearing_error_rad, 0.0)
        self.assertLess(negative.signed_bearing_error_rad, 0.0)
        self.assertAlmostEqual(
            positive.absolute_bearing_error_rad,
            negative.absolute_bearing_error_rad,
        )

    def test_bearing_outside_default_three_degrees_is_rejected(self):
        bearing = math.radians(3.01)
        decision = self.evaluate(
            Pose2D(0.0, 0.0, 0.0),
            0.70 * math.cos(bearing),
            0.70 * math.sin(bearing),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.decision, ARRIVAL_GEOMETRY_REJECTED)
        self.assertEqual(
            decision.reasons,
            (REASON_BEARING_ERROR_ABOVE_MAXIMUM,),
        )
        self.assertFalse(decision.motion_authorized)

    def test_range_envelope_and_bearing_threshold_are_inclusive(self):
        at_minimum = self.evaluate(Pose2D(0.0, 0.0, 0.0), 0.60, 0.0)
        at_maximum = self.evaluate(Pose2D(0.0, 0.0, 0.0), 0.80, 0.0)
        bearing = DEFAULT_MAX_BEARING_ERROR_RAD
        at_bearing_limit = self.evaluate(
            Pose2D(0.0, 0.0, 0.0),
            0.70 * math.cos(bearing),
            0.70 * math.sin(bearing),
        )

        self.assertTrue(at_minimum.accepted)
        self.assertTrue(at_maximum.accepted)
        self.assertTrue(at_bearing_limit.accepted)

    def test_range_misses_return_stable_reasons(self):
        too_close = self.evaluate(Pose2D(0.0, 0.0, 0.0), 0.59, 0.0)
        too_far_and_off_axis = self.evaluate(
            Pose2D(0.0, 0.0, 0.0),
            0.81 * math.cos(math.radians(10.0)),
            0.81 * math.sin(math.radians(10.0)),
        )

        self.assertEqual(too_close.reasons, (REASON_RANGE_BELOW_MINIMUM,))
        self.assertEqual(
            too_far_and_off_axis.reasons,
            (
                REASON_RANGE_ABOVE_MAXIMUM,
                REASON_BEARING_ERROR_ABOVE_MAXIMUM,
            ),
        )

    def test_limits_are_configurable(self):
        decision = evaluate_candidate_arrival_admission(
            Pose2D(0.0, 0.0, 0.0),
            target_x_m=1.0,
            target_y_m=1.0,
            config=CandidateArrivalAdmissionConfig(
                min_range_m=1.0,
                max_range_m=1.5,
                max_bearing_error_rad=math.radians(46.0),
            ),
        )

        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(decision.range_m, math.sqrt(2.0))
        self.assertAlmostEqual(
            decision.absolute_bearing_error_rad,
            math.radians(45.0),
        )

    def test_nonfinite_pose_and_target_raise_stable_error_codes(self):
        with self.assertRaises(CandidateArrivalAdmissionError) as pose_error:
            self.evaluate(Pose2D(math.nan, 0.0, 0.0), 0.70, 0.0)
        with self.assertRaises(CandidateArrivalAdmissionError) as target_error:
            self.evaluate(Pose2D(0.0, 0.0, 0.0), math.inf, 0.0)

        self.assertEqual(pose_error.exception.code, ERROR_INVALID_ROBOT_POSE)
        self.assertEqual(target_error.exception.code, ERROR_INVALID_TARGET)

    def test_invalid_config_raises_stable_error_code(self):
        invalid_arguments = (
            {"min_range_m": -0.1, "max_range_m": 0.7},
            {"min_range_m": 0.8, "max_range_m": 0.7},
            {
                "min_range_m": 0.6,
                "max_range_m": 0.8,
                "max_bearing_error_rad": math.pi + 0.01,
            },
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(CandidateArrivalAdmissionError) as error:
                    CandidateArrivalAdmissionConfig(**arguments)
                self.assertEqual(
                    error.exception.code,
                    ERROR_INVALID_CONFIGURATION,
                )

    def test_config_and_decision_are_frozen(self):
        decision = self.evaluate(Pose2D(0.0, 0.0, 0.0), 0.70, 0.0)

        with self.assertRaises(FrozenInstanceError):
            self.config.min_range_m = 0.0
        with self.assertRaises(FrozenInstanceError):
            decision.motion_authorized = True


if __name__ == "__main__":
    unittest.main()
