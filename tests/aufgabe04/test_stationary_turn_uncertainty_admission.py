"""Stationary Start returns preserve the usual uncertainty reserves."""

import unittest

from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    evaluate_route_uncertainty_admission,
    evaluate_stationary_turn_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from tests.aufgabe04.test_route_uncertainty_admission import config, open_costmap


class StationaryTurnUncertaintyTest(unittest.TestCase):
    def _admit(self, *, covariance=None, route=None, start=None, identity=None):
        return evaluate_stationary_turn_uncertainty_admission(
            open_costmap(),
            route or (Pose2D(5., 5., float("nan")), Pose2D(5., 5., 1.)),
            covariance or PlanarCovariance(.0001, 0., .04),
            config(), start_pose=start or Pose2D(5., 5., 0.),
            target_evidence_sha256="a" * 64 if identity is None else identity,
        )

    def test_stationary_budget_uses_largest_covariance_axis(self):
        result = self._admit()
        self.assertTrue(result.decision.accepted)
        self.assertEqual(len(result.segments), 1)
        self.assertTrue(result.segments[0].is_corner)
        self.assertFalse(result.evidence["sampling"]["translation_permitted"])
        isotropic = self._admit(covariance=PlanarCovariance(.04, 0., .04))
        self.assertEqual(result.decision.remaining_margin_m, isotropic.decision.remaining_margin_m)
        self.assertGreater(
            self._admit(covariance=PlanarCovariance(.0001, 0., .0001)).decision.remaining_margin_m,
            result.decision.remaining_margin_m,
        )

    def test_uncertainty_budget_is_still_enforced(self):
        self.assertFalse(self._admit(covariance=PlanarCovariance(25., 0., 25.)).decision.accepted)

    def test_translation_wrong_start_missing_yaw_or_identity_fail_closed(self):
        for inputs in (
            {"route": (Pose2D(5., 5.), Pose2D(5.01, 5., 1.))},
            {"start": Pose2D(5.01, 5., 0.)},
            {"route": (Pose2D(5., 5.), Pose2D(5., 5., float("nan")))},
            {"route": (Pose2D(5., 5.), Pose2D(5., 5., 0.))},
            {"identity": ""},
        ):
            with self.subTest(inputs=inputs):
                self.assertFalse(self._admit(**inputs).decision.accepted)

    def test_generic_route_admission_still_rejects_zero_translation(self):
        result = evaluate_route_uncertainty_admission(
            open_costmap(), (Pose2D(5., 5.), Pose2D(5., 5., 1.)),
            PlanarCovariance(.0001, 0., .0001), config(),
        )
        self.assertFalse(result.decision.accepted)


if __name__ == "__main__":
    unittest.main()
