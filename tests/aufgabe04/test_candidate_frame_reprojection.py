import math
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    CandidateFrameReprojectionError,
    CandidateFrameReprojectionResult,
    CandidatePoint2D,
    canonical_odom_point_from_frozen_map,
    candidate_frame_provenance_from_mapping,
    candidate_frame_reprojection_result_from_mapping,
    current_map_point_from_canonical_odom,
    reproject_candidate_point,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)


class CandidateFrameReprojectionTest(unittest.TestCase):
    def provenance(
        self,
        *,
        point=CandidatePoint2D(1.731397076, 0.651743266),
        transform=PlanarTransform2D(
            -1.602698535,
            -0.528703478,
            0.018426325,
        ),
    ):
        return CandidateFrameProvenance.from_frozen_map_observation(
            map_frame="map",
            odom_frame="odom",
            frozen_map_point=point,
            frozen_map_from_odom=transform,
        )

    def assertPointAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual.x_m, expected.x_m, places=12)
        self.assertAlmostEqual(actual.y_m, expected.y_m, places=12)

    def test_identity_transform_change_preserves_candidate(self):
        provenance = self.provenance(
            point=CandidatePoint2D(1.25, -0.75),
            transform=PlanarTransform2D(0.4, -0.2, 0.3),
        )

        result = reproject_candidate_point(
            provenance, provenance.frozen_map_from_odom
        )

        self.assertPointAlmostEqual(
            result.current_map_point, provenance.frozen_map_point
        )
        self.assertAlmostEqual(result.diagnostics.candidate_map_displacement_m, 0.0)
        self.assertAlmostEqual(
            result.diagnostics.map_from_odom_translation_drift_m, 0.0
        )
        self.assertAlmostEqual(
            result.diagnostics.map_from_odom_absolute_yaw_drift_rad, 0.0
        )

    def test_reprojects_latest_run_candidate_through_canonical_odom(self):
        provenance = self.provenance()
        current_transform = PlanarTransform2D(
            -1.638358757,
            -0.437467937,
            -0.108542892,
        )

        result = reproject_candidate_point(provenance, current_transform)

        expected_odom_x = 3.355279679688553
        expected_odom_y = 1.118814698588978
        expected_current_x = 1.8183761729346484
        expected_current_y = 0.31128548227724673
        self.assertPointAlmostEqual(
            result.canonical_odom_point,
            CandidatePoint2D(expected_odom_x, expected_odom_y),
        )
        self.assertPointAlmostEqual(
            result.current_map_point,
            CandidatePoint2D(expected_current_x, expected_current_y),
        )
        self.assertAlmostEqual(
            result.diagnostics.candidate_map_displacement_m,
            0.3513927514917973,
            places=12,
        )
        self.assertAlmostEqual(
            result.diagnostics.map_from_odom_translation_drift_m,
            math.hypot(-0.035660222, 0.091235541),
            places=12,
        )
        self.assertAlmostEqual(
            result.diagnostics.map_from_odom_absolute_yaw_drift_rad,
            0.126969217,
            places=12,
        )
        self.assertNotEqual(result.current_map_point, provenance.frozen_map_point)

    def test_rotation_sign_and_transform_order_are_explicit(self):
        provenance = self.provenance(
            point=CandidatePoint2D(0.0, 1.0),
            transform=PlanarTransform2D(0.0, 0.0, math.pi / 2.0),
        )

        result = reproject_candidate_point(
            provenance, PlanarTransform2D(10.0, 20.0, 0.0)
        )

        self.assertPointAlmostEqual(
            result.canonical_odom_point, CandidatePoint2D(1.0, 0.0)
        )
        self.assertPointAlmostEqual(
            result.current_map_point, CandidatePoint2D(11.0, 20.0)
        )
        self.assertPointAlmostEqual(
            canonical_odom_point_from_frozen_map(
                provenance.frozen_map_point,
                provenance.frozen_map_from_odom,
            ),
            CandidatePoint2D(1.0, 0.0),
        )
        self.assertPointAlmostEqual(
            current_map_point_from_canonical_odom(
                CandidatePoint2D(1.0, 0.0),
                PlanarTransform2D(10.0, 20.0, 0.0),
            ),
            CandidatePoint2D(11.0, 20.0),
        )

    def test_fused_candidate_projects_directly_from_canonical_odom_geometry(self):
        provenance = CandidateFrameProvenance(
            map_frame="map",
            odom_frame="odom",
            canonical_odom_point=CandidatePoint2D(1.5, -0.25),
            source_evidence_id="registry:survey_candidate_0003",
        )

        result = reproject_candidate_point(
            provenance, PlanarTransform2D(2.0, 3.0, math.pi / 2.0)
        )

        self.assertPointAlmostEqual(
            result.canonical_odom_point, CandidatePoint2D(1.5, -0.25)
        )
        self.assertPointAlmostEqual(
            result.current_map_point, CandidatePoint2D(2.25, 4.5)
        )
        self.assertFalse(result.diagnostics.frozen_reference_available)
        self.assertIsNone(result.diagnostics.candidate_map_displacement_m)
        self.assertEqual(
            CandidateFrameProvenance.from_mapping(provenance.to_mapping()),
            provenance,
        )

    def test_provenance_mapping_is_strict_and_round_trips(self):
        provenance = self.provenance()
        payload = provenance.to_mapping()

        self.assertEqual(
            candidate_frame_provenance_from_mapping(payload), provenance
        )
        with self.assertRaises(CandidateFrameReprojectionError) as context:
            candidate_frame_provenance_from_mapping({**payload, "extra": 1})
        self.assertEqual(context.exception.code, "mapping_fields_mismatch")

        wrapped = {
            **payload,
            "frozen_map_from_odom": {
                **payload["frozen_map_from_odom"],
                "yaw_rad": 2.0 * math.pi,
            },
        }
        with self.assertRaises(CandidateFrameReprojectionError) as context:
            candidate_frame_provenance_from_mapping(wrapped)
        self.assertEqual(context.exception.code, "invalid_transform")

        with self.assertRaises(CandidateFrameReprojectionError) as context:
            CandidateFrameProvenance(
                map_frame="map",
                odom_frame="odom",
                canonical_odom_point=CandidatePoint2D(99.0, 99.0),
                frozen_map_point=provenance.frozen_map_point,
                frozen_map_from_odom=provenance.frozen_map_from_odom,
            )
        self.assertEqual(
            context.exception.code, "inconsistent_canonical_odom_point"
        )

    def test_result_mapping_round_trip_recomputes_and_rejects_tampering(self):
        result = reproject_candidate_point(
            self.provenance(), PlanarTransform2D(-1.63, -0.44, -0.1)
        )
        payload = result.to_mapping()

        loaded = CandidateFrameReprojectionResult.from_mapping(payload)
        self.assertEqual(loaded, result)

        tampered = {
            **payload,
            "current_map_point": {
                **payload["current_map_point"],
                "x_m": payload["current_map_point"]["x_m"] + 0.01,
            },
        }
        with self.assertRaises(CandidateFrameReprojectionError) as context:
            candidate_frame_reprojection_result_from_mapping(tampered)
        self.assertEqual(context.exception.code, "inconsistent_reprojection")

    def test_nonfinite_bool_and_invalid_frame_inputs_fail_closed(self):
        for value in (math.nan, math.inf, -math.inf, True, "1.0"):
            with self.subTest(value=value):
                with self.assertRaises(CandidateFrameReprojectionError):
                    CandidatePoint2D(value, 0.0)

        with self.assertRaises(CandidateFrameReprojectionError) as context:
            CandidateFrameProvenance(
                map_frame="/map",
                odom_frame="odom",
                canonical_odom_point=CandidatePoint2D(0.0, 0.0),
                frozen_map_point=CandidatePoint2D(0.0, 0.0),
                frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
            )
        self.assertEqual(context.exception.code, "invalid_frame")

        with self.assertRaises(CandidateFrameReprojectionError) as context:
            reproject_candidate_point(self.provenance(), object())
        self.assertEqual(context.exception.code, "invalid_transform")

    def test_forged_result_is_rejected_before_serialization(self):
        result = reproject_candidate_point(
            self.provenance(), PlanarTransform2D(-1.63, -0.44, -0.1)
        )
        object.__setattr__(
            result,
            "current_map_point",
            CandidatePoint2D(
                result.current_map_point.x_m + 1.0,
                result.current_map_point.y_m,
            ),
        )

        with self.assertRaises(CandidateFrameReprojectionError) as context:
            result.to_mapping()
        self.assertEqual(context.exception.code, "inconsistent_reprojection")


if __name__ == "__main__":
    unittest.main()
