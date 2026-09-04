from dataclasses import replace
import math
import unittest

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
    associate_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
    registered_head_roi_attempt,
)
from scripts.aufgabe04.real_robot.observer.registration_evidence import (
    build_backside_target_registration_evidence,
)


class ObserverRegistrationEvidenceTest(unittest.TestCase):
    @staticmethod
    def _scan() -> PlainLaserScan:
        ranges = [2.0] * 31
        ranges[25] = 0.70  # +9 degrees in a scan starting at -16 degrees.
        return PlainLaserScan(
            ranges=tuple(ranges),
            angle_min=math.radians(-16.0),
            angle_increment=math.radians(1.0),
            range_min=0.10,
            range_max=3.50,
            scan_frame_id="base_scan",
            scan_stamp_sec=9.9,
            receipt_sec=10.0,
        )

    @staticmethod
    def _registered_decision():
        proposal = HeadRoiAttempt(
            roi=ImageRoi(200, 100, 600, 460, 80.0),
            source="target_centered_backside_reacquisition",
            padding_scale=4.5,
            expected_center_u_px=400.0,
            expected_center_v_px=300.0,
            expected_head_height_px=80.0,
            backside_target_crop_half_width_ratio=2.25,
        )
        corners = (
            ImagePoint(70.0, 160.0),
            ImagePoint(150.0, 160.0),
            ImagePoint(150.0, 240.0),
            ImagePoint(70.0, 240.0),
        )
        decision = registered_head_roi_attempt(
            proposal,
            corners,
            max_center_offset_ratio=1.5,
        )
        if not decision.accepted:
            raise AssertionError(decision.reason)
        return decision

    def test_nominal_receipt_evidence_uses_legacy_map_cone(self):
        association = associate_candidate_lidar_target(
            self._scan(),
            map_bearing_rad=math.radians(9.0),
            observed_camera_bearing_rad=math.radians(9.5),
            cone_half_angle_rad=math.radians(3.0),
            accepted_range_m=(0.60, 0.80),
            now_sec=10.1,
            max_scan_age_sec=1.0,
        )

        evidence = build_backside_target_registration_evidence(
            final_head_center_error_ratio=0.08,
            candidate_lidar_association=association,
        )

        self.assertEqual(evidence["mode"], "map_projection")
        self.assertEqual(evidence["lidar_search_bearing_source"], "map_bearing")
        self.assertEqual(
            evidence["map_bearing_rad"],
            evidence["lidar_search_bearing_rad"],
        )
        self.assertFalse(evidence["unique_eligible_lidar_cluster_required"])
        self.assertEqual(evidence["eligible_lidar_cluster_count"], 1)

    def test_registered_receipt_evidence_binds_strict_retry_and_scan(self):
        wrapper = associate_camera_registered_candidate_lidar_target(
            self._scan(),
            map_bearing_rad=0.0,
            observed_camera_bearing_rad=math.radians(9.2),
            cone_half_angle_rad=math.radians(3.0),
            accepted_range_m=(0.60, 0.80),
            now_sec=10.1,
            max_scan_age_sec=1.0,
        )
        self.assertTrue(wrapper.associated)
        association = wrapper.search_association
        self.assertIsNotNone(association)

        evidence = build_backside_target_registration_evidence(
            final_head_center_error_ratio=0.01,
            candidate_lidar_association=association,
            registration_decision=self._registered_decision(),
            registered_lidar_association=wrapper,
        )

        self.assertEqual(
            evidence["mode"],
            "bounded_camera_lidar_registration",
        )
        self.assertEqual(
            evidence["lidar_search_bearing_source"],
            "registered_camera_bearing",
        )
        self.assertTrue(evidence["unique_eligible_lidar_cluster_required"])
        self.assertEqual(evidence["eligible_lidar_cluster_count"], 1)
        self.assertAlmostEqual(
            evidence["camera_map_bearing_delta_rad"],
            math.radians(9.2),
        )

    def test_registered_evidence_rejects_unbound_or_rejected_wrapper(self):
        wrapper = associate_camera_registered_candidate_lidar_target(
            self._scan(),
            map_bearing_rad=0.0,
            observed_camera_bearing_rad=math.radians(9.2),
            cone_half_angle_rad=math.radians(3.0),
            accepted_range_m=(0.60, 0.80),
            now_sec=10.1,
            max_scan_age_sec=1.0,
        )
        association = wrapper.search_association
        self.assertIsNotNone(association)
        other = replace(association, map_bearing_rad=math.radians(9.1))
        with self.assertRaisesRegex(ValueError, "not bound"):
            build_backside_target_registration_evidence(
                final_head_center_error_ratio=0.01,
                candidate_lidar_association=other,
                registration_decision=self._registered_decision(),
                registered_lidar_association=wrapper,
            )
        with self.assertRaisesRegex(ValueError, "not accepted"):
            build_backside_target_registration_evidence(
                final_head_center_error_ratio=0.01,
                candidate_lidar_association=association,
                registration_decision=self._registered_decision(),
                registered_lidar_association=replace(
                    wrapper,
                    associated=False,
                    rejection_reason="test_rejection",
                ),
            )

    def test_qr_registration_cannot_be_relabelled_as_backside_evidence(self):
        wrapper = associate_camera_registered_candidate_lidar_target(
            self._scan(),
            map_bearing_rad=0.0,
            observed_camera_bearing_rad=math.radians(9.2),
            cone_half_angle_rad=math.radians(3.0),
            accepted_range_m=(0.60, 0.80),
            now_sec=10.1,
            max_scan_age_sec=1.0,
        )
        association = wrapper.search_association
        self.assertIsNotNone(association)
        decision = self._registered_decision()
        self.assertIsNotNone(decision.attempt)
        qr_decision = replace(
            decision,
            attempt=replace(
                decision.attempt,
                source=REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
            ),
        )

        with self.assertRaisesRegex(ValueError, "strict retry"):
            build_backside_target_registration_evidence(
                final_head_center_error_ratio=0.01,
                candidate_lidar_association=association,
                registration_decision=qr_decision,
                registered_lidar_association=wrapper,
            )

    def test_final_strict_center_gate_cannot_be_relaxed(self):
        association = associate_candidate_lidar_target(
            self._scan(),
            map_bearing_rad=math.radians(9.0),
            observed_camera_bearing_rad=math.radians(9.0),
            cone_half_angle_rad=math.radians(3.0),
            accepted_range_m=(0.60, 0.80),
        )
        with self.assertRaisesRegex(ValueError, "strictly centred"):
            build_backside_target_registration_evidence(
                final_head_center_error_ratio=0.551,
                candidate_lidar_association=association,
            )


if __name__ == "__main__":
    unittest.main()
