import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.current_head_front_observation import (
    validated_current_head_front_evidence, qr_quad_inside_current_head,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    load_recommendation, recommendation_to_dict, recommendation_uses_current_head_front,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint import main
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle
from scripts.aufgabe04.real_robot.configuration.recommendation import build_real_viewpoint_recommendation
from scripts.aufgabe04.stations.arrival_pose_catalog import load_arrival_pose_catalog
from tests.aufgabe04.current_head_front_fixture import current_head_front_evidence
from tests.aufgabe04 import test_scan_target_persistence as persistence_fixtures


def recommendation(**changes):
    return build_real_viewpoint_recommendation(**{
        "stream_id": "stream", "stand_id": "stand", "planning_frame": "map",
        "stand_center": Pose2D(1.5, 1.5), "stand_radius_m": .06, "stand_uncertainty_m": .02,
        "robot_pose": Pose2D(.75, 1.5, 0.), "stand_axis_rad": math.pi/2,
        "axis_confidence": 0., "axis_sample_count": 1, "sensor_stamp_sec": 100.,
        "expected_qr_id": "QR_003", "observed_qr_ids": ("QR_003",), "target_distance_m": .35,
        "observation_unix_sec": 100., "current_head_evidence": current_head_front_evidence(), **changes})


class CurrentHeadFrontContractTests(unittest.TestCase):
    def test_single_current_fit_roundtrips_without_claiming_consensus(self):
        result = load_recommendation(json.loads(json.dumps(recommendation_to_dict(recommendation()))))
        self.assertTrue(recommendation_uses_current_head_front(result))
        self.assertEqual(result.schema_version, 3)
        self.assertEqual(result.axis_sample_count, 1)
        self.assertEqual(result.axis_confidence, 0.)
        self.assertEqual(result.side_evidence.kind, "qr_observation")
        self.assertEqual(result.axis_measurement["qr_id"], "QR_003")

    def test_policy_flag_missing_proof_and_schema_downgrade_cannot_bypass_minimum(self):
        payload = recommendation_to_dict(recommendation())
        for mutation in (lambda p: p.update(schema_version=1),
                         lambda p: p.pop("axis_measurement"),
                         lambda p: p.update(axis_measurement={"policy": "current_head_and_bound_qr"}),
                         lambda p: p["axis"].update(sample_count=7),
                         lambda p: p["axis"].update(confidence=.9),
                         lambda p: p.update(simulation_only=True)):
            value = copy.deepcopy(payload)
            mutation(value)
            with self.subTest(value=value["schema_version"]), self.assertRaises(ValueError):
                load_recommendation(value)
        with self.assertRaisesRegex(ValueError, "at least two"):
            recommendation(current_head_evidence=None)
        legacy = recommendation(current_head_evidence=None, axis_sample_count=7, axis_confidence=.9)
        self.assertFalse(recommendation_uses_current_head_front(legacy))

    def test_freshness_quality_identity_enclosure_and_current_scan_are_checked(self):
        original = current_head_front_evidence()
        cases = (
            lambda p: p["sample_gate_evidence"].update(stationary=False),
            lambda p: p["head_model_quality"].update(yaw_std_deg=3.1),
            lambda p: p["head_model_quality"].update(axis_ambiguous=True),
            lambda p: p["head_admission"].update(accepted=False),
            lambda p: p.update(checked_at_sec=101.1),
            lambda p: p.update(qr_corners_px=((0, 0), (3, 0), (3, 3), (0, 3))),
            lambda p: p.update(head_corners_px=((0, 100), (200, 100), (200, 200), (0, 200))),
            lambda p: p["qr_binding"].update(qr_texts_for_evidence=["QR_OTHER"]),
            lambda p: p["head_lidar_association"].update(scan_stamp_sec=99.),
            lambda p: p["head_lidar_association"].update(eligible_cluster_count=2),
            lambda p: p["qr_binding"]["association"].update(selected_cluster_source_indices=[99]),
            lambda p: p.update(stand_axis_rad=0.),
        )
        for mutation in cases:
            value = copy.deepcopy(original)
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validated_current_head_front_evidence(value)

    def test_brief_qr_binding_does_not_require_prior_head_or_matching_qr_size(self):
        original = current_head_front_evidence()
        value = copy.deepcopy(original)
        value.update(qr_sensor_stamp_sec=99.5, qr_scan_stamp_sec=99.5, qr_checked_at_sec=99.6)
        value["qr_binding"]["current_head_binding"] = None
        value["qr_binding"]["association"]["scan_stamp_sec"] = 99.5
        value["qr_binding"]["association"]["selected_cluster_source_indices"] = [33, 34]
        self.assertIsNotNone(validated_current_head_front_evidence(value))
        value.update(qr_sensor_stamp_sec=99., qr_scan_stamp_sec=99., qr_checked_at_sec=99.1)
        value["qr_binding"]["association"]["scan_stamp_sec"] = 99.
        with self.assertRaisesRegex(ValueError, "brief stopped lifetime"):
            validated_current_head_front_evidence(value)
        self.assertTrue(qr_quad_inside_current_head(((101, 101), (199, 101), (199, 199), (101, 199)), original["head_corners_px"]))
        self.assertTrue(qr_quad_inside_current_head(((145, 145), (155, 145), (155, 155), (145, 155)), original["head_corners_px"]))

    def test_receipt_binds_recommendation_axis_stamp_target_and_qr(self):
        for changes in ({"sensor_stamp_sec": 101.}, {"stand_axis_rad": 0.},
                        {"stream_id": "another"},
                        {"expected_qr_id": "QR_OTHER", "observed_qr_ids": ("QR_OTHER",)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recommendation(**changes)

    def test_live_nested_camera_context_roundtrips_empty_distortion_and_arrays(self):
        signature = ("camera", 640., 640., 400., 300.,
                     (640., 0., 400., 0., 640., 300., 0., 0., 1.), (),
                     (1., 0., 0., 0., 1., 0., 0., 0., 1.),
                     (640., 0., 400., 0., 0., 640., 300., 0., 0., 0., 1., 0.),
                     "", (0., 0., .1), (0., 0., 0., 1.))
        evidence = current_head_front_evidence(camera_signature=signature)
        self.assertIsNotNone(validated_current_head_front_evidence(json.loads(json.dumps(evidence))))
        evidence["camera_signature"] = (*signature, (0., math.nan))
        with self.assertRaisesRegex(ValueError, "camera context"):
            validated_current_head_front_evidence(evidence)

    def test_existing_head_fragment_witness_recomputes_but_qr_stays_uniquely_bound(self):
        fixture = persistence_fixtures.ScanTargetPersistenceTest()
        fixture.setUp()
        fixture.seed()
        resolved, _, _, _ = fixture.observe(10.6, ranges=fixture.fragmented())
        evidence = current_head_front_evidence(stamp=10.6)
        evidence["head_lidar_association"] = asdict(resolved)
        evidence["qr_binding"]["association"]["scan_frame_id"] = "scan"
        self.assertIsNotNone(validated_current_head_front_evidence(evidence))
        for mutation in (
            lambda p: p["head_lidar_association"].update(witnessed_fragmentation=None),
            lambda p: p["head_lidar_association"].update(witnessed_fragmentation={"persistent_target_count": 1}),
            lambda p: p["head_lidar_association"]["witnessed_fragmentation"]["witnesses"].pop(),
            lambda p: p["qr_binding"]["association"].update(eligible_cluster_count=2),
        ):
            changed = copy.deepcopy(evidence)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validated_current_head_front_evidence(changed)

    def test_real_planner_accepts_one_validated_frame_with_cli_minimum_seven(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.pgm").write_text("P2\n50 40\n255\n" + " ".join(["254"] * 2000) + "\n")
            map_yaml = root / "map.yaml"
            map_yaml.write_text("image: map.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.2\nmode: trinary\n")
            path = root / "recommendation.json"
            path.write_text(json.dumps(recommendation_to_dict(recommendation())))
            catalog = root / "catalog.json"
            args = ["--map", str(map_yaml), "--start-x", ".75", "--start-y", "1.5",
                    "--recommended-pose-json", str(path), "--route-csv", str(root / "route.csv"),
                    "--diagnostics-json", str(root / "diagnostics.json"), "--stream-id", "stream",
                    "--environment", "real", "--map-frame", "map", "--start-from-recommendation",
                    "--workflow-mode", "survey-only", "--axis-sample-count", "7",
                    "--arrival-pose-catalog", str(catalog), "--candidate-uid", "stand",
                    "--expected-candidate-uid", "stand", "--world-id", "test_world",
                    "--world-sha256", "a"*64, "--session-id", "test_session",
                    "--expected-map-bundle-sha256", freeze_map_bundle(map_yaml, semantic_map_id="map", planning_frame="map").bundle_sha256,
                    "--candidate-snapshot-sha256", "b"*64, "--arena-length-m", "20", "--arena-width-m", "20"]
            with patch("scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint.time.time", return_value=100.1):
                self.assertEqual(main(args), 0)
            record = load_arrival_pose_catalog(catalog).records[0]
            self.assertEqual(record.axis.sample_count, 1)
            self.assertEqual(record.axis.confidence, 0.)
            self.assertEqual(record.face.evidence_kind, "qr_observation")
            self.assertFalse((root / "route.csv").exists())
            # The same geometry without the explicit receipt retains the
            # original seven-frame minimum; no global threshold was lowered.
            legacy = recommendation(current_head_evidence=None, axis_sample_count=2, axis_confidence=.9)
            path.write_text(json.dumps(recommendation_to_dict(legacy)))
            with patch("scripts.aufgabe04.navigation.missions.plan_synchronized_viewpoint.time.time", return_value=100.1):
                self.assertEqual(main(args), 1)


if __name__ == "__main__":
    unittest.main()
