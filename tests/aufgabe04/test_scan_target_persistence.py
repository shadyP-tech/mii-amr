"""Real prior returns may witness one internal dropout, never competing stands."""

import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext, StoppedScanTargetPersistence, scan_pose_in_map,
    registered_target_is_unique, registered_target_metadata_is_unique,
    validated_witnessed_fragmentation,
    scan_pose_relative_to_robot,
    scan_pose_from_camera_extrinsics,
)
from scripts.aufgabe04.real_robot.observer.scan_target_geometry import scan_target_geometry
from scripts.aufgabe04.real_robot.observer.head_model_admission import measured_head_lidar_rejection
from scripts.aufgabe04.real_robot.observer.registration_evidence import build_backside_target_registration_evidence
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    registered_head_roi_attempt, HeadRoiAttempt, TARGET_CENTERED_REACQUISITION_SOURCE,
)
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.backside_axis_observation import build_backside_axis_observation
from scripts.aufgabe04.artifacts.backside_axis_observation import validate_backside_axis_observation
from tests.aufgabe04.test_backside_axis_observation import valid_inputs


class ScanTargetPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.state = StoppedScanTargetPersistence()
        self.pose = Pose2D(.8, -.2, math.pi)
        self.target = "survey_candidate_0001_attempt_0:candidate_0001:0.200000000:-0.200000000"

    def observe(self, stamp, *, ranges=None, pose=None, context_changes=None,
                scan_changes=None, now=None, state=None):
        ranges = (math.inf, math.inf, math.inf, .6, .601, .602, .603, math.inf, math.inf) if ranges is None else ranges
        scan = PlainLaserScan(ranges, -.05, .01, .1, 3., "scan", stamp, stamp + .01)
        if scan_changes:
            scan = replace(scan, **scan_changes)
        now = stamp + .1 if now is None else now
        context = ScanPersistenceContext(self.target, "stationary-1", pose or self.pose,
                                         pose or self.pose, stamp, .2, -.2, .06, .02, .04,
                                         Pose2D(0., 0., 0.))
        if context_changes:
            context = replace(context, **context_changes)
        dx, dy = .2 - context.scan_pose_map.x_m, -.2 - context.scan_pose_map.y_m
        c, s = math.cos(context.scan_pose_map.yaw_rad), math.sin(context.scan_pose_map.yaw_rad)
        target = scan_target_geometry((c * dx + s * dy, -s * dx + c * dy, 0.),
                                     stand_radius_m=.06, stand_uncertainty_m=.02, lidar_range_tolerance_m=.04)
        raw = associate_camera_registered_candidate_lidar_target(
            scan, map_bearing_rad=0., observed_camera_bearing_rad=0.,
            cone_half_angle_rad=math.radians(3), accepted_range_m=target.accepted_range_m,
            now_sec=now, max_scan_age_sec=.5, min_cluster_sample_count=1)
        resolved = (state or self.state).resolve(raw, scan, context=context, now_sec=now,
                                               max_scan_age_sec=.5)
        return resolved, raw, scan, context

    def seed(self):
        for stamp in (10., 10.2, 10.4):
            self.assertTrue(self.observe(stamp)[0].associated)

    def ingest(self, stamp, **changes):
        # Construct the raw source/context without delivering any camera result
        # to the persistence state under test.
        _, _, scan, context = self.observe(stamp, state=StoppedScanTargetPersistence(), **changes)
        return self.state.ingest_scan(scan, context=context,
            now_sec=changes.get("now", stamp + .1), max_scan_age_sec=.5)

    @staticmethod
    def fragmented():
        return (math.inf, math.inf, math.inf, .6, .601, math.nan, .603, math.inf, math.inf)

    def test_three_real_scans_witness_current_fragments_without_inventing_return(self):
        self.seed()
        resolved, raw, scan, context = self.observe(10.6, ranges=self.fragmented())
        self.assertFalse(raw.associated)
        self.assertTrue(registered_target_is_unique(resolved), self.state.last_metadata)
        self.assertEqual(resolved.search_association.eligible_cluster_count, 2)
        self.assertFalse(resolved.unique_eligible_cluster_required)
        self.assertEqual(resolved.search_association.selected_cluster_source_indices, (3, 4, 6))
        self.assertEqual(resolved.search_association.distance_m, .601)
        self.assertIsNone(measured_head_lidar_rejection(resolved.search_association,
            registered=True, cone_half_angle_rad=math.radians(3), registered_association=resolved))
        proof = json.loads(json.dumps(resolved.witnessed_fragmentation, allow_nan=False))
        self.assertTrue(registered_target_metadata_is_unique(json.loads(json.dumps(asdict(resolved)))))
        self.assertTrue(validated_witnessed_fragmentation(proof).associated)
        again = self.state.resolve(raw, scan, context=context, now_sec=10.7, max_scan_age_sec=.5)
        self.assertTrue(registered_target_is_unique(again))
        self.assertEqual(len(again.witnessed_fragmentation["witnesses"]), 3)
        self.assertEqual([w["scan"]["scan_stamp_sec"] for w in proof["witnesses"]], [10., 10.2, 10.4])

    def test_duplicate_scans_cannot_create_three_witnesses(self):
        for _ in range(4):
            self.observe(10.)
        self.assertFalse(self.observe(10.2, ranges=self.fragmented())[0].associated)

    def test_scan_only_witnesses_support_first_head_after_camera_misses(self):
        for stamp in (10., 10.2, 10.4):
            self.assertTrue(self.ingest(stamp))
        result = self.observe(10.6, ranges=self.fragmented())[0]
        self.assertTrue(registered_target_is_unique(result), self.state.last_metadata)
        witnesses = result.witnessed_fragmentation["witnesses"]
        self.assertEqual([w["scan"]["scan_stamp_sec"] for w in witnesses], [10., 10.2, 10.4])
        self.assertTrue(all(w["input_source"] == "independent_stopped_scan" for w in witnesses))
        self.assertEqual(result.search_association.selected_cluster_source_indices, (3, 4, 6))
        self.assertEqual(result.search_association.eligible_cluster_count, 2)
        self.assertTrue(validated_witnessed_fragmentation(
            json.loads(json.dumps(result.witnessed_fragmentation, allow_nan=False))).associated)

    def test_scan_only_duplicate_and_camera_scan_cannot_double_count(self):
        for _ in range(4):
            self.assertTrue(self.ingest(10.))
        self.assertTrue(self.observe(10.)[0].associated)
        self.assertFalse(self.observe(10.2, ranges=self.fragmented())[0].associated)

    def test_scan_only_motion_staleness_target_and_epoch_withhold_old_witnesses(self):
        for changes in ({"pose": Pose2D(.82, -.2, math.pi)},
                        {"context_changes": {"epoch_key": "stationary-2"}},
                        {"context_changes": {"target_key": "other"}},
                        {"now": 11.1},
                        {"context_changes": {"image_stamp_sec": 10.49}}):
            with self.subTest(changes=changes):
                self.state.reset()
                for stamp in (10., 10.2, 10.4):
                    self.ingest(stamp)
                self.ingest(10.5, **changes)
                self.assertFalse(self.observe(10.6, ranges=self.fragmented())[0].associated)

    def test_scan_only_competing_targets_and_nonbridging_scans_do_not_supply_proof(self):
        for ranges in (self.fragmented(),
                       (math.inf,) * 3 + (.6, .601) + (math.inf,) * 4,
                       (math.inf,) * 3 + (.6, .601, .9, .603) + (math.inf,) * 2):
            with self.subTest(ranges=ranges):
                self.state.reset()
                for stamp in (10., 10.2, 10.4):
                    self.ingest(stamp, ranges=ranges)
                self.assertFalse(self.observe(10.6, ranges=self.fragmented())[0].associated)
        self.state.reset()
        for stamp in (10., 10.2, 10.4):
            self.ingest(stamp)
        self.ingest(10.5, ranges=(math.inf,) * 9)
        self.assertFalse(self.observe(10.6, ranges=self.fragmented())[0].associated)

    def test_scan_only_evidence_expires_without_refreshing_from_camera_results(self):
        for stamp in (10., 10.2, 10.4):
            self.ingest(stamp)
        self.assertFalse(self.observe(13., ranges=self.fragmented())[0].associated)

    def test_three_later_scan_only_returns_recover_after_a_contradiction(self):
        self.ingest(9.8, ranges=(math.inf,) * 9)
        for stamp in (10., 10.2, 10.4):
            self.ingest(stamp)
        self.assertTrue(registered_target_is_unique(self.observe(10.6, ranges=self.fragmented())[0]),
                        self.state.last_metadata)

    def test_scan_only_proof_binds_historical_search_to_current_head_ray(self):
        for stamp in (10., 10.2, 10.4):
            self.ingest(stamp)
        result, _, scan, context = self.observe(10.6, ranges=self.fragmented())
        # A slightly different valid current head bearing must re-register the
        # same raw witnesses, not reuse the previous image's camera bearing.
        raw = associate_camera_registered_candidate_lidar_target(scan,
            map_bearing_rad=0., observed_camera_bearing_rad=.001,
            cone_half_angle_rad=math.radians(3), accepted_range_m=(.42, .64),
            now_sec=10.71, max_scan_age_sec=.5, min_cluster_sample_count=1)
        changed = self.state.resolve(raw, scan, context=context, now_sec=10.71, max_scan_age_sec=.5)
        self.assertTrue(registered_target_is_unique(changed), self.state.last_metadata)
        self.assertAlmostEqual(changed.witnessed_fragmentation["witnesses"][0]["parameters"]
            ["observed_camera_bearing_rad"], .001)
        for mutate in (
            lambda p: p["witnesses"][0]["parameters"].update(observed_camera_bearing_rad=.01),
            lambda p: p["witnesses"][0].update(input_source="camera"),
            lambda p: p["current"].update(input_source="independent_stopped_scan"),
        ):
            proof = copy.deepcopy(result.witnessed_fragmentation)
            mutate(proof)
            with self.assertRaises(ValueError):
                validated_witnessed_fragmentation(proof)

    def test_same_frame_different_camera_searches_do_not_count_as_new_scans(self):
        _, _, scan, context = self.observe(10.)
        for bearing, bounds in ((.001, (.42, .64)), (-.001, (.42, .64)), (.002, (.42, .64))):
            raw = associate_camera_registered_candidate_lidar_target(
                scan, map_bearing_rad=0., observed_camera_bearing_rad=bearing,
                cone_half_angle_rad=math.radians(3), accepted_range_m=bounds,
                now_sec=10.12, max_scan_age_sec=.5, min_cluster_sample_count=1)
            result = self.state.resolve(raw, scan, context=context, now_sec=10.13, max_scan_age_sec=.5)
            self.assertTrue(result.associated)
        self.assertFalse(self.observe(10.2, ranges=self.fragmented())[0].associated)
        self.assertEqual(self.state.last_metadata["reason"], "three independent witnessed scans are required")
        self.observe(10.3); self.observe(10.4); self.observe(10.5)
        result = self.observe(10.6, ranges=self.fragmented())[0]
        self.assertTrue(result.associated)
        self.assertEqual([w["scan"]["scan_stamp_sec"] for w in result.witnessed_fragmentation["witnesses"]],
                         [10.3, 10.4, 10.5])
        self.observe(10.7)
        self.assertEqual(self.state.last_metadata["reason"], "unique_current_cluster")
        wrong_range = associate_camera_registered_candidate_lidar_target(
            scan, map_bearing_rad=0., observed_camera_bearing_rad=0.,
            cone_half_angle_rad=math.radians(3), accepted_range_m=(.49, .71),
            now_sec=10.13, max_scan_age_sec=.5, min_cluster_sample_count=1)
        self.assertFalse(self.state.resolve(wrong_range, scan, context=context,
                                          now_sec=10.13, max_scan_age_sec=.5).associated)

    def test_historical_pose_coordinates_cannot_expand_the_spatial_gate(self):
        self.seed()
        # A1cm allowed stopped-pose change does not bypass current spatial
        # spacing. This gap exceeds4cm despite range values remaining valid.
        ranges = list(self.fragmented()); ranges[4] = .595; ranges[6] = .634
        self.assertFalse(self.observe(10.6, ranges=tuple(ranges),
                                     pose=Pose2D(.805, -.2, math.pi))[0].associated)

    def test_motion_epoch_target_gap_and_stale_sources_invalidate_history(self):
        changes = ({"pose": Pose2D(.82, -.2, math.pi)},
                   {"pose": Pose2D(.8, -.2, math.pi + .04)},
                   {"context_changes": {"target_key": "another"}},
                   {"context_changes": {"epoch_key": "stationary-2"}},
                   {"context_changes": {"image_stamp_sec": 9.}},
                   {"scan_changes": {"receipt_sec": 9.}},
                   {"scan_changes": {"scan_frame_id": "other"}},
                   {"scan_changes": {"angle_increment": -.01}},
                   {"now": 11.2})
        for change in changes:
            with self.subTest(change=change):
                self.state.reset(); self.seed()
                self.assertFalse(self.observe(10.6, ranges=self.fragmented(), **change)[0].associated)
        for stamp in (9.9, 11.6):
            self.state.reset(); self.seed()
            self.assertFalse(self.observe(stamp, ranges=self.fragmented())[0].associated)

    def test_finite_intervening_return_and_competing_persistent_clusters_stay_rejected(self):
        for value in (.9, .1):
            self.state.reset(); self.seed()
            ranges = list(self.fragmented()); ranges[5] = value
            self.assertFalse(self.observe(10.6, ranges=tuple(ranges))[0].associated)
        self.state.reset()
        for stamp in (10., 10.2, 10.4, 10.6):
            self.assertFalse(self.observe(stamp, ranges=self.fragmented())[0].associated)
        self.assertFalse(self.observe(10.8, ranges=self.fragmented())[0].associated)

    def test_missing_bridging_beam_and_second_missing_beam_cannot_be_filled(self):
        self.seed()
        ranges = list(self.fragmented()); ranges[4] = math.nan
        self.assertFalse(self.observe(10.6, ranges=tuple(ranges))[0].associated)
        self.state.reset()
        for stamp in (10., 10.2, 10.4):
            # A prior contiguous cluster exists only on today's left side.
            self.observe(stamp, ranges=(math.inf,) * 3 + (.6, .601) + (math.inf,) * 4)
        self.assertFalse(self.observe(10.6, ranges=self.fragmented())[0].associated)

    def test_partial_scan_endpoints_are_never_temporally_merged(self):
        self.seed()
        ranges = (.6,) + (math.inf,) * 626 + (.603,)
        result = self.observe(10.6, ranges=ranges, scan_changes={
            "angle_min": 0., "angle_increment": math.tau / 628,
            "angle_max": math.tau - math.tau / 628, "scan_topology_profile": "linear"})[0]
        self.assertFalse(result.associated)
        self.assertEqual(result.search_association.eligible_cluster_count, 2)
        self.assertFalse(result.search_association.scan_topology["circular_adjacency_enabled"])

    def test_proof_tampering_or_removing_raw_ambiguity_is_rejected(self):
        self.seed()
        resolved = self.observe(10.6, ranges=self.fragmented())[0]
        for mutate in (
            lambda p: p["witnesses"].pop(),
            lambda p: p["witnesses"][1]["scan"].update(scan_stamp_sec=10.),
            lambda p: p["witnesses"][0]["scan"]["ranges"].__setitem__(5, None),
            lambda p: p["current"]["scan"]["ranges"].__setitem__(5, .9),
            lambda p: p["current"]["context"].update(epoch_key="other"),
            lambda p: p["current"]["context"]["scan_pose_map"].update(x_m=1.2),
        ):
            proof = copy.deepcopy(resolved.witnessed_fragmentation); mutate(proof)
            with self.assertRaises(ValueError):
                validated_witnessed_fragmentation(proof)
        forged = replace(resolved, search_association=replace(resolved.search_association, eligible_cluster_count=1))
        self.assertFalse(registered_target_is_unique(forged))
        self.assertFalse(registered_target_is_unique(replace(resolved, witnessed_fragmentation=None)))

    def test_schema_four_receipt_preserves_raw_two_and_recomputes_proof(self):
        self.seed()
        resolved = self.observe(10.6, ranges=self.fragmented())[0]
        decision = registered_head_roi_attempt(
            HeadRoiAttempt(ImageRoi(0, 0, 200, 200, 80.), TARGET_CENTERED_REACQUISITION_SOURCE,
                           4.5, 100., 100., 80., 2.25),
            tuple(ImagePoint(u, v) for u, v in ((60., 60.), (140., 60.), (140., 140.), (60., 140.))),
            max_center_offset_ratio=1.5)
        registration = build_backside_target_registration_evidence(
            final_head_center_error_ratio=.07, candidate_lidar_association=resolved.search_association,
            registration_decision=decision, registered_lidar_association=resolved)
        inputs = valid_inputs()
        inputs.update(target_registration=registration, sensor_stamp_sec=10.6, stand_x_m=.2,
                      consensus_source="model_backside_current_frame_bounded_camera_lidar_registered")
        receipt = build_backside_axis_observation(**inputs)
        self.assertEqual(receipt["schema_version"], 4)
        self.assertEqual(receipt["target_registration"]["eligible_lidar_cluster_count"], 2)
        validate_backside_axis_observation(json.loads(json.dumps(receipt, allow_nan=False)))
        for mutate in (
            lambda p: p.update(schema_version=3),
            lambda p: p["target_registration"].pop("witnessed_fragmentation"),
            lambda p: p["target_registration"].update(eligible_lidar_cluster_count=1),
            lambda p: p.update(sensor_stamp_sec=10.5),
            lambda p: p.update(stand_id="other"),
            lambda p: p["robot_pose"].update(x_m=.81),
        ):
            altered = copy.deepcopy(receipt); mutate(altered)
            with self.assertRaises(ValueError):
                validate_backside_axis_observation(altered)
        shifted = copy.deepcopy(receipt)
        proof = shifted["target_registration"]["witnessed_fragmentation"]
        for entry in (proof["current"], *proof["witnesses"]):
            entry["context"]["scan_pose_map"]["y_m"] += 100.
        with self.assertRaises(ValueError):
            validate_backside_axis_observation(shifted)

    def test_exact_transform_inverse_and_invalid_geometry(self):
        pose = scan_pose_in_map((1., 2., .3), (0., 0., math.sin(.3), math.cos(.3)))
        self.assertAlmostEqual(pose.x_m, -math.cos(.6) - 2 * math.sin(.6))
        self.assertAlmostEqual(pose.y_m, math.sin(.6) - 2 * math.cos(.6))
        self.assertAlmostEqual(pose.yaw_rad, -.6)
        for translation, rotation in (((math.nan, 0., 0.), (0., 0., 0., 1.)),
                                      ((0., 0., 0.), (0., 0., 0., 0.)),
                                      ((0., 0., 0.), (.1, 0., 0., .99))):
            with self.assertRaises(ValueError):
                scan_pose_in_map(translation, rotation)
        optical = (.5, -.5, .5, -.5)
        static = scan_pose_from_camera_extrinsics((.045, -.004, .125), optical,
                                                 (.077, -.004, -.057), optical)
        self.assertAlmostEqual(static.x_m, -.032)
        self.assertAlmostEqual(static.y_m, 0.)
        self.assertAlmostEqual(static.yaw_rad, 0.)
        rotated = scan_pose_from_camera_extrinsics((1., 2., .3),
            (0., 0., math.sin(.3), math.cos(.3)), (.1, .2, .3), (0., 0., 0., 1.))
        self.assertAlmostEqual(rotated.x_m, 1 - .1 * math.cos(.6) + .2 * math.sin(.6))
        self.assertAlmostEqual(rotated.y_m, 2 - .1 * math.sin(.6) - .2 * math.cos(.6))
        self.assertAlmostEqual(rotated.yaw_rad, .6)
        with self.assertRaises(ValueError):
            scan_pose_from_camera_extrinsics((0., 0., 0.), (.1, 0., 0., .99),
                                            (0., 0., 0.), (0., 0., 0., 1.))

    def test_coherent_scan_pose_shift_cannot_survive_candidate_projection_binding(self):
        self.seed()
        proof = self.observe(10.6, ranges=self.fragmented())[0].witnessed_fragmentation
        for adjust_relative in (False, True):
            corrupted = copy.deepcopy(proof)
            for entry in (corrupted["current"], *corrupted["witnesses"]):
                context = entry["context"]
                context["scan_pose_map"]["y_m"] += 100.
                if adjust_relative:
                    context["scan_pose_robot"] = asdict(scan_pose_relative_to_robot(
                        Pose2D(**context["robot_pose"]), Pose2D(**context["scan_pose_map"])))
            with self.assertRaises(ValueError):
                validated_witnessed_fragmentation(corrupted)
        for field in ("candidate_x_m", "candidate_y_m", "stand_radius_m",
                      "stand_uncertainty_m", "lidar_range_tolerance_m"):
            corrupted = copy.deepcopy(proof)
            corrupted["current"]["context"][field] += .1
            with self.assertRaises(ValueError):
                validated_witnessed_fragmentation(corrupted)

    def test_current_contradiction_and_expiry_withhold_association_and_clear_witnesses(self):
        # A finite competing return fails the unchanged spatial continuity
        # gate. It cannot leave the earlier witnesses usable next frame.
        for intervening in (.64, .7):
            self.state.reset(); self.seed()
            ranges = list(self.fragmented()); ranges[4] = .59; ranges[5] = intervening
            self.assertFalse(self.observe(10.5, ranges=tuple(ranges))[0].associated)
            self.assertFalse(self.observe(10.6, ranges=self.fragmented())[0].associated)
        self.state.reset(); self.seed()
        _, raw, scan, context = self.observe(10.6)
        expired = self.state.resolve(raw, scan, context=context, now_sec=11.2, max_scan_age_sec=.5)
        self.assertFalse(expired.associated)
        self.assertFalse(expired.search_association.associated)
        self.assertIsNone(expired.distance_m)
        self.assertIsNone(expired.witnessed_fragmentation)

    def test_source_hashed_recorded_dropouts_are_internal_and_remain_separate(self):
        path = Path(__file__).parent / "fixtures/scan_target_persistence_20260914T123717Z.json"
        fixture = json.loads(path.read_text())
        self.assertRegex(fixture["replay_sha256"], r"^[0-9a-f]{64}$")
        failures = {}
        for record in fixture["records"]:
            self.assertRegex(record["metadata_sha256"], r"^[0-9a-f]{64}$")
            data = record["scan"]
            scan = PlainLaserScan(**{**data, "ranges": tuple(math.nan if v is None else v
                                                            for v in data["ranges"])})
            context = ScanPersistenceContext(**{**record["context"],
                "robot_pose": Pose2D(**record["context"]["robot_pose"]),
                "scan_pose_map": Pose2D(**record["context"]["scan_pose_map"]),
                "scan_pose_robot": Pose2D(**record["context"]["scan_pose_robot"])})
            raw = associate_camera_registered_candidate_lidar_target(scan, **record["parameters"])
            result = self.state.resolve(raw, scan, context=context,
                now_sec=record["parameters"]["now_sec"], max_scan_age_sec=.5)
            self.assertEqual(raw.search_association.eligible_cluster_count, record["raw_cluster_count"])
            if record["frame"] in {"frame_000012", "frame_000020", "frame_000025"}:
                missing = 4 if record["frame"] == "frame_000025" else 5
                self.assertTrue(math.isnan(scan.ranges[missing]))
                self.assertTrue(math.isfinite(scan.ranges[missing - 1]))
                self.assertTrue(math.isfinite(scan.ranges[missing + 1]))
                self.assertEqual(raw.search_association.eligible_cluster_count, 2)
                self.assertFalse(raw.associated)
                self.assertFalse(record["original_image_result_fresh"],
                                 "historical scan support cannot renew the saved image result")
                failures[record["frame"]] = result.associated
                if result.associated:
                    self.assertEqual(result.search_association.selected_cluster_source_indices, (3, 4, 6))
                    self.assertTrue(registered_target_is_unique(result))
                if record["frame"] == "frame_000025":
                    self.assertTrue(raw.search_association.scan_topology["circular_adjacency_enabled"],
                                    "a valid seam cannot repair an internal missing beam")
        # Only012 has three suitable recent witnessed scans in this sparse
        # saved proposal sequence.020/025 still require more current evidence.
        self.assertEqual(failures, {"frame_000012": True, "frame_000020": False, "frame_000025": False})
