"""Recorded geometry with explicit synthetic negative mutations, no robot/ROS."""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.real_robot.observer.opposite_endpoint_confirmation import (
    HASH_FIELD, KIND, build_opposite_endpoint_hint, confirm_opposite_endpoint,
    load_endpoint_snapshot, opposite_endpoint_validation_scope, validate_opposite_endpoint,
)
from scripts.aufgabe04.real_robot.observer import opposite_endpoint_confirmation as endpoint
from scripts.aufgabe04.real_robot.observer.opposite_target_support import detect_opposite_qr_outline
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    registered_target_metadata_is_unique, validated_witnessed_fragmentation,
)
from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import envelope_is_unique
from scripts.aufgabe04.real_robot.observer.target_reconciliation import (
    RETAINED_POLICY, StoppedTargetReconciliation, validate_reconciliation,
)
from tests.aufgabe04.opposite_endpoint_fixture import recorded_endpoint, rectified_endpoint_image


def rehash(value):
    """Tampering tests recalculate transport hash; geometry must still reject."""
    return content_hashed_payload({k: v for k, v in value.items() if k != HASH_FIELD}, hash_field=HASH_FIELD)


class OppositeEndpointConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.value = recorded_endpoint(Path(self.temp.name))

    def confirmed(self):
        attempt, seed = build_opposite_endpoint_hint(**self.value['hint_kwargs'])
        return attempt, seed, confirm_opposite_endpoint(seed, self.value['outline'], now_sec=self.value['now_sec'])

    def test_recorded_outlines_confirm_real_rays_without_rewriting_raw_topology(self):
        for index in (2, 26):
            with self.subTest(capture_index=index):
                v = recorded_endpoint(Path(self.temp.name) / str(index), index)
                original = v['metadata']['detector_metadata']['identity_crop']['search']['envelope']
                raw = qr_registration_envelope(v['scan'], **v['row']['options'],
                    now_sec=v['now_sec'], max_scan_age_sec=.5)
                self.assertEqual(raw.eligible_cluster_count, 2)
                self.assertFalse(envelope_is_unique(raw))
                self.assertEqual(raw.scan_topology, original['scan_topology'])
                before_scan, before_orientation = asdict(v['scan']), copy.deepcopy(v['orientation'])
                _, seed = build_opposite_endpoint_hint(**v['hint_kwargs'])
                proof = confirm_opposite_endpoint(seed, v['outline'], now_sec=v['now_sec'])
                association = validate_opposite_endpoint(json.loads(json.dumps(proof)))
                self.assertEqual(proof['kind'], KIND)
                self.assertTrue(registered_target_metadata_is_unique(asdict(association)))
                self.assertEqual(association.search_association.eligible_cluster_count, 2)
                self.assertEqual(association.search_association.scan_topology, raw.scan_topology)
                ids = association.search_association.selected_cluster_source_indices
                self.assertGreaterEqual(len(ids), 3)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(math.isfinite(v['scan'].ranges[i]) for i in ids))
                for key in ('motion_authorized', 'candidate_geometry_updated', 'supplies_angle', 'supplies_identity'):
                    self.assertIs(proof[key], False)
                self.assertEqual(asdict(v['scan']), before_scan)
                self.assertEqual(v['orientation'], before_orientation)
                tracker = StoppedTargetReconciliation()
                receipt = tracker.observe(**v['row'], fragmentation=proof)
                self.assertIsNotNone(receipt, tracker.metadata)
                self.assertEqual(receipt['policy'], RETAINED_POLICY)
                self.assertEqual(len(receipt['entries']), 1)
                validate_reconciliation(json.loads(json.dumps(receipt)),
                    image_stamp_sec=v['row']['image_stamp_sec'], scan_stamp_sec=v['scan'].scan_stamp_sec)

    def test_search_hint_and_recorded_visual_miss_never_supply_uniqueness(self):
        for index in (2, 3, 26):
            with self.subTest(capture_index=index):
                v = recorded_endpoint(Path(self.temp.name) / str(index), index)
                _, seed = build_opposite_endpoint_hint(**v['hint_kwargs'])
                with self.assertRaises(ValueError):
                    validate_opposite_endpoint(seed)
                with self.assertRaises(ValueError):
                    validated_witnessed_fragmentation(seed)
                if index == 3:
                    self.assertIsNone(v['outline'])
                    with self.assertRaises(ValueError):
                        confirm_opposite_endpoint(seed, None, now_sec=v['now_sec'])

    def test_original_final_jpeg_produces_outline_with_real_rectification(self):
        import cv2
        v = self.value
        attempt, seed = build_opposite_endpoint_hint(**v['hint_kwargs'])
        image = rectified_endpoint_image(v, v['image_path'])
        outline = detect_opposite_qr_outline(image, cv2, attempt=attempt, model_profile=v['model'],
            image_stamp_sec=v['row']['image_stamp_sec'], now_sec=v['now_sec'],
            max_scan_age_sec=.5, max_elapsed_sec=.5)
        self.assertIsNotNone(outline)
        self.assertFalse(outline.metadata()['supplies_identity'])
        validate_opposite_endpoint(confirm_opposite_endpoint(seed, outline, now_sec=v['now_sec']))

    def test_two_nearby_objects_need_visual_gap_coverage_not_only_center_ray(self):
        _, seed = build_opposite_endpoint_hint(**self.value['hint_kwargs'])
        # Explicit counterfactual: same complete symbol/scale, translated20px.
        # Its center remains inside the3-degree gate but misses one gap ray.
        outline = copy.deepcopy(self.value['outline'])
        outline['corners_px'] = [[x + 20., y] for x, y in outline['corners_px']]
        outline['center_px'] = [sum(p[k] for p in outline['corners_px']) / 4 for k in (0, 1)]
        with self.assertRaisesRegex(ValueError, 'does not span both scan gap endpoints'):
            confirm_opposite_endpoint(seed, outline, now_sec=self.value['now_sec'])

    def test_unrelated_cluster_and_connected_rays_outside_retained_disk_are_rejected(self):
        for case in ('third_cluster', 'outside_retained_disk'):
            with self.subTest(case=case):
                ranges = list(self.value['scan'].ranges)
                if case == 'third_cluster':
                    for i in range(4, 8):
                        ranges[i] = math.nan
                    ranges[8] = ranges[9] = .45
                    reason = 'exactly two bounded raw endpoint groups'
                else:
                    # Synthetic consecutive real-valued returns stay inside
                    # originalrange; one end leaves the retained stand disk.
                    ranges[:6] = (.45, .47, .485, .50, .515, .53)
                    reason = 'exceed certified target geometry'
                with self.assertRaisesRegex(ValueError, reason):
                    build_opposite_endpoint_hint(**{**self.value['hint_kwargs'],
                        'scan': replace(self.value['scan'], ranges=tuple(ranges))})

    def test_missing_retained_target_and_nonendpoint_scan_are_rejected(self):
        with self.assertRaises(ValueError):
            build_opposite_endpoint_hint(**{**self.value['hint_kwargs'],
                'persistence_context': replace(self.value['persistence_context'], retained_orientation=None)})
        for scan in (replace(self.value['scan'], scan_topology_profile='linear'),
                     replace(self.value['scan'], angle_max=self.value['scan'].angle_max-.1)):
            with self.subTest(scan_topology=scan.scan_topology_profile, angle_max=scan.angle_max):
                with self.assertRaises(ValueError):
                    build_opposite_endpoint_hint(**{**self.value['hint_kwargs'], 'scan': scan})

    def test_rehashed_tuple_transform_profile_outline_and_authority_tampering_rejected(self):
        _, _, original = self.confirmed()
        mutations = {
            'stale': lambda p: p['current'].__setitem__('now_sec', p['current']['now_sec'] + .6),
            'wrong_image': lambda p: p['outline'].__setitem__('image_stamp_sec', p['outline']['image_stamp_sec'] - .01),
            'wrong_model': lambda p: p.__setitem__('model_sha256', '0' * 64),
            'wrong_calibration': lambda p: p.__setitem__('calibration_profile_sha256', '0' * 64),
            'changed_intrinsics': lambda p: p['intrinsics'].__setitem__('fx_px', p['intrinsics']['fx_px'] + 1.),
            'changed_transform': lambda p: p['camera_from_map']['translation_xyz_m'].__setitem__(0, p['camera_from_map']['translation_xyz_m'][0] + .1),
            'borrowed_angle': lambda p: p.__setitem__('supplies_angle', True),
            'borrowed_motion': lambda p: p.__setitem__('motion_authorized', True),
            'changed_retained_uncertainty': lambda p: p['current']['context']['retained_orientation']['validated_target_center'].__setitem__('uncertainty_m', .5),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                proof = json.loads(json.dumps(original))
                mutate(proof)
                with self.assertRaises((ValueError, OSError)):
                    validate_opposite_endpoint(rehash(proof))

    def test_callback_scope_reuses_exact_proof_but_never_mutable_results_or_next_scope(self):
        _, _, proof = self.confirmed()
        with patch.object(endpoint, '_validate_current_outline', wraps=endpoint._validate_current_outline) as replay:
            with opposite_endpoint_validation_scope():
                first = validate_opposite_endpoint(copy.deepcopy(proof))
                second = validate_opposite_endpoint(copy.deepcopy(proof))
                self.assertEqual(first, second)
                self.assertIsNot(first, second)
                self.assertEqual(replay.call_count, 1)
                first.witnessed_fragmentation['outline']['supplies_identity'] = True
                second.witnessed_fragmentation['current']['context']['retained_orientation']['validated_target_center']['uncertainty_m'] = 99.
                third = validate_opposite_endpoint(copy.deepcopy(proof))
                self.assertFalse(third.witnessed_fragmentation['outline']['supplies_identity'])
                self.assertEqual(third.witnessed_fragmentation['current']['context']['retained_orientation'],
                    proof['current']['context']['retained_orientation'])
                self.assertEqual(replay.call_count, 1)
            validate_opposite_endpoint(copy.deepcopy(proof))
            self.assertEqual(replay.call_count, 2)
            with opposite_endpoint_validation_scope():
                validate_opposite_endpoint(copy.deepcopy(proof))
                self.assertEqual(replay.call_count, 3)

    def test_callback_scope_rechecks_rehashed_proof_and_each_external_dependency(self):
        _, _, proof = self.confirmed()
        # Never edit repository metrology: a byte-identical temporary model is
        # an equivalent local dependency, with the same authenticated profile.
        local_model = Path(self.temp.name) / 'measured_model.json'
        local_model.write_bytes(Path(proof['model_path']).read_bytes())
        proof['model_path'] = str(local_model)
        proof = rehash(proof)
        with opposite_endpoint_validation_scope():
            validate_opposite_endpoint(proof)
            changed = copy.deepcopy(proof)
            changed['outline']['image_stamp_sec'] -= .01
            with self.assertRaisesRegex(ValueError, 'differs from its current calibrated hint'):
                validate_opposite_endpoint(rehash(changed))
        for dependency in (Path(self.temp.name) / 'axis.json', self.value['snapshot_path'], local_model):
            with self.subTest(dependency=dependency.name):
                original = dependency.read_bytes()
                with opposite_endpoint_validation_scope():
                    validate_opposite_endpoint(copy.deepcopy(proof))
                    try:
                        # Even whitespace changes bytes: cache hits must not
                        # skip source integrity checks on a parse-equivalent edit.
                        dependency.write_bytes(original + b'\n')
                        with self.assertRaisesRegex(ValueError, 'source changed during the current callback'):
                            validate_opposite_endpoint(copy.deepcopy(proof))
                    finally:
                        dependency.write_bytes(original)

    def test_callback_snapshot_reuse_is_defensive_and_rechecks_bytes(self):
        path = self.value['snapshot_path']
        with patch.object(endpoint, 'load_candidate_snapshot', wraps=endpoint.load_candidate_snapshot) as loader:
            with opposite_endpoint_validation_scope():
                snapshot, digest = load_endpoint_snapshot(path)
                again, repeated_digest = load_endpoint_snapshot(path)
                self.assertEqual(snapshot, again)
                self.assertEqual(digest, repeated_digest)
                self.assertEqual(loader.call_count, 1)
                # Deliberately bypass the frozen dataclass only in this test
                # to prove callers cannot mutate the cached object indirectly.
                object.__setattr__(snapshot, 'planning_frame', 'counterfactual_frame')
                self.assertEqual(load_endpoint_snapshot(path)[0].planning_frame, 'map')
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b'\n')
                    with self.assertRaisesRegex(ValueError, 'snapshot changed during the current callback'):
                        load_endpoint_snapshot(path)
                finally:
                    path.write_bytes(original)
            with opposite_endpoint_validation_scope():
                load_endpoint_snapshot(path)
            self.assertEqual(loader.call_count, 2)


if __name__ == '__main__':
    unittest.main()
