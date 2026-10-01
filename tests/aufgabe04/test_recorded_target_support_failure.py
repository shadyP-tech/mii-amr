"""Replay the stopped radiator-view tuples through raw scan and failure policy."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
from scripts.aufgabe04.real_robot.observer.target_support_failure import (
    TargetSupportFailureWindow,
    target_support_binding_sha256,
    validate_target_support_failure,
)


FIXTURE = Path(__file__).resolve().parent / 'fixtures/target_support_20260930'


def _binding(group, record):
    metadata = record['metadata']
    receipt = group['target_receipt']['values']
    center = {key: receipt['target'][key] for key in ('x_m', 'y_m')}
    stream_id = f"{group['run_id']}_{group['candidate_uid']}"
    target_key = (f"{stream_id}:{group['candidate_uid']}:"
                  f"{center['x_m']:.9f}:{center['y_m']:.9f}")
    observed = metadata['outcome'].get('observation_evidence')
    if observed is not None:
        assert target_key == observed['target_key']
    return dict(
        candidate_uid=group['candidate_uid'], stream_id=stream_id,
        target_key=target_key, planning_frame='map', stand_center=center,
        candidate_snapshot_sha256=receipt['projected_candidate_snapshot_sha256'],
        **{key: metadata[key] for key in (
            'robot_profile_sha256', 'calibration_profile_sha256',
            'stand_model_profile_sha256')},
    )


def _frame(record):
    metadata = record['metadata']
    image_stamp, scan_stamp = metadata['image_stamp_sec'], metadata['scan_stamp_sec']
    exact = {
        (sample['target_frame'], sample['source_frame']): sample
        for sample in metadata['tf_samples']
        if sample['query_kind'] == 'exact_sensor_time'
    }
    base, scan = exact.get(('map', 'base_footprint')), exact.get(('base_scan', 'map'))
    validated = all(
        sample is not None
        and sample['query_stamp_sec'] == stamp
        and sample['returned_stamp_sec'] == stamp
        for sample, stamp in ((base, image_stamp), (scan, scan_stamp))
    )
    pose = None
    if base is not None:
        x, y, z, w = base['rotation_xyzw']
        pose = dict(
            x_m=base['translation_xyz_m'][0], y_m=base['translation_xyz_m'][1],
            yaw_rad=math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
        )
    observation = metadata['outcome'].get('observation_evidence') or {}
    # The negative sequence has zero accepted frames throughout; the independent
    # positive control is its stream's first capture and first accepted frame.
    # No cumulative counter is interpreted as a general per-frame admission API.
    return dict(
        frame_stamp_sec=image_stamp, scan_stamp_sec=scan_stamp,
        robot_pose=pose, motion_epoch=observation.get('motion_epoch', 0),
        tf_validated=validated, poisoned=observation.get('poisoned', False),
        motion_epoch_reset=bool(observation.get('motion_reset_count', 0)),
        frame_accepted=bool(observation.get('accepted_frame_count', 0)),
    )


def _replay_association(record, *, broad=False):
    metadata = record['metadata']
    original = metadata['detector_metadata']['preliminary_candidate_lidar_association']
    raw = metadata['sensors']['scan']
    header = raw['header']
    scan = PlainLaserScan(
        ranges=tuple(float(value) for value in raw['ranges']),
        angle_min=raw['angle_min'], angle_increment=raw['angle_increment'],
        angle_max=raw['angle_max'], range_min=raw['range_min'], range_max=raw['range_max'],
        scan_frame_id=header['frame_id'],
        scan_stamp_sec=header['stamp_sec'] + header['stamp_nanosec'] / 1e9,
        receipt_sec=metadata['scan_received_ros_sec'],
        scan_topology_profile=original['scan_topology']['profile'],
    )
    options = dict(
        map_bearing_rad=original['map_bearing_rad'],
        cone_half_angle_rad=original['cone_half_angle_rad'],
        accepted_range_m=tuple(original['accepted_range_m']),
        now_sec=metadata['scan_received_ros_sec'] + original['scan_age_sec'],
        max_scan_age_sec=0.5,
        min_cluster_sample_count=original['min_cluster_sample_count'],
        max_range_jump_m=original['max_range_jump_m'],
        max_point_gap_m=original['max_point_gap_m'],
        observed_camera_bearing_rad=original['observed_camera_bearing_rad'],
    )
    if broad:
        for key in ('min_cluster_sample_count', 'max_range_jump_m', 'max_point_gap_m',
                    'observed_camera_bearing_rad'):
            options.pop(key)
        replay = qr_registration_envelope(
            scan, max_camera_map_bearing_delta_rad=math.radians(12), **options,
        )
    else:
        replay = associate_candidate_lidar_target(scan, **options)
    return json.loads(json.dumps(asdict(replay)))


class RecordedTargetSupportFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads((FIXTURE / 'inputs.json').read_text())
        cls.manifest = json.loads((FIXTURE / 'manifest.json').read_text())

    def observe(self, window, group, record, *, association=None):
        return window.observe(
            association=_replay_association(record, broad=True) if association is None else association,
            frame=_frame(record), target_binding=_binding(group, record),
            now_sec=record['source_freshness']['checked_at_sec'],
        )

    def test_fixture_binding_and_raw_scan_replay_preserve_recorded_evidence(self):
        raw = (FIXTURE / 'inputs.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.manifest['fixture_file']['sha256'])
        self.assertEqual(len(raw), self.manifest['fixture_file']['bytes'])
        source_manifest = {
            item['repository_relative_path']: item for item in self.manifest['original_sources']
        }
        for name in ('negative_sequence', 'excluded_tf_retry', 'valid_control'):
            group = self.inputs[name]
            receipt = group['target_receipt']['source']
            self.assertEqual(receipt, source_manifest[receipt['repository_relative_path']])
            for record in group['frames']:
                with self.subTest(group=name, capture=record['capture_index']):
                    source = record['source']
                    self.assertEqual(source, source_manifest[source['repository_relative_path']])
                    if record['metadata']['detector_metadata'] is None:
                        self.assertIsNone(record['metadata']['detector_metadata'])
                    else:
                        self.assertEqual(
                            _replay_association(record),
                            record['metadata']['detector_metadata']['preliminary_candidate_lidar_association'],
                        )
                        if name == 'negative_sequence':
                            replay = _replay_association(record, broad=True)
                            original = record['metadata']['detector_metadata']['metric_model'][
                                'candidate_head_search']['identity_search']['envelope']
                            # Original identity search queried the same raw
                            # envelope later in image processing. Geometry and
                            # cluster evidence agree; only query age differs.
                            self.assertEqual(
                                {k: v for k, v in replay.items() if k != 'scan_age_sec'},
                                {k: v for k, v in original.items() if k != 'scan_age_sec'},
                            )

    def test_contiguous_attempt_defers_at_capture_twenty_without_authority(self):
        group = self.inputs['negative_sequence']
        window = TargetSupportFailureWindow()
        self.assertEqual([r['capture_index'] for r in group['frames']], list(range(1, 21)))
        processed = 0
        for record in group['frames']:
            with self.subTest(capture=record['capture_index']):
                if record['metadata']['detector_metadata'] is None:
                    self.assertEqual(record['metadata']['observer_state'], 'tf_retry_exhausted')
                    self.assertIsNone(record['source_freshness'])
                    self.assertEqual(len(window.samples), 0)
                    continue
                processed += 1
                self.assertTrue(record['source_freshness']['accepted'])
                self.assertTrue(_frame(record)['tf_validated'])
                self.assertFalse(_frame(record)['frame_accepted'])
                receipt = self.observe(window, group, record)
                if record['capture_index'] < 20:
                    self.assertIsNone(receipt)
                    self.assertEqual(window.metadata['sample_count'], processed)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt['state'], 'target_reconciliation_required')
        self.assertEqual(receipt['disposition'], 'target_reconciliation_required')
        self.assertEqual(receipt['reason'], 'persistent_target_support_missing')
        self.assertEqual(receipt['sample_count'], 19)
        self.assertAlmostEqual(receipt['elapsed_sec'], 5.253284454345703)
        self.assertAlmostEqual(receipt['sample_span_sec'], 5.265655994415283)
        for flag in ('motion_authorized', 'candidate_geometry_updated', 'completion_authorized'):
            self.assertIs(receipt[flag], False)
        binding = _binding(group, group['frames'][-1])
        self.assertEqual(receipt['target_binding'], binding)
        self.assertEqual(receipt['target_binding_sha256'], target_support_binding_sha256(binding))
        self.assertEqual(validate_target_support_failure(receipt, target_binding=binding), receipt)
        self.assertEqual(
            [sample['frame']['frame_stamp_sec'] for sample in receipt['samples']],
            [record['metadata']['image_stamp_sec'] for record in group['frames'][1:]],
        )
        for sample in receipt['samples']:
            self.assertEqual(sample['association']['rejection_reason'], 'no_samples_in_accepted_range')
            self.assertEqual(sample['association']['in_range_sample_count'], 0)
            self.assertGreater(sample['association']['cone_valid_sample_count'], 0)
            self.assertGreater(sample['association']['nearest_cone_distance_m'],
                               sample['association']['accepted_range_m'][1])
            self.assertAlmostEqual(sample['association']['cone_half_angle_rad'], math.radians(15))
        # The narrow nominal cone misses all finite returns at capture 16.
        # The real registration envelope still sees explicit background beyond
        # the range gate, so this intermediate frame must not silently reset it.
        intermediate = group['frames'][15]
        self.assertEqual(_replay_association(intermediate)['rejection_reason'],
                         'no_valid_samples_in_map_cone')
        self.assertEqual(_replay_association(intermediate, broad=True)['rejection_reason'],
                         'no_samples_in_accepted_range')

    def test_original_tf_retry_tuple_cannot_supply_a_negative_sample(self):
        group = self.inputs['excluded_tf_retry']
        record = group['frames'][0]
        self.assertEqual(record['metadata']['observer_state'], 'tf_retry_exhausted')
        self.assertFalse(_frame(record)['tf_validated'])
        self.assertEqual(record['metadata']['tf_samples'], [])
        window = TargetSupportFailureWindow()
        # No original association exists for this tuple. None is deliberate:
        # neither a stale detector result nor the next frame's scan is borrowed.
        self.assertIsNone(window.observe(
            association=None, frame=_frame(record), target_binding=_binding(group, record),
            now_sec=record['metadata']['outcome_ros_sec'],
        ))
        self.assertEqual(window.samples, [])
        self.assertEqual(window.metadata['sample_count'], 0)
        self.assertEqual(window.metadata['reason'], 'current exact TF is not validated')

    def test_independent_positive_capture_stays_viable_and_clears_negative_history(self):
        group = self.inputs['valid_control']
        record = group['frames'][0]
        association = _replay_association(record)
        self.assertTrue(association['associated'])
        self.assertEqual(association['selected_cluster_source_indices'], [0, 1, 2])
        self.assertTrue(record['metadata']['detector_metadata']['metric_model']['head_detection']['head_frame_detected'])
        self.assertTrue(_frame(record)['frame_accepted'])
        window = TargetSupportFailureWindow()
        self.assertIsNone(self.observe(window, group, record, association=association))
        self.assertEqual(window.samples, [])
        self.assertFalse(window.metadata['ready'])

        # An independent stream is a reset check, not a claimed chronological
        # continuation at the wall. Original time, target, pose and flags remain
        # intact: the accepted control can never extend negative history.
        negative = self.inputs['negative_sequence']
        for item in negative['frames'][1:7]:
            self.assertIsNone(self.observe(window, negative, item))
        self.assertEqual(len(window.samples), 6)
        self.assertIsNone(self.observe(window, group, record, association=association))
        self.assertEqual(window.samples, [])
        self.assertEqual(window.metadata['sample_count'], 0)
        self.assertFalse(window.metadata['ready'])


if __name__ == '__main__':
    unittest.main()
