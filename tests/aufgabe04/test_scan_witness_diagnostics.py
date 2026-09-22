"""Recorded raw beams retain strict witness proof with bounded diagnostics."""
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import ScanPersistenceContext, StoppedScanTargetPersistence
from scripts.aufgabe04.real_robot.observer.scan_witness_diagnostics import ScanWitnessDiagnostics


class ScanWitnessDiagnosticsTests(unittest.TestCase):
    def test_recorded_endpoint_fragments_do_not_invent_witnesses(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/backside_framing_20260922.json').read_text())
        state = StoppedScanTargetPersistence()
        for row in fixture['rows']:
            entry = row['input']
            scan = PlainLaserScan(**{**entry['scan'], 'ranges': tuple(
                math.nan if value is None else value for value in entry['scan']['ranges'])})
            context = ScanPersistenceContext(**{**entry['context'], **{
                key: Pose2D(**entry['context'][key]) for key in ('robot_pose', 'scan_pose_map', 'scan_pose_robot')}})
            timing = dict(now_sec=entry['now_sec'], max_scan_age_sec=entry['max_scan_age_sec'])
            raw = associate_camera_registered_candidate_lidar_target(scan, **entry['parameters'], **timing)
            before = state.diagnostics.snapshot()
            state.preview(raw, scan, context=context, **timing)
            self.assertEqual(before, state.diagnostics.snapshot(), row['frame'])
            resolved = state.resolve(raw, scan, context=context, **timing)
            self.assertEqual(len(state._history), row['expected_history_count'], row['frame'])
            self.assertEqual(resolved.associated, row == fixture['rows'][0], row['frame'])
        summary = state.diagnostics.snapshot()
        self.assertEqual(summary['counts']['unique_witness'], 1)
        self.assertEqual(summary['counts']['fragment_only'], 6)
        self.assertIn('gap', state.last_metadata['history_reset_reason'])
        self.assertTrue(any(e.get('raw_fragment_source_indices') for e in summary['recent_events']))
        self.assertTrue(any('gap' in (e['reason'] or '') for e in summary['recent_events'] if e['stage'] == 'history_reset'))
        json.dumps(summary, allow_nan=False)

    def test_bounded_coalesced_snapshot_and_preview_clone_are_isolated(self):
        diagnostics = ScanWitnessDiagnostics()
        for stamp in range(40):
            diagnostics.record('tf_pending', stamp=stamp, reason='exact TF unavailable')
        self.assertEqual(len(diagnostics.events), 32)
        clone = diagnostics.clone()
        clone.record('tf_pending', stamp=39, reason='exact TF unavailable')
        self.assertNotIn('repeated_count', diagnostics.events[-1])
        self.assertEqual(clone.events[-1]['repeated_count'], 2)
        snapshot = diagnostics.snapshot()
        snapshot['recent_events'][-1]['reason'] = 'external mutation'
        self.assertEqual(diagnostics.events[-1]['reason'], 'exact TF unavailable')
        self.assertEqual(diagnostics.counts['tf_pending'], 40)
