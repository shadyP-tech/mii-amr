"""Recorded centering boundary regression."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory
from scripts.aufgabe04.real_robot.observer.inspection_framing import review_centering_destination

FIXTURE = Path(__file__).parent / 'fixtures/backside_framing_20260922.json'


class InspectionFramingTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(FIXTURE.read_text())
        self.advisory = validate_camera_centering_advisory(self.record['advisory'])
        self.search = SimpleNamespace(**self.record['initial_search_association'])

    def test_recorded_turn_is_vetoed_without_changing_hashed_advice(self):
        original = deepcopy(self.record)
        result = review_centering_destination(self.advisory, search_association=self.search)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, 'preserve_view_centering_would_cross_scan_boundary')
        self.assertAlmostEqual(math.degrees(self.advisory.requested_yaw_rad), 3.85828665, places=6)
        self.assertAlmostEqual(math.degrees(result.predicted_point_bearing_rad), -.962, places=2)
        self.assertLess(result.minimum_clearance_rad, result.required_clearance_rad)
        self.assertEqual(original, self.record)
        self.assertFalse(result.metadata()['motion_authorized'])

    def test_safe_boundary_both_scan_directions_and_wrapped_angles(self):
        for start, step in ((-math.pi, math.tau/360), (math.pi, -math.tau/360),
                            (3*math.pi, math.tau/360)):
            self.search.scan_topology = dict(profile='full_rotation', sample_count=360,
                angle_min_rad=start, angle_increment_rad=step)
            self.assertTrue(review_centering_destination(self.advisory,
                search_association=self.search).allowed)

    def test_valid_circular_metadata_does_not_bypass_boundary_veto(self):
        self.search.scan_topology = dict(profile='full_rotation', sample_count=360,
            angle_min_rad=0., angle_increment_rad=math.tau/360, circular_adjacency_enabled=True)
        self.assertFalse(review_centering_destination(self.advisory, search_association=self.search).allowed)

    def test_unavailable_full_rotation_geometry_fails_closed_linear_unchanged(self):
        for change in ({'angle_increment_rad': 0.}, {'sample_count': 0}, {'angle_min_rad': math.nan}):
            self.search.scan_topology = {**self.record['initial_search_association']['scan_topology'], **change}
            result = review_centering_destination(self.advisory, search_association=self.search)
            self.assertFalse(result.allowed)
            self.assertEqual(result.reason, 'centering_boundary_geometry_unavailable')
        self.search.scan_topology = {'profile': 'linear'}
        self.assertTrue(review_centering_destination(self.advisory, search_association=self.search).allowed)
