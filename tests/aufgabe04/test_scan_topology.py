"""Guarded circular adjacency, including measured LDS receipt geometry."""

from dataclasses import replace
import json
import math
import unittest

from scripts.aufgabe04.perception.scan_topology import ScanTopology


# Real N/min/increment copied from stand_explore_exact2_camera_20260909T123820Z
# visibility receipts: vp001 stamp1788957581.1896076 and vp002 1788957646.0310426.
# Those receipts omit angle_max: tests explicitly reconstruct its uniform-grid
# value and do not claim it is the original LaserScan header's endpoint.
SCAN_GEOMETRIES = (
    (360, 0.0, math.radians(1.0)),
    (224, 0.010003812611103058, 0.028046658262610435),
    (222, 0.012739302590489388, 0.028297076001763344),
)


def topology_for(count, minimum, increment, profile="full_rotation"):
    return ScanTopology(count, minimum, increment,
                        minimum + (count - 1) * increment, profile)


class ScanTopologyTest(unittest.TestCase):
    def test_explicit_full_rotation_accepts_only_actual_endpoint_indices(self):
        for geometry in SCAN_GEOMETRIES:
            topology = topology_for(*geometry)
            with self.subTest(geometry=geometry):
                self.assertTrue(topology.joins_endpoints(geometry[0] - 1, 0))
                self.assertFalse(topology.joins_endpoints(geometry[0] - 2, 0))
                self.assertFalse(topology.joins_endpoints(geometry[0] - 1, 1))
                self.assertFalse(topology.joins_endpoints(0, geometry[0] - 1))
                self.assertEqual(topology.evidence()["reason"], "validated_full_rotation")

    def test_circularity_requires_profile_and_original_maximum(self):
        topology = topology_for(*SCAN_GEOMETRIES[0])
        for changed in (replace(topology, profile="linear"), replace(topology, profile="unknown"),
                        replace(topology, angle_max_rad=None)):
            self.assertFalse(changed.joins_endpoints(359, 0))

    def test_partial_two_step_and_overlapping_scans_never_join(self):
        for count, increment in ((180, math.radians(1)), (359, math.radians(1)),
                                 (361, math.radians(1)), (362, math.radians(1)),
                                 (224, math.tau / 225)):
            with self.subTest(count=count, increment=increment):
                self.assertFalse(topology_for(count, 0.0, increment).joins_endpoints(count - 1, 0))

    def test_metadata_mismatch_cannot_certify_an_index_gap_or_header_gap(self):
        normal = topology_for(*SCAN_GEOMETRIES[0])
        variants = (
            replace(normal, angle_max_rad=math.radians(350)),
            replace(normal, angle_max_rad=math.radians(358)),
            replace(normal, sample_count=359),
            replace(normal, angle_max_rad=0.0),
        )
        for topology in variants:
            self.assertFalse(topology.joins_endpoints(topology.sample_count - 1, 0))

    def test_guard_has_bounded_tolerance_and_supports_clockwise_scan(self):
        normal = topology_for(*SCAN_GEOMETRIES[0])
        self.assertTrue(replace(normal, angle_max_rad=normal.angle_max_rad + math.radians(.05)).joins_endpoints(359, 0))
        self.assertFalse(replace(normal, angle_max_rad=normal.angle_max_rad + math.radians(.11)).joins_endpoints(359, 0))
        reverse = topology_for(360, math.pi, -math.radians(1))
        self.assertTrue(reverse.joins_endpoints(359, 0))

    def test_malformed_metadata_has_json_safe_rejection_evidence(self):
        normal = topology_for(*SCAN_GEOMETRIES[0])
        for field, value in (("angle_min_rad", float("nan")), ("angle_max_rad", float("inf")),
                             ("angle_increment_rad", 0.0), ("angle_increment_rad", True),
                             ("sample_count", True), ("sample_count", 1),
                             ("sample_count", None), ("sample_count", "360")):
            topology = replace(normal, **{field: value})
            self.assertFalse(topology.joins_endpoints(359, 0))
            json.dumps(topology.evidence(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
