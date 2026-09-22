"""Bounded raw endpoint gaps, without granting circular scan continuity."""

import math

from scripts.aufgabe04.perception.scan_topology import ScanTopology


def bounded_endpoint_fragments(scan, index_groups):
    """Recognize the LDS gap eligible for a search hint or temporal proof.

    This is only eligibility: callers needing target uniqueness must still
    prove continuity using independent scans. Original angles remain intact.
    """
    if len(index_groups) != 2 or any(not group for group in index_groups):
        return False
    left, right = sorted(index_groups, key=lambda group: group[0])
    if (left[0] != 0 or right[-1] != len(scan.ranges) - 1 or left[-1] >= right[0]
            or any(b != a + 1 for group in (left, right) for a, b in zip(group, group[1:]))):
        return False
    topology = ScanTopology(len(scan.ranges), scan.angle_min, scan.angle_increment,
                            scan.angle_max, scan.scan_topology_profile).evidence()
    if (topology["profile"] != "full_rotation"
            or topology["reason"] not in {"inconsistent_scan_endpoint_metadata",
                                         "seam_is_not_one_sampling_step"}
            or not 0 < abs(scan.angle_increment) <= math.radians(2.)):
        return False
    indexed, reported, error = (topology[key] for key in (
        "indexed_seam_gap_steps", "reported_seam_gap_steps", "endpoint_metadata_error_steps"))
    return (all(type(v) in (int, float) and math.isfinite(v) for v in (indexed, reported, error))
            and .9 <= indexed <= 2.1 and .9 <= reported <= 1.1 and error <= 1.1)
