"""Eligibility for witnessed endpoint fragments, never circular scan authority."""

import math

from scripts.aufgabe04.perception.scan_topology import ScanTopology


ENDPOINT_WITNESS_KIND = "scan_endpoint_fragments_witnessed"
INTERNAL_WITNESS_KIND = "one_internal_missing_beam_witnessed"


def endpoint_fragments(scan, clusters):
    """Bound the original scan gap; a caller must still prove real continuity.

    The LDS receipts report a near-one-step header seam while their indexed
    seam varies between one and two steps. Do not repair their angles or join
    their groups here. Only three independently valid scans can witness this
    limited missing interval, with the unchanged spatial and freshness gates.
    """
    if len(clusters) != 2:
        return False
    left, right = sorted(clusters, key=lambda c: c.start_index)
    if not (left.start_index == 0 and right.end_index == len(scan.ranges) - 1
            and left.end_index < right.start_index
            and right.start_index <= right.end_index):
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
