"""A real fragmented scan supplies a search region, never unique-target proof."""

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.aufgabe04.perception.stand_axis.nearest_scan_head import nearest_scan_head_search
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan


@pytest.fixture
def recorded():
    row = json.loads((Path(__file__).parent / "fixtures/endpoint_search_20260921.json").read_text())
    raw = row["head_target_scan"]
    scan = PlainLaserScan(**{**raw, "ranges": tuple(math.nan if v is None else v for v in raw["ranges"])})
    intr = row["processing_intrinsics"]
    model = Path(__file__).resolve().parents[2] / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
    return dict(scan=scan, image_stamp_sec=row["source_stamp_sec"],
        now_sec=row["received_wall_sec"]+row["detector_completed_monotonic_sec"]-row["received_monotonic_sec"],
        max_scan_age_sec=.5,
        scan_from_camera=SimpleNamespace(**row["calibration"]["scan_from_camera"]),
        base_from_camera=SimpleNamespace(**row["calibration"]["base_from_camera"]),
        model_profile=load_measured_physical_stand_model(model), fx=intr["fx_px"], fy=intr["fy_px"],
        cx=intr["cx_px"], cy=intr["cy_px"], image_shape=(600, 800))


def test_recorded_boundary_fragments_supply_only_a_current_image_search_region(recorded):
    scan = recorded["scan"]
    prior, info = nearest_scan_head_search(**recorded)
    assert prior is not None
    assert prior.center[0] == pytest.approx(403.03, abs=.01)
    # The recorded head spans approximately x=357..452; the prior covers it.
    assert abs(prior.center[0] - 404.5) < 2.
    assert not info["scan_topology"]["circular_adjacency_enabled"]
    assert [c["scan_indices"] for c in info["candidates"][:2]] == [(0, 1), (223, 224, 225)]
    assert not any(c["wraps_scan_seam"] for c in info["candidates"])
    hint = info["endpoint_fragment_search"]
    assert hint["raw_cluster_count"] == 2
    assert not hint["target_uniqueness_proven"]
    assert not hint["supplies_corners"] and not hint["motion_authorized"]
    assert recorded["scan"] is scan  # Original acquisition metadata/ranges retained.


@pytest.mark.parametrize("change", ["linear", "missing_metadata", "large_gap", "endpoint_missing",
                                    "spatial_gap", "third_object", "stale"])
def test_search_hint_cannot_override_missing_context_or_competing_targets(recorded, change):
    scan = recorded["scan"]
    ranges = list(scan.ranges)
    if change == "linear":
        scan = replace(scan, scan_topology_profile="linear")
    elif change == "missing_metadata":
        scan = replace(scan, angle_max=None)
    elif change == "large_gap":
        scan = replace(scan, angle_increment=scan.angle_increment * .98)
    elif change == "endpoint_missing":
        ranges[-1] = math.nan
    elif change == "spatial_gap":
        # Keep radial difference below the nearest ambiguity threshold while
        # increasing the actual endpoint separation past the 40 mm bound.
        ranges[-1] = ranges[0] + .039
    elif change == "third_object":
        for index in (217, 218):
            ranges[index] = .56
    else:
        scan = replace(scan, scan_stamp_sec=scan.scan_stamp_sec-1.)
    recorded["scan"] = replace(scan, ranges=tuple(ranges))
    prior, info = nearest_scan_head_search(**recorded)
    assert "endpoint_fragment_search" not in info
    assert prior is None


def test_internal_two_object_ambiguity_is_preserved(recorded):
    scan = recorded["scan"]
    ranges = [math.nan]*len(scan.ranges)
    for index in (3, 4, 214, 215):
        ranges[index] = .55
    recorded["scan"] = replace(scan, ranges=tuple(ranges))
    prior, info = nearest_scan_head_search(**recorded)
    assert prior is None and info["reason"] == "nearest_head_scan_candidates_ambiguous"
