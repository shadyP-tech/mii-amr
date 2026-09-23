from dataclasses import replace
import math

import pytest

from scripts.aufgabe04.perception.stand_axis.nearest_scan_head import nearest_scan_head_search
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform
from tests.aufgabe04.test_physical_head_pipeline import profile


def scan(clusters=((0.,.5),)):
    ranges = [math.inf]*121
    for bearing, distance in clusters:
        for i in range(121):
            a=-.6+i*.01
            if abs(a-bearing)<=.035:
                ranges[i]=distance/math.cos(a-bearing)
    return PlainLaserScan(tuple(ranges),-.6,.01,.08,3.5,"base_scan",10.,10.01)


def search(profile,current=None,**changes):
    q=(-.5,.5,-.5,.5)
    args=dict(scan=scan() if current is None else current,image_stamp_sec=10.,now_sec=10.1,max_scan_age_sec=.5,
        scan_from_camera=RigidTransform("base_scan","camera",(.05,0.,-.06),q),
        base_from_camera=RigidTransform("base_footprint","camera",(.05,0.,.12),q),
        model_profile=profile,fx=640.,fy=640.,cx=400.,cy=300.,image_shape=(600,800))
    args.update(changes)
    return nearest_scan_head_search(**args)


def test_nearest_current_target_projects_floor_height_and_physical_scale(profile):
    result, info=search(profile,scan(((0.,.5),(.4,1.2))))
    assert result is not None,info
    assert result.center[0]==pytest.approx(400.,abs=1.)
    assert result.height==pytest.approx(640*.078/.45,abs=1.)
    assert result.pixel_size.depth_m==pytest.approx(.45,abs=.005)
    assert result.center[1]==pytest.approx(300-640*(.171-.12)/.45,abs=1.)
    assert info["policy"]=="nearest_current_scan_head"
    assert not info["supplies_corners"] and not info["motion_authorized"]


@pytest.mark.parametrize("change",({"scan_stamp_sec":9.},{"receipt_sec":9.},
    {"scan_stamp_sec":10.2},{"scan_frame_id":"wrong"},{"scan_stamp_sec":None}))
def test_stale_future_or_wrong_frame_scan_never_supplies_target(profile,change):
    result,_=search(profile,replace(scan(),**change))
    assert result is None


def test_missing_floor_extrinsic_and_tied_ranges_stay_unavailable(profile):
    assert search(profile,base_from_camera=None)[0] is None
    result,info=search(profile,scan(((-.25,.5),(.25,.5))))
    assert result is None and info["reason"]=="nearest_head_scan_candidates_ambiguous"


def seam_scan(**changes):
    step = math.tau/360
    ranges = [math.inf]*360
    for index in (-3, -2, -1, 0, 1, 2, 3):
        ranges[index] = .5/math.cos(index*step)
    return replace(PlainLaserScan(tuple(ranges), 0., step, .08, 3.5,
        "base_scan", 10., 10.01, 359*step, "full_rotation"), **changes)


def test_validated_scan_seam_is_joined_before_nearest_target_comparison(profile):
    prior, info = search(profile, seam_scan())
    assert prior is not None, info
    assert info["scan_topology"]["circular_adjacency_enabled"]
    assert len(info["candidates"]) == 1
    assert info["candidates"][0]["scan_indices"] == (357, 358, 359, 0, 1, 2, 3)
    assert info["candidates"][0]["wraps_scan_seam"]
    assert prior.edge_region is not None and prior.center_bounds_px is not None


@pytest.mark.parametrize("changes", ({"scan_topology_profile": "linear"},
    {"angle_max": None}, {"angle_max": 358*math.tau/360},
    {"angle_increment": math.pi/360, "angle_max": 359*math.pi/360}))
def test_partial_or_unproven_scan_seam_never_merges(profile, changes):
    _, info = search(profile, seam_scan(**changes))
    assert not info["scan_topology"]["circular_adjacency_enabled"]
    assert not any(c["wraps_scan_seam"] for c in info["candidates"])


def test_full_rotation_does_not_bridge_a_missing_endpoint(profile):
    current = seam_scan()
    ranges = list(current.ranges)
    ranges[-1] = math.inf
    _, info = search(profile, replace(current, ranges=tuple(ranges)))
    assert not any(c["wraps_scan_seam"] for c in info["candidates"])


def test_viewer_preserves_original_scan_geometry_and_recording_values():
    from collections import deque
    import threading
    from types import SimpleNamespace
    from scripts.aufgabe04.perception.debug.stand_axis_viewer import RosLaserScanRangeSource
    from scripts.aufgabe04.perception.debug.recording_metadata import recording_metadata
    source = object.__new__(RosLaserScanRangeSource)
    source._lock, source._scans = threading.Lock(), deque(maxlen=80)
    source.scan_topology_profile = "full_rotation"
    source._on_scan(SimpleNamespace(ranges=(.5, math.inf, .6), angle_min=0.,
        angle_increment=math.tau/3, angle_max=2*math.tau/3, range_min=.08, range_max=3.5,
        header=SimpleNamespace(frame_id="base_scan", stamp=SimpleNamespace(sec=10, nanosec=0))))
    scan = source.latest_scan()
    assert scan.angle_max == 2*math.tau/3 and scan.scan_topology_profile == "full_rotation"
    metadata = recording_metadata(scan)
    assert metadata["ranges"] == [.5, None, .6]
    assert metadata["angle_increment"] == math.tau/3
    assert metadata["scan_stamp_sec"] == 10.


def test_exploration_selects_nearest_only_after_candidate_region_filter(profile):
    from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
    # A closer neighbor exists at image left, outside this candidate's region.
    current=scan(((0.,.7),(.4,.4)))
    unbounded,info=search(profile,current)
    assert unbounded.center[0] < 200
    bound=CandidateHeadSearch((400.,250.),100.,center_bounds_px=(300.,100.,500.,400.))
    selected,info=search(profile,current,bounded_search=bound)
    assert selected is not None
    assert selected.center[0] == pytest.approx(400.,abs=1.)
    assert info['candidate_association_required']
    assert not info['motion_authorized']


def test_bounded_nearest_acquisition_preserves_tied_target_ambiguity(profile):
    from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
    bound=CandidateHeadSearch((400.,250.),160.,center_bounds_px=(50.,50.,750.,500.))
    result,info=search(profile,scan(((-.2,.5),(.2,.5))),bounded_search=bound)
    assert result is None
    assert info['reason']=='nearest_head_scan_candidates_ambiguous'


def test_candidate_range_excludes_foreground_and_background_before_nearest_selection(profile):
    current = scan(((-.25,.35),(0.,.7),(.3,1.2)))
    result, info = search(profile,current,accepted_range_m=(.6,.8))
    assert result.center[0] == pytest.approx(400.,abs=1.)
    assert len(info['candidates']) == 1
    assert search(profile,current,accepted_range_m=(.8,1.))[0] is None
