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
