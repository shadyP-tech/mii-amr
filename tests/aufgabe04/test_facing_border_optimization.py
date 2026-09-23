"""Frame-local reuse must preserve ambiguity and current raw-pixel decisions."""
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from tests.aufgabe04.facing_geometry_fixture import recorded
from scripts.aufgabe04.perception.stand_axis import head_border_families as families
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.model_pipeline import estimate_stand_axis_from_metric_model
from scripts.aufgabe04.perception.stand_color_support import color_edge_support


def test_shared_rails_extracted_once_and_decisions_match_uncached_reference():
    frame,_,_=recorded('green')
    raw=_canny_edges_from_frame(cv2,frame,edge_preprocess='channel_union',blur_kernel=5,canny_low=20,canny_high=60)
    quads=[tuple(ImagePoint(*p) for p in ((302.,250.+n),(402.,250.+n),(402.,350.),(302.,350.))) for n in range(5)]
    expected={(i,j):families.same_current_border_family(a,b,raw_edges=raw,frame_bgr=frame)
              for i,a in enumerate(quads) for j,b in enumerate(quads)}
    cache=families.CurrentBorderFamilies(raw,frame)
    with patch.object(families,'_rail_pixels',wraps=families._rail_pixels) as extract:
        for i,a in enumerate(quads):
            for j,b in enumerate(quads):assert cache.same(a,b)==expected[i,j]
        calls=extract.call_count
        for a in quads:
            for b in quads:cache.same(a,b)
        assert extract.call_count==calls
        assert calls<4*len(quads)  # Shared bottom rail is reused across quads.
    # A new frame owns its own cache; no earlier colour or support verdict leaks.
    other=families.CurrentBorderFamilies(np.zeros_like(raw),np.zeros_like(frame))
    assert not other.same(quads[0],quads[1])


@pytest.mark.parametrize('name',['blue','green'])
def test_recorded_frames_keep_uncertainty_and_do_not_invent_head_geometry(name):
    frame,search,info=recorded(name)
    model=load_measured_physical_stand_model(Path(__file__).resolve().parents[2]/
        'configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json')
    estimate,debug=estimate_stand_axis_from_metric_model(cv2,frame,model_profile=model,
        camera_fx_px=info.p[0],camera_fy_px=info.p[5],camera_cx_px=info.p[2],camera_cy_px=info.p[6],
        candidate_search=search,qr_observations=(),color_support_mask=color_edge_support(cv2,np,frame))
    assert debug.head_orientation_bounds.accepted
    assert debug.head_orientation_bounds.half_width_rad>0
    assert estimate.usable if name=='blue' else estimate.reason=='head_model_yaw_uncertainty_too_high'
    acquisition=debug.head_acquisition_diagnostics['acquisition']
    assert acquisition['raw_verifications']<=12
    assert acquisition['locator']=='material_rim'


def test_batched_profiles_match_recorded_scalar_policy_on_1000_cross_sections():
    import json
    from scripts.aufgabe04.perception.stand_axis.current_rail_profiles import single_thin_stripe
    reference=json.loads((Path(__file__).parent/'fixtures/facing_geometry_20260923/rail_profiles.json').read_text())
    rng=np.random.default_rng(reference['seed'])
    for name,expected in reference['expected'].items():
        image,_,_=recorded(name)
        actual=[]
        for _ in expected:
            a=rng.uniform([280,240],[400,350])
            delta=rng.uniform([20,-5],[100,5])
            offset=rng.uniform([-1,-6],[1,6])
            points=[ImagePoint(*p) for p in (a,a+delta,a+offset,a+delta+offset)]
            actual.append('1' if single_thin_stripe(image,*points) else '0')
        assert ''.join(actual)==expected
