"""Recorded acquisition with the real source-domain filter and competing heads."""
from dataclasses import replace
import time
import cv2
import numpy as np
import pytest
from pathlib import Path

from tests.aufgabe04.facing_geometry_fixture import recorded
from scripts.aufgabe04.perception.camera_calibration import camera_calibration_from_info,rectified_source_support
from scripts.aufgabe04.perception.stand_axis.image_source_support import ImageSourceSupport
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame
from scripts.aufgabe04.perception.stand_axis.material_rim_acquisition import acquire_material_rim
from scripts.aufgabe04.perception.stand_axis.head_geometry_acquisition import estimate_current_head_geometry
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import LidarHeadEdgeRegion

MODEL=load_measured_physical_stand_model(Path('configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json'))


def inputs(name):
    frame,search,info=recorded(name)
    source=ImageSourceSupport(cv2,rectified_source_support(camera_calibration_from_info(info),cv2,np))
    return frame,search,info,source


def acquire(frame,search,source,**kw):
    raw=_canny_edges_from_frame(cv2,frame,edge_preprocess='channel_union',blur_kernel=5,canny_low=20,canny_high=60)
    return acquire_material_rim(cv2,frame,raw_edges=raw,candidate_search=search,model_profile=MODEL,
        proposal_filter=source.filter(),**kw),raw


@pytest.mark.parametrize('name',['blue','green'])
def test_real_filtered_pipeline_resolves_recorded_failures_without_false_precision(name):
    frame,search,info,source=inputs(name)
    est,debug=estimate_current_head_geometry(cv2,frame,model_profile=MODEL,
        camera_fx_px=info.p[0],camera_fy_px=info.p[5],camera_cx_px=info.p[2],camera_cy_px=info.p[6],
        candidate_search=search,source_support=source)
    assert debug.head_acquisition_diagnostics['source']=='current_material_rim_search'
    assert debug.head_orientation_bounds.accepted
    assert debug.head_orientation_bounds.half_width_rad>0
    assert est.usable if name=='blue' else est.reason=='head_model_yaw_uncertainty_too_high'
    assert debug.head_acquisition_diagnostics['acquisition']['raw_verifications']<=12
    assert debug.head_acquisition_diagnostics['current_boundary_reused']


@pytest.mark.parametrize('other',['same','red','gray'])
def test_other_materials_and_gray_neighbors_are_not_hidden(other):
    frame,search,info,source=inputs('green')
    piece=frame[225:375,285:425].copy()
    if other=='gray':piece=cv2.cvtColor(cv2.cvtColor(piece,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    if other=='red':
        hsv=cv2.cvtColor(piece,cv2.COLOR_BGR2HSV);hsv[:,:,0][hsv[:,:,1]>65]=3
        piece=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    frame[225:375,445:585]=piece
    search=replace(search,center=(415.,300.),center_tolerance_px=170.,
        center_bounds_px=(220.,150.,650.,430.),edge_region=LidarHeadEdgeRegion((600,800),(220,150,650,430)))
    result,_=acquire(frame,search,source)
    assert result is not None and result.proposal is None
    assert result.reason=='head_proposal_ambiguous'


@pytest.mark.parametrize('name',['blue','green'])
@pytest.mark.parametrize('case',['missing_top','gray','dim'])
def test_no_material_proof_leaves_ordinary_fallback_available(name,case):
    frame,search,info,source=inputs(name)
    if case=='missing_top':frame[240:268,285:415]=frame[210:238,285:415]
    if case=='gray':frame=cv2.cvtColor(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    if case=='dim':frame=np.zeros_like(frame)
    result,_=acquire(frame,search,source)
    assert result is None or result.proposal is None


def test_expired_attempt_never_publishes_or_restarts_budget():
    frame,search,_,source=inputs('blue')
    result,_=acquire(frame,search,source,deadline_monotonic_sec=time.monotonic()-1)
    assert result.reason=='head_acquisition_deadline_exceeded' and result.proposal is None


def test_refinement_proof_binds_full_raw_edges_and_exact_current_image():
    frame,search,_,source=inputs('blue');out={}
    result,raw=acquire(frame,search,source,refinement_out=out)
    proof=out['selected'];q=result.proposal.corners
    proof.resolve(frame,raw,model_profile=MODEL,proposal_corners=q)
    changed=raw.copy();changed[260:320,292:302]=255-changed[260:320,292:302]
    with pytest.raises(ValueError,match='supporting pixels changed'):
        proof.resolve(frame,changed,model_profile=MODEL,proposal_corners=q)
    with pytest.raises(ValueError,match='another image'):
        proof.resolve(frame.copy(),raw,model_profile=MODEL,proposal_corners=q)


@pytest.mark.parametrize('hue,label',[(4,'red'),(175,'red'),(140,'purple'),(110,'blue'),(60,'green')])
def test_material_is_acquired_without_station_color_identity(hue,label):
    frame,search,_,source=inputs('green')
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    material=(hsv[:,:,0]>=38)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=65)
    hsv[:,:,0][material]=hue
    hsv[:,:,1][material]=np.maximum(hsv[:,:,1][material],180)
    recolored=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result,_=acquire(recolored,search,source)
    assert result is not None and result.proposal is not None
    assert label in [m['label'] for m in result.joint_border_diagnostics['materials']]


def test_failed_rim_search_reports_consumed_budget_for_ordinary_fallback():
    from unittest.mock import patch
    from scripts.aufgabe04.perception.stand_axis.head_proposal import HeadProposalResult
    frame,search,_,source=inputs('blue');budget={}
    failed=HeadProposalResult(None,'no_current_head_proposal',0,4,'test',{})
    with patch('scripts.aufgabe04.perception.stand_axis.material_rim_acquisition.acquire_cold_head_proposal',return_value=failed):
        result,_=acquire(frame,search,source,budget_out=budget)
    assert result is None and budget['used']==4


def test_current_candidate_filter_remains_an_independent_admission_veto():
    frame,search,_,source=inputs('blue')
    raw=_canny_edges_from_frame(cv2,frame,edge_preprocess='channel_union',blur_kernel=5,canny_low=20,canny_high=60)
    result=acquire_material_rim(cv2,frame,raw_edges=raw,candidate_search=search,model_profile=MODEL,
        proposal_filter=lambda proposal:False)
    assert result is None or result.proposal is None


def test_production_fallback_spends_only_remaining_verification_slots():
    from unittest.mock import patch
    from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import acquire_cold_head_proposal
    frame,search,info,source=inputs('green')
    def miss(*args,**kwargs):
        kwargs['budget_out']['used']=4
        return None
    with patch('scripts.aufgabe04.perception.stand_axis.material_rim_acquisition.acquire_material_rim',side_effect=miss), \
         patch('scripts.aufgabe04.perception.stand_axis.physical_head_pipeline.acquire_cold_head_proposal',wraps=acquire_cold_head_proposal) as fallback:
        estimate_current_head_geometry(cv2,frame,model_profile=MODEL,
            camera_fx_px=info.p[0],camera_fy_px=info.p[5],camera_cx_px=info.p[2],camera_cy_px=info.p[6],
            candidate_search=search,source_support=source)
    assert fallback.call_count==1
    assert fallback.call_args.kwargs['_verification_limit']==8
