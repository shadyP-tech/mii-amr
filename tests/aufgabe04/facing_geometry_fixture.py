"""Rectify recorded frames and restore their original bounded search inputs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np

from scripts.aufgabe04.perception.camera_calibration import rectify_bgr_frame
from scripts.aufgabe04.perception.stand_axis.candidate_head_search import CandidateHeadSearch
from scripts.aufgabe04.perception.stand_axis.lidar_head_edge_region import LidarHeadEdgeRegion
from scripts.aufgabe04.perception.stand_axis.metric_head_search import ProjectedHeadSize


def recorded(name):
    root = Path(__file__).parent/'fixtures/facing_geometry_20260923'
    data = json.loads((root/(name+'.json')).read_text())
    image = root/(name+'.jpg')
    assert hashlib.sha256(image.read_bytes()).hexdigest() == data['image_sha256']
    info = data['camera_info']
    info = SimpleNamespace(**{**info, 'roi': SimpleNamespace(**info['roi'])})
    frame = rectify_bgr_frame(cv2.imread(str(image)), info, cv2, np)
    screen = data['search']
    region, size = screen['edge_region'], screen['pixel_size']
    search = CandidateHeadSearch(
        center=(screen['center_u_px'], screen['center_v_px']), height=screen['height_px'],
        max_center_offset_ratio=screen['max_center_offset_ratio'],
        edge_region=LidarHeadEdgeRegion(tuple(region['image_shape']), tuple(region['bounds_xyxy'])),
        pixel_size=ProjectedHeadSize(**{key: size[key] for key in ProjectedHeadSize.__dataclass_fields__}),
        center_tolerance_px=screen['center_tolerance_px'], center_bounds_px=tuple(screen['center_bounds_px']))
    return frame, search, info
