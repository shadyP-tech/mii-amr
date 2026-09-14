"""Lossless current-image fixture for the 2026-09-14 backside neck failure."""

import hashlib
import json
from pathlib import Path

_DIRECTORY = Path(__file__).parent / "fixtures/backside_neck_20260914"
_METADATA = json.loads((_DIRECTORY / "inputs.json").read_text())
CAMERA = _METADATA["camera"]
CORNERS = tuple(tuple(point) for point in _METADATA["corners"])
EXPECTED_HEAD = tuple(_METADATA["expected_head"])
MIN_EDGE_HEIGHT_PX = _METADATA["min_edge_height_px"]


def recorded_backside_neck(cv2, numpy):
    """Verify source bytes before decoding; no pixels are edited or rescaled."""

    images = []
    for filename, hash_key, mode in (
        ("head_roi.png", "head_roi_sha256", cv2.IMREAD_COLOR),
        ("raw_edges.png", "raw_edges_sha256", cv2.IMREAD_GRAYSCALE),
    ):
        data = (_DIRECTORY / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != _METADATA[hash_key]:
            raise ValueError(f"recorded backside fixture hash mismatch: {filename}")
        images.append(cv2.imdecode(numpy.frombuffer(data, numpy.uint8), mode))
    return tuple(images)
