from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


def parse_roi(value: str) -> Rect:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must use x,y,w,h")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be positive")
    return Rect(x, y, width, height)


def clamp_roi(roi: Rect, frame_shape: Sequence[int]) -> Rect:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    x = max(0, min(roi.x, frame_w))
    y = max(0, min(roi.y, frame_h))
    width = max(0, min(roi.width, frame_w - x))
    height = max(0, min(roi.height, frame_h - y))
    return Rect(x, y, width, height)
