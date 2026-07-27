from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Mapping


CAPTURE_SCHEMA_VERSION = 1


def sensor_frame_status(
    *,
    source_stamp_sec: float | None,
    received_wall_sec: float | None,
    max_frame_age_sec: float,
    future_tolerance_sec: float = 0.25,
) -> str:
    if source_stamp_sec is None:
        return "no_header"
    if received_wall_sec is None:
        return "clock_unverified"
    age_sec = received_wall_sec - source_stamp_sec
    if age_sec < -future_tolerance_sec:
        return "header_future"
    if max_frame_age_sec <= 0.0:
        return "clock_unverified"
    if age_sec > max_frame_age_sec:
        return "header_stale"
    return "header_age_ok"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_png(cv2, path: Path, image) -> None:
    if image is None:
        return
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError(f"could not encode capture image: {path.name}")
    _atomic_write_bytes(path, bytes(payload))


def save_structural_capture(
    cv2,
    directory: Path,
    *,
    original_compressed: bytes | None,
    compressed_format: str | None,
    decoded_frame,
    candidate_roi_frame,
    raw_edges,
    localization_edges,
    side_evidence,
    rectangle_mask,
    annotated_frame,
    metadata: Mapping[str, object],
) -> Path:
    """Persist one unambiguous, content-separated diagnostic capture."""

    directory.mkdir(parents=True, exist_ok=True)
    capture_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    prefix = directory / f"{capture_id}_stand_structure"
    files = {
        "compressed": (
            None
            if original_compressed is None
            else f"{prefix.name}_original.compressed"
        ),
        "decoded_frame": f"{prefix.name}_decoded.png",
        "candidate_roi": f"{prefix.name}_candidate_roi.png",
        "raw_edges": f"{prefix.name}_raw_edges.png",
        "localization_edges": f"{prefix.name}_localization_edges.png",
        "side_evidence": f"{prefix.name}_side_evidence.png",
        "rectangle": f"{prefix.name}_rectangle.png",
        "annotated": f"{prefix.name}_annotated.png",
    }
    if original_compressed is not None:
        _atomic_write_bytes(directory / files["compressed"], original_compressed)
    _write_png(cv2, directory / files["decoded_frame"], decoded_frame)
    _write_png(cv2, directory / files["candidate_roi"], candidate_roi_frame)
    _write_png(cv2, directory / files["raw_edges"], raw_edges)
    _write_png(cv2, directory / files["localization_edges"], localization_edges)
    _write_png(cv2, directory / files["side_evidence"], side_evidence)
    _write_png(cv2, directory / files["rectangle"], rectangle_mask)
    _write_png(cv2, directory / files["annotated"], annotated_frame)

    record = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "capture_id": capture_id,
        "observe_only": True,
        "authoritative": False,
        "compressed_format": compressed_format or "",
        "files": files,
        **dict(metadata),
    }
    metadata_path = directory / f"{prefix.name}.json"
    _atomic_write_json(metadata_path, record)
    print(f"saved structural capture: {metadata_path}")
    return metadata_path
