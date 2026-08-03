"""Immutable metric stand geometry used by model-seeded perception.

The model frame is fixed at the centre of the physical head's front plane:
``+x`` points image-right on a frontal view, ``+y`` points down the stand,
and ``+z`` points through the stand away from the observer.  QR coordinates
describe the detected QR *symbol* boundary, not its white panel or quiet zone.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)


STAND_MODEL_SCHEMA_VERSION = 1
STAND_MODEL_HASH_FIELD = "stand_model_sha256"
MEASUREMENT_STATUSES = frozenset(("measured", "provisional"))


@dataclass(frozen=True)
class ModelPoint3D:
    x_m: float
    y_m: float
    z_m: float = 0.0


@dataclass(frozen=True)
class StandModelProfile:
    schema_version: int
    profile_id: str
    environment: str
    measurement_status: str
    head_width_m: float
    head_height_m: float
    head_depth_m: float
    qr_symbol_width_m: float
    qr_symbol_height_m: float
    qr_center_x_m: float
    qr_center_y_m: float
    stem_width_m: float | None
    stem_visible_height_m: float | None
    tolerance_m: float
    source: str
    sha256: str

    @property
    def committable(self) -> bool:
        return self.measurement_status == "measured"

    @property
    def head_corners(self) -> tuple[ModelPoint3D, ...]:
        half_width = self.head_width_m / 2.0
        half_height = self.head_height_m / 2.0
        return (
            ModelPoint3D(-half_width, -half_height),
            ModelPoint3D(half_width, -half_height),
            ModelPoint3D(half_width, half_height),
            ModelPoint3D(-half_width, half_height),
        )

    @property
    def head_back_corners(self) -> tuple[ModelPoint3D, ...]:
        """Back-plane corners for diagnostic cuboid projection."""

        return tuple(
            ModelPoint3D(point.x_m, point.y_m, self.head_depth_m)
            for point in self.head_corners
        )

    @property
    def qr_corners(self) -> tuple[ModelPoint3D, ...]:
        half_width = self.qr_symbol_width_m / 2.0
        half_height = self.qr_symbol_height_m / 2.0
        center_x = self.qr_center_x_m
        center_y = self.qr_center_y_m
        return (
            ModelPoint3D(center_x - half_width, center_y - half_height),
            ModelPoint3D(center_x + half_width, center_y - half_height),
            ModelPoint3D(center_x + half_width, center_y + half_height),
            ModelPoint3D(center_x - half_width, center_y + half_height),
        )

    @property
    def semantic_landmarks(self) -> dict[str, ModelPoint3D]:
        head = self.head_corners
        head_back = self.head_back_corners
        points = {
            "head_top_left": head[0],
            "head_top_right": head[1],
            "head_bottom_right": head[2],
            "head_bottom_left": head[3],
            "head_back_top_left": head_back[0],
            "head_back_top_right": head_back[1],
            "head_back_bottom_right": head_back[2],
            "head_back_bottom_left": head_back[3],
        }
        for name, point in zip(
            ("qr_top_left", "qr_top_right", "qr_bottom_right", "qr_bottom_left"),
            self.qr_corners,
        ):
            points[name] = point
        if self.stem_width_m is not None:
            half_stem = self.stem_width_m / 2.0
            head_bottom = self.head_height_m / 2.0
            points["stem_junction_left"] = ModelPoint3D(-half_stem, head_bottom)
            points["stem_junction_right"] = ModelPoint3D(half_stem, head_bottom)
            if self.stem_visible_height_m is not None:
                stem_bottom = head_bottom + self.stem_visible_height_m
                points["stem_bottom_left"] = ModelPoint3D(-half_stem, stem_bottom)
                points["stem_bottom_right"] = ModelPoint3D(half_stem, stem_bottom)
        return points


def _finite_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _optional_positive(payload: Mapping[str, object], key: str) -> float | None:
    if payload.get(key) is None:
        return None
    value = _finite_number(payload, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be positive when present")
    return value


def stand_model_from_payload(
    payload: Mapping[str, object],
    *,
    sha256: str | None = None,
) -> StandModelProfile:
    if payload.get("schema_version") != STAND_MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"stand model schema_version must be {STAND_MODEL_SCHEMA_VERSION}"
        )
    profile_id = payload.get("profile_id")
    environment = payload.get("environment")
    measurement_status = payload.get("measurement_status")
    source = payload.get("source")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-empty string")
    if environment not in ("physical", "simulation"):
        raise ValueError("environment must be physical or simulation")
    if measurement_status not in MEASUREMENT_STATUSES:
        raise ValueError("measurement_status must be measured or provisional")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")

    positive_names = (
        "head_width_m",
        "head_height_m",
        "head_depth_m",
        "qr_symbol_width_m",
        "qr_symbol_height_m",
        "tolerance_m",
    )
    values = {name: _finite_number(payload, name) for name in positive_names}
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("stand dimensions and tolerance must be positive")
    if values["qr_symbol_width_m"] >= values["head_width_m"]:
        raise ValueError("QR symbol width must be smaller than the head width")
    if values["qr_symbol_height_m"] >= values["head_height_m"]:
        raise ValueError("QR symbol height must be smaller than the head height")

    center_x = _finite_number(payload, "qr_center_x_m")
    center_y = _finite_number(payload, "qr_center_y_m")
    if abs(center_x) + values["qr_symbol_width_m"] / 2.0 >= values["head_width_m"] / 2.0:
        raise ValueError("QR symbol exceeds the measured head width")
    if abs(center_y) + values["qr_symbol_height_m"] / 2.0 >= values["head_height_m"] / 2.0:
        raise ValueError("QR symbol exceeds the measured head height")

    unhashed = dict(payload)
    digest = payload_sha256(unhashed) if sha256 is None else sha256
    return StandModelProfile(
        schema_version=STAND_MODEL_SCHEMA_VERSION,
        profile_id=profile_id,
        environment=environment,
        measurement_status=measurement_status,
        head_width_m=values["head_width_m"],
        head_height_m=values["head_height_m"],
        head_depth_m=values["head_depth_m"],
        qr_symbol_width_m=values["qr_symbol_width_m"],
        qr_symbol_height_m=values["qr_symbol_height_m"],
        qr_center_x_m=center_x,
        qr_center_y_m=center_y,
        stem_width_m=_optional_positive(payload, "stem_width_m"),
        stem_visible_height_m=_optional_positive(payload, "stem_visible_height_m"),
        tolerance_m=values["tolerance_m"],
        source=source,
        sha256=digest,
    )


def load_stand_model(path: Path) -> StandModelProfile:
    payload = load_content_hashed_json(
        Path(path),
        hash_field=STAND_MODEL_HASH_FIELD,
    )
    return stand_model_from_payload(payload, sha256=payload_sha256(payload))


def write_stand_model(path: Path, profile: StandModelProfile) -> str:
    payload = {
        "schema_version": profile.schema_version,
        "profile_id": profile.profile_id,
        "environment": profile.environment,
        "measurement_status": profile.measurement_status,
        "head_width_m": profile.head_width_m,
        "head_height_m": profile.head_height_m,
        "head_depth_m": profile.head_depth_m,
        "qr_symbol_width_m": profile.qr_symbol_width_m,
        "qr_symbol_height_m": profile.qr_symbol_height_m,
        "qr_center_x_m": profile.qr_center_x_m,
        "qr_center_y_m": profile.qr_center_y_m,
        "stem_width_m": profile.stem_width_m,
        "stem_visible_height_m": profile.stem_visible_height_m,
        "tolerance_m": profile.tolerance_m,
        "source": profile.source,
    }
    return write_content_hashed_json(
        Path(path),
        payload,
        hash_field=STAND_MODEL_HASH_FIELD,
    )
