"""Bind decoded identity to its own image ray and the mapped LiDAR target.

An ROI is a search proposal, never identity authority. QR text can enter the
temporal latch only with that symbol's quadrilateral and a unique cluster in
the existing candidate range/cone. Registered crops retain their certified
bearing correction bound; missing geometry remains diagnostic-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
    associate_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_handoff import rectified_pixel_bearing_in_scan
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners


@dataclass(frozen=True)
class QrTargetBinding:
    accepted: bool
    reason: str
    qr_texts_for_evidence: tuple[str, ...] = ()
    symbol_count: int = 0
    camera_bearing_rad: float | None = None
    association: dict | None = None

    def metadata(self) -> dict:
        return {**asdict(self), "motion_authorized": False,
                "completion_authorized": False}


def bind_qr_observations_to_target(
    observations, *, roi, intrinsics, scan_from_camera, scan,
    map_bearing_rad: float, cone_half_angle_rad: float,
    accepted_range_m: tuple[float, float], now_sec: float,
    max_scan_age_sec: float, min_cluster_sample_count: int,
    camera_registration_accepted: bool,
    max_camera_map_bearing_delta_rad: float,
) -> QrTargetBinding:
    observations = tuple(observations or ())
    count = len(observations)
    if count != 1:
        return QrTargetBinding(False, "multiple_qr_symbols" if count > 1
                               else "no_decoded_qr_geometry", symbol_count=count)
    observation = observations[0]
    corners = validated_qr_corners(
        observation.corners, image_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0),
    )
    if corners is None:
        return QrTargetBinding(False, "decoded_qr_geometry_unavailable", symbol_count=1)
    try:
        bearing = rectified_pixel_bearing_in_scan(
            u_px=sum(p[0] for p in corners) / 4 + roi.x0,
            v_px=sum(p[1] for p in corners) / 4 + roi.y0,
            fx_px=intrinsics.fx_px, fy_px=intrinsics.fy_px,
            cx_px=intrinsics.cx_px, cy_px=intrinsics.cy_px,
            scan_from_camera=scan_from_camera,
        )
    except (TypeError, ValueError) as exc:
        return QrTargetBinding(False, f"decoded_qr_bearing_unavailable: {exc}", symbol_count=1)
    common = dict(
        map_bearing_rad=map_bearing_rad, observed_camera_bearing_rad=bearing,
        cone_half_angle_rad=cone_half_angle_rad, accepted_range_m=accepted_range_m,
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        min_cluster_sample_count=min_cluster_sample_count,
    )
    if camera_registration_accepted:
        association = associate_camera_registered_candidate_lidar_target(
            scan, **common,
            max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        )
        cluster = association.search_association
    else:
        association = associate_candidate_lidar_target(scan, **common)
        cluster = association
    reason = association.rejection_reason
    if association.associated and cluster.eligible_cluster_count != 1:
        reason = "ambiguous_qr_target_clusters"
    if (not reason and not camera_registration_accepted
            and cluster.selected_cluster_bearing_delta_from_camera_rad > cone_half_angle_rad):
        reason = "qr_bearing_outside_target_cluster_cone"
    accepted = association.associated and not reason
    return QrTargetBinding(
        accepted, "decoded_qr_target_associated" if accepted else reason,
        (observation.text,) if accepted else (), 1, bearing, asdict(association),
    )
