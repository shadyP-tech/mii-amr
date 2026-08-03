"""Diagnostic-only rendering for metric stand-model evidence."""

from __future__ import annotations

import math

from scripts.aufgabe04.perception.debug.text_overlay import OverlayTextCursor
from scripts.aufgabe04.perception.stand_axis.model_profile import StandModelProfile
from scripts.aufgabe04.perception.stand_axis.models import (
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)


def draw_dashed_segment(cv2, frame, start, end, color, thickness: int) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    dash_count = max(1, int(math.ceil(length / 8.0)))
    for index in range(0, dash_count, 2):
        first = index / dash_count
        second = min(1.0, (index + 1) / dash_count)
        cv2.line(
            frame,
            (round(start[0] + first * dx), round(start[1] + first * dy)),
            (round(start[0] + second * dx), round(start[1] + second * dy)),
            color,
            thickness,
        )


def draw_dashed_polygon(cv2, frame, points, color, thickness: int) -> None:
    """Render a model prediction without making it look measured."""

    for start, end in zip(points, points[1:] + points[:1]):
        draw_dashed_segment(cv2, frame, start, end, color, thickness)


def annotate_model_prediction(
    cv2,
    frame,
    corners,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    if corners is None:
        return
    points = [
        (round(point.u_px + x_offset), round(point.v_px + y_offset))
        for point in corners
    ]
    draw_dashed_polygon(cv2, frame, points, (255, 0, 255), 1)


def annotate_projected_model_landmarks(
    cv2,
    frame,
    landmarks,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    """Draw non-measured head-depth and stem geometry as dashed predictions."""

    if not landmarks:
        return

    def pixel(name):
        point = landmarks.get(name)
        if point is None:
            return None
        return (
            round(point.u_px + x_offset),
            round(point.v_px + y_offset),
        )

    front_names = (
        "head_top_left",
        "head_top_right",
        "head_bottom_right",
        "head_bottom_left",
    )
    back_names = (
        "head_back_top_left",
        "head_back_top_right",
        "head_back_bottom_right",
        "head_back_bottom_left",
    )
    front = [pixel(name) for name in front_names]
    back = [pixel(name) for name in back_names]
    if all(point is not None for point in back):
        draw_dashed_polygon(cv2, frame, back, (180, 0, 180), 1)
    if all(point is not None for point in (*front, *back)):
        for front_point, back_point in zip(front, back):
            draw_dashed_segment(
                cv2,
                frame,
                front_point,
                back_point,
                (180, 0, 180),
                1,
            )

    stem_left = (pixel("stem_junction_left"), pixel("stem_bottom_left"))
    stem_right = (pixel("stem_junction_right"), pixel("stem_bottom_right"))
    for start, end in (stem_left, stem_right):
        if start is not None and end is not None:
            draw_dashed_segment(
                cv2,
                frame,
                start,
                end,
                (255, 0, 255),
                1,
            )
    if stem_left[1] is not None and stem_right[1] is not None:
        draw_dashed_segment(
            cv2,
            frame,
            stem_left[1],
            stem_right[1],
            (255, 0, 255),
            1,
        )


def annotate_metric_model_status(
    cv2,
    frame,
    *,
    profile: StandModelProfile | None,
    inputs_ready: bool,
    estimate: StandAxisImageEstimate | None,
    artifacts: StandAxisEdgeDebugArtifacts | None,
    text_cursor: OverlayTextCursor,
) -> OverlayTextCursor:
    """Keep model acquisition failures visible when edge fallback wins."""

    if profile is None:
        return text_cursor
    evidence_state = (
        "inputs_unavailable"
        if not inputs_ready or estimate is None
        else estimate.evidence_state
    )
    reason = (
        "metric_inputs_unavailable"
        if not inputs_ready or estimate is None
        else estimate.reason
    )
    qr_detected = bool(artifacts is not None and artifacts.qr_detected)
    seed_source = (
        "none"
        if artifacts is None or artifacts.pose_seed_source is None
        else artifacts.pose_seed_source
    )
    scale_text = "n/a"
    if artifacts is not None and artifacts.qr_detection_scale is not None:
        scale_text = f"{artifacts.qr_detection_scale:g}x"
    line1 = (
        f"model={evidence_state} reason={reason} "
        f"qr={str(qr_detected).lower()} scale={scale_text} seed={seed_source}"
    )
    details = [
        f"profile={profile.measurement_status}",
        f"committable={str(profile.committable).lower()}",
    ]
    if estimate is not None and estimate.pose_reprojection_rmse_px is not None:
        details.append(f"pnp_rmse={estimate.pose_reprojection_rmse_px:.2f}px")
    if artifacts is not None and artifacts.refinement_support_mean is not None:
        details.append(f"support={artifacts.refinement_support_mean:.2f}")
    if (
        artifacts is not None
        and artifacts.model_corridor_half_width_px is not None
    ):
        details.append(
            f"corridor={artifacts.model_corridor_half_width_px:.1f}px"
        )
    if artifacts is not None and artifacts.model_pose_fit_source is not None:
        details.append(f"fit={artifacts.model_pose_fit_source}")
    color = (
        (0, 255, 0)
        if evidence_state == "fresh_refined"
        else (255, 0, 255)
    )
    for text in (line1, " ".join(details)):
        text_cursor.draw(
            cv2,
            frame,
            text,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.45,
            color=color,
            thickness=1,
        )
    return text_cursor
