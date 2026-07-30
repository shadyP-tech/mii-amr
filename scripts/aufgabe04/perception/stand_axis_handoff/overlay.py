"""OpenCV overlay for the observe-only calibrated handoff viewer."""

from __future__ import annotations

import math

from scripts.aufgabe04.perception.debug.text_overlay import OverlayTextCursor
from scripts.aufgabe04.perception.stand_axis_handoff.models import (
    AxisHandoffDecision,
)


def _degrees(value: float | None) -> str:
    return "n/a" if value is None else f"{math.degrees(value):.1f}deg"


def annotate_axis_handoff(
    cv2,
    frame,
    decision: AxisHandoffDecision,
    *,
    text_cursor: OverlayTextCursor | None = None,
) -> OverlayTextCursor:
    accepted_color = (60, 220, 60)
    rejected_color = (60, 60, 240)
    color = accepted_color if decision.accepted else rejected_color
    lines = [
        f"handoff={decision.status} observe_only=true",
        (
            f"lidar={_degrees(decision.lidar.angle_rad)} "
            f"pts={decision.lidar.sample_count} "
            f"lin={decision.lidar.linearity if decision.lidar.linearity is not None else float('nan'):.3f}"
        ),
        (
            f"camera={_degrees(decision.camera.angle_rad)} "
            f"n={decision.camera.sample_count}"
        ),
        f"axis_delta={_degrees(decision.axial_difference_rad)}",
    ]
    cursor = text_cursor or OverlayTextCursor()
    for line in lines:
        cursor.draw(
            cv2,
            frame,
            line,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.52,
            color=color,
            thickness=2,
        )

    if decision.lidar.center_xy_m is None:
        return cursor
    height, width = frame.shape[:2]
    origin = (width - 105, 105)
    scale = 80.0
    cv2.circle(frame, origin, 4, (255, 255, 255), -1)
    center_x, center_y = decision.lidar.center_xy_m
    center_px = (
        int(round(origin[0] + scale * center_x)),
        int(round(origin[1] - scale * center_y)),
    )
    cv2.circle(frame, center_px, 5, (0, 215, 255), -1)
    for angle, axis_color in (
        (decision.lidar.angle_rad, (0, 215, 255)),
        (decision.camera.angle_rad, (255, 255, 0)),
    ):
        if angle is None:
            continue
        dx = int(round(32.0 * math.cos(angle)))
        dy = int(round(32.0 * math.sin(angle)))
        cv2.line(
            frame,
            (center_px[0] - dx, center_px[1] + dy),
            (center_px[0] + dx, center_px[1] - dy),
            axis_color,
            2,
        )
    if decision.approach_pose is not None:
        approach_px = (
            int(round(origin[0] + scale * decision.approach_pose.x_m)),
            int(round(origin[1] - scale * decision.approach_pose.y_m)),
        )
        cv2.circle(frame, approach_px, 5, color, 2)
        cv2.line(frame, approach_px, center_px, color, 1)
    return cursor
