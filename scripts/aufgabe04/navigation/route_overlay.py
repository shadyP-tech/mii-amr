"""Stdlib SVG overlay rendering for Aufgabe 04 route validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.map_io import CELL_FREE, CELL_OCCUPIED, CELL_UNKNOWN, OccupancyGrid
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.station_approach import NavigationTarget
from scripts.aufgabe04.stations.models import Station, StationVisit


CELL_PIXELS = 8.0


@dataclass(frozen=True)
class SvgPoint:
    x: float
    y: float


@dataclass(frozen=True)
class RouteOverlayInput:
    grid: OccupancyGrid
    arena_bounds: ArenaBounds
    stations: Mapping[str, Station]
    visits: Sequence[StationVisit]
    targets: Sequence[NavigationTarget]
    results: Sequence[PlanRouteResult]
    metadata: Mapping[str, object]
    failed: bool = False


def world_to_svg_units(grid: OccupancyGrid, pose: Pose2D) -> SvgPoint:
    origin_x, origin_y, _origin_yaw = grid.metadata.origin
    x = (pose.x_m - origin_x) / grid.metadata.resolution
    y = grid.height - ((pose.y_m - origin_y) / grid.metadata.resolution)
    return SvgPoint(x, y)


def grid_to_svg_units(grid: OccupancyGrid, cell_x: int, cell_y: int) -> SvgPoint:
    return SvgPoint(float(cell_x), float(grid.height - 1 - cell_y))


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _attrs(**attributes: object) -> str:
    parts = []
    for key, value in attributes.items():
        if value is None:
            continue
        attr_name = key[:-1] if key.endswith("_") else key
        parts.append(f'{attr_name.replace("_", "-")}="{escape(str(value), quote=True)}"')
    return " ".join(parts)


def _text(x: float, y: float, value: object, *, css_class: str = "label") -> str:
    return (
        f"<text {_attrs(x=_fmt(x), y=_fmt(y), class_=css_class)}>"
        f"{escape(str(value))}</text>"
    )


def _cell_color(value: int) -> str:
    if value == CELL_FREE:
        return "#f8fafc"
    if value == CELL_OCCUPIED:
        return "#334155"
    if value == CELL_UNKNOWN:
        return "#cbd5e1"
    return "#e2e8f0"


def _render_map_cells(grid: OccupancyGrid) -> list[str]:
    elements = ['<g id="occupancy-map">']
    for grid_y, row in enumerate(grid.cells):
        svg_y = grid.height - 1 - grid_y
        run_start = 0
        run_value = row[0]
        for x in range(1, grid.width + 1):
            value = row[x] if x < grid.width else None
            if value == run_value:
                continue
            elements.append(
                f"<rect {_attrs(x=run_start, y=svg_y, width=x - run_start, height=1, fill=_cell_color(run_value))}/>"
            )
            if x < grid.width:
                run_start = x
                run_value = value
    elements.append("</g>")
    return elements


def _arena_polygon(grid: OccupancyGrid, arena: ArenaBounds) -> str:
    yaw = math.radians(arena.yaw_deg)
    half_length = arena.length_m / 2.0
    half_width = arena.width_m / 2.0
    local_corners = (
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    )
    points = []
    for local_x, local_y in local_corners:
        world = Pose2D(
            arena.center_x_m + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            arena.center_y_m + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
        )
        point = world_to_svg_units(grid, world)
        points.append(f"{_fmt(point.x)},{_fmt(point.y)}")
    return (
        '<polygon id="arena-bounds" '
        f'{_attrs(points=" ".join(points), fill="none", stroke="#2563eb", stroke_width=0.06, stroke_dasharray="0.2 0.16")}/>'
    )


def _render_stations(context: RouteOverlayInput) -> list[str]:
    elements = ['<g id="stations">']
    keepout_style = {"fill": "none", "stroke": "#dc2626", "stroke_width": 0.05}
    for station_id in sorted(context.stations):
        station = context.stations[station_id]
        center = world_to_svg_units(
            context.grid,
            Pose2D(station.pose.x_m, station.pose.y_m, station.pose.yaw_rad),
        )
        radius = station.keepout_radius_m / context.grid.metadata.resolution
        elements.append(
            f"<circle {_attrs(cx=_fmt(center.x), cy=_fmt(center.y), r=_fmt(radius), **keepout_style)}/>"
        )
        elements.append(
            f'<circle {_attrs(cx=_fmt(center.x), cy=_fmt(center.y), r=0.12, fill="#dc2626")}/>'
        )
        elements.append(_text(center.x + 0.18, center.y - 0.18, station.station_id, css_class="station-label"))
    elements.append("</g>")
    return elements


def _render_targets(context: RouteOverlayInput) -> list[str]:
    elements = ['<g id="approach-targets">']
    for target in context.targets:
        point = world_to_svg_units(context.grid, target.pose)
        elements.append(
            f'<circle {_attrs(cx=_fmt(point.x), cy=_fmt(point.y), r=0.11, fill="#16a34a", stroke="#052e16", stroke_width=0.04)}/>'
        )
        elements.append(_text(point.x + 0.16, point.y + 0.28, f"{target.station_id} approach"))
    elements.append("</g>")
    return elements


def _render_routes(context: RouteOverlayInput) -> list[str]:
    colors = ("#f97316", "#7c3aed", "#0891b2", "#ca8a04", "#be123c")
    elements = ['<g id="planned-routes">']
    for leg_index, result in enumerate(context.results):
        if result.route is None:
            continue
        points = [
            f"{_fmt(world_to_svg_units(context.grid, point.pose).x)},{_fmt(world_to_svg_units(context.grid, point.pose).y)}"
            for point in result.route.points
        ]
        color = colors[leg_index % len(colors)]
        elements.append(
            f'<polyline {_attrs(points=" ".join(points), fill="none", stroke=color, stroke_width=0.1, stroke_linejoin="round", stroke_linecap="round")}/>'
        )
        start = world_to_svg_units(context.grid, result.route.requested_start)
        goal = world_to_svg_units(context.grid, result.route.requested_goal)
        elements.append(
            f'<circle {_attrs(cx=_fmt(start.x), cy=_fmt(start.y), r=0.1, fill="#ffffff", stroke=color, stroke_width=0.05)}/>'
        )
        elements.append(
            f'<rect {_attrs(x=_fmt(goal.x - 0.1), y=_fmt(goal.y - 0.1), width=0.2, height=0.2, fill="#ffffff", stroke=color, stroke_width=0.05)}/>'
        )
    elements.append("</g>")
    return elements


def _render_metadata(context: RouteOverlayInput) -> list[str]:
    metadata = context.metadata
    origin = metadata.get("origin", context.grid.metadata.origin)
    lines = [
        "frame_id: map",
        f"map_yaml: {metadata.get('map_yaml', metadata.get('map', context.grid.metadata.yaml_path))}",
        f"resolution: {context.grid.metadata.resolution} m/cell",
        f"origin: {origin}",
    ]
    if "map_image_sha256" in metadata:
        lines.append(f"map_image_sha256: {str(metadata['map_image_sha256'])[:16]}...")
    lines.append("visual check only; not live TF/AMCL/Nav2/sensor/cmd_vel validation")
    elements = ['<g id="coordinate-frame-label">']
    for index, line in enumerate(lines):
        elements.append(_text(0.4, 0.75 + index * 0.45, line, css_class="meta-label"))
    elements.append("</g>")
    return elements


def render_route_overlay_svg(context: RouteOverlayInput) -> str:
    width = context.grid.width
    height = context.grid.height
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'{_attrs(viewBox=f"0 0 {width} {height}", width=_fmt(width * CELL_PIXELS), height=_fmt(height * CELL_PIXELS), role="img")}>'
        ),
        "<style>"
        ".label{font:0.32px sans-serif;fill:#0f172a;paint-order:stroke;stroke:#fff;stroke-width:0.08px}"
        ".station-label{font:0.42px sans-serif;font-weight:700;fill:#7f1d1d;paint-order:stroke;stroke:#fff;stroke-width:0.1px}"
        ".meta-label{font:0.28px monospace;fill:#0f172a;paint-order:stroke;stroke:#fff;stroke-width:0.08px}"
        "</style>",
    ]
    elements.extend(_render_map_cells(context.grid))
    elements.append(_arena_polygon(context.grid, context.arena_bounds))
    elements.extend(_render_stations(context))
    elements.extend(_render_targets(context))
    elements.extend(_render_routes(context))
    elements.extend(_render_metadata(context))
    if context.failed:
        elements.append(
            '<g id="failed-overlay">'
            f'<rect {_attrs(x=0, y=height / 2 - 0.75, width=width, height=1.5, fill="#fee2e2", opacity=0.85)}/>'
            f'{_text(width / 2 - 2.4, height / 2 + 0.16, "FAILED/INCOMPLETE", css_class="station-label")}'
            "</g>"
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"
