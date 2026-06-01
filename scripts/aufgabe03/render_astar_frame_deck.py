#!/usr/bin/env python3
"""
Render a GIF-oriented frame deck explaining the A* path generation used for
Aufgabe 03.

The visuals mirror the relevant choices in scripts/aufgabe03/map_path_planner.py:
an occupancy grid, inflated blocked cells, snapping to traversable cells,
8-neighbor expansion without diagonal corner cutting, Euclidean h, f = g + h,
path reconstruction, and waypoint simplification at direction changes.

Use the bundled Codex Python runtime because it includes Pillow:

    /Users/stephpark/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
        scripts/aufgabe03/render_astar_frame_deck.py
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 1920
CANVAS_H = 1080
GRID_W = 24
GRID_H = 15
CELL = 50
GRID_LEFT = 112
GRID_TOP = 190
GRID_RIGHT = GRID_LEFT + GRID_W * CELL
GRID_BOTTOM = GRID_TOP + GRID_H * CELL
SIDE_LEFT = 1390
SIDE_TOP = 195
SIDE_W = 390

START = (2, 3)
GOAL = (21, 11)
REQUESTED_START = (2.18, 3.16)
REQUESTED_GOAL = (20.72, 11.2)

COLORS = {
    "background": (247, 249, 252),
    "surface": (255, 255, 255),
    "ink": (24, 31, 42),
    "muted": (91, 103, 123),
    "grid": (211, 218, 229),
    "free": (252, 253, 255),
    "blocked": (32, 41, 55),
    "inflated": (255, 218, 225),
    "open": (250, 204, 21),
    "closed": (147, 197, 253),
    "neighbor": (254, 240, 138),
    "current": (249, 115, 22),
    "path": (37, 99, 235),
    "old_path": (107, 114, 128),
    "waypoint": (6, 182, 212),
    "start": (22, 163, 74),
    "goal": (220, 38, 38),
    "overlay": (124, 58, 237),
    "overlay_soft": (221, 214, 254),
    "reject": (239, 68, 68),
}


@dataclass(frozen=True)
class AStarStep:
    index: int
    current: tuple[int, int]
    open_cells: frozenset[tuple[int, int]]
    closed_cells: frozenset[tuple[int, int]]
    neighbors: frozenset[tuple[int, int]]
    rejected: frozenset[tuple[int, int]]
    came_from: dict[tuple[int, int], tuple[int, int]]
    g_score: dict[tuple[int, int], float]
    f_score: float


@dataclass(frozen=True)
class FrameSpec:
    name: str
    phase: str
    caption: str
    duration_ms: int
    show_inflation: bool = True
    show_snap: bool = False
    show_heuristic: bool = False
    step: AStarStep | None = None
    path: tuple[tuple[int, int], ...] = ()
    path_prefix: tuple[tuple[int, int], ...] = ()
    waypoints: tuple[tuple[int, int], ...] = ()
    parent_arrows: bool = False
    current: tuple[int, int] | None = None
    neighbors: tuple[tuple[int, int], ...] = ()
    rejected: tuple[tuple[int, int], ...] = ()
    old_path: tuple[tuple[int, int], ...] = ()
    overlay_obstacles: frozenset[tuple[int, int]] = frozenset()
    overlay_inflated: frozenset[tuple[int, int]] = frozenset()
    replan_path: tuple[tuple[int, int], ...] = ()
    side_mode: str = "formula"


def build_obstacles() -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    for y in list(range(2, 6)) + list(range(9, 13)):
        obstacles.add((9, y))
    for x in range(13, 21):
        if x not in (16, 17):
            obstacles.add((x, 7))
    for y in range(1, 5):
        obstacles.add((17, y))
    return obstacles


def inflate(cells: set[tuple[int, int]], radius_cells: int = 1) -> set[tuple[int, int]]:
    inflated = set(cells)
    for x, y in cells:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                cell = (x + dx, y + dy)
                if in_bounds(cell):
                    inflated.add(cell)
    return inflated


def in_bounds(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def traversable(cell: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    return in_bounds(cell) and cell not in blocked


def movement_cost(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def heuristic(cell: tuple[int, int], goal: tuple[int, int] = GOAL) -> float:
    return movement_cost(cell, goal)


def neighbors_no_corner_cutting(
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    x, y = cell
    accepted: list[tuple[int, int]] = []
    rejected: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = (x + dx, y + dy)
            if not traversable(neighbor, blocked):
                if in_bounds(neighbor):
                    rejected.append(neighbor)
                continue
            if dx != 0 and dy != 0:
                side_a = (x + dx, y)
                side_b = (x, y + dy)
                if not traversable(side_a, blocked) or not traversable(side_b, blocked):
                    rejected.append(neighbor)
                    continue
            accepted.append(neighbor)
    return accepted, rejected


def run_astar(
    blocked: set[tuple[int, int]],
    start: tuple[int, int] = START,
    goal: tuple[int, int] = GOAL,
) -> tuple[list[AStarStep], list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    queue: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(queue, (0.0, 0, start))
    open_cells = {start}
    closed_cells: set[tuple[int, int]] = set()
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    tie_breaker = 0
    steps: list[AStarStep] = []

    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        if current in closed_cells:
            continue
        open_cells.discard(current)
        closed_cells.add(current)

        accepted, rejected = neighbors_no_corner_cutting(current, blocked)
        if current != goal:
            for neighbor in accepted:
                tentative_g = g_score[current] + movement_cost(current, neighbor)
                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                tie_breaker += 1
                open_cells.add(neighbor)
                heapq.heappush(
                    queue,
                    (
                        tentative_g + heuristic(neighbor, goal),
                        tie_breaker,
                        neighbor,
                    ),
                )

        steps.append(
            AStarStep(
                index=len(steps),
                current=current,
                open_cells=frozenset(open_cells),
                closed_cells=frozenset(closed_cells),
                neighbors=frozenset(accepted),
                rejected=frozenset(rejected),
                came_from=dict(came_from),
                g_score=dict(g_score),
                f_score=g_score[current] + heuristic(current, goal),
            )
        )
        if current == goal:
            return steps, reconstruct_path(came_from, start, goal), came_from

    raise RuntimeError("No A* path found for the frame-deck map")


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def direction_between(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    gcd = math.gcd(abs(dx), abs(dy))
    if gcd == 0:
        return 0, 0
    return dx // gcd, dy // gcd


def simplify_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return list(path)
    waypoints = [path[0]]
    previous_direction = direction_between(path[0], path[1])
    for index in range(2, len(path)):
        current_direction = direction_between(path[index - 1], path[index])
        if current_direction != previous_direction:
            waypoints.append(path[index - 1])
            previous_direction = current_direction
    waypoints.append(path[-1])
    return waypoints


def sample_search_steps(steps: list[AStarStep]) -> list[AStarStep]:
    indexes: list[int] = []
    indexes.extend(range(min(9, len(steps))))
    indexes.extend(range(9, len(steps), 3))
    indexes.append(len(steps) - 1)
    seen: set[int] = set()
    sampled: list[AStarStep] = []
    for index in indexes:
        if index in seen or not 0 <= index < len(steps):
            continue
        sampled.append(steps[index])
        seen.add(index)
    return sampled


def font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = "/System/Library/Fonts/SFNSMono.ttf" if mono else "/System/Library/Fonts/SFNS.ttf"
    fallback_paths = [
        font_path,
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in fallback_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def cell_rect(cell: tuple[int, int], pad: int = 2) -> tuple[int, int, int, int]:
    x, y = cell
    left = GRID_LEFT + x * CELL
    top = GRID_TOP + (GRID_H - 1 - y) * CELL
    return left + pad, top + pad, left + CELL - pad, top + CELL - pad


def cell_center(cell: tuple[int, int]) -> tuple[float, float]:
    left, top, right, bottom = cell_rect(cell, pad=0)
    return (left + right) / 2.0, (top + bottom) / 2.0


def blend(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(a[i] + (b[i] - a[i]) * amount) for i in range(3))


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int = 34,
    fill: tuple[int, int, int] = COLORS["ink"],
    mono: bool = False,
) -> None:
    draw.text(xy, text, font=font(size, mono=mono), fill=fill)


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int] = COLORS["ink"],
    size: int = 26,
) -> None:
    text_font = font(size)
    tw, th = text_size(draw, text, text_font)
    x, y = xy
    draw.rounded_rectangle(
        (x, y, x + tw + 34, y + th + 22),
        radius=8,
        fill=fill,
    )
    draw.text((x + 17, y + 9), text, font=text_font, fill=text_fill)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int, int] | tuple[int, int, int],
    width: int = 5,
    head: int = 14,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = []
    for offset in (math.pi * 0.82, -math.pi * 0.82):
        points.append(
            (
                end[0] + math.cos(angle + offset) * head,
                end[1] + math.sin(angle + offset) * head,
            )
        )
    draw.polygon((end, points[0], points[1]), fill=fill)


def draw_path(
    draw: ImageDraw.ImageDraw,
    path: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
    alpha: int = 255,
) -> None:
    if len(path) < 2:
        return
    points = [cell_center(cell) for cell in path]
    draw.line(points, fill=(*color, alpha), width=width, joint="curve")
    for point in points:
        r = max(4, width // 3)
        draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=(*color, alpha))


def draw_requested_snap(draw: ImageDraw.ImageDraw) -> None:
    for requested, snapped, color, label in (
        (REQUESTED_START, START, COLORS["start"], "start"),
        (REQUESTED_GOAL, GOAL, COLORS["goal"], "goal"),
    ):
        req_x = GRID_LEFT + requested[0] * CELL
        req_y = GRID_TOP + (GRID_H - requested[1]) * CELL
        snap_x, snap_y = cell_center(snapped)
        draw.ellipse((req_x - 9, req_y - 9, req_x + 9, req_y + 9), outline=color, width=4)
        dash_count = 7
        for i in range(dash_count):
            t1 = i / dash_count
            t2 = (i + 0.55) / dash_count
            x1 = req_x + (snap_x - req_x) * t1
            y1 = req_y + (snap_y - req_y) * t1
            x2 = req_x + (snap_x - req_x) * t2
            y2 = req_y + (snap_y - req_y) * t2
            draw.line((x1, y1, x2, y2), fill=(*color, 210), width=4)
        draw_label(draw, (int(req_x) + 14, int(req_y) - 18), label, size=22, fill=color)


def draw_grid(
    image: Image.Image,
    spec: FrameSpec,
    obstacles: set[tuple[int, int]],
    inflated: set[tuple[int, int]],
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")

    open_cells: set[tuple[int, int]] = set()
    closed_cells: set[tuple[int, int]] = set()
    neighbors: set[tuple[int, int]] = set(spec.neighbors)
    rejected: set[tuple[int, int]] = set(spec.rejected)
    current = spec.current

    if spec.step is not None:
        open_cells = set(spec.step.open_cells)
        closed_cells = set(spec.step.closed_cells)
        neighbors = set(spec.step.neighbors)
        rejected = set(spec.step.rejected)
        current = spec.step.current

    for y in range(GRID_H):
        for x in range(GRID_W):
            cell = (x, y)
            fill = COLORS["free"]
            if spec.show_heuristic and cell not in inflated:
                distance = heuristic(cell)
                amount = min(1.0, distance / 20.0)
                fill = blend((228, 248, 232), (234, 242, 255), amount)
            if spec.show_inflation and cell in inflated and cell not in obstacles:
                fill = COLORS["inflated"]
            if cell in closed_cells:
                fill = COLORS["closed"]
            if cell in open_cells:
                fill = COLORS["open"]
            if cell in neighbors and cell not in inflated:
                fill = COLORS["neighbor"]
            if cell in spec.overlay_inflated and cell not in spec.overlay_obstacles:
                fill = COLORS["overlay_soft"]
            if cell in obstacles:
                fill = COLORS["blocked"]
            if cell in spec.overlay_obstacles:
                fill = COLORS["overlay"]
            draw.rounded_rectangle(cell_rect(cell), radius=5, fill=fill)
            draw.rectangle(cell_rect(cell, pad=0), outline=(*COLORS["grid"], 170), width=1)

    if spec.parent_arrows and spec.step is not None:
        for child, parent in spec.step.came_from.items():
            if child in closed_cells or child in open_cells:
                start = cell_center(parent)
                end = cell_center(child)
                sx = start[0] + (end[0] - start[0]) * 0.28
                sy = start[1] + (end[1] - start[1]) * 0.28
                ex = start[0] + (end[0] - start[0]) * 0.68
                ey = start[1] + (end[1] - start[1]) * 0.68
                draw_arrow(draw, (sx, sy), (ex, ey), fill=(70, 90, 120, 95), width=2, head=6)

    if spec.old_path:
        draw_path(draw, spec.old_path, COLORS["old_path"], width=14, alpha=145)
    if spec.path:
        draw_path(draw, spec.path, COLORS["path"], width=16, alpha=245)
    if spec.path_prefix:
        draw_path(draw, spec.path_prefix, COLORS["path"], width=18, alpha=255)
    if spec.replan_path:
        draw_path(draw, spec.replan_path, COLORS["path"], width=18, alpha=255)

    for cell in rejected:
        cx, cy = cell_center(cell)
        draw.line((cx - 14, cy - 14, cx + 14, cy + 14), fill=COLORS["reject"], width=5)
        draw.line((cx - 14, cy + 14, cx + 14, cy - 14), fill=COLORS["reject"], width=5)

    if current is not None:
        draw.rounded_rectangle(cell_rect(current, pad=0), radius=7, outline=COLORS["current"], width=7)
        cx, cy = cell_center(current)
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), fill=COLORS["current"])

    for cell in spec.waypoints:
        cx, cy = cell_center(cell)
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=COLORS["waypoint"], outline=COLORS["surface"], width=4)

    for cell, color, letter in ((START, COLORS["start"], "S"), (GOAL, COLORS["goal"], "G")):
        cx, cy = cell_center(cell)
        draw.ellipse((cx - 21, cy - 21, cx + 21, cy + 21), fill=color, outline=COLORS["surface"], width=5)
        label_font = font(25, mono=True)
        tw, th = text_size(draw, letter, label_font)
        draw.text((cx - tw / 2, cy - th / 2 - 2), letter, font=label_font, fill=COLORS["surface"])

    if spec.show_snap:
        draw_requested_snap(draw)

    draw.rounded_rectangle((GRID_LEFT - 1, GRID_TOP - 1, GRID_RIGHT + 1, GRID_BOTTOM + 1), radius=8, outline=(94, 109, 130, 220), width=3)


def draw_header(draw: ImageDraw.ImageDraw, spec: FrameSpec) -> None:
    draw_label(draw, (112, 58), "A* path generation", size=52)
    draw_label(draw, (114, 122), spec.phase, size=34, fill=COLORS["muted"])
    if spec.caption:
        draw_label(draw, (470, 124), spec.caption, size=28, fill=COLORS["muted"])
    draw_pill(draw, (1548, 65), "f = g + h", fill=(229, 236, 255), text_fill=COLORS["path"], size=32)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    items = [
        ("open", COLORS["open"]),
        ("closed", COLORS["closed"]),
        ("current", COLORS["current"]),
        ("blocked", COLORS["blocked"]),
        ("inflated", COLORS["inflated"]),
        ("path", COLORS["path"]),
        ("waypoint", COLORS["waypoint"]),
    ]
    x = 112
    y = 966
    for label, color in items:
        draw.rounded_rectangle((x, y, x + 34, y + 34), radius=7, fill=color)
        draw_label(draw, (x + 45, y + 1), label, size=24, fill=COLORS["muted"])
        x += 165 if label not in {"current", "inflated", "waypoint"} else 190


def draw_formula_panel(draw: ImageDraw.ImageDraw, spec: FrameSpec) -> None:
    draw.rounded_rectangle((SIDE_LEFT, SIDE_TOP, SIDE_LEFT + SIDE_W, SIDE_TOP + 705), radius=8, fill=(255, 255, 255), outline=(218, 226, 237), width=2)

    if spec.side_mode == "map":
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 36), "Occupancy grid", size=34)
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 92), "free cells", size=28, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 140), "blocked cells", size=28, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 188), "start / goal", size=28, fill=COLORS["muted"])
        draw_mini_rule(draw, SIDE_LEFT + 52, SIDE_TOP + 300)
        return

    if spec.side_mode == "snap":
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 36), "Snap", size=38)
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 94), "requested", size=28, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 142), "-> nearest free", size=28, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 190), "within radius", size=28, fill=COLORS["muted"])
        draw_mini_snap(draw, SIDE_LEFT + 64, SIDE_TOP + 295)
        return

    if spec.side_mode == "heuristic":
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 36), "Priority", size=38)
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 108), "g: travelled", size=28, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 156), "h: Euclidean", size=28, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 204), "f: next node", size=28, fill=COLORS["muted"])
        draw_mini_costs(draw, SIDE_LEFT + 55, SIDE_TOP + 310)
        return

    if spec.side_mode == "waypoints":
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 36), "Simplify", size=38)
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 100), "keep turns", size=30, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 150), "drop straight", size=30, fill=COLORS["muted"])
        draw_mini_waypoints(draw, SIDE_LEFT + 50, SIDE_TOP + 300)
        return

    if spec.side_mode == "replan":
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 36), "Replan", size=38)
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 100), "new obstacle", size=30, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 150), "same A*", size=30, fill=COLORS["muted"])
        draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 200), "new path", size=30, fill=COLORS["muted"])
        draw_mini_replan(draw, SIDE_LEFT + 52, SIDE_TOP + 300)
        return

    step = spec.step
    current = spec.current or (step.current if step else START)
    g = (step.g_score.get(current, 0.0) if step else 0.0)
    h = heuristic(current)
    f_value = g + h

    draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 36), "Current cell", size=34)
    draw_label(draw, (SIDE_LEFT + 34, SIDE_TOP + 92), f"({current[0]}, {current[1]})", size=34, fill=COLORS["current"], mono=True)
    draw_score_bar(draw, SIDE_LEFT + 36, SIDE_TOP + 180, "g", g, 30.0, COLORS["closed"])
    draw_score_bar(draw, SIDE_LEFT + 36, SIDE_TOP + 285, "h", h, 30.0, COLORS["goal"])
    draw_score_bar(draw, SIDE_LEFT + 36, SIDE_TOP + 390, "f", f_value, 30.0, COLORS["path"])
    draw_label(draw, (SIDE_LEFT + 36, SIDE_TOP + 528), "lowest f leaves open set", size=25, fill=COLORS["muted"])
    draw_mini_rule(draw, SIDE_LEFT + 60, SIDE_TOP + 585, scale=0.55)


def draw_score_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: float,
    max_value: float,
    color: tuple[int, int, int],
) -> None:
    draw_label(draw, (x, y - 4), label, size=32, fill=color, mono=True)
    bar_x = x + 58
    bar_y = y
    bar_w = 260
    bar_h = 30
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=8, fill=(232, 238, 247))
    fill_w = int(bar_w * min(1.0, value / max_value))
    draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=8, fill=color)
    draw_label(draw, (bar_x, bar_y + 42), f"{value:0.1f}", size=25, fill=COLORS["muted"], mono=True)


def draw_mini_rule(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    size = int(48 * scale)
    center = (x + size * 1.5, y + size * 1.5)
    for row in range(3):
        for col in range(3):
            left = x + col * size
            top = y + row * size
            fill = (248, 250, 252)
            if (col, row) in {(1, 1)}:
                fill = COLORS["current"]
            if (col, row) in {(2, 1), (1, 0)}:
                fill = COLORS["blocked"]
            draw.rounded_rectangle((left + 2, top + 2, left + size - 2, top + size - 2), radius=6, fill=fill, outline=COLORS["grid"])
    end = (x + size * 2.5, y + size * 0.5)
    draw.line((center[0], center[1], end[0], end[1]), fill=COLORS["reject"], width=max(3, int(5 * scale)))
    draw.line((end[0] - 12 * scale, end[1] - 12 * scale, end[0] + 12 * scale, end[1] + 12 * scale), fill=COLORS["reject"], width=max(3, int(5 * scale)))
    draw.line((end[0] - 12 * scale, end[1] + 12 * scale, end[0] + 12 * scale, end[1] - 12 * scale), fill=COLORS["reject"], width=max(3, int(5 * scale)))
    draw_label(draw, (x, y + size * 3 + 18), "no corner cut", size=max(18, int(27 * scale)), fill=COLORS["muted"])


def draw_mini_snap(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.ellipse((x + 72, y + 56, x + 116, y + 100), outline=COLORS["start"], width=5)
    draw.rounded_rectangle((x + 180, y + 52, x + 230, y + 102), radius=8, fill=COLORS["start"])
    draw_arrow(draw, (x + 120, y + 78), (x + 176, y + 78), fill=COLORS["start"], width=5)
    draw_label(draw, (x + 40, y + 145), "world -> grid", size=26, fill=COLORS["muted"])


def draw_mini_costs(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rounded_rectangle((x, y, x + 210, y + 120), radius=8, fill=(248, 250, 252), outline=COLORS["grid"])
    center = (x + 80, y + 62)
    draw.ellipse((center[0] - 13, center[1] - 13, center[0] + 13, center[1] + 13), fill=COLORS["current"])
    draw_arrow(draw, center, (x + 155, y + 62), fill=COLORS["closed"], width=5)
    draw_arrow(draw, center, (x + 153, y + 15), fill=COLORS["goal"], width=5)
    draw_label(draw, (x + 25, y + 148), "straight: 1", size=25, fill=COLORS["muted"])
    draw_label(draw, (x + 25, y + 192), "diagonal: sqrt(2)", size=25, fill=COLORS["muted"])


def draw_mini_waypoints(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    points = [(x, y + 110), (x + 70, y + 110), (x + 120, y + 60), (x + 220, y + 60), (x + 285, y + 10)]
    draw.line(points, fill=COLORS["path"], width=12, joint="curve")
    for point in points:
        draw.ellipse((point[0] - 12, point[1] - 12, point[0] + 12, point[1] + 12), fill=COLORS["waypoint"])


def draw_mini_replan(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    old = [(x, y + 130), (x + 90, y + 90), (x + 210, y + 90), (x + 300, y + 55)]
    new = [(x, y + 130), (x + 90, y + 90), (x + 150, y + 20), (x + 250, y + 20), (x + 300, y + 55)]
    draw.line(old, fill=(*COLORS["old_path"], 140), width=10)
    draw.rounded_rectangle((x + 155, y + 64, x + 205, y + 114), radius=8, fill=COLORS["overlay"])
    draw.line(new, fill=COLORS["path"], width=12, joint="curve")


def render_frame(
    spec: FrameSpec,
    obstacles: set[tuple[int, int]],
    inflated: set[tuple[int, int]],
) -> Image.Image:
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), COLORS["background"])
    draw = ImageDraw.Draw(image, "RGBA")
    draw_header(draw, spec)
    draw_grid(image, spec, obstacles, inflated)
    draw_formula_panel(draw, spec)
    draw_legend(draw)
    return image.convert("RGB")


def build_frame_specs(
    steps: list[AStarStep],
    path: list[tuple[int, int]],
    waypoints: list[tuple[int, int]],
    replan_path: list[tuple[int, int]],
    overlay_obstacles: set[tuple[int, int]],
    overlay_inflated: set[tuple[int, int]],
) -> list[FrameSpec]:
    frames: list[FrameSpec] = [
        FrameSpec(
            name="occupancy_grid",
            phase="Occupancy grid",
            caption="discrete cells from the map",
            duration_ms=900,
            show_inflation=False,
            side_mode="map",
        ),
        FrameSpec(
            name="inflated_clearance",
            phase="Inflate blocked space",
            caption="robot clearance becomes non-traversable",
            duration_ms=900,
            show_inflation=True,
            side_mode="map",
        ),
        FrameSpec(
            name="snap_start_goal",
            phase="Snap start and goal",
            caption="requested poses move to nearest free cells",
            duration_ms=900,
            show_inflation=True,
            show_snap=True,
            side_mode="snap",
        ),
        FrameSpec(
            name="heuristic",
            phase="Score cells",
            caption="travelled cost plus Euclidean goal distance",
            duration_ms=900,
            show_inflation=True,
            show_heuristic=True,
            side_mode="heuristic",
        ),
    ]

    accepted, rejected = neighbors_no_corner_cutting(START, inflate(build_obstacles()))
    frames.append(
        FrameSpec(
            name="expand_start",
            phase="Expand current cell",
            caption="8-neighbors, no diagonal corner cutting",
            duration_ms=800,
            current=START,
            neighbors=tuple(accepted),
            rejected=tuple(rejected),
            side_mode="heuristic",
        )
    )

    for step in sample_search_steps(steps):
        frames.append(
            FrameSpec(
                name=f"search_{step.index:03d}",
                phase="Search loop",
                caption="pop lowest f from open set, update neighbors",
                duration_ms=150 if step.current != GOAL else 700,
                step=step,
                parent_arrows=step.index > 2,
            )
        )

    final_step = steps[-1]
    reverse_path = list(reversed(path))
    checkpoints = [4, 8, 13, len(reverse_path)]
    for index, count in enumerate(checkpoints):
        prefix_reversed = reverse_path[: min(count, len(reverse_path))]
        prefix = tuple(reversed(prefix_reversed))
        frames.append(
            FrameSpec(
                name=f"backtrack_{index:02d}",
                phase="Backtrack parents",
                caption="goal links back to start",
                duration_ms=260 if count < len(reverse_path) else 850,
                step=final_step,
                path_prefix=prefix,
                parent_arrows=True,
            )
        )

    frames.extend(
        [
            FrameSpec(
                name="dense_path",
                phase="Dense path",
                caption="cell-by-cell route",
                duration_ms=950,
                path=tuple(path),
            ),
            FrameSpec(
                name="waypoints",
                phase="Simplify path",
                caption="keep direction changes as waypoints",
                duration_ms=1200,
                path=tuple(path),
                waypoints=tuple(waypoints),
                side_mode="waypoints",
            ),
            FrameSpec(
                name="temporary_obstacle",
                phase="Temporary obstacle",
                caption="old path becomes blocked",
                duration_ms=950,
                old_path=tuple(path),
                overlay_obstacles=frozenset(overlay_obstacles),
                overlay_inflated=frozenset(overlay_inflated),
                side_mode="replan",
            ),
            FrameSpec(
                name="replanned_path",
                phase="Run A* again",
                caption="updated grid gives a detour",
                duration_ms=1300,
                old_path=tuple(path),
                replan_path=tuple(replan_path),
                waypoints=tuple(simplify_path(replan_path)),
                overlay_obstacles=frozenset(overlay_obstacles),
                overlay_inflated=frozenset(overlay_inflated),
                side_mode="replan",
            ),
        ]
    )
    return frames


def save_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in frame_paths:
        thumb = Image.open(path).resize((384, 216), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    cols = 5
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 384, rows * 252), COLORS["background"])
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        col = index % cols
        row = index // cols
        x = col * 384
        y = row * 252
        sheet.paste(thumb, (x, y))
        draw.text((x + 12, y + 222), f"{index:02d}", font=font(20, mono=True), fill=COLORS["muted"])
    sheet.save(output_path, optimize=True)


def save_gif(frame_paths: list[Path], durations: list[int], output_path: Path, gif_width: int) -> None:
    frames: list[Image.Image] = []
    for path in frame_paths:
        frame = Image.open(path).convert("RGB")
        if gif_width != CANVAS_W:
            gif_height = round(CANVAS_H * gif_width / CANVAS_W)
            frame = frame.resize((gif_width, gif_height), Image.Resampling.LANCZOS)
        frames.append(frame)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def write_manifest(output_dir: Path, specs: list[FrameSpec], frame_paths: list[Path]) -> None:
    manifest = {
        "title": "Aufgabe 03 A* GIF frame deck",
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "frame_count": len(specs),
        "algorithm_notes": [
            "Occupancy grid cells are free, blocked, or inflated blocked space.",
            "Start and goal are snapped to traversable cells before planning.",
            "A* expands 8-neighbors and prevents diagonal corner cutting.",
            "g is travelled cost; h is Euclidean distance to the goal; f = g + h.",
            "The dense path is simplified into waypoints at direction changes.",
            "A temporary obstacle overlay can trigger the same A* loop again for replanning.",
        ],
        "frames": [
            {
                "index": index,
                "file": str(frame_paths[index].relative_to(output_dir)),
                "name": spec.name,
                "phase": spec.phase,
                "duration_ms": spec.duration_ms,
            }
            for index, spec in enumerate(specs)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Aufgabe 03 A* GIF animation frames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/presentation/aufgabe03_astar_frame_deck"),
        help="Directory for PNG frames, GIF, contact sheet, and manifest.",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=1280,
        help="Animated GIF width in pixels. PNG frames are always 1920x1080.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    obstacles = build_obstacles()
    inflated = inflate(obstacles)
    steps, path, _came_from = run_astar(inflated)
    waypoints = simplify_path(path)

    overlay_obstacles = {(14, 9), (15, 9)}
    overlay_inflated = inflate(overlay_obstacles)
    _replan_steps, replan_path, _replan_came_from = run_astar(inflated | overlay_inflated)

    specs = build_frame_specs(
        steps,
        path,
        waypoints,
        replan_path,
        overlay_obstacles,
        overlay_inflated,
    )

    frame_paths: list[Path] = []
    durations: list[int] = []
    for index, spec in enumerate(specs):
        image = render_frame(spec, obstacles, inflated)
        frame_path = frames_dir / f"frame_{index:03d}_{spec.name}.png"
        image.save(frame_path, optimize=True)
        frame_paths.append(frame_path)
        durations.append(spec.duration_ms)

    save_contact_sheet(frame_paths, output_dir / "contact_sheet.png")
    save_gif(frame_paths, durations, output_dir / "astar_path_generation.gif", args.gif_width)
    write_manifest(output_dir, specs, frame_paths)

    print(f"Wrote {len(frame_paths)} frames to {frames_dir}")
    print(f"Wrote GIF: {output_dir / 'astar_path_generation.gif'}")
    print(f"Wrote contact sheet: {output_dir / 'contact_sheet.png'}")
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
