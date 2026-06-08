#!/usr/bin/env python3
"""
Render presentation PNGs for Aufgabe 03 Sequence A:

    SLAM map -> occupancy mask -> inflated clearance -> start/goal snapping
    -> A* search -> dense path -> simplified waypoints.

The frames use the real saved arena map and the same planning assumptions as
map_path_planner.py. Use the bundled Codex Python runtime because it includes
Pillow:

    /Users/stephpark/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
        scripts/aufgabe03/render_sequence_a_semantic_layers.py
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

import map_path_planner as planner


CANVAS_W = 1920
CANVAS_H = 1080
CELL_PX = 12
MAP_LEFT = 282
MAP_TOP = 114

DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_DIR = Path("docs/presentation/aufgabe03_sequence_a_semantic_layers")
DEFAULT_START = (0.005, -0.015)
DEFAULT_GOAL = (1.005, 0.285)

COLORS = {
    "background": (246, 248, 251),
    "transparent": (0, 0, 0, 0),
    "surface": (255, 255, 255),
    "ink": (26, 32, 44),
    "muted": (93, 105, 123),
    "grid": (208, 216, 228),
    "free": (252, 253, 255),
    "occupied": (30, 41, 59),
    "unknown": (148, 163, 184),
    "inflated": (255, 217, 224),
    "open": (250, 204, 21),
    "closed": (147, 197, 253),
    "current": (249, 115, 22),
    "path": (37, 99, 235),
    "waypoint": (6, 182, 212),
    "start": (22, 163, 74),
    "goal": (220, 38, 38),
    "requested": (99, 102, 241),
}


@dataclass(frozen=True)
class SearchStep:
    index: int
    current: tuple[int, int]
    open_cells: frozenset[tuple[int, int]]
    closed_cells: frozenset[tuple[int, int]]
    came_from: dict[tuple[int, int], tuple[int, int]]
    g_score: dict[tuple[int, int], float]
    f_score: float


@dataclass(frozen=True)
class FrameSpec:
    filename: str
    title: str
    subtitle: str
    note: str
    mode: str
    search_step: SearchStep | None = None
    show_requested_snap: bool = False
    show_path: bool = False
    show_waypoints: bool = False


def font(size: int, *, mono: bool = False) -> ImageFont.ImageFont:
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


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: tuple[int, int, int] = COLORS["ink"],
    mono: bool = False,
) -> None:
    draw.text(xy, text, font=font(size, mono=mono), fill=fill)


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int] = COLORS["ink"],
) -> None:
    text_font = font(26)
    tw, th = text_size(draw, text, text_font)
    x, y = xy
    draw.rounded_rectangle((x, y, x + tw + 34, y + th + 22), radius=8, fill=fill)
    draw.text((x + 17, y + 9), text, font=text_font, fill=text_fill)


def world_to_canvas(
    occupancy_map: planner.OccupancyMap,
    world: tuple[float, float],
) -> tuple[float, float]:
    grid_x, grid_y = planner.world_to_grid(world[0], world[1], occupancy_map.metadata)
    cell_world = planner.grid_to_world(grid_x, grid_y, occupancy_map.metadata)
    origin_x, origin_y, _ = occupancy_map.metadata.origin
    fractional_x = (world[0] - origin_x) / occupancy_map.metadata.resolution
    fractional_y = (world[1] - origin_y) / occupancy_map.metadata.resolution
    if abs(cell_world[0] - world[0]) < 1e-12:
        fractional_x = grid_x + 0.5
    if abs(cell_world[1] - world[1]) < 1e-12:
        fractional_y = grid_y + 0.5
    return (
        MAP_LEFT + fractional_x * CELL_PX,
        MAP_TOP + (occupancy_map.height - fractional_y) * CELL_PX,
    )


def cell_rect(occupancy_map: planner.OccupancyMap, cell: tuple[int, int]) -> tuple[int, int, int, int]:
    grid_x, grid_y = cell
    left = MAP_LEFT + grid_x * CELL_PX
    top = MAP_TOP + (occupancy_map.height - 1 - grid_y) * CELL_PX
    return left, top, left + CELL_PX, top + CELL_PX


def cell_center(occupancy_map: planner.OccupancyMap, cell: tuple[int, int]) -> tuple[float, float]:
    left, top, right, bottom = cell_rect(occupancy_map, cell)
    return (left + right) / 2.0, (top + bottom) / 2.0


def run_astar_trace(
    occupancy_map: planner.OccupancyMap,
    blocked: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[SearchStep]:
    queue: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(queue, (0.0, 0, start))
    open_cells = {start}
    closed_cells: set[tuple[int, int]] = set()
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    tie_breaker = 0
    steps: list[SearchStep] = []

    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        if current in closed_cells:
            continue
        open_cells.discard(current)
        closed_cells.add(current)

        if current != goal:
            for neighbor in planner.neighbors_8_no_corner_cutting(occupancy_map, blocked, current):
                tentative_g = (
                    g_score[current]
                    + planner.movement_cost(current, neighbor, occupancy_map.metadata.resolution)
                )
                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                tie_breaker += 1
                open_cells.add(neighbor)
                heuristic = planner.movement_cost(neighbor, goal, occupancy_map.metadata.resolution)
                heapq.heappush(queue, (tentative_g + heuristic, tie_breaker, neighbor))

        f_score = g_score[current] + planner.movement_cost(
            current,
            goal,
            occupancy_map.metadata.resolution,
        )
        steps.append(
            SearchStep(
                index=len(steps),
                current=current,
                open_cells=frozenset(open_cells),
                closed_cells=frozenset(closed_cells),
                came_from=dict(came_from),
                g_score=dict(g_score),
                f_score=f_score,
            )
        )
        if current == goal:
            return steps

    raise RuntimeError(f"No path found from {start} to {goal}")


def draw_base_map(
    draw: ImageDraw.ImageDraw,
    occupancy_map: planner.OccupancyMap,
    raw_image: planner.PgmImage,
    mode: str,
    inflated_blocked: set[tuple[int, int]],
) -> None:
    base_blocked = planner.base_blocked_cells(occupancy_map, block_unknown=True)
    for grid_y in range(occupancy_map.height):
        for grid_x in range(occupancy_map.width):
            cell = (grid_x, grid_y)
            left, top, right, bottom = cell_rect(occupancy_map, cell)
            if mode == "slam":
                image_x, image_y = planner.grid_to_image(grid_x, grid_y, occupancy_map.height)
                pixel = raw_image.pixels[image_y][image_x]
                fill = (pixel, pixel, pixel)
            else:
                cell_state = occupancy_map.cells[grid_y][grid_x]
                if cell_state == planner.CELL_OCCUPIED:
                    fill = COLORS["occupied"]
                elif cell_state == planner.CELL_UNKNOWN:
                    fill = COLORS["unknown"]
                else:
                    fill = COLORS["free"]
                if mode in {"inflated", "snap", "search", "path", "waypoints"}:
                    if cell in inflated_blocked and cell not in base_blocked:
                        fill = COLORS["inflated"]
            draw.rectangle((left, top, right, bottom), fill=fill)

    map_right = MAP_LEFT + occupancy_map.width * CELL_PX
    map_bottom = MAP_TOP + occupancy_map.height * CELL_PX
    draw.rectangle((MAP_LEFT, MAP_TOP, map_right, map_bottom), outline=(94, 109, 130), width=3)


def draw_grid_overlay(draw: ImageDraw.ImageDraw, occupancy_map: planner.OccupancyMap) -> None:
    map_right = MAP_LEFT + occupancy_map.width * CELL_PX
    map_bottom = MAP_TOP + occupancy_map.height * CELL_PX
    for x in range(0, occupancy_map.width + 1, 10):
        cx = MAP_LEFT + x * CELL_PX
        draw.line((cx, MAP_TOP, cx, map_bottom), fill=(148, 163, 184, 70), width=1)
    for y in range(0, occupancy_map.height + 1, 10):
        cy = MAP_TOP + y * CELL_PX
        draw.line((MAP_LEFT, cy, map_right, cy), fill=(148, 163, 184, 70), width=1)


def draw_search(
    draw: ImageDraw.ImageDraw,
    occupancy_map: planner.OccupancyMap,
    step: SearchStep,
) -> None:
    for cell in step.closed_cells:
        left, top, right, bottom = cell_rect(occupancy_map, cell)
        draw.rectangle((left + 1, top + 1, right - 1, bottom - 1), fill=COLORS["closed"])
    for cell in step.open_cells:
        left, top, right, bottom = cell_rect(occupancy_map, cell)
        draw.rectangle((left + 1, top + 1, right - 1, bottom - 1), fill=COLORS["open"])
    left, top, right, bottom = cell_rect(occupancy_map, step.current)
    draw.rectangle((left + 1, top + 1, right - 1, bottom - 1), fill=COLORS["current"])
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), outline=COLORS["surface"], width=2)


def draw_path(
    draw: ImageDraw.ImageDraw,
    occupancy_map: planner.OccupancyMap,
    path: list[tuple[int, int]],
    *,
    color: tuple[int, int, int] = COLORS["path"],
    width: int = 9,
) -> None:
    if len(path) < 2:
        return
    points = [cell_center(occupancy_map, cell) for cell in path]
    draw.line(points, fill=color, width=width, joint="curve")
    for point in points:
        r = max(4, width // 2)
        draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color)


def draw_markers(
    draw: ImageDraw.ImageDraw,
    occupancy_map: planner.OccupancyMap,
    plan: planner.PlanResult,
    *,
    requested: bool,
    waypoints: list[tuple[int, int]] | None = None,
) -> None:
    if waypoints:
        for cell in waypoints:
            cx, cy = cell_center(occupancy_map, cell)
            draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=COLORS["waypoint"], outline=COLORS["surface"], width=3)

    for cell, color in (
        (plan.start_cell, COLORS["start"]),
        (plan.goal_cell, COLORS["goal"]),
    ):
        cx, cy = cell_center(occupancy_map, cell)
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=color, outline=COLORS["surface"], width=4)

    if requested:
        for requested_world, snapped_cell in (
            (plan.start_requested_world, plan.start_cell),
            (plan.goal_requested_world, plan.goal_cell),
        ):
            rx, ry = world_to_canvas(occupancy_map, requested_world)
            sx, sy = cell_center(occupancy_map, snapped_cell)
            draw.ellipse((rx - 12, ry - 12, rx + 12, ry + 12), outline=COLORS["requested"], width=4)
            if abs(rx - sx) > 1.0 or abs(ry - sy) > 1.0:
                draw.line((rx, ry, sx, sy), fill=COLORS["requested"], width=3)


def wrap_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def render_frame(
    spec: FrameSpec,
    occupancy_map: planner.OccupancyMap,
    raw_image: planner.PgmImage,
    inflated_blocked: set[tuple[int, int]],
    plan: planner.PlanResult,
) -> Image.Image:
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), COLORS["transparent"])
    draw = ImageDraw.Draw(image, "RGBA")
    draw_base_map(draw, occupancy_map, raw_image, spec.mode, inflated_blocked)
    draw_grid_overlay(draw, occupancy_map)

    if spec.search_step is not None:
        draw_search(draw, occupancy_map, spec.search_step)
    if spec.show_path:
        draw_path(draw, occupancy_map, plan.path)
    if spec.show_requested_snap or spec.show_path or spec.show_waypoints:
        draw_markers(
            draw,
            occupancy_map,
            plan,
            requested=spec.show_requested_snap,
            waypoints=plan.waypoints if spec.show_waypoints else None,
        )
    return image


def frame_specs(search_steps: list[SearchStep]) -> list[FrameSpec]:
    frames = [
        FrameSpec(
            "001_slam_map.png",
            "SLAM map artifact",
            "SLAM map",
            "The saved PGM is the stationary arena map produced before planning.",
            "slam",
        ),
        FrameSpec(
            "002_occupancy_mask.png",
            "Trinary occupancy mask",
            "Occupancy grid",
            "Map pixels become free, occupied, or unknown planning cells.",
            "occupancy",
        ),
        FrameSpec(
            "003_inflated_clearance_mask.png",
            "Inflated blocked-space mask",
            "Clearance mask",
            "Occupied cells are expanded by the configured robot clearance before A*.",
            "inflated",
        ),
        FrameSpec(
            "004_start_goal_snap.png",
            "Start and goal snapping",
            "Snap to free cells",
            "Requested world poses are converted to grid cells and snapped if needed.",
            "snap",
            show_requested_snap=True,
        ),
    ]
    for step in search_steps:
        frames.append(
            FrameSpec(
                f"{len(frames) + 1:03d}_astar_search_{step.index:03d}.png",
                "A* search expansion",
                "Search",
                "A* expands the lowest f-score cell and updates reachable neighbors.",
                "search",
                search_step=step,
                show_requested_snap=True,
            )
        )
    frames.extend(
        [
            FrameSpec(
                f"{len(frames) + 1:03d}_dense_path.png",
                "Dense A* path",
                "Dense path",
                "The full cell-by-cell route is reconstructed from parent links.",
                "path",
                show_path=True,
            ),
            FrameSpec(
                f"{len(frames) + 2:03d}_simplified_waypoints.png",
                "Simplified waypoint route",
                "Waypoints",
                "Straight segments are compressed into waypoint targets for execution.",
                "waypoints",
                show_path=True,
                show_waypoints=True,
            ),
        ]
    )
    return frames


def save_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in frame_paths:
        thumbs.append(Image.open(path).convert("RGBA").resize((240, 135), Image.Resampling.LANCZOS))
    cols = 6
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * 240, rows * 135), COLORS["transparent"])
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        col = index % cols
        row = index // cols
        x = col * 240
        y = row * 135
        sheet.alpha_composite(thumb, (x, y))
    sheet.save(output_path, optimize=True)


def save_search_gif(
    frame_paths: list[Path],
    output_path: Path,
    gif_width: int,
    frame_duration_ms: int,
    final_hold_ms: int,
) -> None:
    search_paths = [path for path in frame_paths if "_astar_search_" in path.name]
    if not search_paths:
        raise ValueError("No A* search frames found for GIF output.")

    frames: list[Image.Image] = []
    for path in search_paths:
        source = Image.open(path).convert("RGBA")
        if gif_width != CANVAS_W:
            gif_height = round(CANVAS_H * gif_width / CANVAS_W)
            source = source.resize((gif_width, gif_height), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", source.size, COLORS["surface"])
        frame.paste(source, mask=source.getchannel("A"))
        frames.append(frame)

    durations = [frame_duration_ms] * len(frames)
    durations[-1] = final_hold_ms
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def write_manifest(
    output_dir: Path,
    frame_paths: list[Path],
    occupancy_map: planner.OccupancyMap,
    plan: planner.PlanResult,
    gif_path: Path,
) -> None:
    manifest = {
        "title": "Aufgabe 03 Sequence A semantic layers",
        "map": str(occupancy_map.metadata.yaml_path),
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "map_size_cells": {"width": occupancy_map.width, "height": occupancy_map.height},
        "resolution_m_per_cell": occupancy_map.metadata.resolution,
        "start_requested_world_m": list(plan.start_requested_world),
        "goal_requested_world_m": list(plan.goal_requested_world),
        "start_cell": list(plan.start_cell),
        "goal_cell": list(plan.goal_cell),
        "path_cells": len(plan.path),
        "waypoint_count": len(plan.waypoints),
        "path_length_m": plan.path_length_m,
        "search_frame_count": sum(1 for path in frame_paths if "_astar_search_" in path.name),
        "search_gif": str(gif_path.relative_to(output_dir)),
        "frames": [
            {
                "index": index + 1,
                "file": str(path.relative_to(output_dir)),
            }
            for index, path in enumerate(frame_paths)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_readme(output_dir: Path) -> None:
    readme = """# Aufgabe 03 Sequence A Semantic Layers

Presentation-ready PNG frames for explaining the static-map planning pipeline:

1. SLAM map artifact
2. Trinary occupancy mask
3. Inflated blocked-space mask
4. Start and goal snapping
5. One frame per A* search expansion step
6. Dense A* path
7. Simplified waypoint route

All frames use `maps/aufgabe03/arena_1p898x3p9_auto.yaml` and the default
Aufgabe 03 start/goal from `map_path_planner.py` usage notes.
"""
    (output_dir / "README.md").write_text(readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Sequence A semantic layer PNGs.")
    parser.add_argument("--map", default=DEFAULT_MAP, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--start", nargs=2, default=DEFAULT_START, type=float, metavar=("X", "Y"))
    parser.add_argument("--goal", nargs=2, default=DEFAULT_GOAL, type=float, metavar=("X", "Y"))
    parser.add_argument("--inflate-radius-m", default=planner.DEFAULT_INFLATE_RADIUS_M, type=float)
    parser.add_argument("--snap-radius-m", default=planner.DEFAULT_SNAP_RADIUS_M, type=float)
    parser.add_argument("--gif-width", default=1280, type=int)
    parser.add_argument("--search-frame-duration-ms", default=70, type=int)
    parser.add_argument("--search-final-hold-ms", default=900, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    occupancy_map = planner.load_occupancy_map(args.map)
    raw_image = planner.read_pgm(occupancy_map.metadata.image_path)
    plan, inflated_blocked, _inflation_cells = planner.plan_path(
        occupancy_map,
        tuple(args.start),
        tuple(args.goal),
        inflate_radius_m=args.inflate_radius_m,
        snap_radius_m=args.snap_radius_m,
    )
    search_trace = run_astar_trace(occupancy_map, inflated_blocked, plan.start_cell, plan.goal_cell)
    specs = frame_specs(search_trace)

    frame_paths: list[Path] = []
    for spec in specs:
        image = render_frame(spec, occupancy_map, raw_image, inflated_blocked, plan)
        path = frames_dir / spec.filename
        image.save(path, optimize=True)
        frame_paths.append(path)

    save_contact_sheet(frame_paths, output_dir / "contact_sheet.png")
    gif_path = output_dir / "astar_search_animation.gif"
    save_search_gif(
        frame_paths,
        gif_path,
        args.gif_width,
        args.search_frame_duration_ms,
        args.search_final_hold_ms,
    )
    write_manifest(output_dir, frame_paths, occupancy_map, plan, gif_path)
    write_readme(output_dir)

    print(f"Wrote {len(frame_paths)} PNG frames to {frames_dir}")
    print(f"Wrote contact sheet: {output_dir / 'contact_sheet.png'}")
    print(f"Wrote A* search GIF: {gif_path}")
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
