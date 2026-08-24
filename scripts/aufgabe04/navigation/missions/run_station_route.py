"""Dry-run Aufgabe 04 station route planner.

This CLI plans route artifacts only. It never imports ROS and never publishes
motion commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.artifacts import (
    write_diagnostics_json,
    write_overlay_metadata_json,
    write_overlay_svg,
    write_route_csv,
)
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.route_context import build_station_route_dry_run
from scripts.aufgabe04.navigation.planning.route_overlay import RouteOverlayInput, render_route_overlay_svg
from scripts.aufgabe04.stations.models import Station
from scripts.aufgabe04.stations.station_map import normalize_station_id
from scripts.aufgabe04.stations.station_layout_io import load_station_layout_json


DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/station_route_diagnostics.json")


def parse_station_ids(text: str) -> List[str]:
    station_ids = [normalize_station_id(part) for part in text.replace(",", " ").split()]
    if not station_ids:
        raise ValueError("at least one station id is required")
    return station_ids


def plan_station_route(
    map_yaml: Path,
    station_ids: Iterable[str],
    *,
    station_map: Mapping[str, Station] | None = None,
    start: Pose2D | None = None,
    inflation_radius_m: float = 0.0,
    snap_radius_m: float = 0.30,
    line_of_sight_optimization: bool = True,
) -> List[PlanRouteResult]:
    dry_run = build_station_route_dry_run(
        map_yaml,
        station_ids,
        station_map=station_map,
        start=start,
        inflation_radius_m=inflation_radius_m,
        snap_radius_m=snap_radius_m,
        line_of_sight_optimization=line_of_sight_optimization,
    )
    return list(dry_run.results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="ROS map YAML path")
    parser.add_argument(
        "--stations",
        required=True,
        help="Station order, comma- or space-separated, for example 'A,B,C'",
    )
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--inflation-radius-m", type=float, default=0.0)
    parser.add_argument("--snap-radius-m", type=float, default=0.30)
    parser.add_argument("--station-layout-json", type=Path, default=None)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--overlay-svg", type=Path, default=None)
    parser.add_argument("--overlay-metadata-json", type=Path, default=None)
    parser.add_argument("--allow-failed-overlay", action="store_true")
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument("--arena-center-x-m", type=float, default=ArenaBounds.center_x_m)
    parser.add_argument("--arena-center-y-m", type=float, default=ArenaBounds.center_y_m)
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    parser.add_argument(
        "--no-line-of-sight-route-optimization",
        action="store_true",
        help=(
            "Disable planning-time collision-checked A* route compaction. "
            "The follower still never takes uncertified shortcuts."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        station_ids = parse_station_ids(args.stations)
        if (args.start_x is None) != (args.start_y is None):
            raise ValueError("--start-x and --start-y must be provided together")
        start = None
        if args.start_x is not None and args.start_y is not None:
            start = Pose2D(args.start_x, args.start_y, args.start_yaw)
        station_map = None
        if args.station_layout_json is not None:
            station_map = load_station_layout_json(args.station_layout_json)
        arena_bounds = ArenaBounds(
            length_m=args.arena_length_m,
            width_m=args.arena_width_m,
            center_x_m=args.arena_center_x_m,
            center_y_m=args.arena_center_y_m,
            yaw_deg=args.arena_yaw_deg,
            margin_m=args.arena_margin_m,
        )
        dry_run = build_station_route_dry_run(
            args.map,
            station_ids,
            station_map=station_map,
            station_layout_json=args.station_layout_json,
            start=start,
            inflation_radius_m=args.inflation_radius_m,
            snap_radius_m=args.snap_radius_m,
            arena_bounds=arena_bounds,
            line_of_sight_optimization=(
                not args.no_line_of_sight_route_optimization
            ),
        )
        results = list(dry_run.results)
        write_route_csv(args.route_csv, dry_run.results)
        write_diagnostics_json(
            args.diagnostics_json,
            dry_run.results,
            metadata=dry_run.metadata,
        )
        route_failed = any(result.failure is not None for result in dry_run.results)
        if args.overlay_svg is not None:
            if route_failed and not args.allow_failed_overlay:
                pass
            else:
                overlay_input = RouteOverlayInput(
                    grid=dry_run.grid,
                    arena_bounds=dry_run.arena_bounds,
                    stations=dry_run.station_map,
                    visits=dry_run.visits,
                    targets=dry_run.targets,
                    results=dry_run.results,
                    metadata=dry_run.metadata,
                    failed=route_failed,
                )
                write_overlay_svg(args.overlay_svg, render_route_overlay_svg(overlay_input))
                if args.overlay_metadata_json is not None:
                    write_overlay_metadata_json(args.overlay_metadata_json, dry_run.metadata)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if any(result.failure is not None for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
