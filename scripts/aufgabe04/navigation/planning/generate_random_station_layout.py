"""Generate reproducible Aufgabe 04 station layouts from a ROS map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.arena_bounds import (
    DEFAULT_ARENA_CENTER_X_M,
    DEFAULT_ARENA_CENTER_Y_M,
    DEFAULT_ARENA_LENGTH_M,
    DEFAULT_ARENA_MARGIN_M,
    DEFAULT_ARENA_WIDTH_M,
    DEFAULT_ARENA_YAW_DEG,
    ArenaBounds,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.random_station_layout import (
    YAW_MODES,
    RandomStationLayoutConfig,
    generate_random_station_layout,
    parse_station_ids,
    station_ids_from_count,
)
from scripts.aufgabe04.stations.station_layout_io import (
    write_station_layout_csv,
    write_station_layout_json,
)


DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_LAYOUT_JSON = Path("results/aufgabe04/layouts/random_station_layout.json")
DEFAULT_LAYOUT_CSV = Path("results/aufgabe04/layouts/random_station_layout.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="ROS map YAML path")
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument("--station-count", type=int)
    station_group.add_argument("--station-ids", help="Station ids, comma- or space-separated")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--clearance-radius-m", type=float, default=0.10)
    parser.add_argument("--min-station-distance-m", type=float, default=0.40)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--min-start-distance-m", type=float, default=0.40)
    parser.add_argument("--yaw-mode", choices=YAW_MODES, default="toward-center")
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--approach-offset-m", type=float, default=0.30)
    parser.add_argument("--keepout-radius-m", type=float, default=0.20)
    parser.add_argument("--arena-length-m", type=float, default=DEFAULT_ARENA_LENGTH_M)
    parser.add_argument("--arena-width-m", type=float, default=DEFAULT_ARENA_WIDTH_M)
    parser.add_argument("--arena-center-x", type=float, default=DEFAULT_ARENA_CENTER_X_M)
    parser.add_argument("--arena-center-y", type=float, default=DEFAULT_ARENA_CENTER_Y_M)
    parser.add_argument("--arena-yaw-deg", type=float, default=DEFAULT_ARENA_YAW_DEG)
    parser.add_argument("--arena-margin-m", type=float, default=DEFAULT_ARENA_MARGIN_M)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--output-csv", type=Path, nargs="?", const=DEFAULT_LAYOUT_CSV, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if (args.start_x is None) != (args.start_y is None):
            raise ValueError("--start-x and --start-y must be provided together")
        station_ids = (
            station_ids_from_count(args.station_count)
            if args.station_count is not None
            else parse_station_ids(args.station_ids)
        )
        start = None
        if args.start_x is not None and args.start_y is not None:
            start = Pose2D(args.start_x, args.start_y, args.start_yaw)
        result = generate_random_station_layout(
            RandomStationLayoutConfig(
                map_yaml=args.map,
                station_ids=station_ids,
                seed=args.seed,
                clearance_radius_m=args.clearance_radius_m,
                min_station_distance_m=args.min_station_distance_m,
                start=start,
                min_start_distance_m=args.min_start_distance_m,
                yaw_mode=args.yaw_mode,
                max_attempts=args.max_attempts,
                approach_offset_m=args.approach_offset_m,
                keepout_radius_m=args.keepout_radius_m,
                arena_bounds=ArenaBounds(
                    length_m=args.arena_length_m,
                    width_m=args.arena_width_m,
                    center_x_m=args.arena_center_x,
                    center_y_m=args.arena_center_y,
                    yaw_deg=args.arena_yaw_deg,
                    margin_m=args.arena_margin_m,
                ),
            )
        )
        write_station_layout_json(args.output_json, result.stations.values(), result.metadata)
        if args.output_csv is not None:
            write_station_layout_csv(args.output_csv, result.stations.values())
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
