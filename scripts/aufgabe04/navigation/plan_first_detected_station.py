"""Plan an Aufgabe 04 route to the first confirmed LiDAR-detected station."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_overlay_metadata_json,
    write_overlay_svg,
    write_route_csv,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.route_context import build_station_route_dry_run, file_sha256
from scripts.aufgabe04.navigation.route_overlay import RouteOverlayInput, render_route_overlay_svg
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
    select_first_confirmed_stand,
)
from scripts.aufgabe04.perception.stand_observation import (
    OBSERVATION_SCHEMA_VERSION,
    StandObservation,
    load_observation_jsonl,
)
from scripts.aufgabe04.stations.detected_station_layout import (
    DetectedStationLayoutConfig,
    detected_station_metadata,
    station_from_confirmed_stand,
)
from scripts.aufgabe04.stations.station_layout_io import (
    write_station_layout_csv,
    write_station_layout_json,
)


DEFAULT_OBSERVATIONS_JSONL = Path("results/aufgabe04/detected_stations/stand_observations.jsonl")
DEFAULT_LAYOUT_JSON = Path("results/aufgabe04/detected_stations/first_detected_station_layout.json")
DEFAULT_LAYOUT_CSV = Path("results/aufgabe04/detected_stations/first_detected_station_layout.csv")
DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/first_detected_station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/first_detected_station_route_diagnostics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-jsonl", type=Path, default=DEFAULT_OBSERVATIONS_JSONL)
    parser.add_argument("--map", required=True, type=Path, help="ROS map YAML path")
    parser.add_argument("--start-x", required=True, type=float)
    parser.add_argument("--start-y", required=True, type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--station-id", default="A")
    parser.add_argument("--stand-yaw-rad", type=float, default=0.0)
    parser.add_argument("--approach-offset-m", type=float, default=0.30)
    parser.add_argument("--keepout-radius-m", type=float, default=0.20)
    parser.add_argument("--merge-distance-m", type=float, default=0.18)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-observation-age-sec", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--required-map-frame", default="map")
    parser.add_argument("--required-base-frame", default="base_footprint")
    parser.add_argument("--required-localization-source", default=None, choices=["amcl", "tf"])
    parser.add_argument("--require-map-hash", action="store_true")
    parser.add_argument("--inflation-radius-m", type=float, default=0.0)
    parser.add_argument("--snap-radius-m", type=float, default=0.30)
    parser.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--layout-csv", type=Path, default=DEFAULT_LAYOUT_CSV)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--overlay-svg", type=Path, default=None)
    parser.add_argument("--overlay-metadata-json", type=Path, default=None)
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument("--arena-center-x-m", type=float, default=ArenaBounds.center_x_m)
    parser.add_argument("--arena-center-y-m", type=float, default=ArenaBounds.center_y_m)
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    return parser


def validate_observation_provenance(
    observation: StandObservation,
    *,
    map_yaml: Path,
    required_map_frame: str,
    required_base_frame: str,
    required_localization_source: str | None,
    max_tf_age_sec: float,
    require_map_hash: bool,
) -> None:
    provenance = observation.provenance
    if provenance.schema_version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported observation schema_version: {provenance.schema_version}")
    if provenance.map_frame != required_map_frame:
        raise ValueError(f"observation {observation.observation_id} map_frame mismatch")
    if provenance.base_frame != required_base_frame:
        raise ValueError(f"observation {observation.observation_id} base_frame mismatch")
    if required_localization_source is not None and provenance.localization_source != required_localization_source:
        raise ValueError(f"observation {observation.observation_id} localization source mismatch")
    if not provenance.scan_frame:
        raise ValueError(f"observation {observation.observation_id} missing scan_frame")
    if provenance.tf_age_sec > max_tf_age_sec:
        raise ValueError(f"observation {observation.observation_id} TF age exceeds limit")
    if require_map_hash:
        expected_hash = file_sha256(map_yaml)
        if provenance.map_yaml_sha256 != expected_hash:
            raise ValueError(f"observation {observation.observation_id} map hash mismatch")


def validated_observations(args) -> tuple[StandObservation, ...]:
    observations = load_observation_jsonl(args.observations_jsonl)
    if not observations:
        raise ValueError("no stand observations found")
    for observation in observations:
        validate_observation_provenance(
            observation,
            map_yaml=args.map,
            required_map_frame=args.required_map_frame,
            required_base_frame=args.required_base_frame,
            required_localization_source=args.required_localization_source,
            max_tf_age_sec=args.max_tf_age_sec,
            require_map_hash=args.require_map_hash,
        )
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        arena_bounds = ArenaBounds(
            length_m=args.arena_length_m,
            width_m=args.arena_width_m,
            center_x_m=args.arena_center_x_m,
            center_y_m=args.arena_center_y_m,
            yaw_deg=args.arena_yaw_deg,
            margin_m=args.arena_margin_m,
        )
        observations = validated_observations(args)
        accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(
                merge_distance_m=args.merge_distance_m,
                min_hits=args.min_hits,
                max_age_sec=args.max_observation_age_sec,
                min_confidence=args.min_confidence,
            ),
            arena_bounds=arena_bounds,
        )
        stands = accumulator.add_observations(observations)
        selected = select_first_confirmed_stand(stands)
        station = station_from_confirmed_stand(
            selected,
            config=DetectedStationLayoutConfig(
                station_id=args.station_id,
                approach_offset_m=args.approach_offset_m,
                keepout_radius_m=args.keepout_radius_m,
                stand_yaw_rad=args.stand_yaw_rad,
                arena_length_m=args.arena_length_m,
                arena_width_m=args.arena_width_m,
                arena_center_x_m=args.arena_center_x_m,
                arena_center_y_m=args.arena_center_y_m,
                arena_yaw_deg=args.arena_yaw_deg,
                arena_margin_m=args.arena_margin_m,
            ),
        )
        metadata = detected_station_metadata(
            selected,
            source_observation_path=str(args.observations_jsonl),
            extra={
                "map_yaml": str(args.map),
                "map_yaml_sha256": file_sha256(args.map),
                "required_map_frame": args.required_map_frame,
                "required_base_frame": args.required_base_frame,
            },
        )
        write_station_layout_json(args.layout_json, [station], metadata)
        write_station_layout_csv(args.layout_csv, [station])

        dry_run = build_station_route_dry_run(
            args.map,
            [station.station_id],
            station_map={station.station_id: station},
            station_layout_json=args.layout_json,
            start=Pose2D(args.start_x, args.start_y, args.start_yaw),
            inflation_radius_m=args.inflation_radius_m,
            snap_radius_m=args.snap_radius_m,
            arena_bounds=arena_bounds,
        )
        route_metadata = dict(dry_run.metadata)
        route_metadata["detected_station"] = metadata
        write_route_csv(args.route_csv, dry_run.results)
        write_diagnostics_json(args.diagnostics_json, dry_run.results, metadata=route_metadata)
        failed = any(result.failure is not None for result in dry_run.results)
        if args.overlay_svg is not None:
            overlay_input = RouteOverlayInput(
                grid=dry_run.grid,
                arena_bounds=dry_run.arena_bounds,
                stations=dry_run.station_map,
                visits=dry_run.visits,
                targets=dry_run.targets,
                results=dry_run.results,
                metadata=route_metadata,
                failed=failed,
            )
            write_overlay_svg(args.overlay_svg, render_route_overlay_svg(overlay_input))
            if args.overlay_metadata_json is not None:
                write_overlay_metadata_json(args.overlay_metadata_json, route_metadata)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
