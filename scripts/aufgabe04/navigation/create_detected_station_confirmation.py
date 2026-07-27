"""Create a confirmation receipt for one LiDAR-detected Aufgabe 04 station.

This command is offline/artifact-only. It does not publish ROS messages, start
missions, or move the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.plan_first_detected_station import (
    validate_observation_provenance,
)
from scripts.aufgabe04.navigation.map_io import freeze_map_bundle
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
    select_confirmed_stand_by_id,
    select_unique_confirmed_stand,
)
from scripts.aufgabe04.perception.stand_observation import (
    DEFAULT_OBSERVATION_TIMING_LIMITS,
    VALID_OBSERVER_CLOCKS,
    ObservationTimingLimits,
    load_observation_jsonl,
    validated_observation_stream_clock,
)


DEFAULT_OBSERVATIONS_JSONL = Path("results/aufgabe04/detected_stations/stand_observations.jsonl")
DEFAULT_CONFIRMATION_JSON = Path(
    "results/aufgabe04/detected_stations/first_detected_station_confirmation.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-jsonl", type=Path, default=DEFAULT_OBSERVATIONS_JSONL)
    parser.add_argument("--map", required=True, type=Path, help="ROS map YAML path")
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument("--station-id", required=True)
    parser.add_argument(
        "--stand-id",
        default="",
        help=(
            "Explicit confirmed stand id to bind. Required when multiple stands "
            "are confirmed unless --list-confirmed is used."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_CONFIRMATION_JSON)
    parser.add_argument(
        "--list-confirmed",
        action="store_true",
        help="Print confirmed stand ids and coordinates, then exit without writing a receipt.",
    )
    parser.add_argument("--confirmation-source", required=True, choices=["operator", "qr"])
    parser.add_argument(
        "--operator-confirmed",
        action="store_true",
        help="Required when --confirmation-source=operator.",
    )
    parser.add_argument(
        "--resolved-station-id",
        default="",
        help="Required when --confirmation-source=qr; must match --station-id.",
    )
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--merge-distance-m", type=float, default=0.18)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-observation-age-sec", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-boundary-clearance-m", type=float, default=0.10)
    parser.add_argument(
        "--max-tf-age-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_age_sec,
    )
    parser.add_argument(
        "--max-scan-age-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_scan_age_sec,
    )
    parser.add_argument(
        "--max-future-timestamp-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_future_timestamp_sec,
    )
    parser.add_argument(
        "--max-tf-scan-skew-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_scan_skew_sec,
    )
    parser.add_argument("--required-map-frame", default="map")
    parser.add_argument("--required-base-frame", default="base_footprint")
    parser.add_argument("--required-localization-source", default=None, choices=["amcl", "tf"])
    parser.add_argument(
        "--required-observer-clock",
        default=None,
        choices=sorted(VALID_OBSERVER_CLOCKS),
    )
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument("--arena-center-x-m", type=float, default=ArenaBounds.center_x_m)
    parser.add_argument("--arena-center-y-m", type=float, default=ArenaBounds.center_y_m)
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    return parser


def build_confirmation_receipt(args) -> dict[str, object]:
    station_id = args.station_id.strip().upper()
    if not station_id:
        raise ValueError("station id must not be empty")
    map_bundle = freeze_map_bundle(
        args.map,
        semantic_map_id=args.semantic_map_id or args.map.stem,
        planning_frame=args.required_map_frame,
    )
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
            max_scan_age_sec=args.max_scan_age_sec,
            max_future_timestamp_sec=args.max_future_timestamp_sec,
            max_tf_scan_skew_sec=args.max_tf_scan_skew_sec,
            required_observer_clock=args.required_observer_clock,
            expected_map_yaml_sha256=map_bundle.yaml_sha256,
            expected_map_bundle_sha256=map_bundle.bundle_sha256,
        )
    observer_clock = validated_observation_stream_clock(
        observations,
        required_observer_clock=args.required_observer_clock,
    )
    timing_limits = ObservationTimingLimits(
        max_scan_age_sec=args.max_scan_age_sec,
        max_future_timestamp_sec=args.max_future_timestamp_sec,
        max_tf_age_sec=args.max_tf_age_sec,
        max_tf_scan_skew_sec=args.max_tf_scan_skew_sec,
    ).validated()

    arena_bounds = ArenaBounds(
        length_m=args.arena_length_m,
        width_m=args.arena_width_m,
        center_x_m=args.arena_center_x_m,
        center_y_m=args.arena_center_y_m,
        yaw_deg=args.arena_yaw_deg,
        margin_m=args.arena_margin_m,
    )
    accumulator = StandConfirmationAccumulator(
        config=StandConfirmationConfig(
            merge_distance_m=args.merge_distance_m,
            min_hits=args.min_hits,
            max_age_sec=args.max_observation_age_sec,
            min_confidence=args.min_confidence,
            min_boundary_clearance_m=args.min_boundary_clearance_m,
        ),
        arena_bounds=arena_bounds,
    )
    stands = accumulator.add_observations(observations)
    if args.list_confirmed:
        for stand in stands:
            print(
                f"{stand.stand_id}: x={stand.x_m:.3f} y={stand.y_m:.3f} "
                f"hits={stand.hit_count} confidence={stand.confidence:.3f}"
            )
        return {}
    stand = (
        select_confirmed_stand_by_id(stands, args.stand_id)
        if args.stand_id
        else select_unique_confirmed_stand(stands)
    )

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmation_source": args.confirmation_source,
        "station_id": station_id,
        "stand_id": stand.stand_id,
        "stand_x_m": stand.x_m,
        "stand_y_m": stand.y_m,
        "stand_confidence": stand.confidence,
        "stand_hit_count": stand.hit_count,
        "source_observation_ids": list(stand.source_observation_ids),
        "observations_jsonl": str(args.observations_jsonl),
        "map_yaml": str(args.map),
        "map_yaml_sha256": map_bundle.yaml_sha256,
        "map_image_sha256": map_bundle.image_sha256,
        "map_bundle_sha256": map_bundle.bundle_sha256,
        "required_map_frame": args.required_map_frame,
        "required_base_frame": args.required_base_frame,
        "observer_clock": observer_clock,
        "required_observer_clock": args.required_observer_clock,
        "observation_timing_limits": timing_limits.as_dict(),
        "operator_note": args.operator_note,
    }
    if args.confirmation_source == "operator":
        if not args.operator_confirmed:
            raise ValueError("operator confirmation requires --operator-confirmed")
        receipt["operator_confirmed"] = True
    else:
        resolved_station_id = args.resolved_station_id.strip().upper()
        if resolved_station_id != station_id:
            raise ValueError("--resolved-station-id must match --station-id for QR confirmation")
        receipt["resolved_station_id"] = resolved_station_id
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = build_confirmation_receipt(args)
        if args.list_confirmed:
            return 0
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
