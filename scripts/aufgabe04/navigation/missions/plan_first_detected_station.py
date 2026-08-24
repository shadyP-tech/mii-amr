"""Plan an Aufgabe 04 route to the first confirmed LiDAR-detected station."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.execution.route_context import build_station_route_dry_run, file_sha256
from scripts.aufgabe04.navigation.planning.route_overlay import RouteOverlayInput, render_route_overlay_svg
from scripts.aufgabe04.navigation.approach.two_stage_approach import pre_approach_candidates
from scripts.aufgabe04.navigation.approach.pre_approach_sampling_state import (
    initial_sampling_state,
    load_sampling_state,
    write_sampling_state,
)
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
    ConfirmedStand,
    select_confirmed_stand_by_id,
    select_unique_confirmed_stand,
)
from scripts.aufgabe04.perception.stand_observation import (
    DEFAULT_OBSERVATION_TIMING_LIMITS,
    OBSERVATION_SCHEMA_VERSION,
    TF_LOOKUP_MODE_SCAN_TIME_EXACT,
    VALID_OBSERVER_CLOCKS,
    ObservationTimingLimits,
    StandObservation,
    load_observation_jsonl,
    observation_timing_limits_from_runtime_config,
    validated_observation_timing,
    validated_observation_stream_clock,
    validated_provenance_observer_clock,
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
    parser.add_argument("--pre-approach-reference-x", type=float, default=None)
    parser.add_argument("--pre-approach-reference-y", type=float, default=None)
    parser.add_argument(
        "--pre-approach-candidate-index",
        type=int,
        default=None,
        help="Orientation-blind sampled inspection pose index; advance after an unusable camera view.",
    )
    parser.add_argument("--pre-approach-sampling-state-json", type=Path, default=None)
    parser.add_argument(
        "--camera-observation-status-json",
        type=Path,
        default=None,
        help="Consume a rejected viewer conditioning status and automatically advance.",
    )
    parser.add_argument(
        "--reject-current-pre-approach",
        default=None,
        help="Record why the current camera view was unusable and advance the sampling state.",
    )
    parser.add_argument("--station-id", default="A")
    parser.add_argument("--stand-yaw-rad", type=float, default=0.0)
    parser.add_argument("--approach-offset-m", type=float, default=0.30)
    parser.add_argument("--keepout-radius-m", type=float, default=0.20)
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
    parser.add_argument(
        "--require-map-hash",
        action="store_true",
        help="Deprecated: map hash/provenance is always required.",
    )
    parser.add_argument(
        "--confirmation-json",
        required=True,
        type=Path,
        help=(
            "Operator/QR confirmation receipt. Must bind the selected detected stand "
            "to --station-id before any route artifact is committed."
        ),
    )
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
    max_scan_age_sec: float = DEFAULT_OBSERVATION_TIMING_LIMITS.max_scan_age_sec,
    max_future_timestamp_sec: float = (
        DEFAULT_OBSERVATION_TIMING_LIMITS.max_future_timestamp_sec
    ),
    max_tf_scan_skew_sec: float = (
        DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_scan_skew_sec
    ),
    required_observer_clock: str | None = None,
    expected_map_yaml_sha256: str | None = None,
    expected_map_bundle_sha256: str | None = None,
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
    if not math.isfinite(observation.observed_at_sec) or observation.observed_at_sec <= 0.0:
        raise ValueError(
            f"observation {observation.observation_id} observed_at_sec is invalid"
        )
    try:
        validated_provenance_observer_clock(
            provenance,
            required_observer_clock=required_observer_clock,
        )
    except ValueError as exc:
        raise ValueError(f"observation {observation.observation_id}: {exc}") from exc
    if provenance.tf_lookup_mode != TF_LOOKUP_MODE_SCAN_TIME_EXACT:
        raise ValueError(
            f"observation {observation.observation_id} did not use exact scan-time TF"
        )
    if not math.isclose(
        provenance.tf_query_stamp_sec,
        provenance.scan_stamp_sec,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"observation {observation.observation_id} TF query/scan timestamp mismatch"
        )

    consumer_limits = ObservationTimingLimits(
        max_scan_age_sec=max_scan_age_sec,
        max_future_timestamp_sec=max_future_timestamp_sec,
        max_tf_age_sec=max_tf_age_sec,
        max_tf_scan_skew_sec=max_tf_scan_skew_sec,
    ).validated()
    timing = validated_observation_timing(
        observer_clock_sec=provenance.observer_clock_sec,
        scan_stamp_sec=provenance.scan_stamp_sec,
        tf_stamp_sec=provenance.tf_lookup_stamp_sec,
        **consumer_limits.as_dict(),
    )
    producer_limits = observation_timing_limits_from_runtime_config(
        provenance.runtime_config
    )
    validated_observation_timing(
        observer_clock_sec=provenance.observer_clock_sec,
        scan_stamp_sec=provenance.scan_stamp_sec,
        tf_stamp_sec=provenance.tf_lookup_stamp_sec,
        **producer_limits.as_dict(),
    )
    for field_name, stored, recomputed in (
        ("scan_age_sec", provenance.scan_age_sec, timing.scan_age_sec),
        ("tf_age_sec", provenance.tf_age_sec, timing.tf_age_sec),
        (
            "tf_scan_skew_sec",
            provenance.tf_scan_skew_sec,
            timing.tf_scan_skew_sec,
        ),
    ):
        if not math.isfinite(stored) or not math.isclose(
            stored,
            recomputed,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"observation {observation.observation_id} inconsistent {field_name}"
            )
    # Callers that have already frozen the map must pass the digest obtained
    # from that same read. Re-reading the path here could validate observation
    # evidence against a newer map revision while planning on the older grid.
    expected_hash = (
        file_sha256(map_yaml)
        if expected_map_yaml_sha256 is None
        else expected_map_yaml_sha256
    )
    if provenance.map_yaml_sha256 != expected_hash:
        raise ValueError(f"observation {observation.observation_id} map hash mismatch")
    if expected_map_bundle_sha256 is not None and (
        provenance.map_bundle_sha256 != expected_map_bundle_sha256
    ):
        raise ValueError(
            f"observation {observation.observation_id} map bundle hash mismatch"
        )


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
            max_scan_age_sec=args.max_scan_age_sec,
            max_future_timestamp_sec=args.max_future_timestamp_sec,
            max_tf_scan_skew_sec=args.max_tf_scan_skew_sec,
            required_observer_clock=args.required_observer_clock,
        )
    validated_observation_stream_clock(
        observations,
        required_observer_clock=args.required_observer_clock,
    )
    return observations


def load_and_validate_confirmation_receipt(
    path: Path,
    *,
    stand: ConfirmedStand,
    station_id: str,
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid confirmation receipt: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("confirmation receipt must be a JSON object")

    confirmed_stand_id = str(payload.get("stand_id", "")).strip()
    if confirmed_stand_id != stand.stand_id:
        raise ValueError(
            f"confirmation stand_id mismatch: expected {stand.stand_id}, got {confirmed_stand_id or '(empty)'}"
        )

    confirmed_station_id = str(payload.get("station_id", "")).strip().upper()
    expected_station_id = station_id.strip().upper()
    if confirmed_station_id != expected_station_id:
        raise ValueError(
            f"confirmation station_id mismatch: expected {expected_station_id}, got {confirmed_station_id or '(empty)'}"
        )

    source = str(payload.get("confirmation_source", "")).strip().lower()
    if source == "operator":
        if payload.get("operator_confirmed") is not True:
            raise ValueError("operator confirmation receipt requires operator_confirmed=true")
    elif source == "qr":
        resolved_station_id = str(payload.get("resolved_station_id", "")).strip().upper()
        if resolved_station_id != expected_station_id:
            raise ValueError(
                "QR confirmation receipt requires resolved_station_id to match station_id"
            )
    else:
        raise ValueError("confirmation_source must be 'operator' or 'qr'")

    return payload


def validate_route_commitment_ready(dry_run) -> None:
    failures = [result.failure for result in dry_run.results if result.failure is not None]
    if failures:
        raise ValueError("route diagnostics contain failure before artifact commitment")
    for index, result in enumerate(dry_run.results):
        if result.route is None:
            raise ValueError(f"route leg {index} has no route")
        if result.route.length_m <= 0.0:
            raise ValueError(f"route leg {index} has no positive final-corridor length")


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
                min_boundary_clearance_m=args.min_boundary_clearance_m,
            ),
            arena_bounds=arena_bounds,
        )
        stands = accumulator.add_observations(observations)
        selected_stand_id = ""
        if args.confirmation_json.exists():
            try:
                confirmation_preview = json.loads(args.confirmation_json.read_text())
                if isinstance(confirmation_preview, dict):
                    selected_stand_id = str(confirmation_preview.get("stand_id", "")).strip()
            except (OSError, json.JSONDecodeError):
                selected_stand_id = ""
        selected = (
            select_confirmed_stand_by_id(stands, selected_stand_id)
            if selected_stand_id
            else select_unique_confirmed_stand(stands)
        )
        confirmation = load_and_validate_confirmation_receipt(
            args.confirmation_json,
            stand=selected,
            station_id=args.station_id,
        )
        start_pose = Pose2D(args.start_x, args.start_y, args.start_yaw)
        if (args.pre_approach_reference_x is None) != (args.pre_approach_reference_y is None):
            raise ValueError(
                "--pre-approach-reference-x and --pre-approach-reference-y must be provided together"
            )
        inspection_reference = (
            start_pose
            if args.pre_approach_reference_x is None
            else Pose2D(args.pre_approach_reference_x, args.pre_approach_reference_y)
        )
        inspection_candidates = pre_approach_candidates(
            Pose2D(selected.x_m, selected.y_m),
            inspection_reference,
            offset_m=args.approach_offset_m,
        )
        sampling_state = None
        if args.pre_approach_sampling_state_json is not None:
            if args.pre_approach_sampling_state_json.exists():
                sampling_state = load_sampling_state(args.pre_approach_sampling_state_json)
                if sampling_state.stand_id != selected.stand_id:
                    raise ValueError("pre-approach sampling state stand_id mismatch")
            else:
                sampling_state = initial_sampling_state(
                    stand_id=selected.stand_id,
                    reference_x_m=inspection_reference.x_m,
                    reference_y_m=inspection_reference.y_m,
                    candidate_count=len(inspection_candidates),
                )
            rejection_reason = args.reject_current_pre_approach
            if args.camera_observation_status_json is not None:
                status_payload = json.loads(
                    args.camera_observation_status_json.read_text()
                )
                if status_payload.get("accepted") is True:
                    raise ValueError(
                        "camera observation is accepted; do not plan another pre-approach"
                    )
                rejection_reason = str(
                    status_payload.get("reason", "no_usable_observation")
                )
            if rejection_reason is not None:
                sampling_state = sampling_state.reject_current(
                    rejection_reason
                )
            selected_candidate_index = sampling_state.candidate_index
        else:
            if (
                args.reject_current_pre_approach is not None
                or args.camera_observation_status_json is not None
            ):
                raise ValueError(
                    "camera rejection inputs require --pre-approach-sampling-state-json"
                )
            selected_candidate_index = (
                0
                if args.pre_approach_candidate_index is None
                else args.pre_approach_candidate_index
            )
        if not 0 <= selected_candidate_index < len(inspection_candidates):
            raise ValueError(
                f"--pre-approach-candidate-index must be in [0, {len(inspection_candidates) - 1}]"
            )
        inspection_pose = inspection_candidates[selected_candidate_index]
        station = station_from_confirmed_stand(
            selected,
            config=DetectedStationLayoutConfig(
                station_id=args.station_id,
                approach_offset_m=args.approach_offset_m,
                keepout_radius_m=args.keepout_radius_m,
                # This yaw describes the orientation-blind inspection target,
                # not the unknown physical stand/QR orientation.  The regular
                # station target convention places the robot opposite this
                # heading and therefore at inspection_pose.
                stand_yaw_rad=inspection_pose.yaw_rad,
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
                "observation_admission": {
                    "observer_clock": observations[0].provenance.observer_clock,
                    "required_observer_clock": args.required_observer_clock,
                    "timing_limits": ObservationTimingLimits(
                        max_scan_age_sec=args.max_scan_age_sec,
                        max_future_timestamp_sec=args.max_future_timestamp_sec,
                        max_tf_age_sec=args.max_tf_age_sec,
                        max_tf_scan_skew_sec=args.max_tf_scan_skew_sec,
                    ).validated().as_dict(),
                },
                "confirmation_json": str(args.confirmation_json),
                "confirmation": confirmation,
                "pre_approach": {
                    "x_m": inspection_pose.x_m,
                    "y_m": inspection_pose.y_m,
                    "yaw_rad": inspection_pose.yaw_rad,
                    "orientation_source": "robot_to_detected_stand_bearing",
                    "hidden_stand_yaw_used": False,
                    "reference_x_m": inspection_reference.x_m,
                    "reference_y_m": inspection_reference.y_m,
                    "candidate_index": selected_candidate_index,
                    "candidate_count": len(inspection_candidates),
                    "candidate_poses": [
                        {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad}
                        for pose in inspection_candidates
                    ],
                },
            },
        )

        dry_run = build_station_route_dry_run(
            args.map,
            [station.station_id],
            station_map={station.station_id: station},
            station_layout_json=args.layout_json,
            start=start_pose,
            inflation_radius_m=args.inflation_radius_m,
            snap_radius_m=args.snap_radius_m,
            arena_bounds=arena_bounds,
        )
        validate_route_commitment_ready(dry_run)
        write_station_layout_json(args.layout_json, [station], metadata)
        write_station_layout_csv(args.layout_csv, [station])

        route_metadata = dict(dry_run.metadata)
        route_metadata["detected_station"] = metadata
        write_route_csv(
            args.route_csv,
            dry_run.results,
            final_yaw_by_leg={0: inspection_pose.yaw_rad},
        )
        write_diagnostics_json(args.diagnostics_json, dry_run.results, metadata=route_metadata)
        if args.pre_approach_sampling_state_json is not None:
            write_sampling_state(args.pre_approach_sampling_state_json, sampling_state)
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
