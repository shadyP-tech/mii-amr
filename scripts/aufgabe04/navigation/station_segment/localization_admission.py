"""Map/odometry admission and uncertainty adapters for one segment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.content_hashed_evidence import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.coverage_replan_coordinator import (
    CoverageReplanCoordinator,
)
from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.mission_execution_gate import DiagnosticsSnapshot
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    odom_pose_to_map,
    pose_route_sha256,
    transform_map_route_to_odom,
    validate_odom_execution_identity,
    write_odom_execution_certificate,
)
from scripts.aufgabe04.navigation.odom_route_adapter import (
    OdomExecutionContext,
    adapt_map_route_update_to_odom,
    evaluate_map_odom_stationary_stability,
)
from scripts.aufgabe04.navigation.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.route_uncertainty_admission import (
    RouteUncertaintyAdmissionConfig,
    evaluate_route_uncertainty_admission,
    route_uncertainty_admission_evidence_sha256,
)
from scripts.aufgabe04.navigation.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.waypoint_csv import (
    SelectedRouteLeg,
    poses_from_waypoints,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid

from .route_bundle import (
    _resolved_map_execution_certificate,
    _runtime_command_owner,
)

def _preflight_pose(
    raw: object,
    *,
    frame_id: str,
    child_frame_id: str,
    name: str,
) -> Pose2D:
    if not isinstance(raw, Mapping):
        raise ValueError(f"preflight did not provide {name}")
    if (
        raw.get("frame_id") != frame_id
        or raw.get("child_frame_id") != child_frame_id
    ):
        raise ValueError(f"preflight {name} frame identity mismatch")
    try:
        pose = Pose2D(
            float(raw["x_m"]),
            float(raw["y_m"]),
            float(raw["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"preflight {name} is malformed: {exc}") from exc
    if not all(
        math.isfinite(value)
        for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError(f"preflight {name} is non-finite")
    return pose

def _preflight_map_from_odom(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
) -> tuple[PlanarTransform2D, float, float]:
    raw = preflight.map_from_odom
    if not isinstance(raw, Mapping):
        raise ValueError("preflight did not provide a direct map->odom transform")
    if (
        raw.get("target_frame") != map_frame
        or raw.get("source_frame") != odom_frame
    ):
        raise ValueError("preflight map->odom transform frame identity mismatch")
    try:
        transform = PlanarTransform2D(
            float(raw["x_m"]),
            float(raw["y_m"]),
            float(raw["yaw_rad"]),
        )
        stamp_sec = float(raw["stamp_sec"])
        capture_time_sec = float(raw["capture_time_sec"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"preflight map->odom transform is malformed: {exc}") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        stamp_sec,
        capture_time_sec,
    )):
        raise ValueError("preflight map->odom timestamps must be finite/non-negative")
    return transform, stamp_sec, capture_time_sec

def _preflight_stationary_map_from_odom_window(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
) -> tuple[tuple[PlanarTransform2D, ...], tuple[dict[str, object], ...]]:
    """Validate ordered direct-TF samples paired with stationary AMCL."""

    raw_samples = preflight.stationary_map_from_odom_samples
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        raise ValueError(
            "preflight did not provide at least two stationary direct "
            "map->odom transform samples"
        )
    transforms: list[PlanarTransform2D] = []
    provenance: list[dict[str, object]] = []
    previous_receipt_nanoseconds: int | None = None
    previous_stamp_nanoseconds: int | None = None
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"preflight stationary map->odom sample {index} is malformed"
            )
        if (
            raw.get("source") != "direct_dynamic_tf"
            or raw.get("target_frame") != map_frame
            or raw.get("source_frame") != odom_frame
            or raw.get("observed_target_frame") != map_frame
            or raw.get("observed_source_frame") != odom_frame
            or raw.get("amcl_sample_index") != index
        ):
            raise ValueError(
                "preflight stationary map->odom sample provenance or frame "
                f"identity mismatch at index {index}"
            )
        try:
            transform = PlanarTransform2D(
                float(raw["x_m"]),
                float(raw["y_m"]),
                float(raw["yaw_rad"]),
            )
            stamp_sec = float(raw["stamp_sec"])
            receipt_time_sec = float(raw["receipt_time_sec"])
            capture_time_sec = float(raw["capture_time_sec"])
            stamp_nanoseconds = raw["stamp_nanoseconds"]
            receipt_time_nanoseconds = raw["receipt_time_nanoseconds"]
            capture_time_nanoseconds = raw["capture_time_nanoseconds"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"preflight stationary map->odom sample {index} is malformed: "
                f"{exc}"
            ) from exc
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (stamp_sec, receipt_time_sec, capture_time_sec)
        ):
            raise ValueError(
                f"preflight stationary map->odom sample {index} has invalid "
                "timestamps"
            )
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in (
                stamp_nanoseconds,
                receipt_time_nanoseconds,
                capture_time_nanoseconds,
            )
        ):
            raise ValueError(
                f"preflight stationary map->odom sample {index} has invalid "
                "nanosecond timestamps"
            )
        for seconds, nanoseconds, name in (
            (stamp_sec, stamp_nanoseconds, "stamp"),
            (receipt_time_sec, receipt_time_nanoseconds, "receipt"),
            (capture_time_sec, capture_time_nanoseconds, "capture"),
        ):
            if not math.isclose(
                seconds,
                nanoseconds / 1_000_000_000.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    f"preflight stationary map->odom sample {index} {name} "
                    "second/nanosecond timestamps disagree"
                )
        if capture_time_nanoseconds < receipt_time_nanoseconds:
            raise ValueError(
                f"preflight stationary map->odom sample {index} was captured "
                "before receipt"
            )
        if (
            previous_receipt_nanoseconds is not None
            and receipt_time_nanoseconds <= previous_receipt_nanoseconds
        ):
            raise ValueError(
                "preflight stationary map->odom samples do not have strictly "
                "newer direct-TF receipts"
            )
        if (
            previous_stamp_nanoseconds is not None
            and stamp_nanoseconds <= previous_stamp_nanoseconds
        ):
            raise ValueError(
                "preflight stationary map->odom samples do not have strictly "
                "newer direct-TF stamps"
            )
        transforms.append(transform)
        provenance.append(
            {
                "sample_index": index,
                "amcl_sample_index": index,
                "source": "direct_dynamic_tf",
                "stamp_sec": stamp_sec,
                "stamp_nanoseconds": stamp_nanoseconds,
                "receipt_time_sec": receipt_time_sec,
                "receipt_time_nanoseconds": receipt_time_nanoseconds,
                "capture_time_sec": capture_time_sec,
                "capture_time_nanoseconds": capture_time_nanoseconds,
            }
        )
        previous_receipt_nanoseconds = receipt_time_nanoseconds
        previous_stamp_nanoseconds = stamp_nanoseconds
    return tuple(transforms), tuple(provenance)

def _conservative_preflight_covariance(
    preflight: RosPreflightResult,
) -> tuple[PlanarCovariance, float, dict[str, object]]:
    if not preflight.stationary_amcl_samples:
        raise ValueError("preflight has no accepted stationary AMCL samples")
    maximum_position_variance_m2 = 0.0
    maximum_yaw_variance_rad2 = 0.0
    sample_evidence = []
    for index, sample in enumerate(preflight.stationary_amcl_samples):
        raw_covariance = sample.get("covariance")
        if not isinstance(raw_covariance, list) or len(raw_covariance) != 36:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is incomplete"
            )
        try:
            values = [float(value) for value in raw_covariance]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is malformed"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"preflight AMCL sample {index} covariance is non-finite"
            )
        xx_m2 = values[0]
        xy_m2 = values[1]
        yx_m2 = values[6]
        yy_m2 = values[7]
        yaw_variance_rad2 = values[35]
        symmetry_tolerance = max(
            1.0e-12,
            1.0e-6 * max(abs(xy_m2), abs(yx_m2)),
        )
        if abs(xy_m2 - yx_m2) > symmetry_tolerance:
            raise ValueError(
                f"preflight AMCL sample {index} covariance is asymmetric"
            )
        covariance = PlanarCovariance(
            xx_m2,
            0.5 * (xy_m2 + yx_m2),
            yy_m2,
        )
        largest_position_variance_m2 = 0.5 * (
            covariance.xx_m2
            + covariance.yy_m2
            + math.hypot(
                covariance.xx_m2 - covariance.yy_m2,
                2.0 * covariance.xy_m2,
            )
        )
        if yaw_variance_rad2 < 0.0:
            raise ValueError(
                f"preflight AMCL sample {index} yaw covariance is negative"
            )
        maximum_position_variance_m2 = max(
            maximum_position_variance_m2,
            largest_position_variance_m2,
        )
        maximum_yaw_variance_rad2 = max(
            maximum_yaw_variance_rad2,
            yaw_variance_rad2,
        )
        sample_evidence.append(
            {
                "sample_index": index,
                "xx_m2": covariance.xx_m2,
                "xy_m2": covariance.xy_m2,
                "yy_m2": covariance.yy_m2,
                "yaw_variance_rad2": yaw_variance_rad2,
                "largest_position_variance_m2": (
                    largest_position_variance_m2
                ),
            }
        )
    # An isotropic envelope at the largest observed eigenvalue dominates each
    # accepted sample in every route-normal direction. This is conservative;
    # it does not turn a five-sample spread into an accuracy claim.
    covariance_envelope = PlanarCovariance(
        maximum_position_variance_m2,
        0.0,
        maximum_position_variance_m2,
    )
    return (
        covariance_envelope,
        math.sqrt(maximum_yaw_variance_rad2),
        {
            "envelope_kind": "isotropic_maximum_eigenvalue",
            "sample_count": len(sample_evidence),
            "samples": sample_evidence,
        },
    )

def _angle_distance_rad(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)

def _covariance_bounded_continuity_limits(
    covariance: PlanarCovariance,
    *,
    heading_sigma_rad: float,
    sigma_multiplier: float,
    translation_hard_cap_m: float,
    yaw_hard_cap_rad: float,
) -> tuple[float, float]:
    """Reuse, rather than duplicate, the route's localization allowance."""

    allocated_translation_m = sigma_multiplier * math.sqrt(
        covariance.xx_m2
    )
    allocated_yaw_rad = sigma_multiplier * heading_sigma_rad
    return (
        min(translation_hard_cap_m, allocated_translation_m),
        min(yaw_hard_cap_rad, allocated_yaw_rad),
    )

def _admit_stationary_map_from_odom_window(
    preflight: RosPreflightResult,
    *,
    map_frame: str,
    odom_frame: str,
    final_map_from_odom: PlanarTransform2D,
    final_stamp_sec: float,
    final_capture_time_sec: float,
    max_translation_drift_m: float,
    max_yaw_drift_rad: float,
) -> tuple[PlanarTransform2D, dict[str, object]]:
    """Bind the final certificate transform to a stable direct-TF window."""

    samples, provenance = _preflight_stationary_map_from_odom_window(
        preflight,
        map_frame=map_frame,
        odom_frame=odom_frame,
    )
    final_provenance = provenance[-1]
    if (
        final_capture_time_sec
        < float(final_provenance["capture_time_sec"])
        or final_stamp_sec < float(final_provenance["stamp_sec"])
    ):
        raise ValueError(
            "preflight final map->odom transform predates its stationary "
            "sample window"
        )
    stability = evaluate_map_odom_stationary_stability(
        (*samples, final_map_from_odom),
        max_translation_drift_m=max_translation_drift_m,
        max_yaw_drift_rad=max_yaw_drift_rad,
    )
    if not stability.accepted:
        raise ValueError(
            "preflight stationary map->odom transform window rejected: "
            f"{stability.reason}"
        )
    admitted = stability.frozen_map_from_odom
    if admitted is None:
        raise ValueError(
            "preflight stationary map->odom admission did not freeze a transform"
        )
    if admitted != final_map_from_odom:
        raise ValueError(
            "preflight stationary map->odom admission changed the final transform"
        )
    evidence = stability.to_evidence()
    evidence["sample_provenance"] = [
        *provenance,
        {
            "sample_index": len(samples),
            "source": "final_preflight_tf_lookup",
            "stamp_sec": final_stamp_sec,
            "capture_time_sec": final_capture_time_sec,
        },
    ]
    return admitted, evidence

def _build_odom_execution_admission(
    *,
    args,
    resolved,
    leg: SelectedRouteLeg,
    preflight: RosPreflightResult,
    diagnostics_snapshot: DiagnosticsSnapshot,
) -> tuple[
    tuple[Pose2D, ...],
    OdomExecutionContext,
    dict[str, object],
    "_OdomRouteUncertaintyGate",
]:
    """Freeze and seal one map-route projection for odom-only control."""

    map_route = poses_from_waypoints(leg.executable_waypoints)
    map_pose = _preflight_pose(
        preflight.route_pose,
        frame_id=resolved.map_frame,
        child_frame_id=resolved.base_frame,
        name="map-frame base pose",
    )
    odom_pose = _preflight_pose(
        preflight.odom_pose,
        frame_id=resolved.odom_frame,
        child_frame_id=resolved.base_frame,
        name="odom-frame base pose",
    )
    map_from_odom, transform_stamp_sec, transform_capture_time_sec = (
        _preflight_map_from_odom(
            preflight,
            map_frame=resolved.map_frame,
            odom_frame=resolved.odom_frame,
        )
    )
    composed_map_pose = odom_pose_to_map(odom_pose, map_from_odom)
    composition_position_error_m = math.hypot(
        composed_map_pose.x_m - map_pose.x_m,
        composed_map_pose.y_m - map_pose.y_m,
    )
    composition_yaw_error_rad = _angle_distance_rad(
        composed_map_pose.yaw_rad,
        map_pose.yaw_rad,
    )
    if composition_position_error_m > args.certified_route_tube_radius_m:
        raise ValueError(
            "preflight map/odom/transform composition exceeds the certified "
            f"route tube: {composition_position_error_m:.6f} m"
        )
    if composition_yaw_error_rad > args.max_stationary_amcl_yaw_spread_rad:
        raise ValueError(
            "preflight map/odom/transform yaw composition is inconsistent: "
            f"{composition_yaw_error_rad:.6f} rad"
        )
    odom_route = transform_map_route_to_odom(map_route, map_from_odom)
    route_yaw_lever_arm_m = max(
        math.hypot(
            pose.x_m - map_route[0].x_m,
            pose.y_m - map_route[0].y_m,
        )
        for pose in map_route
    ) + args.uncertainty_robot_radius_m
    startup_decision = certified_static_startup_decision(
        odom_pose,
        odom_route,
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
    )
    if not startup_decision.ok:
        raise ValueError(
            "odom pose is outside the transformed certified startup segment: "
            + startup_decision.route_check.reason
        )

    covariance, heading_sigma_rad, covariance_evidence = (
        _conservative_preflight_covariance(preflight)
    )
    allocated_translation_drift_m = (
        args.uncertainty_sigma_multiplier
        * math.sqrt(covariance.xx_m2)
    )
    allocated_yaw_drift_rad = (
        args.uncertainty_sigma_multiplier * heading_sigma_rad
    )
    (
        continuity_translation_limit_m,
        continuity_yaw_limit_rad,
    ) = _covariance_bounded_continuity_limits(
        covariance,
        heading_sigma_rad=heading_sigma_rad,
        sigma_multiplier=args.uncertainty_sigma_multiplier,
        translation_hard_cap_m=args.max_map_odom_translation_drift_m,
        yaw_hard_cap_rad=args.max_map_odom_yaw_drift_rad,
    )
    map_from_odom, stationary_stability_evidence = (
        _admit_stationary_map_from_odom_window(
            preflight,
            map_frame=resolved.map_frame,
            odom_frame=resolved.odom_frame,
            final_map_from_odom=map_from_odom,
            final_stamp_sec=transform_stamp_sec,
            final_capture_time_sec=transform_capture_time_sec,
            max_translation_drift_m=continuity_translation_limit_m,
            max_yaw_drift_rad=continuity_yaw_limit_rad,
        )
    )
    arena_bounds = validate_arena_boundary_evidence(
        diagnostics_snapshot.metadata
    )
    base_costmap = Costmap.from_occupancy_grid(
        load_occupancy_grid(args.uncertainty_map_yaml)
    ).with_arena_bounds(arena_bounds)
    admission_config = RouteUncertaintyAdmissionConfig(
        robot_radius_m=args.uncertainty_robot_radius_m,
        collision_margin_m=args.uncertainty_collision_margin_m,
        fixed_odom_tracking_bound_m=args.certified_route_tube_radius_m,
        empirical_odom_drift_bound_m=(
            args.uncertainty_odom_drift_bound_m
        ),
        braking_latency_distance_m=(
            args.uncertainty_braking_latency_distance_m
        ),
        localization_sigma_multiplier=args.uncertainty_sigma_multiplier,
        # The same covariance envelope is used twice: once as reserved route
        # clearance here and once as the maximum live map<-odom correction the
        # monitor may accept. It is not charged a second time.
        heading_sigma_rad=heading_sigma_rad,
        heading_lever_arm_m=(
            args.uncertainty_robot_radius_m
            if args.uncertainty_heading_lever_arm_m is None
            else args.uncertainty_heading_lever_arm_m
        ),
        sampling_spacing_m=args.uncertainty_clearance_sample_spacing_m,
        heading_reference_x_m=map_route[0].x_m,
        heading_reference_y_m=map_route[0].y_m,
    )
    admission = evaluate_route_uncertainty_admission(
        base_costmap,
        map_route,
        covariance,
        admission_config,
    )
    if not admission.decision.accepted:
        limiting = admission.decision.limiting_segment_id or "unknown"
        margin = admission.decision.remaining_margin_m
        margin_text = "unknown" if margin is None else f"{margin:.6f} m"
        raise ValueError(
            "route uncertainty budget exhausted: "
            f"limiting_segment={limiting} remaining_margin={margin_text}"
        )

    map_certificate, map_certificate_sha256 = (
        _resolved_map_execution_certificate(args, diagnostics_snapshot)
    )
    branch_evidence = {
        "schema_version": 1,
        "proof_id": args.localization_branch_proof_id,
        "method": "operator_known_start_or_asymmetric_landmark_attestation",
        "map_frame": resolved.map_frame,
        "map_bundle_sha256": str(
            diagnostics_snapshot.metadata.get("map_bundle_sha256", "")
        ),
        "source_map_execution_certificate_sha256": map_certificate_sha256,
        "claim_boundary": (
            "operator branch selection; covariance alone does not resolve "
            "symmetric-map aliases"
        ),
    }
    ambiguity_evidence_sha256 = payload_sha256(branch_evidence)
    budget_payload = {
        "schema_version": 1,
        "source": "route_uncertainty_admission",
        "admission": admission.to_evidence_dict(),
        "covariance_envelope": covariance_evidence,
        "runtime_map_odom_continuity_allocation": {
            "position_covariance_allocation_m": (
                allocated_translation_drift_m
            ),
            "yaw_covariance_allocation_rad": allocated_yaw_drift_rad,
            "translation_hard_cap_m": (
                args.max_map_odom_translation_drift_m
            ),
            "yaw_hard_cap_rad": args.max_map_odom_yaw_drift_rad,
            "effective_translation_limit_m": (
                continuity_translation_limit_m
            ),
            "effective_yaw_limit_rad": continuity_yaw_limit_rad,
            "route_yaw_lever_arm_m": route_yaw_lever_arm_m,
            "threshold_contract": (
                "live correction must remain within the same covariance "
                "allowance already reserved in route clearance"
            ),
        },
        "stationary_map_from_odom_stability": (
            stationary_stability_evidence
        ),
        "localization_branch_evidence": branch_evidence,
        "preflight_composition": {
            "position_error_m": composition_position_error_m,
            "yaw_error_rad": composition_yaw_error_rad,
        },
    }
    uncertainty_budget_sha256 = write_content_hashed_json(
        args.uncertainty_budget_json,
        budget_payload,
        hash_field="route_uncertainty_artifact_sha256",
    )

    odom_certificate = OdomExecutionCertificate(
        source_map_route_sha256=pose_route_sha256(map_route),
        source_map_execution_certificate_sha256=map_certificate_sha256,
        transformed_odom_route_sha256=pose_route_sha256(odom_route),
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        map_from_odom=map_from_odom,
        transform_stamp_sec=transform_stamp_sec,
        transform_capture_time_sec=transform_capture_time_sec,
        waypoint_count=len(map_route),
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
        command_owner=_runtime_command_owner(resolved.namespace),
        uncertainty_budget_sha256=uncertainty_budget_sha256,
        ambiguity_evidence_sha256=ambiguity_evidence_sha256,
    )
    odom_certificate_sha256 = write_odom_execution_certificate(
        args.odom_execution_certificate_json,
        odom_certificate,
    )
    validate_odom_execution_identity(
        odom_certificate,
        source_map_route=map_route,
        source_map_execution_certificate_sha256=map_certificate_sha256,
        transformed_odom_route=odom_route,
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        tracking_tube_radius_m=args.certified_route_tube_radius_m,
        command_owner=_runtime_command_owner(resolved.namespace),
        map_from_odom=map_from_odom,
        transform_stamp_sec=transform_stamp_sec,
        transform_capture_time_sec=transform_capture_time_sec,
        uncertainty_budget_sha256=uncertainty_budget_sha256,
        ambiguity_evidence_sha256=ambiguity_evidence_sha256,
    )
    context = OdomExecutionContext(
        map_frame=resolved.map_frame,
        odom_frame=resolved.odom_frame,
        base_frame=resolved.base_frame,
        frozen_map_from_odom=map_from_odom,
        certificate_sha256=odom_certificate_sha256,
        max_map_from_odom_translation_drift_m=(
            continuity_translation_limit_m
        ),
        max_map_from_odom_yaw_drift_rad=continuity_yaw_limit_rad,
    )
    replacement_route_gate = _OdomRouteUncertaintyGate(
        costmap=base_costmap,
        covariance=covariance,
        config=admission_config,
        evidence_root=(
            None
            if args.coverage_transient_replan_session_root is None
            else Path(args.coverage_transient_replan_session_root)
            / "odom_execution_replans"
        ),
    )
    return (
        odom_route,
        context,
        {
            "odom_execution_certificate_sha256": odom_certificate_sha256,
            "odom_execution_certificate_json": str(
                args.odom_execution_certificate_json
            ),
            "uncertainty_budget_sha256": uncertainty_budget_sha256,
            "uncertainty_budget_json": str(args.uncertainty_budget_json),
            "ambiguity_evidence_sha256": ambiguity_evidence_sha256,
            "source_map_execution_certificate_sha256": (
                map_certificate_sha256
            ),
            "source_map_route_sha256": pose_route_sha256(map_route),
            "transformed_odom_route_sha256": pose_route_sha256(odom_route),
            "minimum_remaining_margin_m": (
                admission.decision.remaining_margin_m
            ),
            "map_from_odom": {
                "x_m": map_from_odom.x_m,
                "y_m": map_from_odom.y_m,
                "yaw_rad": map_from_odom.yaw_rad,
                "stamp_sec": transform_stamp_sec,
                "capture_time_sec": transform_capture_time_sec,
            },
            "map_execution_certificate_route_kind": map_certificate.route_kind,
        },
        replacement_route_gate,
    )

class _OdomRouteUncertaintyGate:
    """Re-admit each replacement map route before odom transformation."""

    def __init__(
        self,
        *,
        costmap: Costmap,
        covariance: PlanarCovariance,
        config: RouteUncertaintyAdmissionConfig,
        evidence_root: Path | None,
    ) -> None:
        self._costmap = costmap
        self._covariance = covariance
        self._config = config
        self._evidence_root = evidence_root

    def adapt(
        self,
        update: RouteUpdate,
        context: OdomExecutionContext,
    ) -> RouteUpdate:
        if update.kind is not RouteUpdateKind.ADOPT:
            return update
        admission = evaluate_route_uncertainty_admission(
            self._costmap,
            update.waypoints,
            self._covariance,
            self._config,
        )
        evidence_sha256 = route_uncertainty_admission_evidence_sha256(
            admission
        )
        evidence_path = None
        if self._evidence_root is not None:
            route_revision = (
                "unknown"
                if update.route_revision is None
                else f"{update.route_revision:06d}"
            )
            route_hash_prefix = str(update.route_hash or "unhashed")[:16]
            evidence_path = (
                self._evidence_root
                / (
                    f"route_revision_{route_revision}_"
                    f"{route_hash_prefix}_uncertainty.json"
                )
            )
            stored_hash = write_content_hashed_json(
                evidence_path,
                admission.to_evidence_dict(),
                hash_field="route_uncertainty_admission_sha256",
            )
            if stored_hash != evidence_sha256:
                raise ValueError(
                    "replacement route uncertainty evidence hash mismatch"
                )
        evidence_fields = {
            "replacement_route_uncertainty_admission_sha256": (
                evidence_sha256
            ),
            "replacement_route_uncertainty_admission_json": (
                "" if evidence_path is None else str(evidence_path)
            ),
            "replacement_route_uncertainty_accepted": (
                admission.decision.accepted
            ),
            "replacement_route_uncertainty_remaining_margin_m": (
                admission.decision.remaining_margin_m
            ),
            "replacement_route_uncertainty_limiting_segment_id": (
                admission.decision.limiting_segment_id
            ),
        }
        if not admission.decision.accepted:
            return RouteUpdate(
                kind=RouteUpdateKind.REJECT,
                reason="replacement route uncertainty budget exhausted",
                route_revision=update.route_revision,
                target_revision=update.target_revision,
                route_hash=update.route_hash,
                requires_zero_cycle=True,
                event_name="dynamic_route_rejected",
                event_fields={
                    **dict(update.event_fields),
                    **evidence_fields,
                    "fail_closed": True,
                },
            )
        adapted = adapt_map_route_update_to_odom(update, context)
        return RouteUpdate(
            kind=adapted.kind,
            waypoints=adapted.waypoints,
            target_index=adapted.target_index,
            reason=adapted.reason,
            route_revision=adapted.route_revision,
            target_revision=adapted.target_revision,
            route_hash=adapted.route_hash,
            requires_zero_cycle=True,
            event_name=adapted.event_name,
            event_fields={
                **dict(adapted.event_fields),
                **evidence_fields,
            },
        )

class _OdomBlockageRecoveryAdapter:
    """Keep the planner map-native and adapt only its sealed handoff."""

    def __init__(
        self,
        provider,
        context: OdomExecutionContext,
        uncertainty_gate: _OdomRouteUncertaintyGate,
    ) -> None:
        self._provider = provider
        self._context = context
        self._uncertainty_gate = uncertainty_gate

    def __call__(
        self,
        map_pose: Pose2D,
        stop_reason: str,
        stop_details: Mapping[str, object],
    ) -> RouteUpdate | None:
        update = self._provider(map_pose, stop_reason, stop_details)
        if update is None:
            return None
        return self._uncertainty_gate.adapt(update, self._context)

