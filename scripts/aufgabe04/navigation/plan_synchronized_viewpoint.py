"""Plan or validate a synchronized-viewpoint recommendation.

Dynamic acquisition routes remain simulation-only.  A real-robot observer may
use ``--environment real --workflow-mode survey-only`` to validate a committed,
passively collected recommendation and append it to a real survey catalog.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.axis_acquisition_feedback import (
    AXIS_ACQUISITION_FEEDBACK_CONTRACT,
    AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION,
    AXIS_ACQUISITION_STATIC_TARGET_FAILURES,
    axis_acquisition_feedback_binding,
    canonical_json_sha256,
    finite_feedback_pose,
    load_axis_acquisition_feedback,
    write_axis_acquisition_feedback,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    FaceNormalCandidate,
    circular_keepout_cells,
    face_normal_candidates as planner_face_candidates,
    minimum_static_obstacle_inflation_m,
    plan_axis_acquisition,
    plan_dynamic_approach,
    plan_fixed_approach,
    point_clearance_to_blocked_m,
)
from scripts.aufgabe04.navigation.dynamic_replan_policy import (
    DynamicReplanPolicy,
    DynamicReplanState,
)
from scripts.aufgabe04.navigation.map_io import (
    FrozenMapBundle,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.route_smoothing import (
    segment_is_collision_free,
    supercover_segment_cells,
)
from scripts.aufgabe04.navigation.route_revision_store import (
    RouteRevisionError,
    RouteRevisionStore,
    read_committed_revision,
    read_route_revision,
)
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    MaterialTarget,
    angular_distance,
    load_recommendation,
    recommendation_to_dict,
)
from scripts.aufgabe04.perception.arrival_pose_estimator import (
    arrival_pose_record_from_recommendation,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    arrival_pose_catalog_sha256,
    load_arrival_pose_catalog,
    new_arrival_pose_catalog,
    set_expected_candidate_uids,
    upsert_arrival_pose,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import CatalogProvenance


_POINT_APPROACH_AXIS_STATES = frozenset(
    {"axis_acquisition", "viewpoint_sampling"}
)
_PHYSICAL_AXIS_STATES = frozenset({"target_committed", "resolved"})
WORKFLOW_IMMEDIATE_APPROACH = "immediate-approach"
WORKFLOW_SURVEY_ONLY = "survey-only"
_WORKFLOW_MODES = (WORKFLOW_IMMEDIATE_APPROACH, WORKFLOW_SURVEY_ONLY)
_KNOWN_STAND_KEEPOUT_EPSILON_M = 1.0e-10
_KNOWN_STAND_EGRESS_SEARCH_RADIUS_CELLS = 4


def _point_approach_targets(
    recommendation,
    active_publication: Mapping[str, object] | None,
) -> tuple[MaterialTarget, MaterialTarget]:
    """Order unresolved antipodes, retaining an installed safe side first.

    Point-acquisition face IDs carry no physical side identity.  Until the
    observer commits an axis, either antipode is therefore an admissible
    observation target.  Once one has an active route, keep that provisional
    side stable while it remains present in the recommendation.
    """

    if recommendation.material_target.evidence_state != recommendation.axis_state:
        raise ValueError("point_approach_evidence_state_mismatch")
    if any(face.identity_resolved for face in recommendation.face_candidates):
        raise ValueError("point_approach_has_resolved_face_identity")
    if recommendation.side_evidence.hard or recommendation.side_evidence.valid:
        raise ValueError("point_approach_has_side_evidence")
    evidence_state = recommendation.material_target.evidence_state
    targets = [
        MaterialTarget(face.face_id, face.pose, evidence_state)
        for face in recommendation.face_candidates
    ]
    preferred_id = recommendation.material_target.face_id
    targets.sort(key=lambda target: target.face_id != preferred_id)
    if active_publication is not None:
        active_target = active_publication.get("target")
        active_x = active_y = None
        if isinstance(active_target, Mapping):
            try:
                active_x = float(active_target.get("x_m"))
                active_y = float(active_target.get("y_m"))
            except (TypeError, ValueError):
                active_x = active_y = None
        if (
            active_x is not None
            and active_y is not None
            and math.isfinite(active_x)
            and math.isfinite(active_y)
        ):
            # Provisional near/far IDs are regenerated from the observer's
            # current robot-relative view and can swap physical hemispheres.
            # Preserve the installed target geometrically, not by that
            # unstable label, so a heartbeat cannot ping-pong antipodes.
            targets.sort(
                key=lambda target: math.hypot(
                    target.pose.x_m - active_x,
                    target.pose.y_m - active_y,
                )
            )
        else:
            active_id = (
                active_target.get("face_id")
                if isinstance(active_target, Mapping)
                else None
            )
            targets.sort(key=lambda target: target.face_id != active_id)
    if len(targets) != 2:
        raise ValueError("point approach requires exactly two antipodal targets")
    return targets[0], targets[1]


def _plan_point_approach(
    costmap: Costmap,
    start: Pose2D,
    stand: Pose2D,
    targets: Sequence[MaterialTarget],
    *,
    config: DynamicApproachConfig,
):
    """Try the preferred unresolved viewpoint, then its safe antipode."""

    attempts = []
    last_result = None
    for index, target in enumerate(targets):
        result = plan_axis_acquisition(
            costmap,
            start,
            stand,
            target.pose,
            config=config,
        )
        attempts.append(
            {
                "face_id": target.face_id,
                "pose": asdict(target.pose),
                "priority": index,
                "valid": result.plan is not None,
                "diagnostics": asdict(result.diagnostics),
            }
        )
        last_result = result
        if result.plan is not None:
            return result, target, tuple(attempts)
    assert last_result is not None
    return last_result, None, tuple(attempts)


@dataclass(frozen=True)
class KnownStandKeepoutOverlay:
    """One start-aware raster overlay over the static-inflated map."""

    costmap: Costmap
    keepouts: tuple[dict[str, float], ...]
    rasterized_cell_count: int
    blocked_cell_count: int
    start_cell: GridCell | None
    start_cell_exempted: bool
    egress_anchor: Pose2D | None
    egress_cells: tuple[GridCell, ...]
    start_join_clearance_m: float | None


def _normalize_known_stand_keepouts(
    specifications: Sequence[Sequence[float]],
) -> tuple[dict[str, float], ...]:
    normalized = []
    for index, specification in enumerate(specifications):
        if len(specification) != 3:
            raise ValueError(
                f"known stand keepout {index} must contain x, y, and radius"
            )
        x_m, y_m, radius_m = (float(value) for value in specification)
        if not all(math.isfinite(value) for value in (x_m, y_m, radius_m)):
            raise ValueError(f"known stand keepout {index} must be finite")
        if radius_m <= 0.0:
            raise ValueError(
                f"known stand keepout {index} radius must be positive"
            )
        normalized.append({"x_m": x_m, "y_m": y_m, "radius_m": radius_m})
    return tuple(normalized)


def _known_stand_keepout_costmap(
    costmap: Costmap,
    specifications: Sequence[Sequence[float]],
    *,
    start: Pose2D | None = None,
) -> KnownStandKeepoutOverlay:
    """Overlay explicitly supplied robot-center exclusion disks.

    The radius is already a configuration-space keepout radius.  This keeps
    the CLI contract independent of the target recommendation's radius and is
    important for non-target stands, whose live geometry is unavailable to
    this single-target planner.
    """

    normalized = _normalize_known_stand_keepouts(specifications)
    cells = set()
    for keepout in normalized:
        cells.update(
            circular_keepout_cells(
                costmap,
                Pose2D(keepout["x_m"], keepout["y_m"]),
                keepout["radius_m"],
            )
        )
    rasterized_cell_count = len(cells)

    start_cell = None if start is None else costmap.world_to_grid(start)
    start_cell_exempted = False
    egress_anchor = None
    egress_cells: tuple[GridCell, ...] = ()
    start_join_clearance_m = None
    if start is not None and start_cell in cells:
        # A closed disk can touch a grid square even though the exact robot
        # pose lies safely outside it.  Exempt only that containing square,
        # and only from this run-local raster overlay.  Static occupancy and
        # inflation remain immutable and therefore still veto the escape.
        exact_start_outside_all = all(
            math.hypot(
                start.x_m - keepout["x_m"],
                start.y_m - keepout["y_m"],
            )
            > keepout["radius_m"] + _KNOWN_STAND_KEEPOUT_EPSILON_M
            for keepout in normalized
        )
        static_clearance_m = point_clearance_to_blocked_m(costmap, start)
        known_stand_clearance_m = min(
            (
                math.hypot(
                    start.x_m - keepout["x_m"],
                    start.y_m - keepout["y_m"],
                )
                - keepout["radius_m"]
                for keepout in normalized
            ),
            default=math.inf,
        )
        if (
            costmap.is_traversable(start_cell)
            and exact_start_outside_all
            and static_clearance_m > _KNOWN_STAND_KEEPOUT_EPSILON_M
            and known_stand_clearance_m > _KNOWN_STAND_KEEPOUT_EPSILON_M
        ):
            egress_anchor = _find_known_stand_egress_anchor(
                costmap,
                cells,
                normalized,
                start,
            )
        if egress_anchor is not None:
            # Keep the legacy one-cell raster exemption as an explicit
            # handoff certificate.  Planning itself starts at the certified
            # exterior anchor, so the unsafe center of this containing cell
            # can never become an A* waypoint.
            cells.remove(start_cell)
            start_cell_exempted = True
            egress_cells = supercover_segment_cells(
                costmap,
                start,
                egress_anchor,
            )
            start_join_clearance_m = min(
                static_clearance_m,
                known_stand_clearance_m,
            )

    overlay = costmap if not cells else costmap.with_blocked_cells(cells)
    return KnownStandKeepoutOverlay(
        costmap=overlay,
        keepouts=normalized,
        rasterized_cell_count=rasterized_cell_count,
        blocked_cell_count=len(cells),
        start_cell=start_cell,
        start_cell_exempted=start_cell_exempted,
        egress_anchor=egress_anchor,
        egress_cells=egress_cells,
        start_join_clearance_m=start_join_clearance_m,
    )


def _point_to_segment_distance_m(
    point: Pose2D,
    start: Pose2D,
    end: Pose2D,
) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    denominator = dx * dx + dy * dy
    if denominator <= _KNOWN_STAND_KEEPOUT_EPSILON_M**2:
        return math.hypot(point.x_m - start.x_m, point.y_m - start.y_m)
    fraction = max(
        0.0,
        min(
            1.0,
            (
                (point.x_m - start.x_m) * dx
                + (point.y_m - start.y_m) * dy
            )
            / denominator,
        ),
    )
    closest_x = start.x_m + fraction * dx
    closest_y = start.y_m + fraction * dy
    return math.hypot(point.x_m - closest_x, point.y_m - closest_y)


def _segment_outside_known_stands(
    start: Pose2D,
    end: Pose2D,
    keepouts: Sequence[Mapping[str, float]],
) -> bool:
    return all(
        _point_to_segment_distance_m(
            Pose2D(float(keepout["x_m"]), float(keepout["y_m"])),
            start,
            end,
        )
        > float(keepout["radius_m"]) + _KNOWN_STAND_KEEPOUT_EPSILON_M
        for keepout in keepouts
    )


def _find_known_stand_egress_anchor(
    costmap: Costmap,
    rasterized_cells: set[GridCell],
    keepouts: Sequence[Mapping[str, float]],
    start: Pose2D,
) -> Pose2D | None:
    """Find the nearest rigorously safe A* start outside the raster halo.

    A conservative disk raster can contain the exact, collision-free robot
    pose and can even isolate that containing cell.  Grid A* cannot start at
    the cell center in that case because the center itself may lie inside the
    physical disk.  Search a bounded local neighborhood for an exterior cell
    center connected directly to the exact pose by a static-free segment that
    remains strictly outside every supplied disk.
    """

    start_cell = costmap.world_to_grid(start)
    candidates = []
    radius = _KNOWN_STAND_EGRESS_SEARCH_RADIUS_CELLS
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cell = GridCell(start_cell.x + dx, start_cell.y + dy)
            if cell in rasterized_cells or not costmap.is_traversable(cell):
                continue
            anchor = costmap.grid_to_world(cell)
            distance_m = math.hypot(
                anchor.x_m - start.x_m,
                anchor.y_m - start.y_m,
            )
            if distance_m <= _KNOWN_STAND_KEEPOUT_EPSILON_M:
                continue
            candidates.append((distance_m, abs(dx) + abs(dy), cell, anchor))
    for _distance_m, _manhattan, _cell, anchor in sorted(candidates):
        if not segment_is_collision_free(costmap, start, anchor):
            continue
        if not _segment_outside_known_stands(start, anchor, keepouts):
            continue
        return anchor
    return None


def _prepend_certified_known_stand_egress(
    plan_result,
    *,
    source_start: Pose2D,
    overlay: KnownStandKeepoutOverlay,
    target_stand: Pose2D,
    target_keepout_radius_m: float,
):
    """Restore the exact robot pose before a plan made from its safe anchor."""

    if overlay.egress_anchor is None:
        return plan_result
    if plan_result.plan is None:
        return plan_result
    anchor = overlay.egress_anchor
    if not plan_result.plan.waypoints:
        raise ValueError("known stand egress plan has no anchor waypoint")
    first = plan_result.plan.waypoints[0].pose
    if math.hypot(first.x_m - anchor.x_m, first.y_m - anchor.y_m) > 1.0e-8:
        raise ValueError("known stand egress plan lost its certified anchor")
    if _point_to_segment_distance_m(
        target_stand,
        source_start,
        anchor,
    ) <= target_keepout_radius_m + _KNOWN_STAND_KEEPOUT_EPSILON_M:
        raise ValueError("known stand egress intersects target stand keepout")
    if overlay.start_join_clearance_m is None or (
        overlay.start_join_clearance_m <= _KNOWN_STAND_KEEPOUT_EPSILON_M
    ):
        raise ValueError("known stand egress lacks positive start clearance")
    target_start_clearance_m = (
        math.hypot(
            source_start.x_m - target_stand.x_m,
            source_start.y_m - target_stand.y_m,
        )
        - target_keepout_radius_m
    )
    if target_start_clearance_m <= _KNOWN_STAND_KEEPOUT_EPSILON_M:
        raise ValueError("known stand egress starts inside target stand keepout")
    start_join_clearance_m = min(
        overlay.start_join_clearance_m,
        target_start_clearance_m,
    )
    diagnostics = replace(
        plan_result.diagnostics,
        start_join_clearance_m=start_join_clearance_m,
    )
    exact_start = replace(
        plan_result.plan.waypoints[0],
        pose=Pose2D(source_start.x_m, source_start.y_m, math.nan),
        protected=False,
        corridor=False,
    )
    egress_length_m = math.hypot(
        anchor.x_m - source_start.x_m,
        anchor.y_m - source_start.y_m,
    )
    plan = replace(
        plan_result.plan,
        waypoints=(exact_start, *plan_result.plan.waypoints),
        length_m=plan_result.plan.length_m + egress_length_m,
        diagnostics=diagnostics,
    )
    return replace(plan_result, plan=plan, diagnostics=diagnostics)


def _validate_known_stand_route_clearance(
    plan,
    keepouts: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], ...]:
    """Fail closed unless the exact polyline stays outside every disk."""

    poses = tuple(waypoint.pose for waypoint in plan.waypoints)
    if not poses:
        raise ValueError("known stand keepout validation received an empty route")
    clearances = []
    for index, keepout in enumerate(keepouts):
        center = Pose2D(float(keepout["x_m"]), float(keepout["y_m"]))
        radius_m = float(keepout["radius_m"])
        if len(poses) == 1:
            minimum_clearance_m = math.hypot(
                center.x_m - poses[0].x_m,
                center.y_m - poses[0].y_m,
            )
        else:
            minimum_clearance_m = min(
                _point_to_segment_distance_m(center, segment_start, segment_end)
                for segment_start, segment_end in zip(poses, poses[1:])
            )
        if minimum_clearance_m <= (
            radius_m + _KNOWN_STAND_KEEPOUT_EPSILON_M
        ):
            raise ValueError(
                "known_stand_keepout_route_clearance_failed:"
                f"index={index}:clearance_m={minimum_clearance_m:.9f}:"
                f"radius_m={radius_m:.9f}"
            )
        clearances.append(
            {
                "x_m": center.x_m,
                "y_m": center.y_m,
                "radius_m": radius_m,
                "minimum_route_clearance_m": minimum_clearance_m,
            }
        )
    return tuple(clearances)


def route_kind_for_axis_state(
    axis_state: str,
    workflow_mode: str = WORKFLOW_IMMEDIATE_APPROACH,
) -> str:
    if axis_state in _POINT_APPROACH_AXIS_STATES:
        return axis_state
    if axis_state in _PHYSICAL_AXIS_STATES:
        return (
            "survey_complete"
            if workflow_mode == WORKFLOW_SURVEY_ONLY
            else "synchronized_face_approach"
        )
    raise ValueError(f"unsupported synchronized viewpoint axis state: {axis_state}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_survey_arrival(
    *,
    args,
    recommendation,
    costmap: Costmap,
    start: Pose2D,
    planning_start: Pose2D,
    known_stand_overlay: KnownStandKeepoutOverlay,
    config: DynamicApproachConfig,
    known_stand_keepouts: Sequence[Mapping[str, float]],
    map_bundle: FrozenMapBundle,
    now: float,
) -> tuple[dict[str, object], object]:
    """Validate and atomically persist one committed arrival estimate."""

    selected_index, selected_face = next(
        (
            (index, face)
            for index, face in enumerate(recommendation.face_candidates)
            if face.face_id == recommendation.material_target.face_id
        ),
        (None, None),
    )
    if selected_index is None or selected_face is None:
        raise ValueError("committed material target does not reference a face")
    entry_radius = config.standoff_distance_m + config.terminal_corridor_length_m
    entry = Pose2D(
        recommendation.stand.center.x_m
        + entry_radius * math.cos(selected_face.outward_normal_rad),
        recommendation.stand.center.y_m
        + entry_radius * math.sin(selected_face.outward_normal_rad),
        selected_face.pose.yaw_rad,
    )
    fixed = FaceNormalCandidate(
        face_id=selected_index,
        normal_rad=selected_face.outward_normal_rad,
        target=selected_face.pose,
        entry=entry,
    )
    fixed_result = plan_fixed_approach(
        costmap,
        planning_start,
        recommendation.stand.center,
        fixed,
        config=config,
    )
    if fixed_result.plan is None:
        raise ValueError(
            fixed_result.diagnostics.failure_reason
            or "committed arrival pose failed fixed-target validation"
        )
    fixed_result = _prepend_certified_known_stand_egress(
        fixed_result,
        source_start=start,
        overlay=known_stand_overlay,
        target_stand=recommendation.stand.center,
        target_keepout_radius_m=config.stand_keepout_radius_m,
    )
    known_stand_clearances = _validate_known_stand_route_clearance(
        fixed_result.plan,
        known_stand_keepouts,
    )

    map_sha256 = map_bundle.yaml_sha256
    provenance = CatalogProvenance(
        planning_frame=args.map_frame,
        map_yaml_sha256=map_sha256,
        world_id=args.world_id,
        world_sha256=args.world_sha256,
        session_id=args.session_id,
        environment=args.environment,
        map_bundle_sha256=args.expected_map_bundle_sha256,
        candidate_snapshot_sha256=args.candidate_snapshot_sha256,
        station_identity_registry_sha256=(
            args.station_identity_registry_sha256
        ),
        survey_config_sha256=args.survey_config_sha256,
        calibration_profile_sha256=args.calibration_profile_sha256,
        survey_input_binding_sha256=args.survey_input_binding_sha256,
    )
    candidate_uid = args.candidate_uid
    if args.arrival_pose_catalog.exists():
        catalog = load_arrival_pose_catalog(
            args.arrival_pose_catalog,
            required_provenance=provenance,
        )
        if args.expected_candidate_uid:
            catalog = set_expected_candidate_uids(
                catalog,
                args.expected_candidate_uid,
                updated_unix_sec=now,
            )
    else:
        catalog = new_arrival_pose_catalog(
            catalog_id=args.catalog_id,
            provenance=provenance,
            expected_candidate_uids=args.expected_candidate_uid,
            created_unix_sec=now,
        )
    if recommendation.axis_sample_count < args.axis_sample_count:
        raise ValueError(
            "measured axis evidence has fewer stable inlier samples than "
            f"required: {recommendation.axis_sample_count} < {args.axis_sample_count}"
        )
    record = arrival_pose_record_from_recommendation(
        recommendation,
        candidate_uid=candidate_uid,
        map_yaml_sha256=map_sha256,
        corridor_length_m=config.terminal_corridor_length_m,
        # Use the immutable observation time in the record so replaying the
        # same sensor evidence is an idempotent catalog upsert.
        validated_unix_sec=recommendation.observation_unix_sec,
        axis_sample_count=recommendation.axis_sample_count,
        estimator=(
            "simulation/silhouette_head_rectangle"
            if args.environment == "simulation"
            else "real/silhouette_head_rectangle_camera_info_tf"
        ),
        source=(
            "simulation/synchronized_viewpoint"
            if args.environment == "simulation"
            else "real/synchronized_viewpoint"
        ),
    )
    catalog = upsert_arrival_pose(catalog, record, updated_unix_sec=now)
    catalog_sha256 = write_arrival_pose_catalog(args.arrival_pose_catalog, catalog)
    completion = {
        "candidate_uid": candidate_uid,
        "catalog_path": str(args.arrival_pose_catalog),
        "catalog_revision": catalog.revision,
        "catalog_sha256": catalog_sha256,
        "catalog_complete": catalog.complete,
        "arrival_pose": asdict(record.arrival_pose),
        "corridor_entry_pose": asdict(record.corridor_entry_pose),
        "face_id": record.face.face_id,
        "axis_rad": record.axis.axis_rad,
        "axis_confidence": record.axis.confidence,
        "axis_sample_count": record.axis.sample_count,
        "map_yaml_sha256": map_bundle.yaml_sha256,
        "map_image_sha256": map_bundle.image_sha256,
        "map_bundle_sha256": map_bundle.bundle_sha256,
        "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
        "known_stand_keepout_clearances": list(known_stand_clearances),
    }
    # Assert the in-memory digest agrees with the bytes we just published.
    if catalog_sha256 != arrival_pose_catalog_sha256(catalog):
        raise ValueError("arrival-pose catalog digest changed during publication")
    return completion, catalog


def load_recommended_payload(path: Path) -> tuple[Pose2D, Pose2D | None, dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("source") != "synchronized_lidar_camera_viewpoint":
        raise ValueError("recommended viewpoint source mismatch")
    pose = payload.get("pose")
    if not isinstance(pose, dict):
        raise ValueError("recommended viewpoint is missing pose")
    values = (pose.get("x_m"), pose.get("y_m"), pose.get("yaw_rad"))
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("recommended viewpoint pose must be finite")
    target = Pose2D(float(values[0]), float(values[1]), float(values[2]))
    robot = payload.get("robot_pose")
    start = None
    if isinstance(robot, dict):
        robot_values = (robot.get("x_m"), robot.get("y_m"), robot.get("yaw_rad"))
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in robot_values):
            start = Pose2D(*(float(value) for value in robot_values))
    return target, start, payload


def load_recommended_pose(path: Path) -> Pose2D:
    return load_recommended_payload(path)[0]


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _raw_face_mapping(recommendation, config: DynamicApproachConfig) -> tuple[dict[int, str], int | None]:
    stable_faces = recommendation.face_candidates
    stand_axis = stable_faces[0].outward_normal_rad - math.pi / 2.0
    raw_faces = planner_face_candidates(recommendation.stand.center, stand_axis, config)
    direct = angular_distance(raw_faces[0].normal_rad, stable_faces[0].outward_normal_rad) + angular_distance(
        raw_faces[1].normal_rad, stable_faces[1].outward_normal_rad
    )
    swapped = angular_distance(raw_faces[0].normal_rad, stable_faces[1].outward_normal_rad) + angular_distance(
        raw_faces[1].normal_rad, stable_faces[0].outward_normal_rad
    )
    ordered = stable_faces if direct <= swapped else (stable_faces[1], stable_faces[0])
    face_id_by_raw = {0: ordered[0].face_id, 1: ordered[1].face_id}
    hard_raw = None
    evidence = recommendation.side_evidence
    if evidence.hard and evidence.valid:
        if evidence.face_id is None:
            raise ValueError("hard side evidence has no physical face ID")
        hard_face = next(face for face in stable_faces if face.face_id == evidence.face_id)
        if not hard_face.identity_resolved:
            raise ValueError("hard side evidence references unresolved face identity")
        hard_raw = next(index for index, face_id in face_id_by_raw.items() if face_id == evidence.face_id)
    return face_id_by_raw, hard_raw


def _route_csv_text(
    costmap,
    plan,
    *,
    stream_id: str,
    target_revision: int,
    simulation_only: bool = True,
    route_kind: str = "synchronized_viewpoint",
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "leg_index",
            "point_index",
            "grid_x",
            "grid_y",
            "world_x_m",
            "world_y_m",
            "yaw_rad",
            "segment_length_m",
            "cumulative_length_m",
            "protected",
            "corridor",
            "simulation_only",
            "route_kind",
            "stream_id",
            "route_revision",
            "target_revision",
            "manifest_path",
        ]
    )
    cumulative = 0.0
    previous = None
    for index, waypoint in enumerate(plan.waypoints):
        pose = waypoint.pose
        segment = 0.0 if previous is None else math.hypot(
            pose.x_m - previous.x_m, pose.y_m - previous.y_m
        )
        cumulative += segment
        cell = costmap.world_to_grid(pose)
        writer.writerow(
            [
                0,
                index,
                cell.x,
                cell.y,
                pose.x_m,
                pose.y_m,
                "" if not math.isfinite(pose.yaw_rad) else pose.yaw_rad,
                segment,
                cumulative,
                str(waypoint.protected).lower(),
                str(waypoint.corridor).lower(),
                str(simulation_only).lower(),
                route_kind,
                stream_id,
                "",
                target_revision,
                "",
            ]
        )
        previous = pose
    return output.getvalue()


def _diagnostics_payload(
    plan_result,
    recommendation,
    target_revision: int,
    *,
    map_bundle: FrozenMapBundle | None = None,
    candidate_snapshot_sha256: str = "",
) -> dict:
    plan = plan_result.plan
    return {
        "metadata": {
            "stage": "dynamic_synchronized_viewpoint_refinement",
            "simulation_only": recommendation.simulation_only,
            "stream_id": recommendation.stream_id,
            "stand_id": recommendation.stand_id,
            "planning_frame": recommendation.planning_frame,
            "target_revision": target_revision,
            "source": recommendation.source,
            "approach_phase": recommendation.axis_state,
            **(
                {}
                if map_bundle is None
                else {
                    "map_yaml_sha256": map_bundle.yaml_sha256,
                    "map_image_sha256": map_bundle.image_sha256,
                    "map_bundle_sha256": map_bundle.bundle_sha256,
                    "candidate_snapshot_sha256": candidate_snapshot_sha256,
                }
            ),
        },
        "legs": [
            {
                "diagnostics": {
                    "status": "ok" if plan is not None else "failed",
                    "reason": plan_result.diagnostics.failure_reason or "",
                    "route_length_m": 0.0 if plan is None else plan.length_m,
                    "dynamic_approach": asdict(plan_result.diagnostics),
                },
                "failure": (
                    None
                    if plan is not None
                    else {"reason": plan_result.diagnostics.failure_reason or "planning_failed"}
                ),
                "route_length_m": None if plan is None else plan.length_m,
                "route_point_count": 0 if plan is None else len(plan.waypoints),
            }
        ],
    }


def _publish_compatibility_aliases(args, committed) -> None:
    if committed.route_path is not None:
        _atomic_bytes(args.route_csv, committed.route_path.read_bytes())
    if committed.diagnostics_path is not None:
        _atomic_bytes(args.diagnostics_json, committed.diagnostics_path.read_bytes())


def _restart_state(existing) -> DynamicReplanState:
    """Restore the material target represented by an existing manifest."""

    if existing is None:
        return DynamicReplanState()
    revision = existing.target_revision
    target_payload = existing.manifest.get("target", {})
    if revision == 0 and not target_payload:
        return DynamicReplanState()
    if not isinstance(target_payload, dict):
        raise ValueError("existing route manifest target is malformed")
    try:
        pose = Pose2D(
            float(target_payload["x_m"]),
            float(target_payload["y_m"]),
            float(target_payload["yaw_rad"]),
        )
        face_id = str(target_payload["face_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("existing route manifest target is incomplete") from exc
    if not face_id or not all(math.isfinite(value) for value in asdict(pose).values()):
        raise ValueError("existing route manifest target is not finite")
    evidence_state = str(target_payload.get("evidence_state", ""))
    if not evidence_state:
        evidence = existing.manifest.get("evidence", {})
        evidence_state = (
            "hard_qr"
            if isinstance(evidence, dict)
            and evidence.get("hard") is True
            and evidence.get("valid") is True
            and evidence.get("face_id") == face_id
            else "ambiguous_axis"
        )
    current_target = MaterialTarget(face_id, pose, evidence_state)

    planned_start = None
    start_payload = existing.manifest.get("source_robot_pose", {})
    if isinstance(start_payload, dict) and start_payload:
        try:
            candidate = Pose2D(
                float(start_payload["x_m"]),
                float(start_payload["y_m"]),
                float(start_payload["yaw_rad"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("existing route manifest source_robot_pose is malformed") from exc
        if not all(math.isfinite(value) for value in asdict(candidate).values()):
            raise ValueError("existing route manifest source_robot_pose is not finite")
        planned_start = candidate

    active = existing.status == "active"
    return DynamicReplanState(
        target_revision=revision,
        current_target=current_target,
        last_observed_time_sec=float(existing.manifest["published_unix_sec"]),
        last_route_plan_time_sec=(
            float(existing.manifest["published_unix_sec"]) if active else None
        ),
        last_planned_start=planned_start if active else None,
        last_planned_target_revision=revision if active else 0,
    )


def _active_publication(existing) -> dict | None:
    """Load the immutable geometry retained for heartbeat-only revisions."""

    if existing is None or existing.status != "active":
        return None
    if existing.route_path is None or existing.diagnostics_path is None:
        raise ValueError("existing active route is missing immutable artifacts")
    manifest = existing.manifest
    required_mappings = {}
    for name in ("source_robot_pose", "target", "evidence", "safety_diagnostics"):
        value = manifest.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"existing route manifest {name} is malformed")
        required_mappings[name] = dict(value)
    diagnostics = json.loads(existing.diagnostics_path.read_text())
    if not isinstance(diagnostics, dict):
        raise ValueError("existing route diagnostics must be an object")
    with existing.route_path.open("r", newline="") as route_file:
        route_text = route_file.read()
    return {
        "route_text": route_text,
        "diagnostics": diagnostics,
        **required_mappings,
        "route_length_m": float(manifest.get("new_route_length_m", 0.0)),
    }


def _axis_acquisition_rejection_feedback(
    *,
    path: Path,
    recommendation,
    active_publication: Mapping[str, object],
    point_approach_attempts: Sequence[Mapping[str, object]],
    arrival_tolerance_m: float,
    max_search_targets: int,
    now_unix_sec: float,
    max_age_sec: float,
) -> dict[str, object]:
    """Write or retain one exact, monotonic, observer-only rejection event.

    This function deliberately does not publish a route heartbeat. The motion
    side can therefore adopt no geometry from this sidecar: the previously
    certified active manifest remains byte-for-byte untouched while the
    observer advances to the next bounded survey-only ray.
    """

    binding = axis_acquisition_feedback_binding(recommendation)
    search_index = int(binding["search_index"])
    if search_index + 1 >= max_search_targets:
        raise ValueError(
            "axis_acquisition_search_exhausted:"
            + ",".join(
                f"{attempt.get('face_id')}="
                f"{dict(attempt.get('diagnostics', {})).get('failure_reason')}"
                for attempt in point_approach_attempts
            )
        )
    if not math.isfinite(arrival_tolerance_m) or arrival_tolerance_m <= 0.0:
        raise ValueError("axis acquisition arrival tolerance is invalid")
    if not math.isfinite(max_age_sec) or max_age_sec <= 0.0:
        raise ValueError("axis acquisition feedback max age is invalid")

    active_target = active_publication.get("target")
    active_evidence = active_publication.get("evidence")
    if not isinstance(active_target, Mapping) or not isinstance(
        active_evidence, Mapping
    ):
        raise ValueError("axis acquisition feedback active route is malformed")
    if active_target.get("evidence_state") != "axis_acquisition":
        raise ValueError("axis acquisition feedback active route is not unresolved")
    active_face_id = active_target.get("face_id")
    if (
        not isinstance(active_face_id, str)
        or not active_face_id.startswith(
            ("acquisition_near_", "acquisition_far_")
        )
        or active_evidence.get("hard") is True
        or active_evidence.get("valid") is True
    ):
        raise ValueError("axis acquisition feedback active route has side identity")
    active_pose = finite_feedback_pose(
        active_target,
        name="axis acquisition feedback active target",
    )
    hold_distance_m = math.hypot(
        recommendation.robot_pose.x_m - active_pose["x_m"],
        recommendation.robot_pose.y_m - active_pose["y_m"],
    )
    if (
        not math.isfinite(hold_distance_m)
        or hold_distance_m > arrival_tolerance_m
    ):
        raise ValueError("axis acquisition feedback active target is not safely held")

    candidates = binding["face_candidates"]
    if len(point_approach_attempts) != 2:
        raise ValueError("axis acquisition feedback requires two planner attempts")
    rejections = []
    for attempt, candidate in zip(point_approach_attempts, candidates):
        if not isinstance(attempt, Mapping):
            raise ValueError("axis acquisition planner attempt is malformed")
        diagnostics = attempt.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("axis acquisition planner diagnostics are malformed")
        reason = diagnostics.get("failure_reason")
        if (
            attempt.get("face_id") != candidate["face_id"]
            or reason not in AXIS_ACQUISITION_STATIC_TARGET_FAILURES
        ):
            raise ValueError(
                "axis acquisition rejection is not an allowlisted static target failure"
            )
        rejections.append(
            {
                "face_id": str(attempt["face_id"]),
                "failure_reason": str(reason),
            }
        )

    binding_sha256 = canonical_json_sha256(binding)
    if path.exists():
        existing = load_axis_acquisition_feedback(
            path,
            now_unix_sec=now_unix_sec,
            max_age_sec=max_age_sec,
        )
        existing_index = int(existing["binding"]["search_index"])
        if existing_index > search_index:
            raise ValueError("axis acquisition feedback index moved backwards")
        if existing_index == search_index:
            if existing["binding_sha256"] != binding_sha256:
                raise ValueError("axis acquisition feedback binding mismatch")
            # Pending duplicate polls and the short consumed-to-next-observation
            # race retain the exact same event without rewriting or revisioning.
            return existing
        if existing["state"] != "consumed":
            raise ValueError(
                "older axis acquisition feedback was not consumed monotonically"
            )

    payload = {
        "schema_version": AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION,
        "contract": AXIS_ACQUISITION_FEEDBACK_CONTRACT,
        "simulation_only": True,
        "state": "pending",
        "created_unix_sec": now_unix_sec,
        "source_observation_unix_sec": (
            recommendation.observation_unix_sec
        ),
        "source_sensor_stamp_sec": recommendation.sensor_stamp_sec,
        "arrival_tolerance_m": arrival_tolerance_m,
        "binding": binding,
        "binding_sha256": binding_sha256,
        "held_active_target": {
            "face_id": active_face_id,
            "evidence_state": "axis_acquisition",
            "pose": active_pose,
        },
        "held_distance_m": hold_distance_m,
        "rejections": rejections,
    }
    return write_axis_acquisition_feedback(path, payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument(
        "--environment",
        choices=("simulation", "real"),
        default="simulation",
        help=(
            "Evidence environment. Real mode is observe-only and supports only "
            "--workflow-mode survey-only without --watch."
        ),
    )
    parser.add_argument("--start-x", required=True, type=float)
    parser.add_argument("--start-y", required=True, type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument(
        "--start-from-recommendation",
        action="store_true",
        help=(
            "Use the exact passively observed map-frame robot pose as the "
            "one-shot validation start. Required for real survey evidence."
        ),
    )
    parser.add_argument("--recommended-pose-json", required=True, type=Path)
    parser.add_argument("--route-csv", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--route-manifest", type=Path, default=None)
    parser.add_argument("--stream-id", default="sim-stand-viewpoint")
    parser.add_argument("--writer-id", default="synchronized-viewpoint-planner")
    parser.add_argument("--writer-takeover", action="store_true")
    parser.add_argument(
        "--workflow-mode",
        choices=_WORKFLOW_MODES,
        default=WORKFLOW_IMMEDIATE_APPROACH,
        help=(
            "immediate-approach preserves the legacy live physical replan; "
            "survey-only records the committed perpendicular pose and stops"
        ),
    )
    parser.add_argument(
        "--arrival-pose-catalog",
        type=Path,
        default=Path(
            "results/aufgabe04/detected_stations/arrival_pose_catalog.json"
        ),
    )
    parser.add_argument("--catalog-id", default="sim_arrival_survey")
    parser.add_argument("--candidate-uid", default="")
    parser.add_argument(
        "--expected-candidate-uid",
        action="append",
        default=[],
        help="Expected stable candidate UID; repeat once per candidate.",
    )
    parser.add_argument("--world-id", default="")
    parser.add_argument(
        "--world-sha256",
        default="",
        help="SHA-256 of the randomized world/layout artifact.",
    )
    parser.add_argument("--session-id", default="")
    parser.add_argument("--axis-sample-count", type=int, default=7)
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument(
        "--expected-map-bundle-sha256",
        default="",
        help="Fail if the map YAML+image snapshot differs from this digest.",
    )
    parser.add_argument(
        "--candidate-snapshot-sha256",
        default="",
        help="Immutable candidate snapshot bound to this survey attempt.",
    )
    parser.add_argument("--station-identity-registry-sha256", default="")
    parser.add_argument("--survey-config-sha256", default="")
    parser.add_argument("--calibration-profile-sha256", default="")
    parser.add_argument("--survey-input-binding-sha256", default="")
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument(
        "--arena-center-x-m",
        type=float,
        default=ArenaBounds.center_x_m,
    )
    parser.add_argument(
        "--arena-center-y-m",
        type=float,
        default=ArenaBounds.center_y_m,
    )
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    parser.add_argument("--robot-radius-m", type=float, default=0.105)
    parser.add_argument("--tracking-margin-m", type=float, default=0.03)
    parser.add_argument("--collision-margin-m", type=float, default=0.02)
    parser.add_argument("--inflation-radius-m", type=float, default=None)
    parser.add_argument(
        "--known-stand-keepout",
        action="append",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "RADIUS_M"),
        default=[],
        help=(
            "Known stand center and total robot-center exclusion radius in map "
            "coordinates; repeat once per known stand. No keepouts are added "
            "unless this option is supplied."
        ),
    )
    parser.add_argument("--standoff-distance-m", type=float, default=None)
    parser.add_argument("--terminal-corridor-length-m", type=float, default=0.40)
    parser.add_argument("--corridor-sample-spacing-m", type=float, default=0.05)
    parser.add_argument("--lidar-stop-distance-m", type=float, default=0.18)
    parser.add_argument("--scan-origin-to-base-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-clearance-margin-m", type=float, default=0.02)
    parser.add_argument("--max-recommendation-age-sec", type=float, default=3.0)
    parser.add_argument(
        "--axis-acquisition-feedback-json",
        type=Path,
        default=None,
        help=(
            "Simulation survey observer/planner sidecar for one exact static "
            "axis-acquisition target rejection. It is never a motion input."
        ),
    )
    parser.add_argument(
        "--axis-acquisition-feedback-max-age-sec",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--axis-acquisition-arrival-tolerance-m",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--axis-acquisition-search-max-targets",
        type=int,
        default=7,
    )
    parser.add_argument("--target-position-threshold-m", type=float, default=0.06)
    parser.add_argument("--target-yaw-threshold-deg", type=float, default=10.0)
    parser.add_argument("--start-deviation-threshold-m", type=float, default=0.15)
    parser.add_argument(
        "--replan-on-start-deviation",
        action="store_true",
        help=(
            "Legacy behavior: replace route geometry when the moving robot leaves "
            "the last planned start. By default only material target changes replan."
        ),
    )
    parser.add_argument("--refresh-timeout-sec", type=float, default=4.0)
    parser.add_argument(
        "--terminal-route-lock-distance-m",
        type=float,
        default=0.42,
        help=(
            "Keep the installed collision-checked terminal corridor inside this "
            "target distance unless the material camera target changes."
        ),
    )
    parser.add_argument("--snap-radius-m", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--watch", action="store_true", help="Continuously replan from observer robot/target poses.")
    parser.add_argument("--replan-rate-hz", type=float, default=0.75)
    parser.add_argument("--max-replans", type=int, default=0, help="Stop after N plans; zero means run until interrupted.")
    args = parser.parse_args(argv)
    if args.environment == "real":
        if args.workflow_mode != WORKFLOW_SURVEY_ONLY:
            parser.error("real environment supports only --workflow-mode survey-only")
        if args.watch:
            parser.error("real survey validation is one-shot; --watch is simulation-only")
        if args.axis_acquisition_feedback_json is not None:
            parser.error(
                "--axis-acquisition-feedback-json is simulation-only and cannot "
                "be used for a real survey"
            )
        if not args.start_from_recommendation:
            parser.error(
                "real survey validation requires --start-from-recommendation"
            )
    if args.snap_radius_m != 0.0:
        parser.error("dynamic corridor entry snapping is forbidden; --snap-radius-m must be 0")
    if args.max_replans < 0:
        parser.error("--max-replans must be non-negative")
    if args.axis_sample_count < 1:
        parser.error("--axis-sample-count must be positive")
    if args.workflow_mode == WORKFLOW_SURVEY_ONLY:
        if args.axis_sample_count < 7:
            parser.error("survey-only requires --axis-sample-count >= 7")
        required_survey_values = {
            "--candidate-uid": args.candidate_uid,
            "--world-id": args.world_id,
            "--world-sha256": args.world_sha256,
            "--session-id": args.session_id,
        }
        missing = [name for name, value in required_survey_values.items() if not value]
        if missing:
            parser.error(
                "survey-only requires explicit provenance: " + ", ".join(missing)
            )
        if not args.expected_candidate_uid:
            parser.error(
                "survey-only requires the complete candidate set via "
                "--expected-candidate-uid"
            )
        if args.candidate_uid not in args.expected_candidate_uid:
            parser.error("--candidate-uid must occur in --expected-candidate-uid")
        if len(args.world_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in args.world_sha256
        ):
            parser.error("--world-sha256 must be 64 lowercase hexadecimal characters")
        for option, digest in (
            ("--expected-map-bundle-sha256", args.expected_map_bundle_sha256),
            ("--candidate-snapshot-sha256", args.candidate_snapshot_sha256),
        ):
            if not digest or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                parser.error(
                    f"survey-only requires {option} as 64 lowercase hexadecimal characters"
                )
    if (
        any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                args.max_recommendation_age_sec,
                args.refresh_timeout_sec,
                args.terminal_route_lock_distance_m,
                args.axis_acquisition_feedback_max_age_sec,
                args.axis_acquisition_arrival_tolerance_m,
            )
        )
    ):
        parser.error("freshness and refresh timeouts must be positive")
    if args.axis_acquisition_search_max_targets < 1:
        parser.error("--axis-acquisition-search-max-targets must be positive")
    args.route_manifest = args.route_manifest or args.route_csv.with_suffix(".manifest.json")
    if args.axis_acquisition_feedback_json is not None:
        feedback_path = args.axis_acquisition_feedback_json.resolve()
        protected_paths = {
            args.recommended_pose_json.resolve(),
            args.route_csv.resolve(),
            args.diagnostics_json.resolve(),
            args.route_manifest.resolve(),
            args.arrival_pose_catalog.resolve(),
        }
        if feedback_path in protected_paths:
            parser.error(
                "--axis-acquisition-feedback-json must be a distinct "
                "observer/planner-only sidecar"
            )
    try:
        arena_bounds = ArenaBounds(
            length_m=args.arena_length_m,
            width_m=args.arena_width_m,
            center_x_m=args.arena_center_x_m,
            center_y_m=args.arena_center_y_m,
            yaw_deg=args.arena_yaw_deg,
            margin_m=args.arena_margin_m,
        )
        arena_bounds.validate()
        frozen_grid, map_bundle = load_occupancy_grid_with_bundle(
            args.map,
            semantic_map_id=args.semantic_map_id or args.map.stem,
            planning_frame=args.map_frame,
        )
        if (
            args.expected_map_bundle_sha256
            and map_bundle.bundle_sha256 != args.expected_map_bundle_sha256
        ):
            raise ValueError("occupancy map bundle differs from survey manifest")
        base_costmap = Costmap.from_occupancy_grid(
            frozen_grid
        ).with_arena_bounds(arena_bounds)
        required_static_inflation = minimum_static_obstacle_inflation_m(
            robot_radius_m=args.robot_radius_m,
            tracking_margin_m=args.tracking_margin_m,
            lidar_stop_distance_m=args.lidar_stop_distance_m,
            scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
            lidar_clearance_margin_m=args.lidar_clearance_margin_m,
        )
        static_inflation = (
            required_static_inflation
            if args.inflation_radius_m is None
            else args.inflation_radius_m
        )
        if not math.isfinite(static_inflation) or static_inflation <= 0.0:
            raise ValueError(
                "configuration-space inflation radius must be finite and positive"
            )
        if (
            static_inflation + 1.0e-12
            < required_static_inflation
        ):
            raise ValueError(
                "configuration-space inflation must cover both the robot body "
                "and live LiDAR stop distance along the certified tracking tube"
            )
        base_costmap = base_costmap.with_inflation(static_inflation)
        # Validate once up front, but delay the raster overlay until the exact
        # recommendation start is known inside the watch loop.
        known_stand_keepouts = _normalize_known_stand_keepouts(
            args.known_stand_keepout
        )
        store = RouteRevisionStore(
            args.route_manifest,
            stream_id=args.stream_id,
            writer_id=args.writer_id,
            # Use the planner's clock lookup rather than the store module's
            # import-time default so restart and test clock domains match.
            now_fn=lambda: time.time(),
        )
        existing = None
        if args.route_manifest.exists():
            existing = read_route_revision(
                args.route_manifest,
                expected_stream_id=args.stream_id,
            )
        policy = DynamicReplanPolicy(
            target_position_threshold_m=args.target_position_threshold_m,
            target_yaw_threshold_rad=math.radians(args.target_yaw_threshold_deg),
            start_deviation_threshold_m=args.start_deviation_threshold_m,
            refresh_timeout_sec=args.refresh_timeout_sec,
            terminal_route_lock_distance_m=args.terminal_route_lock_distance_m,
            replan_on_start_deviation=args.replan_on_start_deviation,
        )
        state = _restart_state(existing)
        active_publication = _active_publication(existing)
        invalid_active_clearance_reason = None
        if active_publication is not None:
            active_safety = active_publication["safety_diagnostics"]
            try:
                active_static_inflation = float(
                    active_safety["static_inflation_radius_m"]
                )
            except (KeyError, TypeError, ValueError):
                active_static_inflation = float("nan")
            if active_safety.get("arena_bounds") != arena_bounds.to_metadata():
                invalid_active_clearance_reason = (
                    "active_route_arena_bounds_differ_from_current_requirement"
                )
            elif (
                not math.isfinite(active_static_inflation)
                or active_static_inflation + 1.0e-12
                < static_inflation
            ):
                invalid_active_clearance_reason = (
                    "active_route_static_clearance_below_current_requirement"
                )
            if invalid_active_clearance_reason is not None:
                withdrawn = store.withdraw(
                    invalid_active_clearance_reason,
                    target_revision=state.target_revision,
                    observation_unix_sec=time.time(),
                    takeover=args.writer_takeover,
                )
                _publish_compatibility_aliases(args, withdrawn)
                active_publication = None
                state = replace(
                    state,
                    last_route_plan_time_sec=None,
                    last_planned_start=None,
                    last_planned_target_revision=0,
                )
        plans = 0
        previous_route_length = (
            0.0
            if existing is None
            else float(existing.manifest.get("new_route_length_m", 0.0))
        )
        last_withdrawal_reason = invalid_active_clearance_reason
        while True:
            now = time.time()
            try:
                recommendation = load_recommendation(
                    args.recommended_pose_json,
                    expected_frame=args.map_frame,
                    expected_source="synchronized_lidar_camera_viewpoint",
                    expected_simulation_only=args.environment == "simulation",
                    now_unix_sec=now,
                    max_age_sec=args.max_recommendation_age_sec,
                )
                if recommendation.stream_id != args.stream_id:
                    raise ValueError("viewpoint recommendation stream_id mismatch")
                if recommendation.axis_state.startswith("invalid_"):
                    raise ValueError(recommendation.axis_state)
                route_kind = route_kind_for_axis_state(
                    recommendation.axis_state,
                    args.workflow_mode,
                )
                start = (
                    recommendation.robot_pose
                    if args.watch or args.start_from_recommendation
                    else Pose2D(args.start_x, args.start_y, args.start_yaw)
                )
                point_targets = None
                point_policy_target = None
                state_before_preplan = state
                preplan_decision = None
                if (
                    recommendation.axis_state in _POINT_APPROACH_AXIS_STATES
                    and active_publication is not None
                ):
                    point_targets = _point_approach_targets(
                        recommendation,
                        active_publication,
                    )
                    point_policy_target = point_targets[0]
                    # A stable point-acquisition target owns immutable route
                    # geometry between material target changes.  Decide
                    # whether geometry is needed before attempting a live-pose
                    # A* plan: a conservative raster halo may temporarily
                    # surround the moving robot even though the installed
                    # route remains continuously certified and executable.
                    state, preplan_decision = policy.evaluate(
                        state,
                        target=point_policy_target,
                        robot_pose=start,
                        now_sec=now,
                    )
                    if preplan_decision.fail_closed:
                        raise ValueError(";".join(preplan_decision.reasons))
                    if not preplan_decision.should_replan:
                        if not args.watch:
                            return 0
                        time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
                        continue
                    heartbeat_without_geometry = (
                        not preplan_decision.target_changed
                        and set(preplan_decision.reasons) == {"refresh_timeout"}
                    )
                    if heartbeat_without_geometry:
                        rec_payload = recommendation_to_dict(recommendation)
                        committed = store.publish_active(
                            active_publication["route_text"],
                            active_publication["diagnostics"],
                            target_revision=preplan_decision.target_revision,
                            observation_unix_sec=(
                                recommendation.observation_unix_sec
                            ),
                            source_robot_pose=(
                                active_publication["source_robot_pose"]
                            ),
                            target=active_publication["target"],
                            evidence=rec_payload["side_evidence"],
                            previous_route_length_m=(
                                active_publication["route_length_m"]
                            ),
                            new_route_length_m=(
                                active_publication["route_length_m"]
                            ),
                            safety_diagnostics=(
                                active_publication["safety_diagnostics"]
                            ),
                            takeover=args.writer_takeover,
                        )
                        _publish_compatibility_aliases(args, committed)
                        state = policy.mark_route_planned(
                            state,
                            planned_start=Pose2D(
                                float(
                                    active_publication["source_robot_pose"][
                                        "x_m"
                                    ]
                                ),
                                float(
                                    active_publication["source_robot_pose"][
                                        "y_m"
                                    ]
                                ),
                                float(
                                    active_publication["source_robot_pose"][
                                        "yaw_rad"
                                    ]
                                ),
                            ),
                            now_sec=now,
                            target_revision=preplan_decision.target_revision,
                        )
                        last_withdrawal_reason = None
                        if not args.watch:
                            return 0
                        time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
                        continue
                known_stand_overlay = _known_stand_keepout_costmap(
                    base_costmap,
                    args.known_stand_keepout,
                    start=start,
                )
                planning_costmap = known_stand_overlay.costmap
                planning_start = known_stand_overlay.egress_anchor or start
                observed_standoff = math.hypot(
                    recommendation.material_target.pose.x_m - recommendation.stand.center.x_m,
                    recommendation.material_target.pose.y_m - recommendation.stand.center.y_m,
                )
                config = DynamicApproachConfig(
                    stand_radius_m=recommendation.stand.radius_m,
                    stand_position_uncertainty_m=recommendation.stand.uncertainty_m,
                    robot_radius_m=args.robot_radius_m,
                    collision_margin_m=args.collision_margin_m,
                    tracking_margin_m=args.tracking_margin_m,
                    standoff_distance_m=(
                        observed_standoff
                        if args.standoff_distance_m is None
                        else args.standoff_distance_m
                    ),
                    terminal_corridor_length_m=args.terminal_corridor_length_m,
                    corridor_sample_spacing_m=args.corridor_sample_spacing_m,
                    lidar_stop_distance_m=args.lidar_stop_distance_m,
                    scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
                    lidar_clearance_margin_m=args.lidar_clearance_margin_m,
                )
                point_approach_attempts = None
                if recommendation.axis_state in _POINT_APPROACH_AXIS_STATES:
                    if point_targets is None:
                        point_targets = _point_approach_targets(
                            recommendation,
                            active_publication,
                        )
                        point_policy_target = point_targets[0]
                    result, effective_target, point_approach_attempts = _plan_point_approach(
                        planning_costmap,
                        planning_start,
                        recommendation.stand.center,
                        point_targets,
                        config=config,
                    )
                elif args.workflow_mode == WORKFLOW_SURVEY_ONLY:
                    completion, _catalog = _record_survey_arrival(
                        args=args,
                        recommendation=recommendation,
                        costmap=planning_costmap,
                        start=start,
                        planning_start=planning_start,
                        known_stand_overlay=known_stand_overlay,
                        config=config,
                        known_stand_keepouts=known_stand_keepouts,
                        map_bundle=map_bundle,
                        now=now,
                    )
                    current = (
                        read_route_revision(
                            args.route_manifest,
                            expected_stream_id=args.stream_id,
                            verify_artifacts=False,
                        )
                        if args.route_manifest.exists()
                        else None
                    )
                    if current is not None:
                        terminal_target_revision = (
                            current.target_revision
                            if current.status == "survey_complete"
                            else current.target_revision + 1
                        )
                        terminal = store.complete_survey(
                            "arrival pose recorded",
                            completion=completion,
                            target_revision=terminal_target_revision,
                            observation_unix_sec=recommendation.observation_unix_sec,
                            takeover=args.writer_takeover,
                        )
                        completion["route_revision"] = terminal.route_revision
                        completion["route_sha256"] = terminal.route_hash
                    elif args.watch:
                        raise ValueError(
                            "survey-only completion has no active acquisition route"
                        )
                    print(json.dumps(completion, indent=2, sort_keys=True))
                    return 0
                else:
                    face_id_by_raw, hard_raw_face = _raw_face_mapping(recommendation, config)
                    if hard_raw_face is None:
                        # A committed silhouette axis resolves the physical
                        # side currently facing the robot even when QR identity
                        # is still unknown. Never replace it with the opposite
                        # side merely because that route is shorter.
                        selected_physical = [
                            raw_id
                            for raw_id, stable_id in face_id_by_raw.items()
                            if stable_id == recommendation.material_target.face_id
                        ]
                        if selected_physical:
                            hard_raw_face = selected_physical[0]
                    if hard_raw_face is None and state.current_target is not None:
                        # Robot motion alone must not switch an ambiguous physical
                        # face merely because the other face became Euclidean-
                        # shorter. The observer/QR evidence owns face changes.
                        matching = [
                            raw_id
                            for raw_id, stable_id in face_id_by_raw.items()
                            if stable_id == state.current_target.face_id
                        ]
                        if matching:
                            hard_raw_face = matching[0]
                    stand_axis = recommendation.face_candidates[0].outward_normal_rad - math.pi / 2.0
                    result = plan_dynamic_approach(
                        planning_costmap,
                        planning_start,
                        recommendation.stand.center,
                        stand_axis,
                        hard_face_id=hard_raw_face,
                        config=config,
                    )
                    if result.plan is not None:
                        selected_face_id = face_id_by_raw[result.plan.selected_face_id]
                        effective_target = MaterialTarget(
                            face_id=selected_face_id,
                            pose=result.plan.target,
                            evidence_state=recommendation.material_target.evidence_state,
                        )
                result = _prepend_certified_known_stand_egress(
                    result,
                    source_start=start,
                    overlay=known_stand_overlay,
                    target_stand=recommendation.stand.center,
                    target_keepout_radius_m=config.stand_keepout_radius_m,
                )
                if result.plan is None:
                    if point_approach_attempts is not None:
                        failures = ",".join(
                            f"{attempt['face_id']}="
                            f"{attempt['diagnostics']['failure_reason']}"
                            for attempt in point_approach_attempts
                        )
                        if (
                            recommendation.axis_state == "axis_acquisition"
                            and args.workflow_mode == WORKFLOW_SURVEY_ONLY
                            and args.watch
                            and args.axis_acquisition_feedback_json is not None
                            and active_publication is not None
                        ):
                            try:
                                _axis_acquisition_rejection_feedback(
                                    path=args.axis_acquisition_feedback_json,
                                    recommendation=recommendation,
                                    active_publication=active_publication,
                                    point_approach_attempts=point_approach_attempts,
                                    arrival_tolerance_m=(
                                        args
                                        .axis_acquisition_arrival_tolerance_m
                                    ),
                                    max_search_targets=(
                                        args
                                        .axis_acquisition_search_max_targets
                                    ),
                                    now_unix_sec=now,
                                    max_age_sec=(
                                        args
                                        .axis_acquisition_feedback_max_age_sec
                                    ),
                                )
                            except ValueError as feedback_error:
                                feedback_reason = str(feedback_error)
                                if feedback_reason.startswith(
                                    "axis_acquisition_search_exhausted:"
                                ):
                                    raise
                                raise ValueError(
                                    "point_approach_candidates_exhausted:"
                                    + failures
                                    + ";axis_acquisition_feedback_rejected:"
                                    + feedback_reason
                                ) from feedback_error
                            # Policy evaluation tentatively revisioned the
                            # rejected target. Restore it exactly and leave the
                            # active route manifest untouched: the follower can
                            # neither adopt the sidecar nor mistake it for an
                            # unchanged-geometry heartbeat.
                            state = state_before_preplan
                            time.sleep(
                                1.0 / max(args.replan_rate_hz, 0.1)
                            )
                            continue
                        raise ValueError(
                            "point_approach_candidates_exhausted:" + failures
                        )
                    raise ValueError(result.diagnostics.failure_reason or "dynamic planning failed")
                known_stand_clearances = _validate_known_stand_route_clearance(
                    result.plan,
                    known_stand_keepouts,
                )
                # Revision the target the collision-aware planner actually
                # selected.  With ambiguous evidence the shortest valid face
                # may differ from the observer's provisional preference.
                if (
                    preplan_decision is None
                    or effective_target != point_policy_target
                ):
                    evaluation_state = (
                        state
                        if preplan_decision is None
                        else state_before_preplan
                    )
                    state, decision = policy.evaluate(
                        evaluation_state,
                        target=effective_target,
                        robot_pose=start,
                        now_sec=now,
                    )
                else:
                    decision = preplan_decision
                if decision.fail_closed:
                    raise ValueError(";".join(decision.reasons))
                if not decision.should_replan:
                    if not args.watch:
                        return 0
                    time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
                    continue
                heartbeat_only = (
                    active_publication is not None
                    and not decision.target_changed
                    and set(decision.reasons) == {"refresh_timeout"}
                )
                rec_payload = recommendation_to_dict(recommendation)
                if heartbeat_only:
                    committed = store.publish_active(
                        active_publication["route_text"],
                        active_publication["diagnostics"],
                        target_revision=decision.target_revision,
                        observation_unix_sec=recommendation.observation_unix_sec,
                        source_robot_pose=active_publication["source_robot_pose"],
                        target=active_publication["target"],
                        evidence=rec_payload["side_evidence"],
                        previous_route_length_m=active_publication["route_length_m"],
                        new_route_length_m=active_publication["route_length_m"],
                        safety_diagnostics=active_publication["safety_diagnostics"],
                        takeover=args.writer_takeover,
                    )
                    _publish_compatibility_aliases(args, committed)
                    state = policy.mark_route_planned(
                        state,
                        planned_start=Pose2D(
                            float(active_publication["source_robot_pose"]["x_m"]),
                            float(active_publication["source_robot_pose"]["y_m"]),
                            float(active_publication["source_robot_pose"]["yaw_rad"]),
                        ),
                        now_sec=now,
                        target_revision=decision.target_revision,
                    )
                    last_withdrawal_reason = None
                    time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
                    continue
                route_text = _route_csv_text(
                    planning_costmap,
                    result.plan,
                    stream_id=recommendation.stream_id,
                    target_revision=decision.target_revision,
                    simulation_only=recommendation.simulation_only,
                    route_kind=(
                        route_kind
                    ),
                )
                diagnostics = _diagnostics_payload(
                    result,
                    recommendation,
                    decision.target_revision,
                    map_bundle=map_bundle,
                    candidate_snapshot_sha256=args.candidate_snapshot_sha256,
                )
                target_payload = {
                    **asdict(effective_target.pose),
                    "face_id": effective_target.face_id,
                    "evidence_state": effective_target.evidence_state,
                }
                safety_payload = {
                    **asdict(result.diagnostics),
                    "arena_bounds": arena_bounds.to_metadata(),
                    "arena_boundary_overlay": True,
                    "static_inflation_radius_m": static_inflation,
                    "required_static_inflation_radius_m": (
                        required_static_inflation
                    ),
                    "map_bundle_sha256": map_bundle.bundle_sha256,
                    "candidate_snapshot_sha256": args.candidate_snapshot_sha256,
                    "known_stand_keepouts": list(known_stand_keepouts),
                    "known_stand_keepout_cell_count": (
                        known_stand_overlay.blocked_cell_count
                    ),
                    "known_stand_keepout_rasterized_cell_count": (
                        known_stand_overlay.rasterized_cell_count
                    ),
                    "known_stand_start_cell": (
                        None
                        if known_stand_overlay.start_cell is None
                        else asdict(known_stand_overlay.start_cell)
                    ),
                    "known_stand_start_cell_exempted": (
                        known_stand_overlay.start_cell_exempted
                    ),
                    "known_stand_egress_anchor": (
                        None
                        if known_stand_overlay.egress_anchor is None
                        else asdict(known_stand_overlay.egress_anchor)
                    ),
                    "known_stand_egress_cells": [
                        asdict(cell) for cell in known_stand_overlay.egress_cells
                    ],
                    "known_stand_egress_continuous_clearance_validated": (
                        known_stand_overlay.egress_anchor is not None
                    ),
                    "known_stand_keepout_clearances": list(
                        known_stand_clearances
                    ),
                }
                if point_approach_attempts is not None:
                    safety_payload["point_approach_attempts"] = list(
                        point_approach_attempts
                    )
                committed = store.publish_active(
                    route_text,
                    diagnostics,
                    target_revision=decision.target_revision,
                    observation_unix_sec=recommendation.observation_unix_sec,
                    source_robot_pose=asdict(start),
                    target=target_payload,
                    evidence=rec_payload["side_evidence"],
                    previous_route_length_m=previous_route_length,
                    new_route_length_m=result.plan.length_m,
                    safety_diagnostics=safety_payload,
                    takeover=args.writer_takeover,
                )
                _publish_compatibility_aliases(args, committed)
                previous_route_length = result.plan.length_m
                active_publication = {
                    "route_text": route_text,
                    "diagnostics": diagnostics,
                    "source_robot_pose": asdict(start),
                    "target": target_payload,
                    "evidence": rec_payload["side_evidence"],
                    "safety_diagnostics": safety_payload,
                    "route_length_m": result.plan.length_m,
                }
                state = policy.mark_route_planned(
                    state,
                    planned_start=start,
                    now_sec=now,
                    target_revision=decision.target_revision,
                )
                last_withdrawal_reason = None
                plans += 1
            except (OSError, ValueError, RouteRevisionError, json.JSONDecodeError) as exc:
                reason = str(exc)
                if reason != last_withdrawal_reason:
                    store.withdraw(
                        reason,
                        target_revision=state.target_revision,
                        observation_unix_sec=now,
                        takeover=args.writer_takeover,
                    )
                    last_withdrawal_reason = reason
                if not args.watch:
                    return 1
            if not args.watch or (args.max_replans > 0 and plans >= args.max_replans):
                return 0
            time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RouteRevisionError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
