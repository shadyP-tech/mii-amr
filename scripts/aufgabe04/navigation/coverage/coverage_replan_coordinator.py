"""In-process transient A* recovery for one physical coverage leg.

The coordinator is deliberately ROS-free.  It is called by the already
running waypoint follower only while zero Twist is held.  Each accepted front
blockage creates a run-local obstacle overlay, seals a replacement route to the
same inspection viewpoint, and returns one atomic route update.  It never
records a survey stop or mutates the semantic stand registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.coverage.coverage_escape_geometry import (
    DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD,
    DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD,
    EGRESS_MODE_FORWARD,
    EGRESS_MODE_STRAIGHT_REVERSE,
)
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.coverage.stand_blockage_replan import (
    record_transient_blockage_replan,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    CLEARANCE_LIMITED_MOTION_FLOOR,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import (
    load_route_leg,
    poses_from_waypoints,
)


RECOVERABLE_STOP_REASONS = frozenset(
    {
        "stuck no progress",
        "obstacle too close",
        CLEARANCE_LIMITED_MOTION_FLOOR,
    }
)


def _front_evidence(
    stop_reason: str,
    stop_details: Mapping[str, object],
) -> dict[str, object] | None:
    if stop_reason not in RECOVERABLE_STOP_REASONS:
        return None
    confirmation = stop_details.get("stationary_obstacle_confirmation")
    if (
        not isinstance(confirmation, Mapping)
        or confirmation.get("confirmed") is not True
        or confirmation.get("fail_closed") is not False
    ):
        return None
    distinct_sample_count = confirmation.get("distinct_sample_count")
    thresholds = confirmation.get("thresholds")
    if (
        not isinstance(distinct_sample_count, int)
        or isinstance(distinct_sample_count, bool)
        or not isinstance(thresholds, Mapping)
    ):
        return None
    minimum_samples = thresholds.get("min_distinct_samples")
    if (
        not isinstance(minimum_samples, int)
        or isinstance(minimum_samples, bool)
        or minimum_samples < 3
        or distinct_sample_count < minimum_samples
    ):
        return None
    front = stop_details.get("front_clearance")
    if not isinstance(front, Mapping) or front.get("source") != "front_sector":
        return None
    try:
        clearance_m = float(front["nearest_valid_range_m"])
        bearing_rad = float(front["nearest_valid_bearing_rad"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(clearance_m)
        or clearance_m <= 0.0
        or not math.isfinite(bearing_rad)
    ):
        return None
    return dict(front)


def _load_escape_metadata(
    diagnostics_path: Path,
    waypoints: tuple[Pose2D, ...],
    *,
    tracking_tube_radius_m: float,
    forward_translation_heading_limit_rad: float,
    reverse_connector_alignment_tolerance_rad: float,
) -> dict[str, object]:
    """Bind the sealed handoff to the planner's executable prefix proof."""

    try:
        payload = json.loads(Path(diagnostics_path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid transient replan diagnostics: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("transient replan diagnostics metadata is not an object")
    mode = metadata.get("egress_mode")
    if mode not in {EGRESS_MODE_FORWARD, EGRESS_MODE_STRAIGHT_REVERSE}:
        raise ValueError("transient replan has no executable egress mode")
    try:
        certified_tube_m = float(metadata["tracking_tube_radius_m"])
        certified_forward_limit_rad = float(
            metadata["forward_translation_heading_limit_rad"]
        )
        certified_reverse_tolerance_rad = float(
            metadata["reverse_connector_alignment_tolerance_rad"]
        )
        raw_transition_index = metadata[
            "egress_transition_waypoint_index"
        ]
        raw_forward_index = metadata.get("egress_forward_waypoint_index")
        anchor = metadata["egress_anchor"]
        transition_anchor = metadata["egress_transition_anchor"]
        if not isinstance(anchor, Mapping) or not isinstance(
            transition_anchor,
            Mapping,
        ):
            raise TypeError("egress anchors must be objects")
        anchor_x_m = float(anchor["x_m"])
        anchor_y_m = float(anchor["y_m"])
        transition_x_m = float(transition_anchor["x_m"])
        transition_y_m = float(transition_anchor["y_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"transient replan egress metadata is malformed: {exc}") from exc
    if not isinstance(raw_transition_index, int) or isinstance(
        raw_transition_index,
        bool,
    ):
        raise ValueError("transient replan transition index is not an integer")
    transition_index = raw_transition_index
    if raw_forward_index is None:
        forward_index = None
    elif isinstance(raw_forward_index, int) and not isinstance(
        raw_forward_index,
        bool,
    ):
        forward_index = raw_forward_index
    else:
        raise ValueError("transient replan forward index is not an integer")
    if (
        not math.isfinite(certified_tube_m)
        or abs(certified_tube_m - tracking_tube_radius_m) > 1.0e-12
    ):
        raise ValueError("transient replan tracking tube differs from execution")
    if (
        not math.isfinite(certified_forward_limit_rad)
        or abs(
            certified_forward_limit_rad
            - forward_translation_heading_limit_rad
        )
        > 1.0e-12
    ):
        raise ValueError(
            "transient replan forward-heading limit differs from execution"
        )
    if (
        not math.isfinite(certified_reverse_tolerance_rad)
        or abs(
            certified_reverse_tolerance_rad
            - reverse_connector_alignment_tolerance_rad
        )
        > 1.0e-12
    ):
        raise ValueError(
            "transient replan reverse-alignment tolerance differs from execution"
        )
    if not all(
        math.isfinite(value)
        for value in (
            anchor_x_m,
            anchor_y_m,
            transition_x_m,
            transition_y_m,
        )
    ):
        raise ValueError("transient replan egress anchors are not finite")
    if not 1 <= transition_index < len(waypoints):
        raise ValueError("transient replan transition index is outside the route")
    if math.hypot(
        waypoints[1].x_m - anchor_x_m,
        waypoints[1].y_m - anchor_y_m,
    ) > 1.0e-8:
        raise ValueError("sealed route lost the certified egress anchor")
    if math.hypot(
        waypoints[transition_index].x_m - transition_x_m,
        waypoints[transition_index].y_m - transition_y_m,
    ) > 1.0e-8:
        raise ValueError("sealed route lost the certified transition anchor")
    if mode == EGRESS_MODE_FORWARD:
        if transition_index != 1 or (
            forward_index is not None
            and (forward_index != 2 or forward_index >= len(waypoints))
        ):
            raise ValueError("forward egress transition must be waypoint 1")
    elif (
        transition_index < 2
        or forward_index != transition_index + 1
        or forward_index >= len(waypoints)
    ):
        raise ValueError("straight reverse egress handoff indices are malformed")
    return dict(metadata)


@dataclass
class CoverageReplanCoordinator:
    survey_root: Path
    session_root: Path
    map_yaml: Path
    semantic_map_id: str
    target_viewpoint_id: str
    run_id: str
    coverage_leg_index: int
    robot_radius_m: float
    max_replans: int
    tracking_tube_radius_m: float
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    )
    reverse_connector_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    )
    route_leg_index: int = 0
    command_owner: str = "/aufgabe04_simple_waypoint_follower"
    replan_count: int = 0
    overlay_path: Path | None = None
    adopted_route_hashes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.survey_root = Path(self.survey_root)
        self.session_root = Path(self.session_root)
        self.map_yaml = Path(self.map_yaml)
        if self.coverage_leg_index < 0 or self.route_leg_index < 0:
            raise ValueError("coverage and route leg indices must be non-negative")
        if not self.command_owner.startswith("/"):
            raise ValueError("command_owner must be a namespace-qualified node")
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative")
        if (
            not isinstance(self.replan_count, int)
            or isinstance(self.replan_count, bool)
            or self.replan_count < 0
            or self.replan_count > self.max_replans
        ):
            raise ValueError(
                "replan_count must be an integer inside the cumulative budget"
            )
        if self.replan_count > 0 and self.overlay_path is None:
            raise ValueError("resumed replan_count requires an obstacle overlay")
        if self.replan_count == 0 and self.overlay_path is not None:
            raise ValueError("an obstacle overlay requires a positive replan_count")
        if self.overlay_path is not None:
            self.overlay_path = Path(self.overlay_path)
        if not isinstance(self.adopted_route_hashes, set) or any(
            not isinstance(route_hash, str) or not route_hash
            for route_hash in self.adopted_route_hashes
        ):
            raise ValueError("adopted_route_hashes must be a set of non-empty strings")
        if not math.isfinite(self.robot_radius_m) or self.robot_radius_m <= 0.0:
            raise ValueError("robot_radius_m must be finite and positive")
        if (
            not math.isfinite(self.tracking_tube_radius_m)
            or self.tracking_tube_radius_m <= 0.0
        ):
            raise ValueError(
                "tracking_tube_radius_m must be finite and positive"
            )
        if (
            not math.isfinite(self.forward_translation_heading_limit_rad)
            or self.forward_translation_heading_limit_rad <= 0.0
            or self.forward_translation_heading_limit_rad > math.pi / 2.0
        ):
            raise ValueError(
                "forward_translation_heading_limit_rad must be in (0, pi/2]"
            )
        if (
            not math.isfinite(
                self.reverse_connector_alignment_tolerance_rad
            )
            or self.reverse_connector_alignment_tolerance_rad <= 0.0
            or self.reverse_connector_alignment_tolerance_rad > math.pi / 2.0
        ):
            raise ValueError(
                "reverse_connector_alignment_tolerance_rad must be in "
                "(0, pi/2]"
            )

    def __call__(
        self,
        pose: Pose2D,
        stop_reason: str,
        stop_details: Mapping[str, object],
    ) -> RouteUpdate | None:
        front = _front_evidence(stop_reason, stop_details)
        if front is None:
            return None
        if self.replan_count >= self.max_replans:
            return RouteUpdate(
                kind=RouteUpdateKind.STOP,
                reason="coverage blockage replan budget exhausted",
                event_name="transient_navigation_blockage_budget_exhausted",
                event_fields={
                    "replan_count": self.replan_count,
                    "max_replans": self.max_replans,
                    "original_stop_reason": stop_reason,
                    "target_viewpoint_id": self.target_viewpoint_id,
                    "semantic_survey_evidence": False,
                    "fail_closed": True,
                },
            )

        self.replan_count += 1
        replan_index = self.replan_count
        blockage_id = (
            f"blockage_leg_{self.coverage_leg_index:03d}"
            f"_replan_{replan_index:03d}"
        )
        source_root = (
            self.survey_root
            / "replans"
            / (
                f"leg_{self.coverage_leg_index:03d}"
                f"_replan_{replan_index:03d}"
            )
        )
        artifacts = record_transient_blockage_replan(
            survey_root=self.survey_root,
            map_yaml=self.map_yaml,
            semantic_map_id=self.semantic_map_id,
            target_viewpoint_id=self.target_viewpoint_id,
            blockage_id=blockage_id,
            stop_pose=pose,
            stop_reason=stop_reason,
            stop_details={
                **dict(stop_details),
                "front_clearance": front,
            },
            output_dir=source_root,
            robot_radius_m=self.robot_radius_m,
            existing_overlay_path=self.overlay_path,
            tracking_tube_radius_m=self.tracking_tube_radius_m,
            forward_translation_heading_limit_rad=(
                self.forward_translation_heading_limit_rad
            ),
            reverse_connector_alignment_tolerance_rad=(
                self.reverse_connector_alignment_tolerance_rad
            ),
        )
        self.overlay_path = Path(artifacts["transient_obstacle_overlay_json"])

        execution_root = (
            self.session_root
            / "execution"
            / (
                f"coverage_leg_{self.coverage_leg_index:03d}"
                f"_replan_{replan_index:03d}"
            )
        )
        sealed = seal_stand_discovery_route(
            source_route_csv=Path(artifacts["route_csv"]),
            source_diagnostics_json=Path(artifacts["diagnostics_json"]),
            coverage_plan_path=self.survey_root / "coverage_plan.json",
            output_dir=execution_root,
            command_owner=self.command_owner,
            tracking_tube_radius_m=self.tracking_tube_radius_m,
        )
        leg = load_route_leg(
            Path(sealed["route_csv"]),
            self.route_leg_index,
            thinning_min_spacing_m=0.0,
        )
        waypoints = poses_from_waypoints(leg.executable_waypoints)
        if len(waypoints) < 2:
            raise ValueError("sealed transient replan has fewer than two waypoints")
        if leg.route_kind != STAND_DISCOVERY_ROUTE_KIND:
            raise ValueError("sealed transient replan changed the mission phase")
        if leg.source_sha256 in self.adopted_route_hashes:
            raise ValueError("transient replanner repeated an adopted route")
        self.adopted_route_hashes.add(leg.source_sha256)

        escape_metadata = _load_escape_metadata(
            Path(artifacts["diagnostics_json"]),
            waypoints,
            tracking_tube_radius_m=self.tracking_tube_radius_m,
            forward_translation_heading_limit_rad=(
                self.forward_translation_heading_limit_rad
            ),
            reverse_connector_alignment_tolerance_rad=(
                self.reverse_connector_alignment_tolerance_rad
            ),
        )
        reverse_egress = (
            escape_metadata["egress_mode"] == EGRESS_MODE_STRAIGHT_REVERSE
        )
        event_fields = {
            "replan_index": replan_index,
            "original_stop_reason": stop_reason,
            "target_viewpoint_id": self.target_viewpoint_id,
            "semantic_survey_evidence": False,
            "route_kind": STAND_DISCOVERY_ROUTE_KIND,
            "effective_join_limit_m": self.tracking_tube_radius_m,
            "start_egress_vertex_lock": True,
            "start_egress_waypoint_index": 1,
            "start_egress_continuous_clearance_validated": True,
            "start_egress_motion": "reverse" if reverse_egress else "forward",
            "replacement_route_csv": sealed["route_csv"],
            "source_map_route_sha256": leg.source_sha256,
            "replacement_diagnostics_json": sealed["diagnostics_json"],
            "replacement_route_certificate_json": sealed[
                "route_certificate_json"
            ],
            "transient_obstacle_overlay_json": str(self.overlay_path),
            "front_clearance_m": float(front["nearest_valid_range_m"]),
            "front_bearing_rad": float(front["nearest_valid_bearing_rad"]),
            "stationary_obstacle_confirmation": dict(
                stop_details["stationary_obstacle_confirmation"]
            ),
            "egress_mode": escape_metadata["egress_mode"],
            "egress_transition_waypoint_index": escape_metadata[
                "egress_transition_waypoint_index"
            ],
            "egress_forward_waypoint_index": escape_metadata[
                "egress_forward_waypoint_index"
            ],
            "forward_translation_heading_limit_rad": escape_metadata[
                "forward_translation_heading_limit_rad"
            ],
            "reverse_connector_alignment_tolerance_rad": escape_metadata[
                "reverse_connector_alignment_tolerance_rad"
            ],
            "reverse_connector_heading_error_rad": escape_metadata[
                "reverse_connector_heading_error_rad"
            ],
            "minimum_transition_keepout_tube_clearance_m": escape_metadata[
                "minimum_transition_keepout_tube_clearance_m"
            ],
            "tracking_tube_radius_m": escape_metadata[
                "tracking_tube_radius_m"
            ],
        }
        if reverse_egress:
            reverse_until_index = int(
                escape_metadata["egress_transition_waypoint_index"]
            )
            forward_alignment_index = int(
                escape_metadata["egress_forward_waypoint_index"]
            )
            event_fields.update(
                {
                    "start_egress_reverse_until_waypoint_index": (
                        reverse_until_index
                    ),
                    "start_egress_forward_alignment_waypoint_index": (
                        forward_alignment_index
                    ),
                }
            )
        return RouteUpdate(
            kind=RouteUpdateKind.ADOPT,
            waypoints=waypoints,
            target_index=0,
            route_revision=replan_index,
            route_hash=leg.source_sha256,
            requires_zero_cycle=True,
            event_name="transient_navigation_blockage_replanned",
            event_fields=event_fields,
        )
