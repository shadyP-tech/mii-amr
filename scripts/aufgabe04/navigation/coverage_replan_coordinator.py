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
import time
from typing import Mapping

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.stand_blockage_replan import (
    record_transient_blockage_replan,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.waypoint_csv import (
    load_route_leg,
    poses_from_waypoints,
)


RECOVERABLE_STOP_REASONS = frozenset(
    {"stuck no progress", "obstacle too close"}
)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _front_evidence(
    stop_reason: str,
    stop_details: Mapping[str, object],
) -> dict[str, object] | None:
    if stop_reason not in RECOVERABLE_STOP_REASONS:
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


def _reverse_egress_required(pose: Pose2D, waypoints: tuple[Pose2D, ...]) -> bool:
    if len(waypoints) < 2:
        return False
    egress = waypoints[1]
    heading = math.atan2(egress.y_m - pose.y_m, egress.x_m - pose.x_m)
    error = math.atan2(
        math.sin(heading - pose.yaw_rad),
        math.cos(heading - pose.yaw_rad),
    )
    return abs(error) > math.pi / 2.0


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
        if not math.isfinite(self.robot_radius_m) or self.robot_radius_m <= 0.0:
            raise ValueError("robot_radius_m must be finite and positive")
        if (
            not math.isfinite(self.tracking_tube_radius_m)
            or self.tracking_tube_radius_m <= 0.0
        ):
            raise ValueError(
                "tracking_tube_radius_m must be finite and positive"
            )

    @property
    def adaptive_log_path(self) -> Path:
        return self.session_root / "adaptive_replans.jsonl"

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

        reverse_egress = _reverse_egress_required(pose, waypoints)
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
            "replacement_diagnostics_json": sealed["diagnostics_json"],
            "replacement_route_certificate_json": sealed[
                "route_certificate_json"
            ],
            "transient_obstacle_overlay_json": str(self.overlay_path),
            "front_clearance_m": float(front["nearest_valid_range_m"]),
            "front_bearing_rad": float(front["nearest_valid_bearing_rad"]),
        }
        _append_jsonl(
            self.adaptive_log_path,
            {
                "schema_version": 1,
                "event": "transient_navigation_blockage_replanned",
                "timestamp": time.time(),
                "run_id": self.run_id,
                "leg_index": self.coverage_leg_index,
                **event_fields,
            },
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
