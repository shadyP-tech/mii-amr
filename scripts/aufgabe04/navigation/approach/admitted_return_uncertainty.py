"""Select a stopped prefix without changing the full stored-Start route.

Every evaluation uses the current, unchanged localization anchor. A later
stage requires a new stopped observation; this module grants no motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
    RouteUncertaintyAdmissionResult,
    evaluate_route_uncertainty_admission,
    evaluate_stationary_turn_uncertainty_admission,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    evaluate_route_uncertainty_budget, uncertainty_budget_evidence_sha256,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap


MINIMUM_STAGE_DISPLACEMENT_M = .20
MINIMUM_REMAINING_ROUTE_M = .15
MAXIMUM_SAMPLED_CUTS = 32


class ReturnUncertaintyExhausted(ValueError):
    """A valid route exhausted the unchanged budget; retain audit-only evidence."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _attempt_diagnostic(poses, admission, *, final):
    decision = admission.decision
    entries = decision.segment_decisions
    limiting = next((d for d in entries if d.segment_id == decision.limiting_segment_id), None)
    first = next((d for d in entries if not d.accepted), None)
    ids = {d.segment_id for d in (limiting, first) if d is not None}
    samples = [s for s in admission.evidence.get("sampling", {}).get("segments", ())
               if f"segment:{s['canonical_index']:04d}:{s['subsegment_index']:04d}" in ids]
    return {"final_stage": final, "endpoint": asdict(poses[-1]),
        "route_geometry_sha256": payload_sha256(return_route_geometry(poses)),
        "accepted": decision.accepted, "remaining_margin_m": decision.remaining_margin_m,
        "limiting_segment_id": decision.limiting_segment_id,
        "limiting_segment": None if limiting is None else limiting.to_evidence_dict(),
        "first_rejected_segment": None if first is None else first.to_evidence_dict(),
        "limiting_samples": samples}


@dataclass(frozen=True)
class ReturnRouteStage:
    poses: tuple[Pose2D, ...]
    is_final_stage: bool
    stage_target_pose: Pose2D
    end_segment_index: int
    end_fraction: float
    evidence: dict[str, object]


def executable_return_poses(poses: tuple[Pose2D, ...]) -> tuple[Pose2D, ...]:
    """Match CSV: transit headings are unconstrained, terminal yaw is exact."""
    return tuple(replace(p, yaw_rad=math.nan) for p in poses[:-1]) + poses[-1:]


def return_route_geometry(poses: tuple[Pose2D, ...]) -> dict[str, object]:
    return {"poses": [asdict(p) for p in poses]}


def build_return_prefix(
    full_poses: tuple[Pose2D, ...], end_segment_index: int, end_fraction: float,
) -> tuple[Pose2D, ...]:
    """Reconstruct the sole permitted endpoint from the original polyline."""
    if (
        isinstance(end_segment_index, bool) or not isinstance(end_segment_index, int)
        or not 0 <= end_segment_index < len(full_poses) - 1
        or isinstance(end_fraction, bool) or not isinstance(end_fraction, (int, float))
        or not math.isfinite(end_fraction) or not 0. < end_fraction <= 1.
    ):
        raise ValueError("invalid return prefix position")
    i, t = end_segment_index, end_fraction
    a, b = full_poses[i:i + 2]
    if t == 1.:
        end = b
        if i + 2 < len(full_poses):
            # Stop aligned with the arriving segment. Performing the outgoing
            # corner turn here would lose that corner's worst-axis budget when
            # the prefix sampler removes the remainder of the polyline.
            end = replace(end, yaw_rad=math.atan2(b.y_m - a.y_m, b.x_m - a.x_m))
    else:
        end = Pose2D(
            a.x_m + t * (b.x_m - a.x_m), a.y_m + t * (b.y_m - a.y_m),
            math.atan2(b.y_m - a.y_m, b.x_m - a.x_m),
        )
    return (*full_poses[:i + 1], end)


def evaluate_admitted_return_stage_uncertainty(
    costmap, map_route, covariance, config, *, start_pose: Pose2D,
    target_evidence_sha256: str, is_final_stage: bool,
) -> RouteUncertaintyAdmissionResult:
    """Retain worst-axis clearance for initial alignment and a stopped cut.

    The endpoint envelope below is a mathematical all-heading disk check. Its
    quarter-turn headings are sampling inputs only, never planned robot motion.
    They keep the existing stationary evaluator's conservative covariance and
    all reserves, including the original (not reset) heading reference.
    """
    poses = tuple(map_route)
    if (
        not isinstance(is_final_stage, bool) or not isinstance(start_pose, Pose2D)
        or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for v in (start_pose.x_m, start_pose.y_m, start_pose.yaw_rad))
        or len(poses) < 2 or any(not isinstance(p, Pose2D) for p in poses)
        or math.hypot(poses[0].x_m - start_pose.x_m, poses[0].y_m - start_pose.y_m) > 1e-9
    ):
        raise ValueError("return stage requires a bound finite start pose and final-stage flag")
    if len(poses) == 2 and (poses[0].x_m, poses[0].y_m) == (poses[1].x_m, poses[1].y_m):
        return evaluate_stationary_turn_uncertainty_admission(
            costmap, poses, covariance, config, start_pose=start_pose,
            target_evidence_sha256=target_evidence_sha256,
        )
    admission = evaluate_route_uncertainty_admission(costmap, poses, covariance, config)
    if not admission.decision.accepted:
        return admission
    segments = list(admission.segments)
    envelope_evidence = []
    for name, point in (("initial_orientation", start_pose), *(() if is_final_stage else (("stopped_endpoint_orientation", poses[-1]),))):
        point = replace(point, yaw_rad=0.)
        envelope = evaluate_stationary_turn_uncertainty_admission(
            costmap, (point, replace(point, yaw_rad=math.pi / 2)), covariance, config,
            start_pose=point, target_evidence_sha256=target_evidence_sha256,
        )
        if not envelope.segments:
            raise ValueError("return endpoint orientation envelope is invalid")
        segments.extend(replace(s, segment_id=name) for s in envelope.segments)
        envelope_evidence.append({"endpoint": name, "admission": envelope.to_evidence_dict()})
    decision = evaluate_route_uncertainty_budget(tuple(segments))
    evidence = {**admission.evidence,
        "endpoint_orientation_envelopes": envelope_evidence,
        "decision": decision.to_evidence_dict(),
        "decision_evidence_sha256": uncertainty_budget_evidence_sha256(decision),
    }
    evidence["budget_profile"] = [*admission.evidence["budget_profile"], *({
        "profile_index": len(admission.segments) + index,
        "segment_id": s.segment_id, "raw_centerline_clearance_m": s.raw_centerline_clearance_m,
        "segment_normal": {"x": s.segment_normal_x, "y": s.segment_normal_y},
        "is_corner": True, "isotropic_covariance": True,
        "heading_contribution_m": s.heading_contribution_m,
    } for index, s in enumerate(segments[len(admission.segments):]))]
    return RouteUncertaintyAdmissionResult(tuple(segments), decision, evidence)


def select_admitted_return_prefix(
    *, full_poses: tuple[Pose2D, ...], base_costmap: Costmap,
    uncertainty: CandidateRouteUncertaintyContext, target_evidence_sha256: str,
    minimum_prefix_vertex_index: int = 0,
) -> ReturnRouteStage:
    """Prefer the whole route, then existing vertices, then bounded line cuts."""
    if len(full_poses) < 2 or any(
        not isinstance(p, Pose2D) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (p.x_m, p.y_m, p.yaw_rad)
        ) for p in full_poses
    ):
        raise ValueError("full return route requires finite poses")
    if not isinstance(uncertainty, CandidateRouteUncertaintyContext):
        raise ValueError("return uncertainty context is required")
    config = uncertainty.admission_config
    start = full_poses[0]
    if (
        config.heading_reference_x_m != start.x_m
        or config.heading_reference_y_m != start.y_m
        or config.braking_latency_distance_m < .075 - 1e-12
    ):
        raise ValueError("return uncertainty requires the actual start anchor and fast braking reserve")
    if (
        isinstance(minimum_prefix_vertex_index, bool)
        or not isinstance(minimum_prefix_vertex_index, int)
        or not 0 <= minimum_prefix_vertex_index < len(full_poses)
    ):
        raise ValueError("invalid required prefix anchor index")
    stationary = len(full_poses) == 2 and (
        full_poses[0].x_m, full_poses[0].y_m
    ) == (full_poses[1].x_m, full_poses[1].y_m)

    attempt_diagnostics = []

    def evaluate(poses, *, final):
        executable = executable_return_poses(poses)
        admission = evaluate_admitted_return_stage_uncertainty(
            base_costmap, executable, uncertainty.covariance, config,
            start_pose=start, target_evidence_sha256=target_evidence_sha256,
            is_final_stage=final,
        )
        attempt_diagnostics.append(_attempt_diagnostic(poses, admission, final=final))
        return admission

    whole = evaluate(full_poses, final=True)
    evidence = {
        "schema_version": 1, "motion_authorized": False,
        "policy": "full-route-then-stopped-prefix-with-unchanged-anchor",
        "full_route_geometry_sha256": payload_sha256(return_route_geometry(full_poses)),
        "target_evidence_sha256": target_evidence_sha256,
        "covariance": asdict(uncertainty.covariance), "config": asdict(config),
        "source_evidence": dict(uncertainty.source_evidence),
        "full_route_admission": whole.to_evidence_dict(),
        "minimum_prefix_vertex_index": minimum_prefix_vertex_index,
    }

    def selected(poses, index, fraction, admission, final):
        return ReturnRouteStage(poses, final, poses[-1], index, fraction, {
            **evidence, "selected_route_geometry_sha256": payload_sha256(return_route_geometry(poses)),
            "selected_admission": admission.to_evidence_dict(),
            "is_final_stage": final, "end_segment_index": index, "end_fraction": fraction,
        })

    if whole.decision.accepted:
        return selected(full_poses, len(full_poses) - 2, 1., whole, True)
    if stationary:
        raise ReturnUncertaintyExhausted("stationary return uncertainty budget exhausted", attempt_diagnostics)
    lengths = [math.hypot(b.x_m - a.x_m, b.y_m - a.y_m) for a, b in zip(full_poses, full_poses[1:])]
    if any(length <= 0. for length in lengths):
        raise ValueError("return travel route requires positive segment lengths")
    cumulative = [0.]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    def try_cut(index, fraction):
        distance = cumulative[index] + lengths[index] * fraction
        if (
            distance + 1e-9 < cumulative[minimum_prefix_vertex_index]
            or cumulative[-1] - distance < MINIMUM_REMAINING_ROUTE_M
        ):
            return None
        poses = build_return_prefix(full_poses, index, fraction)
        end = poses[-1]
        if math.hypot(end.x_m - start.x_m, end.y_m - start.y_m) < MINIMUM_STAGE_DISPLACEMENT_M:
            return None
        admission = evaluate(poses, final=False)
        return selected(poses, index, fraction, admission, False) if admission.decision.accepted else None

    for vertex in range(len(full_poses) - 2, 0, -1):
        choice = try_cut(vertex - 1, 1.)
        if choice is not None:
            return choice
    # Distances are bounded and deterministic, and lie on the certified line.
    count = min(MAXIMUM_SAMPLED_CUTS, max(1, math.ceil(cumulative[-1] / .25)))
    for sample in range(count, 0, -1):
        distance = cumulative[-1] * sample / (count + 1)
        for index, length in enumerate(lengths):
            if cumulative[index] < distance <= cumulative[index + 1]:
                choice = try_cut(index, (distance - cumulative[index]) / length)
                if choice is not None:
                    return choice
                break
    raise ReturnUncertaintyExhausted("return uncertainty budget exhausted: no meaningful admitted prefix", attempt_diagnostics)
