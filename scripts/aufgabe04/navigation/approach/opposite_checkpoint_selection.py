"""Motion-neutral selection of one stopped waypoint on an opposite route.

The suffix calculation is a feasibility forecast with unchanged covariance,
never a localization receipt or authorization to reset a live drift anchor.
"""
from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import (
    build_return_prefix, evaluate_admitted_return_stage_uncertainty,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


@dataclass(frozen=True)
class OppositeCheckpoint:
    vertex_index: int
    poses: tuple[Pose2D, ...]
    minimum_margin_m: float
    evidence: dict


def select_opposite_checkpoint(*, poses, start_pose, costmap, uncertainty, source_sha256):
    """Choose the largest two-part margin among at most 32 existing vertices."""
    poses = tuple(poses)
    if len(poses) < 3:
        return None
    config = uncertainty.admission_config
    if (config.heading_reference_x_m != start_pose.x_m
            or config.heading_reference_y_m != start_pose.y_m
            or (poses[0].x_m, poses[0].y_m) != (start_pose.x_m, start_pose.y_m)):
        raise ValueError("checkpoint selection requires the unchanged measured start anchor")
    if any(not math.isfinite(v) for p in poses for v in (p.x_m, p.y_m)):
        raise ValueError("checkpoint route coordinates must be finite")
    indices = list(range(1, len(poses) - 1))
    if len(indices) > 32:
        indices = sorted({indices[round(i * (len(indices) - 1) / 31)] for i in range(32)})
    choices = []
    evaluations = []
    for index in indices:
        prefix = build_return_prefix(poses, index - 1, 1.)
        stop = prefix[-1]
        remaining = sum(math.hypot(b.x_m-a.x_m, b.y_m-a.y_m)
                        for a,b in zip(poses[index:], poses[index+1:]))
        if math.hypot(stop.x_m-start_pose.x_m, stop.y_m-start_pose.y_m) < .20 or remaining < .15:
            continue
        first = evaluate_admitted_return_stage_uncertainty(
            costmap, prefix, uncertainty.covariance, config, start_pose=start_pose,
            target_evidence_sha256=source_sha256, is_final_stage=False,
        )
        # Hypothesis only: the real suffix must use newly measured localization.
        second = evaluate_admitted_return_stage_uncertainty(
            costmap, poses[index:], uncertainty.covariance,
            replace(config, heading_reference_x_m=stop.x_m, heading_reference_y_m=stop.y_m),
            start_pose=stop, target_evidence_sha256=source_sha256, is_final_stage=True,
        )
        row = {"vertex_index": index, "prefix": first.to_evidence_dict(),
               "hypothetical_suffix": second.to_evidence_dict()}
        evaluations.append(row)
        if first.decision.accepted and second.decision.accepted:
            margin = min(first.decision.remaining_margin_m, second.decision.remaining_margin_m)
            choices.append((margin, index, prefix))
    if not choices:
        return None
    margin, index, prefix = max(choices, key=lambda c: (c[0], c[1]))
    return OppositeCheckpoint(index, prefix, margin, {
        "schema_version": 1, "selected_vertex_index": index,
        "source_route_sha256": source_sha256, "evaluations": evaluations,
        "hypothetical_suffix_only": True, "fresh_checkpoint_localization_required": True,
        "maximum_checkpoint_count": 1, "motion_authorized": False,
        "route_limits_unchanged": True,
    })
