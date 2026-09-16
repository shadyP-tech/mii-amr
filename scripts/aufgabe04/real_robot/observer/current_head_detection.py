"""Current head detection authority, distinct from a unique admitted angle.

An independently validated orientation bound may locate and associate current
head pixels while the strict single-angle contract remains rejected. Nothing
in this adapter upgrades that strict contract or authorizes motion.
"""

import math

from scripts.aufgabe04.perception.stand_axis.head_model_admission import admit_measured_head_model
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MEASURED_HEAD_AXIS_SOURCE
from scripts.aufgabe04.perception.stand_axis.head_outer_border import current_head_boundary_eligible
from scripts.aufgabe04.perception.stand_axis.head_orientation_bounds import (
    validated_current_head_orientation_bounds,
)
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis
from scripts.aufgabe04.real_robot.observer.contract import BACKSIDE_AXIS_SAMPLE_SOURCE


def current_head_detection_admission(estimate, debug, *, profile_sha256=None):
    """Return the strict decision and a separately validated current bound."""
    yaw = estimate.yaw_deg
    strict = admit_measured_head_model(
        estimate=estimate, debug=debug,
        yaw_rad=math.radians(yaw) if type(yaw) in (int, float) else math.nan,
    )
    bounds = getattr(debug, "head_orientation_bounds", None)
    if not (estimate.source in (MEASURED_HEAD_AXIS_SOURCE, BACKSIDE_AXIS_SAMPLE_SOURCE)
            and debug.model_pose_fit_source == MEASURED_HEAD_AXIS_SOURCE
            and estimate.evidence_state != "predicted_only"
            and debug.evidence_state != "predicted_only"
            and current_head_boundary_eligible(estimate, debug)
            and validated_current_head_orientation_bounds(
                bounds, estimate=estimate, debug=debug, profile_sha256=profile_sha256)):
        bounds = None
    return strict, bounds


def current_head_search_pose(estimate, debug, *, profile_sha256=None):
    """Choose a projection locator only; its angle has no admission authority.

    For a bounded detection, either retained planar pose locates the same
    current head. The next frame must refit all its borders and hypotheses.
    """
    strict, bounds = current_head_detection_admission(
        estimate, debug, profile_sha256=profile_sha256)
    if strict.accepted:
        return debug.model_pose
    if bounds is None:
        return None
    hypothesis = bounds.hypotheses[0]
    return PlanarPoseHypothesis(
        hypothesis.rotation_vector, hypothesis.translation_xyz_m,
        hypothesis.face_normal_xyz, math.degrees(hypothesis.yaw_rad),
        hypothesis.reprojection_rmse_px, hypothesis.all_corners_positive_depth,
    )
