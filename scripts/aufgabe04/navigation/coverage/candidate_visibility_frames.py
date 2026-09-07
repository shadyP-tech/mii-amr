"""Project historical candidate geometry into a visibility receipt's epoch.

Missing frame evidence is inconclusive, never clearance. This pure boundary
reuses canonical-odom reprojection and owns no scan selection or policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    reproject_candidate_point,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    SurveyCandidate,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    LidarVisibilityReceipt,
)


@dataclass(frozen=True)
class CandidateVisibilityProjection:
    target: Pose2D | None
    reason: str | None
    candidate_source_evidence_id: str | None
    receipt_source_evidence_id: str | None
    displacement_m: float | None = None

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "geometry_mode": "canonical_odom_projected_into_receipt_epoch",
            "reason": self.reason,
            "candidate_source_evidence_id": self.candidate_source_evidence_id,
            "receipt_source_evidence_id": self.receipt_source_evidence_id,
            "candidate_map_point": (
                None if self.target is None
                else {"x_m": self.target.x_m, "y_m": self.target.y_m}
            ),
            "candidate_map_displacement_m": self.displacement_m,
        }


def project_candidate_for_visibility(
    candidate: SurveyCandidate,
    receipt: LidarVisibilityReceipt,
) -> CandidateVisibilityProjection:
    """Require consistent frame identities before comparing map coordinates."""

    source = candidate.frame_provenance
    frame = receipt.frame_provenance
    ids = {
        "candidate_source_evidence_id": (
            None if source is None else source.source_evidence_id
        ),
        "receipt_source_evidence_id": (
            None if frame is None else frame.source_evidence_id
        ),
    }

    def unavailable(reason: str) -> CandidateVisibilityProjection:
        return CandidateVisibilityProjection(target=None, reason=reason, **ids)

    if source is None or not source.source_evidence_id:
        return unavailable("candidate_frame_provenance_missing")
    if frame is None:
        return unavailable("visibility_receipt_frame_provenance_missing")
    if (
        source.map_frame != receipt.planning_frame
        or frame.map_frame != receipt.planning_frame
        or source.odom_frame != frame.odom_frame
    ):
        return unavailable("candidate_visibility_frame_identity_mismatch")
    frozen_point = source.frozen_map_point
    if frozen_point is not None and math.hypot(
        candidate.x_m - frozen_point.x_m,
        candidate.y_m - frozen_point.y_m,
    ) > 1.0e-8:
        return unavailable("candidate_frozen_map_point_mismatch")
    if (
        source.source_evidence_id == frame.source_evidence_id
        and source.frozen_map_from_odom is not None
        and source.frozen_map_from_odom != frame.map_from_odom
    ):
        return unavailable("visibility_frame_certificate_transform_mismatch")
    result = reproject_candidate_point(source, frame.map_from_odom)
    point = result.current_map_point
    return CandidateVisibilityProjection(
        target=Pose2D(point.x_m, point.y_m, 0.0),
        reason=None,
        displacement_m=math.hypot(
            point.x_m - candidate.x_m, point.y_m - candidate.y_m,
        ),
        **ids,
    )
