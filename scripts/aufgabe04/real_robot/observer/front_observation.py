"""Front-marker evidence policy independent of metric axis acquisition.

A decoded QR can contradict a QR-free geometry fallback without contradicting
the QR identity itself. This module withholds that axis; target association,
freshness, stationary epochs and temporal identity checks remain authoritative.
"""

from dataclasses import dataclass

from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_SAMPLE_SOURCE,
    REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
)


BACKSIDE_AXIS_SOURCES = frozenset({
    BACKSIDE_AXIS_SAMPLE_SOURCE, REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE,
})


@dataclass(frozen=True)
class FrontObservationDecision:
    marker_observed_now: bool
    marker_seen_in_stationary_epoch: bool
    withhold_backside_axis: bool
    classification: str

    def metadata(self, *, lidar_associated: bool, frame_accepted: bool) -> dict:
        return {
            "classification": self.classification,
            "axis_state": "unresolved",
            "marker_observed_now": self.marker_observed_now,
            "marker_seen_in_stationary_epoch": self.marker_seen_in_stationary_epoch,
            "lidar_associated": bool(lidar_associated),
            "candidate_frame_accepted": bool(frame_accepted),
            "motion_authorized": False,
            "completion_authorized": False,
            "identity_authority": "requires_current_associated_temporal_latch",
        }


def front_observation_decision(
    *, qr_texts: tuple[str, ...], qr_marker_detected: bool,
    estimate_source: str, marker_seen_in_stationary_epoch: bool,
) -> FrontObservationDecision:
    """Keep positive front evidence while rejecting QR-free axis authority.

    An unassociated marker may conservatively veto a backside axis but cannot
    establish the candidate's identity. Only the existing evidence accumulator
    can bind any text to the target, including multiple/conflicting QR checks.
    """

    observed = bool(qr_texts) or qr_marker_detected is True
    seen = observed or marker_seen_in_stationary_epoch
    return FrontObservationDecision(
        marker_observed_now=observed,
        marker_seen_in_stationary_epoch=seen,
        withhold_backside_axis=seen and estimate_source in BACKSIDE_AXIS_SOURCES,
        classification="front_readable" if qr_texts else "front_unreadable",
    )
