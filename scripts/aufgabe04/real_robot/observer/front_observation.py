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
    marker_tentative_now: bool = False

    def metadata(self, *, lidar_associated: bool, frame_accepted: bool) -> dict:
        return {
            "classification": self.classification,
            "axis_state": "unresolved",
            "marker_observed_now": self.marker_observed_now,
            "marker_tentative_now": self.marker_tentative_now,
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
    qr_marker_verified: bool | None = None,
) -> FrontObservationDecision:
    """Keep positive front evidence while rejecting QR-free axis authority.

    An unassociated marker may conservatively veto a backside axis but cannot
    establish the candidate's identity. Only the existing evidence accumulator
    can bind any text to the target, including multiple/conflicting QR checks.
    A tentative quadrilateral vetoes only this frame's backside measurement;
    decoded identity or verified finder pixels can persist through this stopped
    epoch. ``None`` preserves legacy producers; live image processing supplies
    an explicit verification result.
    """

    verified = (
        qr_marker_detected is True if qr_marker_verified is None
        else qr_marker_verified is True
    )
    observed = bool(qr_texts) or verified
    tentative = qr_marker_detected is True and not observed
    seen = observed or marker_seen_in_stationary_epoch
    return FrontObservationDecision(
        marker_observed_now=observed,
        marker_seen_in_stationary_epoch=seen,
        withhold_backside_axis=(seen or tentative) and estimate_source in BACKSIDE_AXIS_SOURCES,
        classification="front_readable" if qr_texts else "front_unreadable",
        marker_tentative_now=tentative,
    )
