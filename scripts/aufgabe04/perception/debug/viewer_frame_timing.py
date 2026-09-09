"""Capture and receipt ages for the real-camera diagnostic viewer."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.aufgabe04.perception.stand_axis.observation_freshness import (
    ObservationFreshness,
    observation_freshness,
    source_stamp_in_monotonic_domain,
)


@dataclass(frozen=True)
class ViewerFrameTiming:
    received_monotonic_sec: float | None
    observed_monotonic_sec: float | None

    @classmethod
    def from_read(cls, read):
        return cls(
            received_monotonic_sec=read.received_monotonic_sec,
            observed_monotonic_sec=source_stamp_in_monotonic_domain(
                source_stamp_sec=read.stamp_sec,
                received_wall_sec=read.received_wall_sec,
                received_monotonic_sec=read.received_monotonic_sec,
            ),
        )

    def source_age_sec(self, now_sec: float) -> float | None:
        return observation_freshness(
            observed_at_sec=self.observed_monotonic_sec,
            now_sec=now_sec,
            max_age_sec=0.0,
        ).age_sec

    def assess(
        self, *, now_sec: float, max_result_age_sec: float, max_frame_age_sec: float
    ) -> ObservationFreshness:
        receipt = observation_freshness(
            observed_at_sec=self.received_monotonic_sec,
            now_sec=now_sec,
            max_age_sec=max_result_age_sec,
        )
        if not receipt.accepted:
            return receipt
        return observation_freshness(
            observed_at_sec=self.observed_monotonic_sec,
            now_sec=now_sec,
            max_age_sec=max_frame_age_sec,
        )
