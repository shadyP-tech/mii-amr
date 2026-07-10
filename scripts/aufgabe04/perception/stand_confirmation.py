"""ROS-free confirmation logic for repeated LiDAR stand observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.perception.stand_observation import StandObservation


@dataclass(frozen=True)
class StandConfirmationConfig:
    merge_distance_m: float = 0.18
    min_hits: int = 3
    max_age_sec: float = 8.0
    min_confidence: float = 0.55


@dataclass(frozen=True)
class ConfirmedStand:
    stand_id: str
    x_m: float
    y_m: float
    confidence: float
    hit_count: int
    first_seen_sec: float
    last_seen_sec: float
    first_confirmed_at_sec: float
    source_observation_ids: tuple[str, ...]
    provenance: dict[str, object]


@dataclass(frozen=True)
class _Track:
    x_m: float
    y_m: float
    confidence_sum: float
    hit_count: int
    first_seen_sec: float
    last_seen_sec: float
    first_confirmed_at_sec: float | None
    source_observation_ids: tuple[str, ...]
    provenance: dict[str, object]

    @property
    def confidence(self) -> float:
        return self.confidence_sum / max(self.hit_count, 1)


class StandConfirmationAccumulator:
    def __init__(
        self,
        *,
        config: StandConfirmationConfig | None = None,
        arena_bounds: ArenaBounds | None = None,
    ) -> None:
        self.config = config or StandConfirmationConfig()
        self.arena_bounds = arena_bounds or ArenaBounds()
        self.arena_bounds.validate()
        self._tracks: list[_Track] = []

    def add_observations(self, observations: Iterable[StandObservation]) -> tuple[ConfirmedStand, ...]:
        for observation in observations:
            self.add_observation(observation)
        return self.confirmed_stands()

    def add_observation(self, observation: StandObservation) -> tuple[ConfirmedStand, ...]:
        if observation.confidence < self.config.min_confidence:
            return self.confirmed_stands()
        if not self.arena_bounds.contains(Pose2D(observation.x_m, observation.y_m, 0.0)):
            return self.confirmed_stands()

        self._expire_before(observation.observed_at_sec - self.config.max_age_sec)
        track_index = self._nearest_track_index(observation)
        if track_index is None:
            self._tracks.append(_track_from_observation(observation, self.config))
        else:
            self._tracks[track_index] = _merge_track(self._tracks[track_index], observation, self.config)
        return self.confirmed_stands()

    def confirmed_stands(self) -> tuple[ConfirmedStand, ...]:
        confirmed = []
        for index, track in enumerate(self._tracks, start=1):
            if track.hit_count < self.config.min_hits or track.first_confirmed_at_sec is None:
                continue
            confirmed.append(
                ConfirmedStand(
                    stand_id=f"detected_stand_{index:02d}",
                    x_m=track.x_m,
                    y_m=track.y_m,
                    confidence=track.confidence,
                    hit_count=track.hit_count,
                    first_seen_sec=track.first_seen_sec,
                    last_seen_sec=track.last_seen_sec,
                    first_confirmed_at_sec=track.first_confirmed_at_sec,
                    source_observation_ids=track.source_observation_ids,
                    provenance=track.provenance,
                )
            )
        return tuple(confirmed)

    def _nearest_track_index(self, observation: StandObservation) -> int | None:
        best_index = None
        best_distance = self.config.merge_distance_m
        for index, track in enumerate(self._tracks):
            distance = math.hypot(track.x_m - observation.x_m, track.y_m - observation.y_m)
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _expire_before(self, cutoff_sec: float) -> None:
        self._tracks = [track for track in self._tracks if track.last_seen_sec >= cutoff_sec]


def _track_from_observation(observation: StandObservation, config: StandConfirmationConfig) -> _Track:
    return _Track(
        x_m=observation.x_m,
        y_m=observation.y_m,
        confidence_sum=observation.confidence,
        hit_count=1,
        first_seen_sec=observation.observed_at_sec,
        last_seen_sec=observation.observed_at_sec,
        first_confirmed_at_sec=observation.observed_at_sec if config.min_hits <= 1 else None,
        source_observation_ids=(observation.observation_id,),
        provenance={"selected_observation": observation.observation_id, "provenance": observation.provenance.__dict__},
    )


def _merge_track(
    track: _Track,
    observation: StandObservation,
    config: StandConfirmationConfig,
) -> _Track:
    next_hit_count = track.hit_count + 1
    x_m = (track.x_m * track.hit_count + observation.x_m) / next_hit_count
    y_m = (track.y_m * track.hit_count + observation.y_m) / next_hit_count
    first_confirmed_at_sec = track.first_confirmed_at_sec
    if first_confirmed_at_sec is None and next_hit_count >= config.min_hits:
        first_confirmed_at_sec = observation.observed_at_sec
    return _Track(
        x_m=x_m,
        y_m=y_m,
        confidence_sum=track.confidence_sum + observation.confidence,
        hit_count=next_hit_count,
        first_seen_sec=track.first_seen_sec,
        last_seen_sec=observation.observed_at_sec,
        first_confirmed_at_sec=first_confirmed_at_sec,
        source_observation_ids=track.source_observation_ids + (observation.observation_id,),
        provenance={"selected_observation": observation.observation_id, "provenance": observation.provenance.__dict__},
    )


def select_first_confirmed_stand(stands: Iterable[ConfirmedStand]) -> ConfirmedStand:
    ordered = sorted(
        stands,
        key=lambda stand: (stand.first_confirmed_at_sec, -stand.confidence, stand.stand_id),
    )
    if not ordered:
        raise ValueError("no confirmed stand available")
    return ordered[0]


def select_unique_confirmed_stand(stands: Iterable[ConfirmedStand]) -> ConfirmedStand:
    ordered = sorted(
        stands,
        key=lambda stand: (stand.first_confirmed_at_sec, -stand.confidence, stand.stand_id),
    )
    if not ordered:
        raise ValueError("no confirmed stand available")
    if len(ordered) > 1:
        stand_ids = ", ".join(stand.stand_id for stand in ordered)
        raise ValueError(f"ambiguous confirmed stands: {stand_ids}")
    return ordered[0]


def select_confirmed_stand_by_id(
    stands: Iterable[ConfirmedStand],
    stand_id: str,
) -> ConfirmedStand:
    selected_id = stand_id.strip()
    if not selected_id:
        raise ValueError("stand_id must not be empty")
    matches = [stand for stand in stands if stand.stand_id == selected_id]
    if not matches:
        available = ", ".join(stand.stand_id for stand in stands) or "(none)"
        raise ValueError(f"confirmed stand {selected_id} not found; available: {available}")
    return matches[0]
