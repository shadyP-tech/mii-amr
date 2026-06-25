"""Adapter-neutral follower result models for Aufgabe 04 navigation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FollowerResult:
    status: str
    stop_reason: str
    duration_sec: float
    distance_estimate_m: float
    motion_published: bool
