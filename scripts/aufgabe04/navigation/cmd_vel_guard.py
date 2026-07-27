"""Fail-closed command-ownership lease for a future cmd_vel mux/guard node.

The waypoint follower still performs graph-level publisher checks, but those
checks cannot revoke an already latched command after the follower crashes.
This ROS-free state machine defines the independent guard contract: one owner,
strictly increasing command sequence numbers, short leases, and an automatic
zero command whenever ownership or timing is invalid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.navigation.waypoint_controller import VelocityCommand


ZERO_VELOCITY = VelocityCommand(0.0, 0.0)


@dataclass(frozen=True)
class CommandLease:
    owner_id: str
    epoch: int
    acquired_monotonic_sec: float
    expires_monotonic_sec: float

    def __post_init__(self) -> None:
        if not str(self.owner_id).strip():
            raise ValueError("owner_id must be non-empty")
        if not isinstance(self.epoch, int) or isinstance(self.epoch, bool) or self.epoch < 1:
            raise ValueError("epoch must be a positive integer")
        if not all(
            math.isfinite(value)
            for value in (self.acquired_monotonic_sec, self.expires_monotonic_sec)
        ):
            raise ValueError("lease timestamps must be finite")
        if self.expires_monotonic_sec <= self.acquired_monotonic_sec:
            raise ValueError("lease expiry must follow acquisition")


@dataclass(frozen=True)
class GuardedCommand:
    owner_id: str
    epoch: int
    sequence: int
    issued_monotonic_sec: float
    command: VelocityCommand

    def __post_init__(self) -> None:
        if not str(self.owner_id).strip():
            raise ValueError("owner_id must be non-empty")
        if not isinstance(self.epoch, int) or isinstance(self.epoch, bool) or self.epoch < 1:
            raise ValueError("epoch must be a positive integer")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 1
        ):
            raise ValueError("sequence must be a positive integer")
        if not math.isfinite(self.issued_monotonic_sec):
            raise ValueError("issued_monotonic_sec must be finite")
        if not all(
            math.isfinite(value)
            for value in (
                self.command.linear_x_mps,
                self.command.angular_z_radps,
            )
        ):
            raise ValueError("velocity command must be finite")


@dataclass(frozen=True)
class CommandGuardDecision:
    command: VelocityCommand
    accepted: bool
    reason: str
    next_sequence: int


def guard_command(
    lease: CommandLease | None,
    candidate: GuardedCommand | None,
    *,
    now_monotonic_sec: float,
    last_accepted_sequence: int = 0,
    max_command_age_sec: float = 0.25,
    max_future_sec: float = 0.02,
) -> CommandGuardDecision:
    """Return the candidate or a zero command; invalid input never propagates."""

    if not math.isfinite(now_monotonic_sec):
        raise ValueError("now_monotonic_sec must be finite")
    if not math.isfinite(max_command_age_sec) or max_command_age_sec <= 0.0:
        raise ValueError("max_command_age_sec must be finite and positive")
    if not math.isfinite(max_future_sec) or max_future_sec < 0.0:
        raise ValueError("max_future_sec must be finite and non-negative")
    if lease is None:
        return CommandGuardDecision(ZERO_VELOCITY, False, "missing command lease", last_accepted_sequence)
    if now_monotonic_sec > lease.expires_monotonic_sec:
        return CommandGuardDecision(ZERO_VELOCITY, False, "expired command lease", last_accepted_sequence)
    if now_monotonic_sec < lease.acquired_monotonic_sec:
        return CommandGuardDecision(ZERO_VELOCITY, False, "command guard clock regressed", last_accepted_sequence)
    if candidate is None:
        return CommandGuardDecision(ZERO_VELOCITY, False, "missing guarded command", last_accepted_sequence)
    if candidate.owner_id != lease.owner_id or candidate.epoch != lease.epoch:
        return CommandGuardDecision(ZERO_VELOCITY, False, "command owner or epoch mismatch", last_accepted_sequence)
    if candidate.sequence <= last_accepted_sequence:
        return CommandGuardDecision(ZERO_VELOCITY, False, "replayed guarded command", last_accepted_sequence)
    age_sec = now_monotonic_sec - candidate.issued_monotonic_sec
    if age_sec < -max_future_sec:
        return CommandGuardDecision(ZERO_VELOCITY, False, "future-dated guarded command", last_accepted_sequence)
    if age_sec > max_command_age_sec:
        return CommandGuardDecision(ZERO_VELOCITY, False, "stale guarded command", last_accepted_sequence)
    return CommandGuardDecision(candidate.command, True, "", candidate.sequence)
