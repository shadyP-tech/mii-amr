"""Operator prompt for manual AMCL initial-pose seeding.

This module does not publish ROS messages or authorize motion. It only owns
the terminal wording and input validation for a stopped, preauthorization
initial-pose refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


@dataclass(frozen=True)
class InitialPosePromptConfig:
    amcl_topic: str
    observation_window_sec: float
    maximum_retry_count: int

    def __post_init__(self) -> None:
        topic = str(self.amcl_topic).strip()
        if not topic:
            raise ValueError("amcl_topic must be non-empty")
        if (
            not math.isfinite(self.observation_window_sec)
            or self.observation_window_sec <= 0.0
        ):
            raise ValueError("observation_window_sec must be finite and positive")
        if (
            type(self.maximum_retry_count) is not int
            or self.maximum_retry_count < 0
        ):
            raise ValueError("maximum_retry_count must be a non-negative integer")
        object.__setattr__(self, "amcl_topic", topic)


def prompt_for_initialpose_attempt(
    *,
    config: InitialPosePromptConfig,
    attempt_index: int,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Pause so the operator can seed AMCL immediately before dry admission."""

    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    if attempt_index > config.maximum_retry_count:
        raise ValueError("attempt_index must not exceed maximum_retry_count")
    if not callable(input_fn):
        raise ValueError("input_fn must be callable")
    if not callable(output_fn):
        raise ValueError("output_fn must be callable")

    if attempt_index == 0:
        output_fn("\nInitial-pose refresh required before first-route readiness.")
    else:
        output_fn("\nInitial-pose refresh required before readiness retry.")
        output_fn(
            "Previous no-motion admission did not leave enough localization "
            "clearance for the first route."
        )
    output_fn("AMCL often publishes only once after RViz 2D Pose Estimate.")
    output_fn("Do not move the robot. Do not send a Nav2 goal.")
    output_fn(f"AMCL topic: {config.amcl_topic}")
    output_fn(
        "Press Enter here, then immediately click 2D Pose Estimate in RViz "
        f"during the next {config.observation_window_sec:.1f}s."
    )
    input_fn("Press Enter, then click 2D Pose Estimate immediately: ")


__all__ = [
    "InitialPosePromptConfig",
    "prompt_for_initialpose_attempt",
]
