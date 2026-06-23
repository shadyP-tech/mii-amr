"""Aufgabe 04 preflight checks wrapping navigation safety assumptions."""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class PreflightStatus:
    ok: bool
    failures: List[str]


def validate_required_topics(available_topics: Iterable[str], required_topics: Iterable[str]) -> PreflightStatus:
    available = set(available_topics)
    missing = [topic for topic in required_topics if topic not in available]
    return PreflightStatus(ok=not missing, failures=[f"missing topic: {topic}" for topic in missing])

