"""Source-time checks at evidence admission and artifact publication."""

from dataclasses import asdict, dataclass
import math

from scripts.aufgabe04.perception.stand_axis.observation_freshness import (
    ObservationFreshness,
    observation_freshness,
)


def _timestamp(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


@dataclass(frozen=True)
class CameraSourceFreshness:
    accepted: bool
    checked_at_sec: float | None
    image: ObservationFreshness
    scan: ObservationFreshness

    def metadata(self):
        return asdict(self)


def camera_source_freshness(*, image_stamp_sec, scan_stamp_sec, now_sec,
                            max_age_sec: float, max_future_sec: float) -> CameraSourceFreshness:
    """Both original sources must still be fresh; creation time cannot renew them."""

    if (type(max_age_sec) not in (int, float) or not math.isfinite(max_age_sec)
            or max_age_sec <= 0):
        raise ValueError("camera publication age limit must be finite and positive")
    checked_at = _timestamp(now_sec)
    checks = [observation_freshness(
        observed_at_sec=_timestamp(stamp),
        now_sec=float("nan") if checked_at is None else checked_at,
        max_age_sec=max_age_sec, max_future_sec=max_future_sec,
    ) for stamp in (image_stamp_sec, scan_stamp_sec)]
    return CameraSourceFreshness(all(check.accepted for check in checks), checked_at, *checks)


class CameraPublicationExpired(Exception):
    """Only the unpublished temporary artifact is discarded on this outcome."""
