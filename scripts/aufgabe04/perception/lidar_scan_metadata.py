"""Original LaserScan geometry and diagnostics, without filling missing rays."""
from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.scan_topology import SCAN_TOPOLOGY_PROFILES, ScanTopology


INVALID_RANGE_REASONS = frozenset(("nan", "positive_infinity", "negative_infinity",
                                 "zero", "below_range_min", "above_range_max"))


@dataclass(frozen=True)
class LidarScanMetadata:
    angle_max_rad: float
    time_increment_sec: float
    scan_time_sec: float
    scan_topology_profile: str
    intensities: tuple[float | None, ...]
    invalid_range_reasons: tuple[str | None, ...]

    def validate(self, ranges):
        for name in ("angle_max_rad", "time_increment_sec", "scan_time_sec"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid original scan {name}")
        if self.time_increment_sec < 0 or self.scan_time_sec < 0:
            raise ValueError("negative original scan timing")
        if self.scan_topology_profile not in SCAN_TOPOLOGY_PROFILES:
            raise ValueError("unknown original scan topology profile")
        if not isinstance(self.intensities, tuple) or len(self.intensities) not in (0, len(ranges)):
            raise ValueError("original scan intensity count mismatch")
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v))
               for v in self.intensities):
            raise ValueError("original scan intensities must be sanitized")
        if not isinstance(self.invalid_range_reasons, tuple) or len(self.invalid_range_reasons) != len(ranges):
            raise ValueError("original scan invalid-reason count mismatch")
        for value, reason in zip(ranges, self.invalid_range_reasons):
            if (value is None and reason not in INVALID_RANGE_REASONS
                    or value is not None and reason is not None):
                raise ValueError("original scan invalid reason differs from ranges")
        return self

    def to_mapping(self):
        return {"angle_max_rad": self.angle_max_rad,
                "time_increment_sec": self.time_increment_sec, "scan_time_sec": self.scan_time_sec,
                "scan_topology_profile": self.scan_topology_profile,
                "intensities": list(self.intensities),
                "invalid_range_reasons": list(self.invalid_range_reasons)}

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("original scan metadata fields do not match schema")
        if any(not isinstance(value[k], (list, tuple)) for k in ("intensities", "invalid_range_reasons")):
            raise ValueError("original scan diagnostic arrays are malformed")
        return cls(**{**value, "intensities": tuple(value["intensities"]),
                      "invalid_range_reasons": tuple(value["invalid_range_reasons"])})

    def topology(self, *, sample_count, angle_min_rad, angle_increment_rad):
        return ScanTopology(sample_count, angle_min_rad, angle_increment_rad,
                            self.angle_max_rad, self.scan_topology_profile)


def metadata_from_message(message, *, topology_profile):
    reasons = []
    for raw in message.ranges:
        if isinstance(raw, bool):
            raise ValueError("scan ranges must be numeric")
        value = float(raw)
        reasons.append("nan" if math.isnan(value) else
                       "positive_infinity" if value == math.inf else
                       "negative_infinity" if value == -math.inf else
                       "zero" if value == 0 else
                       "below_range_min" if value < message.range_min else
                       "above_range_max" if value > message.range_max else None)
    intensities = []
    for value in message.intensities:
        if isinstance(value, bool):
            raise ValueError("scan intensities must be numeric")
        number = float(value)
        intensities.append(number if math.isfinite(number) else None)
    return LidarScanMetadata(message.angle_max, message.time_increment, message.scan_time,
                             topology_profile, tuple(intensities), tuple(reasons))
