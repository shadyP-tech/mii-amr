"""Constant-memory arrival ages, including frames discarded before decoding.

Header-to-receipt age compares two wall clocks. These diagnostics expose that
quantity without attributing it to transport or correcting an unknown offset.
"""

from dataclasses import dataclass, field
import math


@dataclass
class ReceiptAgeSummary:
    count: int = 0
    total_sec: float = 0.
    minimum_sec: float | None = None
    maximum_sec: float | None = None
    last_sec: float | None = None

    def record(self, age_sec):
        self.count += 1
        self.total_sec += age_sec
        self.minimum_sec = age_sec if self.minimum_sec is None else min(self.minimum_sec, age_sec)
        self.maximum_sec = age_sec if self.maximum_sec is None else max(self.maximum_sec, age_sec)
        self.last_sec = age_sec

    def snapshot(self):
        return dict(count=self.count, mean_sec=self.total_sec / self.count if self.count else None,
                    minimum_sec=self.minimum_sec, maximum_sec=self.maximum_sec, last_sec=self.last_sec)


@dataclass
class FrameReceiptDiagnostics:
    arrivals: int = 0
    age_rejections: int = 0
    missing_stamp: int = 0
    future_stamp: int = 0
    all_ages: ReceiptAgeSummary = field(default_factory=ReceiptAgeSummary)
    accepted_ages: ReceiptAgeSummary = field(default_factory=ReceiptAgeSummary)
    rejected_ages: ReceiptAgeSummary = field(default_factory=ReceiptAgeSummary)

    def record(self, *, stamp_sec, received_wall_sec, age_rejected):
        self.arrivals += 1
        self.age_rejections += int(age_rejected)
        if stamp_sec is None or not math.isfinite(stamp_sec):
            self.missing_stamp += 1
            return
        age = received_wall_sec - stamp_sec
        self.future_stamp += int(age < 0.)
        self.all_ages.record(age)
        (self.rejected_ages if age_rejected else self.accepted_ages).record(age)

    def snapshot(self):
        return dict(arrivals=self.arrivals, age_rejections=self.age_rejections,
                    accepted_arrivals=self.arrivals - self.age_rejections,
                    missing_stamp=self.missing_stamp, future_stamp=self.future_stamp,
                    header_receipt_clock_offset_measured=False,
                    all_header_to_receipt_age=self.all_ages.snapshot(),
                    accepted_header_to_receipt_age=self.accepted_ages.snapshot(),
                    rejected_header_to_receipt_age=self.rejected_ages.snapshot())
