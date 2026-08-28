"""ROS-free capture policy for stationary direct ``map <- odom`` samples.

The policy binds transform evidence to a stationary capture epoch using the
direct-TF identity observed at the start of that epoch.  DDS callback arrival
order is deliberately outside the contract: a candidate is admitted when its
transform stamp and local receipt both advance beyond the epoch baseline and
all timing fields advance beyond the last accepted sample.

This module only classifies normalized dictionaries.  It does not import ROS,
call localization services, spin an executor, or publish motion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence


STAMP_NANOSECONDS = "stamp_nanoseconds"
RECEIPT_TIME_NANOSECONDS = "receipt_time_nanoseconds"
CAPTURE_TIME_NANOSECONDS = "capture_time_nanoseconds"
_TIMING_FIELDS = (
    STAMP_NANOSECONDS,
    RECEIPT_TIME_NANOSECONDS,
    CAPTURE_TIME_NANOSECONDS,
)
_MISSING = object()


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _timing_validation_reasons(
    sample: object,
    *,
    prefix: str,
) -> tuple[str, ...]:
    if not isinstance(sample, Mapping):
        return (f"{prefix}_not_mapping",)

    reasons: list[str] = []
    for field_name in _TIMING_FIELDS:
        value = sample.get(field_name, _MISSING)
        if value is _MISSING:
            reasons.append(f"{prefix}_{field_name}_missing")
        elif type(value) is not int or value < 0:
            reasons.append(
                f"{prefix}_{field_name}_not_nonnegative_integer"
            )
    return tuple(reasons)


def _timing_for_audit(sample: object) -> dict[str, object]:
    """Return a compact JSON-safe snapshot of candidate timing metadata."""

    if not isinstance(sample, Mapping):
        return {"candidate_type": type(sample).__name__}

    timing: dict[str, object] = {}
    for field_name in _TIMING_FIELDS:
        value = sample.get(field_name, _MISSING)
        if value is _MISSING:
            timing[field_name] = None
            timing[f"{field_name}_state"] = "missing"
        elif type(value) is int:
            timing[field_name] = value
        else:
            timing[field_name] = repr(value)
            timing[f"{field_name}_type"] = type(value).__name__
    return timing


@dataclass(frozen=True)
class StationaryMapOdomEpochBaseline:
    """Direct-TF identity captured before a stationary sampling epoch."""

    stamp_nanoseconds: int
    receipt_time_nanoseconds: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.stamp_nanoseconds, STAMP_NANOSECONDS)
        _nonnegative_integer(
            self.receipt_time_nanoseconds,
            RECEIPT_TIME_NANOSECONDS,
        )

    def to_log_dict(self) -> dict[str, int]:
        return {
            STAMP_NANOSECONDS: self.stamp_nanoseconds,
            RECEIPT_TIME_NANOSECONDS: self.receipt_time_nanoseconds,
        }


@dataclass(frozen=True)
class StationaryMapOdomCaptureDecision:
    """Admission result for one normalized direct-TF candidate."""

    accepted: bool
    reason: str
    reasons: tuple[str, ...]
    sample: dict[str, object] | None = None

    def to_log_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "sample": deepcopy(self.sample),
        }


@dataclass(frozen=True)
class StationaryMapOdomCaptureRejection:
    """Audit record for a candidate that could not advance the window."""

    candidate_sequence_index: int
    reasons: tuple[str, ...]
    candidate_timing: dict[str, object]
    accepted_head_timing: dict[str, object] | None

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)

    def to_log_dict(self) -> dict[str, object]:
        return {
            "candidate_sequence_index": self.candidate_sequence_index,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "candidate_timing": deepcopy(self.candidate_timing),
            "accepted_head_timing": deepcopy(self.accepted_head_timing),
        }


@dataclass(frozen=True)
class StationaryMapOdomCaptureWindow:
    """Fail-closed view of the retained stationary-epoch evidence.

    ``samples`` is populated only for a complete window.  Its entries are
    oldest-to-newest and receive contiguous indices in both the legacy
    ``amcl_sample_index`` field and the explicit
    ``stationary_epoch_sample_index`` field.  ``retained_samples`` remains
    available for incomplete-window diagnostics without being claimable as a
    complete paired window.
    """

    complete: bool
    required_count: int
    retained_sample_count: int
    samples: tuple[dict[str, object], ...]
    retained_samples: tuple[dict[str, object], ...]
    rejections: tuple[StationaryMapOdomCaptureRejection, ...]

    def to_log_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "required_count": self.required_count,
            "retained_sample_count": self.retained_sample_count,
            "sample_order": "oldest_to_newest",
            "samples": [deepcopy(sample) for sample in self.samples],
            "retained_samples": [
                deepcopy(sample) for sample in self.retained_samples
            ],
            "rejections": [
                rejection.to_log_dict() for rejection in self.rejections
            ],
        }


@dataclass(frozen=True)
class StationaryMapOdomAmclWindowDecision:
    """Binding result between the direct-TF and AMCL stationary windows."""

    accepted: bool
    reason: str
    reasons: tuple[str, ...]
    direct_tf_window: dict[str, object]
    amcl_window: dict[str, object]

    def to_log_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "direct_tf_window": deepcopy(self.direct_tf_window),
            "amcl_window": deepcopy(self.amcl_window),
        }


def evaluate_stationary_map_odom_candidate(
    candidate: object,
    *,
    epoch_start_baseline: StationaryMapOdomEpochBaseline,
    accepted_head: Mapping[str, object] | None = None,
) -> StationaryMapOdomCaptureDecision:
    """Evaluate one candidate without depending on AMCL callback order.

    The baseline must have been captured before the no-motion request (or
    equivalent stationary epoch trigger).  Both TF stamp and local receipt
    must advance beyond it.  Once a sample has been accepted, stamp, receipt,
    and capture time must all advance strictly beyond the accepted head.
    """

    if not isinstance(
        epoch_start_baseline,
        StationaryMapOdomEpochBaseline,
    ):
        raise ValueError(
            "epoch_start_baseline must be a "
            "StationaryMapOdomEpochBaseline"
        )

    reasons = list(
        _timing_validation_reasons(candidate, prefix="candidate")
    )
    if accepted_head is not None:
        reasons.extend(
            _timing_validation_reasons(
                accepted_head,
                prefix="accepted_head",
            )
        )
    if reasons:
        return StationaryMapOdomCaptureDecision(
            accepted=False,
            reason="; ".join(reasons),
            reasons=tuple(reasons),
        )

    assert isinstance(candidate, Mapping)
    stamp_nanoseconds = int(candidate[STAMP_NANOSECONDS])
    receipt_nanoseconds = int(candidate[RECEIPT_TIME_NANOSECONDS])
    capture_nanoseconds = int(candidate[CAPTURE_TIME_NANOSECONDS])

    if stamp_nanoseconds <= epoch_start_baseline.stamp_nanoseconds:
        reasons.append("stamp_not_after_epoch_start")
    if receipt_nanoseconds <= epoch_start_baseline.receipt_time_nanoseconds:
        reasons.append("receipt_not_after_epoch_start")

    if accepted_head is not None:
        head_stamp = int(accepted_head[STAMP_NANOSECONDS])
        head_receipt = int(accepted_head[RECEIPT_TIME_NANOSECONDS])
        head_capture = int(accepted_head[CAPTURE_TIME_NANOSECONDS])
        if stamp_nanoseconds <= head_stamp:
            reasons.append("stamp_not_strictly_increasing")
        if receipt_nanoseconds <= head_receipt:
            reasons.append("receipt_not_strictly_increasing")
        if capture_nanoseconds <= head_capture:
            reasons.append("capture_not_strictly_increasing")

    if reasons:
        return StationaryMapOdomCaptureDecision(
            accepted=False,
            reason="; ".join(reasons),
            reasons=tuple(reasons),
        )

    normalized = deepcopy(dict(candidate))
    return StationaryMapOdomCaptureDecision(
        accepted=True,
        reason="candidate_admitted",
        reasons=(),
        sample=normalized,
    )


def evaluate_stationary_map_odom_amcl_window_binding(
    direct_tf_samples: Sequence[object],
    *,
    amcl_receipt_nanoseconds: Sequence[object],
) -> StationaryMapOdomAmclWindowDecision:
    """Require direct-TF evidence to overlap the accepted AMCL window.

    Callback order is still intentionally ignored.  A direct TF may be received
    before the AMCL callback it belongs with, but a claimable direct-TF window
    may not be entirely older than the accepted stationary AMCL window.
    """

    reasons: list[str] = []
    if (
        isinstance(direct_tf_samples, (str, bytes, bytearray, Mapping))
        or not isinstance(direct_tf_samples, Sequence)
    ):
        reasons.append("direct_tf_window_not_sequence")
        direct_tf_values: tuple[object, ...] = ()
    else:
        direct_tf_values = tuple(direct_tf_samples)

    if (
        isinstance(amcl_receipt_nanoseconds, (str, bytes, bytearray, Mapping))
        or not isinstance(amcl_receipt_nanoseconds, Sequence)
    ):
        reasons.append("amcl_window_not_sequence")
        amcl_receipts: tuple[object, ...] = ()
    else:
        amcl_receipts = tuple(amcl_receipt_nanoseconds)

    if not direct_tf_values:
        reasons.append("direct_tf_window_empty")
    if not amcl_receipts:
        reasons.append("amcl_window_empty")
    if (
        direct_tf_values
        and amcl_receipts
        and len(direct_tf_values) != len(amcl_receipts)
    ):
        reasons.append("stationary_window_sample_count_mismatch")

    if reasons:
        return _amcl_window_decision(
            False,
            reasons,
            direct_tf_values,
            amcl_receipts,
        )

    direct_timing_reasons: list[str] = []
    for index, sample in enumerate(direct_tf_values):
        direct_timing_reasons.extend(
            f"direct_tf_{index}_{reason}"
            for reason in _timing_validation_reasons(
                sample,
                prefix="sample",
            )
        )
    reasons.extend(direct_timing_reasons)

    validated_amcl_receipts: list[int] = []
    for index, receipt in enumerate(amcl_receipts):
        if type(receipt) is not int or receipt < 0:
            reasons.append(
                f"amcl_{index}_receipt_time_nanoseconds_not_nonnegative_integer"
            )
            continue
        if (
            validated_amcl_receipts
            and receipt <= validated_amcl_receipts[-1]
        ):
            reasons.append("amcl_receipts_not_strictly_increasing")
        validated_amcl_receipts.append(receipt)

    if reasons:
        return _amcl_window_decision(
            False,
            reasons,
            direct_tf_values,
            amcl_receipts,
        )

    assert validated_amcl_receipts
    direct_capture_times = [
        int(sample[CAPTURE_TIME_NANOSECONDS])
        for sample in direct_tf_values
        if isinstance(sample, Mapping)
    ]
    if not direct_capture_times:
        reasons.append("direct_tf_window_empty")
        return _amcl_window_decision(
            False,
            reasons,
            direct_tf_values,
            amcl_receipts,
        )

    first_amcl_receipt = validated_amcl_receipts[0]
    last_tf_capture = direct_capture_times[-1]
    if last_tf_capture < first_amcl_receipt:
        reasons.append("direct_tf_window_predates_amcl_window")

    return _amcl_window_decision(
        not reasons,
        reasons or ["direct_tf_window_overlaps_amcl_window"],
        direct_tf_values,
        amcl_receipts,
    )


def _amcl_window_decision(
    accepted: bool,
    reasons: Sequence[str],
    direct_tf_samples: Sequence[object],
    amcl_receipts: Sequence[object],
) -> StationaryMapOdomAmclWindowDecision:
    reason_values = tuple(str(reason) for reason in reasons)
    direct_capture_times = [
        sample.get(CAPTURE_TIME_NANOSECONDS)
        for sample in direct_tf_samples
        if isinstance(sample, Mapping)
    ]
    amcl_receipt_values = list(amcl_receipts)
    return StationaryMapOdomAmclWindowDecision(
        accepted=accepted,
        reason="; ".join(reason_values),
        reasons=reason_values,
        direct_tf_window={
            "sample_count": len(direct_tf_samples),
            "first_capture_time_nanoseconds": (
                direct_capture_times[0] if direct_capture_times else None
            ),
            "last_capture_time_nanoseconds": (
                direct_capture_times[-1] if direct_capture_times else None
            ),
        },
        amcl_window={
            "sample_count": len(amcl_receipt_values),
            "first_receipt_time_nanoseconds": (
                amcl_receipt_values[0] if amcl_receipt_values else None
            ),
            "last_receipt_time_nanoseconds": (
                amcl_receipt_values[-1] if amcl_receipt_values else None
            ),
        },
    )


class StationaryMapOdomEpochCapture:
    """Bounded collector for one stationary direct-TF capture epoch."""

    def __init__(
        self,
        *,
        epoch_start_baseline: StationaryMapOdomEpochBaseline,
        required_count: int,
    ) -> None:
        if not isinstance(
            epoch_start_baseline,
            StationaryMapOdomEpochBaseline,
        ):
            raise ValueError(
                "epoch_start_baseline must be a "
                "StationaryMapOdomEpochBaseline"
            )
        if type(required_count) is not int or required_count <= 0:
            raise ValueError("required_count must be a positive integer")

        self.epoch_start_baseline = epoch_start_baseline
        self.required_count = required_count
        self._retained_samples: list[dict[str, object]] = []
        self._rejections: list[StationaryMapOdomCaptureRejection] = []
        self._candidate_sequence_index = 0

    @property
    def accepted_head(self) -> dict[str, object] | None:
        if not self._retained_samples:
            return None
        return deepcopy(self._retained_samples[-1])

    @property
    def retained_samples(self) -> tuple[dict[str, object], ...]:
        return tuple(deepcopy(sample) for sample in self._retained_samples)

    @property
    def rejections(self) -> tuple[StationaryMapOdomCaptureRejection, ...]:
        return tuple(self._rejections)

    def consider(
        self,
        candidate: object,
    ) -> StationaryMapOdomCaptureDecision:
        """Admit a monotonic candidate or append a non-mutating rejection."""

        sequence_index = self._candidate_sequence_index
        self._candidate_sequence_index += 1
        head = self._retained_samples[-1] if self._retained_samples else None
        decision = evaluate_stationary_map_odom_candidate(
            candidate,
            epoch_start_baseline=self.epoch_start_baseline,
            accepted_head=head,
        )
        if decision.accepted:
            assert decision.sample is not None
            self._retained_samples.append(deepcopy(decision.sample))
            if len(self._retained_samples) > self.required_count:
                self._retained_samples = self._retained_samples[
                    -self.required_count :
                ]
            return decision

        self._rejections.append(
            StationaryMapOdomCaptureRejection(
                candidate_sequence_index=sequence_index,
                reasons=decision.reasons,
                candidate_timing=_timing_for_audit(candidate),
                accepted_head_timing=(
                    _timing_for_audit(head) if head is not None else None
                ),
            )
        )
        return decision

    def window_result(self) -> StationaryMapOdomCaptureWindow:
        """Return reindexed samples only when the required window is full."""

        complete = len(self._retained_samples) == self.required_count
        samples: list[dict[str, object]] = []
        if complete:
            for index, retained_sample in enumerate(self._retained_samples):
                reindexed = deepcopy(retained_sample)
                reindexed["amcl_sample_index"] = index
                reindexed["stationary_epoch_sample_index"] = index
                samples.append(reindexed)

        return StationaryMapOdomCaptureWindow(
            complete=complete,
            required_count=self.required_count,
            retained_sample_count=len(self._retained_samples),
            samples=tuple(samples),
            retained_samples=self.retained_samples,
            rejections=self.rejections,
        )
