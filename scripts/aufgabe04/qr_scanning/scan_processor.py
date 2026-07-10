"""ROS-free QR scan outcome processing for Aufgabe 04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .qr_id_decoder import decode_qr_id


@dataclass(frozen=True)
class ScanProcessorConfig:
    robot_id: str
    run_id: str = ""
    min_repeat_sec: float = 2.0
    max_frame_age_sec: float = 1.0
    confidence: float = 1.0


@dataclass(frozen=True)
class QRScanOutcome:
    status: str
    raw_text: str
    qr_id: str = ""
    reason: str = ""
    row: Mapping[str, object] | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


class QRScanProcessor:
    """Classify detected QR text and construct scan-log evidence rows."""

    def __init__(self, config: ScanProcessorConfig):
        self.config = config
        self._last_accept_sec_by_qr_id: dict[str, float] = {}
        self._last_reject_sec_by_key: dict[tuple[str, str], float] = {}

    def process_texts(
        self,
        raw_texts: tuple[str, ...],
        *,
        source: str,
        receipt_time_sec: float,
        stamp_sec: float | None = None,
    ) -> tuple[QRScanOutcome, ...]:
        reason = self._frame_rejection_reason(
            receipt_time_sec=receipt_time_sec,
            stamp_sec=stamp_sec,
        )
        return tuple(
            self._process_text(
                raw_text,
                source=source,
                receipt_time_sec=receipt_time_sec,
                stamp_sec=stamp_sec,
                frame_rejection_reason=reason,
            )
            for raw_text in raw_texts
        )

    def _process_text(
        self,
        raw_text: str,
        *,
        source: str,
        receipt_time_sec: float,
        stamp_sec: float | None,
        frame_rejection_reason: str,
    ) -> QRScanOutcome:
        evidence_timestamp = stamp_sec if stamp_sec is not None else receipt_time_sec
        if frame_rejection_reason:
            return self._rejected(
                raw_text,
                reason=frame_rejection_reason,
                source=source,
                timestamp_sec=evidence_timestamp,
                now_sec=receipt_time_sec,
            )

        try:
            scanned = decode_qr_id(
                raw_text,
                confidence=self.config.confidence,
                source=source,
                timestamp_sec=evidence_timestamp,
            )
        except ValueError as exc:
            return self._rejected(
                raw_text,
                reason=str(exc),
                source=source,
                timestamp_sec=evidence_timestamp,
                now_sec=receipt_time_sec,
            )

        last_accept_sec = self._last_accept_sec_by_qr_id.get(scanned.qr_id)
        if (
            last_accept_sec is not None
            and receipt_time_sec - last_accept_sec < self.config.min_repeat_sec
        ):
            return QRScanOutcome(
                status="debounced",
                raw_text=raw_text,
                qr_id=scanned.qr_id,
                reason="repeat_scan",
            )

        self._last_accept_sec_by_qr_id[scanned.qr_id] = receipt_time_sec
        row = {
            "timestamp": evidence_timestamp,
            "run_id": self.config.run_id,
            "robot_id": self.config.robot_id,
            "raw_text": scanned.raw_text,
            "qr_id": scanned.qr_id,
            "resolved_station_id": "",
            "source": scanned.source,
            "confidence": scanned.confidence,
            "status": "accepted",
            "reason": "",
        }
        return QRScanOutcome(status="accepted", raw_text=raw_text, qr_id=scanned.qr_id, row=row)

    def _rejected(
        self,
        raw_text: str,
        *,
        reason: str,
        source: str,
        timestamp_sec: float,
        now_sec: float,
    ) -> QRScanOutcome:
        key = (raw_text, reason)
        last_reject_sec = self._last_reject_sec_by_key.get(key)
        if (
            last_reject_sec is not None
            and now_sec - last_reject_sec < self.config.min_repeat_sec
        ):
            return QRScanOutcome(status="debounced", raw_text=raw_text, reason=reason)

        self._last_reject_sec_by_key[key] = now_sec
        row = {
            "timestamp": timestamp_sec,
            "run_id": self.config.run_id,
            "robot_id": self.config.robot_id,
            "raw_text": raw_text,
            "qr_id": "",
            "resolved_station_id": "",
            "source": source,
            "confidence": "",
            "status": "rejected",
            "reason": reason,
        }
        return QRScanOutcome(status="rejected", raw_text=raw_text, reason=reason, row=row)

    def _frame_rejection_reason(
        self,
        *,
        receipt_time_sec: float,
        stamp_sec: float | None,
    ) -> str:
        if stamp_sec is None or self.config.max_frame_age_sec <= 0:
            return ""
        age_sec = receipt_time_sec - stamp_sec
        if age_sec < -0.5:
            return ""
        if age_sec > self.config.max_frame_age_sec:
            return "stale_frame"
        return ""
