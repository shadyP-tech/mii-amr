"""Slow-call forecasts prevent new work without reusing old QR evidence."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading

import pytest

from scripts.aufgabe04.qr_scanning.qr_work_budget import QrWorkBudget, QrWorkHistory
from scripts.aufgabe04.qr_scanning.qr_decoder_resources import QrDecoderResources
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr


def test_cost_admission_records_overruns_and_expires_old_slow_forecasts():
    now, history = [0.], QrWorkHistory()
    budget = QrWorkBudget(deadline=.12, history=history, clock=lambda: now[0])
    assert budget.allow("wechat", 10000)
    budget.measure("wechat", 10000, lambda: now.__setitem__(0, .16))
    now[0] = 0.
    following = QrWorkBudget(deadline=.12, history=history, clock=lambda: now[0])
    assert not following.allow("wechat", 10000)
    assert following.events[-1]["estimated_sec"] > .16
    for _ in range(7):
        history.begin_frame()
    assert following.allow("wechat", 10000)  # A transient stall cannot poison all later probes.


def test_history_storage_is_bounded():
    history = QrWorkHistory()
    for pixels in range(1, 101):
        history.record("native", pixels, .01)
    assert len(history._costs) == len(history._seen) == 32


def test_size_extrapolation_cannot_permanently_exclude_unmeasured_enlarged_views():
    history = QrWorkHistory()
    for call in range(1, 9):
        budget = QrWorkBudget(deadline=.12, history=history, clock=lambda: 0.)
        history.record("wechat", 10000, .05)
        assert budget.allow("wechat", 160000) == (call == 8)


def test_resources_retry_failed_construction_and_reject_another_owner():
    decoder = object()
    backend = SimpleNamespace(QRCodeDetector=Mock(side_effect=[RuntimeError("temporary"), decoder]))
    resources = QrDecoderResources(backend)
    assert resources.decoder("native") is None
    assert resources.decoder("native") is decoder
    assert resources.decoder("native") is decoder
    assert backend.QRCodeDetector.call_count == 2
    failures = []
    def other_owner():
        try:
            resources.decoder("native")
        except RuntimeError as error:
            failures.append(error)
    thread = threading.Thread(target=other_owner)
    thread.start()
    thread.join(1.)
    assert len(failures) == 1
    with pytest.raises(RuntimeError):
        resources.check_owner(object())


def test_reused_decoder_does_not_reuse_payload():
    frame = SimpleNamespace(shape=(100, 100, 3))
    decoder = SimpleNamespace(detectAndDecodeMulti=Mock(return_value=(False, (), None)),
        detectAndDecode=Mock(side_effect=[("QR_003", None), ("", None)]))
    backend = SimpleNamespace(QRCodeDetector=Mock(return_value=decoder))
    resources = QrDecoderResources(backend)
    with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
               side_effect=lambda *_args, **_kwargs: iter(((frame, 1., 0),))):
        first = detect_qr_observations_bgr(frame, backend, resources=resources)
        second = detect_qr_observations_bgr(frame, backend, resources=resources)
    assert [observation.text for observation in first] == ["QR_003"]
    assert second == ()
    assert backend.QRCodeDetector.call_count == 1


def test_reused_backend_does_not_carry_multi_symbol_rejection_to_next_image():
    frame = SimpleNamespace(shape=(100, 100, 3))
    quad = ((20., 20.), (60., 20.), (60., 60.), (20., 60.))
    neighbor = tuple((u + 10., v + 10.) for u, v in quad)
    decoder = SimpleNamespace(
        detectAndDecodeMulti=Mock(side_effect=[(False, (), (quad, neighbor)), (False, (), None)]),
        detectAndDecode=Mock(side_effect=[("QR_003", quad), ("QR_004", quad)]))
    backend = SimpleNamespace(QRCodeDetector=lambda: decoder)
    resources = QrDecoderResources(backend)
    with patch("scripts.aufgabe04.qr_scanning.opencv_qr_detector._qr_decode_candidates_with_geometry",
               side_effect=lambda *_args, **_kwargs: iter(((frame, 1., 0),))):
        first = detect_qr_observations_bgr(frame, backend, resources=resources)
        second = detect_qr_observations_bgr(frame, backend, resources=resources)
    assert first[0].text == "QR_003" and first[0].corners is None
    assert second[0].text == "QR_004" and second[0].corners == quad


def test_known_slow_backend_is_not_started_or_resized_into_remaining_budget():
    frame = SimpleNamespace(shape=(300, 300, 3))
    decoder = SimpleNamespace(detectAndDecodeMulti=Mock(), detectAndDecode=Mock())
    wechat = SimpleNamespace(detectAndDecode=Mock())
    backend = SimpleNamespace(QRCodeDetector=lambda: decoder,
        wechat_qrcode_WeChatQRCode=lambda: wechat, resize=Mock())
    resources = QrDecoderResources(backend)
    for stage in ("opencv_multi", "opencv_single", "wechat"):
        resources.history.record(stage, 90000, .2)
    diagnostics = {}
    assert detect_qr_observations_bgr(frame, backend, resources=resources,
        max_elapsed_sec=.12, diagnostics=diagnostics) == ()
    decoder.detectAndDecodeMulti.assert_not_called()
    decoder.detectAndDecode.assert_not_called()
    wechat.detectAndDecode.assert_not_called()
    backend.resize.assert_not_called()
    assert any(event.get("allowed") is False for event in diagnostics["work_stages"])
