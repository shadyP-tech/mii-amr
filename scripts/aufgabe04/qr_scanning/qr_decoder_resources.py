"""Single-owner reusable decoder objects and costs, without frame evidence."""

import threading

from scripts.aufgabe04.qr_scanning.qr_work_budget import QrWorkHistory


class QrDecoderResources:
    def __init__(self, cv2):
        self.cv2 = cv2
        self.history = QrWorkHistory()
        self._owner = threading.get_ident()
        self._decoders = {}

    def check_owner(self, cv2):
        if cv2 is not self.cv2 or threading.get_ident() != self._owner:
            raise RuntimeError("QR resources require their original backend and owner thread")

    def decoder(self, name):
        self.check_owner(self.cv2)
        if name not in {"native", "wechat"}:
            raise ValueError("unknown QR decoder")
        if name not in self._decoders:
            factory = getattr(self.cv2, "QRCodeDetector" if name == "native"
                              else "wechat_qrcode_WeChatQRCode", None)
            try:
                self._decoders[name] = factory() if factory is not None else None
            except Exception:
                # A transient initialization failure must not disable this
                # backend for every subsequent image in the observer.
                return None
        return self._decoders[name]
