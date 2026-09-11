"""Two bounded views of one native QR proposal for payload confirmation.

The second view retains a small margin of original pixels around the native
quad. Its purpose is to avoid cutting antialiased outer modules at imperfect
native corners; that margin never replaces the original symbol coordinates.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IsolatedQrView:
    name: str
    symbol_size_px: int
    source_margin_ratio: float
    quiet_border_px: int = 16


ISOLATED_QR_VIEWS = (
    IsolatedQrView("native_quad", 224, 0.0),
    IsolatedQrView("source_quiet_margin", 160, 0.04),
)


def rectify_isolated_qr_view(frame, corners, cv2, view: IsolatedQrView):
    """Sample this quad with at most a four-percent source margin per side."""
    import numpy

    margin = round(view.symbol_size_px * view.source_margin_ratio)
    last = margin + view.symbol_size_px - 1
    transform = cv2.getPerspectiveTransform(
        numpy.asarray(corners, dtype=numpy.float32),
        numpy.asarray(((margin, margin), (last, margin), (last, last), (margin, last)),
                      dtype=numpy.float32),
    )
    white = 255 if len(frame.shape) == 2 else (255, 255, 255)
    side = view.symbol_size_px + 2 * margin
    symbol = cv2.warpPerspective(
        frame, transform, (side, side), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=white,
    )
    border = view.quiet_border_px
    return cv2.copyMakeBorder(symbol, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=white)
