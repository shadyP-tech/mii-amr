"""Measured, non-overlapping OpenCV text layout for diagnostic viewers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextBounds:
    left_px: int
    top_px: int
    right_px: int
    bottom_px: int


class OverlayTextCursor:
    """Stack and wrap status rows using OpenCV's measured text dimensions."""

    def __init__(
        self,
        *,
        left_px: int = 12,
        top_px: int = 8,
        right_margin_px: int = 48,
        row_gap_px: int = 4,
    ) -> None:
        self.left_px = max(0, int(left_px))
        self._next_top_px = max(0, int(top_px))
        self.right_margin_px = max(0, int(right_margin_px))
        self.row_gap_px = max(0, int(row_gap_px))

    @property
    def bottom_px(self) -> int:
        return self._next_top_px

    @staticmethod
    def _text_size(cv2, text, font_face, font_scale, thickness):
        (width, height), baseline = cv2.getTextSize(
            text,
            font_face,
            font_scale,
            thickness,
        )
        return int(width), int(height), max(0, int(baseline))

    def _wrapped_lines(
        self,
        cv2,
        frame,
        text: str,
        *,
        font_face,
        font_scale: float,
        thickness: int,
    ) -> tuple[str, ...]:
        maximum_width = max(
            40,
            int(frame.shape[1]) - self.left_px - self.right_margin_px,
        )
        words = str(text).split()
        if not words:
            return ("",)
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            width, _height, _baseline = self._text_size(
                cv2,
                candidate,
                font_face,
                font_scale,
                thickness,
            )
            if width <= maximum_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return tuple(lines)

    def draw(
        self,
        cv2,
        frame,
        text: str,
        *,
        font_face,
        font_scale: float,
        color,
        thickness: int,
    ) -> tuple[TextBounds, ...]:
        """Draw one logical label, wrapping it into reserved non-overlapping rows."""

        bounds: list[TextBounds] = []
        for line in self._wrapped_lines(
            cv2,
            frame,
            text,
            font_face=font_face,
            font_scale=font_scale,
            thickness=thickness,
        ):
            width, height, baseline = self._text_size(
                cv2,
                line,
                font_face,
                font_scale,
                thickness,
            )
            origin_y = self._next_top_px + height
            cv2.putText(
                frame,
                line,
                (self.left_px, origin_y),
                font_face,
                font_scale,
                color,
                thickness,
            )
            bottom = origin_y + baseline
            bounds.append(
                TextBounds(
                    left_px=self.left_px,
                    top_px=self._next_top_px,
                    right_px=self.left_px + width,
                    bottom_px=bottom,
                )
            )
            self._next_top_px = bottom + self.row_gap_px
        return tuple(bounds)
