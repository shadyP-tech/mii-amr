"""Canonical in-memory calibration context for tracking and admission receipts.

ROS CameraInfo arrays can contain NumPy scalar numbers. Convert them at the
producer, before any receipt validation: serialization is too late and can hide
this difference. Calibration agreement remains the caller's responsibility.
"""

import math
from numbers import Real


def _numbers(values):
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("camera context requires finite real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("camera context requires finite real numbers")
        output.append(number)
    return tuple(output)


def _text(value):
    if not isinstance(value, str) or len(value) > 256:
        raise ValueError("camera context requires bounded frame/model text")
    return str(value)


def camera_context_signature(*, camera_frame, intrinsics, camera_info,
                             scan_translation, scan_rotation):
    """Copy the full validated context into immutable Python scalar values.

    Preserve matrix coefficients, distortion and scan extrinsics exactly; a
    change in any of them must still reset the associated tracking/QR context.
    Do not coerce numeric strings, booleans, complex or nonfinite values into a
    calibration. The artifact validators intentionally retain their strict
    built-in numeric type checks.
    """
    return (
        _text(camera_frame), *_numbers(intrinsics),
        *(_numbers(getattr(camera_info, field, ())) for field in ("k", "d", "r", "p")),
        _text(getattr(camera_info, "distortion_model", "")),
        _numbers(scan_translation), _numbers(scan_rotation),
    )
