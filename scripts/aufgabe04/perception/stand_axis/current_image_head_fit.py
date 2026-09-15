"""One-use geometry reuse while recovering QR identity from the same crop.

Create inside one ROI evaluation and discard before the next ROI or image.
Only the undecorated physical head result may be retained: marker evidence and
front/back classification must be recomputed after QR recovery. This object
neither stores observation time nor grants freshness or motion admission.
"""

from hashlib import sha256


class CurrentImageHeadFit:
    """Retain one current fit for at most one exact-context QR refresh."""

    def __init__(self):
        self._pending = None
        self._started = False
        self.reused = False

    def compute(self, frame, *, context, producer):
        # Keep the array itself alive; identical bytes in another frame are
        # never the same observation, and mutation of this frame invalidates it.
        key = (
            tuple(frame.shape), frame.dtype.str, frame.strides,
            frame.__array_interface__["data"][0],
            sha256(frame.tobytes()).digest(), context,
        )
        pending, self._pending = self._pending, None
        self.reused = bool(pending is not None and pending[0] is frame and pending[1] == key)
        if self.reused:
            return pending[2]
        value = producer()  # Failed computations are never retained.
        if not self._started:
            self._pending = (frame, key, value)
        self._started = True
        return value
