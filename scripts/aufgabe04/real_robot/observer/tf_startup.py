"""Nonblocking readiness of this observer's own TF buffer, before tuple retry."""

import time


class ObserverTfStartup:
    """Readiness is only a startup gate; every observation still uses exact TF.

    The parent's existing observation timeout bounds this wait. Neither source
    timestamps nor evidence epochs are extended when the buffer becomes ready.
    """

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._started = clock()
        self._next_report = -float("inf")
        self.ready = False
        self.waited_sec = 0.

    def poll(self, checks):
        if self.ready:
            return True, None
        now = self._clock()
        self.waited_sec = max(0., now - self._started)
        missing = [name for name, available in checks if not available()]
        self.ready = not missing
        report = None
        if self.ready or now >= self._next_report:
            report = dict(ready=self.ready, waited_sec=self.waited_sec, missing=missing,
                          exact_sensor_time_still_required=True)
            self._next_report = now + 1.
        return self.ready, report
