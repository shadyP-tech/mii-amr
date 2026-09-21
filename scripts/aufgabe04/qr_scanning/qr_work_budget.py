"""Cost-aware admission of atomic QR work; no images or decoded evidence kept."""

from collections import OrderedDict
import math


class QrWorkHistory:
    """Bounded recent peak costs, partitioned by operation and pixel count."""

    def __init__(self):
        self._costs = OrderedDict()
        self._generation = 0
        self._seen = {}

    def begin_frame(self):
        self._generation += 1
        for key in tuple(self._costs):
            if self._generation - self._seen[key] >= 8:
                del self._costs[key]
                del self._seen[key]

    def estimate(self, stage, pixels):
        costs = [(size, elapsed) for (name, size), elapsed in self._costs.items()
                 if name == stage]
        if not costs:
            return 0.  # First observation remains a cooperative, measured probe.
        size, elapsed = min(costs, key=lambda item: abs(math.log(pixels / item[0])))
        if pixels > 2 * size and self._generation % 8 == 0:
            # Extrapolation must not permanently exclude a never-measured
            # enlarged view that may be the only decodable one. Periodically
            # measure it under the same cooperative source-age protection.
            return 0.
        return .002 + 1.2 * elapsed * max(1., pixels / size)

    def record(self, stage, pixels, elapsed):
        key = stage, pixels
        # Slow calls remain conservative, but do not permanently set the cost
        # after a transient scheduling stall. No data-dependent result survives.
        self._costs[key] = max(elapsed, .95 * self._costs.get(key, 0.))
        self._seen[key] = self._generation
        self._costs.move_to_end(key)
        while len(self._costs) > 32:
            old, _ = self._costs.popitem(last=False)
            del self._seen[old]


def image_pixels(image):
    return max(1, int(image.shape[0]) * int(image.shape[1]))


class QrWorkBudget:
    def __init__(self, *, deadline, history, clock):
        self.deadline, self.history, self.clock = deadline, history, clock
        self.events = []
        history.begin_frame()

    def allow(self, stage, pixels):
        remaining = None if self.deadline is None else self.deadline - self.clock()
        estimate = self.history.estimate(stage, pixels)
        allowed = remaining is None or remaining > estimate
        if not allowed and len(self.events) < 32:
            self.events.append(dict(stage=stage, pixels=pixels, allowed=False,
                                   remaining_sec=remaining, estimated_sec=estimate))
        return allowed

    def measure(self, stage, pixels, operation):
        started = self.clock()
        try:
            return operation()
        finally:
            elapsed = max(0., self.clock() - started)
            self.history.record(stage, pixels, elapsed)
            if len(self.events) < 32:
                self.events.append(dict(stage=stage, pixels=pixels, elapsed_sec=elapsed))
