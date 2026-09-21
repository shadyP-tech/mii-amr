"""Frame-local extraction shared by cold-search orderings, never measurements.

Only one invocation and its recursive retries own this cache. No pose, selected
proposal, association or identity is retained; those decisions rerun normally.
"""


class HeadSearchInputs:
    NAMES = frozenset({"locator_edges", "contours", "gray", "lsd", "hough", "distance"})

    def __init__(self, frame, raw_edges, edge_region):
        self._context = (frame, raw_edges, edge_region)
        self._values = {}

    def check_context(self, frame, raw_edges, edge_region):
        if any(old is not new for old, new in zip(
                self._context, (frame, raw_edges, edge_region))):
            raise ValueError("head search inputs require the same frame, edges and region")

    def get(self, name, extract):
        if name not in self.NAMES:
            raise ValueError("unsupported head search extraction")
        if name not in self._values:
            self._values[name] = extract()
        return self._values[name]
