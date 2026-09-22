"""Bounded witness lifecycle diagnostics, isolated from admission state."""
from collections import Counter, deque
from copy import deepcopy
import math


class ScanWitnessDiagnostics:
    def __init__(self):
        self.counts = Counter()
        self.events = deque(maxlen=32)

    def record(self, stage, *, stamp=None, reason=None, witness_count=None, association=None, index_groups=None):
        self.counts[stage] += 1
        event = dict(stage=stage, scan_stamp_sec=(stamp if type(stamp) in (int, float)
            and math.isfinite(stamp) else None), reason=None if reason is None else str(reason)[:256], witness_count=witness_count)
        if index_groups is not None:
            event["raw_fragment_source_indices"] = [list(group[:16]) for group in index_groups[:8]]
        if association is not None:
            search = association.search_association
            if search is not None:
                indices = search.selected_cluster_source_indices
                event.update(eligible_cluster_count=search.eligible_cluster_count,
                    selected_source_indices=list(indices[:16]), selected_source_index_count=len(indices),
                    topology=dict(search.scan_topology or {}))
        if self.events and event == {k: v for k, v in self.events[-1].items() if k != "repeated_count"}:
            self.events[-1]["repeated_count"] = self.events[-1].get("repeated_count", 1)+1
        else:
            self.events.append(event)

    def snapshot(self):
        return dict(counts=dict(self.counts), recent_events=deepcopy(list(self.events)),
                    maximum_events=32, diagnostic_only=True, motion_authorized=False)

    def clone(self):
        result = ScanWitnessDiagnostics()
        result.counts = self.counts.copy()
        # Recording only mutates a top-level repeat count. Nested scan summaries
        # are immutable internally; public snapshots receive independent copies.
        result.events.extend(dict(event) for event in self.events)
        return result
