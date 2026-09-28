"""Bound fresh opposite-view association misses without authorizing motion."""
import math

STATE = 'opposite_identity_association_exhausted'


class OppositeIdentityOpportunity:
    def __init__(self):
        self.reset()

    def reset(self):
        self.context = self.anchor = self.start = self.last = None

    def observe(self, *, target_key, epoch, pose, image_stamp_sec, scan_stamp_sec,
                now_sec, poisoned, motion_epoch_reset, conflict):
        if (poisoned or motion_epoch_reset or not conflict
                or not all(math.isfinite(v) for v in (*pose, image_stamp_sec, scan_stamp_sec, now_sec))
                or any(not 0 <= now_sec-v <= .5 for v in (image_stamp_sec, scan_stamp_sec))
                or abs(image_stamp_sec-scan_stamp_sec) > .1):
            self.reset()
            return None
        context = (target_key, epoch)
        moved = self.anchor is not None and (math.dist(pose[:2], self.anchor[:2]) > .02
            or abs(math.remainder(pose[2]-self.anchor[2], math.tau)) > math.radians(2))
        if (context != self.context or moved or self.last is None
                or not 0 < image_stamp_sec-self.last <= 1.5):
            self.context, self.anchor, self.start = context, pose, image_stamp_sec
        self.last = image_stamp_sec
        if image_stamp_sec-self.start < 5.:
            return None
        return dict(reason='persistent_opposite_target_association_conflict',
            elapsed_sec=image_stamp_sec-self.start, image_stamp_sec=image_stamp_sec,
            recovery='bounded_inspection_view', motion_authorized=False,
            candidate_geometry_updated=False)
