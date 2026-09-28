"""Short, observation-only opportunity to bind an epoch-recovered target."""
import math

from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import is_epoch_recovery
from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation

STATE = 'candidate_position_epoch_inconsistent'


class PositionEpochOpportunity:
    """Five seconds of fresh, unbound frames; gaps and productive frames reset it."""
    def __init__(self):
        self.context = None
        self.start = self.last = None

    def reset(self):
        self.context = self.start = self.last = None

    def observe(self, proof, frame, *, now_sec):
        if (not is_epoch_recovery(proof) or not frame or frame.get('poisoned') is not False
                or frame.get('motion_epoch_reset') or frame.get('frame_accepted')):
            self.reset()
            return None
        stamp = frame['frame_stamp_sec']
        if not math.isfinite(now_sec) or not 0 <= now_sec-stamp <= .5:
            self.reset()
            return None
        try:
            validate_reconciliation(proof, image_stamp_sec=stamp, scan_stamp_sec=frame['scan_stamp_sec'])
        except (ValueError, TypeError, KeyError, OSError):
            self.reset()
            return None
        context = (proof['target_key'], proof['epoch'], proof['snapshot_sha256'],
                   proof['entries'][-1]['position_epoch']['sha256'])
        if context != self.context or self.last is None or not 0 < stamp-self.last <= 1.5:
            self.context, self.start = context, stamp
        self.last = stamp
        if stamp-self.start < 5.:
            return None
        return dict(reason='persistent_current_cluster_without_camera_identity_binding',
                    elapsed_sec=stamp-self.start, image_stamp_sec=stamp,
                    recovery='bounded_inspection_view', motion_authorized=False,
                    candidate_geometry_updated=False)
