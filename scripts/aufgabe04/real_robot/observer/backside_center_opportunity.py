"""Bound the wait between backside angle readiness and independent center proof."""
import time

MAX_WAIT_SEC = 1.5


def backside_center_pending(adapter, *, center_ready, metadata, now=None):
    # Standalone angle observers without a candidate population cannot perform
    # reconciliation. Their receipts keep the existing explicit limitation.
    if getattr(adapter.args, 'candidate_crop_snapshot', None) is None:
        return False
    now = time.monotonic() if now is None else now
    evidence = adapter.observation_evidence.snapshot()
    context = (evidence.target_key, evidence.motion_epoch)
    previous = getattr(adapter, '_backside_center_opportunity', None)
    if previous is None or previous[0] != context:
        previous = (context, now)
        adapter._backside_center_opportunity = previous
    pending = not center_ready and 0 <= now-previous[1] < MAX_WAIT_SEC
    adapter._backside_center_pending_until = previous[1]+MAX_WAIT_SEC if pending else None
    metadata['backside_center_opportunity'] = dict(
        ready=center_ready, pending=pending, budget_sec=MAX_WAIT_SEC,
        reason=('validated_center_ready' if center_ready else 'collecting_center_after_angle'
                if pending else 'center_opportunity_exhausted'), motion_authorized=False)
    return pending


def backside_center_grace_pending(adapter):
    deadline = getattr(adapter, '_backside_center_pending_until', None)
    previous = getattr(adapter, '_backside_center_opportunity', None)
    if deadline is None or previous is None:
        return False
    snapshot = adapter.observation_evidence.snapshot()
    return (previous[0] == (snapshot.target_key, snapshot.motion_epoch)
            and time.monotonic() < deadline and not snapshot.poisoned)
