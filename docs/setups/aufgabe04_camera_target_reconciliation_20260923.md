# Bounded nearest acquisition and candidate reconciliation

Implements the correction identified in
[the first-inspection audit](aufgabe04_camera_admission_audit_20260923T101959Z.md)
for `survey_candidate_0005` / `QR_004` in run `20260923T101959Z`.

## Behavior

The physical exploration observer uses the viewer's shared nearest-scan head
selector. It first excludes clusters outside the selected candidate's image
search region and existing scan range interval. The selected location guides
current-pixel head fitting; it does not authorize candidate association. An
invalid candidate region cannot trigger unrestricted full-image acquisition.

`observer/finite_target_bearing.py` intersects a calibrated camera ray with
the current scan range, including camera translation. It carries the bearing
uncertainty over the full accepted range interval. QR association, opposite-side
target support and centering share this projection implementation.

`observer/target_reconciliation.py` reconciles displaced current support with
the admitted candidate snapshot. It requires three independently stamped,
fresh, synchronized scans/images while stopped. Each scan must contain one
compact cluster with at least three samples in the original registration
envelope; even one-sample competing clusters prevent uniqueness. The cluster
centroid remains within the original 12-degree registration bound. Candidate
displacement and cluster diameter are bounded by twice radius plus uncertainty,
capped at 0.16 m. Another stored candidate within the combined uncertainty
envelopes prevents reconciliation.

History expires after 1.5 seconds. Robot displacement above 0.02 m or 2 degrees,
target displacement above 0.03 m, reused samples, invalid sources or changed
candidate/epoch reset it. Snapshot identity, raw scans, exact transform values,
source stamps and original bounds travel with the proof. Discovery and
centering receipt validators recheck this evidence. The frozen survey geometry
is never rewritten.

The current camera bearing, including its uncertainty interval, must then fall
within 3 degrees of that validated cluster. Validated current head support can
feed the existing bounded centering controller. Its rotation limits and fresh
post-turn observation requirements remain unchanged. A fresh, uniquely bound
QR completes discovery immediately after reconciliation, without angle consensus
or the ordinary geometry grace. Certified opposite-side identity continues to
retain its backside orientation.

Cluster comparison ignores elapsed-age diagnostic differences caused by image
processing, while independently enforcing freshness. Decoded-but-unbound QR
counts and the last association reason survive subsequent TF status updates.

## Validation and limits

- Focused observer, candidate, QR, receipt, centering and acquisition suites:
  **291 passed, 1 skipped, 313 subtests passed**.
- The recorded three-scan fixture admits the previously rejected `QR_004`
  corners. Tests reject neighboring candidates, stale/reused/moving sources,
  changed bounds, altered receipts and epoch rebinding. They also exercise
  processing-time advancement and immediate completion with zero axis samples.
- Local OpenCV 5 replay selects the intended recorded nearest cluster at
  approximately image x=222. Its head-border fit still reports
  `model_current_head_border_unavailable`; this is not evidence of improved
  geometry detection rate. QR association regression uses the actual recorded
  decoder corners rather than claiming a new decoder replay.
- The workstation OpenCV 4.5.4/WeChat replay was not executed: automatic approval
  review rejected sending modified repository source to `mii001` over SSH
  without explicit payload/destination authorization. No code was deployed and
  no robot motion was executed.

The existing exploration command needs no additional flags. Ordinary candidate
captures now receive the admitted candidate snapshot automatically. Live ROS
timing, deployed decoder behavior and robot execution remain to be validated.
