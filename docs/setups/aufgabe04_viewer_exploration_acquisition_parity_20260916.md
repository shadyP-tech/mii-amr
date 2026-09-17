# Shared viewer and exploration acquisition — 2026-09-16

Camera exploration now uses the same full-image geometry entry point and pose
search tracker as the supplied real-camera debug viewer command. The physical
model first acquires or tracks current head borders, fits their 3D geometry,
then applies the mission's current candidate association. Projected candidate
position, expected head size and preliminary LiDAR proposal filtering no longer
control which geometry the acquisition stage can see.

This addresses the orchestration mismatch documented in the
[latest first-candidate audit](aufgabe04_first_candidate_viewer_mismatch_audit_20260916T150723Z.md).
It does not change the shared fitter's handling of competing physical borders
or establish that every scene can cold-acquire successfully.

## Modules and behavior

- `perception/stand_axis/head_geometry_acquisition.py` is the shared entry point
  used by the viewer and observer. Both pass original image coordinates and
  matching intrinsics. The observer uses the viewer's eight-pixel minimum edge
  height instead of deriving that acquisition parameter from the map projection.
- The shared tracker factory uses a 250 ms observation freshness limit, a
  two-second search-hint lifetime and at most two soft misses. A hint locates
  new pixels; every angle is fitted again from the current image. Mission
  target, calibration, image shape, stationary epoch and cumulative pose changes
  still invalidate its search context.
- `real_robot/observer/viewer_head_acquisition.py` runs geometry once, then
  scopes QR decoding and finder checks to the complete detected head. On a head
  miss, identity probing can use the bounded projected fallback crop. QR corners
  translate to full-image coordinates exactly once. QR size or pose never
  selects or rescales head geometry.
- Backside classification is attached after geometry using current marker
  checks. An empty decode or a skipped/late finder check cannot prove absence.
  Side classification preserves the exact fitted angle and current boundary.
- `tracked_head_registration.py` now accepts explicit full-image acquisition
  provenance, including cold detections. The current geometry must still match
  the candidate's original projection, scale, unique current scan cluster and
  freshness checks before it supplies a receipt. A cold fit is not mislabeled
  as a reused track or a proposal retry.

Geometry completion is timed before optional QR work. A timely geometry fit can
therefore retain a search hint even when later QR processing overruns the current
result budget. That late combined result still cannot admit a candidate.
Invalid or behind-camera map projections do not stop image acquisition; they
remain rejected during association and serialize as `null` diagnostic pixels.

The existing immediate frontside rule remains: fresh accepted current geometry
plus a correctly bound QR can admit on the first successful frame. A failed
association may retain geometry search information, but grants no candidate
identity, receipt or motion authority. Current observer image/scan publication
limits and route-admission policy remain in force.

## Validation and deployment

Regression tests exercise full-image input parity, cold/tracked arguments,
current-pixel fitting with missing-pixel rejection, first-frame frontside
admission, association failure and recovery, timely geometry with late QR,
late geometry, context invalidation, head-scoped QR coordinates, explicit
marker absence, backside angle preservation and invalid candidate projection.

Final `tests/aufgabe04` results: **3,735 passed, 21 failed, 1 skipped**, plus
3,221 passing subtests. All 21 failures reproduce in an archive of unchanged
commit `ae5f39a6cdfff56055ca5a5396ad97350c072a36` under the same local environment;
there are no new failing cases. The suite is not globally green. Python 3.10
syntax compatibility and `git diff --check` pass. Test logs and the comparison
are in `results/aufgabe04/debug_audits/stand_explore_exact2_camera_all5_20260916T150723Z/`,
including `viewer_acquisition_test_summary.json`.

The implementation is local. The original mission images and recordings remain
on mii001. Two attempts at the authorized read-only, in-memory patched replay
failed to connect to mii001 (`10.42.0.1:22` timed out). No patched replay result or
hardware success is claimed. The prior audit's successful diagnostic fits used
the earlier deployed code and must not be relabeled as validation of this patch.

No new experiment flags are required after the updated code is deployed. The
existing ROI flags now scope fallback identity search for the physical model;
they do not switch its geometry back to projected-crop acquisition. No robot
motion or remote checkout modification was performed.
