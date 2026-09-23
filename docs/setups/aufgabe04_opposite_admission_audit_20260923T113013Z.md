# Opposite-side admission audit: 20260923T113013Z

The second inspected candidate, `survey_candidate_0001` (the visible `Start`
stand), timed out at its opposite-side view. The robot faced the stored candidate
center accurately, but that point disagreed with the current stand. The observer
then rejected every identity crop before payload decoding or centering could run.

## Evidence

- Run: `stand_explore_exact2_camera_all5_20260923T113013Z`.
- Workstation/run revision: `58f9c95678a4fb9083bf69073789002b50a5ca71`.
- Failing observer: `candidates/001_survey_candidate_0001/camera_lidar_attempt_01`.
- Viewer: `recording_20260923_134105_124402383`, 52 recorded frames.
- Local evidence: `results/implementation_checks/opposite_admission_audit_20260923T113013Z/`.
  `provenance.json` records the archive hash. `summarize.py` reproduces
  `summary.json`; `probe_geometry.py` reproduces the local geometry calculations.
  `deployed_decoder_probe.log` records the read-only workstation decoder check.
- Production code was not changed. No ROS or robot-motion commands were issued.

## The angle survived; the observed position did not reach planning

The first local view produced seven backside samples. Its retained orientation
uncertainty was ±7.96 degrees. The angle was re-expressed in the planning and
arrival map frames; the opposite observer records `current_angle_refit=false`.
The source angle was -1.50942 rad and the arrival-frame angle -1.54169 rad, with
the difference explicitly accounted for by the map/odom frame transform.

The source observation already contained a successful three-sample target
reconciliation. Its stored candidate center was (-1.13468, -0.38330) m; its
current scan search center was (-1.11069, -0.46276) m: **8.30 cm apart**.
The measured head center was around image x=248 while the stored center projected
near x=376. Thus the discrepancy was available before the opposite-side motion.

That reconciliation authorizes observation association, not a rewritten survey
position or motion target. `backside_axis_observation.py` still emits the supplied
`stand_x_m`/`stand_y_m` as `stand_center`. `_move_certified_opposite_face` in
`candidate/approach.py` uses the projected snapshot candidate geometry for its
route. `candidate_preapproach_compute.py:286` derives terminal yaw with `atan2`
toward that candidate center. There is no separate validated current-center
handoff into this planning step, and this terminal-yaw calculation does not
include the calibrated camera offset.

The 0.50 m and 0.45 m opposite routes failed uncertainty admission. The 0.40 m
route executed and reported `motion_completed`. At arrival:

| Quantity | Recorded value |
| --- | --- |
| Robot pose | (-1.55717, -0.35084, 0.01994 rad) |
| Projected stored candidate | (-1.15052, -0.34161) m |
| Stored-target range | 0.40675 m |
| Stored-target bearing error | **0.159 degrees** |
| `camera_centered` | `false` |

Nine sampled opposite-view scans place current cluster support 8.76–10.29 cm
from the projected stored center. These are scan-surface estimates, not an
independent ground-truth measurement of the stand center. The audit establishes
the disagreement, but does not uniquely attribute its origin to survey error,
localization drift, or support geometry.

## Why decoding and centering never started

The opposite observer processed **396 source-fresh tuples**:

| Result | Frames |
| --- | ---: |
| `target_crop_overlap_unresolved` | 388 |
| `qr_search_cluster_not_unique` | 8 |
| Payload decoder invoked | **0** |
| Accepted observation / QR samples | **0** |

The first frame's current scan search projects near x=548 and makes a broad
identity box. Stored candidate `0003` projects to a competing box
`[553,317,603,366]`, overlapping that identity region. The existing overlap
resolver can preserve and isolate the complete foreground QR, but only after
`detect_opposite_target_support` has validated its current outline and bearing.

That support still uses the original stored-target bearing. For nine sampled
mission images, native detection locates the complete QR near x=567. Its
translation-corrected bearing differs from the stored target by **13.30–13.51
degrees**, with another **1.56–1.58 degrees** of range-interval uncertainty.
The support function requires the entire interval within 12 degrees, so it
returns no support. Without that support, projected overlap remains unresolved;
`process_opposite_identity` consequently never calls the payload decoder.

The recent ordinary-observer reconciliation does not run on this branch:
`observer/node.py:1591` dispatches opposite identity and returns before
`StoppedTargetReconciliation.observe` at approximately line 1799. Opposite-side
centering also requires both accepted support and an accepted crop
(`opposite_identity.py:84`). The same rejection therefore blocks decoding and
the very centering adjustment intended to improve framing.

There is a second, independent centering limit. The calibrated diagnostic turn
needed to center this QR is approximately **13.2 degrees clockwise**. The existing
solver requires the complete correction to fit within its **12-degree total
budget**; replaying it raises `head cannot be centered within the bounded yaw
interval`. This is a counterfactual additional blocker: the live run never
reached that solver because target support failed first. Merely enabling the
shared reconciliation is therefore not a complete correction.

## What the recording and deployed decoder confirm

The viewer has comparable visible framing: its accepted head centers span
x=562.8–566.9, median 563.1, in an 800-pixel image. It reports 21 usable head
fits, 26 ambiguous proposals, and 5 obsolete results. It uses `head_target=nearest`
and `no_qr_decode=true`; its overlay is not proof of mission admission. It begins
73 seconds after the last processed opposite-view tuple. Similar images do not
prove exact equality of robot pose.

A read-only check using the workstation's installed **OpenCV 4.5.4/WeChat** on
the actual failed-run `frame_000002` detects the full QR outline and decodes
**`Start`** from the current scan crop at scale 4. The native outline agrees with
the local OpenCV 5 geometry replay. No replacement repository code was uploaded.
The saved image was decodable; the live branch's preceding gates prevented the
decode attempt.

TF retry exhaustion occurred 33 times, but all 396 processed observations had
accepted source freshness (image ages 0.222–0.418 s, scan ages 0.187–0.426 s).
The final TF status does not explain the repeated crop failures. The observer
ended at its deadline, was interrupted and reaped with return code 130. The
parent then tried bounded route recovery. The copied artifacts do not establish
a terminal mission-wide failure.

## Recommended correction

1. Carry a separately validated current stand-center estimate and uncertainty
   alongside the immutable survey candidate and retained backside angle. Recheck
   candidate identity, competing candidates and frame provenance before using
   that estimate for opposite-side planning.
2. Share current-target reconciliation across ordinary and opposite acquisition,
   before the branch split. Neither nearest selection nor a historical angle
   alone can establish current target identity.
3. Separate target support for a bounded centering action from permission to
   decode an exclusive identity crop. Keep the overlap check for admission, but
   avoid making an unresolved projected overlap block all target-directed recovery.
4. Compute arrival framing from the validated current target and calibrated
   camera transform. If the correction exceeds the existing turn budget, emit an
   explicit bounded recovery request instead of silently returning no advisory
   and spending the full camera timeout. Do not widen motion/identity bounds
   merely to make this replay pass.
5. Decode the preserved complete target and immediately admit a fresh associated
   `Start`, retaining the backside orientation without fitting another stand angle.

Regression coverage should include this opposite-side producer path, the
8–10 cm stored/current discrepancy, foreground QR with projected background
overlap, and an out-of-budget centering request. The previous ordinary QR
reconciliation tests did not exercise this branch.
