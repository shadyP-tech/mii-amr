# Second candidate: QR fallback, backside axis, and viewer audit — 22 September 2026

The third local view did not reach QR-only admission because the operational
geometry-first path starved QR decoding of its reserved time. No associated QR
sample existed, so the new 1.5-second grace never started. A separate offline
replay also exposes a downstream binding problem: the visible QR is about 6.6°
away from the mapped bearing, outside the unregistered QR path's 3° cone.

At the earlier backside view, visual angle estimation mostly succeeded. The
blocking gate was current LiDAR association across the scan endpoints, with
insufficient independent witnesses. Only one of the required seven backside
samples was admitted. Consequently no certified axis receipt existed to enter
the opposite-side branch.

## Evidence and scope

- Mission: `stand_explore_exact2_camera_all5_20260922T121851Z`.
- Workstation and local revision: `bab57a20687ccdde1f225cb7b7ecc7662c180d27`.
- Second candidate: `001_survey_candidate_0001`.
- Third physical view: `camera_lidar_attempt_02`, including two centering turns
  and final observer `recenter_02/capture`.
- Latest viewer: `recording_20260922_143442_350245680`, 62 frames from
  12:34:41.978 to 12:34:50.177 UTC (14:34 CEST).
- The user confirmed reduced window lighting was already in place during this run.
- Run and recording were copied read-only from `mii001`. Archive hashes and
  retrieval timestamp are in
  `results/implementation_checks/camera_admission_audit_20260922T121851Z/snapshot_provenance.json`.

The first candidate, `survey_candidate_0003`, successfully produced a combined
geometry/QR recommendation. The second candidate recorded three unresolved
physical views. These observers returned advisory artifacts normally; their
process evidence has exit code 0, no signals, and no deadline expiry. This audit
does not infer a final mission outcome from the absence of later camera results.

## 1. Third view: QR decoding never gets its opportunity

Across the initial third-view observer and its two recentered captures, **all
36 processed images** record `performed=false` and
`identity_deferred_for_source_freshness` for QR work. The final observer contains
14 captured tuples: 12 processed images, one exact-TF retry exhaustion, and one
stale input. Among the 12 detector results, 11 report
`head_acquisition_deadline_exceeded` and one `head_proposal_ambiguous`.

| Final third-view timing | Median |
| --- | ---: |
| Camera header to local receipt | 300.8 ms |
| Camera header to detector start | 333.3 ms |
| Initial geometry work allowance | 105.8 ms |
| Detector duration | 109.5 ms |
| Remaining QR work after geometry | -1.75 ms |

The work deadline also accounts for the synchronized scan and retains a 50 ms
publication reserve. Remaining QR time ranges from -5.34 to +1.66 ms in this
last capture. The native marker/identity entry gate requires **70 ms**, so no
native or full decoder is called.

The code contains an **80 ms identity reserve after previous head misses**, but
`observer/node.py:1614` applies it only under `if not viewer_geometry`. The real
physical-head branch uses `evaluate_viewer_head` and passes the unshortened head
deadline (`node.py:1815`). That function performs geometry first, then checks
remaining QR time (`viewer_head_acquisition.py:149`). Repeated cold geometry
attempts can therefore consume every frame's available identity budget.

`QrObservationPoseFallback.observe` starts its grace only after a fresh,
independently associated QR sample (`qr_observation_pose.py:103`). Here the final
capture records eight `fresh_independently_bound_qr_required` and four
`current_sensor_frame_not_admitted` outcomes, zero QR samples, and no tentative
or latched identity. **There is no `same_pose_geometry_grace_pending` state.**
The configured 1.5-second delay is present and is not the cause of waiting.

Instead, eight ordinary associated sensor frames spanning **2.200 seconds**
satisfy the separate unresolved-inspection policy. It publishes an
`unobservable` advisory at 12:29:12.416 UTC, about four seconds after the final
observer starts. The parent moves on to generic view planning. The 90-second
camera timeout is a maximum, not a minimum observation duration.

## 2. Offline QR replay: readable pixels, unsuitable crop, then binding rejection

The saved image visibly contains a large front-facing QR. Replaying the exact
recorded wide identity ROI `[20,5,592,577]` on the 12 final images produced no
payload using local native decoding and the bounded 120 ms full decoder.

A **manually selected diagnostic crop** `[330,210,470,360]`, with the recorded
camera rectification and a one-second offline allowance, decodes **`Start` in
all 12 images**, with the decoder returning its own QR quadrilateral. Actual
local elapsed times were approximately 34–45 ms, using enlarged decode variants.
This demonstrates that the recorded pixels contain recoverable identity; it
does not establish an automatic crop selector or a live-time admission result.
The short-budget local replay did not reach this successful variant.

Restoring those returned corners to full-image coordinates and replaying the
unchanged QR target binder against each original scan and recorded evaluation
time rejects **12/12** with `camera_bearing_outside_map_cone`:

- QR camera bearing: approximately -0.60°.
- Mapped candidate bearing: approximately +5.98°.
- Difference: approximately 6.58°; permitted original-cone half-angle: 3°.
- Projected head center is approximately u=306, whereas the actual QR/head is
  near u=400.

The operational binder permits a camera-centered search only after a camera
registration/current head association has already been accepted
(`node.py:2105`, `qr_target_binding.py:73`). A QR-only observation with failed
head geometry cannot use that branch. Thus restoring decoder runtime alone is
not sufficient to guarantee fallback in this view.

For diagnostic comparison only, forcing the registered branch in offline
replay yields nine associations and three `no_samples_in_accepted_range`
rejections. This is **not valid registration evidence** and must not be turned
into an unconditional production override. It shows why independent bounded
QR-to-candidate registration deserves a dedicated implementation and tests.

The cause of the map/image bearing displacement cannot be assigned uniquely to
candidate geometry, localization, or calibration from these records alone.

## 3. Backside: the angle was computed, then association vetoed it

After the first view's centering turn, the observer processed 13 images in
approximately 2.69 seconds of source time:

| Result | Frames |
| --- | ---: |
| Missing current head border | 1 |
| Head geometry/quality accepted | 12 |
| Of those: current unique LiDAR target | 1 |
| Of those: ambiguous endpoint fragments | 11 |

The accepted visual fits report yaw approximately **-16.25° to -22.78°**. For
example, frame 3 measures -21.17°, with estimated yaw standard deviation 1.46°,
0.668 px reprojection RMSE, and accepted current raw border support. This is a
successful conditional visual angle estimate, not yet a certified candidate axis.

Deterministic replay reproduces all 12 original associations. Every rejected
case has two groups at opposite ends of the raw scan array, such as `[0,1]`
and `[225,226]`. The temporal correction reports
`three independent witnessed scans are required` on all 11 rejected frames.
The correction is active, but the required independent support did not become
available before this observer completed. Six scan witnesses also expired
before exact TF was available during this observer's startup.

The final axis/backside window contains and peaks at **one sample**, against
the required seven. Appearance identifies a backside candidate, but the crop
registration requires an associated current head. The final status therefore
reports `backside_complete_head_crop_unverified`. Later
`current_head_crop_proof_evaluation_mismatch` diagnostics follow the replacement
of the rejected evaluation; the recorded underlying target-association veto is
still `ambiguous_registered_camera_clusters`.

Ten unresolved samples spanning 2.047 seconds then produce a
`backside_unresolved` advisory at 12:25:59.136 UTC. No
`axis_observation.json` is committed. `candidate/inspection_execution.py:144`
enters `move_opposite` only for an actual axis receipt. The following view is
generic approximately +90° search, not an attempted certified opposite route.

The second physical view then has 25 missing-border results and no QR decode.
This is separate from the first view's successful visual fitting and association
failure, and from the third view's decoder starvation.

## 4. Latest viewer: endpoint search fixed, physical border choice unresolved

The recording shows the same visible front-facing `Start` head and the reduced
window illumination. It starts about 5.5 minutes after the third mission view
ends, so it is supporting evidence from a separate observer, not a simultaneous
trace of that mission capture.

| Viewer result | Frames |
| --- | ---: |
| `head_proposal_ambiguous` | 57 |
| `head_model_planar_axis_ambiguous` | 2 |
| `axis_estimated_current_measured_head` | 3 |

All **62** scan selections yield `nearest_head_current_scan_hint`; **46** use
the new endpoint-fragment search region. The previous viewer scan-selection
blocker is therefore resolved in this recording.

The dominant failure has moved to competing verified rectangles. In 51 of the
57 ambiguous frames, three proposals pass raw refinement but selection reports
`distinct_current_heads_ambiguous`. Frame 0's accepted proposals overlap around
the same visible panel, with differing top/bottom and inner/outer side rails.
The selector cannot certify which rectangle is the physical outer head frame;
an arbitrary largest/first rectangle would not be sufficient pose evidence.

Median detector duration is 106.6 ms. All 62 tracker updates report
`pose_observation_stale`, and all predictions report `no_tracked_pose`.
The tracker accepts source ages up to 250 ms (`pose_tracking.py:41,107`), while
the observed complete-frame ages are already roughly 344–491 ms at rendering.
Even the three accepted fits do not establish a tracked search hint, leaving
repeated cold acquisition. Only 38/62 results pass the viewer's overall
freshness assessment. This viewer has `no_qr_decode=true` and no observation
output path; its overlays cannot demonstrate QR fallback or mission admission.

## Recommended next corrections

1. Give the actual physical-head path a bounded periodic QR opportunity after
   geometry misses. Reserve decoder time before geometry or schedule a separate
   current-image identity probe. Keep the same image/scan timestamps and final
   freshness gates. Prevent an unresolved advisory from treating a never-run
   decoder as a completed unsuccessful identity search.
2. Use a bounded, automatically verified QR search region and independent
   QR-corner-to-candidate registration. It must preserve range, competing-target,
   motion, freshness, and bearing-displacement bounds, without requiring a head
   yaw or granting the existing registered flag without proof. Replay these
   12 original scans before claiming this candidate can be admitted.
3. Resolve physical outer-frame versus inset/double-edge proposals using current
   border evidence and the measured stand model. Preserve true multi-head and
   unresolved planar-orientation rejection. Use this recording as regression
   input, including frames 0, 13, 29, 39, and 58.
4. Diagnose the measured camera header-to-receipt age increase: approximately
   46 ms median at the backside view versus 301 ms at the final front view.
   Verify camera/workstation clock offset, capture timestamps, exposure,
   compression, transport queues, and receipt scheduling before assigning a
   cause. The viewer explicitly records that header/receipt clock offset was
   not measured. Reduced lighting alone does not establish why the age changed.
5. For backside acquisition, expose independent-witness availability and give
   an actually detected, currently associated head a bounded opportunity to
   complete its proof before the generic advisory wins. Preserve the three
   independent scans and seven axis samples; do not turn visual appearance or
   nearby fragments into a certified opposite-side angle.

Increasing the QR grace, reducing consensus counts, or widening the original
3° cone globally would not address this combination safely.

## Reproduction and limitations

Local audit files are in
`results/implementation_checks/camera_admission_audit_20260922T121851Z/`:
`analyze.py`, `audit_summary.json`, `qr_replay.py`, `qr_replay.json`,
`qr_binding_replay.py`, `qr_binding_replay.json`, and the explicitly hypothetical
`qr_registered_probe.json`. Run scripts from the repository root with
`PYTHONPATH=.` and a Python environment containing NumPy/OpenCV.

Association replay uses the original scans, camera transforms, and recorded
times. Image replay used local OpenCV 5.0.0 and a manually chosen tight crop;
its timings are not a workstation/ROS benchmark. The manually selected crop and
relaxed offline decode allowance are diagnostic tools, not production changes.
No production code, workstation configuration, ROS state, or robot motion was
changed during this audit.

## Implemented correction after operator confirmation

The operator confirmed that `Start` is the correct payload. Production code
continues treating identities generically; there is no `Start` special case.

- `observer/qr_acquisition_policy.py` and `viewer_head_acquisition.py` now give
  one periodic image an identity-first opportunity after a head-fitting miss.
  The existing image/scan deadline, publication reserve and one-full-crop budget
  remain in force. Geometry still runs on the same original image with the
  remaining budget. Empty early decoding cannot claim backside marker absence.
- `observer/qr_candidate_search.py` derives a small identity search crop from
  the unique current scan return and calibrated head-height projection. The
  decoder can prioritize a 4x variant, capped at 1.5 million pixels and subject
  to its existing work forecast. This crop supplies neither QR corners nor an
  axis; decoded corners are restored by the existing decoder.
- `observer/qr_target_binding.py` supports independent registration for physical
  QR-discovery runs. It requires the decoded symbol's own quadrilateral, a
  unique cluster over the entire correction envelope (including single-beam
  competitors), and the same cluster under the narrow QR ray cone. The original
  range interval and 12-degree correction cap remain. Scan source freshness
  and TF frame identity are checked explicitly. The receipt retains and
  validates the independent envelope evidence.
- `observer/inspection_progress.py` gives discovery observers five seconds from
  the first accepted stationary frame before an unresolved advisory can finish.
  Soft misses and consensus restarts do not renew this opportunity. Successful
  stronger paths still finish earlier. The existing 1.5-second geometry grace
  after a valid bound QR, three independent scan witnesses, seven axis samples,
  and parent observer timeout are unchanged. Status exposes the opportunity's
  deadline and remaining time.

Regression fixtures preserve the original compressed images and scan/TF data
for final-view frames 3 and 14, with image hashes and operator-confirmed payload.
Both decode `Start` using the automatic scan crop and bind independently without
a head angle; the old nominal binding rejects both. Clock-controlled tests
reproduce geometry starvation, verify identity-first scheduling, and cover
finite advisory delay, motion resets, conflicts and source/association rejection.

Validation: 238 tests and 174 subtests passed in the focused camera acquisition,
QR decoder/binding/receipt, inspection, discovery, transport and geometry suites.
One deployed-WeChat-backend test was skipped locally. `git diff --check` passed.
The local environment uses OpenCV 5.0.0; this is not deployed robot validation.

An additional budgeted offline replay is retained as
`implemented_qr_replay.json` in the audit artifact directory. It exercises shared
decoder resources and records both successes and cooperative-call overruns.
Binding uses original recorded scan/outcome times, so these results must not be
interpreted as live deadline admissions. The real observer rechecks source age
before accepting or publishing any result. Outer/inset border ambiguity and
upstream camera timestamp latency remain separate measured limitations; no
backside angle is manufactured when independent scan proof is unavailable.
