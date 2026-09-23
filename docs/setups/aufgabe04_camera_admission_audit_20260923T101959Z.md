# First local inspection admission audit: 20260923T101959Z

The fourth inspected candidate, `survey_candidate_0005`, timed out at its first
local view because the decoded QR could not pass candidate association. This
was not another truncated-QR failure: the observer decoded `QR_004` on 71
processed frames. Its camera-to-map bearing check rejected 69; the other two
had an ambiguous independent scan envelope. No decoded identity became bound
evidence, so immediate QR-only admission never became eligible.

## Scope and evidence

- Run: `stand_explore_exact2_camera_all5_20260923T101959Z`.
- Run bundle revision: `6c31049b8e6f714476e46d3c644abe1c15ba9194`, clean worktree.
- Candidate directory: `candidates/003_survey_candidate_0005`.
- Observer: `camera_lidar_attempt_00`, first local view (index 0).
- Latest viewer: `recording_20260923_123550_406882849`, 175 frames.
- Local evidence: `results/implementation_checks/camera_admission_audit_20260923T101959Z/`.
  `provenance.json` records the three archive hashes; `summarize.py` reproduces
  the counts in `summary.json`; `bearing_parallax_replay.json` records the
  diagnostic association comparison. All workstation access was read-only.

The prior correction worked for the earlier candidate: `survey_candidate_0001`
completed its opposite-side observation as `Start`. Candidate `0002` then
completed QR-only admission as `QR_002`; candidate `0003` had already completed
geometry-based admission as `QR_003`. The latest unresolved view is ordinary
frontside acquisition, with no retained backside orientation for this candidate.

## Why this view did not admit

The arrival check accepted the robot at 0.5418 m from the stored candidate,
with only -0.093 degrees of bearing error relative to that stored point. This
did not establish that the visible stand was centered: the admission artifact
explicitly says `camera_centered=false` and requires live target association.

The recorded image shows the complete blue stand QR around image x=230, while
the stored candidate projects near x=390. The unique current scan search can
shift its search region toward the visible stand, but that search hint does not
update the candidate or authorize final association. In 165 accepted search
hints, the current scan target and stored candidate differ by 0.103–0.119 m
(median 0.115 m). This residual establishes disagreement; it does not alone
identify whether its origin is survey localization, odometry drift, candidate
center estimation, or scan geometry.

Of 236 processed frames:

| Outcome | Count |
| --- | ---: |
| Decoded `QR_004`, rejected by the 12-degree registration limit | 69 |
| Decoded `QR_004`, independent scan envelope not unique | 2 |
| No decoded QR observation | 165 |
| Accepted geometry/axis sample | 1 |

Decoded QR bearing deltas are 14.412–14.650 degrees. The rejected identity is
not latched, so the QR-only fallback reports that fresh independently bound QR
evidence is required. This is an association failure, not a requirement to wait
for seven angle samples after a valid identity admission.

### Camera translation is missing from ordinary QR association

`real_robot/observer/qr_target_binding.py:67` calls
`rectified_pixel_bearing_in_scan` without `optical_depth_m`. The helper then
rotates a ray but does not translate its origin. The recorded camera is
0.077537 m forward and -0.004971 m lateral relative to the scan origin. At this
short distance, comparing the camera-origin ray directly with a scan-origin
bearing introduces material parallax.

A diagnostic replay intersects each rotated horizontal ray, from its calibrated
camera origin, with a circle at the median range of the unique current scan
cluster. The existing 12-degree limit, 3-degree cone, range gate and clustering
remain unchanged. For the 45 saved frames rejected specifically by bearing:

- corrected nominal deltas are 11.993–12.147 degrees;
- one passes association, 44 still exceed 12 degrees;
- the passing frame selects the same current scan indices as its envelope.

This is a nominal-range counterfactual, not a validated replacement depth
estimator or motion authorization. For frame 6, the existing accepted range
interval gives a corrected delta interval of 10.951–12.207 degrees, straddling
the limit. Translation correction is necessary but does not independently
resolve the remaining target-position discrepancy.

### The same boundary also suppresses geometry and centering

The measured-head preview already uses finite optical depth. Its proposals lie
near the same 12-degree boundary. All 1,135 previews reported as
`scan_persistence_current_input_invalid` already exceeded that bearing limit.
The persistence wrapper obscures the original reason: a bearing rejection has
no search association, `_entry` raises `no current camera cone`, and the
wrapper returns the generic persistence-input error.

Only one geometry sample survived, roughly 79 seconds into the observer window.
That frame deferred centering with `preserve_productive_geometry_view` and
started front reacquisition. No later accepted current frame produced a
centering receipt. The remaining frames failed association, and the single
axis sample expired. Thus enabling centering did not guarantee a turn: ordinary
centering still needs accepted current target support. Turning toward the image
alone would also not repair the stored candidate-position discrepancy.

## Timing and bounded recovery

TF delays were present: 107 tuple retries exhausted. However, 236 tuples reached
processing, including the 71 decoded frames. All 47 decoded frames retained in
capture history completed at source ages 0.324–0.455 s, below the 0.5 s limit.
Their admission failures are not stale-decoder-result failures. The final
status happens to be `tf_pending_exact_time`, which hides the dominant bearing
rejection if only the last status is inspected.

The parent stopped and reaped the observer at its deadline (SIGINT, return 130).
It correctly classified the view as candidate-local unavailability and began
bounded view recovery. Four proposals in one direction were statically blocked;
an alternate route passed dry preflight. The copied artifacts contain no
`mission_failure.json` and do not establish a completed terminal mission failure.

## What the debug viewer adds

The viewer starts 168 seconds after the first observer's last event. It shows a
substantially different framing: the blue head is near x=442 and more centered,
versus approximately x=230 during the failed view. Exact equality of robot pose
cannot be assumed from these recordings.

The viewer accepts head geometry in 157/175 frames (median detector time 10.5 ms).
It uses `head_target=nearest`, rather than the mission's named map candidate,
and `no_qr_decode=true`. Its green geometry overlay therefore demonstrates a
visible, measurable head at that later view, not a decoded and map-associated
mission admission. The recording provides no evidence that lighting prevented
the earlier QR read; the mission itself already decoded the complete symbol.

## Recommended correction

1. Use one calibrated finite-distance bearing implementation for QR and head
   association, including camera translation and a defensible depth interval.
2. Add a bounded stopped-target reconciliation step for the persistent
   candidate/current-scan discrepancy. Require unique current scan support,
   source/frame consistency, uncertainty and competing-candidate exclusion;
   do not silently rewrite the frozen candidate or simply widen the limit.
3. Let validated current support feed bounded centering without a new head-angle
   fit, then require a fresh candidate-associated decode for completion.
4. Preserve the original bearing rejection through persistence diagnostics and
   report decoded-but-unbound identity explicitly in timeout summaries. Add this
   recorded boundary case to regression coverage.

Production code was not changed during this audit.
