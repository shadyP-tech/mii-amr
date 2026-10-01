# Real-run audit: 20261001T104618Z

Run `stand_explore_exact2_camera_all5_20261001T104618Z`, October 1, 2026,
**12:46:18–12:57:10 Europe/Berlin**, on workstation SSH alias `mii002`
(hostname `mii001`). All nine parent/motion-child bundles record clean
revision `fbe692684c5fb0fdcc840d6ed77a56d9e2509c80`, including the latest
persistent-target-support correction. All **688 copied files** match their
workstation SHA-256 hashes.

## Outcome

The robot found **all five distinct QR identities**. The mission then failed
closed during return-to-Start planning because Start's original rasterized
candidate keepout blocked the exact stored destination. No return leg or
return velocity command was issued. All eight previously executed motion
children completed successfully.

| Completion time, Berlin | Candidate in this run | QR identity | Result |
|---|---|---|---|
| 12:49:54.751 | `0003` | `QR_003` | Certified facing geometry |
| 12:53:00.448 | `0001` | `Start` | Certified retained facing geometry |
| 12:54:29.685 | `0002` | `QR_002` | Certified facing geometry |
| 12:56:09.091 | `0006` | `QR_004` | QR-verified observation pose only |
| 12:57:03.995 | `0005` | `QR_001` | QR-verified observation pose only |

The configured completion policy is
`geometry_or_qr_verified_observation_pose`. Consequently:

- Identity discovery: **5/5**, `goal_completed=true`.
- Certified facing geometry: **3/5**, `facing_complete=false`.
- Return to Start: **failed before motion**, `start_pose_reached=false`.
- Overall mission: `failed_closed`, parent exit code **2**.
- No FastAPI request was sent; server identity binding remains pending.

The two coverage observation windows lasted approximately eight seconds each;
final coverage was 95.31%. Start required an opposite-side inspection. Its
first dry route failed the route-margin gate by 3.27 cm; a smaller-standoff
alternative subsequently completed. This was a handled planning rejection,
separate from the terminal return failure.

There is also a reporting inconsistency: `coverage/survey_summary.json` and
its last printed terminal summary still say three confirmed stands and
exploration incomplete. That registry summary was last updated by candidate
`0002`'s geometry result; the subsequent two QR-only discoveries are written
through a separate path. Use final `candidate_goal_progress.json`,
`observed_station_identities.json` and `mission_summary.json` for the five-QR
discovery outcome. The return exception prevents the final successful-run
summary from being printed. The legacy summary should be labeled or updated
to avoid presenting its narrower registry count as final mission discovery.

## Exact cause of the return failure

The saved Start facing pose was `(-1.475728, -0.496324, 0.040198)` in its
observation map epoch. Fresh localization reprojected it to
`(-1.457460, -0.481547, -0.008407)` for the return, a **2.35 cm** translation.
The same rigid transform was applied to the candidate geometry, preserving
their exact **0.341708 m** separation.

The original candidate keepout radius is **0.34 m**. Thus the continuous
endpoint has only **1.71 mm** of reserve. The planner rasterizes this radius
onto a **5 cm grid** using `ceil(0.34 / 0.05) = 7` cells:

| Check | At retained-pose validation | At return planning |
|---|---:|---:|
| Exact candidate-to-target distance | 0.341708 m | 0.341708 m |
| Target grid cell | `(26, 23)` | `(27, 24)` |
| Start-candidate grid cell | `(33, 24)` | `(34, 24)` |
| Relative cell offset | `(-7, -1)` | `(-7, 0)` |
| Original candidate raster keepout | Free | Blocked |

The new goal cell lies on the included seven-cell disk boundary. The exact
stored-pose planner then raises:

> exact stored target is blocked; goal snapping is forbidden

Offline replay of the **production planner** reproduces this exact error.
Layer-by-layer checks isolate the original Start-candidate keepout as the
sole endpoint blocker. The static map, arena bounds, 0.25 m static inflation,
other candidate keepouts and measured-target keepout all permit the endpoint.
The nearest occupied static cell square is **0.45845 m** away. This is a
continuous-versus-raster admission inconsistency, not evidence that the return
target was against the wall or that the controller failed.

Relevant implementation:

- `scripts/aufgabe04/navigation/planning/costmap.py:195`: candidate keepout rasterization.
- `scripts/aufgabe04/navigation/approach/admitted_pose_route.py:330`: exact endpoint rejection.
- `scripts/aufgabe04/real_robot/mission/stored_pose_navigation.py:37`: stored pose reprojection.

The audit identified a mismatch between stored-pose admission and return
endpoint collision semantics. The correction below addresses that mismatch
while preserving the exact goal, full candidate keepouts and physical
clearance requirements.

### Implemented return correction

Automatic return-to-Start travel now handles a candidate-raster-only endpoint
block through one separately certified final segment. A* and smoothing still
use the unchanged complete planning costmap. They end at a nearby free grid
waypoint; the exact saved position and heading are appended afterward. The
connection is limited to three grid cells or 15 cm, whichever is smaller.

The entire connection must be free in the separate statically inflated map,
and a sampled Lipschitz bound must prove continuous static/arena clearance at
the original inflation radius. Analytic segment checks retain every original
candidate keepout, the measured-center envelope and active stand standoff.
The planner neither clears a keepout cell nor substitutes a new saved goal.
The child reloads the bound map and candidate snapshot and recomputes the
proof. Missing or altered proof fails validation. Its grid waypoint remains
protected from thinning, and uncertainty staging cannot stop inside the final
connection. Stored-pose tours and stationary heading corrections retain their
existing blocked-goal policy.

Offline replay of the recorded failure now produces a **2.829 m route with six
vertices**, ending at the original reprojected Start position and yaw. The
final connection is **4.1025 cm**. Its continuous static clearance lower bound
is **0.45617 m**, above the unchanged **0.25 m** requirement. Every route
segment passes independent static and candidate clearance checks. The
minimum margin outside the original candidate keepout remains **1.7084 mm**;
no keepout radius was reduced. Reload validation accepts the sealed route.

With the original return-localization samples and the same run's recorded
0.105 m robot radius / two-sigma policy, the production readiness loader and
planner admit **the first 2.2255 m return stage**, stopping at
`(-1.145, -0.065, -3.038899)` before the final approach. The complete route
remains bound to the exact saved Start pose; the recorded uncertainty does
not admit the whole return as one leg. Later stages require new stopped
localization evidence. This replay does not claim completed physical return
or invent later localization epochs.

Validation: **115 distinct tests and 100 subtests across 14 files pass**,
including recorded reprojection, staged return, source/proof tampering, anchor
retention, static obstacles, neighboring candidates and measured-center
blockers. The correction is local; deployment and a new hardware run remain
unperformed. Fresh live localization, uncertainty admission and motion permits
are still required.

## Comparison with the previous wall-facing failure

Candidate numbers are local to each run. Today's rejected `0004` is **not the
same physical hypothesis** as September 30's `0004`: their frozen positions
are approximately **2.06 m apart**. Today's `0004` was rejected before travel
for `target_static_map_incompatible`, and its obstacle keepout remained.

Candidate `0005` is spatially comparable across the two runs. It lies near the
same previously observed `QR_001` stand:

| Quantity | September 30, 15:05 UTC run | October 1, 10:46 UTC run |
|---|---:|---:|
| Frozen survey position | `(1.16218, 0.77162)` | `(1.15457, 0.74690)` |
| Projected arrival position | `(1.31082, 0.31381)` | `(1.21830, 0.66766)` |
| Position shift | 48.13 cm | 10.17 cm |
| Position-epoch recovery limit | 35 cm | 35 cm |
| Current target reconciliation | Rejected, beyond bound | Three fresh stopped scans accepted |
| Camera outcome | No accepted stand/QR; user stopped run | `QR_001` verified |

The latest ordinary range gate still initially had no in-range samples:
expected approximately 0.377–0.597 m, with the real target near 0.604 m.
Three fresh scans each recovered one compact four-beam cluster using the
authenticated frozen/current hypotheses. Recovered surface centroids were
4.10–4.38 cm from the frozen target and 6.23–6.46 cm from its projected point.
The final QR ray independently associated with a unique three-beam cluster
around 0.603 m. Offline production reconciliation validation succeeds with
only source-path rebinding; recorded candidate geometry remains unchanged.

The new missing-support stop is deployed, but **its terminal branch was not
exercised in this run**. Positive current reconciliation took priority. There
are no `target_support_failure.json` receipts, no 90-second camera timeout and
no repeat of the previous unsupported radiator-only inspection. The smaller
position shift is an observed run difference; this audit does not attribute
it to the new failure-window policy.

## Why QR_001 and QR_004 are not facing-ready

Both original camera captures show real complete stands. The remaining issue
is orientation confidence, rather than missing physical targets.

- **QR_004:** all five processed frames associated with the target. Four
  failed yaw uncertainty checks and one was planar-axis ambiguous. The final
  yaw standard deviation was about **4.09 degrees**, above the **3-degree**
  threshold. The three-sample bounded interval reached **±15.58 degrees**,
  beyond the **15-degree** limit. Discovery completed in **4.248 seconds**.
- **QR_001:** three processed frames were planar-axis ambiguous; the third
  obtained valid reconciliation, target association and QR identity. Its final
  competing orientations were approximately **10.48 and 0.22 degrees**, with
  only **0.0703 pixel** reprojection-error separation. The one-sample bounded
  interval was **±17.28 degrees**, beyond the same limit. Discovery completed
  in **2.953 seconds**.

Both attempts saved bounded front-geometry evidence for inspection, but no
certified facing recommendation. Their QR observation receipts explicitly
have `facing_ready=false`, no stand-axis estimate and no motion authority.
Under the configured discovery policy, a valid QR receipt ends the candidate
inspection before further geometry acquisition. Do not treat these two QR
poses as certified docking/facing poses or relax the orientation gates solely
to change the completion count.

Original source images, copied byte-for-byte and verified against capture
image hashes:

![QR_001, green stand](../../results/implementation_checks/run_audit_20261001T104618Z/camera_qr001.jpg)

![QR_004, blue stand](../../results/implementation_checks/run_audit_20261001T104618Z/camera_qr004.jpg)

## Additional debug-viewer recording

The subsequent user recording is
`recording_20261001_131416_741681612`, **13:14:16.604–13:14:23.536 Berlin**.
All **67 recording files** were copied and SHA-256 verified separately in
`viewer_source_integrity.json`. Its 63 distinct source sequences, timestamps
and PNG hashes span **6.932 seconds**. The nominal 15 fps AVI plays those
frames in 4.2 seconds; that playback length is not the sensor observation time.

**The viewer also admits no head angle: 0/63 frames.** It detects the current
head border in 58 frames, all rejected as
`head_model_planar_axis_ambiguous`. Three frames have ambiguous nearest-scan
head selection, and two have no visible scan-selected head. All inputs are
fresh at rendering, with source ages approximately 40–179 ms. The head-center
positions among the 58 fits differ by at most **0.164 pixel**. There is no
recorded robot-pose history, so this is evidence of a stable image target,
not a measured proof that the robot remained stationary.

The visible yellow rectangle means **current borders detected**, not accepted
orientation. `display_estimate.usable=false`, `yaw_deg=null`,
`yaw_reliable=false`, and the geometry/model overlays are `rejected_fit`.
The overlay's `current_head_selection_not_fresh` reason refers here to an
unaccepted selection; it is not a sensor-age failure. Likewise,
`model.committable=true` refers to the measured physical profile, not acceptance
of this image's angle. `head_orientation_bounds.accepted=true` means that a
valid uncertainty interval was constructed; its width still has to pass the
later bounded-window and facing checks. No tracker-held or historical angle
was substituted.

### Why a nearly frontal-looking head still fails

For recording frame 10, the selected 78 mm square head has refined corners
`(299,254), (392,254), (392,347), (299,347)`: a 93-pixel square. The same
current image supports two positive-depth planar pose solutions:

| Camera-relative yaw | Reprojection error | Local yaw standard deviation |
|---|---:|---:|
| 0.177 degrees | 0.0104 pixel | 4.766 degrees |
| 10.584 degrees | 0.6196 pixel | 4.721 degrees |

Their errors differ by **0.6091 pixel**, less than the configured **0.75-pixel
corner-noise allowance**, while their yaws differ by more than the **5-degree**
ambiguity threshold. A very small best-fit residual therefore does not make
the alternative sufficiently implausible. The strict gate rejects the pair.
Independently, both yaw uncertainties exceed the **3-degree** limit.

The bounded path retains both solutions with their three-sigma engineering
allowances. That frame's range is centered at **5.313 degrees with a
19.433-degree half-width**. Across all 58 fits, half-widths remain
**18.85–19.43 degrees**, above the **15-degree** bounded-window limit.
Hypothesis uncertainty is also above 3 degrees throughout. Simply selecting
the near-zero angle or collecting more copies of this nearly unchanged view
does not satisfy either admission rule. These are engineering uncertainty
bounds in the current estimator, not ground-truth measurements of the stand's
angle.

Production image replay independently reproduces the recorded border corners
and ambiguity rejection on original PNG frames 0, 10 and 62. An explicitly
synthetic sensitivity check on frame 10 moves only its top-right corner one
pixel to the right: the best-fit yaw changes from **0.177 to 9.081 degrees**,
while the two fits' residuals differ by only **0.0040 pixel**. This was an
offline perturbation, not a measured movement. It demonstrates sensitivity of
the current near-frontal planar fit to pixel-scale corner changes; it does
not establish a wrong calibration or a particular true stand angle.

The viewer and mission use the same measured stand profile and camera
intrinsics/extrinsics, and share the measured-head quality gate. The viewer
uses nearest-scan head selection, no mission target coordinates and
`no_qr_decode=true`; its absent QR result is therefore expected, not a decode
failure. It does not write a mission pose or authorize motion. The recording
is useful evidence of the same geometric rejection, but is not a complete
replay of mission target/QR association.

The viewer recording does not embed a Git revision. Its rejections and
configuration are directly recorded evidence; code interpretation and offline
replay use the inspected `fbe6926` implementation. The workstation was still
at that revision when the recording was copied.

### Why exploration stops before obtaining facing geometry

The mission has a separate QR-only completion path. Its normal short geometry
grace is reduced to **zero** when the accepted QR binding carries a current
target-reconciliation proof (`observer/qr_observation_pose.py:126`). Thus
QR_001 can finish discovery on its third processed frame, with only the final
frame associated, before the seven associated frames required by the bounded
front-facing path are available. Stronger strict or bounded geometry has
priority on that same frame, but neither qualified.

This is a real workflow distinction: the configured mission is allowed to
finish **identity discovery** without finishing **facing geometry**. It should
not be interpreted as successful front-facing admission. A longer wait alone
is not shown to fix this case: all 58 usable-border fits in the longer viewer
recording remain too ambiguous and too wide. If certified front-facing poses
are required under that original policy, geometry must remain an explicit
unfinished task after QR success and obtain additional geometric evidence,
such as a separately certified changed view. The recorded evidence does not justify reducing
uncertainty or discarding the alternate pose simply to pass the gate.

An original annotated-video frame, extracted without changing its pixels:

![Viewer rejects the angle while showing detected head borders](../../results/implementation_checks/run_audit_20261001T104618Z/viewer_annotated_first.png)

## Accepted approximate-front correction

After this audit, the operator specified that **up to 30 degrees from
straight-on, including estimated uncertainty**, is sufficient for front-facing
admission. The correction therefore changes the task's viewing requirement;
it does not make the measured angle artificially more precise.

The distinct `current_head_coarse_front_interval` policy retains both planar
solutions and every sample's original uncertainty. It requires seven fresh,
independently target-associated QR/front samples from one stopped context,
with consistent borders. The selected pose uses the interval midpoint as its
nominal normal. The full interval, endpoint misalignment, target-center
uncertainty and 3 cm terminal-position reserve must together fit within
30 degrees. The existing candidate route still checks arena bounds, static
obstacles, the active stand and other candidate keepouts. No facing route is
authorized to move by these receipts.

At the configured 0.35 m distance, 2 cm target uncertainty plus the existing
3 cm terminal reserve consume **8.213 degrees**. With the original angular
bounds, the ideal midpoint-normal endpoint needs **25.495 degrees** for the
recorded QR_001 sample, **20.777 degrees** for QR_004, and **28.129 degrees**
for the union of all 58 detected viewer frames. These are angular feasibility
calculations, not successful mission replays or route certificates. The
mission recordings do not contain seven admitted frames for those two
stands; the viewer had QR decoding disabled and no authenticated mission
target/pose timeline. A compact fixture retains seven actual consecutive
viewer frames and both pose alternatives for offline angular regression.

The observer now gives eligible coarse-front geometry a fixed **five-second**
opportunity before QR-only fallback, even when current target reconciliation
would otherwise make that fallback immediate. Fresh current QR evidence is
still required when the opportunity expires. Soft misses cannot extend the
deadline. Strong strict geometry can finish immediately; retained backside
discovery keeps its existing behavior. The strict single-angle quality gate
and the legacy/backside 15-degree interval / 20-degree viewing policies remain
unchanged.

Coarse-front receipts are restricted to current schema-2 QR-front
recommendations. They cannot enter backside or retained-facing contracts, or
be flattened into the legacy logistics arrival catalog. A recommendation's
failed route validation still stops the existing recommendation branch; this
change does not add automatic QR-only downgrade after a route failure.

Implementation and validation are local. Deployment and a new real-robot run
remain unperformed.

Validation covers **202 distinct tests and 233 subtests across 18 test files**,
all passing after a test-fixture scan-frame mismatch was corrected and the
affected file rerun. The node regression uses current stopped-scan
reconciliation and a 20.776-degree interval: the first six frames cannot exit
through QR-only fallback, and the seventh commits the bounded front
recommendation. Expiry tests retain the five-second deadline across soft
misses and require a fresh QR after missing or stale frames. Route tests
reject occupied map cells and neighboring stand keepouts. Strict-angle,
backside, retained-facing, wall-target and source-freshness guards also pass.

## Evidence and scope

Audit root: `results/implementation_checks/run_audit_20261001T104618Z/`.

- `source_integrity.json`, `source.tar.gz`, `source/`: all 688 verified source files.
- `audit.py`, `audit_summary.json`: file integrity, content-hashed catalogs and cross-checked outcome counts.
- `timeline_audit.py`, `timeline_audit.json`: run/child provenance, candidate timing and execution results.
- `return_audit.py`, `return_audit.json`: authentic production endpoint replay and per-layer blocker checks.
- `return_correction_replay.json`, `return_correction_replay_artifacts/`: corrected exact-target planning replay and independently checked full-route clearance.
- `return_correction_readiness_replay.json`, `return_correction_readiness_artifacts/`: first return-stage admission using original localization samples and the run's recorded uncertainty policy; no later epoch or physical arrival claim.
- `start_return_correction_validation.json`, `start_return_*_pytest.txt`: return correction test scope, unique result counts and source hashes.
- `population_compare.py`, `population_comparison.json`: cross-run spatial comparison and current scan reconciliation.
- `camera_audit.py`, `camera_audit.json`: source frame/association/geometry outcomes and clean observer exits.
- `viewer_source_integrity.json`, `viewer_source.tar.gz`: the separately verified user recording.
- `viewer_statistics.py`, `viewer_statistics.json`: all 63 source frames, freshness, border/angle outcomes and uncertainty statistics.
- `viewer_parity.py`, `viewer_parity.json`: shared quality gates and recorded viewer/mission configuration differences.
- `viewer_geometry.py`, `viewer_geometry.json`: offline geometric replay and explicitly labeled corner-perturbation analysis.
- `coarse_front_feasibility.py`, `coarse_front_feasibility.json`: original-uncertainty angular calculations for the accepted 30-degree front requirement.
- `coarse_front_guard_summary.json`, `coarse_front_guard_pytest.txt`, `coarse_front_guard_corrected_pytest.txt`: exact test files, result counts, source hashes and the affected-file rerun.
- `final_workstation_state.json`: at **13:11:25 Berlin**, this remained the latest run, the workstation was clean at `fbe6926`, its mission summary was unchanged, and no matching mission processes were running.

Workstation access was read-only. The initial audit wrote only local evidence;
the subsequently authorized correction changes local production code and
tests. No deployment or robot commands were made.
