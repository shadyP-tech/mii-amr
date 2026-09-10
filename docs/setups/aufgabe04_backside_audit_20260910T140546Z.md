# Second-candidate backside audit — 10 September 2026

**The first view contained a clear backside, and the detector computed 27
usable backside poses. Every one arrived too late to be admitted.** The robot
therefore never obtained the certified backside-axis observation required to
select the opposite face. It tried a generic 90° view instead. That view was
nearly edge-on and also failed; a separate timeout-classification defect then
aborted the mission before the remaining view budget could be used.

Audited run: `stand_explore_exact2_camera_all5_20260910T140546Z`, clean commit
`770ea6b`, **14:05:46–14:17:49 UTC (16:05:46–16:17:49 CEST)**, parent exit code
**2**. Inspection order was `survey_candidate_0003`, then
`survey_candidate_0001`. The first candidate resolved to `QR_003`; the second
candidate caused the failure. Final progress was **1/5 distinct QR identities**.

## What the camera saw and computed

The [first-view image](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260910T140546Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_00/perception_debug/latest_frame.png)
shows the stand's broad gray backside and neck. The
[second-view image](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260910T140546Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_01/perception_debug/latest_frame.png)
shows the QR face at a highly oblique angle.

| Recorded outcome | Initial backside view | After generic +90° move |
| --- | ---: | ---: |
| Observer event interval, UTC | 14:13:31.921–14:15:00.882 | 14:16:18.300–14:17:47.761 |
| Processed images | 125 | 149 |
| Results rejected as obsolete | 100 | 128 |
| Remaining results | 25 | 21 |
| Usable geometry before freshness admission | **27** | 0 |
| Accepted candidate frames / peak axis consensus | **0 / 0 of 7** | **0 / 0 of 7** |
| Exhausted exact-time TF tuples | 4 | 14 |
| Raw captures saved | 64 | 64 |

All 27 usable geometries in attempt 00 have
`estimator_reason: axis_estimated_model_backside_current_frame`. Their completed
image ages were **708–882 ms**, median **786 ms**, beyond the existing **500 ms**
operational freshness limit. The images themselves arrived only **29–39 ms**
after their sensor stamps. The detector consumed **657–826 ms**, median
**723 ms**. For these successful geometries, processing was the dominant delay.

The [observer](../../scripts/aufgabe04/real_robot/observer/node.py) checks result
freshness at lines 1408–1420 and returns before final candidate association or
axis-sample admission. This correctly prevented stale motion evidence, but it
also explains why the logs say zero accepted axes despite successful geometric
estimates. The angle estimates were not independent physical ground truth;
seven fresh, consistent, candidate-bound samples still need to be demonstrated.

## Why the successful backside path was slow

The nominal projected head center was approximately **u=388 px**, while the
recovered head was around **u=297 px**. The nominal crop missed much of the
actual head. Every usable backside result required three evaluations: nominal
ROI, wider target-centered acquisition, and strict registered evaluation.

Recorded QR identity decoding cost a median **231 ms** on these usable frames,
about **32%** of detector time. Repeated geometry/acquisition and registration
account for substantial additional work. The current per-image cache already
reuses QR decoding when the wide and strict evaluations use the identical
crop. This is not evidence that the previously implemented QR cache is absent.

The existing [metric pose tracker](../../scripts/aufgabe04/perception/stand_axis/pose_tracking.py)
requires `fresh_refined` evidence and a directed `model_pose`. Backside results
deliberately provide `fresh_backside` and an undirected axis instead. All 274
processed frames first hit its 250 ms age rejection; even a faster backside
frame would still fail its different pose-type requirement. Backside processing
therefore needs its own bounded proposal reuse or faster current-frame path;
it cannot assume the existing QR-pose tracker will make subsequent frames cheap.

## Target association is recoverable without widening gates

Every preliminary LiDAR check missed the expected target range. In the first
view, the predicted bearing found returns around **1.59–1.61 m**, while the
accepted range was approximately **0.548–0.768 m**. This was the old projected
bearing, before camera registration corrected the visible head location.

An [exact registered-binding replay](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/registered_binding_replay.py)
rectified original capture `000008` with its recorded CameraInfo and reran the
strict metric geometry. It reproduced a usable camera-relative backside angle
of **−2.631965°**. Its regenerated corners and the original scan/TF then passed
the existing registered LiDAR association: **8.503256°** bearing correction,
**0.758 m** range, one eligible cluster with **three samples**.

That original result was discarded at **0.824 s** age before this final
association could run. The replay establishes that the candidate can be bound
using the current gates; it does not authorize reuse of that old observation
or guarantee complete seven-sample consensus or route admission. The
[replay receipt](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/registered_binding_replay.json)
records inputs, code hashes and the temporal limitation explicitly.

## Geometry failures also occurred

The 25 non-obsolete results in attempt 00 all reported
`model_backside_head_and_neck_unavailable`. Of the obsolete results, 36 had
`model_backside_planar_pose_unavailable`, 37 lacked head-and-neck geometry and
27 were geometrically usable. In attempt 01, all 149 selected estimates lacked
usable head-and-neck geometry, including its 21 non-obsolete results.

A [second replay](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/backside_replay.py)
reproduced nine saved ROI evaluations across four representative captures.
All four selected terminal geometry reasons matched the recording. One
registered head had good edge support but planar reprojection RMSE
**2.563 px**, above the **2 px** gate. Other samples lacked a valid head/neck;
one oblique-view crop found a QR quadrilateral but no pose seed or decoded
identity. These results distinguish real geometry rejections from obsolete
usable poses. The broad head-and-neck reason alone does not isolate blur,
illumination, contour quality or neck support as the cause.

The [geometry replay results](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/backside_replay.json)
are diagnostic. They do not replay full live timing, consensus or motion.

## Why the robot did not select the opposite face

The desired behavior already exists in
[inspection_execution.py](../../scripts/aufgabe04/real_robot/candidate/inspection_execution.py):
after a committed `axis_observation_path`, it calls `move_opposite` before
generic view search. That artifact requires fresh backside classification,
seven source-matched axis samples, stationary synchronized evidence, candidate
association and no conflicting QR evidence. Neither attempt produced it.

The first 90-second deadline ended on `metric_model_measurement_unavailable`,
which was classified as a local observation failure. Generic `unobservable`
view selection consequently chose +90°, not a face derived from a stand axis.
That motion completed at **14:16:12.080 UTC**.

A direct opposite-face selection means choosing the other face as the next
goal. It still requires a route around the stand: the target stand and every
other candidate remain keepouts. Existing opposite-side planning derives the
two normals from the certified stand axis, chooses the far side relative to
the observer, refreshes frame binding, and performs ordinary route, clearance
and motion admission. No straight-through-stand shortcut is appropriate or
present in this path.

## Separate defect: the final status accidentally ends all retries

Attempt 01 reached its deadline with `state: obsolete_detector_result`. The
last image age was **0.508793 s**, so rejecting that image was correct. However,
[timeout classification](../../scripts/aufgabe04/real_robot/observer/diagnostics.py)
omits this observer soft-miss state from its local-timeout handling. The parent
therefore raises a generic `RuntimeError` instead of the typed local-observation
exception caught by the inspection loop.

The attempt had valid, unpoisoned status and 21 prior fresh TF-ready frames
that failed candidate evidence admission. Its preceding status, just **0.651 s**
earlier, was `metric_model_measurement_unavailable`. A
[classifier replay](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/timeout_classification_replay.py)
shows that changing only the final status in memory to that value, or to a
trailing TF-pending state, makes the same attempt eligible for bounded local
recovery. The actual obsolete-result status makes it fatal.

The second view was consequently neither recorded as completed nor followed
by another local view. `inspection_progress.json` remains at **one recorded
view out of eight**, with no view-budget or route-search exhaustion. This is
a terminal-snapshot classification defect, not exhaustion of the configured
inspection budget. The [replay result](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/timeout_classification_replay.json)
preserves all original process/status evidence.

## Required corrections and validation

1. **Make backside evidence timely.** Reduce repeated acquisition and strict
   registration work on the same pixels; bound expensive QR absence searches.
   Reuse only a candidate-bound proposal as a search hint and refit the current
   image. Keep undirected backside-axis evidence separate from directed QR-pose
   tracking. Validate the full capture-to-admission budget on hardware rather
   than extending the freshness limits.
2. **Preserve registered target association.** The saved frame passes the
   existing corrected-bearing gate. The faster path must retain that binding
   and its source timestamps, not use the incorrect nominal bearing or widen
   candidate-association thresholds to hide the projection discrepancy.
3. **Fix bounded timeout classification.** Treat a trailing obsolete-result
   soft miss using valid, unpoisoned accumulated processing evidence, record the
   failed view, then continue within the existing budget. Malformed evidence,
   crashes, identity conflicts and genuine hard failures must remain terminal.
   No stale axis or recommendation may become authoritative.
4. **Test the actual transition.** Reuse these original fixtures to establish
   fresh seven-sample backside consensus, artifact commit, immediate selection
   of the opposite face, collision-checked execution, and QR decoding there.
   Add a regression for the recorded attempt-01 status so a final-frame timing
   accident cannot again bypass the remaining inspection budget.

No additional generic orbit is needed once a valid backside-axis artifact
exists; the existing opposite-face path should be used. Hardware validation is
still required for the full transition and for angle accuracy.

## Evidence integrity

All **773 copied source artifacts** match the read-only transfer's
[manifest](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/source_manifest.json).
The 64-frame quota on each attempt captured its initial portion; later raw
images were dropped by the diagnostic quota, while the full observer event
logs continued. The 27 usable-pose count and hardware timing statistics come
from those full event logs, not just the saved images.

[audit_metrics.json](../../results/audits/stand_explore_exact2_camera_all5_20260910T140546Z/derived/audit_metrics.json)
contains per-attempt counts, per-result timings, association metadata and
terminal evidence. Replays used Python 3.12.14, OpenCV 4.13.0 and NumPy 2.5.3.
Twenty-seven focused geometry, axis-policy, freshness and tracker tests passed.
This audit changed no production code or original artifacts and issued no
robot commands.
