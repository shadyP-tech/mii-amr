# Camera acquisition and navigation failure audit — 10 September 2026

The second relocation of candidate `survey_candidate_0001` did reach a close,
front-facing view. The robot was **0.413 m** from the candidate instead of the
preferred **0.70 m**. However, the original image contains the entire head,
and an offline diagnostic can fit its border at this distance. The strongest
evidence is for **incorrect crop placement and inconsistent QR/head geometry**,
with substantial image latency. Proximity alone does not explain the failure.

The mission ultimately stopped later, during another inspection move: a
localization correction required resealing, and the replacement route failed
its uncertainty budget by **18.091 mm**. This was a separate navigation gate.

## Run and view sequence

Run: `stand_explore_exact2_camera_all5_20260910T145318Z`, clean commit
`cfd1439a45e35faced2d4719090f68329dac3d68`, source host `mii001`, container
host `mii0002`. The parent bundle spans **14:53:18–15:12:25 UTC**, or
**16:53:18–17:12:25 CEST**, and records exit code **2**.

Both survey stops completed at **95.31% modelled coverage**. Candidate
`survey_candidate_0003` resolved to `QR_003` in its first camera view. Candidate
`survey_candidate_0001` did not resolve; final progress is **1/5 QR identities**.

Here, “second inspection pose” means the **second relocation**, recorded as
`camera_lidar_attempt_02`: the initial camera observation has index 00.

| Candidate 001 view | Base-to-candidate distance | Projected optical depth | Expected head height | Recorded image |
| --- | ---: | ---: | ---: | --- |
| 00: initial backside | 0.708 m | 0.665 m | 75 px | [Backside](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260910T145318Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_00/perception_debug/latest_frame.png) |
| 01: first relocation | 0.682 m | 0.635 m | 79 px | [Oblique QR view](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260910T145318Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_01/perception_debug/latest_frame.png) |
| 02: second relocation | 0.413 m | 0.372 m | 134 px | [Front-facing view](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260910T145318Z/candidates/001_survey_candidate_0001/camera_lidar_attempt_02/perception_debug/latest_frame.png) |

All three camera attempts lasted about 90 seconds and ended at zero axis
consensus samples out of seven. The newly implemented timeout classification
worked: each trailing obsolete-result deadline was recorded as a bounded local
failure and the inspection loop continued.

## Why the robot moved closer

For view 02, route search tried offsets **0.70, 0.65, 0.60, 0.55 and 0.50 m**;
each was rejected because the target cell was blocked. The **0.45 m** proposal
completed, with measured arrival distance **0.412517 m**. This is explicit in
`candidates/001_survey_candidate_0001/inspection_route_proposals.jsonl`.

The current [standoff search](../../scripts/aufgabe04/real_robot/candidate/inspection_route_search.py)
descends toward collision/raster floors. It receives no camera observability
constraint. [Arrival admission](../../scripts/aufgabe04/real_robot/candidate/approach.py)
accepted the configured **0.33–0.90 m** base-to-target range and bearing error
within **3°**. Those checks do not establish usable optical depth or complete
head/neck visibility.

The observer checks projected center visibility and a minimum 18-pixel head
size; its ROI helper clips image boundaries. Moreover,
[view novelty](../../scripts/aufgabe04/real_robot/candidate/inspection_policy.py)
is angular, requiring at least 20° separation. There is no ordinary distance-only
correction that retains a promising front-facing bearing. The existing bearing
alignment exception preserves range and therefore cannot back out for framing.

This is a real planning limitation. It does not prove that this particular
distance is intrinsically too close: the whole head is present in the image.

## Crop placement fails before angle fitting

In saved front-view frame 000045, the nominal crop ends at **x=525**; nearby
frames end around **x=523**. The replayed QR quadrilateral spans approximately
**x=456–575**. Thus roughly **50 pixels of the symbol's right side** are outside
the nominal crop, including the right finder region. The wider crop contains
the complete symbol and head.

The current projection predicts the head around x=403, while the registered
head center in frame 000045 is x=514.4. Distance increases the pixel size of
the target, but the direct clipping mechanism is this registration error.
The logs cannot isolate its physical cause among candidate-position error,
localization, camera calibration and other frame/model discrepancies.

The user's proposed pipeline addresses a circular dependency in
[`camera_target_registration.py`](../../scripts/aufgabe04/real_robot/observer/camera_target_registration.py):
the QR-mode wide proposal must already carry `debug.model_pose` before it can
seed strict recentering. A wide image can therefore contain the head and QR
text yet fail to correct the crop when pose acquisition is unavailable.
Recorded frame 000008's wide result, `model_qr_text_without_geometry`, proves
that live identity decoding sometimes found text without usable geometric
binding. It does not establish a confirmed candidate identity.

**Keep ROI processing, but let a bounded current-image head proposal relocate
the crop before a successful pose exists.** Associate its bearing, size and
LiDAR evidence with the candidate; include the complete head, all four borders,
neck and margin; then run the strict current-pixel fit with correctly adjusted
intrinsics. Wider or full-image work should be bounded reacquisition, not the
steady-state processing path. A proposal remains distinct from identity,
axis consensus and motion authority.

## Recentring alone is insufficient

The [front replay](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/derived/front_camera_replay.json)
reproduces frame 000045's QR-only pose: **8.694°**, estimated depth
**0.3467 m**, reprojection RMSE **1.9349 px**. The following outer-head
refinement fails before the joint QR/head fit: the **6.168 px** search corridor
finds acceptable left/right lines but no coherent top or bottom rail.

A separate [border diagnostic](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/derived/front_border_diagnostic.json)
uses a disclosed manual visible-border proposal on the same original pixels,
then the unchanged production raw-edge, corner and fitting gates. It recovers
all four sides with at least **0.990 support**, and the head-only fit achieves
**1.580 px RMSE**. Combining those head corners with the recorded QR corners
fails at **5.076 px RMSE**. The observed head/QR image ratios are approximately
**1.175 × 1.110**, versus the configured **1.258 × 1.258**.

This establishes that useful border information exists at the recorded
distance, and that the QR/head measurements are inconsistent under the current
model. It does not establish automatic acquisition success or physical ground
truth. Investigate head-versus-paper-versus-symbol association, QR corner bias,
the printed symbol dimensions and calibration. Do not silently enlarge the
refinement corridor or change measured dimensions to force this frame to pass.

## Earlier backside observation was blocked by a false QR marker

The initial view contains a broad plain backside and small text label. The
[frame 000022 replay](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/derived/backside_front_latch_replay.json)
reproduces a spurious native OpenCV quadrilateral from its 2× search, with
neither decoded identity nor a valid pose. It nevertheless sets `qr_detected`
and the stationary `front_seen` latch. The wider crop finds no QR marker.

At image stamp **1789052448.765649**, a usable backside result was genuinely
fresh (**239.9 ms**), but was withheld as `front_seen_axis_unresolved`. Another
usable registered backside result was obsolete (**697.8 ms**). Neither created
an axis sample or an eligible search hint. The hint path was never used in
this candidate's **433 processed images**; its absence here is not a failure
to deploy the fix.

The persistent front-face veto is stronger than the unverified quadrilateral
evidence that triggered it. Marker validation needs its own correction while
preserving rejection of verified QR conflicts. The observation evidence's
`poisoned` flag remained false; this was the separate marker latch.

## Timing remains an independent obstacle

| Camera view | Processed | Obsolete results | Usable geometry before admission | Median image age at receipt | Median detector time | Median completed image age |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 149 | 117 | 2 | 27.8 ms | 487.9 ms | 555.2 ms |
| 01 | 140 | 94 | 0 | 36.7 ms | 433.6 ms | 520.0 ms |
| 02 | 144 | 133 | 0 | 184.5 ms | 446.5 ms | 647.7 ms |

The operational age limit remains **500 ms**. In the front view, **133/144**
results expired, and the other eleven still lacked usable geometry. Faster
processing alone cannot repair that geometric failure; fixing geometry alone
does not resolve the roughly 185 ms median age already present at receipt.
These logs do not distinguish publisher, transport, queue or timestamp causes
of the increased receipt age.

## Terminal navigation failure

After the three camera failures, `candidate_001_inspection_003` began another
admitted inspection route. Its 43 control cycles commanded zero linear speed
while rotating **38.506°**. At **15:12:11 UTC**, a fresh localization yaw
correction of **0.980°** exceeded the frozen certificate's **0.881°** allowance.
The follower stopped and requested resealing.

Recovery obtained fresh localization and replanned the same endpoint. Raw
limiting clearance remained about **322.52 mm**, but position sigma increased
from **21.43 to 48.05 mm**, and yaw sigma from **0.007689 to 0.050339 rad**.
Required clearance grew to **340.61 mm**, leaving **−18.091 mm**. Replacement
preflight correctly rejected the route; no replacement permit or motion followed.
The replay cannot identify the physical cause of covariance growth.

The [navigation replay](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/derived/navigation_failure_replay.json)
reproduces the continuity decision and both complete 88-segment uncertainty
budgets. This was a structured replacement-preflight failure, not stale TF or
exhaustion of the view/reseal budget. Raising a retry count alone does not
change that policy. A future stopped replan/readiness extension would need a
fresh bounded admission transaction, with all clearance limits preserved.

Append-only route evidence records **26 proposals**, although the last saved
`inspection_progress.json` still says 23. There were **3/8 camera views**;
camera or proposal-budget exhaustion did not terminate this run.

## Corrections to implement next

1. Separate neutral 2-D head acquisition and candidate association from pose
   fitting, then recenter a complete, bounded ROI on current image evidence.
2. Validate QR/head/paper border association using the original frames and
   measured printed geometry; retain strict current-pixel fit quality checks.
3. Validate candidate-local QR marker evidence before setting a persistent
   front-face veto. An arbitrary undecoded quadrilateral must not defeat
   stronger current backside geometry indefinitely.
4. Give inspection planning a calibrated camera observability envelope and
   a bounded distance-only correction from promising front views. A collision-safe
   smaller standoff is not automatically a camera-valid pose.
5. Measure capture-to-receipt delay separately from decoder/fitter time, and
   verify the full current-frame-to-consensus budget on hardware.

## Evidence limits and integrity

All **1,050 copied source artifacts** match the read-only transfer
[manifest](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/source_manifest.json).
Each camera attempt retained 64 raw captures; later images exceeded the capture
quota. Full observer event logs support the counts and timing statistics in
[`camera_metrics.json`](../../results/audits/stand_explore_exact2_camera_all5_20260910T145318Z/derived/camera_metrics.json).

Replays use original images and recorded CameraInfo rectification. The local
OpenCV 4.13 environment lacks WeChat decoding, so absent offline identity text
does not prove absent live decoding, and not every recorded per-ROI outcome
was reproduced. Frame 000045's QR seed and border failure, and frame 000022's
false-marker branch, were reproduced explicitly. The manual border diagnostic
is labelled as such and creates no fresh sensor or motion evidence.

This audit changed no production code and issued no robot commands.
