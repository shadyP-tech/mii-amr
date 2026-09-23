# Second-candidate admission audit: 20260922T150953Z

The latest run failed because the new opposite-side identity crop removed the
bottom of the target QR code. A separate timeout integration omission then
terminated the mission instead of permitting the existing bounded inspection
recovery. Both issues are in the change deployed as `ab89c3c`.

## Evidence and scope

- Run: `stand_explore_exact2_camera_all5_20260922T150953Z` (17:09:53 local start).
- Recorded revision: `ab89c3cc018c6de0b4caf9d822bb5e90958fea5c`, clean worktree.
- First candidate: `survey_candidate_0003`, admitted as `QR_003`.
- Second candidate: `survey_candidate_0001`, backed by seven backside samples.
- Artifacts copied read-only from `mii001` into
  `results/implementation_checks/opposite_identity_audit_20260922T150953Z/`.
  `provenance.json` records archive hashes; `summary.json` records counts.
- Counterfactual crop replay used the existing Apptainer image, OpenCV 4.5.4,
  WeChat backend available, repository mounted read-only, and recorded images.
  No ROS node, robot command, deployment, or production-code change was made.

## What worked

The opposite route completed at the 0.40 m standoff after the larger proposals
failed dry preflight. Arrival was admitted at 0.4067 m from the stored candidate,
with only -0.345 degrees of bearing error relative to that stored center.

The observer loaded the retained backside orientation: 85.293 degrees with
6.207 degrees of half-width uncertainty, seven source samples. Every processed
opposite-side image reported `current_angle_refit=false`. It did not wait for a
new frontside angle or reject a decoded identity for missing QR corners.

There were 404 processed images: 360 accepted identity crops and 44 ambiguous
searches containing two eligible LiDAR clusters. Every frame reported
`no_decoded_qr_identity`; no QR payload reached the completion gate. The event
stream also contains 27 exhausted exact-TF tuples. Those losses did not prevent
360 associated frames, and all 251 processed tuples retained in capture history
finished below 0.47 seconds of source age. Timing was not the primary blocker.

## Regression 1: neighbor exclusion truncates the QR

`real_robot/observer/opposite_identity_crop.py:49` clips a candidate crop to one
side of any overlapping projected neighboring head. It chooses the largest
remaining rectangle containing the projected target center. Containing a point
does not establish that the complete target QR fits inside the rectangle.

All 360 accepted crops ended at image row **320**. For captured frame 50:

- Crop sent to the decoder: `[386, 186, 589, 320]` (x0, y0, x1, y1).
- Excluded neighbor `survey_candidate_0003`: `[448, 320, 496, 368]`.
- The visible foreground target QR continues to approximately row **330**.

The neighboring stand is in the background. The exclusion logic uses its
projected rectangle without resolving occlusion, and the projected-center test
allows a crop that has lost the target symbol's bottom rows. It still labels
that crop `exclusive_current_target_crop` and sends it to the decoder.

The stored candidate location still differs from the observed foreground head
position. This makes projected neighbor/target boundaries especially unsuitable
as an unconditional pixel cut. The crop regression is independently demonstrated
by changing only the bottom boundary on the exact same recorded images.

### Controlled replay with the deployed decoder

Twelve frames were sampled evenly across the 223 accepted crops retained in the
256-frame capture history. Each variant had the same 120 ms decoder budget,
preferred scale 4, and its own reused decoder resources. Recorded CameraInfo was
used for the production rectification function.

| Diagnostic crop | Successful decodes | Median elapsed time |
| --- | ---: | ---: |
| Actual recorded crop ending at row 320 | 0/12 | 107.3 ms |
| Same crop, bottom restored to row 343 | 12/12, all `Start` | 24.7 ms |
| Complete target-head crop `[442,207,578,343]` | 12/12, all `Start` | 21.5 ms |

Restored variants decoded through `opencv_quad_wechat_rectified`. These are
diagnostic comparisons that deliberately bypass the conflicting neighbor cut;
they are not candidate-admission proofs or proposed hard-coded production ROIs.
See `replay_crops.py`, `deployed_crop_replay.log`, and
`deployed_crop_replay.json` in the evidence directory.

The terminal contains missing-QUIRC messages from native OpenCV. That is a
backend limitation, but WeChat successfully decodes these same frames once the
whole QR is preserved. Increasing camera timeout, changing lighting, or waiting
for additional angle samples would not restore pixels removed by the crop.

## Regression 2: new observer state bypasses bounded recovery

The observer reached its 90-second deadline in `opposite_identity_collecting`.
The parent sent SIGINT and reaped it with return code 130; this was expected
deadline cleanup, not evidence of an independent process crash.

`real_robot/observer/timeout_policy.py:21` does not include the new state in
`CANDIDATE_LOCAL_OBSERVER_TIMEOUT_STATES`. Its accumulated-evidence fallbacks
also do not include this state. Consequently, the parent at
`real_robot/autonomous_runner/runtime.py:1647` rejected candidate-local timeout
classification and raised a plain `RuntimeError`. The inspection loop did not
receive `CandidateObservationUnavailableError`, so it did not attempt another
view despite `--max-candidate-inspection-views 8`.

A read-only policy replay of the actual status/process artifacts returns false
for candidate-local timeout. Changing only the state to an existing recognized
quality state returns true. There is no status parse error or poisoned identity.
`mission_failure.json` records `failed_closed` and prohibits continued motion.

The diagnostic text `without a usable axis; consensus=0/7` is misleading here:
zero new axis samples are intentional, and the retained orientation exists.
The fallback's initial `delay_sec=1.5` and null axis diagnostic likewise do not
mean the branch was waiting for grace or refitting; no QR decode reached the
code that switches those diagnostic fields to retained-angle completion.

## Targeted corrections

1. Preserve the complete target symbol and its border before decoding. Resolve
   projected neighbor overlap using current target support and depth/occlusion,
   without requiring a fresh angle fit. If association is unresolved, report a
   crop conflict and allow bounded recovery instead of accepting a truncated
   crop. Do not blindly drop all neighbor exclusion or hard-code row 343.
2. Integrate opposite-identity acquisition states with the existing typed,
   bounded timeout policy. Preserve conflict, source, and cleanup checks.
3. Distinguish retained-axis identity acquisition from new-axis consensus in
   timeout diagnostics. Add a recorded foreground/background overlap regression
   and a state-to-parent-timeout integration regression; the previous synthetic
   crop tests did not exercise this partial-symbol case.

No fix was implemented during this audit.
