# Candidate head recentering: recorded feasibility and integration solution

Status: offline geometry verified; live rotation behavior is not implemented or
deployed by this audit. No robot motion was performed.

## Finding

The initial inspection arrival points the base at the stored map candidate. It
does not close the loop on the currently observed head center. In
`stand_explore_exact2_camera_all5_20260917T140949Z`, candidate
`survey_candidate_0001` passed the 3-degree map-bearing arrival gate with a
1.188-degree error, while its head appeared at u=270 in an 800-pixel image.
The mapped candidate projects to u=373.5 using the recorded camera extrinsics.
Thus the mounting offset alone does not explain the displacement. Current
camera-associated scan returns also lie left of the stored target.

The effective correction is a small turn at the current base position, based
on the measured image center and current associated scan. It must not wait for
the seven-sample stand-angle consensus. Centering is distinct from estimating
which direction the head faces and does not admit a stand angle or QR identity.

## Recorded geometric feasibility

The offline check automatically selected all five accepted current-head
associations in the saved capture metadata. It used rectified image centers,
intrinsics, full camera-to-base and camera-to-scan transforms, and associated
scan range. It did not use fitted head translation, face normal or stand yaw.

| Quantity | Result |
| --- | --- |
| Observed head centers | u=269.50–270.51 px |
| Desired horizontal center | u=400 px |
| Required base rotation across the five captures | 10.50–10.58 degrees left |
| First capture's simulated 6-degree turn | u=344.65 px |
| Simulated remaining 4.54-degree turn | u=400 px |
| First usable capture | capture 25, about 5.47 seconds into capture history |
| Old map-bearing error after the full nominal turn | −9.35 degrees: old arrival gate would reject it |

For capture 25, perturbing the range proxy by ±6 cm changes the computed turn
from 10.41 to 10.64 degrees. Applying the nominal turn to those perturbed
points leaves the projected center within 1.55 px of the target. This is a
sensitivity experiment, not a certified uncertainty interval: the scanner
measures a surface, not the exact head center. A new image must close the loop.

The calculation intersects the calibrated camera ray through the measured
center with a cylinder of the associated scan range. It then rotates that
point into the hypothetical new base frame and projects through the full
camera extrinsic. Solving for u=400 accounts for camera translation around the
base pivot, mounting yaw, pitch and principal point. A rotation-only pixel ray
does not include that near-range parallax.

These are counterfactual projections. There is no post-turn recording proving
that the border fitter improves after this correction. In particular, turning
does not guarantee recovery of the faint right border seen in failed frames.

## Runtime behavior to implement

1. Keep the existing map-based approach and initial arrival admission. Begin
   the normal stopped observation at every inspection view, including a view
   reached through the opposite-side branch.
2. As soon as a fresh, current head is uniquely associated with the candidate's
   scan cluster, expose its image center as a separate centering advisory.
   Accept a strict head fit or the existing independently validated head
   orientation bounds; do not require a precise angle or seven-frame consensus.
   A missing raw border remains a missing observation. A completed, bound QR
   observation retains priority and can finish discovery immediately.
3. Compute a signed base-yaw correction with calibrated geometry. Use an
   approximately one-degree image deadband (`fx * tan(1 degree)`, 11.2 px for
   this calibration). Initial test bounds: at most two turns, each at most
   6 degrees, at most 12 degrees total measured angular travel per physical
   inspection view, zero linear velocity and angular speed at most 0.12 rad/s.
   These are proposed test bounds, not existing certified runtime settings.
4. Execute each turn through a separately admitted inspection-rotation action
   in the existing sole command-publishing follower. Obtain fresh scan and
   odometry at motion preflight, verify the observation's candidate and pose
   anchor, monitor clearance during rotation, then stop and prove stationarity.
   Use small-angle deceleration and stopping tolerance; the startup rotation's
   default 4-degree tolerance is too coarse.
5. Explicitly begin a new observation epoch after every turn, including turns
   smaller than the existing 2-degree stationarity threshold. Discard pre-turn
   axis samples, QR latches, tracking authority and scan witnesses. Require
   advancing image and scan timestamps before checking the new center or
   proposing the second turn. Count actual absolute angular travel, including
   overshoot/reversal; capture restarts and failures cannot reset the budget.
6. Use the admitted turn receipt and fresh candidate-bound observation for
   post-turn admission. Preserve range, displacement and candidate-identity
   checks, but do not reapply the obsolete map-facing yaw as the centering
   criterion. Do not rewrite the global candidate location or QR identity from
   this local advisory. If no trustworthy observation arrives or the budget
   expires, return to the bounded existing inspection policy without claiming
   the view is centered.

## Code seams

| Component | Required change |
| --- | --- |
| `real_robot/observer/current_head_detection.py` and `current_head_association.py` | Reuse the current measured-center and independently validated orientation-bound contracts. |
| `real_robot/observer/node.py` | Prepare an early centering advisory after current crop and identity-conflict checks. Publish only after `_record_observation_frame()` accepts the frame in an unpoisoned stationary epoch. Keep QR completion ahead of optional centering. |
| Observer status / observation process boundary | Return a separate motion-neutral centering receipt before the slow consensus/progress path. Existing inspection progress waits seven frames and suppresses output while consensus is collecting. |
| `real_robot/candidate/inspection_execution.py` and `inspection_adapters.py` | Handle centering within the current physical view, share it across initial/diverse/opposite views, persist the turn/time budget, and restart observation after the stopped turn. Capture restarts must retain the original view's observation deadline. |
| `navigation/execution/mission_leg_motion_permit.py` and follower | Add a candidate-bound yaw-only inspection purpose and guarded small-angle action. There is currently no generic sealed inspection-turn contract. Do not manufacture a duplicate-waypoint route or reuse a startup-localization permit. |
| `real_robot/candidate/approach.py` arrival admission | Distinguish original map-facing arrival from receipt-bound post-turn admission; otherwise a successful visual turn is rejected or undone. |

Keep the implementation limited to a pure geometry/policy helper, the early
observation receipt, one follower turn action, and lifecycle integration. No
second detector, broad scan-only target selector, or global Canny change is
required to address this centering defect.

Widening the map-centered scan cone is not an equivalent fix. In capture 5,
an experimental ±12-degree search contains two clusters; the legacy helper
still ranks one as associated and picks the fragment nearer the old map ray.
The existing 12-degree camera/map allowance is justified by current image
geometry plus a narrow unique scan match, not by an unconstrained broad search.

## Validation and evidence

The executable check and its JSON are in
`results/aufgabe04/debug_audits/recording_20260917_161701_719967086/`:

- `recenter_projection_check.py`: offline calculation and analytic sign,
  left/right/center, out-of-budget and projection round-trip checks.
- `recenter_projection_check.json`: all five observations, input SHA-256s,
  range sensitivity and predicted failure of the old map-bearing gate.
- `camera_center_offset_audit.json`: recorded map, camera and scan evidence.

Run from the repository root, with copied mission metadata available:

```bash
PYTHONPATH=. python3 results/aufgabe04/debug_audits/recording_20260917_161701_719967086/recenter_projection_check.py --mission-root /path/to/copied/mission
```

Before live integration, test receipt candidate/epoch/calibration binding,
stale and conflicting evidence rejection, immediate QR completion, explicit
epoch reset for sub-2-degree turns, budgets surviving restarts, and a successful
post-turn observation with a map-bearing error above 3 degrees. Follower tests
must cover turn sign, tight stopping, sensor loss, obstacle stop and exclusive
command ownership. An eventual stationary robot trial must supply actual
before/after images; offline projections do not substitute for that evidence.
