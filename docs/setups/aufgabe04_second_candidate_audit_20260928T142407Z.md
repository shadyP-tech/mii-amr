# Second candidate admission audit: 20260928T142407Z

The second visited stand (`001_survey_candidate_0001`) reached its opposite-side
view, but the observer rejected its visible QR before payload decoding. Navigation
and camera association used different target centers. A read-only replay in the
workstation's OpenCV 4.5.4 environment reproduced the rejection; substituting the
already validated target bearing allowed two sampled frames to pass the crop
checks and decode `Start`.

This is an investigation, not a production fix or a successful live admission
test. No robot motion or ROS nodes were started for the audit.

## Run and evidence

- Workstation connection: `mii002`.
- Run: `stand_explore_exact2_camera_all5_20260928T142407Z`.
- Checkout inspected: `f644ecb3037926a87a2a9fab48827ea581a03e60`.
- Local evidence: `results/implementation_checks/run_audit_20260928T142407Z/`.
- Runtime evidence: `mission/candidates/001_survey_candidate_0001/`, particularly
  `camera_lidar_attempt_00/axis_observation.json`,
  `inspection_opposite_01/camera_attempt_01_arrival/`, and
  `camera_lidar_attempt_01/{observer_status.json,observer_events.jsonl,observer_process.json}`.
- Replay script/output: `replay_opposite.py`, `replay_workstation.txt`.
- Saved opposite-view images and matching sensor/TF metadata: `capture/`, with
  compressed image samples 10, 100 and 200. The full metadata capture contains
  256 frames; the observer processed 429 frames.

## What succeeded

The first camera attempt produced seven accepted backside-angle samples. The
opposite route passed preflight and finished at approximately 14:32:24 UTC.
The arrival-frame retained axis was 1.473094 rad, with uncertainty half-width
0.110560 rad (6.335 degrees); the opposite face normal was 3.043891 rad.
The opposite observer correctly reported `current_angle_refit=false`.

The stopped arrival passed the strict three-degree heading check with only
0.6295 degrees error relative to the validated target center. Distance was
0.42537 m. In rectified frame 100, the complete foreground QR center was near
x=406 in an 800-pixel-wide image. This run's evidence does not support a large
camera-centering error at the failed opposite view. Arrival admission itself
does not certify live camera association (`camera_centered=false`).

## Root cause: two position hypotheses

| Consumer | Target map position, metres |
| --- | --- |
| Opposite arrival check, using validated backside target | (-1.164355, -0.445027) |
| Observer search and association, using reprojected survey candidate | (-1.175352, -0.348405) |

These positions differ by about 9.72 cm. The validated center carries a 2.41 cm
engineering uncertainty bound.

In `real_robot/observer/node.py`, the scan target is still constructed from
`args.stand_x` and `args.stand_y`. Its bearing and range interval are passed into
`process_opposite_identity`, even though that context contains the retained
orientation and validated target center.

In recorded frame 100, the survey-based scan bearing is +12.626 degrees. The
validated-center bearing is +0.379 degrees. The detected QR's finite-distance,
camera-offset-corrected bearing is -1.518 degrees. Including its uncertainty,
the difference from the survey reference is 14.493 degrees, exceeding the
existing 12-degree association limit. Relative to the validated center it is
only 2.247 degrees.

`opposite_target_support.py` therefore discards the complete QR outline before
it can supply foreground support to the crop resolver. Without that support,
the generic search crop overlaps the projected background stand
`survey_candidate_0003`. `opposite_identity_crop.py` rejects that unresolved
overlap. `opposite_identity.py` invokes payload decoding only after an accepted
crop. This explains why a visibly readable QR never became an admitted identity.
The same absent support also prevents a centering receipt.

## Additional range and recovery limitations

The survey-based range interval in frame 100 is [0.260276, 0.480276] m. Returns
near the visible QR bearing are 0.484–0.485 m and are excluded. Other sampled
frames contain 0.488–0.494 m returns. The search often retains only one or two
beams at its upper range boundary; the forward target also spans the scan's
0/360-degree seam, where recorded topology diagnostics disable circular joining.
This contributes to empty or split search results and prevents reliable
three-beam reconciliation.

`target_reconciliation.py` applies the original candidate range and envelope
before it uses the retained center as a bearing reference. It cannot reliably
repair a target already clipped by those inputs. All 429 status records reported
no target reconciliation in the decoded-identity binding.

The recently added position-epoch recovery requires an **empty** ordinary search
envelope. Most frames here have a nonempty envelope, so that recovery does not
address their mismatch. The candidate's recorded localization displacement is
10.10 cm, inside the recovery displacement limits; displacement size alone is
not the blocker. The five-second opportunity timer also requires a valid epoch
recovery proof, so it does not bound this persistent crop-conflict case.

## Runtime outcome

| Opposite observation outcome | Frames |
| --- | ---: |
| `target_crop_overlap_unresolved` | 348 |
| `qr_search_cluster_not_unique` | 77 |
| `stale_or_unsynchronized_search` | 4 |
| Accepted observation frames | 0 |

There were 431 synchronized tuples and 429 TF-ready processed tuples. The
observer reached its 90-second deadline without an artifact and was stopped by
the parent with SIGINT (return code 130). The dominant failure is association,
not missing TF or another head-angle fit.

The parent subsequently prepared another inspection view. Its preflight passed,
but the captured terminal log ends at 14:34:44 UTC while collecting pre-run
diagnostics. The available evidence does not establish why the overall process
ended after that point; this audit establishes the preceding camera timeout.

## Offline counterfactual

Replay used the recorded images, calibration, exact sensor/TF metadata, candidate
snapshot, and production crop/QR functions in the workstation container.
Only the search/reference bearing was replaced by the bearing to the retained
validated center. Range limits and competing-candidate crop checks remained.

| Saved frame | Original reference | Validated-center reference |
| --- | --- | --- |
| 10 | Crop rejected | Still rejected: range-clipped ray support |
| 100 | Crop rejected | Crop accepted; decoded `Start` |
| 200 | Crop rejected | Crop accepted; decoded `Start` |

Support-stage elapsed time was frozen for this geometric diagnosis, and the
payload decoder received a one-second offline budget. These results demonstrate
the association/crop failure and recoverable identity, not compliance with the
live timing budget or full end-to-end admission.

## Recommended correction

1. Resolve one opposite-view target-position context from the validated retained
   center, its uncertainty, candidate identity, and current planning frame. Use
   it consistently for search bearing, range bounds, current-cluster validation,
   crop support, and centering. Preserve the retained angle and its uncertainty.
2. Keep reconciliation to the survey candidate and exclusion of competing
   candidates as separate checks. Validate current scan support, including
   topology constraints; do not globally widen association thresholds or bypass
   foreground/background identity checks.
3. Make persistent opposite-view crop/association conflicts enter bounded
   recovery even when there is no position-epoch proof. Record explicit support
   rejection reasons, including bearing mismatch and excluded range returns.
4. Once a complete current candidate-associated QR decodes, complete admission
   using the retained opposite orientation without another front-angle fit.

Validation should replay these three frames plus neighboring-target and invalid
retained-center negatives, then verify the full receipt/admission chain and the
unchanged live freshness/decode budgets.

## Implemented correction and validation

The correction uses `observer/opposite_target_geometry.py` for both camera/scan
search and independent scan witnesses. It retains the candidate snapshot as the
identity reference while projecting the certified target center, including its
uncertainty on both ends of the scan range interval. It does not enlarge the
camera-bearing limit or fit a new stand angle.

The new `certified_center_current_scan_confirmation` receipt confirms an already
certified center with one fresh, unique, compact cluster containing at least
three real beams. It revalidates the retained source chain, current frame, range,
candidate displacement, opposite side, and competing candidates. Ordinary
survey reconciliation still requires three distinct stopped scans. Updated
opposite-view decoding requires this current confirmation; retained position
alone cannot authorize identity admission.

`observer/opposite_identity_opportunity.py` bounds repeated fresh association/crop
conflicts at five seconds. Its clean child exit enters the existing bounded
inspection recovery. Motion, stale data, target changes, repeated timestamps,
poisoned identity, and productive crop results reset that opportunity. A QR or
centering artifact has priority over the failure exit.

The corrected complete range envelope exposes two raw scan-boundary groups in
many recorded frames. These remain ambiguous. Frame 24 provides a unique current
cluster and is preserved, with its exact original camera payload, in
`tests/aufgabe04/fixtures/opposite_center_20260928/`. Tests also preserve the
ambiguous frames and verify that neither their groups nor their identity are
silently admitted.

Validation completed:

- 190 targeted tests passed, plus 217 subtests; one local test skipped because it
  requires the deployed WeChat decoder backend.
- On `mii002`, the production opposite-view functions replayed frame 24 using
  OpenCV 4.5.4 and the real payload decoder. The result was `Start`, an accepted
  complete-symbol crop, and a validated QR observation receipt with the retained
  angle unchanged.
- That replay kept the production support/decode budgets and advanced its clock
  with actual processing time: approximately 80 ms processing and 209 ms image
  age at receipt creation, below the existing 500 ms freshness bound.
- Replay output is saved locally as
  `results/implementation_checks/run_audit_20260928T142407Z/production_replay_result.json`.
  The temporary remote replay directory is
  `/tmp/a04-retained-center-replay.3sTWlY`; both checkout and source overlay were
  mounted read-only in the container. No ROS nodes or robot motion were started.

The correction is in the local working tree. The workstation production checkout
was not changed by the offline replay. Live sensor delivery, navigation and the
complete real-run lifecycle remain to be verified on the robot after deployment.
