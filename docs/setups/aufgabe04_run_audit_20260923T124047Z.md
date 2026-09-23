# Workstation run audit: 20260923T124047Z

Audited on 2026-09-23. Read-only inspection of workstation run artifacts, captured camera frames, and the recorded source revision. No robot commands or production changes were made.

## Outcome and provenance

- Run: `stand_explore_exact2_camera_all5_20260923T124047Z`.
- Bundle records revision `6c0734cedb6e3b104c592fc7e7693f25a8d99664`, branch `main`, and clean working tree.
- Started 12:40:47 UTC / 14:40:47 CEST; finished 12:53:41 UTC / 14:53:41 CEST. Elapsed **12m54s**, exit code **0**.
- Mission completed with five distinct identities. Two have facing geometry; three are QR-only. `facing_complete=false` and `camera_geometry_complete=false` are consistent with this result.
- Six LiDAR hypotheses existed. Candidate 0004 was left unvisited because the five-identity goal was reached; it remains a keepout. Station/server identity binding remains pending.

| Inspection order | Candidate | Identity | Outcome | Observer capture elapsed | Processed images |
|---|---|---|---|---|---|
| 1 | 0003 | QR_003 | Geometry recommendation | 2.43 s | 1 |
| 2 | 0001 | Start | Backside, then opposite-side QR-only | 9.39 s + 2.26 s | 25 + 1 |
| 3 | 0002 | QR_002 | Geometry recommendation | 9.25 s | 24 |
| 4 | 0006 | QR_004 | QR-only | 3.60 s | 6 |
| 5 | 0005 | QR_001 | QR-only | 10.10 s | 17 |

Capture elapsed is the orchestration interval, including observer startup and cleanup, not just detector compute. All six observer processes finished with an artifact, return code zero, and no deadline expiration. Their combined capture intervals were about **37 seconds** of the 774-second run. Camera compute alone therefore does not explain the mission duration.

## Start succeeded, but the center handoff is still broken

The backside receipt retained a seven-sample orientation interval: **83.02 degrees ±6.18 degrees**. The opposite-side QR receipt explicitly records `current_angle_refit=false`; the observer did not need a new stand-angle fit to admit Start.

However, the validated target center was not carried into opposite planning:

1. Backside reconciliation was ready in **23 of 25 captured frames**. Fifteen frames contained both a current association proof and metric head-position evidence at the outer `detector_metadata` level.
2. The final frame, `frame_000025.json`, contains both proofs for exactly the same sensor stamp (`1790167648.32168`) used in `axis_observation.json`.
3. The saved `axis_observation.json` omits both `target_reconciliation` and `head_position_evidence`.
4. Opposite route diagnostics and arrival admission consequently contain `validated_target_center: null`.

The code explains the loss deterministically:

- `observer/node.py` populates `axis_metadata["current_head_candidate_association"]` and `axis_metadata["head_position_evidence"]` around lines 2152–2160.
- Its `prepare_bounded_head(...)` call instead supplies `metadata=model_metadata` around line 2279.
- `observer/bounded_head_observation.py`, around lines 154–155, reads the two proof fields from `current.metadata`. Those keys are absent from `model_metadata`, so the bounded receipt builder receives `None` for both.
- The ordinary precise-angle path passes the proofs directly; this run used the distinct bounded-angle path.

This is a metadata wiring defect, not failure to gather three scans or a late-arriving proof. The suggested correction is to pass current reconciliation and position evidence explicitly through the bounded-head structure, with a regression test covering the real observer-to-bounded-receipt handoff. Preserve candidate identity, provenance, timestamps, and uncertainty; do not substitute an unchecked center.

## What the opposite-side image shows

The admitted QR center is **(509.51, 272.99) px** in an 800×600 image. The calibrated optical center is x=405.87 px: the QR remains **103.64 px to the right**, corresponding to an approximately 9.18-degree horizontal image ray. This image-ray angle is not directly a robot yaw command; translation and camera rotation must still be applied.

The finite-range, calibrated camera-to-map association discrepancy is **7.54 degrees**, below the existing 12-degree bound, with a unique current LiDAR cluster. The previous failed run's corresponding discrepancy was roughly 13.3–13.5 degrees. This run therefore had framing that passed the original association gate despite the missing center handoff.

- Arrival explicitly records `camera_centered=false`; no centering-turn artifacts were produced.
- Start admitted on the **first processed opposite-side image**. Two capture-history entries exist because the earlier captured tuple did not reach detector processing.
- Decoder result: `opencv_quad_wechat_rectified`; isolated native-quad confirmation returned `Start`. Decoder elapsed was approximately **7.65 ms**.
- Publication freshness passed: image age 0.373 s, scan age 0.352 s.

Immediate candidate-associated QR admission worked. Successful admission does **not** demonstrate that the validated-center planning handoff or bounded centering correction worked. With immediate QR completion, no centering turn was needed for this run's discovery goal.

The latest standalone viewer recording found during the audit was still `recording_20260923_134105_124402383`, preceding this run. The run's own captured images are the contemporaneous evidence used here.

## Remaining delays and recoveries

**One pre-motion map→odom acquisition failure.** Before the second coverage leg, preflight/sealing had obtained map→odom, but the follower's initial acquisition later reported that `map` did not exist in its buffer. All 31 recorded execution-pose lookups were ready; the global-consistency edge was unavailable. The attempt stopped after 5.15 s, with zero distance and `motion_published=false`. A startup reseal recovered automatically and the replacement leg completed. About 52.3 s elapsed from the recorded stop to replacement permit issuance. This supports investigating TF-listener lifecycle/readiness across the preflight-to-follower boundary; it does not prove AMCL globally stopped publishing, and it is separate from camera load because it occurred before candidate inspection.

**Two opposite-route admission rejections.** The 0.50 m and 0.45 m standoff options exhausted the route uncertainty budget (remaining margins −0.121 m and −0.0278 m). The 0.40 m option passed and completed. No motion was published for the rejected options. Screening equivalent uncertainty constraints earlier could avoid repeated child setup, while preserving the admission limits.

**Residual camera freshness losses.** The final candidate recorded 46 overwritten ingress images and two obsolete detector results, but admitted QR_001 in 10.10 s. Overwrites represent latest-frame coalescing, not 46 independent processing failures. Candidate 0002 processed 24 images over 9.25 s. No candidate consumed its 90-second timeout.

**Other elapsed time.** Startup active localization took 54.96 s of controlled rotation. Each of the five initial arrival checks took about 5.1 s, separate from camera capture. Route following, preflight, localization, and process orchestration remain substantial portions of mission time; the recorded capture intervals should not be used as an exhaustive timing decomposition.

## Recommended order

1. Correct the bounded-angle receipt's proof handoff and test it end to end through opposite route materialization and arrival projection. This is the remaining correctness issue directly demonstrated by the run.
2. Diagnose the cold TF-buffer boundary that caused the coverage startup reseal; retain a live listener or transfer readiness within a single validated lifecycle rather than weakening freshness gates.
3. Profile route admission and repeated child startup before further QR decoder optimization. The successful opposite decode itself took only milliseconds.

## Evidence retained locally

Under `results/implementation_checks/run_audit_20260923T124047Z/`:

- `evidence.tar.gz`: run JSON/JSONL/CSV evidence excluding image/large binary payloads.
- `captures.tar.gz`: candidate capture history, including original compressed images.
- `bundle.tar.gz`: manifest, recorded revision, git status, and terminal log.
- Extracted original paths under `results/`.
- `audit_summary.json`: compact verified handoff/QR measurements.
- `start_opposite_admitted.jpg`: byte-identical copy of the admitted opposite image.

The audit checked same-stamp proof availability and absence in the committed receipt with assertions. These are forensic checks, not a new hardware validation or a production regression test.
