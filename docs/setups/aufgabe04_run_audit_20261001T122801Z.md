# Real-run audit: 20261001T122801Z

Run `stand_explore_exact2_camera_all5_20261001T122801Z`, October 1, 2026,
**14:28:01–14:37:28.630 Europe/Berlin**, workstation SSH alias `mii002`.
All seven parent/motion-child bundles record clean revision
`915250ef42d5e7fa9e80a0127c761a13b8baaa72`, containing the coarse-front
30-degree and Start-return corrections. All **542 copied source files**
match their workstation SHA-256 hashes. A second read-only workstation check
at **14:46:09.600 Berlin** found the same latest run, revision and clean
checkout, with all 542 files unchanged, including their modification times.

## Finding

The opposite-side move succeeded. The camera then showed the complete front
of the **Start stand**, and offline decoding of two original frames reads
`Start`. Live admission stopped **before QR decoding**, because the current
LiDAR target was split across the beginning and end of the scan array.
Reconciliation saw two eligible clusters and could not certify one target.

This is an unresolved target-association case at the LiDAR scan seam. The
retained head angle was present and valid; the opposite-side identity path
does not request a new head-angle fit. No evidence indicates a wall-only
target, a stale retained-center projection, a QR decoder failure, or rejection
by the new 30-degree front-facing policy in this attempt.

Only **QR_003** was admitted and facing-ready before the run ended. Start
was not entered into the live discovery catalog. All six candidate keepouts
remained. There was **no return-to-Start attempt**, so this run does not
validate the recently implemented Start-return correction.

## Sequence

| Time, Berlin | Observed event |
|---|---|
| 14:34:04.672 | Last backside capture; the first camera attempt yields seven-frame certified orientation. |
| 14:34:21.981 | Initial opposite route fails dry uncertainty admission, before motion. A smaller-standoff alternative is selected. |
| 14:35:45.867 | Alternative opposite-side motion completes normally. |
| 14:35:54.936–14:36:00.002 | 25 fresh, exact-TF camera/scan tuples all fail current target reconciliation. |
| 14:36:00.329 | Observer publishes terminal `opposite_identity_association_exhausted` status. |
| 14:36:00.766 | Parent receives the no-artifact result and starts bounded inspection recovery. |
| 14:36:55.546 | A diverse inspection route begins after four blocked route proposals. |
| 14:37:25.274 | Localization consistency monitor stops that recovery drive. |
| 14:37:28.630 | Parent records `KeyboardInterrupt`; no final parent exit code or mission summary is present. |

Five motion legs completed. The sixth, recovery inspection, stopped after
approximately **0.649 m**. Its independent stop reason was map-to-odometry
yaw drift **3.7848 degrees**, exceeding the certified **3.6820-degree** limit.
Translation drift was **0.02717 m**, below its **0.06406 m** limit, and TF
validation passed. The monitor requested `FORCE_ZERO_RESEAL`. This occurred
3.356 seconds before the recorded parent interruption; the artifacts do not
identify who or what initiated that interruption.

## Why the visible candidate was not admitted

The post-opposite observation has a certified retained orientation based on
seven backside frames, with **±7.839893 degrees** uncertainty. Its retained
metric target center is `(-1.134750, -0.474741)` in the arrival map frame,
with **0.026208 m** uncertainty. The corresponding projected survey center
is `(-1.143944, -0.417203)`; these are deliberately different geometry sources.

The observer uses the retained metric center correctly. In the final tuple,
its scan bearing is **0.011871126 rad**, exactly matching the search bearing.
The original survey center would project to **0.137015391 rad**. Thus this is
not an accidental search at the unreconciled survey position. Static target
admission and arrival geometry also passed; the recorded optical bearing
error was **2.615 degrees**, below its **3-degree** arrival limit.

Every one of the 25 usable tuples contains **five or six in-range LiDAR
returns**, but those returns form **two eligible clusters**. The final tuple
illustrates the split:

| Scan indices | Bearings in degrees | Ranges |
|---|---|---|
| 0, 1, 2 | +0.320, +1.920, +3.520 | Approximately 0.480–0.485 m |
| 221, 222, 223 | −6.080, −4.480, −2.880 | Approximately 0.480–0.485 m |

These returns lie on either side of zero bearing, where the indexed scan
ends meet. Circular adjacency is disabled in **all 25 tuples**: 19 fail
endpoint metadata consistency, and six fail the one-sampling-step seam-gap
check. The final scan has 224 samples, an endpoint metadata error of about
**0.97184 sampling steps**, and an indexed seam gap of **2.00020 steps**.
It would be incorrect to assume that its first and last beams are ordinary
adjacent samples and globally join them without further evidence.

Offline replay of all 25 selected scans through the unchanged production
association, numeric reconciliation and QR-search functions reproduces the
same rejection and absent crop. All recorded envelope fields match within
`1e-12`. All 25 fragment pairs also satisfy the existing *bounded endpoint
fragment eligibility* check:

| Diagnostic over the 25 scans | Recorded range | Existing geometric bound |
|---|---|---|
| Distance between seam-end returns | 15.18–27.02 mm | 40 mm |
| Range jump across seam | 0–6.00 mm | 50 mm |
| Diameter of both raw groups together | 55.69–80.83 mm | 160 mm reconciliation limit |
| Combined raw centroid to retained metric center | 18.93–31.51 mm | Diagnostic only |
| Combined raw centroid to projected survey center | 70.88–88.80 mm | 160 mm displacement limit |

Thus real returns are present near the intended target. Eligibility for
bounded fragmentation handling does not itself certify that two groups are
one object. The existing historical proof still needs three independently
unique earlier scans; none of the 25 captured scans can become such a witness.
The full independent 58-scan witness stream was not saved as raw scan/TF
inputs here, so its precise registration-history state cannot be replayed.
The saved sequence cannot establish a new continuity proof by itself; the
live result establishes that no accepted proof reached reconciliation.

The observed causal chain is:

1. Current target reconciliation requires a unique compact cluster with at
   least three current beams, or a validated historical fragmentation proof.
2. Every tuple fails with `reconciliation requires one compact three-beam cluster`.
3. Because the retained target lacks that current confirmation, the opposite
   identity path marks its crop `retained_target_reconciliation_pending`.
4. Target support reports `search_unavailable`; no target crop is admitted,
   and the live QR decoder is never invoked.
5. After **5.065701 seconds** of fresh stationary association misses, the
   observer exits cleanly with `persistent_opposite_target_association_conflict`.

The parent message's `fresh_detector_results=0` and `consensus=0/7` do not
identify an angle-estimation failure: this branch reuses its certified angle.
The decisive counters are **25 LiDAR rejections and zero accepted candidate
frames**. There were no TF retries, poisoned observation evidence, process
timeout, or observer crash. One earlier stale tuple was discarded; all 25
processed tuples passed source freshness at the evidence gate.

## Camera evidence and replay limits

The following image is an unchanged JPEG byte copy of original capture 26,
verified against the capture's image hash:

![Start stand after the opposite-side move](/Users/stephpark/Documents/stephsWorld/mii-amr/results/implementation_checks/run_audit_20261001T122801Z/camera_start_after.jpg)

The production QR decoder reads `Start` from original captures 2 and 26 in
an offline diagnostic check. This confirms readable pixels. The check uses
the original full frame and a one-second desktop budget; live processing
would use a rectified, admitted target crop and at most 0.12 seconds. It does
not establish live timing compliance or replace current candidate association.

## Relationship to the recent corrections

The scan topology, target reconciliation, opposite identity, conflict-window,
endpoint fragment and persistence modules are unchanged across both
`99ac300 → fbe6926` and `fbe6926 → 915250e`. Endpoint topology checks date to
September 9; the unique three-beam reconciliation requirement dates to
September 23, with shared witnessed-fragment support added September 28.
The retained-target reconciliation prerequisite and five-second opposite
conflict exit were introduced September 28 (`696b1ef`).

The October 1 commit changes coarse-front admission and Start-return
planning. Its observer change reorders ordinary front evidence and QR-only
fallback handling; it does not change these opposite-side gates. This
establishes that the failing policy predates the latest fixes. It does not
establish that this particular physical run would succeed on older code.

## Correction identified by the audit

The audit identified **current target confirmation for bounded scan-end
fragments** as the correction scope. Increasing the angle limit or waiting
longer at the same pose does not address the observed prerequisite failure.

The existing persistence mechanism requires three independently unique
historical scan witnesses; ambiguous fragments cannot certify themselves.
A correction must preserve current beam geometry, target/neighbor exclusion,
retained-center uncertainty and fresh visual identity binding. Simply choosing
the nearer of two clusters, accepting the QR solely because it is visible,
or forcing full circular adjacency would discard evidence the current gate
is intended to preserve. The implementation below uses this recording to
validate a scoped confirmation based on a current visual outline.

The investigation itself changed no production code or robot state. The
subsequently requested local implementation is described below.

## Implemented correction

The retained-opposite path now has a separate current visual confirmation for
this bounded endpoint case. A search hint first checks the authenticated
retained metric center, all current scan rays, the existing endpoint metadata,
gap and diameter limits, and competing candidates. That hint supplies neither
target uniqueness nor identity. Confirmation then requires one complete QR
outline in the current image. Its finite-distance projection must cover both
gap endpoints throughout the accepted range interval. Every selected return
must fit the stand radius plus retained-center uncertainty. The ordinary
single-scan retained-target reconciliation then consumes this proof.

The original **two raw clusters** and disabled circular adjacency remain in
the receipt. The temporal three-witness policy is unchanged. There is no
global endpoint merge, invented beam, new head-angle sample, relaxed scan
range, or enlarged freshness limit. Image-time and scan-time transforms are
compared using the existing stopped-pose bounds; their distinct timestamps
are preserved. The proof binds source orientation, candidate snapshot,
measured stand model, camera intrinsics and extrinsics, and current tuple.

Replay exposed an additional payload issue after target admission: the local
native decoder reads the original pixels but misses the lens-rectified
version of this frame. The new endpoint branch therefore decodes a small
original-image crop inverse-mapped from the confirmed current outline using
that same image's CameraInfo. The crop uses the existing four-percent quiet
margin. Exactly one decoded symbol must supply its **own corners**, which
must map back to the confirmed rectified quadrilateral. The recorded frame's
maximum discrepancy is **0.718 pixels**. The shared decoder ceiling remains
**0.12 seconds**, further limited by remaining source freshness.

Both the live crop and persisted QR receipt require this exact isolated
outline. A receipt cannot replace it with a broad rectangular crop or a
different symbol. Original/rectified pixel mappings, decoded corners, source
stamp, calibration, candidate proof and identity are replayed when the
artifact is loaded. The retained angle and its full uncertainty are preserved
through the retained-facing and projected-facing loaders.

The raw-image binding carries the full sealed camera calibration. Its
recomputed hash must match the retained target and QR receipt, and the current
K/D/R/P matrices, dimensions and optical frame must match that profile. The
original workstation profile was copied separately for verification; its
hash matches the run's `ef9251020036e0ab201ae6dc599594dfc382bd83a995bafdee6b194123b1b2c7`.
Tests reject altered calibration even when all dependent pixel coordinates
and the outer receipt hash are recomputed consistently.

Repeated proof replay initially consumed the freshness budget. Reuse is now
limited to one synchronous callback: each hit still checks the complete proof
hash and the bytes of every external source, and returns defensive copies.
Snapshot validation uses the same bounded scope. Changed sources fail closed;
nothing survives into the next callback, and fresh-clock publication checks
remain active.

The saved advancing-clock replay uses original frame 26, its original sensor
timestamps and starting age, and actual desktop processing elapsed time. It
decodes **Start**, admits one current QR observation with **zero new axis
samples**, publishes the QR receipt, and reloads the retained-facing geometry.
The callback took **0.13854 seconds**; publication image age was **0.45775
seconds**, inside the unchanged **0.5-second** limit. This is offline desktop
evidence, not a workstation timing guarantee or a completed hardware run.

Validation across 23 correction and existing observer test modules completed
with **334 tests and 158 subtests passed**. One existing test was skipped
because it requires the deployed WeChat decoder backend. Changed Python files
also parsed successfully and `git diff --check` passed.

The implementation is local. It has not been deployed or exercised with robot
motion. The independent localization-drift stop and return-to-Start behavior
are unchanged.

## Reproducible evidence

Local audit root:
`results/implementation_checks/run_audit_20261001T122801Z/`.

- `source_integrity.json`, `source_integrity_recheck.json`: independent
  workstation inventories, revision and all original file hashes.
- `source/`: the unchanged run and seven matching real-run bundles.
- `camera_audit.py`, `camera_audit.json`: every camera outcome, source
  freshness, retained angle, scan envelope, decoder diagnostics and image hashes.
- `timeline_audit.py`, `timeline_audit.json`: route/controller/observer/parent
  chronology, 54 hashed inputs, live admission state and stop thresholds.
- `target_scan_audit.py`, `target_scan_audit.json`: production numeric replay
  of all 25 selected scans, fragment geometry, projection and proof limits.
- `endpoint_correction_replay.py`, `endpoint_correction_replay/summary.json`:
  advancing-clock native-pixel replay and persisted corrected receipt.
- `sealed_camera_calibration.json`: original workstation calibration, matched
  to the run's recorded profile hash and included in the regression fixture.
- `endpoint_correction_pytest.txt`: correction and existing observer regression
  test results.
- `camera_start_before.jpg`, `camera_start_after.jpg`: byte-identical
  representative source images.

Relevant implementation anchors at executed revision:

- `scripts/aufgabe04/perception/scan_topology.py:69`: circular adjacency admission.
- `scripts/aufgabe04/real_robot/observer/target_reconciliation.py:100`: unique current cluster requirement.
- `scripts/aufgabe04/real_robot/observer/node.py:1742`: retained-angle opposite identity dispatch.
- `scripts/aufgabe04/real_robot/observer/opposite_identity.py:43`: reconciliation prerequisite for QR search.
- `scripts/aufgabe04/real_robot/observer/scan_target_persistence.py:298`: independent historical witnesses.
- `scripts/aufgabe04/real_robot/observer/opposite_identity_opportunity.py:29`: bounded association-conflict exit.
