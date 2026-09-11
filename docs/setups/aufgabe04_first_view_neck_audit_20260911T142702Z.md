# First-view camera audit: 2026-09-11 14:27:02 UTC run

Follow-up: the correction is now implemented; see [implementation and production replay](aufgabe04_first_view_camera_correction_20260911.md). The audit below records the pre-correction investigation.

The first candidate **was classified `front_readable` and decoded as `QR_003` at the first inspection point**. It did not receive an admitted angle. The principal defect is the head/neck junction validator's fixed-column edge test. A separate successful-fit registration asymmetry then rejects the rare geometry result that passes that test. Reducing the approach distance alone cannot repair either decision.

## Evidence and scope

- Run: `stand_explore_exact2_camera_all5_20260911T142702Z`.
- Commit: `75d625f2c97ceb3258a215de2110f5853d007a5e`; remote checkout clean when inspected.
- Accessed through SSH alias `mii001`; bundle records hostname `mii0002`.
- Snapshot: `2026-09-11T14:37:08.704645+00:00`; all **587 copied original files** match their remote SHA-256 hashes, with no files changing during their reads. The directory snapshot is not an atomic snapshot of a running process.
- The parent and third-position child wrapper manifests have no recorded end time or exit code. There are two completed camera observations, one completed extra local move, and a second extra move whose wrapper started. There is no third camera observation or terminal mission-failure artifact in this snapshot. This audit explains the repeated camera-view selection, not an unrecorded terminal failure.
- No ROS nodes, robot motion, or deployed code changes were used for this audit.

Original evidence and runnable diagnostic replays are under [the local audit directory](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z). [Source manifest](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/remote_source_manifest.json).

## What the closer approach changed

The recorded command explicitly used `--candidate-approach-offset-m 0.50`. The first observation's recorded robot-base-to-candidate-center map distance was **0.5323 m**, compared with **0.7234 m** in the preceding `135346Z` run. These are map-derived distances, not independent range metrology.

The median smaller vertical head-side span grew from **72.23 px to 98.61 px**, about **36.5% more pixels**. The full head and neck fit inside the first-view nominal ROI. Reacquisition also found and recentered the correct head. Further reducing standoff is not the next corrective action.

![First inspection point: complete head and QR visible](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T142702Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/perception_debug/latest_frame.png)

## Recorded decisions

| Evidence | First camera point | Second camera point |
| --- | ---: | ---: |
| Fresh processed results | 151 | 124 |
| Rejected with `head_neck_junction_gap_too_large` | 150 | 123 |
| Geometrically usable measured-head fits | 1 | 1 |
| Fits admitted to angle consensus | 0 | 0 |
| QR-sample frames | 130 | 21 |
| Final classification and identity | `front_readable`, `QR_003` | `front_readable`, `QR_003` |
| Final angle | null | null |
| Stopped front-view recovery budget | 30 s, exhausted | 30 s, exhausted |

Both first-view classification and identity therefore worked. The unresolved quantity was the admitted head-plane angle. First-view processing freshness was not the dominant rejection in this run: all 151 processed results were fresh, and 150 frames were associated for the separate advisory/QR path.

At the first point, the junction diagnostics were **3 px gap in 148 results**, 7 px in two, and 2 px in one; the allowed gap was 2 px throughout. At the second point, 123 results reported 3 px and one reported 2 px. Both usable fits were rejected with `camera_bearing_outside_map_cone`.

Counts come from distinct final processed-result states in `observer_events.jsonl`, not from counting repeated TF-wait status messages. [Decision summary and replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/audit_camera_decisions.py).

## Root cause 1: the validator measures the start of straight rails, not the head junction

[`head_model_neck.py`](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_model_neck.py:91) searches for two fixed image columns containing simultaneous uninterrupted raw edge runs. In the latest saved detector crop it requires 13 rows at columns **54 and 71**, beginning at row **124**. Relative to the fitted bottom-edge midpoint, this produces a three-row gap and rejects the fit.

The original raw edge image contains connected junction pixels leading into those rails:

- Left, traced upward: `(54,124) → (54,123) → (53,122) → (52,121)`.
- Right, traced upward: `(71,124) → (72,123) → (72,122) → (73,121)`.

The neck edges taper sideways at the junction. They exist before they settle into the fixed columns. Each path reaches the first row below the fitted bottom edge evaluated at that path's own x coordinate. The visible head is correctly selected; this rejection does not demonstrate a physically detached neck or an inner-paper rectangle.

[`fit_current_measured_head`](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_model_fit.py:64) computes a current measured-head IPPE pose, then makes the junction rejection decisive. For the exact saved latest raw edges, the head-only pose has **0.578 px reprojection RMSE**, but the neck check rejects it before the rest of uncertainty admission can succeed. QR/head joint-fit disagreement is recorded separately and is not the angle authority in this path.

## Root cause 2: a successful nominal head fit skips the registration used by reacquired fits

[`select_camera_target_measurement`](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/camera_target_registration.py:225) returns immediately for a usable primary fit. [`node.py`](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/node.py:1789) selects the bounded camera-centered LiDAR search only when the selected ROI carries registered provenance. A good nominal fit stays on the map-centered cone.

The sole usable first-view result, captured as `frame_000012.json`, estimates **26.338°**. Its head ray differs from the original map ray by **3.507°**, outside the **3°** map cone. Its center is only **0.339 expected head heights** away from the projected center, within the existing bounded registration limit of 1.5. The existing registered association helper accepts a unique cluster with the original range/freshness limits and unchanged 3° cone centered on the head ray.

Replaying the recorded refined head rays for all 62 saved detector crops gives **52 unique-cluster accepts and 10 ambiguous-cluster rejects** under the bounded camera-centered policy. All 62 rays fail the legacy map cone. This is an association-only counterfactual; it does not on its own certify the rejected head geometry or create registration provenance.

## Why another local inspection is selected

[`execute_candidate_inspection`](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/candidate/inspection_execution.py:79) resolves a candidate when it receives the validated recommendation artifact. A `front_readable` advisory with null angle does not satisfy that contract. After the bounded stationary recovery expires, [`candidate_view_options`](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/candidate/inspection_policy.py:28) proposes a diverse view, beginning with +45° for this classification. The second view suffers the same rejection, so another view is selected.

There is no fixed three-view requirement. Reducing `--max-candidate-inspection-views` or accepting a QR read as a complete pose would hide the missing angle evidence.

## Validated direction for the correction

1. **Keep the current measured 3D head model and correct the raw neck-connectivity measurement.** Extract a small ROS-free connectivity helper, invoked from `head_model_neck.py`, that follows the paired current raw edges back from their supported rail runs to their own intersections with the fitted bottom edge. The audit prototype uses at most one pixel of horizontal movement per row within a two-pixel lateral corridor around each supported rail. Every traversed row must contain an actual edge pixel. Preserve the existing two-pixel vertical-gap cap and physical paper-inset resolution check.
2. **Separate current-head candidate association from how its crop was acquired.** Add a small observer helper that accepts a fully quality-admitted current measured head, checks displacement against the original projection, then uses the existing bounded camera-centered narrow cone, original surface-range interval, scan freshness, and unique-cluster requirement. Record explicit current-fit association evidence. Keep actual nominal/reacquired ROI provenance; do not invent a strict-retry receipt or require another expensive fit merely to obtain one.
3. **Retain seven fresh stationary angle samples and independent current QR binding.** Once these produce the validated recommendation, the existing controller already exits the local-view search. Keep the 0.50 m preferred approach and existing physical-clearance/route gates.

The purple dashed overlay depicts projected model geometry. The angle authority remains the admitted current-pixel head fit; drawing a convincing overlay does not bypass the junction, association, uncertainty, or consensus checks.

## Offline validation and remaining boundary

[Neck replay prototype](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/neck_pixel_replay.py) reconstructs the saved detector crops from the original compressed images and CameraInfo, with rectification and crop-adjusted intrinsics, then uses their recorded current-image proposal corners. The baseline reproduced the recorded reasons in all **62/62** saved detector frames: 61 junction failures and one usable fit.

Changing only junction connectivity produced **62/62 usable head fits**, with estimated yaw **26.152–26.433°**, mean **26.292°**. The exact saved latest raw-edge image also passes, with estimated yaw **26.218°** and local-model yaw standard deviation **1.462°**. Tight frame-to-frame agreement is not a measurement of absolute physical angle accuracy.

The prototype retained rejection of a centered 71/78 inner-panel rectangle on the same real raw pixels, and blank, single-rail, disconnected-row, and genuinely detached-rail fixtures. All **five existing head-neck tests pass** under the prototype, including multi-angle/distance paper-only negatives and prior outer-border recovery. Rejected connectivity probes preserve the original junction diagnostics so they do not inadvertently alter the existing outward-border recovery search. It does not manufacture missing pixels or relax the physical gap limit.

A [combined evidence-window counterfactual](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/first_view_consensus_counterfactual.py) uses those new fit corners, bounded current-head association, recorded QR binding and sensor timestamps, and the production `PassiveObserverEvidence` accumulator. It reaches **seven axis samples plus current `QR_003` front identity at saved frame 10**, spanning **1.396 s of saved image timestamps**, while retaining ten ambiguous-cluster rejections across the subset. This establishes that the saved first view contains enough evidence under the proposed corrections. It is not a measured runtime completion time: acquisition/decoder scheduling changes, fresh processing time and recommendation publication were not replayed.

![Raw-pixel junction diagnosis](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/neck_pixel_junction.png)

This is a diagnostic prototype, not a production patch. Before integration, explicitly validate bottom-edge geometry and image bounds, retain distinct paired paths and rail separation, make traced-path versus straight-run diagnostics unambiguous, and run adversarial inner-panel tests across distance and angle. Test nominal/reacquired association equivalence, ambiguous clusters, stale scans, wrong range/identity, excessive displacement, and successful first-view termination.

Local OpenCV is 4.13.0 versus deployed 4.5.4; the baseline decisions match on this saved subset, but the replay does not exercise changed full-node acquisition/decoder scheduling, runtime processing latency, recommendation publication, or hardware motion. No production detector code was changed during this audit.

## Should the neck check be removed?

The angle already comes exclusively from the four current outer-head corners and measured head geometry. The neck supplies boundary-identity evidence; it is not an additional angle landmark. Removing it without replacement allows an inner paper rectangle to supply a plausible planar fit with the wrong physical boundary. The evidence supports correcting its connectivity measurement while retaining head-only angle computation. A future neck-optional path would need independently validated outer-border identity and its own negative fixtures; QR visibility alone is insufficient.
