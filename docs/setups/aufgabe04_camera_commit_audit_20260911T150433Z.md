# First-view camera result commit failure — 2026-09-11

The latest retrieved run, `stand_explore_exact2_camera_all5_20260911T150433Z`, successfully observed the first candidate at its first inspection point. It decoded `QR_003`, accumulated seven accepted measured-head angle samples and published a camera recommendation. The mission then stopped because the decision recorder rejected the new `pose_provenance` field inside `planning_frame_admission`.

This is a reproducible producer/consumer schema regression in the planning-frame handoff. Further changes to camera distance, angle thresholds or inspection budgets would not correct this failure.

## Run and evidence

- Run commit: `301805e63e953ad869fa821309f6a12802013bac`; local and remote HEAD matched at retrieval.
- Parent execution: **15:04:33–15:10:17 UTC / 17:04:33–17:10:17 CEST**, exit code **2**.
- Read-only retrieval through `mii001`; bundle hostname records `mii0002`.
- Snapshot: `2026-09-11T15:12:18.734825+00:00`. All **312 original files** verified against the retrieval manifest, with no changed-while-reading files.
- First candidate: `survey_candidate_0003`, stored under `candidates/000_survey_candidate_0003`.
- Two survey legs completed, with approximately **2.024 m** summed recorded travel estimates and **95.31% modelled coverage**. The fused registry retained **six hypotheses**: two pending camera validation and four provisional. Five distinct QR identities remain the goal; the six hypotheses do not establish six physical stands.
- Final persistent goal state: **zero confirmed stations**, despite one successfully observed QR identity. Candidate 0003 remains `inspection_started` in goal progress and `pending_camera` in the survey registry.

The [source manifest](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/source_manifest.json), [computed audit evidence](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/audit_run.json) and [terminal log](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/results/real_runs/stand_explore_exact2_camera_all5_20260911T150433Z/terminal_run.log) preserve the supporting records.

## Failure sequence

| UTC | Recorded event | Interpretation |
| --- | --- | --- |
| 15:07:11.899 | Survey leg 000 completed | Coverage motion succeeded. |
| 15:08:30.077 | Survey leg 001 completed | Exact-two coverage handoff became available. |
| 15:09:29.394 | Initial candidate approach stopped for localization consistency | Recoverable interruption; see below. |
| 15:10:05.722 | Runtime-localization reseal child completed | Candidate approach recovered and reached the observation point. |
| 15:10:12.253 | Camera observer first status | Waiting for synchronized sensor data. |
| 15:10:13.584 | First accepted head-angle sample | Current measured-head geometry and candidate association accepted. |
| 15:10:15.346 | `recommendation_committed` | Seven axis samples, QR consensus and a facing-pose recommendation available. |
| After recommendation, before 15:10:17 termination | `planning_frame_admission fields mismatch` | Persistent candidate-decision validation failed. |
| 15:10:17 | Parent bundle completed with exit 2 | Mission failed closed. |

The terminal contains both `error: planning_frame_admission fields mismatch` and the adapter's `error: failed to commit camera candidate decision`. `mission_failure.json` records the latter and `motion_continues_authorized=false`. Buffered recovery text appears later in terminal output; the timestamped child events establish that recovery happened before camera observation.

## Camera performance at the first point

The saved full image shows the complete front-facing head and QR. Production diagnostics confirm angle source **`model_current_measured_head`**, with pose model **`measured_head_only`**. The 3D measured outer-head fit supplies the angle; the rendered overlay is its visualization. QR evidence supplies identity and front-side classification.

| Measurement | Recorded result |
| --- | --- |
| Local inspection views | **1** |
| Additional local route proposals | **0** |
| Inspection termination | `joint_observation_ready` |
| Processed / fresh / verified head results | **10 / 10 / 10** |
| Candidate-associated / accepted axis samples | **7 / 7** |
| QR sample frames | **5**, latched identity `QR_003` |
| Axis confidence | **0.98747** |
| Last camera-relative head yaw | **26.704°** |
| Last head reprojection RMSE | **0.655 px** |
| Last minimum head edge length | **100.315 px** |
| Last conditional yaw uncertainty estimate | **1.198°**, within the 3° gate |
| Detector duration | **62.81–93.99 ms** across the ten processed images |
| Result age at detector completion | **120.28–164.04 ms** |
| First observer status to recommendation | **3.094 s**, including sensor startup |
| First through last processed image timestamp | **1.767 s** |

All ten fits used the nominal crop; no wider registration retry was required. In the last frame the head center differed from the map projection by about **4.09°**. The new current-head association accepted it using the measured camera bearing and a unique LiDAR cluster while retaining the configured **3° search cone**, **12° map/camera displacement bound**, range and freshness checks. The QR was independently bound inside that head and to the same scan cluster.

The final neck-junction check passed with a one-pixel straight-core gap. **None of the ten frames needed raw-edge continuation**, so this run validates the combined first-view pipeline and the new association path, but does not exercise that particular neck recovery branch. The angle still uses head corners; neck connectivity remains an outer-head identity check.

The joint head/QR geometry diagnostic still reports `model_head_qr_geometry_mismatch` in the last frame, with a 2.400 px joint fit error. It did not prevent admission of the independently verified head-only angle. This is consistent with the chosen angle-source policy; it is not the terminal failure.

The camera subprocess exited **0** after publishing `recommendation.json`. A **0.35 m** facing-pose route also passed A* and continuous-clearance validation, with route length **0.291 m**. This was validation only; that final facing route was not executed before the commit failure.

Evidence: [observer status](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T150433Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/observer_status.json), [inspection progress](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T150433Z/candidates/000_survey_candidate_0003/inspection_progress.json), [recommendation](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T150433Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/recommendation.json), [saved full image](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T150433Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/perception_debug/latest_frame.png).

## Exact code defect

Commit `b1d1ddf` added optional `pose_provenance` to `CandidatePlanningFrame`. The live frame builder supplies direct TF/odom capture evidence, and [the serializer](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/approach/candidate_frame_projection.py:106) emits it. The outer projection schema remained version 1.

The camera-decision consumer still requires exactly four fields:

```text
current_pose, map_from_odom, map_frame, odom_frame
```

The saved arrival projection contains all four plus **`pose_provenance`**. [Its strict reader](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/approach/camera_decision_geometry_binding.py:362) rejects that fifth field with `planning_frame_admission fields mismatch`.

The v3 `candidate_decision.json` is a prepared confirmation request. Its projection reference and canonical JSON hash match the saved arrival projection (`a793e5c71a851bae0118a6da1c042af2ea911e1b923db144234f96f68728e515`). Parsing fails before [registry mutation and canonical receipt publication](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/approach/record_stand_candidate_decision.py:546). A camera-local `target_committed` state therefore does not mean the persistent mission decision committed.

The duplicated [backside projection reader](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/approach/backside_axis_frame_projection.py:736) rejects the same payload. That branch was not exercised on this run, but it would obstruct a later opposite-side handoff. A third independent reader in [the autonomous arrival catalog](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/stations/autonomous_arrival_catalog.py:85) drops the provenance rather than validating and retaining it.

### Offline reproduction

The [schema replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/schema_contract_replay.py) invokes both actual production parsers on the original saved nested payload. [Results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/schema_contract_replay.json):

| Consumer | Original saved payload | In-memory diagnostic copy omitting only provenance |
| --- | --- | --- |
| Camera decision | `planning_frame_admission fields mismatch` | Nested parser accepts |
| Backside projection | `planning_frame_admission has unexpected fields` | Nested parser accepts |

This isolates the offending field. Omitting evidence is not the recommended correction. The diagnostic does not modify or rehash originals, publish a receipt, establish complete decision acceptance, or authorize motion.

## Recovered interruption and remaining observations

The initial candidate approach stopped when `map ← odom` translation drift reached **0.13128 m**, exceeding its **0.12392 m** allowance. TF freshness validation passed. The controller requested zero and resealing; a freshly admitted runtime-localization recovery child then completed the approach. This interruption added delay but was not the terminal cause.

During camera observation, three processed frames were rejected as `ambiguous_registered_camera_clusters`, each with two eligible clusters. Their scan topology evidence disabled circular adjacency because endpoint metadata was inconsistent. The retained seven valid samples still reached consensus promptly. These records justify a separate scan-metadata/seam audit; they do not establish that the two reported clusters are safe to merge or justify relaxing uniqueness checks. Three exact-time TF waits also recovered within the observation.

This run supplies no physical ground-truth angle error, no >35° validation, no opposite-side execution, and no full five-station completion evidence. It supports a successful first-view camera observation followed by a deterministic persistence failure.

## Recommended modular correction and validation

1. Define one shared planning-frame evidence encoder/decoder contract and use it in camera-decision binding, backside projection and arrival-catalog loading. Explicitly support the four-field legacy form and the defined optional provenance extension; retain rejection of unknown fields.
2. Preserve and validate provenance structure, finite capture values, frame identities and consistency with the composed pose and transform. Historical capture evidence must not become fresh motion authorization merely because it parses.
3. Keep the existing artifact hashes, candidate population/reprojection checks, recommendation binding, route admission and registry checks intact.
4. Add a saved-preflight → real frame builder → serialization → actual decision-recorder integration regression that proves one registry confirmation and canonical receipt publication. Feed the same produced artifact through the backside and catalog consumers. Cover legacy artifacts, malformed/inconsistent provenance, unknown keys and tampered hashes.

Existing exact-two camera decision fixtures construct frames without provenance, while a runtime test stubs the final commit. That bypassed the failing production contract. The previous 379 passing camera tests are evidence for their exercised paths, not this recorder handoff.

Audit work used saved artifacts and ROS-free parsing only. No production correction, deployment or robot motion was performed.
