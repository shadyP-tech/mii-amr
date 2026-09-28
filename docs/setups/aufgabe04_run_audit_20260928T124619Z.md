# Run audit: 20260928T124619Z

Read-only audit on 2026-09-28 of `stand_explore_exact2_camera_all5_20260928T124619Z` on `mii001`. The run bundle records clean revision `313ae38234d781632f67210466a46b7763dfcab5`, matching the local checkout and containing the preceding opposite-route recovery correction.

## Conclusion

The robot stopped at the same second-candidate inspection location, but at a different execution stage. **The corrected opposite-route recovery succeeded in obtaining an admissible route.** The new execution follower then stopped before moving because it received no `/odom` message within its two-second initial sensor window.

Only `survey_candidate_0003` / `QR_003` was admitted, with facing geometry. `survey_candidate_0001` retained a backside observation but remained QR-unresolved. Three other candidates were unvisited. No opposite-side arrival, centering, or front-side QR observation occurred.

## The previous correction ran successfully

| Planning attempt | Standoff | Result / minimum margin |
| --- | ---: | --- |
| Initial epoch | 0.50 m | Rejected: −0.091105 m |
| Initial epoch | 0.45 m | Rejected: −0.003543 m |
| Initial epoch | 0.400839 m | Static goal blocked |
| Fresh stationary epoch | 0.50 m | Rejected: −0.004594 m |
| Fresh stationary epoch | 0.45 m, dry | **Accepted: +0.052903 m** |
| Fresh stationary epoch | 0.45 m, execute | **Accepted: +0.059551 m** |

The proposal floor included the current validated center uncertainty; no invalid 0.35 m request was generated. Exactly one `opposite_localization_refresh_requested` event appears. The stationary observations improved sufficiently for the unchanged uncertainty budget to pass. Both general preflight and stationary map←odom stability passed.

## Final child timeline (UTC)

Child suffix: `candidate_001_inspection_001_localization_001_opposite_standoff_001`.

- **12:53:15.136:** dry route admitted with +52.9 mm minimum margin.
- **12:53:27.588:** execute preflight passed all 16 observations. `/odom` header age was **12 ms**, receipt age **2 ms**; scan header age **85 ms**; map→odom and odom→base checks passed.
- **12:53:29.047:** execute odom route certificate sealed with +59.6 mm minimum margin.
- **12:53:29.052:** mission-leg permit consumed; `motion_started` logged with explicit semantics `child_execution_attempt_started_before_follower`, **motion_published=false**.
- **12:53:29.068:** execution TF buffer created.
- **12:53:31.333:** follower stopped: **`missing odom`**, before motion, distance 0.

The single controller trace row is `initial_runtime_input_stop`, with zero linear/angular command and no odom or map pose. The event named `motion_started` is not evidence that the robot moved.

## Confirmed failure mechanism

The final controller evidence reports:

- `source=message_freshness`, `sensor=odom`, `has_message=false`.
- No odometry receipt/header age exists: this is absence of a first message in this follower, not an old odometry sample.
- Elapsed initial acquisition **2.024594 s**; initial sensor allowance **2.0 s**.
- A separate **3.0 s** cold-TF allowance is configured, but `extension_used=false`, `denial_reason=sensor_inputs_not_fresh`.
- TF executor alive and ready, **40 heartbeats**, latest heartbeat age **27.6 ms**.
- Execution TF receipt tracer: **zero** `odom←base_footprint` deliveries and **zero** `map←odom` deliveries into this new buffer; no ingestion exceptions.
- `edges={}`: actual required-edge lookup/admission was not attempted because sensor admission failed first.

`waypoint_follower/runtime.py::run_waypoint_follower` constructs new sensor subscriptions and a dedicated TF listener after preflight/route sealing. The persistent preflight listener being ready does not make the new follower's subscriptions ready.

`runtime_components/initial_runtime_inputs.py::_sensor_failure` checks scan before odom. Returning `missing odom` therefore implies that scan freshness passed at that evaluation. The logs do not contain a full sensor/matched-publisher snapshot at the stop.

`initial_tf_acquisition.py::InitialTfAcquisition.can_continue` refuses the additional TF acquisition window unless scan and odom are already fresh. Thus the nominal five-second maximum was not available for this missing-first-odom case: the run stopped at the two-second sensor boundary. A ROS-free replay using the recorded failure fields reproduces `False / sensor_inputs_not_fresh`.

## Why recovery did not continue

The earlier opposite-route retry policy intentionally admits only dry uncertainty rejections with no motion permit. This final child had already obtained and consumed its one-use routine permit and failed during execution startup. It could not safely be treated as another dry route proposal.

The existing startup reseal classifier expects a narrowly defined initial global-TF lookup failure with fresh sensors. This evidence has `source=message_freshness`, no odometry, and no sampled TF edge, so it returns **`invalid_initial_map_tf_stop`**. That wording is a recovery-classification rejection; it does not establish that a bad map→odom transform caused the stop. The original `missing odom` reason is preserved in `mission_failure.json`.

## What remains uncertain

The run proves a **new execution-subscriber readiness failure**, not that the robot's odometry publisher stopped. The bundle's separate `/odom` echo succeeded before execution, and execute preflight received very fresh odometry immediately beforehand.

Delayed DDS discovery/first delivery is a plausible explanation. The bundle's separate five-second `view_frames` probe also retained only about 1.5 seconds of odom→base history, which is compatible with delayed startup delivery, but it does not record receive times and is not proof. A transient transport problem or subscription QoS compatibility cannot be distinguished from this recording. No incompatible-QoS warning appears in the copied terminal logs. An executor heartbeat proves callback service, not topic delivery.

The camera had already committed seven backside samples with angle **1.513408 rad ±0.154742 rad**. Its final image/scan ages were approximately **150/103 ms**. This stop is not a camera admission or front-side centering failure.

## Next correction

1. Make **the actual execution follower's** stopped sensor/TF readiness part of the execution handoff. Warm its subscribers before consuming the final motion permit where the architecture allows, then revalidate freshness, certificate continuity, and ownership immediately before nonzero motion. A separate preflight subscriber is insufficient evidence for this boundary.
2. Add an explicit bounded **first-message acquisition** policy for missing initial sensor data. For example, allow one fixed startup window up to five seconds while publishing zero, with healthy executors and diagnostics; admit only after fresh scan, odom, and both TF edges pass. Distinguish never-received data from an established stream becoming stale or disappearing. Do not let repeated misses renew the deadline or relax runtime freshness limits.
3. Persist first-receipt times/counts, matched publishers and QoS, and both follower/TF executor health at this boundary. This distinguishes discovery, compatibility, and callback-service failures in the next recording.
4. If recovery after a terminal startup failure is added, use a dedicated no-motion sensor-startup outcome and explicit permit retirement/new authorization. Do not broaden the existing TF reseal classifier or reuse the consumed permit.

The route geometry and uncertainty limits should remain unchanged. Increasing only the TF extension cannot fix the currently observed path because missing odometry prevents entry into that extension.

## Evidence locations

Copied artifacts are under `results/implementation_checks/run_audit_20260928T124619Z/`:

- `audit_summary.json`: complete final failure, opposite-route budgets, and deterministic startup-policy replay result.
- Nested autonomous run: `mission_failure.json`, `candidate_goal_progress.json`, `station_segment_runs.csv`, `candidate_selection.jsonl`, final child run events, and dry/execute uncertainty certificates.
- Nested top-level bundle: recorded command, revision, clean status, terminal output.
- Nested final-child bundle: `controller_trace.jsonl`, `terminal_run.log`, `odom_once.txt`, `tf_frames.txt`, command and ROS topic/node snapshots.

No production code changes, ROS nodes, or robot motion were performed for this audit.
