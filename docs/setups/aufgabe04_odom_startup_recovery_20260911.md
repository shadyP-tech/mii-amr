# Candidate odom startup recovery

This correction addresses the failure in `stand_explore_exact2_camera_all5_20260911T124500Z`: a dry admission accepted a pose 23.449 mm from the certified route, then execution admission rejected a fresh pose 30.458 mm away from the same 30 mm tube. The old generic `preflight_failed` result could not enter bounded startup recovery. See the [recorded audit](aufgabe04_run_audit_20260911T124500Z.md).

## Resulting behavior

Candidate planning now composes the admitted direct `map <- odom` transform with the captured odom/base pose. Candidate reprojection uses that same transform. The chained map/base pose remains diagnostic evidence, including its timestamps and difference from the composed pose. Missing or inconsistent odom evidence does not fall back to a different pose basis.

For odom-controlled candidate and opposite-face legs, the odom admission check owns startup containment. Map-controlled and coverage startup checks retain their existing behavior. Sensor preflight can pass while route admission rejects the pose; the rejection retains its specific geometry and before-motion phase.

Only the new typed `odom_startup_route_mismatch` enters this recovery path. Its proof includes the original map route, transformed odom route, captured poses and transforms, frame identities, timestamp/freshness limits, source hashes and recomputable tube distances. Generic admission exceptions, malformed transforms, stale evidence, wrong identities and any recorded motion remain ineligible. The outcome still says `preflight_failed`; it is not relabelled as a follower stop.

An eligible candidate failure follows this sequence:

1. Validate the exact child, routine, geometric rejection and terminal semantic event sequence.
2. Close its unused authority. An issued permit is retired atomically at its existing exclusive claim path. A failed dry attempt must instead prove that no permit was issued. Neither disposition claims that motion occurred.
3. With retry budget remaining, acquire fresh stationary localization, reproject the same candidate and replan from the newly admitted start.
4. Re-run route clearance, uncertainty and dry admission, then seal a distinct replacement permit against the new route and evidence.
5. Let the child revalidate the permit and claim it once before starting its follower. Camera observation still waits for a completed approach.

If the last replacement fails, its unused permit is also retired before reporting budget exhaustion. Startup/runtime handoffs preserve both cumulative counters and the actual permit class. A replacement may fail during either dry or execution admission without resetting those counters.

The corridor remains **30 mm**. Sensor freshness, collision clearance, uncertainty, motion ownership and one-use authorization gates remain enforced. The existing startup master scope/schema is preserved; this is a specific classification of the existing same-target, before-motion startup mismatch scope.

## Module boundaries

| Responsibility | Module |
| --- | --- |
| Shared pose composition and capture validation | `navigation/localization/candidate_planning_pose.py` |
| Typed geometry proof and pure eligibility decision | `navigation/localization/startup_route_admission.py` |
| Completed-child event and source-artifact binding | `navigation/execution/startup_route_rejection_evidence.py` |
| Atomic unused-permit disposition | `navigation/execution/startup_reseal_permit_retirement.py` |
| Candidate outcome-to-permit adapter | `real_robot/candidate/startup_permit_retirement.py` |
| Bounded coordination and existing phase handoffs | `real_robot/candidate/startup_recovery.py`, `runtime_recovery.py` |

The existing station-segment producer, planning-frame builder and startup permit validator use these modules. Recovery coordinators retain injected effects; importing a classifier does not start ROS or publish velocity.

Replacement summaries explicitly mark the composed planning-pose basis. The validator derives that pose from the saved fresh evidence for every candidate startup recovery source. Existing summaries without the marker retain their original validation contract. Recorded audit files are unchanged.

## Verification and next experiment

Regression coverage includes the actual dry/execution geometry, coherent recorded candidate selection, typed terminal events, relative-path dry launches, repeated failures, exhausted budgets, startup/runtime handoffs, and the full replacement-permit issue/validation/consumption sequence. It also checks that consumption and retirement cannot both win the old permit's claim, and that altered source evidence, foreign permits, wrong targets and arbitrary preflight failures cannot authorize replacement motion.

The recorded selection fixture moves the planning anchor by 4.409 mm to match the transform already used for candidate projection. This is a software consistency correction, not a claim of measured physical localization accuracy.

Final local verification is recorded in [startup correction validation](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/startup_correction_validation.json). The pre-existing observer-module inventory assertion also fails on clean `0620d78`; its expected module list omits existing camera modules. It is outside this correction.

No new experiment flags are required. The previous command requires the updated source files in the robot checkout. Hardware validation remains pending: first verify one admitted candidate approach followed by a stopped, fresh QR/head-model observation. Offline tests do not establish that all live localization, route and camera gates will pass on the next run.
