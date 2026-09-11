# Run audit: 20260911T124500Z

The latest run, `stand_explore_exact2_camera_all5_20260911T124500Z`, failed before camera inspection. The first candidate's fresh odometry pose, expressed through the newly captured map transform, was **30.458 mm from a route certified with a 30 mm tracking radius**. The startup gate correctly refused motion. Its generic `preflight_failed` result then fell outside every applicable recovery policy, terminating the mission without trying a replacement route.

This run does not demonstrate a camera, QR, or 3D head-model failure. The new camera implementation was present but never reached.

## Run identity and outcome

| Item | Recorded value |
| --- | --- |
| Run | `stand_explore_exact2_camera_all5_20260911T124500Z` |
| Time | 11 September 2026, 12:45:00–12:49:20 UTC / 14:45:00–14:49:20 CEST |
| Revision | Clean `0620d78733fad5c219e90d10bd72179bacec3e52` |
| Retrieval | SSH alias `mii001`; the run's manifest records hostname `mii0002` |
| Mode | `execute-exact-two-camera`, five expected QR identities |
| Survey | Both legs completed; recorded distance estimates 0.908781 + 1.186691 = **2.095472 m** |
| Coverage | **95.3100% modelled coverage**; not proof of visibility of every physical stand |
| Candidate population | Five retained: two pending camera, three provisional |
| First selection | `survey_candidate_0003`, child suffix `_candidate_000` |
| Candidate route | Two vertices, **0.086076 m**, certified tube radius **0.030 m** |
| Candidate outcome | `preflight_failed`; zero recorded motion, duration and distance |
| Camera result | No candidate camera observation started; **0/5 confirmed QR identities** |
| Mission exit | `failed_closed`, command exit code 2 |

Sources: [bundle manifest](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/real_runs/stand_explore_exact2_camera_all5_20260911T124500Z/manifest.txt), [revision](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/real_runs/stand_explore_exact2_camera_all5_20260911T124500Z/git_rev.txt), [segment results](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T124500Z/station_segment_runs.csv), [survey summary](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T124500Z/coverage/survey_summary.json), [mission failure](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T124500Z/mission_failure.json).

## Failure sequence

1. The bounded startup localization rotation completed. Both coverage legs subsequently completed, followed by their stopped LiDAR observations.
2. Candidate selection captured localization at **12:48:52.530 UTC**. The planned route began at map position `(0.389790772, -0.036268282)` and ended at `(0.355, -0.115)`.
3. The candidate dry run captured fresh localization at **12:49:05.831**. Its odometry startup check passed at **23.449 mm** from the route. The route's original start was retained.
4. The parent issued the exact routine-leg permit at **12:49:07.374**. The child bundle collected diagnostics before starting its execution command at **12:49:13**.
5. Execution preflight captured another fresh localization result at **12:49:18.531** and reported all 15 observations passing. Its separate map-frame startup check also passed.
6. Odom execution admission recomputed the route using the fresh direct `map <- odom` transform. The odom pose was **30.457545 mm** from the first segment, exceeding the 30 mm limit by **0.457545 mm**. Admission failed at **12:49:18.568**, before follower startup or motion-permit consumption.
7. The parent rejected startup recovery with `outcome_not_stopped`. `startup_reseal_index=0` and `maximum_startup_reseal_count=3` show that none of the configured reseal attempts was used.

The candidate permit was issued but never claimed. Only the two coverage-leg consumption receipts exist. The code returns from failed odom admission before permit validation/claim and before starting the follower. Thus this failure occurred earlier than the previous missing/stale-TF failure in a follower's separate listener.

Sources: [candidate events](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T124500Z/run_events/stand_explore_exact2_camera_all5_20260911T124500Z_candidate_000.jsonl), [adaptive events](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T124500Z/adaptive_replans.jsonl).

## Why a passed preflight still failed admission

The preflight independently requests the latest chained `map <- base`, latest `odom <- base`, and latest direct `map <- odom`. These do not necessarily represent the same localization update. The executable route's startup check uses the odom pose and direct transform; the earlier map check uses the chained map pose.

| Representation checked against the same route | Dry run | Execution |
| --- | ---: | ---: |
| Chained `map <- base` pose | 24.827 mm — pass | 28.439 mm — pass |
| Odom pose against the transformed route | 23.449 mm — pass | **30.458 mm — reject** |
| Difference between chained and composed map poses | 3.224 mm | 2.092 mm |

At execution, the map/base and odom/base transform stamp is `1789130958.4726343`; the direct map/odom stamp is `1789130959.458113`. The stamps differ by **0.985479 seconds**, although the lookups were captured less than a millisecond apart. The direct transform's future stamp is within the explicitly configured 1.1-second AMCL allowance; receipt freshness passed. **There is no missing or stale TF failure in this run.**

The execute odom pose composed through the direct transform is `(0.420165096, -0.038518273)`, matching the final stationary AMCL sample's position. Its nearest point on the certified segment is the original route start. The production transform and segment calculations reproduce this rejection; neither transform inversion nor segment indexing is wrong.

Across candidate selection, dry run and execution, odometry x/y are identical: `(1.962919626, 0.564411377)`. Yaw changes by approximately 0.125 degrees. The direct map/odom translation changes by 27.962 mm between selection and execution, and its composed base position changes by 26.089 mm. Each stationary window can pass stability while the estimate shifts between windows. The route anchor is about **26 seconds old** by execution admission.

The evidence supports a localization correction invalidating the earlier route start, with a smaller discrepancy between the two TF representations explaining why only the odom check rejected it. Recorded odometry is not independent physical ground truth; the logs do not prove absolute robot position or identify the underlying reason for AMCL's estimate changes.

Code: [independent TF captures](../../scripts/aufgabe04/navigation/localization/ros_preflight.py#L999), [latest lookup](../../scripts/aufgabe04/navigation/localization/ros_preflight.py#L1736), [odom composition and startup admission](../../scripts/aufgabe04/navigation/station_segment/localization_admission.py#L345), [certified startup selection](../../scripts/aufgabe04/navigation/waypoint_follower/startup.py#L74). Exact numerical results and input provenance: [geometry replay](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/startup_admission_replay.json).

## Why recovery did not run

The two analogous startup failures have different representations in code. The map route check returns a structured `stopped` outcome with route-check evidence. The odom route check throws a generic `ValueError`, which is converted into `preflight_failed` with only a general `odom_execution_admission_failed` fault and message. The specific pose, distances and transformed segment are lost from the failure record.

Replaying the recorded failure through the current pure policies gives:

| Recovery policy | Decision |
| --- | --- |
| Startup reseal | Ineligible: `outcome_not_stopped` |
| Localization-readiness retry | Ineligible: `failure_not_route_uncertainty_exhaustion` |
| Candidate route-admission deferral | Ineligible: `motion_permit_was_issued` |

`outcome_not_stopped` describes the outcome enum; it does **not** mean that this child was physically moving. The alternate readiness/deferral paths are narrowly designed for uncertainty-budget exhaustion, not this startup corridor mismatch. Issued-but-unclaimed authorization also prevents treating the failure as an ordinary no-permit deferral.

Code: [map startup rejection](../../scripts/aufgabe04/navigation/station_segment/route_bundle.py#L118), [generic odom rejection](../../scripts/aufgabe04/navigation/station_segment/localization_admission.py#L407), [failure serialization](../../scripts/aufgabe04/navigation/station_segment/reporting.py#L18), [admission exception handling](../../scripts/aufgabe04/navigation/station_segment/runtime.py#L1016). Policy reproduction: [recovery replay](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/recovery_policy_replay.json).

## Recommended correction and validation

1. **Preserve a typed odom startup-corridor rejection.** Include the exact pose, transformed segment, radius, measured distances, transform timestamps and explicit no-motion evidence. Keep malformed transforms, ownership/identity failures and other admission exceptions distinct.
2. **Add a bounded stopped re-admission path for that exact failure.** Reacquire coherent localization, reproject the same candidate, rebuild the exact route start, and repeat collision, uncertainty and certificate checks. Account for the issued-but-unclaimed old permit and bind any replacement authorization to the new artifacts. Do not merely relabel every `preflight_failed` result as recoverable.
3. **Use one explicit localization basis for planning and execution readiness.** Retain the source timestamps in the receipt and report sensor health separately from route admission. A sensor preflight PASS should not imply that the latest execution pose fits the previously sealed route.
4. **Validate this exact recorded transition offline**, including bounded exhaustion, no motion before admission, changed route/permit identities, and refusal of unrelated failures. Then repeat one candidate approach into a stopped camera observation on hardware.

Keep the 30 mm tracking bound. Increasing it or adding a blanket epsilon would suppress this particular rejection while leaving the planning-to-execution drift and recovery gap unresolved. The next hardware milestone remains a successfully admitted candidate approach followed by a fresh QR identity and validated 3D head/facing observation.

The camera timing precheck did receive a synchronized header tuple, but no candidate image processing occurred. This run supplies no new evidence about QR binding, crop acquisition, backside classification, large-angle head fitting or seven-frame camera consensus.

## Audit integrity and repeatability

The audit copied the complete session and its parent, candidate and two coverage bundles. **All 215 original files match independently computed SHA-256 hashes from the remote originals.** Derived replay scripts/results are separate from those files. The saved revision and clean status establish that the latest camera changes were included in this run.

- [Local source manifest](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/source_manifest.json)
- [Remote source manifest](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/remote_source_manifest.json)
- [ROS-free geometry replay script](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/replay_startup_admission.py)
- [ROS-free recovery-policy replay script](../../results/audits/stand_explore_exact2_camera_all5_20260911T124500Z/replay_recovery_policy.py)

This audit changed no runtime source, ran no ROS nodes and commanded no robot motion. Its numerical reconstruction establishes why the recorded software rejected admission; it does not certify that a replacement route would have passed every downstream live gate.
