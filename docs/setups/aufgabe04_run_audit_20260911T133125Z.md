# Real-run audit: candidate planning pose binding

Run `stand_explore_exact2_camera_all5_20260911T133125Z` failed because the candidate planner and the uncertainty-evidence loader used different definitions of the robot's map pose. This is an integration regression in correction `b1d1ddf`, reproduced with the unchanged saved evidence. The failure occurred before choosing the first camera target or launching its approach.

The experiment ran **2026-09-11 13:31:25–13:35:20 UTC / 15:31:25–15:35:20 CEST**, on clean commit `b1d1ddf7a5b72a2789a4a814319fb1ccb0c00510`, and exited with code 2. The bundle records hostname `mii0002`; the audit retrieved it through SSH alias `mii001`. The command contained the intended all-five mode and the existing recovery limits. Changing those arguments does not repair this contract mismatch.

## Recorded sequence

| Time (UTC) | Recorded outcome |
| --- | --- |
| Before survey execution | Startup active localization completed, publishing motion for a recorded 6.222804 rad rotation over 54.682 s. |
| 13:33:58.855 | Coverage leg 000 completed; recorded translation 0.896530 m. |
| 13:35:01.673 | Coverage leg 001 completed; recorded translation 1.174846 m. |
| 13:35:11.661 | Exact-two camera handoff marked the five-candidate pool ready. |
| 13:35:19.146 | Candidate-selection localization passed all 15 observations; the planning-frame projection was saved. |
| By 13:35:20 | Parent uncertainty-context loading raised `route uncertainty preflight route pose does not match the admitted planning start`; mission failed closed. |

Both survey legs moved the robot, totaling **2.071376 m** of recorded translation. The survey reports **95.3100159% modelled coverage**, two viewpoints, and five retained hypotheses: two multi-view candidates pending camera validation and three provisional candidates supported by one viewpoint. Modelled coverage does not establish visual visibility of all stands.

The camera phase recorded **0/5 confirmed QR identities**, an empty inspection order, and five unvisited candidates. No selected first UID, candidate route/certificate, candidate child preflight/event log, candidate motion permit, or startup-reseal attempt exists in the copied inventory. `selection_000` identifies the projected candidate pool, not a chosen stand. This run therefore supplies no new evidence about camera geometry, QR decoding, head angle, or backside classification.

## Exact cause and reproduction

The saved selection preflight contains the independently chained TF `map <- base_footprint` result in `route_pose`. The new planning-frame builder deliberately computes its authoritative start using the captured direct `map <- odom` transform multiplied by the captured `odom <- base_footprint` pose, sharing the transform used to project the frozen candidates.

| Source | x (m) | y (m) | yaw (rad) |
| --- | ---: | ---: | ---: |
| Original preflight `route_pose` | 0.3824413242684872 | -0.03337124734143321 | -0.020110468861719293 |
| Admitted composed planning start | 0.38630830346895917 | -0.03197293366089271 | -0.02084861665981000 |

These poses differ by **4.112032 mm** and **0.000738148 rad / 0.042293 degrees**. The discrepancy is already present within the same saved preflight; this exception does not establish movement between planning and execution.

The code path at audited commit `b1d1ddf` is:

1. [Planning-frame admission](../../scripts/aufgabe04/real_robot/readiness/candidate_planning_frame.py) derives the composed pose using `admitted_candidate_planning_pose` (line 311).
2. [Candidate approach coordination](../../scripts/aufgabe04/real_robot/candidate/approach.py) supplies that pose as `expected_start` to the uncertainty adapter (lines 2039–2067), before calling the initial selector.
3. [Candidate uncertainty readiness](../../scripts/aufgabe04/real_robot/candidate/route_uncertainty_readiness.py) calls the shared loader without a pose-basis declaration (line 50).
4. [Shared preflight uncertainty context](../../scripts/aufgabe04/navigation/localization/preflight_route_uncertainty_context.py) still reads `payload['route_pose']` (line 85) and requires exact `Pose2D` equality (line 161). It raises the recorded error before constructing the covariance envelope or evaluating a candidate route.

This is a pose-identity contract failure. It is not a rejection of the 30 mm tracking tube or an exhausted uncertainty margin. Localization freshness and stationary checks passed. The chained and odom TF captures have stamp `1789133719.0831099`, while the latest direct map/odom capture has stamp `1789133720.0371642`. The latter is 0.891 s ahead of capture time and was accepted under the existing 1.1 s AMCL future allowance; its recorded receipt age is about 45 ms. Those different temporal bases explain why exact equality cannot be assumed. The audit does not establish physical pose accuracy or a DDS fault.

The [offline replay](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/replay_planning_pose_binding.py) runs the actual planning-frame builder and actual candidate uncertainty adapter on the original preflight. It reconstructs the complete saved planning-frame admission exactly, then reproduces the exact `ValueError`. A diagnostic control using the old chained pose loads the old uncertainty contract successfully; this control does not certify a route or recommend reverting the composed planning basis. [Replay results](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/planning_pose_binding_replay.json) record the values and source hashes.

The configured three startup reseals were never reached: there was no failed candidate child or permit to retire. The new recovery handles a later execution-admission failure and cannot repair this earlier synchronous evidence-contract error. Increasing timeouts, retries, or positional tolerances would not resolve the mismatched definitions.

## Required correction and missing validation

Candidate uncertainty readiness must explicitly use the composed planning basis, reconstructed from the same immutable preflight bytes with the existing shared pose helper. Keep the original chained pose as diagnostic evidence. Preserve the original source hash, exact expected-start binding, frame/capture validation, covariance envelope, and route-clearance gates. Record the selected pose basis and provenance in uncertainty evidence. The shared loader also serves survey startup, so its existing chained-pose contract must remain explicit for those callers. Do not overwrite the original preflight, accept either pose opportunistically, or replace identity validation with a distance tolerance.

The previous correction updated the planning producer and replacement-permit validation but missed this uncertainty consumer. Its reported 440-test suite omitted the candidate uncertainty-adapter and shared-context test modules. More significantly, their existing fixtures assume `route_pose == expected_start`, and the autonomous runner test stubs both planning admission and uncertainty loading. The **27 existing tests** across those two modules and planning-frame readiness still pass on the failing commit; [test transcript](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/existing_boundary_tests.txt). The earlier command check verified valid arguments and deployed code, but did not exercise this real producer-to-consumer handoff.

The necessary regression test must connect actual planning-frame admission to actual candidate uncertainty readiness using recorded evidence with distinct chained/composed poses, then verify that initial selection is reached without invoking motion. Add rejection coverage for wrong starts, unsupported pose bases, missing/wrong frame and capture evidence, non-finite data, and changed source content. Preserve survey-start tests and existing candidate recovery/permit tests. Passing this boundary test will address the demonstrated blocker; it will not prove that subsequent route execution or camera validation succeeds on hardware.

## Evidence preservation

The audit copied and SHA-256 verified **176 original files**, including the parent and both coverage debug bundles. All originals remain unchanged under [the audit directory](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/). The remote source manifest, compressed source archive, local verification manifest, replay script/results, and test transcript are retained there. Key originals are `mission_failure.json`, `preflight/candidate_selection_000_localization.json`, `candidate_frame_projections/selection_000/candidate_frame_projection.json`, `candidate_goal_progress.json`, `coverage/survey_summary.json`, and the parent bundle's manifest and terminal log.

This audit changed no production code, issued no ROS commands, and initiated no robot motion. The correction described above remains to be implemented.
