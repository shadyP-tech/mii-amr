# Candidate uncertainty pose binding correction

This fixes the parent planning/admission failure in `stand_explore_exact2_camera_all5_20260911T133125Z`, audited in [the run report](aufgabe04_run_audit_20260911T133125Z.md). The planner used direct `map <- odom` composed with the captured odom/base pose, while uncertainty validation compared against the chained map/base pose. The recorded difference was 4.112 mm and 0.042 degrees, causing an exact-identity rejection before candidate selection.

Candidate uncertainty validation now reconstructs the same composed planning pose from the original preflight document. The coordinator supplies the admitted frame's explicit odom identity, including nondefault frame names. Both pose reconstruction and covariance extraction use one strict JSON read, and the returned evidence hashes those original bytes. The original diagnostic `route_pose` is not overwritten.

The implementation uses the existing module boundaries:

| Responsibility | Module |
| --- | --- |
| Explicit pose-basis selection, frame validation, strict start binding, covariance and source hash | `navigation/localization/preflight_route_uncertainty_context.py` |
| Shared composed-pose reconstruction from successful captured TF/odom evidence | Existing `navigation/localization/candidate_planning_pose.py` |
| Candidate-specific basis and provenance in selection evidence | `real_robot/candidate/route_uncertainty_readiness.py` |
| Passing the admitted map/odom frame to candidate uncertainty readiness | `real_robot/candidate/approach.py` |
| Explicit retention of the chained-pose contract for survey startup | `navigation/missions/startup_route_uncertainty_selection.py` |

Exact expected-start equality, source hashing, frame/capture validation, covariance policy and the 30 mm tracking tube remain enforced. Unsupported bases, missing or inconsistent captured poses, and mismatched starts fail closed. Candidate requests cannot fall back to the chained pose. The original survey-start contract remains available and is explicitly selected by its production caller. Source hashes identify the document loaded; this change does not add a separate protocol for detecting arbitrary file replacement between earlier admission and loading.

The recorded fixture in [test inputs](../../tests/aufgabe04/fixtures/candidate_planning_frame_20260911T133125Z.json) includes the complete preflight, expected planning-frame admission and original source hashes. [Handoff tests](../../tests/aufgabe04/test_candidate_uncertainty_handoff.py) connect the real planning-frame builder to the real uncertainty adapter, then exercise that connection through the actual candidate coordinator up to selector invocation. Motion and camera effects must remain uncalled. The tests cover the recorded distinct poses, tiny wrong-start changes, inconsistent captures and explicit nondefault odom-frame propagation. Shared-loader tests retain survey behavior and cover malformed/duplicate JSON, symlinks, unsupported bases, non-finite inputs, failed observations and missing frame/capture evidence.

Validation on Python 3.12.14 passed **489 unique tests and 471 subtests**, with no skips, failures or errors. All seven changed/new Python files passed Python 3.10 syntax parsing; `git diff --check` passed. Source files remained unchanged during the test run. All **176 original run artifacts** were rechecked against their source hashes and remain unchanged. See [validation manifest](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/uncertainty_correction_validation.json) and [test transcript](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/implementation_validation/focused_tests.txt).

The [corrected replay](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/replay_corrected_uncertainty.py) reconstructs the full saved planning-frame admission exactly and successfully loads candidate uncertainty from the untouched original preflight. [Replay results](../../results/audits/stand_explore_exact2_camera_all5_20260911T133125Z/corrected_uncertainty_replay.json) include the source hash, pose provenance, covariance and admission parameters. The original failing replay and its results remain preserved as audit evidence.

No new experiment flags are needed. The robot checkout must include this correction before reusing the previous command. This is local offline validation: no ROS nodes, robot commands, remote deployment, or real-robot run were initiated. Subsequent candidate-route feasibility, execution and stopped camera validation still require hardware evidence.
