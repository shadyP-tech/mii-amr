# Camera inspection audit: 2026-10-02 15:24 run

## Conclusion

Camera inspection succeeded for the first candidate. The user confirmed that
they manually stopped the run while it was preparing the second candidate.
The evidence does not show a camera rejection or establish that the next
preflight hung.

Run: `stand_explore_exact2_camera_all5_20261002T132409Z`.
Workstation SSH alias: `mii002` (reported hostname `mii001`). Both the recorded
run revision and the current clean workstation checkout were `f7a6f15`.
The local checkout used for code tracing matched that revision.

Raw logs and recordings were inspected on the workstation and remain there.
No robot process or motion was started for this audit.

## Recorded sequence

Times below are Europe/Berlin (UTC+2).

| Time | Evidence |
| --- | --- |
| 15:28:42.457 | First approach to `survey_candidate_0003` completed. |
| 15:28:48.476 | Stopped arrival admission returned accepted; no new LiDAR reacquisition required. |
| 15:28:48.524 | Camera capture started. |
| 15:28:53.045 | Observer status reported `recommendation_committed`. |
| 15:28:53.307 | Camera capture returned successfully, approximately 4.783 seconds after start. |
| 15:29:13.107 | Next candidate, `survey_candidate_0005`, completed dry-run preflight. |
| 15:29:20.359 | Its live child started. |
| 15:29:20.405 | Route validation completed; this is its last recorded event. |

First-candidate inspection processed 13 images, accepted four measured-head
geometry results, and recorded four decoded QR frames. The current identity
binding was accepted as `decoded_qr_target_associated` for `QR_003`.
`inspection_progress.json` terminated with `joint_observation_ready` after one
view, with no LiDAR recovery, distance recovery, or generic route proposals.
Optional centering remained blocked at the scan boundary, but the usable
camera observation completed successfully.

The mission's persisted goal progress confirms one of five required stands,
`QR_003`, with one facing-ready stand. The other four candidates remained
unvisited. This is a partial, operator-stopped run, not a completed mission.

## Why there was no second camera inspection

The second candidate had not begun its approach motion. Its dry-run preflight
passed all 16 checks, including localization, TF, sensor freshness and velocity
ownership. Separate camera/LiDAR timing readiness also passed, with about
8.92 ms image/scan skew.

The live execution then entered the interval between route validation and
completed ROS preflight. There is no live `candidate_001_execute.json`, no
`preflight_passed` or `preflight_failed` event, no motion-start event, and no
camera attempt directory for that candidate. The child terminal log is empty;
the parent ends at the wrapped-command launch. Neither bundle contains a
terminal exit outcome. A subsequent process check found no remaining mission
process. The user's confirmation supplies the missing stop explanation.

In `scripts/aufgabe04/navigation/station_segment/runtime.py`, the live path
calls `run_ros_preflight` at line 835, writes its result at lines 917–924, and
emits `preflight_passed` at line 961. The actual command did not request the
optional initial-pose prompt. Its environment selected the mission preflight
session socket. A normal preflight rejection or handled session timeout would
produce failure events; none was recorded here. The saved evidence cannot
identify a finer preflight substage or how long it ran before interruption.

## Follow-up

This run verifies that the first-view camera changes reached decoding and
admission on the real robot. It does not exercise the opposite-side branch or
prove completion of the five-stand task.

Improve operator-visible progress and interruption diagnostics: print the
admitted candidate/QR identity, explicitly announce live preflight for the
next candidate, and persist an interrupted outcome when the workflow is
stopped. This audit provides no basis for loosening camera or motion gates.

No production code changed. This audit used recorded artifacts and source
tracing; no new runtime tests were needed.

## Workstation evidence

Paths are relative to the workstation repository:

- `results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20261002T132409Z/candidates/000_survey_candidate_0003/inspection_progress.json`
- The same candidate's `inspection_handoff_events.jsonl`, `candidate_arrival_admission.json`, and `camera_lidar_attempt_00/observer_status.json`.
- The run root's `candidate_goal_progress.json`, `run_events/*candidate_001.jsonl`, and `preflight/*candidate_001_dry.json`.
- `results/real_runs/stand_explore_exact2_camera_all5_20261002T132409Z/terminal_run.log` and `git_rev.txt`.
- `results/real_runs/stand_explore_exact2_camera_all5_20261002T132409Z_candidate_001/manifest.txt` and the selected preflight-session environment field.
