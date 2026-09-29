# Regression audit: 20260929T121800Z

The latest run failed because a newly supported LiDAR evidence format was not
handled by the backside receipt validator. The observer raised `KeyError:
'current'` after obtaining seven usable backside orientation samples. This is a
confirmed software regression, introduced by `4dc00c5` on September 28 at 14:13
CEST, and still present at the recorded run revision `d2a8f98`.

## Provenance and outcome

- Run: `stand_explore_exact2_camera_all5_20260929T121800Z`, September 29,
  approximately 14:18–14:23 CEST (12:18–12:23 UTC).
- Recorded revision: `d2a8f98fa4a8cad1ce42ea2641e5a60febd0bbdd`, clean `main`.
- Read through SSH alias `mii002`; the workstation reports hostname `mii001`.
- Both survey motions and both initial candidate approaches completed.
- First visited candidate `survey_candidate_0003` was confirmed as `QR_003`.
- Second candidate `survey_candidate_0001` reached backside consensus but
  produced no terminal axis receipt. Its QR identity remains unresolved.
- Final result: `failed_closed`, one of five required identities confirmed.
  No opposite-side route or checkpoint was attempted in this run.

The observer exited with return code 1, `completion_kind=child_exit`,
`deadline_expired=false`, and no supervisory signals. The recorded Python
exception, rather than an observer timeout, sensor-startup stop, or route
uncertainty rejection, explains this run's termination.

## Failure chain

1. The stopped head observation has seven orientation samples and repeated
   backside appearance support. The final debug frame has image timestamp
   `1790684584.7385798`, approximately 12:23:04.739 UTC.
2. Its narrow camera-associated LiDAR ray contains two fragments. Three scan
   witnesses support one persistent target in the broader envelope. The shared
   scan code accepts the narrow ray using proof kind
   `witnessed_registration_envelope_subset`.
3. `current_head_association.py` calls `bind_ray_to_envelope`;
   `registration_evidence.py` propagates that proof into target registration.
   `build_backside_axis_observation` consequently constructs schema 4.
4. `_validate_target_registration` recomputes and accepts the nested proof.
   The subsequent receipt-context check in
   `artifacts/backside_axis_observation.py:325` still reads
   `target_registration["witnessed_fragmentation"]["current"]`.
5. The subset proof stores the current tuple at
   `target_registration["witnessed_fragmentation"]["envelope"]["current"]`.
   The direct lookup raises `KeyError`.
6. `commit_bounded_head` catches `TypeError` and `ValueError`, so this exception
   escapes the observer process. The parent then stops the mission for missing
   terminal observation evidence.

The final subset retains source indices `[0, 1, 2, 226]`, two eligible fragments,
and one independently witnessed target. Its camera/map bearing difference is
only 0.007936 rad, below the three-degree bound. This was not a bearing-limit
failure or insufficient backside confidence.

`observer_status.json` is one frame behind the crash: it records image timestamp
`1790684584.5494156` and registration without the subset wrapper. The later
`perception_debug/latest_metadata.json` contains the actual triggering shape.
Using only the status file would miss this defect.

## Change that introduced the mismatch

Commit `4dc00c5d4121820499613968ef2ce571f4f7eda6` added
`shared_scan_cluster.py`, the subset producer in `current_head_association.py`,
and subset dispatch in `validated_witnessed_fragmentation`. The backside
context check retained its direct-proof assumption from `6baeb22` (September
14). Producer and general proof validator were updated, but this consumer was
not. Later commits inherited the inconsistency.

The QR receipt path already unwraps this proof correctly in
`shared_scan_cluster.py::validate_cluster_receipt_context`. The latest
checkpoint change `d2a8f98` does not explain this crash: its execution stage was
never reached.

## Offline reproduction and test gap

Evidence and replay files are under
`results/implementation_checks/run_audit_20260929T121800Z/`.

`replay_backside_schema.py` reconstructs the receipt builder inputs from the
final debug frame, including its complete registration proof, bounded angle,
three-scan target reconciliation, and measured head-position evidence. Only
artifact paths are relocated to local copies; their content checks remain.

- The unmodified witness validator accepts and recomputes the subset.
- The unmodified receipt builder reproduces `KeyError: 'current'`.
- An in-memory change that unwraps the already validated subset before the
  existing context checks accepts the reconstructed schema-4 receipt.
- Candidate-center, robot-pose, and image-timestamp mismatches still reject.
- The resulting validated target center is approximately
  `(-1.128366, -0.484700)` m with 0.026737 m uncertainty.

This is a receipt-boundary replay. It uses the recorded temporal-window result
and supplied sensor/QR gate flags; it does not rerun pixel fitting, ROS delivery,
or the whole mission. It proves the cause and a narrow correction for this
crash, not downstream route feasibility or eventual mission completion.

The unchanged production code also passes **61 tests** across
`test_backside_axis_observation`, `test_measured_backside_observer_handoff`,
`test_bounded_head_observation`, and `test_scan_target_persistence`, run with
`python -m unittest`. Existing tests cover direct fragmentation proofs and
reconciled unfragmented backside observations separately. They miss the
combination of reconciliation, subset fragmentation, and backside publication.

## Relation to earlier runs

The exact nearly successful first September 28 run has not been established
from the available records. Available saved runs establish this sequence:

| Run (UTC) | Revision | Recorded outcome |
| --- | --- | --- |
| September 23, 12:40:47 | `6c0734c` | Five identities, exit 0; three identities were QR-only |
| September 23, 14:46:40 | `3ad9b00` | Five identities; return to Start rejected by route uncertainty |
| September 28, 12:17:06 | `4dc00c5` | One identity; invalid opposite standoff raised uncaught `ValueError` |
| September 28, 12:46:19 | `313ae38` | One identity; admitted opposite route then missing initial odom |
| September 28, 13:09:11 | `0329c68` | Three identities; fourth candidate target reprojection mismatch |
| September 28, 14:24:07 | `f644ecb` | One identity; opposite arrival and observer used different centers |
| September 28, 15:06:41 | `696b1ef` | One identity; opposite routes rejected by uncertainty, later interruption |
| September 29, 12:18:00 | `d2a8f98` | One identity; subset-proof receipt validation crash |

The successful September 28 backside receipts inspected from the 15:06 run do
not carry this subset proof and therefore do not exercise the crash. A branch
can be broken since September 28 while other scans continue to succeed.

The core route-uncertainty evaluator is byte-identical from `3ad9b00` to
`d2a8f98`. Budget defaults and robot profile are unchanged from `6c0734c`.
The earlier failures concern different geometry, evidence handoffs, and
startup/recovery defects; they are not evidence that the current crash came
from tightened uncertainty thresholds. Restoring the old all-five checkout
would also restore its documented loss of validated target-center evidence.

## Correction boundary

Normalize the validated proof's context access for direct and subset forms,
then retain every candidate, image, and stopped-pose binding check. Add a
regression through actual backside publication using this combined branch,
plus invalid context and invalid witness cases. Merely catching `KeyError`
would hide the crash while still losing the valid observation.

The original investigation left production code unchanged and evaluated the
proposed change only in memory. The subsequent implementation is recorded below.
No deployment, ROS nodes, or robot motion were performed.

## Implemented correction

The receipt validator now unwraps `SUBSET_KIND` only after full registration
and witness recomputation. Direct witness proofs retain their existing path;
the serialized subset proof and all candidate/image/stopped-pose checks remain
unchanged.

`test_bounded_head_observation.py` now drives the real persistence,
reconciliation, association, and bounded publication functions through six
ordinary frames and a fragmented seventh frame. Three independent synthetic
scan witnesses produce the subset proof. The test reproduced the exact
`KeyError` before the fix, then passed with the correction. It checks durable
schema-4 publication, retained nested evidence and target center, and rejection
of five receipt-context mismatches and a tampered witness.

Validation: **98 tests and 135 subtests passed** across ten related test modules.
The original recorded-frame replay also passes the production builder and
validator with `replay_backside_schema.py --expect-fixed`; results are saved as
`implementation_replay_result.json`. This remains an offline receipt replay,
not a real-robot mission completion result.

An additional architecture check found two outdated file-inventory assertions
in `test_orchestration_module_boundaries.py`. Both failures were reproduced
against an exported, unmodified `d2a8f98` tree. They are outside this fix; their
baseline output is saved as `baseline_architecture_tests.txt`.

## Primary evidence

All paths below are relative to the local audit evidence directory.

- `results/real_runs/stand_explore_exact2_camera_all5_20260929T121800Z/terminal_run.log`,
  lines 351–373: observer traceback and propagated mission error.
- The matching autonomous mission's `mission_failure.json`,
  `station_segment_runs.csv`, and `candidate_goal_progress.json`.
- `candidates/001_survey_candidate_0001/camera_lidar_attempt_00/observer_process.json`,
  `observer_status.json`, and `perception_debug/latest_metadata.json` within that mission.
- `replay_backside_schema.py` and `replay_result.json`: reproduction and
  in-memory correction results.
- `remote_sha256_and_revision.txt` and `verified_source_hashes.json`: ten key
  copied evidence files independently verified against workstation SHA-256
  hashes. The workstation checkout was still clean at `d2a8f98` when checked.
