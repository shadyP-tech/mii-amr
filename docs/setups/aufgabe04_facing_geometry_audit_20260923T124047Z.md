# Why three admitted candidates lacked facing-ready geometry

Run: `stand_explore_exact2_camera_all5_20260923T124047Z`, recorded revision `6c0734cedb6e3b104c592fc7e7693f25a8d99664`. This follow-up examines the same completed run as `aufgabe04_run_audit_20260923T124047Z.md`. It uses saved receipts, every relevant capture-history metadata record, the admitted images, and the corresponding decision code. No production files were changed for this audit.

## Findings

| Candidate / identity | Why no facing-ready record | Evidence |
|---|---|---|
| 0001 / Start | Opposite-side identity-only completion intentionally emits a discovery-only receipt, even with retained orientation. | Backside orientation was 83.02° ±6.18°; opposite receipt retains its frame-projected angle and records no current angle refit. `build_qr_verified_observation_pose` explicitly sets `facing_ready=false`. |
| 0006 / QR_004 | No complete head was selected before associated QR completion. | Six processed frames: four acquisition deadlines and two `head_proposal_ambiguous` results; zero accepted angle samples. |
| 0005 / QR_001 | Candidate border search repeatedly exhausted its comparison budget. | Seventeen processed frames: ten `head_cold_acquisition_verification_budget_exceeded` and seven acquisition deadlines; zero accepted angle samples. |

“Three QR-only candidates” therefore does not mean three absent orientation measurements. Start has a retained orientation; the other two have no admitted orientation. Facing-ready here means a separate validated facing recommendation/catalog entry, not simply an available QR or historical angle.

## Start: artifact semantics, not another failed angle fit

`observer/opposite_identity.py` deliberately bypasses a new head-angle fit. After current candidate association and a fresh decode it supplies the retained backside orientation to the QR observation builder. `artifacts/qr_verified_observation_pose.py::build_qr_verified_observation_pose` preserves that axis but always writes `completion_scope=discovery_only`, `facing_ready=false`, and `motion_authorized=false`.

No step promotes this QR-only result into a facing recommendation using the retained angle. This behavior preserves immediate QR admission and the instruction not to refit the angle, but means the facing catalog remains incomplete. To obtain a facing record without a new fit would require a separate validated derivation from retained orientation, its uncertainty, the current stand-center evidence, and approach constraints. Changing the flag alone would not provide that validation.

The previously identified bounded-angle metadata handoff defect also removed the validated center from Start's opposite planning receipt. That is a separate correctness issue; fixing it alone will not change the QR-only artifact's explicit `facing_ready=false` contract.

## QR_004: alternative borders of the visible head, plus deadlines

The blue stand is fully visible in the admitted image. The final head acquisition expired during `current_head_raw_refinement`; geometry processing took about 166 ms and the complete geometry/QR evaluation took about 207 ms. Across six evaluations the total processing times were 155–236 ms.

Two earlier evaluations completed comparison but returned `distinct_current_heads_ambiguous`. In `frame_000004.json`, seven strict verifications included five accepted border refinements. Their bounding boxes all lie around the same visible head:

- x roughly 295–395 px;
- y roughly 245–351 px, with several competing upper-border placements.

The recorded alternatives are spatially overlapping head-border interpretations, not evidence that five distinct physical stands were present. The selector could not prove them to be equivalent current borders and retained ambiguity. Candidate-association rejection counters were zero for this comparison. The failure precedes an admitted pose/angle; it is not a seven-sample consensus shortage or a rejected fitted-angle uncertainty.

On the final image, three-scan reconciliation supported a unique current target and QR_004 decoded with an accepted association. `QrObservationPoseFallback` sets the geometry grace to zero for a reconciled target, so discovery completed immediately. The five-second acquisition opportunity is not a mandatory delay after successful identity admission.

## QR_001: “border unavailable” masks incomplete verification

The green stand is fully visible against the wall. The final status says `model_current_head_border_unavailable`, but the producer diagnostic is more specific: `head_cold_acquisition_verification_budget_exceeded`.

Final-frame diagnostics show:

- 49 considered proposals and 43 distinct locator hypotheses;
- 32 border families;
- 12 allowed strict verifications, of which nine accepted a refinement;
- eight resolved physical-frame aliases;
- **23 still-unverified hypotheses**.

Accepted refinements cluster around x=302–402 px, y=251–350 px. Plausible borders were found. The algorithm could not finish proving that the remaining alternatives were aliases, texture, or invalid competitors within its bounded work. `head_cold_acquisition.py` refuses to select a head while uncovered hypotheses remain; `physical_head_pipeline.py` translates this into the generic border-unavailable estimator status.

Across the 17 processed images, ten hit that verification limit and seven hit the time deadline, often while checking uncovered families or texture structure. Total evaluation time was 238–383 ms. No geometry frame reached accepted angle sampling. Two detector results became obsolete, and six exact-TF retries exhausted during an approximately 1.3-second interval; these further delayed completion but do not explain the persistent head-selection failure.

The first associated QR reached the fallback at approximately 12:53:33.889 UTC and started the 1.5-second geometry grace. The next accepted fallback completion was at 12:53:40.023 UTC, about 6.13 seconds later. Intervening frames lacked a fresh independently bound QR or failed source admission. The opportunity for geometry was therefore already longer than the configured grace; simply increasing that delay would not address the repeated per-frame search limits.

## Implications

1. Preserve the distinction between retained orientation and facing-ready admission. Start can reuse its backside angle, but needs a validated facing-record construction if a complete facing catalog is desired.
2. The next geometry target is physical-border hypothesis consolidation and bounded comparison work. The recorded blue/green frames are useful regression fixtures: require evidence-backed treatment of overlapping border interpretations, retain their uncertainty, and preserve rejection of genuinely distinct heads.
3. Do not relax candidate association, arbitrarily pick the largest rectangle, or globally increase deadlines. The latter would consume the 0.5-second freshness budget and worsen the already observed stale results.
4. Immediate front admission can use one valid current head fit plus associated identity (`ImmediateFrontAdmission`); reducing `--axis-sample-count 7` would not solve these zero-fit cases.

## Evidence locations

All run evidence is under `results/implementation_checks/run_audit_20260923T124047Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260923T124047Z/`.

- `candidates/001_survey_candidate_0001/camera_lidar_attempt_00/axis_observation.json`
- `candidates/001_survey_candidate_0001/camera_lidar_attempt_01/qr_observation_pose.json`
- `candidates/003_survey_candidate_0006/camera_lidar_attempt_00/capture_history/frame_000004.json`, `frame_000006.json`, and `frame_000007.json` / `.compressed`
- `candidates/004_survey_candidate_0005/camera_lidar_attempt_00/capture_history/frame_000023.json` / `.compressed`, plus earlier frame metadata and `observer_events.jsonl`
- Both candidates' final `observer_status.json`, `stand_facing_catalog.json`, and `qr_observation_pose_catalog.json`.

Spatial overlap of alternative borders is directly observable in the recorded corner coordinates. Whether every remaining alternative can safely be consolidated requires a targeted replay and tests; this audit does not claim that proof has already been established.
