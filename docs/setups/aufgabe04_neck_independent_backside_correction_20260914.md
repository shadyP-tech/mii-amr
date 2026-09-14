# Neck-independent backside and camera handoff correction — 2026-09-14

The active measured-head camera path now computes angle and backside classification without requiring neck, stem or head/neck-junction validation. This implements the user's correction to the preceding audit. A visible rectangle still needs current raw-border support, measured-head pose quality, candidate association and complete-head marker checks before it can supply an opposite-side observation.

## Module boundaries and behavior

| Module | Responsibility |
| --- | --- |
| [head_proposal.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_proposal.py) | Locate a complete current rectangle without neck evidence. Parallel line extents are search seeds only; current raw borders/corner arms must verify them. |
| [head_outer_border.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_outer_border.py) | Prefer enclosing current raw head borders using measured panel/symbol dimensions; reject a positively identified QR-paper inset using an independent current symbol size reference. No neck or QR pose supplies the angle. |
| [head_model_fit.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_model_fit.py), [head_model_quality.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_model_quality.py) | Fit the measured 3D head to four current borders. Keep raw support, corner arms, minimum pixel span, positive depth, planar ambiguity, reprojection and yaw-uncertainty gates. Neck fields are diagnostic compatibility fields and cannot reject the fit. |
| [head_backside_classification.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/head_backside_classification.py) | Attach a backside-candidate label to the same measured-head angle using physical-profile/projection checks and current QR-marker absence. No neck/junction condition remains. |
| [backside_head_crop.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/backside_head_crop.py) | Require a current complete-head crop, associated with a unique scan cluster, before the observer uses backside evidence. A clipped nominal result or old crop hint cannot satisfy this gate. |
| [scan_target_geometry.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/scan_target_geometry.py) | Compute bearing and range bounds from the same exact-time scan-frame candidate point. Preserve numerical radius, uncertainty, tolerance and clustering rules. |
| [timeout_policy.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/timeout_policy.py) | Treat a trailing stale tuple consistently with an obsolete detector result only when valid, explicitly unpoisoned prior processing evidence and expected deadline cleanup exist. |
| [tf_delivery_trace.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/tf_delivery_trace.py) | Record bounded ingestion and exact-lookup diagnostics in the executing observer's actual TF buffer, without additional subscriptions or queries. |

The physical model pipeline no longer falls back to a neck-dependent locator when head acquisition fails. QR/tracker inputs may locate a search, but current head pixels own the angle. A lone rectangle does not establish physical scale by itself: the model interpretation remains conditional on the candidate and calibration, and observer association is still required. No tight scale threshold was invented to distinguish the measured 71 mm paper from the 78 mm head under uncertain projection.

A usable nominal backside result now enters the bounded wider current-head proposal path. After unique scan association, the complete head crop is recentered and fitted with adjusted intrinsics. Current QR/finder evidence is evaluated there. Positive evidence from any evaluated crop retains its veto or identity-conflict effect. The final crop gate checks the proposal, fitted-head extent, complete borders and current marker absence. Warm search hints repeat these current-image checks; they never reuse a head angle, QR absence or scan association.

The existing stationary seven-sample receipt contract and opposite-side route priority remain. Missing neck pixels cannot block them; ambiguous head geometry, ambiguous LiDAR association, stale data and current front markers still can. The purple geometric overlay retains the current measured-head angle and its quality checks.

## Other run corrections

The recorded front view compared scanner returns with a base-frame distance even though the scanner sits 32 mm behind the base. Range bounds now use the already available exact-time `scan_from_map` candidate point. The separate base-frame distance remains available for camera-distance diagnostics and recovery semantics. Recorded replay retains the original tolerances and changes 20 of 21 proposal checks to unique associations; the remaining ambiguous scan stays rejected.

A trailing `stale_sensor_tuple` with valid prior processing no longer becomes fatal solely because it arrived after an `obsolete_detector_result` status. This authorizes only a typed candidate-local failure for the existing bounded retry/defer policy. It does not admit a stale observation or authorize movement without fresh motion gates. Stale-only attempts, missing/malformed evidence, poisoned identity history and unrelated crash exits remain terminal.

Terminal capture/validation errors now persist the actual attempted camera view and bounded error metadata before re-raising the original exception. Interruptions and terminal failures do not continue motion. This prevents the old third-view failure from leaving a two-view checkpoint.

Observer status now includes `tf_delivery`, capped by default at 16 edges, four receipts per edge and eight recent lookups. It records source validity timestamps, local ROS/monotonic ingestion times, counters and exact query outcomes. Ingestion is entry into the executing buffer's `set_transform` call, not proof of DDS arrival time or broadcaster publication. Static transforms are explicitly timeless. Trace faults cannot change the wrapped insertion/lookup return or exception.

## Validation and limits

The new lossless recorded fixtures preserve source image, metadata, calibration and physical-profile hashes. Tests erase all below-head neck pixels and verify unchanged head-fit outcomes. One latest backside image becomes usable with no neck; a genuinely ambiguous image still fails pose admission. The complete front decodes `Start`; its current finder patterns independently veto backside classification even when payload decoding is injected as empty. A synthetic seven-frame epoch exercises the real observer association, freshness, receipt validation and opposite-side branch.

The [17-image current registration replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/BACKSIDE_REGISTERED_REPLAY.md) exercises the actual current head-proposal, scan-registration and complete-crop paths:

| Saved-image outcome | Count |
| --- | ---: |
| Usable backside geometry, association and complete-crop evidence | 5 |
| Planar head ambiguity | 2 |
| Excessive yaw uncertainty | 1 |
| No usable current head proposal | 6 |
| Ambiguous scan association | 3 |

All five usable results have `outer_border_verified=true` and false neck/junction compatibility flags. This improves the recording from zero usable backside results, but is **not** seven-sample live consensus or hardware success. The replay does not renew timestamps or generate motion authority; local OpenCV 4.13.0 differs from the recorded 4.5.4 runtime. The pose uncertainty is a local pixel-noise model, not measured physical angle accuracy.

The final 120-module suite reports **1,285 passed, six failed, one skipped and 1,064 passing subtests** under Python 3.12.14 / OpenCV 4.13.0. All six failures are the exact legacy `test_stand_axis_image.py` failures independently reproduced on the original clean `1cf07084` checkout under the same runtime; no new failures appeared. The [final test output](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/final_focused_tests.log) and [baseline output](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/baseline_stand_axis_image.log) are preserved separately. That legacy image estimator is not modified by this correction.

All 41 changed/new Python files parse with the Python 3.10 grammar, and `git diff --check` passes. SHA256 verification confirms that all **841 original run artifacts remain unchanged**. The [validation manifest](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/correction_validation/validation_summary.json) records the code hashes, runtime versions, exact baseline failures and replay limits.

Implementation and validation stayed local. No workstation upload, deployment, ROS session, robot movement or new motion certificate was performed. The next hardware acceptance milestone remains a first-view backside receipt, admitted opposite-side route, current front QR/facing recommendation, and progression to the next candidate.
