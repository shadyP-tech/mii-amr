# First-candidate admission and viewer parity audit — 16 September 2026

The first candidate was not admitted because exploration exhausted cold head acquisition before obtaining a usable 3D fit. Independently, its QR-only fallback hit a reproducible camera-context serialization/type defect on every otherwise qualifying QR observation. The later request to move to another inspection point was a consequence of these perception/admission failures.

The debug viewer really did produce accepted, fresh current-pixel geometry. It was already tracking a head, however; the recording does not show automatic cold acquisition. The shared 3D fitter works on the mission's saved images too when given a head-location hint. Making the observer behave like the working viewer therefore requires repairing acquisition and context handling, rather than replacing the 3D model or increasing an angle limit.

## Evidence scope

- Mission: `stand_explore_exact2_camera_all5_20260916T133025Z`, clean commit `aac5510b3fa5aa63b1e089330491ac1bcec3fa6c` on mii001.
- First candidate: `survey_candidate_0003`, `candidates/000_survey_candidate_0003/camera_lidar_attempt_00`.
- Viewer: `recording_20260916_153656_477415458`, 53 saved frames, **15:36:56–15:37:01 CEST**. User reports the same physical observation location.
- Model SHA256: `56fe19dcbfc8aa58682ea460e702a499c65cc719423940e6a892ca581e6d0b5f`; same measured physical model and camera calibration in both paths.
- Read-only remote source inspection and saved-image replay. No ROS nodes, commands to the robot, source edits on mii001, or recording/image transfer. Replay mounts were read-only; before/after source hashes match.

Original files remain under `/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/` on mii001. Derived numeric diagnostics and replay scripts are in [`results/aufgabe04/debug_audits/stand_explore_exact2_camera_all5_20260916T133025Z`](../../results/aufgabe04/debug_audits/stand_explore_exact2_camera_all5_20260916T133025Z/).

## What happened

| Time, CEST | Recorded event |
|---|---|
| 15:30:25 | Mission started |
| 15:32:52 / 15:34:13 | Both LiDAR survey legs completed; 0.877142 m and 1.164595 m |
| 15:34:23 | Camera handoff, five candidates, 95.31% modelled coverage |
| 15:35:18.912 | First approach completed, 0.279381 m |
| 15:35:26.822 | First uniquely associated `QR_003`; fallback receipt rejected |
| 15:35:56.825 | Observer returned `unobservable` inspection progress after its 30-second front recovery window |
| 15:36:10.961 | Proposed extra inspection route failed uncertainty admission |
| 15:36:33.864 | Another inspection route validated; recorded events end before its preflight result |

The camera observer exited normally with return code 0. It produced `inspection_observation.json`, not a geometry recommendation or `qr_observation_pose.json`. The goal ledger remained 0/5. No subsequent camera attempt or executed movement is recorded. The parent bundle has no final exception, mission completion, or completed post-run bundle; these files cannot establish whether the overall process was interrupted, waiting, or stopped without final logging.

## Why exploration did not obtain an angle

The first observer processed **82 fresh, candidate-associated frames**, with zero accepted angles:

- 80 `head_acquisition_deadline_exceeded`.
- 2 `head_cold_acquisition_verification_budget_exceeded`.
- 74 frames reached zero strict border verifications.
- Deadline locations: 73 `cold_hypothesis_support`, six `cold_strict_verification`, one `cold_fragment_corner_support`.

The camera was delivering images promptly. Median header-to-receipt age was **33.1 ms**; median detector processing was **334.1 ms**, with **330.5 ms** spent in head acquisition. Median source-to-completion age was **400.8 ms**. All 82 processed observations passed this observer's source freshness and evidence association checks; this is not evidence that a stale 3D angle was available but rejected. There was no accepted angle at all.

The saved candidate ROI already includes the visible head. One representative crop is `(120, 68)–(589, 538)` in the 800×600 image; projected center `(354.47, 303.13)`, expected head height 104.14 px. The viewer's measured head is approximately `(289, 228)–(391, 337)`. Crop clipping is not the demonstrated blocker in this run.

The deadline stage label does not distinguish gradient work from association work. [`head_cold_acquisition.py:254`](../../scripts/aufgabe04/perception/stand_axis/head_cold_acquisition.py#L254) scores supported hypotheses and invokes the per-proposal association callback inside the same stage. That callback computes the head ray and scan association, then previews persistent scan evidence. [`node.py:1391`](../../scripts/aufgabe04/real_robot/observer/node.py#L1391) rebuilds the persistence context; [`scan_target_persistence.py:401`](../../scripts/aufgabe04/real_robot/observer/scan_target_persistence.py#L401) previews through `resolve`, reprocessing retained and pending scans. The viewer does not run this candidate-association callback.

Consequently, the live logs alone do not support blaming all of the latency on background edges. Repeated scan-witness work and cold border comparison both need separate measurements. The frame metadata shows successful proposal association previews, not a blanket candidate-association rejection.

A saved-scan microprofile used frame 16's actual scan, exact-time transforms and 16 logged proposal centers. Current scan association alone cost **1.58 ms** for all 16. Association plus context construction and persistence preview cost **9.74 ms** with no history, **38.21 ms** with three preceding saved scans, and **60.93 ms** with five retained scans. All proposals remained associated. The five-scan case is about 24% of that frame's recorded 257.7 ms acquisition time: material, but insufficient to explain the entire failure alone. This reconstructs bounded history from saved scans, not the exact live pending queue; timing was repeated 30 times per configuration. See `association_microprofile.py` and `association_microprofile.json` in the diagnostic directory.

## What the latest viewer recording establishes

All **53/53** frames have:

- `model.usable=true`, `model.committable=true`.
- `geometry_overlay.state=current_head_fit`, `current_fit_accepted=true`.
- `model.pose_fit_source=model_current_measured_head`.
- Accepted current four-border/corner support, raw support mean 1.0.
- Fresh results and accepted tracker updates.

Yaw is **24.514–24.827°**, median **24.661°**. Reprojection error is **0.249–0.384 px**. Median model processing is **25.78 ms**; median source-to-completion age **94.81 ms**. This is a measured overlay, not merely a retained or predicted drawing. Consistency and low reprojection error do not independently establish ground-truth physical angle accuracy.

Every saved frame uses `tracked_head_search`. The first saved source sequence is 208 and already contains an earlier pose prediction. The viewer processes the full calibrated image, has no mission candidate projection or scan target binding, and uses `--no-qr-decode`. Its recording therefore proves current head tracking/fitting, but not cold acquisition, QR identity, or mission admission.

Both use `estimate_stand_axis_from_metric_model`. The tracked branch refines borders near the projected prior head; the cold branch must locate, compare, select and bind a physical border family first. Sharing the final fitter does not make these acquisition paths equivalent.

## Saved-pixel replay

Replay used the unchanged remote code/OpenCV, original saved pixels and original calibration. Optional QR work was disabled to isolate geometry. Cold runs retained production comparison limits but had no wall-time deadline or LiDAR callback. The geometry minimum edge height was held at 14 px across the diagnostic modes. These are isolation tests, not live admission tests.

| Input and diagnostic mode | Accepted geometry | Median processing |
|---|---:|---:|
| Viewer frames, recorded prior tracking hint | 53/53 | 16.85 ms |
| Same viewer frames, full-image cold start | 2/53 | 86.70 ms |
| Viewer frames, mission ROI/projection, no scan callback | 0/53 | 132.24 ms |
| Actual mission frames, full-image cold start | 0/55 | 89.78 ms |
| Actual mission frames, recorded ROI/projection, no scan callback | 0/55 | 137.62 ms |
| Actual mission frames, later viewer location hint — diagnostic only | 55/55 | 16.43 ms |

The last row intentionally uses a later recording's head-location hint. It proves that the mission pixels support the unchanged current-pixel fitter; it is noncausal evidence and cannot authorize an earlier observation. No later angle was installed as a mission measurement. The resulting fitted mission yaws were 24.60–25.63°.

The mission ROI replay still produced **49 verification-budget failures and six ambiguous proposals**, even without a processing deadline or association cost. Full-image replay instead produced 36 selected/fitted border-binding rejections and 19 ambiguities. Increasing the time allowance or processing the entire image therefore does not solve this sequence by itself.

The two live verification-budget failures retained 17 individually accepted raw border refinements. Fitting all 17 current-image seeds diagnostically yielded 12 accepted 3D fits spanning **26.77–34.95°**. On frame 5 alone, accepted seeds yielded 26.77° and 34.95°; the latter even had a lower reprojection error. These were competing physical-border choices, not 17 independently selected heads. Accepting the first successful fit or the lowest reprojection error would not establish the desired border. Cold selection must recover the same physical head boundary that the successful tracker follows.

## Why the QR-only fallback also failed

`QR_003` decoded in 30 processed frames. **28** had valid unique target binding; two had ambiguous scan clusters. All 28 qualifying fallback attempts reached the same failure:

```text
qr_observation_receipt_rejected
QR observation needs its calibrated camera context
```

The calibration is present. Its runtime numeric types are the problem:

1. ROS Humble's `CameraInfo.k/r/p` fields are NumPy arrays of `numpy.float64`.
2. [`node.py:1617`](../../scripts/aufgabe04/real_robot/observer/node.py#L1617) uses `tuple(CameraInfo.field)`, preserving those NumPy scalars in `candidate_context.camera_signature`.
3. [`qr_verified_observation_pose.py:34`](../../scripts/aufgabe04/artifacts/qr_verified_observation_pose.py#L34) accepts exact built-in `int`/`float` types. It rejects these otherwise valid values before publishing the receipt.
4. The publication wrapper converts that exception into a failed fallback; front recovery eventually returns advisory `unobservable` progress and the parent plans another view.

An in-memory full fallback reproduction accepts both the sensor frame and QR sample, but fails receipt publication with the same error. Normalizing the same values to Python `float` makes it publish and complete. The immediate-front artifact uses the same strict numeric predicate, so this shared context also needs correction before relying on successful geometry to commit.

Existing tests used Python-float tuples or a JSON roundtrip. JSON converts these NumPy float64 values to ordinary floats and hides the live producer defect. Normalize numeric values once at the camera-context producer, using a small shared module, and test ROS-shaped arrays before serialization. The strict artifact validators need not be weakened.

## Later navigation rejection

The proposed extra inspection route had **0.32255 m available clearance versus 0.39410 m required**, leaving **−0.071551 m**. Its uncertainty contributions included 0.10097 m localization and 0.10313 m heading uncertainty. The stationary map→odom check passed with zero measured drift and 4.1 mm pose composition error. This evidence does not establish an AMCL jump as the first-pose perception cause.

Repairing admission at the original observation point avoids this unnecessary extra-view plan. It does not justify bypassing route clearance checks for routes that are still needed.

## Targeted correction and validation

1. **Normalize the shared camera context.** Cover immediate geometry/QR admission and QR observation-pose fallback with live-shaped NumPy calibration arrays. A fresh, uniquely associated decode should produce the discovery-only fallback when geometry is unavailable, without waiting for another view.
2. **Repair automatic cold border selection.** Deduplicate and compare supporting physical rails efficiently, distinguish inset/printed rectangles from the physical frame, and retain bounded search hints across brief misses. Use the same acquisition, tracking and current-pixel fitting policy in viewer and exploration; a hint may locate pixels but must not supply an angle.
3. **Remove repeated immutable scan preparation.** Prepare scan/context/history once per image, then evaluate each current head bearing without rebuilding the same witnesses. Preserve competing-cluster handling and a final fresh association. Add separate timing for preprocessing, proposal support, scan preview, strict refinement, fitting and receipt creation.
4. **Replay cold start and track loss, then test live timing.** Use all 53 viewer and 55 mission frames, including QR-present images, without later hints or manual seeds for acceptance tests. Record selected physical borders, angle consistency and end-to-end commit latency. Test one fresh associated QR with geometry and the angle-unavailable fallback separately; no seven paired successes are required for these immediate paths.

No production correction was made in this audit. The reproducible fallback defect and acquisition failures are separate: fixing the former allows discovery to continue, while fixing the latter is necessary for reliable angle admission matching the debug viewer.
