**First-candidate QR and stopped-view recovery — implementation, 11 September 2026**

The subsequent [independent measured-head implementation](/Users/stephpark/Documents/stephsWorld/mii-amr/docs/setups/aufgabe04_independent_head_model_20260911.md) supersedes this snapshot's joint QR/head angle prerequisite and adds quality-based admission beyond 35°. The decoder and bounded stopped-recovery changes below remain applicable.

This change addresses the code-level causes in the [latest run audit](/Users/stephpark/Documents/stephsWorld/mii-amr/docs/setups/aufgabe04_run_audit_20260911T104837Z.md). The first front view can recover a decoded symbol's corners, distinguish incompatible joint geometry, and continue observing while stopped before requesting another viewpoint. The command-line options remain compatible with the previous experiment command.

| Module | Responsibility |
| --- | --- |
| [isolated_qr_views.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/qr_scanning/isolated_qr_views.py) | Two bounded rectifications of one current native QR quadrilateral. The recovery view preserves a 4% source-pixel margin at each side and adds a white border. The original symbol coordinates remain the geometry. |
| [qr_decoder_runtime.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/qr_scanning/qr_decoder_runtime.py) | Reuse native and WeChat decoder objects within one image call, and collect bounded decoder provenance. |
| [opencv_qr_detector.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/qr_scanning/opencv_qr_detector.py) | Connect both native single- and multi-quad detection to isolated payload confirmation, preserving identity conflicts and restored crop coordinates. |
| [geometry_contract.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/perception/stand_axis/geometry_contract.py) | Classify a rejected joint QR/head fit when its independent current-pixel fits are good, using existing diagnostics without an additional pose solve. |
| [front_view_recovery.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/observer/front_view_recovery.py) | A fixed stopped recovery deadline for missing QR corners or rejected joint geometry, with current candidate association, freshness and stationary-epoch checks. |
| [camera_distance_recovery.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/real_robot/candidate/camera_distance_recovery.py) | Generate useful outward standoffs within the existing camera/arrival envelope; the preferred approach distance is no longer the recovery ceiling. |
| [observation_node_lifecycle.py](/Users/stephpark/Documents/stephsWorld/mii-amr/scripts/aufgabe04/navigation/foundation/observation_node_lifecycle.py) | Clean up preflight observation nodes with `try_shutdown`, retaining the initiating exception if cleanup also fails. |

Decoded text is still insufficient to supply corners. Each recovered quadrilateral must validate in the original crop, and the isolated image must decode its own payload. Whole-input WeChat rectangles, malformed quads and multiple isolated payloads do not acquire geometry authority. Backend reuse is confined to one call; image evidence is not cached across frames. Equivalent native single/multi quads are only deduplicated within the same processed pixel variant.

Native single-quad isolation is deferred until all ordinary and multi-quad variants fail, with at most four current-image proposals retained. If native detection reports multiple individually valid, original-crop quadrilaterals, that ambiguity persists across preprocessing variants: isolated recovery is blocked and later unique observations retain text without corners. Actual multiple decoded payloads remain conflict evidence. Bounded provenance records backend availability/version, validation reasons, raw/restored bounds, and the selected rectification.

The stopped recovery window lasts **30 seconds from the first qualifying current observation**. Repeated bad frames, changed failure reasons, new QR latch samples and soft misses do not renew it. Ordinary valid QR/axis completion keeps priority. The parent camera timeout remains an independent upper limit. Conflict stops the hold; motion or target changes reset the stationary epoch. A readable-front failure whose neutral head association is ambiguous and whose own QR is unbound cannot accumulate or publish the advisory that ends this view. Ordinary unobservable and backside inspection paths remain available.

For the recorded 0.726 m arrival, outward recovery can propose approximately **0.900 m and 0.826 m** within the existing 0.90 m arrival ceiling and camera framing envelope. Each still requires the normal fresh localization, route admission, motion authorization and arrival checks. Existing per-candidate and per-view budgets apply.

The geometry classifier reports **`model_head_qr_geometry_mismatch`**, preserving the underlying joint-fit rejection, current head/QR residuals and ratios, configured dimensions, and profile provenance. It clears the rejected pose from the returned model artifacts. Diagnostic angles cannot enter axis consensus through this path. The **62 mm symbol profile, 2 px joint-fit limit, 35° QR-bound angle limit and freshness gates remain unchanged**.

Saved-frame verification uses the original JPEG bytes and recorded camera information in [the regression fixture](/Users/stephpark/Documents/stephsWorld/mii-amr/tests/aufgabe04/fixtures/qr_recovery_20260911/inputs.json). The original audit files remain separate from the implementation replays.

The read-only deployed decoder replay compares baseline and patched code using an in-memory source overlay in **OpenCV 4.5.4 / NumPy 1.21.5 / Python 3.10.12**. All four sampled crops now yield `QR_003` with validated original-crop corners. Three previously returned text only. Decoder timing is measured on saved pixels and excludes camera receipt, synchronization, TF waiting, model fitting and ROS executor delay; it is not a live camera throughput measurement. Per-crop timings and exact source hashes are recorded in [qr_decoder_fix_replay.json](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/qr_decoder_fix_replay.json).

| Saved crop | Baseline median | Patched median | Patched output |
| --- | --- | --- | --- |
| Frame 000011, nominal | 149.45 ms | 48.02 ms | `QR_003` with corners |
| Frame 000011, recentered | 121.10 ms | 66.71 ms | `QR_003` with corners |
| Frame 000024, nominal | 160.14 ms | 47.05 ms | `QR_003` with corners |
| Frame 000024, recentered | 22.93 ms | 23.03 ms | `QR_003` with the original recorded corners |

Each median uses three alternating baseline/patched calls after an untimed warmup. All five patched decoder source hashes in the replay match the final local implementation.

[The geometry replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/geometry_contract_fix_replay.json) preserves frame 000024's raw corners and **3.23076 px** rejection while reporting the specific contract mismatch. No extra PnP solve or accepted yaw/normal is introduced. The actual twelve-fresh-frame inspection sequence is covered by policy regression tests: its ambiguous final frame can no longer publish an early front-view advisory, and qualifying failures keep one deadline starting at approximately receipt time **3.260 s**, expiring at **33.260 s**.

The saved joint model still fails. Independent physical measurement of the printed QR symbol and verification of head-versus-paper boundary selection are needed before a dimensional correction is justified. The synthetic 71 mm diagnostic fit is not a replacement calibration, and its approximately 36.13° yaw also exceeds the unchanged 35° admission limit. The code changes therefore establish decoder recovery and bounded inspection behavior; they do not establish successful facing-pose completion on the robot.

Final focused verification: **292 tests passed, 206 subtests passed, one test skipped locally** because local OpenCV 4.13 has no WeChat factory. The deployed OpenCV 4.5.4 replay exercises that real backend on the saved fixtures. Tests cover original-image recovery, restored coordinates, decoder reuse/provenance, same- and cross-variant multiplicity, the actual first-view decision sequence, freshness/association/conflict/motion resets, unchanged angle/fit limits, outward admission bounds, and exception-preserving cleanup. Python 3.10 syntax and whitespace checks pass. All **863 copied original source files** and both fixture JPEG hashes are unchanged; [the validation manifest](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T104837Z/derived/fix_validation_manifest.json) records implementation hashes.

No live ROS observation, motion, or remote deployment was performed during implementation validation. The deployed tests read saved images in a container with read-only host mounts. Production changes are in the local checkout for review and deployment.
