# First-view camera correction — 2026-09-11

Implemented the two code corrections identified in the [closer-approach audit](aufgabe04_first_view_neck_audit_20260911T142702Z.md). The angle still comes only from the measured outer-head model. No angle, freshness, range, collision-clearance or sample-count threshold was relaxed.

## Changed behavior

`head_model_neck.py` retains its paired straight-rail evidence and physical paper-inset test. When a straight rail begins too far below the head, the new `head_neck_connectivity.py` checks whether actual raw pixels connect that rail back to the fitted bottom edge. It allows at most six backward rows, a two-pixel lateral corridor and one-pixel steps, with distinct paired paths and unchanged rail-width limits. Missing rows are never filled. Steep/degenerate/clipped bottom edges cannot use this continuation.

Diagnostics now distinguish the straight core's `core_start_gap_px`, `run_start_row_px` and `rail_columns_px` from `raw_continuation.paths_px` and the accepted junction gap. Failed continuations preserve the original rejected gap, keeping existing outer-border recovery behavior intact. The 45-degree image-bottom slope limit is not a stand-yaw limit.

The new `observer/current_head_association.py` admits a current quality-verified head against the original map projection independently of nominal/reacquired crop provenance. It checks the active model profile, unchanged expected-size gate, complete current crop geometry and original projected-center displacement, then uses the existing bounded camera-centered LiDAR association. It keeps the original maximum 1.5 head-height displacement, 12-degree map/camera difference, configured narrow cone, surface-range interval, fresh scan and unique-cluster requirement. A failure cannot inherit acceptance from preliminary or QR association.

`observer/current_head_qr_binding.py` keeps the newly available registration tied to the head's own QR: the decoded quadrilateral must lie inside the measured head, and both independent ray associations must share samples from the same scan cluster. Overlapping sample subsets are sufficient; disjoint clusters are rejected. Conflicting QR symbols still poison an associated stationary epoch. Current-front proof, seven-sample consensus and final publication freshness remain required.

`observer/node.py` integrates these helpers while retaining the real ROI source and strict-retry diagnostics. Successful nominal fits do not require a wider crop or another fit just to obtain registration provenance. The existing candidate controller stops local inspection when the first view supplies a validated recommendation. No controller threshold or view-budget change was needed.

## Validation

- **379 focused tests passed**, zero failures/errors/skips, across 39 camera, head geometry, association, evidence and candidate-inspection modules.
- New regressions cover the exact saved raw junction, connected/sloping/rounded rails, paper-only and disconnected/crossing-path negatives, nominal/reacquired association equivalence, excessive displacement, stale scans, ambiguous clusters, wrong range/profile/scale, neighboring QR symbols, conflict poisoning and late publication.
- The observer integration test commits a seven-sample recommendation for a shifted nominal head outside the old map cone with `strict_retry_applied=false`. A controller regression confirms that a recommendation at the first view requests no additional local or opposite move.
- `git diff --check` passed. All 587 original audit files still match their retrieved hashes.

Two unpatched production replays used saved data from `stand_explore_exact2_camera_all5_20260911T142702Z`:

| Replay | Result |
| --- | --- |
| Original selected crops/current proposal seeds through production head fitting | 62/62 usable; estimated yaw 26.152–26.433 degrees |
| Cold-start nominal crops, fresh QR decoding and production head acquisition with no saved pose/proposal/QR measurement supplied | 62/62 usable; estimated yaw 25.538–26.435 degrees |
| Production current-head association on the latter replay | 52 accepted, 10 ambiguous-cluster rejections |
| Independent QR binding | 49 accepted; remaining frames rejected by map/cluster association |
| Production evidence accumulator | Seven axis samples and current QR_003 front identity at saved frame 10, spanning 1.396 seconds of saved image timestamps |

The second replay uses original compressed images, CameraInfo, recorded nominal ROI/projection configuration and saved scan/TF data. The association/evidence windows use recorded sensor outcome times. It exercises current image acquisition and decoding, not the entire live observer scheduling/transport loop or recommendation publication. Local OpenCV is 4.13.0; the recorded robot environment used 4.5.4. Timing and physical angle accuracy still need a real-robot run. No ROS, robot motion, deployment or remote code modification occurred.

Replays and test evidence:

- [Production nominal-image replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/production_first_view_replay.py)
- [Production replay results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/production_first_view_replay.json)
- [Focused test runner](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/validate_first_view_correction.py)
- [Focused test results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T142702Z/validate_first_view_correction.json)

No new command-line options are required. Preserve `--candidate-approach-offset-m 0.50` when running the corrected checkout. The previous uncommitted 0.50 m default change and audit documents were retained.
