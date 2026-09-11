# Planning-frame decision commit correction — 2026-09-11

Implemented the correction for the [latest run's camera commit failure](aufgabe04_camera_commit_audit_20260911T150433Z.md). A successfully observed candidate can now pass its TF/odom provenance through camera-decision validation and enter the persistent registry.

## Shared contract

`navigation/approach/planning_frame_evidence.py` owns the planning-frame wire validation. `CandidatePlanningFrame.from_evidence()` provides the typed reader, and the constructor and serializer use the same validator. Camera-decision binding, backside projection and autonomous catalog loading now call that reader; the duplicated readers were removed.

Version-one projection artifacts support the existing four-field legacy representation and the defined optional `pose_provenance` extension. Unknown outer/provenance fields, explicit null provenance and incomplete diagnostic groups are rejected. Provenance is copied and retained, including extensible finite capture metadata.

Validation checks capture availability, frame identities, finite coordinates/timestamps, normalized transform geometry and direct map/odom × odom/base composition. Optional chained-pose diagnostics must agree with their captures and reported deltas. Capture identities share the same normalization as the existing preflight admission code. Numerical pose comparisons allow only `1e-12` absolute floating-point error; stored age arithmetic allows `1e-6` seconds because epoch floats lose precision relative to ROS integer nanoseconds.

The ROS-free capture helpers in `navigation/localization/candidate_planning_pose.py` serve both live preflight reconstruction and persisted evidence validation. Historical parsing checks internal consistency; it does not refresh localization or grant motion authority. A missing legacy provenance field does not create substitute evidence. Existing content hashes, population/reprojection checks and actual decision-recorder validation still apply.

## Validation

**150 focused tests passed**, with zero failures, errors or skips, across 13 modules. The 27 new tests cover shared-contract validation and real-producer integration through the actual decision recorder, backside writer/loader and complete catalog promotion. Resealed malformed provenance is rejected before registry or canonical-receipt writes. Python 3.10 syntax validation and `git diff --check` passed.

The saved `stand_explore_exact2_camera_all5_20260911T150433Z` decision was also replayed through the actual recorder in a disposable copy of its original run directory:

- All three saved planning-frame projections round-trip without losing provenance.
- The original prepared decision is accepted, return code **0**.
- Candidate `survey_candidate_0003` transitions from **`pending_camera` to `confirmed`**.
- The canonical receipt equals the original prepared request; other registry candidates are unchanged.
- All **312 original artifacts** retain their retrieved hashes.

The replay does not run the later mission steps or establish hardware completion. No ROS, robot motion or deployment occurred. No command-line option changes are required for this correction; the robot checkout needs the updated source files, including the new shared module.

Evidence:

- [Focused test results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/validate_planning_frame_contract.json)
- [Focused test runner](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/validate_planning_frame_contract.py)
- [Saved decision replay results](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/validate_camera_commit_correction.json)
- [Saved decision replay](/Users/stephpark/Documents/stephsWorld/mii-amr/results/audits/stand_explore_exact2_camera_all5_20260911T150433Z/validate_camera_commit_correction.py)
