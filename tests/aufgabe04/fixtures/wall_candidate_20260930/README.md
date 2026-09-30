# Recorded wall-target regression, 2026-09-30

These four JSON files are exact-byte copies from
`stand_explore_exact2_camera_all5_20260930T141147Z`, executed at commit
`06b8bdad6e40b2905e4a4751277baeacb281b662`. The manifest records original
run-relative paths, byte hashes, map-file hashes, and the exact arena-bounds
excerpt from the original coverage plan. The map bytes already live in the
repository and are not duplicated here. No camera images or raw scan logs
are needed for this admission regression.

All six original candidates, their source evidence, perception advisories,
and keepout radii are preserved. The test must not remove a candidate or its
keepout merely because current camera acquisition is deferred.

`survey_candidate_0004` initially had 0.068618944 m static clearance: its
0.06 m nominal radius fit, but its 0.08 m radius-plus-uncertainty envelope
did not. Two later morphology conflicts remained unresolved. The selection
projection moved the target 0.479542373 m into a blocked map cell; the arrival
projection placed it 0.537991062 m from its frozen position, also blocked.
Both clearances are zero under the production map geometry policy. This
concerns the stand target, independently of the robot's approach waypoint.

The original snapshot, registry, and projection records are unchanged,
including all recorded content hashes. Any test adaptation should happen
only in memory, never by rewriting this source evidence.
