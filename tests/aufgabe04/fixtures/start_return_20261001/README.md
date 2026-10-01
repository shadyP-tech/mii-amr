# Recorded Start return, 2026-10-01

This fixture reproduces the exact Start-return geometry from
`stand_explore_exact2_camera_all5_20261001T104618Z`.

`candidate_snapshot.json` is a byte-for-byte copy of the recorded return-leg
snapshot, including all six candidates, their original ancestry, and full
0.34 m keepouts. `inputs.json` extracts the source/current frame admissions,
the saved Start record, measured stand center, all source candidate geometries
and current reprojection proofs, physical clearance, and recorded failure.
Each contributing original file is identified by repository-relative path,
SHA256, and byte count. Extracted values are not rounded or altered. The
versioned original map YAML/PGM are reused and checked against those hashes.

The original coverage plan contains about 300 KB of survey cell lists. Only
its actual config, arena bounds, map identity, and scalar metadata are copied;
the test constructs a `CoverageSurveyPlan` with empty unused survey arrays.
This is explicitly a projection for the return planner, not a new claim about
survey coverage or the original full-plan content hash. The two fixture files
are bound as source artifacts when creating test route evidence.

The saved Start pose is reprojected through the two original `map <- odom`
transforms. Its exact distance from the Start candidate stays
0.3417084174064008 m, outside the original 0.34 m circular keepout by only
1.7084174064 mm. The current 5 cm goal cell lies on the seven-cell raster
keepout; the earlier frame's goal cell was free. The original run failed with
`exact stored target is blocked; goal snapping is forbidden` and published no
return motion.

Regression tests require the corrected full route to preserve the exact
reprojected target and yaw, retain all circular keepouts and the additional
measured-center keepout, and pass independent static/continuous geometry and
artifact validation. They do not claim a live localization uncertainty budget,
controller execution, physical arrival, or motion authorization. Planner
outputs are created only in temporary test directories.
