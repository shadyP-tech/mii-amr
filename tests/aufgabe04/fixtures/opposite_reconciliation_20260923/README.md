# Opposite-side reconciliation regression

Recorded run `stand_explore_exact2_camera_all5_20260923T113013Z`, selected candidate
`survey_candidate_0001`. Opposite inputs are fresh tuples 000002–000004 and the
unaltered compressed JPEG from tuple 000004. Source-side evidence is tuple
000007 and its committed backside receipt. Projection and snapshot files retain
the recorded geometry, provenance and hashes.

The source receipt is augmented with the reconciliation and metric head proof
that were present in that exact tuple's diagnostics but not handed to planning
by the deployed revision. Tests remap local file references and rehash those
path-bearing projection wrappers; sensor stamps, scans, images, head fits,
candidate populations and frame transforms are unchanged.

Tests exercise the production opposite observer with deterministic decoder output,
plus an independent native decode of the recorded search crop. Native OpenCV 5
can decode `Start` at 4x there but not in the rectified isolated quad; deployed
WeChat support remains covered by the earlier overlap fixture and requires a
real-run confirmation. This fixture makes no claim of end-to-end hardware success.
