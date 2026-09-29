# Radial position recovery fixture

Source: `stand_explore_exact2_camera_all5_20260929T135018Z`, candidate visit
`004_survey_candidate_0005`, first three saved stopped camera/scan tuples.
Snapshot and projection are unchanged arrival artifacts. `observations.json`
retains raw scan/calibration, exact-time transforms and original timestamps;
its corners and projection come from the third frame's detector metadata.

The current range ends near 0.572 m; the visible cluster is near 0.665 m.
Tests replay reconciliation from the original source options. QR corners are
from an offline production-decoder replay of hash-checked frame 3 after
rectification, with its crop offset restored to full-image coordinates.
Synthetic head-quality contracts isolate consumer integration; they are not
claims of an admitted recorded head-axis estimate. No fixture authorizes
motion or replaces the surveyed stand center.
