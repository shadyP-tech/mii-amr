# Inspection proposal 004: complete QR, clipped head

Stopped run `stand_explore_exact2_camera_all5_20260930T135242Z`, candidate
`004_survey_candidate_0001`, `camera_lidar_attempt_02`, capture 27. The three
JSON source artifacts are byte-preserved; `capture_context.json` selects the
same frame's TF samples and records their artifact hashes. Raw scans are in
the QR receipt's three-frame reconciliation proof. Only proof paths are
rebound in memory by tests, preserving the referenced content hashes.

The recorded QR center is (726.1566, 269.7904) in an 800×600 image, despite
the arrival target projection near x=380.2. No complete candidate head or
stand angle was admitted. These fixtures test current QR/scan framing and
the observer handoff, not image decoding, angle estimation, or physical
motion. No image is required or synthesized. The recorded capture completed
QR-only discovery; the regression must instead publish bounded centering
advice first when all current framing gates pass.
