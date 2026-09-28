# Opposite-view retained-center regression

Source: `mii002`, real run `stand_explore_exact2_camera_all5_20260928T142407Z`,
second visited stand `001_survey_candidate_0001`, camera attempt 01.

`inputs.json` retains the exact image/scan timestamps, camera calibration, scans,
TF samples and original search results for frames 10, 24, 100 and 200.
`frame_000024.jpg` is the original compressed camera payload, without re-encoding.
The other JSON files preserve the preceding backside receipt and its source,
arrival and canonical candidate snapshots/projections. The fixture helper
rebuilds local paths and their content hashes without changing recorded geometry.

Frame 24 contains one unique current cluster. The corrected bearing/range admits
its complete foreground `Start` QR while retaining the certified backside angle.
Frames 10, 100 and 200 retain unresolved scan-boundary ambiguity when the complete
range envelope is used; they must not be silently joined or admitted.

The deployed OpenCV 4.5.4 decoder was checked separately in a read-only container
replay. Local producer tests mock payload decoding only; they run real outline,
range, competing-candidate, retained-source and receipt validation.
