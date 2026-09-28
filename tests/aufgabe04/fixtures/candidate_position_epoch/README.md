# Candidate-position epoch recovery fixture

Source mission: `stand_explore_exact2_camera_all5_20260928T130911Z`, physical
view `candidates/003_survey_candidate_0005/camera_lidar_attempt_00`.

- `candidate_frame_projection.json`: unchanged arrival projection artifact.
- `candidate_snapshot.json`: unchanged projected camera candidate snapshot.
- `observations.json`: sensor, timestamp, and exact-time TF fields from capture
  frames 100, 102, and 104. Also includes the refined head corners of the first
  usable frame in later viewer recording `recording_20260928_152428_945201347`.
- `late_run_frame.png`: unchanged saved late camera image from this mission,
  already downloaded during the audit. It is not the image paired with scan 104.

The three-scan replay proves bounded association of current LiDAR evidence with
the selected candidate. The saved-image test proves acquisition of the visible
blue head rather than the radiator. The viewer-corner replay checks calibrated
turn geometry. Neither image/corner replay is claimed as a synchronized,
end-to-end camera/scan replay. QR receipt tests explicitly use a synthetic symbol
at the visible head center and do not claim a recorded decode.

The exact matching compressed camera frame could not be retrieved during
implementation: mii001's SSH host key changed. Host verification was preserved.
No fixture was obtained by bypassing that check; no robot motion was performed.
