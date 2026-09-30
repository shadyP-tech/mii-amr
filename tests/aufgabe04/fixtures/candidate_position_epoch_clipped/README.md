# Clipped position recovery fixture

Source: `stand_explore_exact2_camera_all5_20260929T152359Z`, candidate visit
`001_survey_candidate_0001`, stopped camera/scan tuples 3, 4 and 5. Snapshot
and projection are unchanged arrival artifacts. `observations.json` contains
their original raw scans, exact-time transforms, sensor timestamps, camera
calibration, profile hashes and source candidate envelopes. It also records
SHA-256 digests of the original frame metadata and copied arrival artifacts.
`now_sec` is the saved scan timestamp plus the recorded scan age.

The ordinary 15-degree envelope selects only raw beam 6. The broader envelope
contains one complete contiguous cluster at indices 6 through 10. These three
distinct stopped tuples can recover the target without dropping ordinary
returns or ignoring a competing cluster. Tests with changed bearings, ranges
or timestamps explicitly create synthetic negative controls.

`frame_000005.jpg` is the original compressed image from the fifth tuple; its
digest is `image_sha256`. The image test uses that tuple's calibration, image
and scan times and transforms. It does not substitute later images or fitted
corners. Its offline detector clock excludes wall-clock performance claims.
This fixture never authorizes motion or replaces the surveyed stand center;
one image does not establish the seven-frame backside consensus or a complete
mission result.
