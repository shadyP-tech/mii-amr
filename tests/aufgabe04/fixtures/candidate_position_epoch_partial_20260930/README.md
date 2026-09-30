# Partial candidate envelope with an ambiguous camera angle

Source: `stand_explore_exact2_camera_all5_20260930T125903Z`, candidate visit
`002_survey_candidate_0005`, camera attempt 00. The source run used revision
`e018878a583131f5dc051c0ea7ce777df91429cf`.

`candidate_snapshot.json` and `candidate_frame_projection.json` are byte-for-byte
copies of the `alignment_01/arrival/arrival_frame_projection` artifacts.
`observations.json` contains the original raw scans, timestamps, exact-time
transforms, camera calibration, original candidate envelopes and profile hashes
for stopped frames 1–3 and 29–31. `source_sha256` records the copied arrival
artifacts and original full frame-metadata digests. `now_sec` is reconstructed
as the saved scan receipt time (`scan_received_ros_sec`) plus the preliminary
LiDAR association's recorded `scan_age_sec`, matching production `_scan_age`.

Frames 1–3 alternate between two and three ordinary-envelope beams. All are
strict subsets of the same unique broader cluster. They establish the first
three-scan recovery without throwing away the three-beam observation. Frames
29–31 provide the independent later proof used with the saved camera image.

`frame_000031.jpg` preserves the original compressed bytes, checked against the
recorded image digest `f824d1ec22b88de55e47e05cfc229cc84d4e3b90511d52eb8aea3f8c311ded86`.
The regression derives corners and QR observations from this image, rather than
supplying fitted geometry. It retains the detector's rejected single angle,
valid bounded head detection and same-scan target association. The resulting
centering advisory is replayed through stopped-frame admission and publication
with zero admitted axis samples. It does not certify perpendicular alignment.

The offline detector clock excludes wall-clock performance claims; historical
sensor-age checks remain active. The fixture grants no motion authority and
does not replace surveyed candidate geometry. A scan-safe first turn deliberately
retains residual image offset and requires a fresh stopped observation.
