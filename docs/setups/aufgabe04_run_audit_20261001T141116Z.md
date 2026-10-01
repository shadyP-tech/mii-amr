# Post-opposite camera audit: 20261001T141116Z

Run `stand_explore_exact2_camera_all5_20261001T141116Z`, started October 1,
2026 at **16:11:16 Europe/Berlin**, workstation alias `mii002` (hostname
`mii001`). All eight recorded bundles use clean revision
`144f95425e36256b2db82b84cc555e6864c726f1`, including the previous endpoint
correction and 10° passive camera acquisition allowance.

The audit copied and hash-verified **638 source files, 64,838,431 bytes**.
A second read-only workstation check at **16:31:17 Berlin** found the same
latest run, clean revision, and unchanged hashes and modification times for
all 638 files. No production code or robot state was changed by this audit.

## Finding

The opposite-side move succeeded. Camera admission then failed **before the
payload decoder was invoked**, because the native QR outline detector could
not provide geometry for the foreground crop. The broad fallback crop
overlapped a projected background candidate, so it was rejected as
`target_crop_overlap_unresolved`.

The intended QR role is **decoded identity only**. The workstation's actual
WeChat decoder reads **`Start` without valid QR corners** from all three
tested post-opposite frames. Its native OpenCV outline detector finds no
outline in those same crops. Thus the native-outline prerequisite prevents
a working identity decoder from running.

This is a different blocker from the previous run's scan-seam ambiguity.
Current target reconciliation becomes ready here. The retained seven-sample
head angle is available with **±3.883272°** uncertainty; no new angle fit is
requested. Neither the 10° acquisition limit nor the 30° front-facing policy
rejects this attempt.

![Original final post-opposite frame](/Users/stephpark/Documents/stephsWorld/mii-amr/results/implementation_checks/run_audit_20261001T141116Z/camera_after_opposite_final.jpg)

## Live causal chain

The post-opposite observer is
`candidates/001_survey_candidate_0001/camera_lidar_attempt_02`.

1. It processes **22 identity-only frames**. Twenty-one pass source freshness;
   one is marginally stale at 0.500450 seconds. Nine additional saved tuples
   do not reach detector processing: eight exhaust exact-TF retry and one
   expires before processing.
2. Three-scan target reconciliation becomes ready in **18 frames**. Its
   position-epoch recovery identifies the actual current five-beam cluster.
3. In **21 frames**, native support reports `no_complete_target_outline`.
4. The fallback identity crop overlaps the projected head of candidate 0003.
   Without supported foreground geometry, the crop has no occlusion proof
   and returns `target_crop_overlap_unresolved`.
5. The returned crop attempt is `None`. The payload-decoder block is skipped:
   no WeChat invocation, no native payload invocation, and no raw-pixel
   identity helper occur in the live post-opposite attempt.
6. The observer finishes after **8.437 seconds** with an advisory
   `inspection_observation`: classification `unobservable`, reason
   `opposite_identity_crop_conflict`, **18 associated progress samples and
   zero QR samples**. It exits normally with code 0; no timeout or crash
   caused this result. The artifact grants no completion or motion authority.

The final recorded frame is still fresh at publication: image age
**0.385034 seconds**, scan age **0.406364 seconds**, below the 0.5-second
limit. TF delays and the one marginally stale processed frame are secondary;
the same crop rejection repeats with fresh, reconciled observations.

## The overlap and current target are concrete

For original frame 31:

| Quantity | Recorded or reproduced value |
|---|---|
| Current LiDAR group | Five beams, indices 206–210, range 0.549000 m |
| Current projected search center | (501.981, 299.198) pixels |
| Expected head height | 106.967 pixels |
| Broad search crop | `[373, 170, 631, 428]` |
| Fallback identity crop | `[416, 213, 588, 385]` |
| Candidate 0003 projected head | `[503, 315, 556, 368]` |
| Projected candidate 0003 depth | 1.399–1.517 m |
| Crop overlap area | 2,809 pixels² |

The search follows the current cluster: an independent production numeric
replay reproduces its projection to within `1.71e-13` pixels. Projecting the
survey center alone would put the head at x=382.92, rather than x=501.98.
The current target displacement is about **9.3–9.8 cm** and is handled by the
existing position-epoch recovery. Its 35° search envelope is an authenticated
recovery envelope, not a relaxation of the ordinary registration limits.

The final image shows a large foreground stand. Other visible QR stands are
outside the recorded search crop. However, the crop validator acts on projected
candidate geometry: it cannot classify the overlapping, deeper candidate as
occluded without a supported foreground region. Simply dropping all neighbor
checks would discard the stand-to-ID association rather than solve it.

## Workstation decoder replay

The audit executed only offline decoder calls against unchanged saved images
inside the same Apptainer image and repository mount as the run. It started
no ROS nodes or robot commands. Runtime OpenCV is **4.5.4**, with the WeChat
backend available. Three relevant module hashes match the local executed
revision exactly.

| Original capture | Native outline, raw/rectified at 4× and 1× | Payload, raw/rectified at preferred 4× and 2× |
|---|---|---|
| 5 | No outline | `Start`, corners absent |
| 16 | No outline | `Start`, corners absent |
| 31 | No outline | `Start`, corners absent |

All **12 payload calls** return `Start`. WeChat's reported points cover the
whole input crop and are correctly rejected as QR geometry (`full_input_extent`);
the decoded text remains valid. This is exactly the distinction required by
the user's ID-only contract.

On the rectified 4× crops, the first WeChat stage obtains the text in
**5.28–6.31 ms**. The shared decoder subsequently spends time searching for
unneeded QR corners, returning the same text-only result after approximately
**105–114 ms**. One separate preferred-2× replay exceeded the cooperative
120 ms budget at 144 ms; individual OpenCV calls cannot be preempted.
These are offline measurements, not a complete live admission replay or a
freshness guarantee for every captured tuple.

A desktop OpenCV 5.0.0 installation without WeChat reads no ID from the tested
pixels. That local limitation does not describe the workstation decoder;
the recorded runtime replay above is the relevant evidence.

## Why the previous correction did not apply

The endpoint correction is deliberately scoped to missing current confirmation
with a certified retained metric center and two bounded endpoint fragments.
This attempt has a single current cluster and successful three-scan
reconciliation, so its `endpoint_confirmation` diagnostic remains empty.
The new original-pixel decoder is also restricted to endpoint-confirmed QR
support; it cannot rescue this ordinary opposite-side path.

The retained source is orientation-only. During the previous backside stop,
frames 5, 10, 11 and 12 had valid current metric-center evidence. At frame 13,
current reconciliation reset because position hypotheses had competing
clusters. Subsequent TF gaps lasted beyond the 1.5-second center opportunity;
the final frame had only one newly collected scan. The observer therefore
published the valid seven-frame angle without a current metric center.
That center was not accidentally lost during the opposite frame projection.

## Correction implied by the ID-only requirement

The payload decoder must not require a native QR outline before it may read
the ID. For this retained-angle branch:

- Read identity from a bounded current candidate/head region using the working
  payload backend; keep QR corners optional.
- Establish which stand owns that region using current head/LiDAR evidence
  and the existing candidate/depth exclusions. Foreground head-frame geometry
  can supply this role without computing a QR pose or refitting the retained
  head angle.
- Once the region is associated and one nonempty ID is decoded, use the
  existing text-only binding and retained orientation. No seven-frame QR
  geometry consensus is needed.
- Give identity-only decoding an explicit early-return path after a valid
  text result instead of spending the remaining budget seeking corners.

`bind_crop_text` already accepts one ID without inspecting QR corners.
The missing capability is the upstream associated-region/decoding path,
which currently depends on the native QR outline to resolve overlap. The
opposite branch returns before ordinary current-head processing, so that
alternative head-region evidence is not currently produced here.

This audit establishes decoder capability and the live blocking gates. It
does not claim that a corrected end-to-end admission artifact has been
implemented or validated. No production changes were made in this turn.

## Subsequent run state

The opposite motion completed at **16:18:31.932**. The post-opposite camera
artifact was published at **16:18:46.154**, and the parent continued recovery.
Four route proposals were blocked; a different route passed dry preflight
at **16:19:26.056**. At **16:19:29.527**, the handoff log records
`KeyboardInterrupt` during pre-run diagnostics, before that recovery motion
started. The artifacts do not identify who initiated the interruption.

Six actual motion legs completed. The copied parent has no sealed end/exit
entry or final mission result; it should not be described as a controller
safety stop or a confirmed ongoing process. The last discovery state is
**1/5, QR_003 only**. No return-to-Start motion occurred.

## Evidence and code anchors

Audit root: `results/implementation_checks/run_audit_20261001T141116Z/`.

- `source_integrity.json`, `source_integrity_recheck.json`: original hashes,
  revision and unchanged-source verification.
- `camera_audit.py/.json`: per-frame outcomes, all 72 saved image hashes and
  byte-identical representative images across the four observers.
- `timeline_audit.py/.json`: motion, camera and interruption chronology.
- `retained_center_audit.py/.json`: retained-center lineage and current-cluster
  geometry replay.
- `pixel_replay.py/.json`: desktop native detector diagnostic.
- `probe_workstation_pixels.py`, `workstation_pixel_replay.json`: actual
  deployed backend, exact image hashes, decoder stages and source hashes.

Relevant functions:

- `real_robot/observer/opposite_identity.py`: endpoint dispatch, support/crop
  acquisition and the payload block guarded by an accepted crop.
- `real_robot/observer/opposite_target_support.py`: native `detectMulti`,
  current-outline filters and supported QR geometry.
- `real_robot/observer/opposite_identity_crop.py`: neighbor overlap and
  text-only `bind_crop_text`.
- `qr_scanning/opencv_qr_detector.py`: provisional text results followed by
  continued QR-corner search.
