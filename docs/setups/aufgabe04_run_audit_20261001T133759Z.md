# Real-run audit: 20261001T133759Z

Run `stand_explore_exact2_camera_all5_20261001T133759Z`, October 1, 2026,
**15:37:59–15:51:29 Europe/Berlin**, workstation SSH alias `mii002`.
The parent and all eight motion bundles record clean revision
`915250ef42d5e7fa9e80a0127c761a13b8baaa72`. The opposite-side endpoint
correction in local commit `cad847b` was **not deployed** for this run.
All **1,186 copied source files**, totaling 170,264,845 bytes, were verified
against workstation SHA-256 hashes. The source directory is unchanged.
A second read-only check at **16:03:17 Berlin** found the same latest run and
clean revision, with all 1,186 source hashes and modification times unchanged.

## Main finding

The mission exited with code **2**, reporting `candidate_qr_goal_incomplete`:
it confirmed **QR_001, QR_002 and QR_003**, but required five identities.
All eight motion legs completed normally. There was no controller safety
stop or parent interruption.

Two visited candidates failed the stopped **camera acquisition bearing**
check before an observer could start:

| Candidate | Optical bearing error | Acquisition limit | Excess | Base-to-target range |
|---|---:|---:|---:|---:|
| `survey_candidate_0006` | −6.088224° | ±6° | 0.088224° | 0.514985 m |
| `survey_candidate_0001` | +7.019018° | ±6° | 1.019018° | 0.530401 m |

Both passed range and static-map target admission. Both receipts explicitly
record `observer_started: false`; neither has a camera capture from that
arrival. Their identities cannot be confirmed from this run. Their locations
are near previously observed QR_004 and Start, respectively, but that is a
geometric inference, not current QR evidence.

This bearing is **where the camera points relative to the candidate**. It is
different from the head plane's front-facing angle and its allowed 30°
deviation. Neither failed arrival reached head-angle estimation, QR decoding,
or opposite-side identity processing. Deploying the previous opposite-side
correction alone would therefore not remove this run's immediate blockers.

## Why navigation succeeded but acquisition failed

The controller checks its fixed route goal with **30 mm position tolerance
and 3° yaw tolerance**. It does not continuously aim the calibrated camera
at the candidate after subsequent frame reprojection.

| Candidate | Stopped position error to executed goal | Stopped yaw error to executed goal |
|---|---:|---:|
| `0006` | 25.470 mm | −2.54970° |
| `0001` | 24.972 mm | +2.33213° |

These values satisfy the controller's contract. The independent arrival check
then measures bearing against the candidate in the fresh map frame and
includes the camera mount's displacement and orientation.

There is also a planning/execution-frame contribution. Mapping the original
planned targets through the later execution certificates puts them
**42.160 mm / 27.597 mm** from their canonical odometry targets. Even exact
attainment of the executed goals would leave base-bearing errors of
**−4.84193° / +3.01646°**. The stopped residuals produce base-bearing errors
of **−7.05474° / +4.92606°**; camera calibration produces the optical errors
in the first table. The decomposition uses stopped localization and execution
certificates, not the last control-trace row, which precedes completion.

Offline replay with the authentic sealed calibration reproduces all five
initial arrival decisions and their measurements within `1e-12`. The
replayed geometry functions are verified against the executed revision.

## Why the candidate was abandoned immediately

The passive acquisition allowance was coupled to `MAX_CENTERING_STEP_RAD`,
the **6° maximum for one corrective turn**. This incorrectly made the size
of one motion step the limit for even looking at an off-center target.
The existing centering mechanism already supports two turns and 12° total
travel, acquiring fresh evidence after each turn.

At the first calibrated arrival, `inspection_adapters.py` calls plain
`admit(...)`. It bypasses the older `admit_corrected(...)` loop and raises
the arrival error before entering the inspection state machine. Consequently
the current-head centering and alternative-view recovery never get a chance
to inspect these two arrivals.

The parent does not classify this arrival rejection as retryable. With one
candidate observation episode allowed, it records `inspection_exhausted`.
That label does **not** mean that eight camera inspection views were tried;
no camera process started for either candidate.

## Other events and candidate selection

| Time, Berlin | Event |
|---|---|
| 15:42:55.414 | QR_003 observation committed. |
| 15:45:34 | Initial QR_002 observer reaches its 90-second deadline and receives SIGINT. |
| 15:47:18.876 | A different inspection view successfully commits QR_002. |
| 15:48:55.126 | Candidate 0006 rejected at arrival, before camera processing. |
| 15:49:57.471 | QR_001 observation committed. |
| 15:51:28.881 | Candidate 0001 rejected at arrival, before camera processing. |
| 15:51:29 | Parent exits with the 3/5 identity shortfall. |

The earlier QR_002 timeout recovered successfully and was not the terminal
cause. Its first view shows the real stand from an oblique rear/side angle.
Among 362 processed frames, 337 lacked a usable head border; the four
backside samples were separated by 9.22–23.96 seconds and could not supply
seven samples inside the five-second consensus window. The alternative view
committed QR_002 after two processed frames. The initial capture history
reached its 256-image limit, so its 408 submitted captures are not all saved;
the status/event stream still records the 362 processed outcomes.
QR_001 successfully used the approximate front-facing path with seven frames
and a retained **±17.233°** orientation bound; the 30° policy was operational.

![Initial oblique QR_002 view](/Users/stephpark/Documents/stephsWorld/mii-amr/results/implementation_checks/run_audit_20261001T133759Z/camera_qr002_initial.jpg)

![QR_002 after inspection recovery](/Users/stephpark/Documents/stephsWorld/mii-amr/results/implementation_checks/run_audit_20261001T133759Z/camera_qr002_recovery_committed.jpg)

Two further candidates, 0004 and 0007, were initially excluded correctly:
their current projected static clearance was only **16.75 mm / 20.54 mm**,
less than the stand's **60 mm nominal radius**, before adding uncertainty.
Candidate 0004 also had an unresolved morphology conflict and never became
eligible in the saved projections. Those wall safeguards should remain.

A separate limitation affects 0007: its later projected clearance increased
to **103.22 mm / 124.36 mm**, which passes the existing 80 mm requirement.
However, initial deferral had removed it from `unresolved`, and subsequent
selection only considers that reduced set. It was never reconsidered.
This is a lost inspection opportunity, not proof that it is a real stand or
that observing it would have met the QR goal. A future correction can
reconsider unvisited, static-only deferrals on fresh geometry under a bounded
budget while retaining all wall and morphology checks.

No opposite-side branch or return-to-Start motion occurred in this run.

## Requested bearing correction

Following the user's request for approximate rather than perfect aiming,
passive acquisition now has an independent **10°** bearing allowance when
camera centering capability is available. Both recorded failed arrivals fit
this allowance. They enter as `acquisition_only`, with live target association
still required; they are not declared centered or front-facing.

The strict 3° arrival metadata, 30° front-facing policy, range limits, static
target admission, camera/LiDAR association, freshness checks, and motion
limits remain unchanged. Each measured centering turn remains capped at 6°,
with two turns and 12° total travel. The calibrated map-based correction loop
has not been enabled as part of this tolerance change.

The correction is local and has not been deployed or exercised with robot
motion. The two failed arrivals have no recorded camera frames, so this audit
can verify removal of their acquisition rejection, not successful QR decoding
or a completed five-stand mission.

Focused verification passed **162 tests and 113 subtests** across nine arrival,
centering and target-admission modules, with no failures or skips. Regressions
use both recorded poses and the authentic calibration, test the inclusive 10°
boundary and rejection beyond it, and retain a wall-target rejection at 7°.
`git diff --check` also passed.

## Reproducible evidence

Audit root: `results/implementation_checks/run_audit_20261001T133759Z/`.

- `source_integrity.json`, `source_integrity_recheck.json`: workstation
  revision, source hashes and unchanged-file recheck.
- `source/`: unchanged mission artifacts and nine real-run bundles.
- `timeline_audit.py`, `timeline_audit.json`: termination, motion, candidate
  outcomes and source-hashed chronology.
- `arrival_replay.py`, `arrival_replay.json`: original acquisition decisions
  and the 10° passive-acquisition comparison using the sealed calibration.
- `controller_arrival_decomposition.py`, `.json`: stopped goal errors and
  planning/execution-frame bearing contributions.
- `static_population_audit.py`, `.json`: reproduction of all 17 recorded
  candidate decisions and all seven candidates across five selection frames.
- `camera_audit.py`, `camera_audit.json`: per-attempt camera outcomes and
  original-image export hashes.
- `acquisition_correction_pytest.txt`: focused correction verification.

Implementation anchors:

- `navigation/approach/candidate_arrival_admission.py`: independent passive
  acquisition bearing policy.
- `real_robot/candidate/approach.py`, `_camera_arrival_decision` and
  `_admit_camera_arrival_geometry`: calibrated arrival and acquisition gates.
- `real_robot/candidate/inspection_adapters.py`, `admit_corrected` and initial
  `admit`: recovery ordering.
- `real_robot/observer/candidate_centering.py` and
  `real_robot/candidate/centering_execution.py`: measured, bounded turns.
- `real_robot/candidate/approach.py`, `defer_ineligible_targets`: persistent
  removal of initially excluded candidates.
