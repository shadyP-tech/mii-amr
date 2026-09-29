# Run audit: 20260929T132022Z

The latest run completed both survey legs and all executed candidate motions,
then stopped during selection of the next candidate. **QR_003 and Start were
both facing-ready (2/5)**. This was an uncertainty-admission rejection before
motion, not another waypoint timeout.

The software defect was immediate termination after one rejected stationary
localization window. Initial candidate selection treated uncertainty rejection
as permanent geometric infeasibility and retired all three remaining candidates.
The correction gives this specific failure one fresh, complete stationary
localization and planning attempt before stopping.

## Recorded evidence

- Run: `stand_explore_exact2_camera_all5_20260929T132022Z`.
- Parent and executed child bundles record clean `main` at
  `b9bf375ae25f6f919339fbf78727d8671ef59c36`, including the previous deadline fix.
- The last executed opposite-face approach completed at 13:28:49.813790 UTC
  in **55.950 seconds**, with 1.269286 m estimated travel.
- Final selection failed at **13:29:24.533818 UTC / 15:29:24.533818 CEST**.
- `mission_failure.json` reports
  `no geometrically reachable camera candidate route passed uncertainty admission`.
- All three route-time budgets were accepted. No third candidate approach was
  dispatched after the selection failure.

| Remaining candidate | Minimum uncertainty margin | Limiting geometry |
| --- | ---: | --- |
| `survey_candidate_0002` | −0.017016 m | First approximately 5 mm interval |
| `survey_candidate_0004` | −0.313818 m | Final segment, far from the localization reference |
| `survey_candidate_0005` | −0.214128 m | Final segment, far from the localization reference |

The common planning start was `(-1.627347, -0.368245, -0.158989)` in map
coordinates. The nearest candidate's limiting interval had **0.340168 m**
clearance but required **0.357184 m**. The requirement comprised 0.190 m of
robot radius, tracking, drift, collision, and braking reserves, plus 0.148498 m
of planar localization uncertainty and 0.018686 m of heading uncertainty.
The rejection was correct for the admitted evidence.

The five stationary samples spanned 13:29:04.450759–13:29:06.753271 UTC
(2.302512 seconds). Position variance decreased from 0.005512890 to
0.004296115 m²; yaw variance decreased from 0.007218076 to 0.005452006 rad².
Admission correctly used the conservative envelope over the entire window.

As a sensitivity calculation only, using the final sample's uncertainty with
the unchanged recorded route makes all 254 sampled intervals for candidate
0002 pass, with a minimum margin of +0.002838 m. The other two routes still
fail. This suggests another complete stationary acquisition could help; it
does **not** authorize discarding earlier samples, reusing the final sample as
fresh evidence, or predicting mission completion.

The run itself contains a successful example of stationary recovery: an
opposite-face approach initially failed uncertainty admission, obtained fresh
localization through the existing opposite-face retry, then passed dry and live
admission and completed. Initial candidate selection lacked that opportunity.

## Root cause and implemented correction

At the recorded revision, `real_robot/candidate/approach.py:2291` caught
`NoUncertaintyAdmittedCameraCandidateError` together with static
`NoFeasibleCameraCandidateError`. It immediately marked every eligible UID
`no_feasible_route`, removed those UIDs from the unresolved set, and raised.
The later child route-admission retry ledger could never see this failure,
because no candidate had been selected or dispatched.

Git history attributes this combined exception handling to `fb1eedb0`
(September 7) and eligible-candidate retirement to `03111cdf` (September 8).
The latest deadline correction did not introduce this behavior.

The local correction adds at most **one** refresh for the typed uncertainty
failure when the required live localization effects are available. Each attempt
performs the complete existing sequence:

1. Admit a new stationary pose and map/odom transform.
2. Reproject the original frozen candidate registry into that planning frame.
3. Load uncertainty from the new complete stationary sample window.
4. Replan and strictly admit the same eligible candidates.

The first rejection does not retire candidates or dispatch motion. Separate
epoch paths preserve both localization/projection records, and selection events
preserve the rejected evidence. The successful frame is carried through the
child handoff and observation. A second rejection stops with
`route_admission_exhausted`; static infeasibility remains `no_feasible_route`.
Malformed localization/readiness failures remain terminal.

No covariance samples are filtered, no safety margins or motion limits are
relaxed, and ordinary child dry/live checks remain authoritative.

Validation: **135 tests and 87 subtests passed** across candidate orchestration,
frame projection and readiness, uncertainty admission, route deferral, and
startup/opposite/runtime recovery. Seven new regression tests cover fresh
pose/covariance/reprojection inputs, preservation of the first rejection,
winning-frame motion and camera handoff, first-attempt success, bounded
exhaustion without motion, static infeasibility, malformed readiness, and the
unbound legacy path. A separate read-only review found no remaining issues.

## Reproduction and scope

Evidence is under `results/implementation_checks/run_audit_20260929T132022Z/`.
Seven copied source files were verified against workstation SHA-256 hashes.
`analyze_uncertainty_failure.py` exactly reproduces all three recorded failures
with the production budget evaluator and writes `audit_summary.json`, including
the explicitly counterfactual sensitivity calculation.

The principal sources are `candidate_selection.jsonl:17`,
`candidate_goal_progress.json`, `station_segment_runs.csv`, and
`preflight/candidate_selection_002_localization.json` within the copied mission
directory. The workstation was read only; the correction is local and has not
been deployed or exercised on the robot.
