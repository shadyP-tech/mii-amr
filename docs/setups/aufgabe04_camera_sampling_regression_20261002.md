# Camera inspection sampling regression, 2026-10-02

## Observed failure

Workstation run `stand_explore_exact2_camera_all5_20261002T121410Z`,
candidate `000_survey_candidate_0003`, used commit `ef3ef71`.
The first approach passed camera arrival admission with a calibrated signed
bearing error of -2.394 degrees at 0.521 m range. Camera inspection processed
38 frames, accepted 14 current-head associations, and accumulated seven angle
samples. The measured head angle, including uncertainty, was within the
configured 30-degree front-facing allowance.

The identity decoder never ran on those accepted head observations. A second,
wider LiDAR search found two eligible clusters and blocked the identity crop.
Offline decoding of the 14 accepted head crops on the workstation produced
`QR_003` in all 14. This verifies readable pixels, not a complete replay of
identity admission or mission success. The recordings and raw logs remain on
the workstation.

The resulting `measured_head_front_identity_unresolved` progress was treated
as unavailable geometry. Camera inspection invoked LiDAR recovery, which
selected a +16-degree scan-boundary sampling turn. The actual turn was
+15.844 degrees with about 1.6 mm translation. This was a successful sampling
motion, not a camera-alignment correction: both alignment verification flags
remained false. It changed the camera bearing error to -19.905 degrees, outside
the 10-degree acquisition limit, so the second camera attempt never started.
The canonical target point did not change across the turn.

The inspected run bundle had no terminal mission record. These findings
explain the first candidate's exhaustion; they do not establish the final
outcome of the whole mission.

## Correction boundaries

The concurrent **Review camera exploration gates** task owns accepted-head
identity reuse, usable-observation priority over optional centering, opposite
identity acquisition, and observation approaches to temporarily hidden
candidates. This correction complements those changes:

1. Preserve the explicit unresolved measured-head identity reason when
   creating inspection progress. An estimator diagnostic or advisory angle
   must not erase that distinction.
2. Give that validated progress one additional passive capture at the same
   stopped frame. Bypass optional centering for this retry. If it still does
   not produce usable identity or a valid axis observation, defer the candidate
   locally instead of invoking distance recovery, LiDAR recovery, or generic
   viewpoint search. The retry consumes the existing inspection budget.
3. Remove scan-boundary sampling from the camera recovery factory. The generic
   sampling controller and explicit sampling child retain their independent
   contracts. Camera recovery can still use its existing admitted support or
   alignment routes when geometry is genuinely unavailable.

A successful QR observation or recommendation still completes inspection.
A valid axis receipt, including the existing usable-head plus actual-empty-
decode receipt, retains its opposite-side route. Failure to reach the decoder
does not create such a receipt. No QR outline, new angle threshold, motion
permission, collision allowance, or compensating restoration turn is added
by this correction.

## Verification

The focused regressions cover passive retry and deferral, successful decoding
on retry, certified opposite routing, unchanged view budgets, preservation of
the producer reason, and production camera recovery with a sampling effect
available but never dispatched. Integration checks also cover LiDAR recovery
handoffs, retained target frames, centering, and candidate inspection.

The combined local run passed **267 tests and 51 subtests** across the new
identity-pending and reason suites, candidate inspection and route search,
LiDAR acquisition and handoffs, retained recovery frames, explicit sampling,
observer progress, unidentified-head orientation, current-head identity, and
centering. `git diff --check` passed. This includes the concurrent changes
present at verification time; it is not a full-repository or hardware test.

Changes are local. Hardware validation remains outstanding.
