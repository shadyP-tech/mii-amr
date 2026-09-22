# Next backside improvement: preserve observable framing before centering

## Conclusion

Prioritize camera framing that accounts for the LiDAR scan boundary. In the
recorded second-candidate backside view, the classifier and angle solver were
already working. Automatic image centering moved an admitted head into the
scanner's endpoint ambiguity. Preserve a complete, uniquely associated current
view while it accumulates axis evidence; veto a centering destination that puts
its association cone across an unreliable scan boundary.

The investigation below led to the bounded implementation described at the end.
No robot commands were issued. It builds on run `stand_explore_exact2_camera_all5_20260922T121851Z`, candidate
`001_survey_candidate_0001`, first local view `camera_lidar_attempt_00`.
Reduced window lighting was already in place.

## Recorded chain of events

| Evidence | Before centering | After centering |
| --- | --- | --- |
| Head center | u=353 px | approximately u=399–403 px |
| Head border/pose quality | accepted | accepted in 12 of 13 images |
| Backside classification proof | accepted, confidence 0.964 | accepted in all 12 valid head fits |
| Camera-relative head yaw | -15.30 degrees | approximately -16.25 to -22.78 degrees |
| Associated scan indices | one cluster `[0,1,2,3]` | one unique scan; 11 split endpoint cases |
| Fitted head bearing in scan | +2.676 degrees | -0.806 to -0.484 degrees |
| Axis samples accumulated | 1 before observer exits to center | 1 of required 7 |

The original head occupied approximately x=306–400 and y=241–342 in an 800x600
image. Visual inspection confirms a complete backside, with ample image margin.
The observer accepted current raw borders, explicit marker absence and a unique
LiDAR cluster. Nevertheless, `candidate_centering_committed` ended this observer
after its first accepted axis sample.

The centering solver requested +3.8583 degrees. The recorded controller completed
+3.6036 degrees in 3.183 seconds. Replaying the existing solver with the saved
advisory's calibration and range reproduces the requested angle exactly.

The destination is hard-coded to `intrinsics.width_px / 2`, with a one-degree
pixel deadband in `observer/candidate_centering.py:97`. It does not assess the
scan boundary or the value of remaining at an already observable pose.
`observer/node.py:3073` commits a centering advisory before evaluating unresolved
inspection progress. The newly implemented five-second advisory opportunity
therefore does not suppress this earlier centering action. QR grace protects a
bound QR view, but cannot protect a QR-free backside this way.

## Why waiting for witnesses alone is insufficient

The rejected scans have indexed endpoint gaps of 1.410–1.787 sampling steps,
with inconsistent reported endpoint metadata. Original raw groups therefore
remain separate. The existing temporal proof correctly requires three distinct
continuous scans that independently witness the missing interval.

A deterministic replay of the twelve saved camera-selected scans reproduces
all raw acceptance decisions. Frame 2 seeds one real witness. Frames 3–7 retain
it while the current endpoint fragments remain geometrically compatible. At
frame 8 the witness-to-current interval exceeds `MAX_SCAN_GAP_SEC=1.0`; the partial
proof rejects and history clears. Later fragments do not become new witnesses.
Thus the visible failure is not a blanket reset on every fragmented frame.

The replay is incomplete for the independent scan stream: the observer received
37 scans, but only the camera-selected scans are preserved in capture history.
Six witnesses expired before exact TF became available. Existing telemetry does
not identify the successful ingestion, registration or rejection of every other
scan. We cannot conclude from these artifacts that every missing witness was
caused by topology, or that none was lost through TF/context handling.

Two diagnostic gaps deserve a small separate correction:

- `scan_witness_collection.py:83` ignores the boolean result of `ingest_scan`;
  rejected ingestion has no dedicated counter or retained rejection event.
- `scan_target_persistence.py:578` replaces metadata with a generic failure;
  the partial-proof exception that clears history at line 589 is not retained.
  Witness counts, source stamps and the reason for clearing them are absent.

A longer timeout cannot create continuous observations if the target remains
on an unreliable boundary. Relaxing the seam checks or counting repeated
fragments as witnesses would change the proof, not solve acquisition.

## Recommended bounded implementation

1. **Preserve productive observations.** Give fresh, complete, uniquely
   associated backside geometry an opportunity to reach the existing seven
   samples before optional image centering. A 47-pixel center offset with
   accepted border and pose quality is not itself a reason to abandon the view.
   Keep the overall observation deadline and motion/conflict reset behavior.
2. **Check the proposed framing against scan topology.** A pure helper should
   use the measured head, camera/scan extrinsics, associated range, original
   scan endpoint geometry and the existing narrow cone to evaluate current and
   proposed framing. Reject centering that introduces endpoint dependence when
   current framing is usable. If a new framing target is necessary, choose the
   smallest valid adjustment within existing turn/travel limits and full-head
   image margins. Reacquire all evidence after movement.
3. **Record the witness lifecycle.** Persist bounded scan-only events for TF
   pending/expired, ingestion accepted/rejected, uniquely registered witness,
   fragment-only observation, witness count/age and exact history-reset reason.
   Include selected raw indices and topology diagnostics, without duplicating
   unbounded scans in status. Keep source-time TF and freshness mandatory.

Keep these responsibilities separate: a pure framing policy, observer
completion scheduling, and witness diagnostics. An alternative centering target
must be represented and recomputed in the hashed advisory and permit validation;
it cannot be an unverified pixel override. Retain three independent witnesses
where temporal seam proof is needed, seven axis samples, the current range and
bearing bounds, and existing motion permits.

Calibration-only projection checks illustrate the available room: from the
initial saved view, a hypothetical -3-degree turn projects the center near
u=316 and the physical scan point near +5.51 degrees; -6 degrees projects u=279
and +8.34 degrees. These are geometric predictions, not new scans or recommended
blind turns. The lowest-risk first change is to preserve the original valid
view. The recording contains only one frame at that original pose, so it cannot
prove that staying there would have guaranteed consensus.

## Border disambiguation remains a separate next step

The later viewer recording `recording_20260922_143442_350245680` contains a
front-facing QR scene, not this backside observation. Its overlapping rectangle
alternatives, and the separate closer-view first-candidate audit, establish a
border-selection problem in other conditions. They do not establish border
selection as the limiting factor in this backside sequence: twelve current
backside classifications already passed.

After the framing issue, improve scale-aware outer/inset edge discrimination
using current border evidence and measured geometry. Preserve genuine multiple
heads and unresolved planar orientation. Do not select the largest rectangle
or use absence of decoded QR as sufficient backside evidence.

## Reproduction and checks

`results/implementation_checks/backside_next_improvement_20260922/analyze.py`
replays saved association/persistence inputs and the actual centering advisory.
`analysis.json` stores the results and projection-only counterfactuals. Run from
the repository root with `PYTHONPATH=.` and Python supporting the repository.
No new scan samples, angles or witness timestamps are fabricated.

Before implementation, targeted regression suites passed: 85 tests and 95 subtests covering
centering, observer centering, backside classification, scan persistence,
endpoint persistence and independent witness collection. These checks validate
the inspected behavior before the implementation below.

## Implemented policy and validation

`observer/inspection_framing.py` owns two motion-neutral policies:

- The first accepted axis sample starts one fixed five-second opportunity per
  stationary candidate/calibration context. Optional centering waits during
  this opportunity; misses and further accepted samples do not renew it.
  Stronger completion paths still run and the parent deadline is unchanged.
- The proposed, already bounded centering step is projected with calibrated
  camera/base/scan extrinsics and current range. Both the depth-aware point
  and camera ray must clear the original indexed scan boundary by the existing
  association cone plus half the endpoint gap plus one angular sample. This
  applies even when one current scan has valid circular metadata. Invalid
  full-rotation geometry vetoes advice; linear scanners retain existing policy.

The policy only vetoes existing centering advice. It does not generate the
alternative turn targets discussed above, change hashed advisory/permit fields,
or weaken classification, scan continuity, freshness or consensus requirements.
`candidate_centering_receipt.py` applies these policies before staging advice.

`observer/scan_witness_diagnostics.py` retains counters and at most 32 events,
with capped raw index summaries. Collection distinguishes TF delay/expiry and
accepted/rejected ingestion; persistence records unique witnesses, fragments,
proof acceptance, pruning and history reset reasons. Observer status exposes
`scan_witness_diagnostics`. Preview evaluation clones diagnostics so it cannot
alter live counters/history. No raw image or full scan arrays enter this trace.

The portable fixture `tests/aufgabe04/fixtures/backside_framing_20260922.json`
contains the original advisory, initial search metadata and actual post-turn
scan/context inputs from frames 2–8, including source frame SHA-256 values.
Regression replay verifies that the original 3.85828665-degree turn is vetoed,
one real witness remains one witness, and frame 8 reports the gap-related
history loss. This is not evidence that staying at the original pose guarantees
consensus: the full independent scan stream was not recorded.

The local Python 3.14 / OpenCV 5.0.0 checks cover framing, fixed deadlines,
preview isolation, witness continuity, QR completion, camera processing and
centering permit/runtime/child validation: 186 tests and 140 subtests pass,
with the one separately reproduced baseline failure deselected. The existing historical-image test
`test_subsequent_current_marker_uncertainty_cannot_inherit_backside_hint`
fails identically on untouched HEAD `bab57a20687ccdde1f225cb7b7ecc7662c180d27`
in this environment (`hint_retained` is true). Its assertion was preserved.
These changes still require a real-run validation; no new CLI flags are needed.
