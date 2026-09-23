# Camera facing geometry optimization — 2026-09-23

This change addresses the missing facing geometry in
`stand_explore_exact2_camera_all5_20260923T124047Z`. Discovery completed with five
QR identities, but Start, QR_004 (blue), and QR_001 (green) had no facing receipt.
See the [recorded-run audit](aufgabe04_facing_geometry_audit_20260923T124047Z.md).

## Retain the measured backside geometry

After candidate-associated QR discovery, the candidate workflow can now build a
separate schema-4 facing recommendation from the retained bounded backside
orientation and validated measured center. It preserves the angle interval,
sample count, center uncertainty, current QR timestamp, and immutable source
receipt chain. It performs no additional camera capture or angle fit.

The recommendation is validated against the current stopped pose, original
candidate keepout, measured-center keepout, bounded-angle endpoint requirements,
and continuous route clearance. The frozen candidate snapshot remains unchanged.
Successful validation adds one facing record through the normal decision path.
Missing proof or failed pose/route validation preserves QR-only discovery and
writes `retained_facing_status.json` with the rejection reason. The original
QR-only receipt remains on disk and retains its discovery-only meaning.

The existing `--final-facing-offset-m` controls the proposed endpoint. No new
command-line options are needed. This optional geometry validation happens
after QR decoding; it introduces no new geometry-acquisition grace period.

The implementation is separated into artifact construction/validation
(`artifacts/retained_facing.py`) and candidate-workflow integration
(`real_robot/candidate/retained_facing.py`). It depends on the current bounded
observer handoff retaining the complete center/orientation evidence.

## Reduce repeated border work

`CurrentBorderFamilies` reuses raw-pixel rail signatures and ordered rail
comparisons within one current frame. `current_rail_profiles.py` batches the
existing five bilinear color cross sections. Texture coverage examines exact
duplicate corner sets only once. Distinct measurements remain distinct.

The 1.5-pixel proximity rule, 70% raw support overlap, single-stripe proof,
multiple-ridge veto, strict verification budget, and freshness deadlines remain
unchanged. No earlier frame's pixel evidence is reused.

Local paired replay of the saved blue and green images reduced median cold-head
acquisition time by approximately 9–10% (blue: 167.7 to 153.1 ms; green: 110.2 to
99.2 ms). Both versions
made the same acquisition decisions. These are host-side image replay timings,
not ROS end-to-end latency or real-robot measurements; the workstation uses a
different OpenCV version. Raw paired measurements are saved under
`results/implementation_checks/facing_optimization_20260923/benchmark.json`.

## Validation and limits

Recorded image fixtures include camera intrinsics, bounded candidate search,
image hashes, and provenance. Regression coverage checks cache isolation,
unchanged blue ambiguity, nonzero green orientation uncertainty, the 12-fit
verification limit, and 1,000 cross-section decisions against the prior scalar
implementation. Retained-facing tests cover immutable receipt binding,
tampering, invalid face structure, measured-center route validation, blocked
routes preserving QR discovery, and parent-workflow bookkeeping. The final
focused suite passed 221 tests and 79 subtests.

The broader Aufgabe 04 suite reported 4,132 passing tests and 71 failures
(including subtest failures). Every failing test node also failed when rerun in
an isolated checkout of baseline `6c0734cedb6e3b104c592fc7e7693f25a8d99664`.
Failures include sandbox Unix socket restrictions and existing fixture/layout
expectations. The comparison is saved in `baseline_comparison.json` beside the
benchmark output.

The blue frame still has unresolved border alternatives and remains rejected for
facing geometry. Green produces a bounded orientation locally; neither this
replay nor the faster processing guarantees workstation acceptance within its
live deadline. These changes do not guarantee facing geometry for all stands.

Schema-4 records provide validated exploration-stage facing geometry. Automatic
logistics arrival-catalog promotion currently accepts only its existing
current-head or seven-frame consensus policies, so it continues to reject these
new retained-angle records. Supporting that separate consumer requires explicit
common-frame projection of the provenance and measured-center clearance; this
change does not bypass that check. No facing motion is authorized by this record.

No robot execution, workstation deployment, or hardware timing validation was
performed for this implementation.
