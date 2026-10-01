# Recorded coarse-front angular policy fixture

`inputs.json` contains the exact `model.head_orientation_bounds` objects from
the first seven consecutive metadata records (frame indices 0–6) in the
2026-10-01 `recording_20261001_131416_741681612` viewer recording. Seven distinct
source timestamps span 0.7998490333557129 s. No records are skipped or repeated.
Both recorded IPPE alternatives, their engineering three-sigma allowances,
corner coordinates, model/calibration values, and original bounds binding
hashes are preserved without rounding. No images are checked in.

Provenance records the complete original metadata file SHA256, each exact
metadata line SHA256 (including its newline), and original PNG SHA256. Original
files remain in `results/implementation_checks/run_audit_20261001T104618Z/source/`.
To reproduce the extraction, parse lines 1–7 of the recorded metadata JSONL;
copy each `model.head_orientation_bounds` object and recorded frame identifiers
verbatim, then call production `enclose_intervals` on their center/half-width
pairs. The enclosing half-width is 19.78878162939927 degrees. It retains every
alternative and does not divide uncertainty by sample count.

The viewer had QR decoding disabled, and this fixture contains no authenticated
mission target/TF timeline. Seven distinct images establish seven recorded
geometry inputs, **not** seven admitted mission observations. Tests exercise
the pure angular policy with explicit coarse-front permission; they do not
claim the recording has earned that permission or authorize movement.

The hypothetical endpoint is 0.35 m from the interval midpoint-normal with
0.03 m terminal reserve. The 0.02 m candidate-center uncertainty is copied from
the QR_001 mission candidate arrival snapshot, with its source SHA256 recorded;
applying it to this viewer sequence is an explicit offline scenario. Testing
1/3 degree endpoint offsets changes only the synthetic endpoint, never a
recorded measurement. Existing mission clips are not relabeled as seven-sample
windows. Route clearance, freshness, target association, and QR identity remain
separate runtime requirements.
