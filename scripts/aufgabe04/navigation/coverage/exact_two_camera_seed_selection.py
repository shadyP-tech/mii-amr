"""Pure camera-seed selection for an exact-two LiDAR checkpoint.

Static-map-admitted candidates are the strict population.  Boundary-
provisional candidates are never allowed to displace a strict candidate: they
may fill only an exact strict-population deficit.  When the strict population
already matches the expected stand count, every boundary candidate remains
explicitly audit-only.

The decision is motion-neutral and ROS-free.  Invalid inputs and ambiguous
populations produce immutable ``ready=False`` evidence instead of selecting a
best-effort subset.  Candidate support remains caller-owned: this module
handles only UID identity, population counts, partition priority, and ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


EXACT_TWO_CAMERA_SEED_SELECTION_SCHEMA_VERSION = 1

SELECTION_MODE_NOT_READY = "not_ready"
SELECTION_MODE_STRICT_EXACT = "strict_exact"
SELECTION_MODE_STRICT_EXACT_BOUNDARY_AUDIT_ONLY = (
    "strict_exact_boundary_audit_only"
)
SELECTION_MODE_EXACT_BOUNDARY_DEFICIT_FILL = "exact_boundary_deficit_fill"

REASON_EXPECTED_STAND_COUNT_MISSING = "expected_stand_count_missing"
REASON_EXPECTED_STAND_COUNT_NOT_INTEGER = "expected_stand_count_not_integer"
REASON_EXPECTED_STAND_COUNT_NOT_POSITIVE = "expected_stand_count_not_positive"
REASON_CANDIDATE_UID_PARTITION_OVERLAP = "candidate_uid_partition_overlap"
REASON_STRICT_CANDIDATE_COUNT_EXCEEDS_EXPECTED = (
    "strict_candidate_count_exceeds_expected"
)
REASON_USABLE_CANDIDATE_COUNT_BELOW_EXPECTED = (
    "usable_candidate_count_below_expected"
)
REASON_BOUNDARY_CANDIDATE_SURPLUS_AMBIGUOUS = (
    "boundary_candidate_surplus_ambiguous"
)

_SAFE_CANDIDATE_UID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ExactTwoCameraSeedSelectionDecision:
    """Immutable, JSON-ready result of one camera-seed population decision."""

    schema_version: int
    ready: bool
    reasons: tuple[str, ...]
    selection_mode: str
    expected_stand_count: int | None
    raw_strict_static_map_admitted_candidate_uids: tuple[str, ...]
    raw_boundary_provisional_candidate_uids: tuple[str, ...]
    strict_static_map_admitted_candidate_uids: tuple[str, ...]
    boundary_provisional_candidate_uids: tuple[str, ...]
    selected_candidate_uids: tuple[str, ...]
    boundary_fill_candidate_uids: tuple[str, ...]
    boundary_audit_only_candidate_uids: tuple[str, ...]
    excluded_candidate_uids: tuple[str, ...]
    motion_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != EXACT_TWO_CAMERA_SEED_SELECTION_SCHEMA_VERSION:
            raise ValueError("unsupported exact-two camera seed schema")
        if type(self.ready) is not bool or self.motion_authorized is not False:
            raise ValueError("camera seed decision must be motion-neutral")
        tuple_fields = (
            self.reasons,
            self.raw_strict_static_map_admitted_candidate_uids,
            self.raw_boundary_provisional_candidate_uids,
            self.strict_static_map_admitted_candidate_uids,
            self.boundary_provisional_candidate_uids,
            self.selected_candidate_uids,
            self.boundary_fill_candidate_uids,
            self.boundary_audit_only_candidate_uids,
            self.excluded_candidate_uids,
        )
        if any(not isinstance(values, tuple) for values in tuple_fields):
            raise TypeError("camera seed UID and reason fields must be tuples")
        if any(
            not isinstance(value, str)
            for values in tuple_fields
            for value in values
        ):
            raise TypeError("camera seed UID and reason values must be text")
        canonical_uid_fields = {
            "strict": self.strict_static_map_admitted_candidate_uids,
            "boundary": self.boundary_provisional_candidate_uids,
            "selected": self.selected_candidate_uids,
            "boundary fill": self.boundary_fill_candidate_uids,
            "boundary audit-only": self.boundary_audit_only_candidate_uids,
            "excluded": self.excluded_candidate_uids,
        }
        for name, values in canonical_uid_fields.items():
            if values != tuple(sorted(set(values))):
                raise ValueError(
                    f"{name} camera seed UIDs must be canonical and unique"
                )

        strict = set(self.strict_static_map_admitted_candidate_uids)
        boundary = set(self.boundary_provisional_candidate_uids)
        selected = set(self.selected_candidate_uids)
        fill = set(self.boundary_fill_candidate_uids)
        audit_only = set(self.boundary_audit_only_candidate_uids)
        excluded = set(self.excluded_candidate_uids)
        usable = strict.union(boundary)
        if selected.intersection(excluded) or selected.union(excluded) != usable:
            raise ValueError("selected and excluded UIDs must partition usable UIDs")
        if not fill.issubset(boundary) or not audit_only.issubset(boundary):
            raise ValueError("boundary fill and audit-only UIDs must be boundary UIDs")

        if self.ready:
            if self.reasons:
                raise ValueError("ready camera seed decision cannot contain reasons")
            if (
                type(self.expected_stand_count) is not int
                or self.expected_stand_count <= 0
                or len(self.selected_candidate_uids)
                != self.expected_stand_count
            ):
                raise ValueError(
                    "ready camera seed decision must select the expected count"
                )
            if not strict.issubset(selected) or strict.intersection(excluded):
                raise ValueError(
                    "ready camera seed decision cannot exclude a strict candidate"
                )
            expected_mode = (
                SELECTION_MODE_EXACT_BOUNDARY_DEFICIT_FILL
                if fill
                else SELECTION_MODE_STRICT_EXACT_BOUNDARY_AUDIT_ONLY
                if audit_only
                else SELECTION_MODE_STRICT_EXACT
            )
            if self.selection_mode != expected_mode:
                raise ValueError(
                    "ready camera seed selection mode differs from partitions"
                )
            if self.selected_candidate_uids != tuple(
                sorted(
                    self.strict_static_map_admitted_candidate_uids
                    + self.boundary_fill_candidate_uids
                )
            ):
                raise ValueError(
                    "ready camera seed UIDs must use global canonical order"
                )
            nonselected_boundary = boundary.difference(selected)
            if audit_only != nonselected_boundary or excluded != audit_only:
                raise ValueError(
                    "all nonselected boundary UIDs must be excluded audit-only"
                )
        elif (
            not self.reasons
            or self.selection_mode != SELECTION_MODE_NOT_READY
            or selected
            or fill
            or audit_only
        ):
            raise ValueError("not-ready camera seed decision must fail closed")

    @property
    def strict_candidate_count(self) -> int:
        return len(self.strict_static_map_admitted_candidate_uids)

    @property
    def boundary_candidate_count(self) -> int:
        return len(self.boundary_provisional_candidate_uids)

    @property
    def usable_candidate_count(self) -> int:
        return len(
            set(self.strict_static_map_admitted_candidate_uids).union(
                self.boundary_provisional_candidate_uids
            )
        )

    @property
    def selected_candidate_count(self) -> int:
        return len(self.selected_candidate_uids)

    def to_evidence_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, explicitly motion-neutral payload."""

        return {
            "schema_version": self.schema_version,
            "decision": "ready" if self.ready else "not_ready",
            "ready": self.ready,
            "reasons": list(self.reasons),
            "selection_mode": self.selection_mode,
            "expected_stand_count": self.expected_stand_count,
            "counts": {
                "strict_static_map_admitted": self.strict_candidate_count,
                "boundary_provisional": self.boundary_candidate_count,
                "usable": self.usable_candidate_count,
                "selected": self.selected_candidate_count,
                "boundary_fill": len(self.boundary_fill_candidate_uids),
                "boundary_audit_only": len(
                    self.boundary_audit_only_candidate_uids
                ),
                "excluded": len(self.excluded_candidate_uids),
            },
            "raw_strict_static_map_admitted_candidate_uids": list(
                self.raw_strict_static_map_admitted_candidate_uids
            ),
            "raw_boundary_provisional_candidate_uids": list(
                self.raw_boundary_provisional_candidate_uids
            ),
            "strict_static_map_admitted_candidate_uids": list(
                self.strict_static_map_admitted_candidate_uids
            ),
            "boundary_provisional_candidate_uids": list(
                self.boundary_provisional_candidate_uids
            ),
            "selected_candidate_uids": list(self.selected_candidate_uids),
            "boundary_fill_candidate_uids": list(
                self.boundary_fill_candidate_uids
            ),
            "boundary_audit_only_candidate_uids": list(
                self.boundary_audit_only_candidate_uids
            ),
            "excluded_candidate_uids": list(self.excluded_candidate_uids),
            "motion_authorized": self.motion_authorized,
        }

    def to_evidence(self) -> dict[str, object]:
        """Compatibility spelling for callers that persist evidence."""

        return self.to_evidence_dict()


@dataclass(frozen=True)
class _UidPartition:
    raw_text_uids: tuple[str, ...]
    canonical_uids: tuple[str, ...]
    reasons: tuple[str, ...]


def select_exact_two_camera_seed_candidates(
    *,
    expected_stand_count: object,
    static_map_admitted_candidate_uids: object,
    boundary_provisional_candidate_uids: object,
) -> ExactTwoCameraSeedSelectionDecision:
    """Select exactly one expected-size, strict-first camera seed population.

    Input order cannot affect the result.  UIDs are canonicalized
    lexicographically, matching the survey registry's stable candidate order.
    No branch authorizes motion.
    """

    normalized_expected, expected_reasons = _expected_count(
        expected_stand_count
    )
    strict = _uid_partition(
        static_map_admitted_candidate_uids,
        partition_name="strict",
    )
    boundary = _uid_partition(
        boundary_provisional_candidate_uids,
        partition_name="boundary",
    )
    reasons = [*expected_reasons, *strict.reasons, *boundary.reasons]

    overlap = set(strict.canonical_uids).intersection(
        boundary.canonical_uids
    )
    if overlap:
        reasons.append(REASON_CANDIDATE_UID_PARTITION_OVERLAP)

    selected: tuple[str, ...] = ()
    boundary_fill: tuple[str, ...] = ()
    boundary_audit_only: tuple[str, ...] = ()
    selection_mode = SELECTION_MODE_NOT_READY

    inputs_valid = not reasons
    if inputs_valid:
        assert normalized_expected is not None
        strict_count = len(strict.canonical_uids)
        boundary_count = len(boundary.canonical_uids)
        deficit = normalized_expected - strict_count
        if strict_count > normalized_expected:
            reasons.append(REASON_STRICT_CANDIDATE_COUNT_EXCEEDS_EXPECTED)
        elif deficit == 0:
            selected = strict.canonical_uids
            boundary_audit_only = boundary.canonical_uids
            selection_mode = (
                SELECTION_MODE_STRICT_EXACT_BOUNDARY_AUDIT_ONLY
                if boundary_audit_only
                else SELECTION_MODE_STRICT_EXACT
            )
        elif boundary_count < deficit:
            reasons.append(REASON_USABLE_CANDIDATE_COUNT_BELOW_EXPECTED)
        elif boundary_count > deficit:
            reasons.append(REASON_BOUNDARY_CANDIDATE_SURPLUS_AMBIGUOUS)
        else:
            boundary_fill = boundary.canonical_uids
            selected = tuple(sorted(strict.canonical_uids + boundary_fill))
            selection_mode = SELECTION_MODE_EXACT_BOUNDARY_DEFICIT_FILL

    ready = not reasons
    if not ready:
        selected = ()
        boundary_fill = ()
        boundary_audit_only = ()
        selection_mode = SELECTION_MODE_NOT_READY

    excluded = _excluded_uids(
        strict.canonical_uids,
        boundary.canonical_uids,
        selected,
    )
    return ExactTwoCameraSeedSelectionDecision(
        schema_version=EXACT_TWO_CAMERA_SEED_SELECTION_SCHEMA_VERSION,
        ready=ready,
        reasons=tuple(reasons),
        selection_mode=selection_mode,
        expected_stand_count=normalized_expected,
        raw_strict_static_map_admitted_candidate_uids=strict.raw_text_uids,
        raw_boundary_provisional_candidate_uids=boundary.raw_text_uids,
        strict_static_map_admitted_candidate_uids=strict.canonical_uids,
        boundary_provisional_candidate_uids=boundary.canonical_uids,
        selected_candidate_uids=selected,
        boundary_fill_candidate_uids=boundary_fill,
        boundary_audit_only_candidate_uids=boundary_audit_only,
        excluded_candidate_uids=excluded,
        motion_authorized=False,
    )


def _expected_count(value: object) -> tuple[int | None, tuple[str, ...]]:
    if value is None:
        return None, (REASON_EXPECTED_STAND_COUNT_MISSING,)
    if type(value) is not int:
        return None, (REASON_EXPECTED_STAND_COUNT_NOT_INTEGER,)
    if value <= 0:
        return value, (REASON_EXPECTED_STAND_COUNT_NOT_POSITIVE,)
    return value, ()


def _uid_partition(
    value: object,
    *,
    partition_name: str,
) -> _UidPartition:
    if value is None:
        return _UidPartition(
            raw_text_uids=(),
            canonical_uids=(),
            reasons=(f"{partition_name}_candidate_uids_missing",),
        )
    if not isinstance(value, (tuple, list)):
        return _UidPartition(
            raw_text_uids=(),
            canonical_uids=(),
            reasons=(f"{partition_name}_candidate_uids_not_sequence",),
        )

    text_uids = tuple(item for item in value if isinstance(item, str))
    raw_text_uids = tuple(sorted(text_uids))
    reasons: list[str] = []
    if len(text_uids) != len(value) or any(
        _SAFE_CANDIDATE_UID.fullmatch(uid) is None for uid in text_uids
    ):
        reasons.append(f"malformed_{partition_name}_candidate_uid")
    if len(text_uids) != len(set(text_uids)):
        reasons.append(f"duplicate_{partition_name}_candidate_uid")

    canonical_uids = tuple(
        sorted(
            {
                uid
                for uid in text_uids
                if _SAFE_CANDIDATE_UID.fullmatch(uid) is not None
            }
        )
    )
    return _UidPartition(
        raw_text_uids=raw_text_uids,
        canonical_uids=canonical_uids,
        reasons=tuple(reasons),
    )


def _excluded_uids(
    strict_uids: tuple[str, ...],
    boundary_uids: tuple[str, ...],
    selected_uids: tuple[str, ...],
) -> tuple[str, ...]:
    selected = set(selected_uids)
    excluded: list[str] = []
    for uid in strict_uids + boundary_uids:
        if uid not in selected and uid not in excluded:
            excluded.append(uid)
    return tuple(excluded)


__all__ = [
    "EXACT_TWO_CAMERA_SEED_SELECTION_SCHEMA_VERSION",
    "ExactTwoCameraSeedSelectionDecision",
    "REASON_BOUNDARY_CANDIDATE_SURPLUS_AMBIGUOUS",
    "REASON_CANDIDATE_UID_PARTITION_OVERLAP",
    "REASON_EXPECTED_STAND_COUNT_MISSING",
    "REASON_EXPECTED_STAND_COUNT_NOT_INTEGER",
    "REASON_EXPECTED_STAND_COUNT_NOT_POSITIVE",
    "REASON_STRICT_CANDIDATE_COUNT_EXCEEDS_EXPECTED",
    "REASON_USABLE_CANDIDATE_COUNT_BELOW_EXPECTED",
    "SELECTION_MODE_EXACT_BOUNDARY_DEFICIT_FILL",
    "SELECTION_MODE_NOT_READY",
    "SELECTION_MODE_STRICT_EXACT",
    "SELECTION_MODE_STRICT_EXACT_BOUNDARY_AUDIT_ONLY",
    "select_exact_two_camera_seed_candidates",
]
