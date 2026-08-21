"""Pure width-quality policy for LiDAR stand-observation tracks.

The raw cluster detector sees chords and sparse beam returns rather than a
perfect stand diameter.  This module therefore derives explicit track-level
tolerances from the survey candidate envelope instead of treating the midpoint
of one broad generic width interval as the most plausible shape.  The survey
radius is a navigation/configuration proxy, not a metrology claim about the
physical stand.

The module is intentionally ROS-free and does not select a fixed number of
candidates.  Every track is assessed independently and receives deterministic
admission evidence or rejection reasons.  Proposal extraction must remain
broad enough to retain the complete width distribution; narrowing the producer
before this gate can censor the very samples needed to reject a broad fixture.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from scripts.aufgabe04.perception.models import LidarStandDetectorConfig
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.perception.stand_observation import StandObservation


STAND_WIDTH_PROFILE_SCHEMA_VERSION = 1
STAND_WIDTH_ASSESSMENT_SCHEMA_VERSION = 1
STAND_MORPHOLOGY_ADMISSION_SCHEMA_VERSION = 1
MORPHOLOGY_POLICY_SOURCE = "survey_candidate_envelope_proxy"
MORPHOLOGY_PROFILE_EVIDENCE_KEY = "lidar_track_morphology_profile"
MORPHOLOGY_PROFILE_SHA256_KEY = "lidar_track_morphology_profile_sha256"
PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY = "lidar_proposal_detector_config"
MINIMUM_PROPOSAL_MAX_DIAMETER_RATIO = 3.5


@dataclass(frozen=True)
class StandWidthProfile:
    """Survey-envelope proxy and explicit sparse-LiDAR track tolerances."""

    expected_diameter_m: float
    detector_lower_tolerance_m: float
    detector_upper_tolerance_m: float
    track_median_upper_tolerance_m: float
    track_max_median_absolute_deviation_m: float
    minimum_track_inlier_fraction: float
    minimum_track_observation_count: int

    @property
    def detector_min_width_m(self) -> float:
        return self.expected_diameter_m - self.detector_lower_tolerance_m

    @property
    def detector_max_width_m(self) -> float:
        return self.expected_diameter_m + self.detector_upper_tolerance_m

    @property
    def track_median_min_width_m(self) -> float:
        return self.detector_min_width_m

    @property
    def track_median_max_width_m(self) -> float:
        return self.expected_diameter_m + self.track_median_upper_tolerance_m

    @property
    def track_upper_quartile_max_width_m(self) -> float:
        return self.detector_max_width_m

    def validated(self) -> "StandWidthProfile":
        _finite_positive(self.expected_diameter_m, "expected_diameter_m")
        _finite_nonnegative(
            self.detector_lower_tolerance_m,
            "detector_lower_tolerance_m",
        )
        _finite_nonnegative(
            self.detector_upper_tolerance_m,
            "detector_upper_tolerance_m",
        )
        _finite_nonnegative(
            self.track_median_upper_tolerance_m,
            "track_median_upper_tolerance_m",
        )
        _finite_nonnegative(
            self.track_max_median_absolute_deviation_m,
            "track_max_median_absolute_deviation_m",
        )
        if self.detector_lower_tolerance_m >= self.expected_diameter_m:
            raise ValueError(
                "detector_lower_tolerance_m must be smaller than "
                "expected_diameter_m"
            )
        if (
            self.track_median_upper_tolerance_m
            > self.detector_upper_tolerance_m
        ):
            raise ValueError(
                "track_median_upper_tolerance_m must not exceed "
                "detector_upper_tolerance_m"
            )
        if not math.isfinite(self.minimum_track_inlier_fraction) or not (
            0.0 < self.minimum_track_inlier_fraction <= 1.0
        ):
            raise ValueError(
                "minimum_track_inlier_fraction must be finite and in (0, 1]"
            )
        if (
            type(self.minimum_track_observation_count) is not int
            or self.minimum_track_observation_count < 1
        ):
            raise ValueError(
                "minimum_track_observation_count must be a positive integer"
            )
        return self

    def to_evidence_dict(self) -> dict[str, object]:
        self.validated()
        return {
            "schema_version": STAND_WIDTH_PROFILE_SCHEMA_VERSION,
            "policy_source": MORPHOLOGY_POLICY_SOURCE,
            "metrology_claim": False,
            "expected_diameter_m": self.expected_diameter_m,
            "tolerances_m": {
                "detector_lower": self.detector_lower_tolerance_m,
                "detector_upper": self.detector_upper_tolerance_m,
                "track_median_upper": self.track_median_upper_tolerance_m,
                "track_max_median_absolute_deviation": (
                    self.track_max_median_absolute_deviation_m
                ),
            },
            "detector_width_bounds_m": {
                "minimum": self.detector_min_width_m,
                "maximum": self.detector_max_width_m,
            },
            "track_width_gates": {
                "median_minimum_m": self.track_median_min_width_m,
                "median_maximum_m": self.track_median_max_width_m,
                "upper_quartile_maximum_m": (
                    self.track_upper_quartile_max_width_m
                ),
                "maximum_median_absolute_deviation_m": (
                    self.track_max_median_absolute_deviation_m
                ),
                "minimum_inlier_fraction": (
                    self.minimum_track_inlier_fraction
                ),
                "minimum_observation_count": (
                    self.minimum_track_observation_count
                ),
            },
        }


@dataclass(frozen=True)
class StandWidthStatistics:
    """Deterministic robust summary of one candidate track's widths."""

    sample_count: int
    sorted_widths_m: tuple[float, ...]
    minimum_width_m: float
    lower_quartile_width_m: float
    median_width_m: float
    upper_quartile_width_m: float
    maximum_width_m: float
    median_absolute_deviation_m: float
    inlier_count: int
    inlier_fraction: float

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "sorted_widths_m": list(self.sorted_widths_m),
            "minimum_width_m": self.minimum_width_m,
            "lower_quartile_width_m": self.lower_quartile_width_m,
            "median_width_m": self.median_width_m,
            "upper_quartile_width_m": self.upper_quartile_width_m,
            "maximum_width_m": self.maximum_width_m,
            "median_absolute_deviation_m": (
                self.median_absolute_deviation_m
            ),
            "detector_inlier_count": self.inlier_count,
            "detector_inlier_fraction": self.inlier_fraction,
        }


@dataclass(frozen=True)
class StandWidthAssessment:
    """Fail-closed track morphology decision with audit-ready evidence."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    profile: StandWidthProfile
    statistics: StandWidthStatistics
    observation_count_met: bool
    median_width_met: bool
    upper_quartile_width_met: bool
    median_absolute_deviation_met: bool
    inlier_fraction_met: bool

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": STAND_WIDTH_ASSESSMENT_SCHEMA_VERSION,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "profile": self.profile.to_evidence_dict(),
            "statistics": self.statistics.to_evidence_dict(),
            "gates": {
                "observation_count_met": self.observation_count_met,
                "median_width_met": self.median_width_met,
                "upper_quartile_width_met": self.upper_quartile_width_met,
                "median_absolute_deviation_met": (
                    self.median_absolute_deviation_met
                ),
                "inlier_fraction_met": self.inlier_fraction_met,
            },
        }


@dataclass(frozen=True)
class StandMorphologyEvidence:
    """One confirmed track joined losslessly to its source observations."""

    stand_id: str
    source_observation_ids: tuple[str, ...]
    assessment: StandWidthAssessment

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "stand_id": self.stand_id,
            "source_observation_ids": list(self.source_observation_ids),
            **self.assessment.to_evidence_dict(),
        }


@dataclass(frozen=True)
class StandMorphologyAdmission:
    """Independent per-track morphology admission for one stopped epoch."""

    schema_version: int
    profile: StandWidthProfile
    source_observation_count: int
    evidence: tuple[StandMorphologyEvidence, ...]
    admitted_stands: tuple[ConfirmedStand, ...]
    rejected_stands: tuple[ConfirmedStand, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate": "lidar_stand_track_morphology_admission",
            "motion_authorized": False,
            "selection_policy": "independent_per_track_no_expected_count_ranking",
            "profile": self.profile.to_evidence_dict(),
            "source_observation_count": self.source_observation_count,
            "counts": {
                "evaluated": len(self.evidence),
                "admitted": len(self.admitted_stands),
                "rejected": len(self.rejected_stands),
            },
            "admitted_stand_ids": [
                stand.stand_id for stand in self.admitted_stands
            ],
            "rejected_stand_ids": [
                stand.stand_id for stand in self.rejected_stands
            ],
            "track_evidence": [item.to_evidence_dict() for item in self.evidence],
        }


def stand_width_profile_from_radius(
    stand_radius_m: float,
    *,
    detector_min_diameter_ratio: float = 0.25,
    detector_max_diameter_ratio: float = 1.50,
    track_median_max_diameter_ratio: float = 1.25,
    track_max_mad_diameter_ratio: float = 0.30,
    minimum_track_inlier_fraction: float = 0.75,
    minimum_track_observation_count: int = 3,
) -> StandWidthProfile:
    """Derive a track policy from the survey candidate-radius proxy.

    The default lower ratio permits a short chord from two LDS beams; the upper
    and track-level limits exclude broad wall or arena-fixture clusters.  The
    resulting ``detector_*`` bounds describe morphology inliers.  They must not
    replace the producer's broad proposal-extraction bounds unless rejected
    proposal clusters are also retained.
    """

    _finite_positive(stand_radius_m, "stand_radius_m")
    expected_diameter_m = 2.0 * float(stand_radius_m)
    _finite_ratio_in_range(
        detector_min_diameter_ratio,
        "detector_min_diameter_ratio",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    _finite_ratio_in_range(
        detector_max_diameter_ratio,
        "detector_max_diameter_ratio",
        lower=1.0,
        upper=math.inf,
        lower_inclusive=True,
        upper_inclusive=False,
    )
    _finite_ratio_in_range(
        track_median_max_diameter_ratio,
        "track_median_max_diameter_ratio",
        lower=1.0,
        upper=detector_max_diameter_ratio,
        lower_inclusive=True,
        upper_inclusive=True,
    )
    _finite_nonnegative(
        track_max_mad_diameter_ratio,
        "track_max_mad_diameter_ratio",
    )

    return StandWidthProfile(
        expected_diameter_m=expected_diameter_m,
        detector_lower_tolerance_m=(
            expected_diameter_m * (1.0 - detector_min_diameter_ratio)
        ),
        detector_upper_tolerance_m=(
            expected_diameter_m * (detector_max_diameter_ratio - 1.0)
        ),
        track_median_upper_tolerance_m=(
            expected_diameter_m
            * (track_median_max_diameter_ratio - 1.0)
        ),
        track_max_median_absolute_deviation_m=(
            expected_diameter_m * track_max_mad_diameter_ratio
        ),
        minimum_track_inlier_fraction=minimum_track_inlier_fraction,
        minimum_track_observation_count=minimum_track_observation_count,
    ).validated()


def lidar_detector_config_for_stand_width(
    profile: StandWidthProfile,
    *,
    base_config: LidarStandDetectorConfig | None = None,
) -> LidarStandDetectorConfig:
    """Return an admission-shaped detector config for offline/calibration use.

    The autonomous survey intentionally does not use this helper at proposal
    extraction time because doing so would censor broad-track evidence.
    """

    _validated_profile(profile)
    return replace(
        base_config or LidarStandDetectorConfig(),
        min_width_m=profile.detector_min_width_m,
        max_width_m=profile.detector_max_width_m,
    )


def validated_broad_proposal_width_bounds(
    *,
    profile: StandWidthProfile,
    proposal_min_width_m: float,
    proposal_max_width_m: float,
) -> dict[str, object]:
    """Prove proposal extraction is broad enough for track-level rejection.

    The upper requirement deliberately covers the widest sample in the audited
    false-track regression.  This is an evidence-preservation contract, not an
    assertion that such a wide cluster is a valid stand.
    """

    _validated_profile(profile)
    minimum = _finite_positive(proposal_min_width_m, "proposal_min_width_m")
    maximum = _finite_positive(proposal_max_width_m, "proposal_max_width_m")
    if minimum > profile.detector_min_width_m + 1.0e-12:
        raise ValueError(
            "proposal minimum width would censor morphology inlier evidence"
        )
    required_maximum = (
        profile.expected_diameter_m * MINIMUM_PROPOSAL_MAX_DIAMETER_RATIO
    )
    if maximum + 1.0e-12 < required_maximum:
        raise ValueError(
            "proposal maximum width would censor broad-track rejection evidence"
        )
    if maximum <= minimum:
        raise ValueError("proposal maximum width must exceed its minimum")
    return {
        "proposal_min_width_m": minimum,
        "proposal_max_width_m": maximum,
        "required_proposal_min_width_at_most_m": (
            profile.detector_min_width_m
        ),
        "required_proposal_max_width_at_least_m": required_maximum,
        "minimum_proposal_max_diameter_ratio": (
            MINIMUM_PROPOSAL_MAX_DIAMETER_RATIO
        ),
        "preserves_track_morphology_evidence": True,
    }


def assess_stand_width_samples(
    widths_m: Iterable[float],
    *,
    profile: StandWidthProfile,
) -> StandWidthAssessment:
    """Assess one track without ranking it against any other candidate."""

    _validated_profile(profile)
    widths = _validated_widths(widths_m)
    statistics_value = _width_statistics(widths, profile=profile)

    observation_count_met = (
        statistics_value.sample_count >= profile.minimum_track_observation_count
    )
    median_width_met = (
        profile.track_median_min_width_m
        <= statistics_value.median_width_m
        <= profile.track_median_max_width_m
    )
    upper_quartile_width_met = (
        statistics_value.upper_quartile_width_m
        <= profile.track_upper_quartile_max_width_m
    )
    median_absolute_deviation_met = (
        statistics_value.median_absolute_deviation_m
        <= profile.track_max_median_absolute_deviation_m
    )
    inlier_fraction_met = (
        statistics_value.inlier_fraction
        >= profile.minimum_track_inlier_fraction
    )

    reasons: list[str] = []
    if not observation_count_met:
        reasons.append("insufficient_width_observations")
    if statistics_value.median_width_m < profile.track_median_min_width_m:
        reasons.append("median_width_below_minimum")
    elif statistics_value.median_width_m > profile.track_median_max_width_m:
        reasons.append("median_width_above_maximum")
    if not upper_quartile_width_met:
        reasons.append("upper_quartile_width_above_maximum")
    if not median_absolute_deviation_met:
        reasons.append("median_absolute_deviation_above_maximum")
    if not inlier_fraction_met:
        reasons.append("width_inlier_fraction_below_minimum")

    return StandWidthAssessment(
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        profile=profile,
        statistics=statistics_value,
        observation_count_met=observation_count_met,
        median_width_met=median_width_met,
        upper_quartile_width_met=upper_quartile_width_met,
        median_absolute_deviation_met=median_absolute_deviation_met,
        inlier_fraction_met=inlier_fraction_met,
    )


def assess_stand_observation_track(
    observations: Iterable[StandObservation],
    *,
    profile: StandWidthProfile,
) -> StandWidthAssessment:
    """Assess ``StandObservation.approximate_width_m`` for one track."""

    items = tuple(observations)
    if any(not isinstance(item, StandObservation) for item in items):
        raise ValueError("observations must contain only StandObservation values")
    return assess_stand_width_samples(
        (item.approximate_width_m for item in items),
        profile=profile,
    )


def evaluate_stand_morphology_admission(
    stands: Iterable[ConfirmedStand],
    observations: Iterable[StandObservation],
    *,
    profile: StandWidthProfile,
) -> StandMorphologyAdmission:
    """Join confirmed tracks to source observations and gate them independently.

    Missing, duplicated, or cross-claimed observation IDs are structural
    evidence failures and raise ``ValueError``.  A well-formed track that does
    not meet the width policy is a normal, persisted rejection.
    """

    _validated_profile(profile)
    observation_snapshot = tuple(observations)
    if any(not isinstance(item, StandObservation) for item in observation_snapshot):
        raise ValueError("observations must contain only StandObservation values")
    observations_by_id: dict[str, StandObservation] = {}
    for observation in observation_snapshot:
        if observation.observation_id in observations_by_id:
            raise ValueError(
                "duplicate source observation ID in morphology input: "
                f"{observation.observation_id!r}"
            )
        observations_by_id[observation.observation_id] = observation

    stand_snapshot = tuple(stands)
    if any(not isinstance(item, ConfirmedStand) for item in stand_snapshot):
        raise ValueError("stands must contain only ConfirmedStand values")
    ordered_stands = tuple(
        sorted(stand_snapshot, key=lambda item: (item.x_m, item.y_m, item.stand_id))
    )
    seen_stand_ids: set[str] = set()
    claimed_observation_ids: set[str] = set()
    evidence: list[StandMorphologyEvidence] = []
    admitted: list[ConfirmedStand] = []
    rejected: list[ConfirmedStand] = []
    for stand in ordered_stands:
        if not stand.stand_id.strip() or stand.stand_id in seen_stand_ids:
            raise ValueError(f"invalid or duplicate confirmed stand ID: {stand.stand_id!r}")
        seen_stand_ids.add(stand.stand_id)
        source_ids = tuple(stand.source_observation_ids)
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError(
                f"confirmed stand {stand.stand_id!r} has invalid source IDs"
            )
        overlap = claimed_observation_ids.intersection(source_ids)
        if overlap:
            raise ValueError(
                "source observation IDs are claimed by multiple confirmed stands: "
                + ", ".join(sorted(overlap))
            )
        missing = tuple(
            observation_id
            for observation_id in source_ids
            if observation_id not in observations_by_id
        )
        if missing:
            raise ValueError(
                f"confirmed stand {stand.stand_id!r} references missing observations: "
                + ", ".join(missing)
            )
        claimed_observation_ids.update(source_ids)
        assessment = assess_stand_observation_track(
            (observations_by_id[item] for item in source_ids),
            profile=profile,
        )
        evidence.append(
            StandMorphologyEvidence(
                stand_id=stand.stand_id,
                source_observation_ids=source_ids,
                assessment=assessment,
            )
        )
        (admitted if assessment.accepted else rejected).append(stand)

    return StandMorphologyAdmission(
        schema_version=STAND_MORPHOLOGY_ADMISSION_SCHEMA_VERSION,
        profile=profile,
        source_observation_count=len(observation_snapshot),
        evidence=tuple(evidence),
        admitted_stands=tuple(admitted),
        rejected_stands=tuple(rejected),
    )


def _width_statistics(
    widths_m: Sequence[float],
    *,
    profile: StandWidthProfile,
) -> StandWidthStatistics:
    ordered = tuple(sorted(widths_m))
    median_width = float(statistics.median(ordered))
    absolute_deviations = tuple(abs(value - median_width) for value in ordered)
    inlier_count = sum(
        profile.detector_min_width_m
        <= value
        <= profile.detector_max_width_m
        for value in ordered
    )
    return StandWidthStatistics(
        sample_count=len(ordered),
        sorted_widths_m=ordered,
        minimum_width_m=ordered[0],
        lower_quartile_width_m=_linear_quantile(ordered, 0.25),
        median_width_m=median_width,
        upper_quartile_width_m=_linear_quantile(ordered, 0.75),
        maximum_width_m=ordered[-1],
        median_absolute_deviation_m=float(
            statistics.median(absolute_deviations)
        ),
        inlier_count=inlier_count,
        inlier_fraction=inlier_count / len(ordered),
    )


def _linear_quantile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(
        ordered[lower_index]
        + fraction * (ordered[upper_index] - ordered[lower_index])
    )


def _validated_widths(widths_m: Iterable[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in widths_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("width samples must be finite positive numbers") from exc
    if not values:
        raise ValueError("width samples must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("width samples must be finite positive numbers")
    return values


def _validated_profile(profile: StandWidthProfile) -> StandWidthProfile:
    if not isinstance(profile, StandWidthProfile):
        raise ValueError("profile must be a StandWidthProfile")
    return profile.validated()


def _finite_positive(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _finite_nonnegative(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _finite_ratio_in_range(
    value: float,
    name: str,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is outside its permitted range") from exc
    lower_met = parsed >= lower if lower_inclusive else parsed > lower
    upper_met = parsed <= upper if upper_inclusive else parsed < upper
    if not math.isfinite(parsed) or not lower_met or not upper_met:
        raise ValueError(f"{name} is outside its permitted range")
    return parsed


__all__ = [
    "MORPHOLOGY_POLICY_SOURCE",
    "MORPHOLOGY_PROFILE_EVIDENCE_KEY",
    "MORPHOLOGY_PROFILE_SHA256_KEY",
    "PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY",
    "MINIMUM_PROPOSAL_MAX_DIAMETER_RATIO",
    "STAND_MORPHOLOGY_ADMISSION_SCHEMA_VERSION",
    "STAND_WIDTH_ASSESSMENT_SCHEMA_VERSION",
    "STAND_WIDTH_PROFILE_SCHEMA_VERSION",
    "StandMorphologyAdmission",
    "StandMorphologyEvidence",
    "StandWidthAssessment",
    "StandWidthProfile",
    "StandWidthStatistics",
    "assess_stand_observation_track",
    "assess_stand_width_samples",
    "evaluate_stand_morphology_admission",
    "lidar_detector_config_for_stand_width",
    "stand_width_profile_from_radius",
    "validated_broad_proposal_width_bounds",
]
