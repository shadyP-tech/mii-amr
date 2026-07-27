"""Pure synchronized-viewpoint decisions for the Gazebo stand pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M,
    DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
    ViewpointSamplingArrivalLatch,
    ViewpointSamplingHoldConfig,
    ViewpointSamplingMaterialTarget,
)
from scripts.aufgabe04.stations.arrival_pose_geometry import (
    ArrivalGeometryConfig,
    arrival_face_candidates,
)


T = TypeVar("T")
DEFAULT_SAMPLING_ARRIVAL_TOLERANCE_M = (
    DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M
)
DEFAULT_TANGENTIAL_CORRECTION_GAIN = 0.5
VIEWPOINT_SAMPLING_ODOM_QUATERNION_NORM_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    stamp_sec: float
    value: T


@dataclass(frozen=True)
class ViewpointSamplingOdomSample:
    """ROS-free odometry evidence retained between image-processing ticks."""

    stamp_sec: float
    pose: Pose2D
    parent_frame: str
    child_frame: str
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewpointSamplingOdomReplayBatch:
    """Chronological, target-bounded odometry accepted for latch replay."""

    samples: tuple[ViewpointSamplingOdomSample, ...]
    diagnostics: dict[str, object]


def viewpoint_sampling_history_identity(
    stream_id: str,
    target: ViewpointSamplingMaterialTarget,
) -> dict[str, object]:
    """Return the persisted stream-and-material-target cursor identity."""

    if not isinstance(stream_id, str) or not stream_id.strip():
        raise ValueError("stream_id must be a non-empty string")
    payload = {
        "stream_id": stream_id,
        "material_target": target.to_status_dict(),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def viewpoint_sampling_history_identity_changed(
    *,
    previous_stream_id: str | None,
    previous_target: ViewpointSamplingMaterialTarget | None,
    current_stream_id: str,
    current_target: ViewpointSamplingMaterialTarget | None,
) -> bool:
    """Return whether all buffered evidence must be discarded."""

    return (
        previous_stream_id != current_stream_id
        or previous_target != current_target
    )


def select_viewpoint_sampling_odom_replay(
    samples: Sequence[ViewpointSamplingOdomSample],
    *,
    target_pose: Pose2D,
    target_activation_stamp_sec: float,
    last_checked_odom_stamp_sec: float,
    current_image_stamp_sec: float,
    current_pose_stamp_sec: float | None = None,
    expected_parent_frame: str,
    expected_child_frame: str,
    strict_entry_tolerance_m: float,
    quaternion_norm_tolerance: float = (
        VIEWPOINT_SAMPLING_ODOM_QUATERNION_NORM_TOLERANCE
    ),
) -> ViewpointSamplingOdomReplayBatch:
    """Select new exact-frame samples without weakening strict arrival.

    Only odometry observed after both target activation and the previous
    high-water mark, and no later than the image currently being processed,
    may contribute arrival evidence. Any malformed sample inside that window
    rejects the complete batch so partial sensor evidence cannot arm the
    latch.
    """

    scalar_inputs = {
        "target_activation_stamp_sec": target_activation_stamp_sec,
        "last_checked_odom_stamp_sec": last_checked_odom_stamp_sec,
        "current_image_stamp_sec": current_image_stamp_sec,
        "strict_entry_tolerance_m": strict_entry_tolerance_m,
        "quaternion_norm_tolerance": quaternion_norm_tolerance,
    }
    if current_pose_stamp_sec is not None:
        if not math.isfinite(current_pose_stamp_sec):
            raise ValueError("current_pose_stamp_sec must be finite or None")
        scalar_inputs["current_pose_stamp_sec"] = current_pose_stamp_sec
    for name, value in scalar_inputs.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if strict_entry_tolerance_m <= 0.0:
        raise ValueError("strict_entry_tolerance_m must be positive")
    if quaternion_norm_tolerance <= 0.0:
        raise ValueError("quaternion_norm_tolerance must be positive")
    if not expected_parent_frame or not expected_child_frame:
        raise ValueError("expected odometry frames must be non-empty")
    if not all(
        math.isfinite(value)
        for value in (target_pose.x_m, target_pose.y_m, target_pose.yaw_rad)
    ):
        raise ValueError("target_pose must be finite")

    lower_bound_sec = max(
        target_activation_stamp_sec,
        last_checked_odom_stamp_sec,
    )
    diagnostics: dict[str, object] = {
        "state": "no_new_samples",
        "fail_closed": False,
        "reason": "no_new_odom_in_target_window",
        "target_activation_stamp_sec": target_activation_stamp_sec,
        "last_checked_odom_stamp_sec_before": last_checked_odom_stamp_sec,
        "current_image_stamp_sec": current_image_stamp_sec,
        "history_replay_upper_bound_stamp_sec": (
            current_image_stamp_sec
            if current_pose_stamp_sec is None
            else min(current_image_stamp_sec, current_pose_stamp_sec)
        ),
        "history_lower_bound_stamp_sec": lower_bound_sec,
        "expected_parent_frame": expected_parent_frame,
        "expected_child_frame": expected_child_frame,
        "strict_entry_tolerance_m": strict_entry_tolerance_m,
        "quaternion_norm_tolerance": quaternion_norm_tolerance,
        "buffered_sample_count": len(samples),
        "eligible_window_sample_count": 0,
        "processed_sample_count": 0,
        "excluded_at_or_before_activation_count": 0,
        "excluded_at_or_before_last_checked_count": 0,
        "excluded_after_image_count": 0,
        "excluded_at_or_after_current_pose_count": 0,
        "invalid_stamp_count": 0,
        "frame_mismatch_count": 0,
        "nonfinite_pose_count": 0,
        "invalid_quaternion_count": 0,
        "nonmonotonic_stamp_count": 0,
        "first_processed_stamp_sec": None,
        "last_processed_stamp_sec": None,
        "minimum_target_distance_m": None,
        "minimum_target_distance_stamp_sec": None,
        "strict_entry_sample_count": 0,
        "first_strict_entry_stamp_sec": None,
        "last_strict_entry_stamp_sec": None,
    }

    eligible: list[ViewpointSamplingOdomSample] = []
    previous_eligible_stamp = -math.inf
    invalid_reasons: list[str] = []
    for sample in samples:
        stamp_sec = sample.stamp_sec
        if not math.isfinite(stamp_sec):
            diagnostics["invalid_stamp_count"] = (
                int(diagnostics["invalid_stamp_count"]) + 1
            )
            invalid_reasons.append("nonfinite_odom_stamp")
            continue
        if stamp_sec <= target_activation_stamp_sec:
            diagnostics["excluded_at_or_before_activation_count"] = (
                int(diagnostics["excluded_at_or_before_activation_count"]) + 1
            )
            continue
        if stamp_sec <= last_checked_odom_stamp_sec:
            diagnostics["excluded_at_or_before_last_checked_count"] = (
                int(diagnostics["excluded_at_or_before_last_checked_count"]) + 1
            )
            continue
        if stamp_sec > current_image_stamp_sec:
            diagnostics["excluded_after_image_count"] = (
                int(diagnostics["excluded_after_image_count"]) + 1
            )
            continue
        if (
            current_pose_stamp_sec is not None
            and stamp_sec >= current_pose_stamp_sec
        ):
            diagnostics["excluded_at_or_after_current_pose_count"] = (
                int(
                    diagnostics[
                        "excluded_at_or_after_current_pose_count"
                    ]
                )
                + 1
            )
            continue

        diagnostics["eligible_window_sample_count"] = (
            int(diagnostics["eligible_window_sample_count"]) + 1
        )
        if stamp_sec <= previous_eligible_stamp:
            diagnostics["nonmonotonic_stamp_count"] = (
                int(diagnostics["nonmonotonic_stamp_count"]) + 1
            )
            invalid_reasons.append("nonmonotonic_odom_stamp")
        previous_eligible_stamp = max(previous_eligible_stamp, stamp_sec)

        if (
            sample.parent_frame != expected_parent_frame
            or sample.child_frame != expected_child_frame
        ):
            diagnostics["frame_mismatch_count"] = (
                int(diagnostics["frame_mismatch_count"]) + 1
            )
            invalid_reasons.append("odom_frame_mismatch")

        pose_values = (
            sample.pose.x_m,
            sample.pose.y_m,
            sample.pose.yaw_rad,
        )
        if not all(math.isfinite(value) for value in pose_values):
            diagnostics["nonfinite_pose_count"] = (
                int(diagnostics["nonfinite_pose_count"]) + 1
            )
            invalid_reasons.append("nonfinite_odom_pose")

        quaternion = sample.quaternion_xyzw
        quaternion_valid = (
            len(quaternion) == 4
            and all(math.isfinite(value) for value in quaternion)
        )
        if quaternion_valid:
            quaternion_norm = math.sqrt(
                sum(value * value for value in quaternion)
            )
            quaternion_valid = (
                math.isfinite(quaternion_norm)
                and abs(quaternion_norm - 1.0)
                <= quaternion_norm_tolerance
            )
        if not quaternion_valid:
            diagnostics["invalid_quaternion_count"] = (
                int(diagnostics["invalid_quaternion_count"]) + 1
            )
            invalid_reasons.append("invalid_odom_quaternion")

        eligible.append(sample)

    if invalid_reasons:
        diagnostics.update(
            {
                "state": "rejected",
                "fail_closed": True,
                "reason": sorted(set(invalid_reasons))[0],
                "rejection_reasons": sorted(set(invalid_reasons)),
            }
        )
        return ViewpointSamplingOdomReplayBatch((), diagnostics)

    if not eligible:
        return ViewpointSamplingOdomReplayBatch((), diagnostics)

    distances = tuple(
        math.hypot(
            sample.pose.x_m - target_pose.x_m,
            sample.pose.y_m - target_pose.y_m,
        )
        for sample in eligible
    )
    minimum_distance_index = min(
        range(len(distances)),
        key=distances.__getitem__,
    )
    strict_samples = tuple(
        sample
        for sample, distance_m in zip(eligible, distances)
        if distance_m <= strict_entry_tolerance_m
    )
    diagnostics.update(
        {
            "state": "replay_ready",
            "reason": "new_exact_frame_odom_selected",
            "processed_sample_count": len(eligible),
            "first_processed_stamp_sec": eligible[0].stamp_sec,
            "last_processed_stamp_sec": eligible[-1].stamp_sec,
            "minimum_target_distance_m": min(distances),
            "minimum_target_distance_stamp_sec": (
                eligible[minimum_distance_index].stamp_sec
            ),
            "strict_entry_sample_count": len(strict_samples),
            "first_strict_entry_stamp_sec": (
                None if not strict_samples else strict_samples[0].stamp_sec
            ),
            "last_strict_entry_stamp_sec": (
                None if not strict_samples else strict_samples[-1].stamp_sec
            ),
        }
    )
    return ViewpointSamplingOdomReplayBatch(tuple(eligible), diagnostics)


@dataclass(frozen=True)
class ViewpointConfig:
    min_distance_m: float = 0.28
    max_distance_m: float = 0.35
    target_distance_m: float = 0.30
    max_center_error_rad: float = math.radians(12.0)
    max_obliqueness_rad: float = math.radians(30.0)
    max_linear_speed_mps: float = 0.01
    max_angular_speed_radps: float = 0.05
    max_tangential_step_rad: float = math.radians(20.0)
    tangential_correction_gain: float = DEFAULT_TANGENTIAL_CORRECTION_GAIN
    settle_time_sec: float = 0.40

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.tangential_correction_gain)
            or not 0.0 < self.tangential_correction_gain <= 1.0
        ):
            raise ValueError("tangential correction gain must be in (0, 1]")


@dataclass(frozen=True)
class ViewpointMeasurement:
    image_stamp_sec: float
    scan_stamp_sec: float
    robot_pose: Pose2D
    linear_speed_mps: float
    angular_speed_radps: float
    stand_x_m: float
    stand_y_m: float
    distance_m: float | None
    camera_center_error_rad: float | None
    camera_yaw_rad: float | None
    silhouette_usable: bool


@dataclass(frozen=True)
class ViewpointDecision:
    state: str
    reason: str
    score: float
    synchronized_delta_sec: float
    recommended_pose: Pose2D
    geometrically_ready: bool
    stationary: bool


@dataclass(frozen=True)
class ViewpointSamplingUpdate:
    active: bool
    target_pose: Pose2D | None
    advanced: bool
    reason: str
    arrival_status: dict[str, object]


class ViewpointSamplingLatch:
    """Hold one closer camera target until it is safely sampled.

    The raw silhouette yaw is deliberately used only by
    :func:`refined_viewpoint_pose` to choose a bounded tangential step.  This
    latch prevents the step from becoming a per-frame moving carrot while the
    follower is driving.  At the initial, deliberately distant acquisition
    pose, either an oblique silhouette or a well-conditioned silhouette can
    seed the closer sampling target.  The latter is still only viewpoint
    evidence: the physical stand axis remains uncommitted until the robot
    reaches the final observation band and passes the complete settle gate.

    A later tangential step is accepted only after the current one has first
    entered the strict arrival gate and then remains inside the follower's
    shared target tube and inferred-stand annulus while the robot is
    stationary and the view remains oblique.  Missing, undersized, or
    non-face silhouette evidence can never start sampling.  An explicitly
    supplied observer-side recenter pose may revise an arrived target while
    the image remains off-center; the raw per-frame candidate is never used
    for that revision.
    """

    _START_REASONS = frozenset({"oblique_silhouette", "well_conditioned"})

    def __init__(
        self,
        *,
        arrival_tolerance_m: float = DEFAULT_SAMPLING_ARRIVAL_TOLERANCE_M,
        hold_tolerance_m: float = (
            INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
        ),
        target_distance_m: float = (
            DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M
        ),
        target_envelope_radius_m: float = (
            INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
        ),
    ) -> None:
        self.arrival_tolerance_m = arrival_tolerance_m
        self.target_pose: Pose2D | None = None
        self._arrival_target: ViewpointSamplingMaterialTarget | None = None
        self._arrival_latch = ViewpointSamplingArrivalLatch(
            strict_entry_tolerance_m=arrival_tolerance_m,
            hold_config=ViewpointSamplingHoldConfig(
                entry_tolerance_m=arrival_tolerance_m,
                hold_tolerance_m=hold_tolerance_m,
                target_distance_m=target_distance_m,
                target_envelope_radius_m=target_envelope_radius_m,
            ),
        )

    def reset(self, *, reason: str = "explicit_reset") -> None:
        self.target_pose = None
        self._arrival_target = None
        self._arrival_latch.reset(reason=reason)

    def invalidate_arrival_evidence(self, *, reason: str) -> None:
        """Fail closed without changing the observer-owned material target."""

        self._arrival_latch.reset(reason=reason)

    def arrival_status(self) -> dict[str, object]:
        return self._arrival_latch.to_status_dict()

    @property
    def arrival_target_pose(self) -> Pose2D | None:
        """Return the exact material target currently checked for arrival."""

        if self._arrival_target is None:
            return None
        return self._arrival_target.pose

    @property
    def arrival_material_target(
        self,
    ) -> ViewpointSamplingMaterialTarget | None:
        """Return the complete identity currently owning arrival evidence."""

        return self._arrival_target

    def observe_arrival_pose(
        self,
        robot_pose: Pose2D,
    ) -> dict[str, object] | None:
        """Replay one odometry pose through only the strict-arrival latch.

        Perception, recentering, sampling advancement, and settle gates remain
        owned by :meth:`update`; buffered odometry may recover only positional
        evidence for the already-active exact material target.
        """

        if self._arrival_target is None:
            return None
        arrival = self._arrival_latch.update(
            pose=robot_pose,
            target=self._arrival_target,
        )
        return arrival.to_status_dict()

    @staticmethod
    def _target_changed(first: Pose2D, second: Pose2D) -> bool:
        return (
            math.hypot(
                first.x_m - second.x_m,
                first.y_m - second.y_m,
            )
            > 1.0e-6
            or abs(normalize_angle(first.yaw_rad - second.yaw_rad))
            > 1.0e-6
        )

    @staticmethod
    def _finite_pose(pose: object) -> bool:
        if not isinstance(pose, Pose2D):
            return False
        try:
            return all(
                math.isfinite(value)
                for value in (pose.x_m, pose.y_m, pose.yaw_rad)
            )
        except TypeError:
            return False

    def update(
        self,
        *,
        robot_pose: Pose2D,
        stationary: bool,
        axis_input_reason: str,
        candidate_pose: Pose2D,
        allow_start: bool,
        view_centered: bool = True,
        view_settled: bool = True,
        recenter_pose: Pose2D | None = None,
        equivalent_target_poses: Sequence[Pose2D] = (),
        target_face_id: str = "sampling_near",
        target_revision: int | None = None,
        equivalent_target_face_ids: Sequence[str] = (),
        equivalent_target_revisions: Sequence[int | None] = (),
    ) -> ViewpointSamplingUpdate:
        poses_to_validate = (
            ("robot", robot_pose),
            ("candidate", candidate_pose),
            *(
                (f"equivalent_target_poses[{index}]", pose)
                for index, pose in enumerate(equivalent_target_poses)
            ),
        )
        for name, pose in poses_to_validate:
            if not all(
                math.isfinite(value)
                for value in (pose.x_m, pose.y_m, pose.yaw_rad)
            ):
                raise ValueError(f"{name} sampling pose must be finite")
        if (
            type(stationary) is not bool
            or type(allow_start) is not bool
            or type(view_centered) is not bool
            or type(view_settled) is not bool
        ):
            raise ValueError("sampling state flags must be boolean")
        if equivalent_target_face_ids and (
            len(equivalent_target_face_ids) != len(equivalent_target_poses)
        ):
            raise ValueError(
                "equivalent target face IDs must match equivalent target poses"
            )
        if equivalent_target_revisions and (
            len(equivalent_target_revisions) != len(equivalent_target_poses)
        ):
            raise ValueError(
                "equivalent target revisions must match equivalent target poses"
            )

        if self.target_pose is None:
            if axis_input_reason not in self._START_REASONS:
                return ViewpointSamplingUpdate(
                    False,
                    None,
                    False,
                    "axis_not_sampleable",
                    self.arrival_status(),
                )
            if (
                not stationary
                or not allow_start
                or not view_centered
                or not view_settled
            ):
                return ViewpointSamplingUpdate(
                    False,
                    None,
                    False,
                    "acquisition_not_settled",
                    self.arrival_status(),
                )
            self.target_pose = candidate_pose
            self._arrival_target = ViewpointSamplingMaterialTarget(
                pose=candidate_pose,
                face_id=target_face_id,
                target_revision=target_revision,
            )
            arrival = self._arrival_latch.update(
                pose=robot_pose,
                target=self._arrival_target,
            )
            return ViewpointSamplingUpdate(
                True,
                self.target_pose,
                True,
                "sampling_started",
                arrival.to_status_dict(),
            )

        target_candidates = [
            ViewpointSamplingMaterialTarget(
                pose=self.target_pose,
                face_id=target_face_id,
                target_revision=target_revision,
            )
        ]
        for index, pose in enumerate(equivalent_target_poses):
            face_id = (
                equivalent_target_face_ids[index]
                if equivalent_target_face_ids
                else f"sampling_equivalent_{index}"
            )
            revision = (
                equivalent_target_revisions[index]
                if equivalent_target_revisions
                else target_revision
            )
            target = ViewpointSamplingMaterialTarget(
                pose=pose,
                face_id=face_id,
                target_revision=revision,
            )
            if target not in target_candidates:
                target_candidates.append(target)
        if self._arrival_latch.arrived and self._arrival_target in target_candidates:
            selected_target = self._arrival_target
        else:
            selected_target = min(
                target_candidates,
                key=lambda target: math.hypot(
                    robot_pose.x_m - target.pose.x_m,
                    robot_pose.y_m - target.pose.y_m,
                ),
            )
        self._arrival_target = selected_target
        arrival = self._arrival_latch.update(
            pose=robot_pose,
            target=selected_target,
        )
        if arrival.arrived and stationary and not view_centered:
            recenter_available = (
                axis_input_reason in self._START_REASONS
                and self._finite_pose(recenter_pose)
                and self._target_changed(recenter_pose, selected_target.pose)
            )
            if recenter_available:
                assert recenter_pose is not None
                self.target_pose = recenter_pose
                self._arrival_target = ViewpointSamplingMaterialTarget(
                    pose=recenter_pose,
                    face_id=target_face_id,
                    target_revision=target_revision,
                )
                arrival = self._arrival_latch.update(
                    pose=robot_pose,
                    target=self._arrival_target,
                )
                return ViewpointSamplingUpdate(
                    True,
                    self.target_pose,
                    True,
                    "sampling_recentered_after_uncentered_arrival",
                    arrival.to_status_dict(),
                )
            return ViewpointSamplingUpdate(
                True,
                self.target_pose,
                False,
                "sampling_recenter_unavailable",
                arrival.to_status_dict(),
            )
        if (
            axis_input_reason == "oblique_silhouette"
            and stationary
            and arrival.arrived
            and view_centered
            and view_settled
        ):
            changed = self._target_changed(candidate_pose, self.target_pose)
            if changed:
                self.target_pose = candidate_pose
                self._arrival_target = ViewpointSamplingMaterialTarget(
                    pose=candidate_pose,
                    face_id=target_face_id,
                    target_revision=target_revision,
                )
                arrival = self._arrival_latch.update(
                    pose=robot_pose,
                    target=self._arrival_target,
                )
                return ViewpointSamplingUpdate(
                    True,
                    self.target_pose,
                    True,
                    "sampling_advanced",
                    arrival.to_status_dict(),
                )
        return ViewpointSamplingUpdate(
            True,
            self.target_pose,
            False,
            "sampling_target_held",
            arrival.to_status_dict(),
        )


def nearest_timed_sample(
    samples: Sequence[TimedSample[T]], stamp_sec: float, *, max_delta_sec: float
) -> TimedSample[T] | None:
    if max_delta_sec < 0.0:
        raise ValueError("max synchronization delta must be non-negative")
    if not samples:
        return None
    nearest = min(samples, key=lambda sample: abs(sample.stamp_sec - stamp_sec))
    return nearest if abs(nearest.stamp_sec - stamp_sec) <= max_delta_sec else None


def newest_synchronized_triplet(
    images: Sequence[TimedSample[T]],
    scans: Sequence[TimedSample[T]],
    odometry: Sequence[TimedSample[T]],
    *,
    min_image_stamp_exclusive: float,
    max_delta_sec: float,
) -> tuple[TimedSample[T], TimedSample[T], TimedSample[T]] | None:
    """Select the newest buffered image that has both sensor partners.

    Camera topics commonly publish faster than LaserScan. Processing only the
    newest image therefore creates periodic false synchronization failures
    while the next scan is still in flight. Retaining the image buffer and
    selecting the newest complete tuple adds bounded latency without relaxing
    the timestamp contract.
    """

    if not math.isfinite(max_delta_sec) or max_delta_sec < 0.0:
        raise ValueError("maximum synchronization delta must be non-negative")
    for image in reversed(tuple(images)):
        if image.stamp_sec <= min_image_stamp_exclusive:
            break
        scan = nearest_timed_sample(scans, image.stamp_sec, max_delta_sec=max_delta_sec)
        odom = nearest_timed_sample(
            odometry, image.stamp_sec, max_delta_sec=max_delta_sec
        )
        if scan is not None and odom is not None:
            return image, scan, odom
    return None


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def sampling_recenter_pose(
    *,
    current_target_pose: Pose2D | None,
    robot_pose: Pose2D,
    center_error_rad: float | None,
    target_distance_m: float,
    max_correction_rad: float,
) -> Pose2D | None:
    """Build one bounded off-center correction without moving the stand.

    The current material target is the only source of stand geometry.  Its
    yaw points from the target toward the stand, so adding one target distance
    recovers the frozen center.  The corrected target keeps that exact center
    and radius while applying the bounded image error around the robot's
    observed yaw.  Invalid or unavailable inputs return ``None`` so callers
    cannot accidentally fall back to a per-frame candidate.
    """

    if not isinstance(current_target_pose, Pose2D) or not isinstance(
        robot_pose, Pose2D
    ):
        return None
    values = (
        current_target_pose.x_m,
        current_target_pose.y_m,
        current_target_pose.yaw_rad,
        robot_pose.x_m,
        robot_pose.y_m,
        robot_pose.yaw_rad,
        center_error_rad,
        target_distance_m,
        max_correction_rad,
    )
    try:
        inputs_are_finite = all(
            value is not None and math.isfinite(value)
            for value in values
        )
    except TypeError:
        return None
    if not inputs_are_finite:
        return None
    if target_distance_m <= 0.0 or max_correction_rad <= 0.0:
        return None

    correction_rad = max(
        -max_correction_rad,
        min(max_correction_rad, center_error_rad),
    )
    corrected_yaw_rad = normalize_angle(
        robot_pose.yaw_rad - correction_rad
    )
    stand_x_m = (
        current_target_pose.x_m
        + target_distance_m * math.cos(current_target_pose.yaw_rad)
    )
    stand_y_m = (
        current_target_pose.y_m
        + target_distance_m * math.sin(current_target_pose.yaw_rad)
    )
    corrected = Pose2D(
        stand_x_m - target_distance_m * math.cos(corrected_yaw_rad),
        stand_y_m - target_distance_m * math.sin(corrected_yaw_rad),
        corrected_yaw_rad,
    )
    if not all(
        math.isfinite(value)
        for value in (corrected.x_m, corrected.y_m, corrected.yaw_rad)
    ):
        return None
    return corrected


def refined_viewpoint_pose(
    measurement: ViewpointMeasurement, config: ViewpointConfig
) -> Pose2D:
    stand = Pose2D(measurement.stand_x_m, measurement.stand_y_m)
    observer_bearing = math.atan2(
        measurement.robot_pose.y_m - stand.y_m,
        measurement.robot_pose.x_m - stand.x_m,
    )
    correction = 0.0
    if measurement.camera_yaw_rad is not None and math.isfinite(measurement.camera_yaw_rad):
        damped_camera_yaw_rad = (
            config.tangential_correction_gain * measurement.camera_yaw_rad
        )
        correction = max(
            -config.max_tangential_step_rad,
            min(config.max_tangential_step_rad, damped_camera_yaw_rad),
        )
    refined_bearing = normalize_angle(observer_bearing + correction)
    return Pose2D(
        stand.x_m + config.target_distance_m * math.cos(refined_bearing),
        stand.y_m + config.target_distance_m * math.sin(refined_bearing),
        normalize_angle(refined_bearing + math.pi),
    )


def evaluate_viewpoint(
    measurement: ViewpointMeasurement,
    config: ViewpointConfig = ViewpointConfig(),
) -> ViewpointDecision:
    delta = abs(measurement.image_stamp_sec - measurement.scan_stamp_sec)
    stationary = (
        abs(measurement.linear_speed_mps) <= config.max_linear_speed_mps
        and abs(measurement.angular_speed_radps) <= config.max_angular_speed_radps
    )
    distance_ok = (
        measurement.distance_m is not None
        and config.min_distance_m <= measurement.distance_m <= config.max_distance_m
    )
    centered = (
        measurement.camera_center_error_rad is not None
        and abs(measurement.camera_center_error_rad) <= config.max_center_error_rad
    )
    conditioned = (
        measurement.camera_yaw_rad is not None
        and abs(normalize_angle(measurement.camera_yaw_rad))
        <= config.max_obliqueness_rad
    )
    usable = measurement.silhouette_usable
    checks = (distance_ok, centered, conditioned, usable)
    score = sum(1.0 for value in checks if value) / len(checks)
    geometrically_ready = all(checks)
    if not usable:
        state, reason = "tracking", "silhouette_unavailable"
    elif not distance_ok:
        state, reason = "tracking", "distance_outside_observation_band"
    elif not centered:
        state, reason = "tracking", "stand_not_centered"
    elif not conditioned:
        state, reason = "viewpoint_optimization", "oblique_silhouette"
    elif not stationary:
        state, reason = "settling", "robot_moving"
    else:
        state, reason = "stationary_consensus", "viewpoint_ready"
    return ViewpointDecision(
        state=state,
        reason=reason,
        score=score,
        synchronized_delta_sec=delta,
        recommended_pose=refined_viewpoint_pose(measurement, config),
        geometrically_ready=geometrically_ready,
        stationary=stationary,
    )


class StationarySettleGate:
    def __init__(self, settle_time_sec: float = 0.40) -> None:
        if settle_time_sec <= 0.0:
            raise ValueError("settle time must be positive")
        self.settle_time_sec = settle_time_sec
        self._ready_since_sec: float | None = None

    def update(self, *, stamp_sec: float, ready: bool) -> bool:
        if not ready:
            self._ready_since_sec = None
            return False
        if self._ready_since_sec is None:
            self._ready_since_sec = stamp_sec
            return False
        return stamp_sec - self._ready_since_sec >= self.settle_time_sec


@dataclass(frozen=True)
class DynamicTargetConfig:
    approach_offset_m: float = 0.32
    freeze_distance_m: float = 0.42
    min_axis_samples: int = 7
    max_axis_samples: int = 12
    min_axis_confidence: float = 0.80
    max_axis_deviation_rad: float = math.radians(8.0)
    max_observation_linear_speed_mps: float = 0.12
    max_observation_angular_speed_radps: float = 0.35
    min_target_translation_m: float = 0.06
    min_target_yaw_change_rad: float = math.radians(10.0)
    side_switch_hysteresis: float = 0.20


@dataclass(frozen=True)
class DynamicTargetUpdate:
    accepted: bool
    reason: str
    pose: Pose2D | None
    stand_axis_rad: float | None
    axis_confidence: float
    side_index: int | None
    frozen: bool
    axis_sample_count: int = 0


class AxialAngleFilter:
    """Weighted circular mean for a 180-degree-symmetric stand axis."""

    def __init__(self, max_samples: int = 12) -> None:
        if max_samples < 1:
            raise ValueError("max axis samples must be positive")
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)

    def add(self, axis_rad: float, confidence: float) -> tuple[float, float]:
        if not math.isfinite(axis_rad):
            raise ValueError("axis angle must be finite")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("axis confidence must be in [0, 1]")
        if confidence > 0.0:
            self._samples.append((axis_rad, confidence))
        return self.estimate()

    def reset(self) -> None:
        self._samples.clear()

    def estimate(self) -> tuple[float, float]:
        if not self._samples:
            return 0.0, 0.0
        x = sum(weight * math.cos(2.0 * angle) for angle, weight in self._samples)
        y = sum(weight * math.sin(2.0 * angle) for angle, weight in self._samples)
        total = sum(weight for _, weight in self._samples)
        return normalize_angle(0.5 * math.atan2(y, x)), min(1.0, math.hypot(x, y) / total)

    def max_deviation_rad(self, reference_rad: float | None = None) -> float:
        """Largest 180-degree-symmetric residual in the retained window."""

        if not self._samples:
            return 0.0
        if reference_rad is None:
            reference_rad, _ = self.estimate()
        return max(
            0.5
            * abs(
                normalize_angle(2.0 * (sample_rad - reference_rad))
            )
            for sample_rad, _weight in self._samples
        )

    def stable_inlier_estimate(
        self,
        *,
        max_deviation_rad: float,
        min_samples: int,
    ) -> tuple[float, float, int] | None:
        """Return the densest stable axial cluster, ignoring isolated outliers."""

        if min_samples < 1 or len(self._samples) < min_samples:
            return None
        if not math.isfinite(max_deviation_rad) or max_deviation_rad <= 0.0:
            raise ValueError("maximum axial deviation must be positive")

        def axial_distance(a: float, b: float) -> float:
            return 0.5 * abs(normalize_angle(2.0 * (a - b)))

        clusters = []
        for center, _center_weight in self._samples:
            inliers = tuple(
                (angle, weight)
                for angle, weight in self._samples
                if axial_distance(angle, center) <= max_deviation_rad
            )
            clusters.append(
                (len(inliers), sum(weight for _, weight in inliers), inliers)
            )
        count, _weight, inliers = max(clusters, key=lambda item: (item[0], item[1]))
        if count < min_samples:
            return None
        x = sum(weight * math.cos(2.0 * angle) for angle, weight in inliers)
        y = sum(weight * math.sin(2.0 * angle) for angle, weight in inliers)
        total = sum(weight for _, weight in inliers)
        axis = normalize_angle(0.5 * math.atan2(y, x))
        confidence = min(1.0, math.hypot(x, y) / total)
        if any(
            axial_distance(angle, axis) > max_deviation_rad
            for angle, _weight in inliers
        ):
            return None
        return axis, confidence, count

    @property
    def sample_count(self) -> int:
        return len(self._samples)


def face_normal_candidates(stand: Pose2D, axis_rad: float, offset_m: float) -> tuple[Pose2D, Pose2D]:
    if offset_m <= 0.0:
        raise ValueError("approach offset must be positive")
    return tuple(
        Pose2D(
            face.target_pose.x_m,
            face.target_pose.y_m,
            face.target_pose.yaw_rad,
        )
        for face in arrival_face_candidates(
            stand,
            axis_rad,
            ArrivalGeometryConfig(standoff_distance_m=offset_m),
        )
    )


class DynamicPreApproachTracker:
    """Select a stable face-normal target while the robot approaches a stand."""

    def __init__(self, stand: Pose2D, config: DynamicTargetConfig = DynamicTargetConfig()) -> None:
        self.stand = stand
        self.config = config
        self.filter = AxialAngleFilter(config.max_axis_samples)
        self.current_pose: Pose2D | None = None
        self.current_side: int | None = None
        self.current_normal_rad: float | None = None
        self.committed_axis_rad: float | None = None
        self.committed_axis_confidence: float | None = None
        self.committed_axis_sample_count: int = 0
        self.frozen = False

    def reset_uncommitted_samples(self) -> None:
        """Require one consecutive settled evidence window before commitment."""

        if self.current_pose is None and not self.frozen:
            self.filter.reset()

    def _held_commit(self) -> DynamicTargetUpdate:
        """Return the one immutable axis/side/pose tuple accepted at commit."""

        if (
            self.current_pose is None
            or self.current_side is None
            or self.committed_axis_rad is None
            or self.committed_axis_confidence is None
        ):
            raise RuntimeError("committed dynamic target state is incomplete")
        return DynamicTargetUpdate(
            False,
            "target_committed",
            self.current_pose,
            self.committed_axis_rad,
            self.committed_axis_confidence,
            self.current_side,
            True,
            self.committed_axis_sample_count,
        )

    def _remember_physical_side(
        self, candidates: tuple[Pose2D, Pose2D], selected_side: int
    ) -> None:
        selected = candidates[selected_side]
        self.current_normal_rad = math.atan2(
            selected.y_m - self.stand.y_m,
            selected.x_m - self.stand.x_m,
        )
        self.current_side = selected_side

    def update(
        self,
        *,
        robot_pose: Pose2D,
        axis_rad: float | None,
        measurement_confidence: float,
        linear_speed_mps: float,
        angular_speed_radps: float,
        freeze_allowed: bool = False,
        allow_opposite_side_switch: bool = False,
        candidate_penalties: tuple[float, float] = (0.0, 0.0),
    ) -> DynamicTargetUpdate:
        # Retained for CLI/API compatibility; neither can relax the physical
        # side lock after a silhouette consensus has committed it.
        _ = freeze_allowed, allow_opposite_side_switch
        if self.frozen:
            return self._held_commit()
        if self.current_pose is not None:
            # A robust axis estimate commits the robot-facing physical side.
            # QR evidence may identify that side, but its absence must never
            # make motion jump through the stand to the opposite face.
            self.frozen = True
            return self._held_commit()
        if axis_rad is None:
            return DynamicTargetUpdate(False, "axis_unavailable", self.current_pose, None, 0.0, self.current_side, False)
        if abs(linear_speed_mps) > self.config.max_observation_linear_speed_mps:
            return DynamicTargetUpdate(False, "linear_motion_too_fast", self.current_pose, None, 0.0, self.current_side, False)
        if abs(angular_speed_radps) > self.config.max_observation_angular_speed_radps:
            return DynamicTargetUpdate(False, "angular_motion_too_fast", self.current_pose, None, 0.0, self.current_side, False)
        filtered_axis, confidence = self.filter.add(axis_rad, measurement_confidence)
        if self.filter.sample_count < self.config.min_axis_samples:
            return DynamicTargetUpdate(False, "insufficient_axis_samples", self.current_pose, filtered_axis, confidence, self.current_side, False)
        stable = self.filter.stable_inlier_estimate(
            max_deviation_rad=self.config.max_axis_deviation_rad,
            min_samples=self.config.min_axis_samples,
        )
        if stable is None:
            return DynamicTargetUpdate(False, "axis_samples_not_stable", self.current_pose, filtered_axis, confidence, self.current_side, False)
        filtered_axis, confidence, inlier_count = stable
        if confidence < self.config.min_axis_confidence:
            return DynamicTargetUpdate(False, "axis_consensus_uncertain", self.current_pose, filtered_axis, confidence, self.current_side, False)
        candidates = face_normal_candidates(self.stand, filtered_axis, self.config.approach_offset_m)
        costs = []
        for index, candidate in enumerate(candidates):
            cost = math.hypot(candidate.x_m - robot_pose.x_m, candidate.y_m - robot_pose.y_m)
            cost += max(0.0, candidate_penalties[index])
            costs.append(cost)
        selected_side = min(range(2), key=costs.__getitem__)
        selected = candidates[selected_side]
        self.current_pose = selected
        self._remember_physical_side(candidates, selected_side)
        # The stable inlier estimate, selected physical side, and target pose
        # form one atomic commitment.  The filter may retain rejected outliers,
        # so recomputing its full-window mean after freezing would report an
        # axis that no longer generated current_pose.
        self.committed_axis_rad = filtered_axis
        self.committed_axis_confidence = confidence
        self.committed_axis_sample_count = inlier_count
        self.frozen = True
        return DynamicTargetUpdate(
            True,
            "target_committed",
            selected,
            filtered_axis,
            confidence,
            selected_side,
            True,
            inlier_count,
        )
