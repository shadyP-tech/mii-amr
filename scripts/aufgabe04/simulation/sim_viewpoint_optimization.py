"""Pure synchronized-viewpoint decisions for the Gazebo stand pipeline."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from scripts.aufgabe04.navigation.models import Pose2D


T = TypeVar("T")


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    stamp_sec: float
    value: T


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
    settle_time_sec: float = 0.40


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


class ViewpointSamplingLatch:
    """Hold one tangential camera target until it is safely sampled.

    The raw silhouette yaw is deliberately used only by
    :func:`refined_viewpoint_pose` to choose a bounded tangential step.  This
    latch prevents the step from becoming a per-frame moving carrot while the
    follower is driving.  A new step is accepted only after the current one is
    reached, the robot is stationary, and the view remains oblique.
    """

    def __init__(self, *, arrival_tolerance_m: float = 0.10) -> None:
        if not math.isfinite(arrival_tolerance_m) or arrival_tolerance_m <= 0.0:
            raise ValueError("sampling arrival tolerance must be finite and positive")
        self.arrival_tolerance_m = arrival_tolerance_m
        self.target_pose: Pose2D | None = None

    def reset(self) -> None:
        self.target_pose = None

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
    ) -> ViewpointSamplingUpdate:
        for name, pose in (("robot", robot_pose), ("candidate", candidate_pose)):
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

        if self.target_pose is None:
            if axis_input_reason != "oblique_silhouette":
                return ViewpointSamplingUpdate(
                    False, None, False, "axis_not_oblique"
                )
            if (
                not stationary
                or not allow_start
                or not view_centered
                or not view_settled
            ):
                return ViewpointSamplingUpdate(
                    False, None, False, "acquisition_not_settled"
                )
            self.target_pose = candidate_pose
            return ViewpointSamplingUpdate(
                True, self.target_pose, True, "sampling_started"
            )

        reached = math.hypot(
            robot_pose.x_m - self.target_pose.x_m,
            robot_pose.y_m - self.target_pose.y_m,
        ) <= self.arrival_tolerance_m
        if (
            axis_input_reason == "oblique_silhouette"
            and stationary
            and reached
            and view_centered
            and view_settled
        ):
            changed = (
                math.hypot(
                    candidate_pose.x_m - self.target_pose.x_m,
                    candidate_pose.y_m - self.target_pose.y_m,
                )
                > 1.0e-6
                or abs(normalize_angle(candidate_pose.yaw_rad - self.target_pose.yaw_rad))
                > 1.0e-6
            )
            if changed:
                self.target_pose = candidate_pose
                return ViewpointSamplingUpdate(
                    True, self.target_pose, True, "sampling_advanced"
                )
        return ViewpointSamplingUpdate(
            True, self.target_pose, False, "sampling_target_held"
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
        correction = max(
            -config.max_tangential_step_rad,
            min(config.max_tangential_step_rad, measurement.camera_yaw_rad),
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
    normals = (axis_rad + math.pi / 2.0, axis_rad - math.pi / 2.0)
    return tuple(
        Pose2D(
            stand.x_m + offset_m * math.cos(normal),
            stand.y_m + offset_m * math.sin(normal),
            normalize_angle(normal + math.pi),
        )
        for normal in normals
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
        filtered_axis, confidence, _inlier_count = stable
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
        self.frozen = True
        return DynamicTargetUpdate(True, "target_committed", selected, filtered_axis, confidence, selected_side, True)
