"""Calibrated, motion-neutral centering advice from one admitted current head.

The scan surface range is a depth proxy. Every turn needs a new stopped image;
this solver neither estimates the stand angle nor authorizes robot motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector, transform_point
from scripts.aufgabe04.real_robot.observer.finite_target_bearing import point_on_scan_range
from scripts.aufgabe04.perception.stand_axis_handoff.models import RigidTransform
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics, validate_intrinsics
from scripts.aufgabe04.real_robot.observer.evidence import EvidencePose
from scripts.aufgabe04.real_robot.observer.opposite_target_support import (
    OppositeTargetSupport, validate_target_support, POLICY as QR_SUPPORT_SOURCE,
)

MAX_CENTERING_STEP_RAD = math.radians(6.)
MAX_CENTERING_TRAVEL_RAD = math.radians(12.)
MAX_CENTERING_TURNS = 2
CENTERING_DEADBAND_RAD = math.radians(1.)
_HASH_KEY = "camera_centering_advisory_sha256"
_SOURCE = "current_measured_head_unique_lidar_cluster"


def _finite(values):
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        raise ValueError("centering geometry must contain finite numbers")


def _check_transform(transform):
    if (not transform.parent_frame or not transform.child_frame
            or len(transform.translation_xyz_m) != 3 or len(transform.rotation_xyzw) != 4):
        raise ValueError("invalid camera transform")
    _finite((*transform.translation_xyz_m, *transform.rotation_xyzw))
    rotate_vector((0., 0., 1.), transform.rotation_xyzw)


def center_point_in_base(*, center_px, intrinsics, distance_m,
                         scan_from_camera, base_from_camera):
    """Intersect a rectified camera ray with the associated scan-range cylinder."""
    validate_intrinsics(intrinsics)
    if len(center_px) != 2:
        raise ValueError("head center must have two coordinates")
    _finite((*center_px, distance_m))
    u, v = center_px
    if not (0 <= u < intrinsics.width_px and 0 <= v < intrinsics.height_px) or distance_m <= 0:
        raise ValueError("head center or associated range is outside its valid domain")
    for transform in (scan_from_camera, base_from_camera):
        _check_transform(transform)
    if scan_from_camera.child_frame != base_from_camera.child_frame:
        raise ValueError("camera transforms have different optical frames")
    _, depth = point_on_scan_range(center_px=center_px,intrinsics=intrinsics,
        scan_from_camera=scan_from_camera,distance_m=distance_m)
    return transform_point((depth*(u-intrinsics.cx_px)/intrinsics.fx_px,
                            depth*(v-intrinsics.cy_px)/intrinsics.fy_px,depth),base_from_camera)



def project_center_after_turn(*, point_base, yaw_rad, intrinsics, base_from_camera):
    """Project a stationary point after base yaw, including the camera lever arm."""
    _finite((*point_base, yaw_rad))
    _check_transform(base_from_camera)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    new_base = (c*point_base[0]+s*point_base[1], -s*point_base[0]+c*point_base[1], point_base[2])
    qx, qy, qz, qw = base_from_camera.rotation_xyzw
    camera = rotate_vector(tuple(a-b for a, b in zip(new_base, base_from_camera.translation_xyz_m)),
                           (-qx, -qy, -qz, qw))
    if camera[2] <= 1e-9:
        raise ValueError("head projects behind the camera")
    return (intrinsics.fx_px*camera[0]/camera[2]+intrinsics.cx_px,
            intrinsics.fy_px*camera[1]/camera[2]+intrinsics.cy_px)


def solve_camera_centering(*, center_px, intrinsics, distance_m,
                           scan_from_camera, base_from_camera,
                           remaining_rotation_rad=MAX_CENTERING_TRAVEL_RAD):
    """Return signed left-positive yaw; zero means inside the pixel deadband.

    A correction that cannot fit entirely in the remaining view budget raises
    ValueError. The caller must still limit each physical step to six degrees.
    """
    _finite((remaining_rotation_rad,))
    if not 0 < remaining_rotation_rad <= MAX_CENTERING_TRAVEL_RAD+1e-12:
        raise ValueError("invalid remaining centering budget")
    point = center_point_in_base(center_px=center_px, intrinsics=intrinsics,
        distance_m=distance_m, scan_from_camera=scan_from_camera, base_from_camera=base_from_camera)
    target = intrinsics.width_px/2.
    if abs(center_px[0]-target) <= intrinsics.fx_px*math.tan(CENTERING_DEADBAND_RAD):
        return 0.
    def residual(yaw):
        return project_center_after_turn(point_base=point, yaw_rad=yaw, intrinsics=intrinsics,
                                         base_from_camera=base_from_camera)[0]-target
    lo, hi = -remaining_rotation_rad, remaining_rotation_rad
    # Reject exotic/invalid mount geometry instead of assuming the turn sign.
    samples = [residual(lo+(hi-lo)*index/8) for index in range(9)]
    if samples[0] > 0 or samples[-1] < 0 or any(b <= a for a, b in zip(samples, samples[1:])):
        raise ValueError("head cannot be centered within the bounded yaw interval")
    for _ in range(48):
        middle = (lo+hi)/2
        if residual(middle) < 0:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CameraCenteringAdvisory:
    candidate_uid: str
    target_key: str
    stream_id: str
    planning_frame: str
    motion_epoch: int
    anchor_pose: EvidencePose
    anchor_odom_pose: EvidencePose
    odom_stamp_sec: float
    image_stamp_sec: float
    scan_stamp_sec: float
    created_at_sec: float
    robot_profile_sha256: str
    calibration_profile_sha256: str
    stand_model_profile_sha256: str
    intrinsics: CameraIntrinsics
    scan_from_camera: RigidTransform
    base_from_camera: RigidTransform
    measured_center_px: tuple[float, float]
    associated_range_m: float
    selected_cluster_sample_count: int
    requested_yaw_rad: float
    required_yaw_rad: float
    consumed_rotation_rad: float
    completed_turn_count: int

    target_support: dict | None = None
    target_reconciliation: dict | None = None

    @property
    def deadband_px(self):
        return self.intrinsics.fx_px*math.tan(CENTERING_DEADBAND_RAD)

    def metadata(self):
        payload = {**asdict(self), "schema_version": 1, "association_source": (_SOURCE if self.target_support is None else QR_SUPPORT_SOURCE),
            "eligible_cluster_count": 1, "image_width_px": self.intrinsics.width_px,
            "target_u_px": self.intrinsics.width_px/2., "deadband_px": self.deadband_px,
            "remaining_rotation_budget_rad": MAX_CENTERING_TRAVEL_RAD-self.consumed_rotation_rad,
            "motion_authorized": False, "completion_authorized": False}
        if self.target_reconciliation is None:
            payload.pop('target_reconciliation')
        if self.target_support is None:
            payload.pop("target_support")
        return {**payload, _HASH_KEY: _digest(payload)}


def build_camera_centering_advisory(*, association, intrinsics, scan_from_camera,
        base_from_camera, candidate_uid, target_key, stream_id, planning_frame,
        motion_epoch, anchor_pose, anchor_odom_pose, odom_stamp_sec,
        image_stamp_sec, now_sec, robot_profile_sha256, calibration_profile_sha256,
        stand_model_profile_sha256, max_age_sec=.5, max_image_scan_skew_sec=.1,
        consumed_rotation_rad=0., completed_turn_count=0):
    """Prepare advice only from an already admitted, uniquely associated head.

    The observer must additionally commit it only after the same frame passes
    its stationary-epoch and identity gates. This helper does not replace them.
    """
    try:
        lidar = association.lidar_association
        search = lidar.search_association if lidar is not None else None
        qr_support = isinstance(association, OppositeTargetSupport)
        if qr_support:
            validate_target_support(association.metadata())
        if (not association.accepted or not (qr_support or association.head_admission.accepted
                or association.head_orientation_bounds is not None)
                or lidar is None or not lidar.associated or search is None
                or not search.associated or search.eligible_cluster_count != 1
                or search.selected_cluster_sample_count < 1
                or search.scan_frame_id != scan_from_camera.parent_frame):
            return None
        scan_stamp = search.scan_stamp_sec
        _finite((now_sec, image_stamp_sec, scan_stamp, odom_stamp_sec, max_age_sec,
                 max_image_scan_skew_sec, consumed_rotation_rad))
        if not 0 < max_age_sec <= .5 or not 0 < max_image_scan_skew_sec <= .1:
            return None
        if (any(not 0 <= now_sec-stamp <= max_age_sec for stamp in
                (image_stamp_sec, scan_stamp, odom_stamp_sec))
                or abs(image_stamp_sec-scan_stamp) > max_image_scan_skew_sec
                or abs(image_stamp_sec-odom_stamp_sec) > max_image_scan_skew_sec):
            return None
        if (type(completed_turn_count) is not int or not 0 <= completed_turn_count < MAX_CENTERING_TURNS
                or not 0 <= consumed_rotation_rad < MAX_CENTERING_TRAVEL_RAD):
            return None
        required = solve_camera_centering(center_px=association.full_image_center_px,
            intrinsics=intrinsics, distance_m=lidar.distance_m, scan_from_camera=scan_from_camera,
            base_from_camera=base_from_camera,
            remaining_rotation_rad=MAX_CENTERING_TRAVEL_RAD-consumed_rotation_rad)
        if not required:
            return None
        advisory = CameraCenteringAdvisory(candidate_uid, target_key, stream_id, planning_frame,
            motion_epoch, anchor_pose, anchor_odom_pose, odom_stamp_sec, image_stamp_sec,
            scan_stamp, now_sec, robot_profile_sha256, calibration_profile_sha256,
            stand_model_profile_sha256, intrinsics, scan_from_camera, base_from_camera,
            tuple(association.full_image_center_px), lidar.distance_m,
            search.selected_cluster_sample_count,
            math.copysign(min(abs(required), MAX_CENTERING_STEP_RAD), required), required,
            consumed_rotation_rad, completed_turn_count,
            target_support=association.metadata() if qr_support else None,
            target_reconciliation=getattr(association,'target_reconciliation',None))
        # Same structural and derived-value validation applies at the process boundary.
        return validate_camera_centering_advisory(advisory.metadata())
    except (AttributeError, TypeError, ValueError, ArithmeticError):
        return None


def validate_camera_centering_advisory(payload: Mapping, *, candidate_uid=None,
        target_key=None, stream_id=None, robot_profile_sha256=None,
        calibration_profile_sha256=None, stand_model_profile_sha256=None,
        now_sec=None, max_receipt_age_sec=5., min_image_stamp_sec=None,
        min_scan_stamp_sec=None):
    """Parse a self-hashed observer receipt; reject stale, rebound or changed advice.

    The hash detects changed evidence, not malicious authorship. Only the
    observer's trusted output may enter a separately admitted motion action.
    """
    try:
        values = dict(payload)
        digest = values.pop(_HASH_KEY)
        if digest != _digest(values):
            raise ValueError("centering advisory hash mismatch")
        extras = ("schema_version", "association_source", "eligible_cluster_count", "image_width_px",
                  "target_u_px", "deadband_px", "remaining_rotation_budget_rad",
                  "motion_authorized", "completion_authorized")
        body = {key: value for key, value in values.items() if key not in extras}
        for key in ("anchor_pose", "anchor_odom_pose"):
            body[key] = EvidencePose(**body[key])
        body["intrinsics"] = CameraIntrinsics(**body["intrinsics"])
        for key in ("scan_from_camera", "base_from_camera"):
            tf = dict(body[key])
            for vector in ("translation_xyz_m", "rotation_xyzw"):
                tf[vector] = tuple(tf[vector])
            body[key] = RigidTransform(**tf)
        body["measured_center_px"] = tuple(body["measured_center_px"])
        result = CameraCenteringAdvisory(**body)
        if result.metadata() != dict(payload):
            # JSON round trips turn tuples into lists; canonical digests compare values.
            if _digest(result.metadata()) != _digest(dict(payload)):
                raise ValueError("centering advisory schema or derived metadata mismatch")
        for key, expected in (("candidate_uid", candidate_uid), ("target_key", target_key),
                ("stream_id", stream_id), ("robot_profile_sha256", robot_profile_sha256),
                ("calibration_profile_sha256", calibration_profile_sha256),
                ("stand_model_profile_sha256", stand_model_profile_sha256)):
            value = getattr(result, key)
            if not isinstance(value, str) or not value or expected is not None and value != expected:
                raise ValueError(f"centering {key} mismatch")
        if not isinstance(result.planning_frame, str) or not result.planning_frame:
            raise ValueError("centering planning frame missing")
        for key in ("robot_profile_sha256", "calibration_profile_sha256", "stand_model_profile_sha256"):
            value = getattr(result, key)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid profile hash")
        if (type(result.motion_epoch) is not int or result.motion_epoch < 0
                or type(result.completed_turn_count) is not int
                or not 0 <= result.completed_turn_count < MAX_CENTERING_TURNS
                or type(result.selected_cluster_sample_count) is not int
                or result.selected_cluster_sample_count < 1):
            raise ValueError("invalid observation epoch or centering count")
        _finite((*asdict(result.anchor_pose).values(), *asdict(result.anchor_odom_pose).values(),
            result.image_stamp_sec, result.scan_stamp_sec, result.odom_stamp_sec, result.created_at_sec,
            result.required_yaw_rad, result.requested_yaw_rad, result.consumed_rotation_rad))
        if (not 0 <= result.consumed_rotation_rad < MAX_CENTERING_TRAVEL_RAD
                or any(not 0 <= result.created_at_sec-stamp <= .5 for stamp in
                    (result.image_stamp_sec, result.scan_stamp_sec, result.odom_stamp_sec))
                or abs(result.image_stamp_sec-result.scan_stamp_sec) > .1
                or abs(result.image_stamp_sec-result.odom_stamp_sec) > .1):
            raise ValueError("centering sensor tuple or travel budget is invalid")
        if result.target_support is not None:
            proof = validate_target_support(result.target_support)
            cluster = proof['lidar_association']['search_association']
            if (tuple(proof['center_px']) != result.measured_center_px
                    or proof['image_stamp_sec'] != result.image_stamp_sec
                    or cluster['scan_stamp_sec'] != result.scan_stamp_sec
                    or cluster['scan_frame_id'] != result.scan_from_camera.parent_frame
                    or tuple(proof['image_shape']) != (result.intrinsics.height_px, result.intrinsics.width_px)
                    or proof['lidar_association']['distance_m'] != result.associated_range_m
                    or cluster['selected_cluster_sample_count'] != result.selected_cluster_sample_count):
                raise ValueError('centering QR support differs from current observation')
        if result.target_reconciliation is not None:
            from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
            from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
            proof = result.target_reconciliation
            proof_scan, envelope, _, reference = validate_reconciliation(proof,candidate_uid=result.candidate_uid,
                image_stamp_sec=result.image_stamp_sec,scan_stamp_sec=result.scan_stamp_sec)
            if (proof['target_key'] != result.target_key or proof['epoch'] != result.motion_epoch
                    or proof['planning_frame'] != result.planning_frame
                    or tuple(proof['entries'][-1]['robot_pose']) != tuple(asdict(result.anchor_pose).values())):
                raise ValueError('centering reconciliation epoch changed')
            bearing, uncertainty, _ = finite_target_bearing(center_px=result.measured_center_px,
                intrinsics=result.intrinsics,scan_from_camera=result.scan_from_camera,
                distance_m=envelope.distance_m,range_interval_m=envelope.accepted_range_m)
            if abs(math.remainder(bearing-reference,math.tau))+uncertainty > math.radians(3)+1e-9:
                raise ValueError('centering ray misses reconciled target')
            from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
            current = associate_camera_registered_candidate_lidar_target(proof_scan,
                map_bearing_rad=reference,observed_camera_bearing_rad=bearing,
                cone_half_angle_rad=math.radians(3),accepted_range_m=envelope.accepted_range_m,
                now_sec=result.created_at_sec,max_scan_age_sec=.5,min_cluster_sample_count=1,
                max_camera_map_bearing_delta_rad=math.radians(3))
            if (not current.associated or current.distance_m != result.associated_range_m
                    or current.search_association.selected_cluster_sample_count != result.selected_cluster_sample_count):
                raise ValueError('centering range differs from reconciled current cluster')
        required = solve_camera_centering(center_px=result.measured_center_px,
            intrinsics=result.intrinsics, distance_m=result.associated_range_m,
            scan_from_camera=result.scan_from_camera, base_from_camera=result.base_from_camera,
            remaining_rotation_rad=MAX_CENTERING_TRAVEL_RAD-result.consumed_rotation_rad)
        step = math.copysign(min(abs(required), MAX_CENTERING_STEP_RAD), required)
        if (not required or abs(result.required_yaw_rad-required) > 1e-10
                or abs(result.requested_yaw_rad-step) > 1e-10):
            raise ValueError("centering rotation differs from calibrated observation")
        if now_sec is not None:
            _finite((now_sec, max_receipt_age_sec))
            if not 0 < max_receipt_age_sec <= 5. or not 0 <= now_sec-result.created_at_sec <= max_receipt_age_sec:
                raise ValueError("centering advisory is stale or future dated")
        for stamp, minimum in ((result.image_stamp_sec, min_image_stamp_sec),
                               (result.scan_stamp_sec, min_scan_stamp_sec)):
            if minimum is not None:
                _finite((minimum,))
                if stamp <= minimum:
                    raise ValueError("centering sensor timestamps did not advance")
        return result
    except (KeyError, TypeError, AttributeError, ArithmeticError) as exc:
        raise ValueError("malformed centering advisory") from exc
