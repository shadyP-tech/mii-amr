"""Decode original pixels only inside a confirmed current endpoint QR quad."""

from dataclasses import asdict
import math
import time
from types import SimpleNamespace

from scripts.aufgabe04.perception.camera_calibration import (
    CameraCalibration, validate_camera_calibration,
)
from scripts.aufgabe04.qr_scanning.isolated_qr_views import ISOLATED_QR_VIEWS
from scripts.aufgabe04.qr_scanning.opencv_qr_detector import detect_qr_observations_bgr
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation, validated_qr_corners
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.configuration.profile import (
    CameraCalibrationProfile, RigidTransform as CalibrationRigidTransform,
    camera_calibration_sha256, camera_info_mismatches,
)
from scripts.aufgabe04.real_robot.observer.opposite_endpoint_confirmation import KIND, HASH_FIELD
from scripts.aufgabe04.real_robot.observer.opposite_identity_crop import require_endpoint_outline_binding
from scripts.aufgabe04.real_robot.observer.opposite_target_support import validate_target_support
from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import SUBSET_KIND

POLICY = 'current_endpoint_raw_qr_pixels'
PIXEL_SOURCE = 'current_raw_camera_image'
MARGIN_RATIO = ISOLATED_QR_VIEWS[1].source_margin_ratio
_FIELDS = {'schema_version', 'policy', 'endpoint_confirmation_sha256', 'image_stamp_sec',
    'image_shape', 'calibration', 'sealed_calibration_profile', 'intrinsics',
    'calibration_profile_sha256', 'camera_frame',
    'confirmed_corners_px', 'raw_source_quad_px', 'source_bounds_xyxy', 'source_margin_ratio',
    'source_margin_px', 'decoded_raw_corners_px', 'mapped_rectified_corners_px',
    'maximum_corner_error_px', 'corner_error_px', 'qr_id', 'decoder', 'decoder_scale',
    'motion_authorized', 'supplies_angle'}


def _endpoint_proof(crop):
    envelope = (crop.get('search') or {}).get('envelope') or {}
    proof = envelope.get('witnessed_fragmentation') or {}
    if proof.get('kind') == SUBSET_KIND:
        proof = proof.get('envelope') or {}
    if proof.get('kind') != KIND:
        raise ValueError('raw QR pixels require a current endpoint confirmation')
    return envelope, proof


def _sealed_profile(value):
    if (not isinstance(value, dict) or set(value) != set(CameraCalibrationProfile.__dataclass_fields__)
            or not isinstance(value.get('base_to_camera'), dict)):
        raise ValueError('raw QR sealed calibration profile is malformed')
    return CameraCalibrationProfile(**{**value,
        'base_to_camera': CalibrationRigidTransform(**value['base_to_camera'])})


def _context(calibration, calibration_profile, intrinsics, support, crop, image_stamp_sec, image_shape):
    validate_camera_calibration(calibration)
    envelope, proof = _endpoint_proof(crop)
    if (not isinstance(calibration_profile, CameraCalibrationProfile)
            or camera_calibration_sha256(calibration_profile) != proof['calibration_profile_sha256']):
        raise ValueError('raw QR sealed calibration digest differs from the certified target')
    camera_info = SimpleNamespace(width=calibration.width_px, height=calibration.height_px,
        distortion_model=calibration_profile.distortion_model, d=calibration.distortion,
        k=calibration.camera_matrix, r=calibration.rectification_matrix, p=calibration.projection_matrix,
        header=SimpleNamespace(frame_id=calibration.frame_id))
    if camera_info_mismatches(calibration_profile, camera_info):
        raise ValueError('raw QR current calibration differs from the sealed calibration')
    require_endpoint_outline_binding(envelope, support, sampling=crop.get('sampling'))
    validate_target_support(support)
    if (crop.get('accepted') is not True or crop.get('target_support') != support
            or crop.get('policy') != 'opposite_current_scan_exclusive_identity_crop'
            or crop.get('image_stamp_sec') != image_stamp_sec
            or support['image_stamp_sec'] != image_stamp_sec
            or tuple(image_shape) != tuple(support['image_shape'])
            or tuple(image_shape) != (calibration.height_px, calibration.width_px)
            or asdict(intrinsics) != proof['intrinsics']
            or (intrinsics.width_px, intrinsics.height_px, intrinsics.fx_px, intrinsics.fy_px,
                intrinsics.cx_px, intrinsics.cy_px) != (calibration.width_px, calibration.height_px,
                calibration.fx_px, calibration.fy_px, calibration.cx_px, calibration.cy_px)
            or calibration.frame_id != proof['scan_from_camera']['child_frame']):
        raise ValueError('raw QR calibration differs from the current confirmed image')
    return proof


def _matrices(calibration):
    import numpy as np
    return (np.asarray(calibration.camera_matrix, dtype=float).reshape(3, 3),
        np.asarray(calibration.distortion, dtype=float),
        np.asarray(calibration.rectification_matrix, dtype=float).reshape(3, 3),
        np.asarray(calibration.projection_matrix, dtype=float).reshape(3, 4)[:, :3])


def _raw_quad(corners, calibration, cv2):
    import numpy as np
    try:
        k, d, r, p = _matrices(calibration)
        rays = np.linalg.solve(p, np.c_[corners, np.ones(4)].T)
        rays = np.linalg.solve(r, rays).T
        if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 0):
            raise ValueError('confirmed QR rays do not project into the raw camera')
        points = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, d)[0].reshape(4, 2)
    except (cv2.error, np.linalg.LinAlgError) as exc:
        raise ValueError('raw QR calibration cannot project the current quad') from exc
    result = validated_qr_corners(points, image_shape=(calibration.height_px, calibration.width_px))
    if result is None:
        raise ValueError('confirmed QR projection is outside the raw image')
    return result


def _rectified_quad(corners, calibration, cv2):
    import numpy as np
    k, d, r, p = _matrices(calibration)
    try:
        points = cv2.undistortPoints(np.asarray(corners, dtype=float).reshape(4, 1, 2),
            k, d, R=r, P=p).reshape(4, 2)
    except cv2.error as exc:
        raise ValueError('raw QR calibration cannot rectify decoded corners') from exc
    result = validated_qr_corners(points, image_shape=(calibration.height_px, calibration.width_px))
    if result is None:
        raise ValueError('decoded raw QR is outside the rectified image')
    return result


def _bounds(corners, image_shape):
    margin = round(MARGIN_RATIO*min(math.dist(corners[i], corners[(i+1)%4]) for i in range(4)))
    bounds = (math.floor(min(p[0] for p in corners))-margin,
        math.floor(min(p[1] for p in corners))-margin,
        math.ceil(max(p[0] for p in corners))+1+margin,
        math.ceil(max(p[1] for p in corners))+1+margin)
    if not 0 <= bounds[0] < bounds[2] <= image_shape[1] or not 0 <= bounds[1] < bounds[3] <= image_shape[0]:
        raise ValueError('confirmed QR source margin is outside the raw image')
    return bounds, margin


def _corner_error(actual, expected):
    # A detector may choose another first corner or winding for the same quad.
    return min(max(math.dist(actual[i], order[(i+offset)%4]) for i in range(4))
        for order in (expected, tuple(reversed(expected))) for offset in range(4))


def _same_quad(actual, expected):
    corners = validated_qr_corners(actual)
    return corners is not None and all(math.dist(a, b) <= 1e-7 for a, b in zip(corners, expected))


def validate_opposite_raw_qr_binding(binding, *, crop, qr_id, image_stamp_sec,
        image_shape, calibration_profile_sha256=None, camera_frame=None, cv2_module=None):
    """Replay the original/rectified pixel mapping when a QR receipt is loaded."""
    if (not isinstance(binding, dict) or set(binding) != _FIELDS
            or type(binding.get('schema_version')) is not int or binding['schema_version'] != 1
            or binding.get('policy') != POLICY or binding.get('motion_authorized') is not False
            or binding.get('supplies_angle') is not False or binding.get('qr_id') != qr_id
            or crop.get('payload_pixel_source') != PIXEL_SOURCE):
        raise ValueError('invalid current raw QR pixel binding')
    calibration = CameraCalibration(**binding['calibration'])
    calibration_profile = _sealed_profile(binding['sealed_calibration_profile'])
    intrinsics = CameraIntrinsics(**binding['intrinsics'])
    proof = _context(calibration, calibration_profile, intrinsics, crop.get('target_support'), crop,
        image_stamp_sec, image_shape)
    if (binding['image_stamp_sec'] != image_stamp_sec or tuple(binding['image_shape']) != tuple(image_shape)
            or binding['endpoint_confirmation_sha256'] != proof[HASH_FIELD]
            or binding['calibration_profile_sha256'] != proof['calibration_profile_sha256']
            or calibration_profile_sha256 not in (None, binding['calibration_profile_sha256'])
            or binding['camera_frame'] != calibration.frame_id
            or camera_frame not in (None, calibration.frame_id)
            or binding['source_margin_ratio'] != MARGIN_RATIO
            or not isinstance(binding['decoder'], str) or not binding['decoder']
            or type(binding['decoder_scale']) not in (int, float)
            or not math.isfinite(binding['decoder_scale']) or binding['decoder_scale'] < 1):
        raise ValueError('raw QR pixel binding differs from the confirmed tuple')
    if cv2_module is None:
        import cv2 as cv2_module
    return _validate_binding_pixels(binding, proof=proof, calibration=calibration,
        image_shape=image_shape, cv2=cv2_module)


def _validate_binding_pixels(binding, *, proof, calibration, image_shape, cv2):
    """Replay pixel geometry without rereading the already checked target chain."""
    confirmed = tuple(tuple(p) for p in proof['outline']['corners_px'])
    raw_quad = _raw_quad(confirmed, calibration, cv2)
    bounds, margin = _bounds(raw_quad, image_shape)
    if (not _same_quad(binding['confirmed_corners_px'], confirmed)
            or not _same_quad(binding['raw_source_quad_px'], raw_quad)
            or tuple(binding['source_bounds_xyxy']) != bounds or binding['source_margin_px'] != margin):
        raise ValueError('raw QR source crop differs from the confirmed quad')
    own = validated_qr_corners(binding['decoded_raw_corners_px'], image_shape=image_shape)
    if own is None or any(not bounds[0] <= x < bounds[2] or not bounds[1] <= y < bounds[3] for x, y in own):
        raise ValueError('decoded QR corners do not belong to the bounded raw source crop')
    mapped = _rectified_quad(own, calibration, cv2)
    limit = max(2., .04*min(math.dist(confirmed[i], confirmed[(i+1)%4]) for i in range(4)))
    error = _corner_error(mapped, confirmed)
    if (error > limit or not _same_quad(binding['mapped_rectified_corners_px'], mapped)
            or binding['maximum_corner_error_px'] != limit
            or type(binding['corner_error_px']) not in (int, float)
            or not math.isfinite(binding['corner_error_px']) or abs(binding['corner_error_px']-error) > 1e-7):
        raise ValueError('decoded raw QR does not match the confirmed current quad')
    return binding


def decode_opposite_raw_qr(raw_frame, cv2, *, calibration, calibration_profile, intrinsics, support,
        crop, image_stamp_sec, max_elapsed_sec, decoder_options=None, diagnostics=None):
    """Decode a bounded original-pixel crop; publish its binding only on success."""
    started = time.monotonic()
    crop.pop('raw_pixel_binding', None)
    crop.pop('payload_pixel_source', None)
    def miss(reason):
        if diagnostics is not None:
            diagnostics['raw_pixel_binding_reason'] = reason
        return ()
    if (type(max_elapsed_sec) not in (int, float) or not math.isfinite(max_elapsed_sec)
            or not 0 < max_elapsed_sec <= .12):
        return miss('raw QR payload budget unavailable')
    try:
        shape = tuple(raw_frame.shape[:2])
        metadata = support.metadata() if hasattr(support, 'metadata') else support
        proof = _context(calibration, calibration_profile, intrinsics, metadata, crop, image_stamp_sec, shape)
        confirmed = tuple(tuple(p) for p in proof['outline']['corners_px'])
        raw_quad = _raw_quad(confirmed, calibration, cv2)
        bounds, margin = _bounds(raw_quad, shape)
        remaining = max_elapsed_sec-(time.monotonic()-started)
        if remaining <= 0:
            return miss('raw QR payload budget exhausted')
        x0, y0, x1, y1 = bounds
        observations = detect_qr_observations_bgr(raw_frame[y0:y1, x0:x1], cv2,
            preferred_scale=2, max_elapsed_sec=remaining, diagnostics=diagnostics,
            **(decoder_options or {}))
        if len(observations) != 1 or observations[0].corners is None:
            return miss('raw QR requires exactly one decoded symbol with its own corners')
        observation = observations[0]
        own = tuple((x+x0, y+y0) for x, y in observation.corners)
        mapped = _rectified_quad(own, calibration, cv2)
        limit = max(2., .04*min(math.dist(confirmed[i], confirmed[(i+1)%4]) for i in range(4)))
        binding = dict(schema_version=1, policy=POLICY,
            endpoint_confirmation_sha256=proof[HASH_FIELD], image_stamp_sec=image_stamp_sec,
            image_shape=shape, calibration=asdict(calibration),
            sealed_calibration_profile=asdict(calibration_profile), intrinsics=asdict(intrinsics),
            calibration_profile_sha256=proof['calibration_profile_sha256'], camera_frame=calibration.frame_id,
            confirmed_corners_px=confirmed, raw_source_quad_px=raw_quad, source_bounds_xyxy=bounds,
            source_margin_ratio=MARGIN_RATIO, source_margin_px=margin, decoded_raw_corners_px=own,
            mapped_rectified_corners_px=mapped, maximum_corner_error_px=limit,
            corner_error_px=_corner_error(mapped, confirmed), qr_id=observation.text,
            decoder=observation.detector, decoder_scale=observation.scale,
            motion_authorized=False, supplies_angle=False)
        _validate_binding_pixels(binding, proof=proof, calibration=calibration,
            image_shape=shape, cv2=cv2)
        if time.monotonic()-started > max_elapsed_sec:
            return miss('raw QR payload budget exhausted')
        crop['raw_pixel_binding'] = binding
        crop['payload_pixel_source'] = PIXEL_SOURCE
        if diagnostics is not None:
            diagnostics['raw_pixel_binding_reason'] = 'current raw QR matches confirmed quad'
        return (DecodedQrObservation(observation.text, mapped,
            'opposite_raw_source:'+observation.detector, observation.scale),)
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError, cv2.error) as exc:
        return miss(str(exc))
