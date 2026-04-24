"""
pose_estimator.py — Classify markers and estimate robot pose.

Marker layout assumption:
    - Two large green circles are attached to the rear of the robot.
    - One smaller green circle is attached to the front.

Marker classification:
    - Front marker = smallest by radius.
    - Left / right rear markers are distinguished using the cross product
      of the heading vector, NOT by image x-coordinate.  This ensures
      correct classification regardless of robot orientation.

Pose computation:
    - Robot center = rear midpoint + configurable forward offset along
      the heading direction.
    - Yaw = atan2 of the heading vector (rear midpoint → front marker).
"""

import numpy as np
import config


def classify_markers(centers_world):
    """Classify three detected markers into left-rear, right-rear, and front.

    Parameters
    ----------
    centers_world : list of (np.ndarray, float)
        Each element is ``(world_xy, radius)`` where ``world_xy`` is a
        2-element array [x, y] in world meters and ``radius`` is the
        pixel-space radius (used only for size classification).

    Returns
    -------
    dict or None
        ``{"left": np.array, "right": np.array, "front": np.array}``
        with values in world meters.  Returns None if input has fewer
        than 3 markers.
    """
    if len(centers_world) < 3:
        return None

    # Sort by radius descending — two largest are rear, smallest is front
    sorted_markers = sorted(centers_world, key=lambda m: m[1], reverse=True)

    rear_a_pos = sorted_markers[0][0]
    rear_b_pos = sorted_markers[1][0]
    front_pos = sorted_markers[2][0]

    # Heading: rear midpoint → front marker
    rear_mid = (rear_a_pos + rear_b_pos) / 2.0
    heading = front_pos - rear_mid

    # Classify left / right using cross product.
    # In a right-handed 2D frame:
    #   cross(heading, v) > 0  ⟹  v is to the LEFT of heading
    #   cross(heading, v) < 0  ⟹  v is to the RIGHT of heading
    vec_a = rear_a_pos - rear_mid
    side_a = np.cross(heading, vec_a)

    if side_a > 0:
        left = rear_a_pos
        right = rear_b_pos
    else:
        left = rear_b_pos
        right = rear_a_pos

    return {
        "left": left,
        "right": right,
        "front": front_pos,
    }


def estimate_pose(classified):
    """Compute robot center and heading from classified marker positions.

    Parameters
    ----------
    classified : dict
        Output of :func:`classify_markers` — world-coordinate positions
        for "left", "right", and "front" markers.

    Returns
    -------
    (float, float, float)
        ``(x, y, yaw)`` where x, y are in meters and yaw is in radians
        (measured from the world x-axis).
    """
    left = classified["left"]
    right = classified["right"]
    front = classified["front"]

    rear_mid = (left + right) / 2.0

    heading = front - rear_mid
    heading_norm = np.linalg.norm(heading)

    if heading_norm < 1e-6:
        # Degenerate case — markers are coincident
        return rear_mid[0], rear_mid[1], 0.0

    heading_unit = heading / heading_norm
    yaw = float(np.arctan2(heading_unit[1], heading_unit[0]))

    # Apply forward offset to get approximate robot center
    center = rear_mid + config.CENTER_FORWARD_OFFSET * heading_unit

    return float(center[0]), float(center[1]), yaw
