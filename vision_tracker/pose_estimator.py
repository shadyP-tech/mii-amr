"""
pose_estimator.py — Classify markers and estimate robot pose.

Marker layout assumption:
    - Two large green circles are attached to the front of the robot.
    - One smaller green circle is attached to the rear.

Marker classification:
    - Rear marker = smallest by radius.
    - Left / right front markers are distinguished using the cross product
      of the heading vector, NOT by image x-coordinate.  This ensures
      correct classification regardless of robot orientation.

Pose computation:
    - Robot center = rear marker + configurable forward offset along
      the heading direction.
    - Yaw = atan2 of the heading vector (rear marker → front midpoint).
"""

import numpy as np
import config


def classify_markers(centers_world):
    """Classify three detected markers into front-left, front-right, and rear.
    
    Expects a 90-degree right triangle layout where the rear marker is on
    the same forward vector as one of the front markers.

    Parameters
    ----------
    centers_world : list of (np.ndarray, float)
        Each element is ``(world_xy, radius)`` where ``world_xy`` is a
        2-element array [x, y] in world meters and ``radius`` is the
        pixel-space radius (used only for size classification).

    Returns
    -------
    dict or None
        ``{"front_left": np.array, "front_right": np.array, "rear": np.array}``
        with values in world meters.  Returns None if input has fewer
        than 3 markers.
    """
    if len(centers_world) < 3:
        return None

    # Sort by radius descending — two largest are front, smallest is rear
    sorted_markers = sorted(centers_world, key=lambda m: m[1], reverse=True)

    front_a_pos = sorted_markers[0][0]
    front_b_pos = sorted_markers[1][0]
    rear_pos = sorted_markers[2][0]

    # Because the markers form a right triangle, the front marker closest to
    # the rear marker forms the straight forward heading line.
    dist_a = np.linalg.norm(front_a_pos - rear_pos)
    dist_b = np.linalg.norm(front_b_pos - rear_pos)

    if dist_a < dist_b:
        straight_front = front_a_pos
        diagonal_front = front_b_pos
    else:
        straight_front = front_b_pos
        diagonal_front = front_a_pos

    # True heading is from rear directly to the straight_front marker
    heading = straight_front - rear_pos
    vec_diagonal = diagonal_front - rear_pos

    # Classify left / right using cross product.
    # If cross product > 0, the diagonal vector is to the LEFT of the heading
    cross_prod = np.cross(heading, vec_diagonal)

    if cross_prod > 0:
        front_left = diagonal_front
        front_right = straight_front
    else:
        front_left = straight_front
        front_right = diagonal_front

    return {
        "front_left": front_left,
        "front_right": front_right,
        "rear": rear_pos,
    }


def estimate_pose(classified):
    """Compute robot center and heading from classified marker positions.

    Parameters
    ----------
    classified : dict
        Output of :func:`classify_markers` — world-coordinate positions
        for "front_left", "front_right", and "rear" markers.

    Returns
    -------
    (float, float, float)
        ``(x, y, yaw)`` where x, y are in meters and yaw is in radians
        (measured from the world x-axis).
    """
    front_left = classified["front_left"]
    front_right = classified["front_right"]
    rear = classified["rear"]

    # Since the layout is a right triangle, the true heading is formed by
    # either (front_left - rear) or (front_right - rear), depending on which
    # is the straight edge. We recreate that straight heading.
    dist_left = np.linalg.norm(front_left - rear)
    dist_right = np.linalg.norm(front_right - rear)
    
    if dist_left < dist_right:
        heading = front_left - rear
    else:
        heading = front_right - rear

    heading_norm = np.linalg.norm(heading)

    if heading_norm < 1e-6:
        # Degenerate case — markers are coincident
        return rear[0], rear[1], 0.0

    heading_unit = heading / heading_norm
    yaw = float(np.arctan2(heading_unit[1], heading_unit[0]))

    # Apply forward offset to get approximate robot center.
    # Note: Because the rear marker is on the side of the robot (forming the straight edge),
    # adding the forward offset purely along the heading will place the 'center'
    # on that same side edge, not the physical center of the chassis.
    # If a true geometric center is needed, consider calculating a lateral shift
    # perpendicular to the heading vector.
    center = rear + config.CENTER_FORWARD_OFFSET * heading_unit

    return float(center[0]), float(center[1]), yaw
