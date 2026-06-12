from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


RVIZ_COLOR_PATH = (0.0, 0.55, 1.0, 0.95)
RVIZ_COLOR_WAYPOINT = (0.0, 0.75, 1.0, 0.85)
RVIZ_COLOR_CURRENT = (1.0, 0.82, 0.16, 0.95)
RVIZ_COLOR_GOAL = (0.95, 0.18, 0.14, 0.95)
RVIZ_COLOR_LABEL = (0.95, 0.95, 0.95, 1.0)
RVIZ_COLOR_CONFIRMED_OBSTACLE = (0.05, 0.95, 0.22, 0.85)
RVIZ_COLOR_INFLATED_OBSTACLE = (0.9, 0.25, 1.0, 0.30)
RVIZ_COLOR_BLOCKED_CORRIDOR = (1.0, 0.45, 0.0, 0.80)


@dataclass(frozen=True)
class RvizMessageTypes:
    point: object
    pose_stamped: object
    nav_path: object
    marker: object
    marker_array: object
    qos_profile: object | None = None
    durability_policy: object | None = None


@dataclass(frozen=True)
class RvizNodeContext:
    build_rviz_path_message: Callable[..., object]
    build_rviz_waypoint_markers: Callable[..., object]
    build_rviz_obstacle_markers: Callable[..., object]


def rviz_messages_available(message_types):
    return all(
        message_type is not None
        for message_type in (
            message_types.nav_path,
            message_types.pose_stamped,
            message_types.point,
            message_types.marker,
            message_types.marker_array,
        )
    )


def rviz_qos_profile(message_types):
    if message_types.qos_profile is None or message_types.durability_policy is None:
        return 1
    qos = message_types.qos_profile(depth=1)
    qos.durability = message_types.durability_policy.TRANSIENT_LOCAL
    return qos


def set_header(message, frame_id, stamp):
    message.header.frame_id = frame_id
    message.header.stamp = stamp


def set_pose_xy(pose, x, y, z=0.0):
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0


def point_msg(message_types, x, y, z=0.0):
    point = message_types.point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def set_marker_color(marker, color):
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]


def marker_delete_all(message_types, frame_id, stamp):
    marker = message_types.marker()
    set_header(marker, frame_id, stamp)
    marker.action = message_types.marker.DELETEALL
    return marker


def apply_marker_common(
    message_types,
    marker,
    frame_id,
    stamp,
    namespace,
    marker_id,
    marker_type,
    color,
):
    set_header(marker, frame_id, stamp)
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = marker_type
    marker.action = message_types.marker.ADD
    marker.pose.orientation.w = 1.0
    set_marker_color(marker, color)


def build_pose_stamped(message_types, frame_id, stamp, x, y):
    pose = message_types.pose_stamped()
    set_header(pose, frame_id, stamp)
    set_pose_xy(pose.pose, x, y)
    return pose


def build_rviz_path_message(message_types, waypoints, frame_id, stamp, current_pose=None):
    if message_types.nav_path is None or message_types.pose_stamped is None:
        raise RuntimeError("ROS nav_msgs/geometry_msgs are unavailable.")
    path = message_types.nav_path()
    set_header(path, frame_id, stamp)
    if current_pose is not None:
        path.poses.append(
            build_pose_stamped(message_types, frame_id, stamp, current_pose.x, current_pose.y)
        )
    for waypoint in waypoints:
        path.poses.append(
            build_pose_stamped(message_types, frame_id, stamp, waypoint.x, waypoint.y)
        )
    return path


def waypoint_point(message_types, waypoint, z=0.04):
    return point_msg(message_types, waypoint.x, waypoint.y, z)


def build_point_layer_marker(
    message_types,
    frame_id,
    stamp,
    namespace,
    marker_id,
    marker_type,
    points,
    color,
    scale_m,
):
    if not points:
        return None
    marker = message_types.marker()
    apply_marker_common(
        message_types,
        marker,
        frame_id,
        stamp,
        namespace,
        marker_id,
        marker_type,
        color,
    )
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    marker.points = list(points)
    return marker


def build_single_waypoint_marker(
    message_types,
    frame_id,
    stamp,
    namespace,
    marker_id,
    waypoint,
    color,
    scale_m,
    z,
):
    marker = message_types.marker()
    apply_marker_common(
        message_types,
        marker,
        frame_id,
        stamp,
        namespace,
        marker_id,
        message_types.marker.SPHERE,
        color,
    )
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    set_pose_xy(marker.pose, waypoint.x, waypoint.y, z)
    return marker


def build_waypoint_label_marker(message_types, frame_id, stamp, marker_id, waypoint):
    marker = message_types.marker()
    apply_marker_common(
        message_types,
        marker,
        frame_id,
        stamp,
        "planned_waypoint_labels",
        marker_id,
        message_types.marker.TEXT_VIEW_FACING,
        RVIZ_COLOR_LABEL,
    )
    set_pose_xy(marker.pose, waypoint.x, waypoint.y, 0.20)
    marker.scale.z = 0.08
    marker.text = str(waypoint.index)
    return marker


def build_rviz_waypoint_markers(
    message_types,
    waypoints,
    frame_id,
    stamp,
    current_waypoint_index=0,
):
    if (
        message_types.marker is None
        or message_types.marker_array is None
        or message_types.point is None
    ):
        raise RuntimeError("ROS visualization messages are unavailable.")
    waypoints = list(waypoints)
    markers = [marker_delete_all(message_types, frame_id, stamp)]
    points = [waypoint_point(message_types, waypoint) for waypoint in waypoints]
    waypoint_layer = build_point_layer_marker(
        message_types,
        frame_id,
        stamp,
        "planned_waypoints",
        1,
        message_types.marker.SPHERE_LIST,
        points,
        RVIZ_COLOR_WAYPOINT,
        0.07,
    )
    if waypoint_layer is not None:
        markers.append(waypoint_layer)
    if waypoints:
        current_index = max(0, min(int(current_waypoint_index), len(waypoints) - 1))
        markers.append(
            build_single_waypoint_marker(
                message_types,
                frame_id,
                stamp,
                "current_waypoint",
                2,
                waypoints[current_index],
                RVIZ_COLOR_CURRENT,
                0.14,
                0.08,
            )
        )
        markers.append(
            build_single_waypoint_marker(
                message_types,
                frame_id,
                stamp,
                "goal_waypoint",
                3,
                waypoints[-1],
                RVIZ_COLOR_GOAL,
                0.12,
                0.10,
            )
        )
        for label_index, waypoint in enumerate(waypoints):
            markers.append(
                build_waypoint_label_marker(
                    message_types,
                    frame_id,
                    stamp,
                    1000 + label_index,
                    waypoint,
                )
            )
    return message_types.marker_array(markers=markers)


def build_cell_layer_marker(
    message_types,
    grid_to_world,
    run_local_map,
    frame_id,
    stamp,
    namespace,
    marker_id,
    cells,
    color,
    z,
    height_m,
):
    cells = sorted(cells or ())
    if not cells:
        return None
    metadata = run_local_map.static_map.metadata
    marker = message_types.marker()
    apply_marker_common(
        message_types,
        marker,
        frame_id,
        stamp,
        namespace,
        marker_id,
        message_types.marker.CUBE_LIST,
        color,
    )
    marker.scale.x = metadata.resolution
    marker.scale.y = metadata.resolution
    marker.scale.z = height_m
    marker.points = [
        point_msg(
            message_types,
            *grid_to_world(cell[0], cell[1], metadata),
            z,
        )
        for cell in cells
    ]
    return marker


def append_marker(markers, marker):
    if marker is not None:
        markers.append(marker)


def build_rviz_obstacle_markers(
    message_types,
    grid_to_world,
    run_local_map,
    frame_id,
    stamp,
    blocked_cells=None,
):
    if (
        message_types.marker is None
        or message_types.marker_array is None
        or message_types.point is None
    ):
        raise RuntimeError("ROS visualization messages are unavailable.")
    markers = [marker_delete_all(message_types, frame_id, stamp)]
    if run_local_map is None:
        return message_types.marker_array(markers=markers)
    append_marker(
        markers,
        build_cell_layer_marker(
            message_types,
            grid_to_world,
            run_local_map,
            frame_id,
            stamp,
            "run_local_inflated_obstacle_cells",
            1,
            run_local_map.inflated_obstacle_cells,
            RVIZ_COLOR_INFLATED_OBSTACLE,
            0.005,
            0.02,
        ),
    )
    append_marker(
        markers,
        build_cell_layer_marker(
            message_types,
            grid_to_world,
            run_local_map,
            frame_id,
            stamp,
            "run_local_confirmed_obstacle_cells",
            2,
            run_local_map.confirmed_raw_cells,
            RVIZ_COLOR_CONFIRMED_OBSTACLE,
            0.045,
            0.05,
        ),
    )
    append_marker(
        markers,
        build_cell_layer_marker(
            message_types,
            grid_to_world,
            run_local_map,
            frame_id,
            stamp,
            "run_local_blocked_corridor_cells",
            3,
            blocked_cells or set(),
            RVIZ_COLOR_BLOCKED_CORRIDOR,
            0.075,
            0.06,
        ),
    )
    return message_types.marker_array(markers=markers)


def rviz_visualization_enabled(node):
    return (
        not getattr(node.args, "no_rviz_visualization", False)
        and node.rviz_path_pub is not None
        and node.rviz_waypoint_marker_pub is not None
        and node.rviz_obstacle_marker_pub is not None
    )


def rviz_stamp(node):
    return node.get_clock().now().to_msg()


def publish_rviz_route(
    node,
    waypoints,
    current_pose=None,
    current_waypoint_index=0,
    *,
    context: RvizNodeContext,
):
    if not node.rviz_visualization_enabled():
        return
    waypoints = list(waypoints)
    stamp = node.rviz_stamp()
    node.rviz_path_pub.publish(
        context.build_rviz_path_message(
            waypoints,
            node.args.map_frame,
            stamp,
            current_pose=current_pose,
        )
    )
    node.rviz_waypoint_marker_pub.publish(
        context.build_rviz_waypoint_markers(
            waypoints,
            node.args.map_frame,
            stamp,
            current_waypoint_index=current_waypoint_index,
        )
    )


def publish_rviz_obstacles(
    node,
    blocked_cells=None,
    *,
    context: RvizNodeContext,
):
    if not node.rviz_visualization_enabled():
        return
    if blocked_cells is not None:
        node.rviz_last_blocked_cells = set(blocked_cells)
    node.rviz_obstacle_marker_pub.publish(
        context.build_rviz_obstacle_markers(
            node.run_local_map,
            node.args.map_frame,
            node.rviz_stamp(),
            blocked_cells=node.rviz_last_blocked_cells,
        )
    )


def publish_rviz_route_if_available(
    node,
    waypoints,
    current_pose=None,
    current_waypoint_index=0,
):
    publish = getattr(node, "publish_rviz_route", None)
    if callable(publish):
        publish(
            waypoints,
            current_pose=current_pose,
            current_waypoint_index=current_waypoint_index,
        )


def publish_rviz_obstacles_if_available(node, blocked_cells=None):
    publish = getattr(node, "publish_rviz_obstacles", None)
    if callable(publish):
        publish(blocked_cells=blocked_cells)
